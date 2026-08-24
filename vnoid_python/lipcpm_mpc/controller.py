"""High-level one-axis LIPCPM-MPC controller.

This module is intentionally independent of vnoid / Choreonoid.  It only
combines the supplied ``parameters.py``, ``model.py`` and ``mpc.py``.

State order
-----------
    X = [x_robot, dx_robot, x_liquid, dx_liquid]

The MPC tracks a *future state reference*.  DCM/ZMP are not MPC states and are
not generated here.  Their conversion for the vnoid interface is handled by
``vnoid_adapter.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import DiscreteSystem, discretize, lipcpm_continuous_matrices
from .mpc import CondensedMPC
from .parameters import LIPCPMParameters

Array = np.ndarray


@dataclass(frozen=True, slots=True)
class LIPCPMControlResult:
    """Result of one MPC solve.

    ``predicted_states[i]`` is X[k+i+1|k].  ``mpc_inputs[i]`` is the optimized
    decision input, while ``known_inputs[i]`` is an additive known acceleration
    supplied by the integration layer.  The model is propagated with
    ``total_inputs = mpc_inputs + known_inputs``.
    """

    state: Array
    reference: Array
    predicted_states: Array
    mpc_inputs: Array
    known_inputs: Array
    total_inputs: Array

    first_mpc_input: float
    first_known_input: float
    first_total_input: float

    objective: float
    hessian_condition: float
    input_was_limited: bool


class LIPCPMController:
    """One-axis LIPCPM-MPC wrapper around :class:`CondensedMPC`.

    Parameters
    ----------
    params:
        Physical LIPCPM parameters.
    dt:
        MPC sampling period [s].
    horizon:
        Number of future samples N.
    state_weights:
        Q diagonal or 4x4 Q.  Defaults to the values stored in ``parameters``.
    input_weight:
        R.  Defaults to the value stored in ``parameters``.
    discretization:
        ``"zoh"`` or ``"euler"``; passed directly to ``model.discretize``.
    mpc_input_limit:
        Optional post-solve clamp for the optimized input.  Because the supplied
        ``mpc.py`` is unconstrained, this is a safety clamp, not a constrained-QP
        solution.
    """

    def __init__(
        self,
        params: LIPCPMParameters,
        *,
        dt: float = 0.02,
        horizon: int = 50,
        state_weights: Array | Sequence[float] | None = None,
        input_weight: float | Array | None = None,
        discretization: str = "zoh",
        regularization: float = 1.0e-12,
        mpc_input_limit: float | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if discretization not in ("zoh", "euler"):
            raise ValueError("discretization must be 'zoh' or 'euler'")
        if mpc_input_limit is not None and mpc_input_limit <= 0.0:
            raise ValueError("mpc_input_limit must be positive when specified")

        self.params = params
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.discretization = str(discretization)
        self.mpc_input_limit = (
            None if mpc_input_limit is None else float(mpc_input_limit)
        )

        self.continuous_A, self.continuous_B = lipcpm_continuous_matrices(params)
        self.system: DiscreteSystem = discretize(
            self.continuous_A,
            self.continuous_B,
            self.dt,
            method=self.discretization,
        )

        q = (
            np.asarray(params.PAPER_STATE_WEIGHTS, dtype=float)
            if state_weights is None
            else np.asarray(state_weights, dtype=float)
        )
        r = params.PAPER_INPUT_WEIGHT if input_weight is None else input_weight

        self.mpc = CondensedMPC(
            self.system,
            horizon=self.horizon,
            state_weights=q,
            input_weight=r,
            regularization=regularization,
        )

    def normalize_state(self, state: Array | Sequence[float]) -> Array:
        x = np.asarray(state, dtype=float).reshape(4)
        if not np.all(np.isfinite(x)):
            raise ValueError("state must contain only finite values")
        return x

    def normalize_reference(self, reference: Array | Sequence[float]) -> Array:
        """Return a ``(N,4)`` LIPCPM reference.

        ``(N,2)`` means ``[x_robot_ref, dx_robot_ref]``.  In that case the
        desired liquid state is set equal to the robot state, i.e. desired
        relative sloshing is zero.
        """
        ref = np.asarray(reference, dtype=float)

        if ref.ndim == 1:
            if ref.size not in (2, 4):
                raise ValueError("1-D reference must contain 2 or 4 values")
            ref = np.repeat(ref[None, :], self.horizon, axis=0)

        if ref.shape == (self.horizon, 2):
            ref = np.column_stack((ref[:, 0], ref[:, 1], ref[:, 0], ref[:, 1]))
        elif ref.shape != (self.horizon, 4):
            raise ValueError(
                "reference must have shape "
                f"{(self.horizon, 2)} or {(self.horizon, 4)}, got {ref.shape}"
            )

        if not np.all(np.isfinite(ref)):
            raise ValueError("reference must contain only finite values")
        return ref

    def normalize_known_input(
        self,
        known_acceleration: float | Array | Sequence[float],
    ) -> Array:
        d = np.asarray(known_acceleration, dtype=float)
        if d.ndim == 0:
            result = np.full(self.horizon, float(d), dtype=float)
        else:
            result = d.reshape(-1)
            if result.size != self.horizon:
                raise ValueError(
                    "known_acceleration must be a scalar or contain "
                    f"{self.horizon} values"
                )
        if not np.all(np.isfinite(result)):
            raise ValueError("known_acceleration must contain only finite values")
        return result

    def derivative(
        self,
        state: Array | Sequence[float],
        total_input: float,
    ) -> Array:
        """Continuous derivative ``dX/dt = A X + B u``."""
        x = self.normalize_state(state)
        u = float(total_input)
        return self.continuous_A @ x + self.continuous_B[:, 0] * u

    def make_discrete_system(self, dt: float) -> DiscreteSystem:
        """Create the same LIPCPM model on another time grid."""
        return discretize(
            self.continuous_A,
            self.continuous_B,
            float(dt),
            method=self.discretization,
        )

    def solve(
        self,
        state: Array | Sequence[float],
        reference: Array | Sequence[float],
        *,
        known_acceleration: float | Array | Sequence[float] = 0.0,
    ) -> LIPCPMControlResult:
        """Solve one receding-horizon problem.

        The supplied ``mpc.py`` solves

            X = Abar x + Bbar U_mpc

        while vnoid may provide a known additive acceleration ``D`` (for
        example the base-orientation recovery term).  We want

            X = Abar x + Bbar (U_mpc + D).

        Therefore ``Bbar D`` is moved into the effective reference and the
        original ``CondensedMPC`` can be reused without changing ``mpc.py``.
        """
        x = self.normalize_state(state)
        ref = self.normalize_reference(reference)
        known = self.normalize_known_input(known_acceleration)

        known_offset = (
            self.mpc.Bbar @ known.reshape(-1)
        ).reshape(self.horizon, self.system.nx)
        effective_reference = ref - known_offset

        solution = self.mpc.solve(x, effective_reference)
        mpc_inputs = np.asarray(solution.sequence, dtype=float).reshape(
            self.horizon, self.system.nu
        )[:, 0]

        input_was_limited = False
        if self.mpc_input_limit is not None:
            clipped = np.clip(
                mpc_inputs,
                -self.mpc_input_limit,
                self.mpc_input_limit,
            )
            input_was_limited = not np.array_equal(clipped, mpc_inputs)
            mpc_inputs = clipped

        total_inputs = mpc_inputs + known
        predicted = (
            self.mpc.Abar @ x
            + self.mpc.Bbar @ total_inputs.reshape(-1)
        ).reshape(self.horizon, self.system.nx)

        # Re-evaluate J in the original coordinates.  This is also valid after
        # the optional post-solve clamp.
        error = (predicted - ref).reshape(-1)
        uvec = mpc_inputs.reshape(-1)
        objective = float(error @ self.mpc.L @ error + uvec @ self.mpc.W @ uvec)

        return LIPCPMControlResult(
            state=x.copy(),
            reference=ref.copy(),
            predicted_states=predicted.copy(),
            mpc_inputs=mpc_inputs.copy(),
            known_inputs=known.copy(),
            total_inputs=total_inputs.copy(),
            first_mpc_input=float(mpc_inputs[0]),
            first_known_input=float(known[0]),
            first_total_input=float(total_inputs[0]),
            objective=objective,
            hessian_condition=float(solution.hessian_condition),
            input_was_limited=bool(input_was_limited),
        )

    # Backward-compatible spelling used by older integration snippets.
    step = solve
