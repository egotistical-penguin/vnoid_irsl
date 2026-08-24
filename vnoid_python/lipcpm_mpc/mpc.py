"""Condensed unconstrained MPC matching Eqs. (20)-(22)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .model import DiscreteSystem

Array = np.ndarray


def lifted_prediction_matrices(A: Array, B: Array, horizon: int) -> tuple[Array, Array]:
    """Build A_bar and B_bar for stacked X=[x1,...,x_h]."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    nx, nu = B.shape
    Abar = np.zeros((horizon * nx, nx), dtype=float)
    Bbar = np.zeros((horizon * nx, horizon * nu), dtype=float)

    powers = [np.eye(nx, dtype=float)]
    for _ in range(horizon):
        powers.append(powers[-1] @ A)

    for row in range(horizon):
        Abar[row * nx : (row + 1) * nx] = powers[row + 1]
        for col in range(row + 1):
            Bbar[
                row * nx : (row + 1) * nx,
                col * nu : (col + 1) * nu,
            ] = powers[row - col] @ B
    return Abar, Bbar


@dataclass(slots=True)
class MPCSolution:
    control: float | Array
    sequence: Array
    predicted_states: Array
    objective: float
    hessian_condition: float


class CondensedMPC:
    """Dense QP controller for the paper's unconstrained formulation.

    J = (Abar*x + Bbar*U - Xref)^T L (...) + U^T W U
    P = 2(Bbar^T L Bbar + W)
    q = 2 Bbar^T L(Abar*x-Xref)

    With no inequality constraints in the paper, the optimum satisfies P U=-q.
    """

    def __init__(
        self,
        system: DiscreteSystem,
        *,
        horizon: int,
        state_weights: Array,
        input_weight: float | Array,
        regularization: float = 1.0e-12,
    ) -> None:
        self.system = system
        self.horizon = int(horizon)
        self.nx = system.nx
        self.nu = system.nu
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

        q = np.asarray(state_weights, dtype=float)
        if q.ndim == 1:
            if q.size != self.nx:
                raise ValueError(f"state_weights must contain {self.nx} values")
            Q = np.diag(q)
        elif q.shape == (self.nx, self.nx):
            Q = q
        else:
            raise ValueError("state_weights has invalid shape")
        if np.any(np.linalg.eigvalsh(0.5 * (Q + Q.T)) < -1.0e-12):
            raise ValueError("state weight matrix must be positive semidefinite")

        r = np.asarray(input_weight, dtype=float)
        if r.ndim == 0:
            if float(r) < 0.0:
                raise ValueError("input_weight must be non-negative")
            R = np.eye(self.nu) * float(r)
        elif r.shape == (self.nu, self.nu):
            R = r
        else:
            raise ValueError("input_weight has invalid shape")

        self.Abar, self.Bbar = lifted_prediction_matrices(
            system.A, system.B, self.horizon
        )
        self.L = np.kron(np.eye(self.horizon), Q)
        self.W = np.kron(np.eye(self.horizon), R)
        H = self.Bbar.T @ self.L @ self.Bbar + self.W
        H = 0.5 * (H + H.T)
        scale = max(1.0, float(np.linalg.norm(H, ord=2)))
        H += np.eye(H.shape[0]) * float(regularization) * scale
        self.P = 2.0 * H
        self._H = H
        self._condition = float(np.linalg.cond(H))
        try:
            self._cho = cho_factor(H, lower=True, check_finite=True)
        except np.linalg.LinAlgError:
            self._cho = None

    def _reference_stack(self, reference: Array) -> Array:
        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 1:
            if ref.size != self.nx:
                raise ValueError(f"reference must contain {self.nx} values")
            ref = np.repeat(ref[None, :], self.horizon, axis=0)
        if ref.shape != (self.horizon, self.nx):
            raise ValueError(
                f"reference must have shape {(self.horizon, self.nx)}, got {ref.shape}"
            )
        return ref.reshape(-1)

    def solve(self, state: Array, reference: Array) -> MPCSolution:
        x = np.asarray(state, dtype=float).reshape(self.nx)
        xref = self._reference_stack(reference)
        offset = self.Abar @ x - xref
        linear_half = self.Bbar.T @ self.L @ offset
        # P U = -q is equivalent to H U = -B^T L offset.
        if self._cho is not None:
            sequence = cho_solve(self._cho, -linear_half, check_finite=True)
        else:
            sequence = np.linalg.lstsq(self._H, -linear_half, rcond=None)[0]
        predicted = (self.Abar @ x + self.Bbar @ sequence).reshape(
            self.horizon, self.nx
        )
        error = predicted.reshape(-1) - xref
        objective = float(error @ self.L @ error + sequence @ self.W @ sequence)
        controls = sequence.reshape(self.horizon, self.nu)
        first = controls[0].copy()
        control: float | Array = float(first[0]) if self.nu == 1 else first
        return MPCSolution(
            control=control,
            sequence=controls,
            predicted_states=predicted,
            objective=objective,
            hessian_condition=self._condition,
        )
