"""Adapter from vnoid walking targets to the one-axis LIPCPM-MPC.

Design used here
----------------
1. Keep vnoid's stepping-controller / footstep-planner target generation.
2. Do *not* add a separate preview controller.
3. Reuse ``FootstepPlanner.plan()`` and ``FootstepPlanner.generate_dcm()`` on
   copied future footsteps so the live vnoid footstep queue is never mutated.
4. Expand vnoid's piecewise-constant ZMP target and time-varying DCM target on
   the simulator time grid, then generate a nominal future CoM target with the
   same relation used by vnoid when ``dcm_ref = dcm_target``:

       com_vel = (dcm_target - com_pos) / T
       com_pos += com_vel * dt

5. LIPCPM-MPC tracks that future CoM target together with the measured liquid
   state.
6. Hold the first optimized LIPCPM input over one MPC interval (ZOH) and
   propagate the LIPCPM state at the vnoid simulator period.  This avoids the
   position/velocity inconsistency of linear interpolation between MPC samples.
7. Convert the propagated CoM state to vnoid-facing DCM/ZMP references.  The
   ZMP conversion is a standard-LIPM *interface conversion*, not a claim that
   the paper's LIPCPM directly outputs ZMP.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from .controller import LIPCPMControlResult, LIPCPMController

Array = np.ndarray
_EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class VnoidTargetPhase:
    """One nominal vnoid phase used only for prediction."""

    duration: float
    zmp: Array
    dcm_start: Array
    support_side: int


@dataclass(frozen=True, slots=True)
class VnoidNominalReference:
    """Future *target* trajectory supplied to LIPCPM-MPC.

    All sampled arrays have length ``controller.horizon`` and correspond to
    k+1 ... k+N.  ``mpc_reference`` is the two-column robot CoM target consumed
    by :class:`LIPCPMController`; the controller expands it to four states by
    setting the desired liquid motion equal to the robot motion.
    """

    time: Array
    zmp_target: Array
    dcm_target: Array
    com_position_target: Array
    com_velocity_target: Array
    phases: tuple[VnoidTargetPhase, ...]

    @property
    def mpc_reference(self) -> Array:
        return np.column_stack(
            (self.com_position_target, self.com_velocity_target)
        )


@dataclass(frozen=True, slots=True)
class VnoidAdapterStep:
    """Result of one vnoid simulator tick."""

    mpc_was_updated: bool
    control: LIPCPMControlResult
    nominal_reference: VnoidNominalReference | None
    propagated_state: Array
    applied_total_input: float
    com_acceleration_ref: float
    dcm_ref: float
    zmp_ref_raw: float
    zmp_ref_applied: float
    recovery_acceleration: float


class VnoidTargetReferenceBuilder:
    """Generate future CoM targets with vnoid's own DCM-based target logic.

    This class deliberately does not implement Kajita Preview Control.  It uses
    the existing vnoid ``FootstepPlanner.plan`` / ``generate_dcm`` methods on a
    prediction-only copy of the corrected future footsteps.
    """

    _MOTION_FIELDS = (
        "stride",
        "sway",
        "turn",
        "spacing",
        "climb",
        "duration",
        "stepping",
    )

    def __init__(
        self,
        controller: LIPCPMController,
        *,
        simulation_dt: float,
        axis: int = 0,
    ) -> None:
        if simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive")
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (x) or 1 (y)")

        ratio_float = controller.dt / float(simulation_dt)
        ratio = int(round(ratio_float))
        if ratio < 1 or not np.isclose(
            ratio_float, ratio, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(
                "controller.dt must be an integer multiple of simulation_dt; "
                f"got controller.dt={controller.dt}, simulation_dt={simulation_dt}"
            )

        self.controller = controller
        self.simulation_dt = float(simulation_dt)
        self.axis = int(axis)
        self.update_ratio = ratio

    @staticmethod
    def _steps(container: Any) -> list[Any]:
        if container is None or not hasattr(container, "steps"):
            return []
        return list(container.steps)

    @staticmethod
    def _copy_step(step: Any) -> Any:
        if hasattr(step, "copy") and callable(step.copy):
            return step.copy()
        return copy.deepcopy(step)

    @staticmethod
    def _make_footstep_like(template: Any, steps: list[Any]) -> Any:
        cls = type(template)
        try:
            return cls(steps=steps)
        except Exception:
            return SimpleNamespace(steps=steps)

    @staticmethod
    def _set_motion_from(dst: Any, src: Any) -> None:
        for name in VnoidTargetReferenceBuilder._MOTION_FIELDS:
            if hasattr(src, name):
                setattr(dst, name, copy.deepcopy(getattr(src, name)))

    @staticmethod
    def _foot_positions(step: Any) -> Array:
        """Return 2x3 foot positions for either vnoid Step representation."""
        if hasattr(step, "foot_pos"):
            pos = np.asarray(step.foot_pos, dtype=float)
            if pos.shape == (2, 3):
                return pos.copy()
        if hasattr(step, "foot_coords"):
            coords = step.foot_coords
            if coords is not None and len(coords) >= 2:
                return np.vstack(
                    [
                        np.asarray(coords[0].pos, dtype=float).reshape(3),
                        np.asarray(coords[1].pos, dtype=float).reshape(3),
                    ]
                )
        raise AttributeError("step must provide foot_pos or foot_coords[].pos")

    def _terminal_phase(self, step: Any, param: Any) -> VnoidTargetPhase:
        feet = self._foot_positions(step)
        zmp = 0.5 * (feet[0] + feet[1])
        dcm = zmp + np.array([0.0, 0.0, float(param.com_height)])
        duration = max(float(getattr(step, "duration", self.controller.dt)), self.controller.dt)
        return VnoidTargetPhase(
            duration=duration,
            zmp=zmp,
            dcm_start=dcm,
            support_side=int(getattr(step, "side", -1)),
        )

    def _make_corrected_future_plan(
        self,
        *,
        param: Any,
        footstep: Any,
        footstep_buffer: Any,
        footstep_planner: Any,
        landing_dcm: Array,
    ) -> Any | None:
        """Build a prediction-only future plan beginning after the next landing.

        The first predicted step uses the corrected geometry in
        ``footstep_buffer.steps[1]``.  Motion commands for that and later steps
        are taken from the remaining planner queue.  ``plan()`` then regenerates
        later foot positions relative to the corrected landing, and
        ``generate_dcm()`` regenerates future ZMP/DCM values.
        """
        planner_steps = self._steps(footstep)
        buffer_steps = self._steps(footstep_buffer)

        if len(planner_steps) < 2 and len(buffer_steps) < 2:
            return None

        if len(buffer_steps) >= 2:
            first = self._copy_step(buffer_steps[1])
        elif len(planner_steps) >= 2:
            first = self._copy_step(planner_steps[1])
        else:
            return None

        if len(planner_steps) >= 2:
            self._set_motion_from(first, planner_steps[1])

        predicted_steps = [first]
        for src in planner_steps[2:]:
            predicted_steps.append(self._copy_step(src))

        prediction = self._make_footstep_like(footstep, predicted_steps)

        # Re-plan later foot positions from the corrected first-step geometry.
        if len(predicted_steps) >= 2:
            footstep_planner.plan(param, prediction)
            prediction.steps[0].dcm = np.asarray(landing_dcm, dtype=float).copy()
            footstep_planner.generate_dcm(param, prediction)
        else:
            # generate_dcm() treats the only step as the terminal stop and would
            # overwrite the externally specified landing DCM.  Keep this case as
            # an explicit terminal hold instead.
            phase = self._terminal_phase(prediction.steps[0], param)
            prediction.steps[0].zmp = phase.zmp.copy()
            prediction.steps[0].dcm = phase.dcm_start.copy()

        return prediction

    def _build_phases(
        self,
        *,
        param: Any,
        stepping_controller: Any,
        footstep: Any,
        footstep_buffer: Any,
        footstep_planner: Any,
        centroid: Any,
    ) -> tuple[VnoidTargetPhase, ...]:
        h = float(param.com_height)
        T = float(param.T)
        if h <= 0.0 or T <= 0.0:
            raise ValueError("param.com_height and param.T must be positive")

        current_zmp = np.asarray(centroid.zmp_target, dtype=float).reshape(3)
        current_dcm = np.asarray(centroid.dcm_target, dtype=float).reshape(3)
        if not np.all(np.isfinite(current_zmp)) or not np.all(np.isfinite(current_dcm)):
            raise ValueError("centroid target values must be finite")

        remaining = float(getattr(stepping_controller, "time_to_landing", 0.0))
        if not np.isfinite(remaining):
            remaining = 0.0
        remaining = max(0.0, remaining)

        buffer_steps = self._steps(footstep_buffer)
        planner_steps = self._steps(footstep)
        current_side = (
            int(getattr(buffer_steps[0], "side", -1))
            if buffer_steps
            else int(getattr(planner_steps[0], "side", -1))
            if planner_steps
            else -1
        )

        phases: list[VnoidTargetPhase] = []
        offset = np.array([0.0, 0.0, h])

        if remaining > _EPS:
            phases.append(
                VnoidTargetPhase(
                    duration=remaining,
                    zmp=current_zmp.copy(),
                    dcm_start=current_dcm.copy(),
                    support_side=current_side,
                )
            )
            landing_dcm = (
                current_zmp
                + offset
                + math.exp(remaining / T)
                * (current_dcm - (current_zmp + offset))
            )
        else:
            landing_dcm = current_dcm.copy()

        future = self._make_corrected_future_plan(
            param=param,
            footstep=footstep,
            footstep_buffer=footstep_buffer,
            footstep_planner=footstep_planner,
            landing_dcm=landing_dcm,
        )

        if future is not None:
            for step in self._steps(future):
                duration = float(getattr(step, "duration", self.controller.dt))
                if not np.isfinite(duration) or duration <= 0.0:
                    duration = self.controller.dt
                zmp = np.asarray(step.zmp, dtype=float).reshape(3)
                dcm = np.asarray(step.dcm, dtype=float).reshape(3)
                phases.append(
                    VnoidTargetPhase(
                        duration=duration,
                        zmp=zmp.copy(),
                        dcm_start=dcm.copy(),
                        support_side=int(getattr(step, "side", -1)),
                    )
                )

        if not phases:
            # Standing/terminal fallback: hold the current targets.
            phases.append(
                VnoidTargetPhase(
                    duration=self.controller.dt * self.controller.horizon,
                    zmp=current_zmp.copy(),
                    dcm_start=current_dcm.copy(),
                    support_side=current_side,
                )
            )

        return tuple(phases)

    @staticmethod
    def _phase_value(
        phases: tuple[VnoidTargetPhase, ...],
        time_from_now: float,
        T: float,
        h: float,
    ) -> tuple[Array, Array]:
        """Evaluate nominal ZMP/DCM target at ``time_from_now``."""
        t = max(0.0, float(time_from_now))
        start = 0.0
        phase = phases[-1]
        tau = phase.duration
        for candidate in phases:
            end = start + candidate.duration
            if t < end - _EPS:
                phase = candidate
                tau = max(0.0, t - start)
                break
            start = end
        # If t exceeds the explicit phase list, hold the final phase at its end.
        tau = min(tau, phase.duration)
        offset = np.array([0.0, 0.0, h])
        support_dcm = phase.zmp + offset
        dcm = support_dcm + math.exp(tau / T) * (
            phase.dcm_start - support_dcm
        )
        return phase.zmp.copy(), dcm

    def build(
        self,
        *,
        param: Any,
        stepping_controller: Any,
        footstep: Any,
        footstep_buffer: Any,
        footstep_planner: Any,
        centroid: Any,
        initial_com_position: float | None = None,
    ) -> VnoidNominalReference:
        """Build k+1...k+N robot CoM targets for LIPCPM-MPC."""
        phases = self._build_phases(
            param=param,
            stepping_controller=stepping_controller,
            footstep=footstep,
            footstep_buffer=footstep_buffer,
            footstep_planner=footstep_planner,
            centroid=centroid,
        )

        N = self.controller.horizon
        ratio = self.update_ratio
        sim_dt = self.simulation_dt
        T = float(param.T)
        h = float(param.com_height)

        if initial_com_position is None:
            com_pos = float(np.asarray(centroid.com_pos_ref, dtype=float)[self.axis])
        else:
            com_pos = float(initial_com_position)

        times = np.arange(1, N + 1, dtype=float) * self.controller.dt
        zmp_samples = np.empty(N, dtype=float)
        dcm_samples = np.empty(N, dtype=float)
        com_pos_samples = np.empty(N, dtype=float)
        com_vel_samples = np.empty(N, dtype=float)

        sample_index = 0
        com_vel = float(np.asarray(centroid.com_vel_ref, dtype=float)[self.axis])

        # Reproduce vnoid's simple DCM->CoM target generation on the same
        # simulator time grid, then sample at the MPC period.
        for tick in range(1, N * ratio + 1):
            t = tick * sim_dt
            zmp_vec, dcm_vec = self._phase_value(phases, t, T, h)
            dcm_axis = float(dcm_vec[self.axis])

            com_vel = (dcm_axis - com_pos) / T
            com_pos += com_vel * sim_dt

            if tick % ratio == 0:
                zmp_samples[sample_index] = float(zmp_vec[self.axis])
                dcm_samples[sample_index] = dcm_axis
                com_pos_samples[sample_index] = com_pos
                com_vel_samples[sample_index] = com_vel
                sample_index += 1

        if sample_index != N:
            raise RuntimeError("internal sampling error while building vnoid reference")

        return VnoidNominalReference(
            time=times,
            zmp_target=zmp_samples,
            dcm_target=dcm_samples,
            com_position_target=com_pos_samples,
            com_velocity_target=com_vel_samples,
            phases=phases,
        )


class VnoidLIPCPMAdapter:
    """Run LIPCPM-MPC above vnoid's selected horizontal axis.

    Recommended hook point in Python ``Stabilizer.Update``::

        CalcBaseTilt(...)
        CalcDcmDynamics(...)          # leave ordinary vnoid update for y/z
        lipcpm_adapter.update_after_calc_dcm_dynamics(...)
        centroid.force_ref = ...      # MUST be after the adapter
        CalcForceDistribution(...)

    If ``use_cpp_stabilizer=True``, the C++ Stabilizer performs force
    distribution inside ``Update()``.  In that mode a C++/binding hook at the
    same internal position is required; calling this adapter after Update() is
    too late.
    """

    def __init__(
        self,
        controller: LIPCPMController,
        *,
        simulation_dt: float,
        axis: int = 0,
        clip_single_support_zmp: bool = True,
    ) -> None:
        if simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive")
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (x) or 1 (y)")

        ratio_float = controller.dt / float(simulation_dt)
        ratio = int(round(ratio_float))
        if ratio < 1 or not np.isclose(
            ratio_float, ratio, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(
                "controller.dt must be an integer multiple of simulation_dt; "
                f"got controller.dt={controller.dt}, simulation_dt={simulation_dt}"
            )

        self.controller = controller
        self.simulation_dt = float(simulation_dt)
        self.axis = int(axis)
        self.update_ratio = ratio
        self.clip_single_support_zmp = bool(clip_single_support_zmp)

        self.reference_builder = VnoidTargetReferenceBuilder(
            controller,
            simulation_dt=self.simulation_dt,
            axis=self.axis,
        )
        self.fast_system = controller.make_discrete_system(self.simulation_dt)

        self._ticks_since_solve = ratio  # force solve on first call
        self._last_control: LIPCPMControlResult | None = None
        self._last_nominal: VnoidNominalReference | None = None
        self._fast_state: Array | None = None
        self._held_total_input = 0.0

    @property
    def mpc_update_due(self) -> bool:
        return self._last_control is None or self._ticks_since_solve >= self.update_ratio

    @property
    def last_control(self) -> LIPCPMControlResult | None:
        return self._last_control

    @property
    def last_nominal_reference(self) -> VnoidNominalReference | None:
        return self._last_nominal

    def reset(self) -> None:
        self._ticks_since_solve = self.update_ratio
        self._last_control = None
        self._last_nominal = None
        self._fast_state = None
        self._held_total_input = 0.0

    @staticmethod
    def make_state(
        *,
        robot_position: float,
        robot_velocity: float,
        liquid_position: float,
        liquid_velocity: float,
    ) -> Array:
        x = np.array(
            [robot_position, robot_velocity, liquid_position, liquid_velocity],
            dtype=float,
        )
        if not np.all(np.isfinite(x)):
            raise ValueError("robot/liquid state must contain only finite values")
        return x

    @staticmethod
    def _rotate(orientation: Any, vector: Array) -> Array:
        if hasattr(orientation, "apply"):
            return np.asarray(orientation.apply(vector), dtype=float)
        if hasattr(orientation, "rot"):
            return np.asarray(orientation.rot, dtype=float) @ vector
        matrix = np.asarray(orientation, dtype=float)
        if matrix.shape != (3, 3):
            raise TypeError("orientation must provide apply(), rot, or be 3x3")
        return matrix @ vector

    @staticmethod
    def _inv_rotate(orientation: Any, vector: Array) -> Array:
        if hasattr(orientation, "inv") and hasattr(orientation.inv(), "apply"):
            return np.asarray(orientation.inv().apply(vector), dtype=float)
        if hasattr(orientation, "rot"):
            return np.asarray(orientation.rot, dtype=float).T @ vector
        matrix = np.asarray(orientation, dtype=float)
        if matrix.shape != (3, 3):
            raise TypeError("orientation must provide inv()/apply(), rot, or be 3x3")
        return matrix.T @ vector

    def recovery_acceleration(
        self,
        *,
        param: Any,
        stabilizer: Any,
        base: Any,
        theta: Array | Sequence[float],
        omega: Array | Sequence[float],
    ) -> float:
        """Reproduce vnoid CalcDcmDynamics' orientation-recovery acceleration."""
        theta = np.asarray(theta, dtype=float).reshape(3)
        omega = np.asarray(omega, dtype=float).reshape(3)

        omegadd_local = np.array(
            [
                -(stabilizer.orientation_ctrl_gain_p * theta[0]
                  + stabilizer.orientation_ctrl_gain_d * omega[0]),
                -(stabilizer.orientation_ctrl_gain_p * theta[1]
                  + stabilizer.orientation_ctrl_gain_d * omega[1]),
                0.0,
            ],
            dtype=float,
        )
        inertia = np.asarray(param.nominal_inertia, dtype=float).reshape(3)
        recovery_moment_local = inertia * omegadd_local
        recovery_moment_local = np.clip(
            recovery_moment_local,
            -float(stabilizer.recovery_moment_limit),
            float(stabilizer.recovery_moment_limit),
        )
        recovery_moment_world = self._rotate(base.ori_ref, recovery_moment_local)

        mass = float(param.total_mass)
        height = float(param.com_height)
        if mass <= 0.0 or height <= 0.0:
            raise ValueError("param.total_mass and param.com_height must be positive")

        delta = np.array(
            [
                -recovery_moment_world[1] / (mass * height),
                recovery_moment_world[0] / (mass * height),
                0.0,
            ],
            dtype=float,
        )
        return float(delta[self.axis])

    def _clip_zmp_if_single_support(
        self,
        *,
        candidate_world: Array,
        param: Any,
        foot: Sequence[Any] | None,
    ) -> Array:
        if not self.clip_single_support_zmp or foot is None or len(foot) < 2:
            return candidate_world

        c0 = bool(getattr(foot[0], "contact_ref", False))
        c1 = bool(getattr(foot[1], "contact_ref", False))
        if c0 == c1:  # both contact or both not contact
            return candidate_world

        sup = 0 if c0 else 1
        pos_ref = np.asarray(foot[sup].pos_ref, dtype=float).reshape(3)
        ori_ref = foot[sup].ori_ref
        local = self._inv_rotate(ori_ref, candidate_world - pos_ref)
        local = np.clip(
            local,
            np.asarray(param.zmp_min, dtype=float),
            np.asarray(param.zmp_max, dtype=float),
        )
        return pos_ref + self._rotate(ori_ref, local)

    def _propagate_and_apply(
        self,
        *,
        centroid: Any,
        param: Any,
        foot: Sequence[Any] | None,
    ) -> tuple[Array, float, float, float]:
        if self._fast_state is None:
            raise RuntimeError("MPC state is not initialized")

        self._fast_state = (
            self.fast_system.A @ self._fast_state
            + self.fast_system.B[:, 0] * self._held_total_input
        )
        derivative = self.controller.derivative(
            self._fast_state,
            self._held_total_input,
        )

        com_position = float(self._fast_state[0])
        com_velocity = float(self._fast_state[1])
        com_acceleration = float(derivative[1])

        T = math.sqrt(
            self.controller.params.com_height / self.controller.params.gravity
        )
        dcm_ref = com_position + T * com_velocity

        # Standard-LIPM interface conversion for the existing vnoid force/ZMP
        # path.  LIPCPM-MPC itself does not optimize a ZMP state.
        zmp_raw = (
            com_position
            - self.controller.params.com_height
            / self.controller.params.gravity
            * com_acceleration
        )

        centroid.com_pos_ref[self.axis] = com_position
        centroid.com_vel_ref[self.axis] = com_velocity
        centroid.com_acc_ref[self.axis] = com_acceleration
        centroid.dcm_ref[self.axis] = dcm_ref

        candidate = np.asarray(centroid.zmp_ref, dtype=float).reshape(3).copy()
        candidate[self.axis] = zmp_raw
        applied = self._clip_zmp_if_single_support(
            candidate_world=candidate,
            param=param,
            foot=foot,
        )
        centroid.zmp_ref[:] = applied

        return self._fast_state.copy(), com_acceleration, dcm_ref, float(applied[self.axis])

    def update_after_calc_dcm_dynamics(
        self,
        *,
        timer: Any,
        param: Any,
        stabilizer: Any,
        centroid: Any,
        base: Any,
        theta: Array | Sequence[float],
        omega: Array | Sequence[float],
        liquid_position: float,
        liquid_velocity: float,
        stepping_controller: Any,
        footstep: Any,
        footstep_buffer: Any,
        footstep_planner: Any,
        foot: Sequence[Any] | None = None,
        robot_position: float | None = None,
        robot_velocity: float | None = None,
    ) -> VnoidAdapterStep:
        """Update the LIPCPM axis for one vnoid control tick.

        Call immediately after the ordinary ``CalcDcmDynamics`` and before
        ``centroid.force_ref`` / ``CalcForceDistribution``.
        """
        if not np.isclose(float(timer.dt), self.simulation_dt):
            raise ValueError(
                f"timer.dt={timer.dt} does not match simulation_dt={self.simulation_dt}"
            )

        # The nominal vnoid target generator and the LIPCPM interface conversion
        # must use the same pendulum height/gravity.  Otherwise DCM target,
        # DCM ref and ZMP ref are defined with different time constants.
        if not np.isclose(
            float(param.com_height),
            self.controller.params.com_height,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "vnoid param.com_height and LIPCPM params.com_height must match"
            )
        if not np.isclose(
            float(param.gravity),
            self.controller.params.gravity,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "vnoid param.gravity and LIPCPM params.gravity must match"
            )

        if robot_position is None:
            if not hasattr(centroid, "com_pos"):
                raise AttributeError("pass robot_position or provide centroid.com_pos")
            robot_position = float(np.asarray(centroid.com_pos)[self.axis])
        if robot_velocity is None:
            if not hasattr(centroid, "com_vel"):
                raise AttributeError("pass robot_velocity or provide centroid.com_vel")
            robot_velocity = float(np.asarray(centroid.com_vel)[self.axis])

        recovery = self.recovery_acceleration(
            param=param,
            stabilizer=stabilizer,
            base=base,
            theta=theta,
            omega=omega,
        )

        mpc_was_updated = self.mpc_update_due
        nominal: VnoidNominalReference | None = None

        if mpc_was_updated:
            nominal = self.reference_builder.build(
                param=param,
                stepping_controller=stepping_controller,
                footstep=footstep,
                footstep_buffer=footstep_buffer,
                footstep_planner=footstep_planner,
                centroid=centroid,
            )
            state = self.make_state(
                robot_position=float(robot_position),
                robot_velocity=float(robot_velocity),
                liquid_position=float(liquid_position),
                liquid_velocity=float(liquid_velocity),
            )

            self._last_control = self.controller.solve(
                state,
                nominal.mpc_reference,
                # Current orientation recovery is treated as a known additive
                # acceleration and held over the prediction horizon.  This is
                # an integration assumption, not part of mpc.py/model.py.
                known_acceleration=recovery,
            )
            self._last_nominal = nominal
            self._fast_state = state.copy()
            self._held_total_input = self._last_control.first_total_input
            self._ticks_since_solve = 0

        if self._last_control is None:
            raise RuntimeError("MPC solve did not produce a control result")

        propagated, com_acc, dcm_ref, zmp_applied = self._propagate_and_apply(
            centroid=centroid,
            param=param,
            foot=foot,
        )
        zmp_raw = (
            float(propagated[0])
            - self.controller.params.com_height
            / self.controller.params.gravity
            * com_acc
        )

        self._ticks_since_solve += 1

        return VnoidAdapterStep(
            mpc_was_updated=mpc_was_updated,
            control=self._last_control,
            nominal_reference=nominal,
            propagated_state=propagated,
            applied_total_input=float(self._held_total_input),
            com_acceleration_ref=float(com_acc),
            dcm_ref=float(dcm_ref),
            zmp_ref_raw=float(zmp_raw),
            zmp_ref_applied=float(zmp_applied),
            recovery_acceleration=float(recovery),
        )
