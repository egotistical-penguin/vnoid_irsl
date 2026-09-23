"""Joint-PD Gaussian-RBF feedback-error learning for G1 walking.

No-W0 / no-prior-data total-target variant.

The phase RBF geometry is generated analytically from:

    [sin(2*pi*phase), cos(2*pi*phase), time_to_landing / step_duration]

For each leg joint, that phase basis is additionally conditioned only by the
planned joint reference velocity dq_ref.  q_ref is deliberately omitted because
q_ref * psi was found to be almost collinear with the phase basis itself.
No actual joint state/error is used as an RBF input.

Learning is delayed to support-step boundaries.  During one support step the
weights used for prediction are frozen.  The target feedforward torque is:

    tau_target = clip(tau_FEL_injected + tau_PD_remaining, FEL output limits)

A ridge fit estimates the complete target FF model for the completed step, then
the old model is blended toward that target.  This differs from residual
accumulation: the remaining PD is not repeatedly added to the existing weights.
"""

import os

import numpy as np


class GaussianRBFFELController:
    """Joint-PD FEL with analytic Gaussian RBFs and step-wise ridge learning."""

    LEG_SUFFIXES = (
        'hip_yaw_link',
        'hip_roll_link',
        'hip_pitch_link',
        'knee_link',
        'ankle_pitch_link',
        'ankle_roll_link',
    )

    PHASE_FEATURE_NAMES = (
        'phase_sin',
        'phase_cos',
        'time_to_landing_norm',
    )
    FEATURE_NAMES = PHASE_FEATURE_NAMES + (
        'dq_ref_radps',
    )

    SIDE_COUNT = 2
    ACTIVE_JOINT_COUNT = 12

    def __init__(self):
        self.enabled = False
        self.ridge_lambda = 1.0e-2
        self.learning_gain = 0.05
        self.forgetting_factor = 0.95
        self.output_limit_ratio = 0.10
        self.teacher_limit_ratio = 0.20
        self.contact_guard_time = 0.05
        self.coverage_threshold = 1.0e-4
        self.coverage_sum_threshold = 1.0e-6
        self.min_samples_per_step = 20
        self.num_centers = 32
        self.rbf_width_scale = 1.5
        # Reference-conditioning scale keeps dq_ref dimensionless.
        # It can be overridden without changing the SimMain API.
        self.dq_ref_scale_radps = 1.0
        self.design_size = 0

        self.num_joints = 0
        self.active_mask = None
        self.active_indices = None
        self.active_joint_names = ()
        self.output_enabled_local = None
        self.disabled_joint_names = ()
        self.torque_scale = None
        self.fel_limit = None
        self.teacher_limit = None

        self.feature_mean = None
        self.feature_std = None
        self.centers = None
        self.sigma = 1.0
        self.weights = None
        self.model_ready = False

        self.support_side = -1
        self.phase = np.nan
        self.step_key = None
        self.step_tbegin = np.nan
        self.step_duration = np.nan
        self.time_to_landing = np.nan
        self.time_to_landing_norm = np.nan
        self.time_from_step_start = np.nan
        self.valid_active_step = False
        self.walking_active = False
        self.contact_guard_active = False
        self.learning_gate = False
        self.coverage_valid = False

        self.last_feature = np.full(len(self.PHASE_FEATURE_NAMES), np.nan)
        self.last_features = None
        self.last_psi = None
        self.last_max_activation = 0.0
        self.last_activation_sum = 0.0
        self.last_torque = None
        self.last_raw_torque = None
        self.last_teacher_pd = None
        self.last_teacher_pd_raw = None
        self.last_target_ff = None
        self.last_target_ff_raw = None
        self.last_output_clipped = None
        self.last_anti_windup_blocked = None
        self.last_weight_delta_l2_norm = 0.0
        self.last_update_joint_delta_l2 = np.zeros(
            self.ACTIVE_JOINT_COUNT, dtype=float)

        self._active_key = None
        self._active_side = -1
        self._step_was_actually_walking = False
        self._step_xtx = None
        self._step_xty = None
        self._step_samples_joint = None
        self._step_anti_windup_blocked = None

        self.update_count = 0
        self.commit_count = 0
        self.total_training_samples = 0
        self.training_samples_by_side = np.zeros(
            (self.SIDE_COUNT, self.ACTIVE_JOINT_COUNT), dtype=int)
        self.completed_valid_steps = np.zeros(
            (self.SIDE_COUNT, self.ACTIVE_JOINT_COUNT), dtype=int)
        self.anti_windup_blocked_samples = np.zeros(
            self.ACTIVE_JOINT_COUNT, dtype=int)
        self.last_commit_samples = 0
        self.last_commit_delta_l2_norm = 0.0
        self.last_update_side = -1
        self.learning_time_s = 0.0
        self.clip_count = 0
        self.prediction_count = 0

    def configure(self, enabled=True,
                  ridge_lambda=1.0e-2,
                  learning_gain=0.05,
                  forgetting_factor=0.95,
                  output_limit_ratio=0.10,
                  teacher_limit_ratio=0.20,
                  contact_guard_time=0.05,
                  coverage_threshold=1.0e-4,
                  coverage_sum_threshold=1.0e-6,
                  min_samples_per_step=20,
                  num_centers=32,
                  rbf_width_scale=1.5):
        self.enabled = bool(enabled)
        self.ridge_lambda = float(ridge_lambda)
        self.learning_gain = float(learning_gain)
        self.forgetting_factor = float(forgetting_factor)
        self.output_limit_ratio = float(output_limit_ratio)
        self.teacher_limit_ratio = float(teacher_limit_ratio)
        self.contact_guard_time = float(contact_guard_time)
        self.coverage_threshold = float(coverage_threshold)
        self.coverage_sum_threshold = float(coverage_sum_threshold)
        self.min_samples_per_step = int(min_samples_per_step)
        self.num_centers = int(num_centers)
        self.rbf_width_scale = float(rbf_width_scale)
        self.dq_ref_scale_radps = float(os.environ.get(
            'HUMANOID_GAUSSIAN_RBF_FEL_DQ_REF_SCALE_RADPS', '1.0'))

        disabled = os.environ.get(
            'HUMANOID_GAUSSIAN_RBF_FEL_DISABLED_JOINTS', '').strip()
        self.disabled_joint_names = tuple(
            name.strip() for name in disabled.replace(';', ',').split(',')
            if name.strip())

        if not np.isfinite(self.ridge_lambda) or self.ridge_lambda <= 0.0:
            raise ValueError('ridge_lambda must be finite and positive')
        if (not np.isfinite(self.learning_gain) or
                not 0.0 <= self.learning_gain <= 1.0):
            raise ValueError('learning_gain must be finite and in [0, 1]')
        if (not np.isfinite(self.forgetting_factor) or
                not 0.0 <= self.forgetting_factor <= 1.0):
            raise ValueError('forgetting_factor must be finite and in [0, 1]')
        if (not np.isfinite(self.output_limit_ratio) or
                not 0.0 < self.output_limit_ratio <= 1.0):
            raise ValueError('output_limit_ratio must be finite and in (0, 1]')
        if (not np.isfinite(self.teacher_limit_ratio) or
                not 0.0 < self.teacher_limit_ratio <= 1.0):
            raise ValueError('teacher_limit_ratio must be finite and in (0, 1]')
        if (not np.isfinite(self.contact_guard_time) or
                self.contact_guard_time < 0.0):
            raise ValueError('contact_guard_time must be finite and non-negative')
        if (not np.isfinite(self.coverage_threshold) or
                self.coverage_threshold < 0.0):
            raise ValueError('coverage_threshold must be finite and non-negative')
        if (not np.isfinite(self.coverage_sum_threshold) or
                self.coverage_sum_threshold < 0.0):
            raise ValueError(
                'coverage_sum_threshold must be finite and non-negative')
        if self.min_samples_per_step <= 0:
            raise ValueError('min_samples_per_step must be positive')
        if self.num_centers < 4:
            raise ValueError('num_centers must be at least 4')
        if (not np.isfinite(self.rbf_width_scale) or
                self.rbf_width_scale <= 0.0):
            raise ValueError('rbf_width_scale must be finite and positive')
        if (not np.isfinite(self.dq_ref_scale_radps) or
                self.dq_ref_scale_radps <= 0.0):
            raise ValueError(
                'dq_ref_scale_radps must be finite and positive')

        self._initialize_analytic_geometry()

    def _initialize_analytic_geometry(self):
        """Generate the same phase-only RBF geometry used by posture FEL."""
        k = int(self.num_centers)

        # Theoretical normalization for p ~ U[0, 1):
        # sin/cos mean=0, std=1/sqrt(2); TTL_norm=1-p mean=1/2,
        # std=1/sqrt(12).  No measured walking data are used.
        self.feature_mean = np.array([0.0, 0.0, 0.5], dtype=float)
        self.feature_std = np.array([
            1.0 / np.sqrt(2.0),
            1.0 / np.sqrt(2.0),
            1.0 / np.sqrt(12.0),
        ], dtype=float)

        phase = (np.arange(k, dtype=float) + 0.5) / float(k)
        raw_centers = np.column_stack([
            np.sin(2.0 * np.pi * phase),
            np.cos(2.0 * np.pi * phase),
            1.0 - phase,
        ])
        self.centers = (
            (raw_centers - self.feature_mean[None, :]) /
            self.feature_std[None, :])
        spacing = np.linalg.norm(np.diff(self.centers, axis=0), axis=1)
        base_spacing = float(np.median(spacing))
        self.sigma = max(
            self.rbf_width_scale * base_spacing,
            1.0e-6)
        self.model_ready = True

    @staticmethod
    def _joint_vector(body, attribute):
        return np.asarray([
            float(getattr(body.joint(index), attribute))
            for index in range(body.numJoints)
        ], dtype=float)

    def initialize(self, body):
        self.num_joints = int(body.numJoints)
        joint_names = tuple(
            body.joint(index).name for index in range(self.num_joints))
        self.active_mask = np.asarray([
            (name.startswith('left_') or name.startswith('right_')) and
            any(name.endswith(suffix) for suffix in self.LEG_SUFFIXES)
            for name in joint_names
        ], dtype=bool)
        if np.count_nonzero(self.active_mask) != self.ACTIVE_JOINT_COUNT:
            raise ValueError(
                'G1 RBF-FEL leg mask expected 12 joints, found {}: {}'.format(
                    int(np.count_nonzero(self.active_mask)),
                    ', '.join(np.asarray(joint_names)[self.active_mask])))
        self.active_indices = np.flatnonzero(self.active_mask)
        self.active_joint_names = tuple(
            joint_names[index] for index in self.active_indices)

        unknown_disabled = sorted(
            set(self.disabled_joint_names) - set(self.active_joint_names))
        if unknown_disabled:
            raise ValueError(
                'Unknown HUMANOID_GAUSSIAN_RBF_FEL_DISABLED_JOINTS entries: '
                + ', '.join(unknown_disabled))
        disabled_set = set(self.disabled_joint_names)
        self.output_enabled_local = np.asarray([
            name not in disabled_set for name in self.active_joint_names
        ], dtype=bool)

        torque_lower = self._joint_vector(body, 'u_lower')
        torque_upper = self._joint_vector(body, 'u_upper')
        self.torque_scale = np.maximum(
            np.abs(torque_lower), np.abs(torque_upper))
        if (not np.all(np.isfinite(self.torque_scale)) or
                np.any(self.torque_scale <= 0.0)):
            raise ValueError('RBF-FEL requires finite positive torque limits')
        self.fel_limit = self.output_limit_ratio * self.torque_scale
        self.teacher_limit = self.teacher_limit_ratio * self.torque_scale

        if not self.model_ready:
            self._initialize_analytic_geometry()
        # One phase-RBF block plus one dq_ref-modulated copy.
        # 32 centers -> 64 parameters per joint (previously 96).
        self.design_size = 2 * self.num_centers
        self.weights = np.zeros(
            (self.SIDE_COUNT, self.ACTIVE_JOINT_COUNT, self.design_size),
            dtype=float)

        self.last_torque = np.zeros(self.num_joints, dtype=float)
        self.last_raw_torque = np.zeros(self.num_joints, dtype=float)
        self.last_teacher_pd = np.zeros(self.num_joints, dtype=float)
        self.last_teacher_pd_raw = np.zeros(self.num_joints, dtype=float)
        self.last_target_ff = np.zeros(self.num_joints, dtype=float)
        self.last_target_ff_raw = np.zeros(self.num_joints, dtype=float)
        self.last_output_clipped = np.zeros(self.num_joints, dtype=bool)
        self.last_anti_windup_blocked = np.zeros(
            self.num_joints, dtype=bool)
        self.last_features = np.full(
            (self.ACTIVE_JOINT_COUNT, len(self.FEATURE_NAMES)),
            np.nan, dtype=float)
        self.last_psi = np.zeros(
            (self.ACTIVE_JOINT_COUNT, self.design_size), dtype=float)

        self.reset_learning()

    @staticmethod
    def _active_step(walking_control):
        if (hasattr(walking_control, 'footstep_buffer') and
                walking_control.footstep_buffer is not None and
                len(walking_control.footstep_buffer.steps) > 0):
            return walking_control.footstep_buffer.steps[0]
        if (hasattr(walking_control, 'footstep') and
                walking_control.footstep is not None and
                len(walking_control.footstep.steps) > 0):
            return walking_control.footstep.steps[0]
        return None

    def _read_context(self, walking_control):
        self.valid_active_step = False
        self.walking_active = False
        self.support_side = -1
        self.phase = np.nan
        self.step_key = None
        self.step_tbegin = np.nan
        self.step_duration = np.nan
        self.time_to_landing = np.nan
        self.time_to_landing_norm = np.nan
        self.time_from_step_start = np.nan
        self.contact_guard_active = False

        active_step = self._active_step(walking_control)
        if active_step is None:
            return
        try:
            side = int(active_step.side)
            duration = float(active_step.duration)
            tbegin = float(active_step.tbegin)
            now = float(walking_control.timer.time)
            ttl = float(
                walking_control.stepping_controller.time_to_landing)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        if (side not in (0, 1) or not np.isfinite(duration) or
                duration <= 0.0 or not np.isfinite(tbegin) or
                not np.isfinite(now) or not np.isfinite(ttl)):
            return

        phase = float(np.clip(
            (now - tbegin) / duration,
            0.0, np.nextafter(1.0, 0.0)))
        ttl_norm = float(np.clip(ttl / duration, 0.0, 1.0))

        self.support_side = side
        self.phase = phase
        self.step_tbegin = tbegin
        self.step_duration = duration
        self.time_to_landing = ttl
        self.time_to_landing_norm = ttl_norm
        self.time_from_step_start = now - tbegin
        self.step_key = (side, round(tbegin, 9))
        self.walking_active = bool(getattr(active_step, 'stepping', False))
        self.valid_active_step = True
        self.contact_guard_active = bool(
            self.time_from_step_start < self.contact_guard_time or
            (0.0 <= ttl < self.contact_guard_time))

    def _feature_from_context(self):
        if not (self.valid_active_step and np.isfinite(self.phase) and
                np.isfinite(self.time_to_landing_norm)):
            return None
        return np.asarray([
            np.sin(2.0 * np.pi * self.phase),
            np.cos(2.0 * np.pi * self.phase),
            self.time_to_landing_norm,
        ], dtype=float)

    def _basis(self, feature):
        xz = ((feature - self.feature_mean) / self.feature_std)
        diff = self.centers - xz[None, :]
        d2 = np.sum(diff * diff, axis=1)
        sig = max(float(self.sigma), 1.0e-9)
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            phi = np.exp(-0.5 * d2 / (sig * sig))
        activation_sum = float(np.sum(phi))
        max_activation = float(np.max(phi)) if phi.size else 0.0
        valid = bool(
            np.isfinite(activation_sum) and
            activation_sum > self.coverage_sum_threshold and
            max_activation >= self.coverage_threshold)
        if not valid:
            return None, max_activation, activation_sum
        psi = phi / max(activation_sum, 1.0e-12)
        return psi, max_activation, activation_sum

    def _joint_design(self, phase_psi, dq_ref):
        """Phase-RBF design conditioned only by planned joint velocity.

        q_ref * psi is intentionally omitted because it is nearly collinear
        with the phase basis for the nominal walking trajectory.  Keeping only
        dq_ref * psi reduces the per-joint model from 96 to 64 parameters.
        """
        dqn = float(np.clip(
            dq_ref / self.dq_ref_scale_radps, -3.0, 3.0))
        return np.concatenate((
            phase_psi,
            phase_psi * dqn,
        ))

    def _clear_step_accumulator(self):
        k = self.design_size
        self._step_xtx = np.zeros(
            (self.ACTIVE_JOINT_COUNT, k, k), dtype=float)
        self._step_xty = np.zeros(
            (self.ACTIVE_JOINT_COUNT, k), dtype=float)
        self._step_samples_joint = np.zeros(
            self.ACTIVE_JOINT_COUNT, dtype=int)
        self._step_anti_windup_blocked = np.zeros(
            self.ACTIVE_JOINT_COUNT, dtype=int)

    def _commit_pending(self):
        """Commit one completed step using total-target ridge regression.

        For each active joint j, the samples accumulated during the step fit
        the complete desired feedforward target:

            W_target_j = argmin ||Psi W - tau_target_j||^2
                                   + lambda ||W||^2

        and the next-step model is blended toward that target:

            W_j <- k_f * W_j + k_l * W_target_j

        With the recommended k_f=0.95 and k_l=0.05 this is a convex 5% move
        toward the newly fitted FF target.  There is no W0.
        """
        if (not self.enabled or self._active_side not in (0, 1) or
                not self._step_was_actually_walking or
                self._step_samples_joint is None or
                not np.any(self._step_samples_joint > 0)):
            self._clear_step_accumulator()
            self._step_was_actually_walking = False
            return

        side = int(self._active_side)
        eye = np.eye(self.design_size, dtype=float)
        old_side = self.weights[side].copy()
        self.last_update_joint_delta_l2.fill(0.0)
        updated = False

        for local_index in range(self.ACTIVE_JOINT_COUNT):
            if not self.output_enabled_local[local_index]:
                continue
            samples = int(self._step_samples_joint[local_index])
            if samples <= 0:
                continue

            self.completed_valid_steps[side, local_index] += 1
            self.training_samples_by_side[side, local_index] += samples
            if samples < self.min_samples_per_step:
                continue

            matrix = (
                self._step_xtx[local_index] + self.ridge_lambda * eye)
            rhs = self._step_xty[local_index]
            try:
                target_weights = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                target_weights = np.linalg.lstsq(
                    matrix, rhs, rcond=None)[0]

            old = self.weights[side, local_index].copy()
            proposed = (
                self.forgetting_factor * old +
                self.learning_gain * target_weights)
            # Do not clip weights.  Saturate only the actual injected torque.
            self.weights[side, local_index] = proposed
            self.last_update_joint_delta_l2[local_index] = float(
                np.linalg.norm(proposed - old))
            updated = True

        step_samples = int(np.max(self._step_samples_joint))
        self.total_training_samples += step_samples
        if updated:
            delta = self.weights[side] - old_side
            self.last_weight_delta_l2_norm = float(np.linalg.norm(delta))
            self.last_commit_delta_l2_norm = self.last_weight_delta_l2_norm
            self.last_commit_samples = step_samples
            self.last_update_side = side
            self.update_count += 1
            self.commit_count += 1
        else:
            self.last_weight_delta_l2_norm = 0.0
            self.last_commit_delta_l2_norm = 0.0
            self.last_commit_samples = step_samples

        self._clear_step_accumulator()
        self._step_was_actually_walking = False

    def _handle_step_transition(self):
        current_key = self.step_key if self.valid_active_step else None

        if current_key != self._active_key:
            if self._active_key is not None:
                self._commit_pending()
            self._active_key = current_key
            self._active_side = (
                self.support_side if current_key is not None else -1)
            self._clear_step_accumulator()
            self._step_was_actually_walking = False
        elif self._step_was_actually_walking and not self.walking_active:
            # The final real support step can finish while the buffered key
            # remains unchanged.  Commit exactly once on stepping -> inactive.
            self._commit_pending()

        if self.walking_active:
            self._step_was_actually_walking = True

    def predict(self, walking_control, q_ref=None, dq_ref=None, *unused,
                **unused_kwargs):
        """Predict joint FEL torque from planned gait/reference quantities only.

        dq_ref is the filtered planned joint-velocity reference.  q_ref is
        accepted only for drop-in compatibility and is not used by the model.
        Actual q/dq and tracking errors are intentionally excluded.
        """
        del unused, unused_kwargs
        if self.last_torque is None:
            raise RuntimeError(
                'GaussianRBFFELController.initialize() was not called')

        if dq_ref is None:
            raise ValueError(
                'total-target joint RBF-FEL requires dq_ref')
        dq_ref = np.asarray(dq_ref, dtype=float)
        if dq_ref.shape != (self.num_joints,):
            raise ValueError('dq_ref shape mismatch')

        self._read_context(walking_control)
        self._handle_step_transition()
        self.last_torque.fill(0.0)
        self.last_raw_torque.fill(0.0)
        self.last_output_clipped.fill(False)
        self.last_anti_windup_blocked.fill(False)
        self.last_feature[:] = np.nan
        self.last_features.fill(np.nan)
        self.last_psi.fill(0.0)
        self.last_max_activation = 0.0
        self.last_activation_sum = 0.0
        self.coverage_valid = False

        if not (self.enabled and self.model_ready and self.walking_active and
                self.valid_active_step):
            return self.last_torque.copy()

        phase_feature = self._feature_from_context()
        if phase_feature is None or not np.all(np.isfinite(phase_feature)):
            return self.last_torque.copy()
        phase_psi, max_activation, activation_sum = self._basis(phase_feature)
        self.last_feature[:] = phase_feature
        self.last_max_activation = max_activation
        self.last_activation_sum = activation_sum
        if phase_psi is None:
            return self.last_torque.copy()

        if not np.all(np.isfinite(dq_ref[self.active_mask])):
            return self.last_torque.copy()

        self.coverage_valid = True
        side = self.support_side
        for local_index, joint_index in enumerate(self.active_indices):
            feature = np.asarray([
                phase_feature[0],
                phase_feature[1],
                phase_feature[2],
                dq_ref[joint_index],
            ], dtype=float)
            self.last_features[local_index] = feature

            design = self._joint_design(
                phase_psi, dq_ref[joint_index])
            self.last_psi[local_index] = design

            if not self.output_enabled_local[local_index]:
                continue
            raw = float(np.dot(self.weights[side, local_index], design))
            limit = float(self.fel_limit[joint_index])
            clipped = float(np.clip(raw, -limit, limit))
            self.last_raw_torque[joint_index] = raw
            self.last_torque[joint_index] = clipped
            clipped_flag = bool(abs(raw - clipped) > 1.0e-12)
            self.last_output_clipped[joint_index] = clipped_flag
            self.prediction_count += 1
            if clipped_flag:
                self.clip_count += 1
        return self.last_torque.copy()

    def update(self, tau_pd, ik_solver_ok, ik_command_ok, dt,
               actuator_saturated=None):
        if self.last_torque is None:
            raise RuntimeError(
                'GaussianRBFFELController.initialize() was not called')
        tau_pd = np.asarray(tau_pd, dtype=float)
        if tau_pd.shape != (self.num_joints,):
            raise ValueError('tau_pd shape mismatch')

        if actuator_saturated is None:
            actuator_saturated = np.zeros(self.num_joints, dtype=bool)
        else:
            actuator_saturated = np.asarray(
                actuator_saturated, dtype=bool)
            if actuator_saturated.shape != (self.num_joints,):
                raise ValueError('actuator_saturated shape mismatch')

        self.last_teacher_pd_raw.fill(0.0)
        self.last_teacher_pd.fill(0.0)
        self.last_target_ff_raw.fill(0.0)
        self.last_target_ff.fill(0.0)
        self.last_anti_windup_blocked.fill(False)
        if np.all(np.isfinite(tau_pd)):
            self.last_teacher_pd_raw[self.active_mask] = tau_pd[self.active_mask]
            for joint_index in self.active_indices:
                limit = float(self.teacher_limit[joint_index])
                self.last_teacher_pd[joint_index] = float(np.clip(
                    tau_pd[joint_index], -limit, limit))

        dt = float(dt)
        self.learning_gate = bool(
            self.enabled and self.model_ready and self.walking_active and
            self.valid_active_step and self.coverage_valid and
            bool(ik_solver_ok) and bool(ik_command_ok) and
            not self.contact_guard_active and self.time_to_landing >= 0.0 and
            np.all(np.isfinite(tau_pd)) and np.isfinite(dt) and dt > 0.0)
        self.last_weight_delta_l2_norm = 0.0
        if not self.learning_gate:
            return self.last_teacher_pd.copy()

        if self.last_psi is None or not np.all(np.isfinite(self.last_psi)):
            return self.last_teacher_pd.copy()

        any_sample = False
        for local_index, joint_index in enumerate(self.active_indices):
            if not self.output_enabled_local[local_index]:
                continue
            teacher = float(self.last_teacher_pd[joint_index])
            raw_output = float(self.last_raw_torque[joint_index])
            injected_output = float(self.last_torque[joint_index])
            target_raw = injected_output + teacher
            limit = float(self.fel_limit[joint_index])
            target = float(np.clip(target_raw, -limit, limit))
            self.last_target_ff_raw[joint_index] = target_raw
            self.last_target_ff[joint_index] = target

            same_direction = bool(raw_output * teacher > 0.0)
            own_saturated = bool(self.last_output_clipped[joint_index])

            # Conditional integration anti-windup.  If the FEL command is
            # already clipped and the remaining PD asks for still more torque
            # in the same direction, that sample must not push W outward.
            # Samples where the physical actuator itself saturated are also
            # discarded because the measured tracking error no longer
            # corresponds to the commanded closed-loop torque faithfully.
            blocked = bool(
                (own_saturated and same_direction) or
                actuator_saturated[joint_index])
            if blocked:
                self.last_anti_windup_blocked[joint_index] = True
                self.anti_windup_blocked_samples[local_index] += 1
                self._step_anti_windup_blocked[local_index] += 1
                continue

            design = self.last_psi[local_index]
            self._step_xtx[local_index] += np.outer(design, design)
            self._step_xty[local_index] += design * target
            self._step_samples_joint[local_index] += 1
            any_sample = True

        if any_sample:
            self.update_count += 0  # commit_count is incremented at step end.
        self.learning_time_s += dt
        return self.last_teacher_pd.copy()

    def finalize(self):
        # Match posture FEL semantics: simulation termination is not proof that
        # the current support step completed.  Completed steps were already
        # committed on step-key changes / stepping->inactive transitions;
        # discard any partial final step.
        if self._active_key is not None:
            self._clear_step_accumulator()
            self._step_was_actually_walking = False
            self._active_key = None
            self._active_side = -1

    def reset_learning(self):
        if self.last_torque is None:
            raise RuntimeError(
                'GaussianRBFFELController.initialize() was not called')
        if self.weights is not None:
            # No W0: every trial starts from a genuinely unlearned FF model.
            self.weights.fill(0.0)
        self._clear_step_accumulator()

        self.last_torque.fill(0.0)
        self.last_raw_torque.fill(0.0)
        self.last_teacher_pd.fill(0.0)
        self.last_teacher_pd_raw.fill(0.0)
        self.last_target_ff.fill(0.0)
        self.last_target_ff_raw.fill(0.0)
        self.last_output_clipped.fill(False)
        self.last_anti_windup_blocked.fill(False)
        self.last_feature[:] = np.nan
        self.last_features.fill(np.nan)
        self.last_psi.fill(0.0)
        self.last_max_activation = 0.0
        self.last_activation_sum = 0.0
        self.learning_gate = False
        self.coverage_valid = False
        self.last_weight_delta_l2_norm = 0.0
        self.last_update_joint_delta_l2.fill(0.0)

        self.support_side = -1
        self.phase = np.nan
        self.step_key = None
        self.step_tbegin = np.nan
        self.step_duration = np.nan
        self.time_to_landing = np.nan
        self.time_to_landing_norm = np.nan
        self.time_from_step_start = np.nan
        self.valid_active_step = False
        self.walking_active = False
        self.contact_guard_active = False
        self._active_key = None
        self._active_side = -1
        self._step_was_actually_walking = False

        self.update_count = 0
        self.commit_count = 0
        self.total_training_samples = 0
        self.training_samples_by_side.fill(0)
        self.completed_valid_steps.fill(0)
        self.anti_windup_blocked_samples.fill(0)
        self.last_commit_samples = 0
        self.last_commit_delta_l2_norm = 0.0
        self.last_update_side = -1
        self.learning_time_s = 0.0
        self.clip_count = 0
        self.prediction_count = 0

    def global_log_fields(self):
        weight_l2 = (
            float(np.linalg.norm(self.weights))
            if self.weights is not None else 0.0)
        weight_absmax = (
            float(np.max(np.abs(self.weights)))
            if self.weights is not None and self.weights.size else 0.0)
        clip_rate = (
            float(self.clip_count) / float(self.prediction_count)
            if self.prediction_count > 0 else 0.0)
        completed_right = int(np.max(self.completed_valid_steps[0]))
        completed_left = int(np.max(self.completed_valid_steps[1]))
        train_right = int(np.sum(self.training_samples_by_side[0]))
        train_left = int(np.sum(self.training_samples_by_side[1]))

        header = [
            'rbf_fel_enabled',
            'rbf_fel_algorithm',
            'rbf_fel_model_ready',
            'rbf_fel_num_centers',
            'rbf_fel_ridge_lambda',
            'rbf_fel_learning_gain',
            'rbf_fel_forgetting_factor',
            'rbf_fel_output_limit_ratio',
            'rbf_fel_teacher_limit_ratio',
            'rbf_fel_contact_guard_time_s',
            'rbf_fel_min_samples_per_step',
            'rbf_fel_support_side',
            'rbf_fel_step_tbegin_s',
            'rbf_fel_step_duration_s',
            'rbf_fel_phase',
            'rbf_fel_time_to_landing_s',
            'rbf_fel_time_to_landing_norm',
            'rbf_fel_walking_active',
            'rbf_fel_learning_gate',
            'rbf_fel_contact_guard_active',
            'rbf_fel_coverage_valid',
            'rbf_fel_max_activation',
            'rbf_fel_activation_sum',
            'rbf_fel_weight_l2_Nm',
            'rbf_fel_weight_absmax_Nm',
            'rbf_fel_last_weight_delta_l2_Nm',
            'rbf_fel_update_count',
            'rbf_fel_commit_count',
            'rbf_fel_last_commit_samples',
            'rbf_fel_last_commit_delta_l2_Nm',
            'rbf_fel_last_update_side',
            'rbf_fel_total_training_samples',
            'rbf_fel_training_samples_right_sum',
            'rbf_fel_training_samples_left_sum',
            'rbf_fel_completed_steps_right',
            'rbf_fel_completed_steps_left',
            'rbf_fel_clip_count',
            'rbf_fel_clip_rate',
            'rbf_fel_anti_windup_blocked_samples_sum',
            'rbf_input_phase_sin',
            'rbf_input_phase_cos',
            'rbf_input_time_to_landing_norm',
        ]
        values = [
            int(self.enabled),
            'step_ridge_total_target_dqref_no_w0',
            int(self.model_ready),
            self.num_centers,
            self.ridge_lambda,
            self.learning_gain,
            self.forgetting_factor,
            self.output_limit_ratio,
            self.teacher_limit_ratio,
            self.contact_guard_time,
            self.min_samples_per_step,
            self.support_side,
            self.step_tbegin,
            self.step_duration,
            self.phase,
            self.time_to_landing,
            self.time_to_landing_norm,
            int(self.walking_active),
            int(self.learning_gate),
            int(self.contact_guard_active),
            int(self.coverage_valid),
            self.last_max_activation,
            self.last_activation_sum,
            weight_l2,
            weight_absmax,
            self.last_weight_delta_l2_norm,
            self.update_count,
            self.commit_count,
            self.last_commit_samples,
            self.last_commit_delta_l2_norm,
            self.last_update_side,
            self.total_training_samples,
            train_right,
            train_left,
            completed_right,
            completed_left,
            self.clip_count,
            clip_rate,
            int(np.sum(self.anti_windup_blocked_samples)),
            self.last_feature[0],
            self.last_feature[1],
            self.last_feature[2],
        ]
        return header, values

    def joint_log_fields(self, joint_name, index):
        local = None
        if self.active_indices is not None:
            found = np.flatnonzero(self.active_indices == index)
            if found.size:
                local = int(found[0])

        if local is None:
            step_samples = 0
            weight_l2 = 0.0
            last_delta = 0.0
            aw_total = 0
        else:
            step_samples = int(self._step_samples_joint[local])
            side = self.support_side if self.support_side in (0, 1) else 0
            weight_l2 = float(np.linalg.norm(self.weights[side, local]))
            last_delta = float(self.last_update_joint_delta_l2[local])
            aw_total = int(self.anti_windup_blocked_samples[local])

        header = [
            '{}_tau_rbf_fel_Nm'.format(joint_name),
            '{}_tau_rbf_fel_raw_Nm'.format(joint_name),
            '{}_rbf_teacher_pd_raw_Nm'.format(joint_name),
            '{}_rbf_teacher_pd_clipped_Nm'.format(joint_name),
            '{}_rbf_target_ff_raw_Nm'.format(joint_name),
            '{}_rbf_target_ff_Nm'.format(joint_name),
            '{}_rbf_output_clipped'.format(joint_name),
            '{}_rbf_anti_windup_blocked'.format(joint_name),
            '{}_rbf_step_train_samples'.format(joint_name),
            '{}_rbf_weight_l2_Nm'.format(joint_name),
            '{}_rbf_last_weight_update_l2_Nm'.format(joint_name),
            '{}_rbf_anti_windup_blocked_samples'.format(joint_name),
        ]
        values = [
            self.last_torque[index],
            self.last_raw_torque[index],
            self.last_teacher_pd_raw[index],
            self.last_teacher_pd[index],
            self.last_target_ff_raw[index],
            self.last_target_ff[index],
            int(bool(self.last_output_clipped[index])),
            int(bool(self.last_anti_windup_blocked[index])),
            step_samples,
            weight_l2,
            last_delta,
            aw_total,
        ]
        return header, values