import os

import numpy as np


class PostureGaussianRBFFELController:
    """Posture-PD FEL for stabilizer recovery moments using Gaussian RBFs.

    The predictor only sees quantities from the planned walking reference.
    The teacher is collected *after* Stabilizer.Update() from the remaining
    orientation-feedback recovery moment. Weights are updated only when a
    real VNOID support step changes, so the predictor is fixed within a step.
    """

    # Compact posture predictor input.  X and Y use independent RBF bases
    # (centers / sigma / activations), but both see only planned gait phase.
    FEATURE_NAMES = (
        'phase_sin',
        'phase_cos',
        'time_to_landing_norm',
    )
    AXIS_NAMES = ('x', 'y')
    SIDE_COUNT = 2
    AXIS_COUNT = 2

    def __init__(self):
        self.configure()

    def configure(self, enabled=False, mode='frozen', model_path='',
                  save_path='', ridge_lambda=1.0e-2,
                  output_limit_nm=10.0, teacher_limit_nm=100.0,
                  contact_guard_time=0.05,
                  coverage_threshold=1.0e-4,
                  coverage_sum_threshold=1.0e-6,
                  payload_calibration_steps_per_side=1,
                  payload_learning_gain_x=0.30,
                  payload_learning_gain_y=0.20,
                  payload_forgetting_factor=1.0,
                  payload_min_samples_per_step=20):
        self.enabled = bool(enabled)
        self.mode = str(mode).lower()
        if self.mode not in ('baseline', 'payload', 'frozen'):
            raise ValueError(
                "Posture RBF-FEL mode must be baseline, payload, or frozen")
        self.model_path = str(model_path or '')
        self.save_path = str(save_path or '')
        self.ridge_lambda = max(float(ridge_lambda), 1.0e-12)
        self.output_limit_nm = abs(float(output_limit_nm))
        self.teacher_limit_nm = abs(float(teacher_limit_nm))
        self.contact_guard_time = max(0.0, float(contact_guard_time))
        self.coverage_threshold = max(0.0, float(coverage_threshold))
        self.coverage_sum_threshold = max(
            0.0, float(coverage_sum_threshold))
        # Online payload adaptation is updated once at every completed
        # support step.  The X/Y learning gains intentionally differ: lateral
        # adaptation is kept more conservative because walking has a smaller
        # stability margin in that direction.
        self.payload_calibration_steps_per_side = max(
            1, int(payload_calibration_steps_per_side))
        self.payload_learning_gain = np.asarray([
            max(0.0, float(payload_learning_gain_x)),
            max(0.0, float(payload_learning_gain_y)),
        ], dtype=float)
        self.payload_forgetting_factor = float(np.clip(
            payload_forgetting_factor, 0.0, 1.0))
        self.payload_min_samples_per_step = max(
            1, int(payload_min_samples_per_step))

        feature_dim = len(self.FEATURE_NAMES)
        # Geometry is axis-specific: [support_side, axis, ...].  This lets X
        # and Y choose different center placement / bandwidth even though both
        # use the same three gait-phase features.
        self.feature_mean = np.zeros(
            (self.SIDE_COUNT, self.AXIS_COUNT, feature_dim), dtype=float)
        self.feature_std = np.ones_like(self.feature_mean)
        self.centers = np.zeros(
            (self.SIDE_COUNT, self.AXIS_COUNT, 0, feature_dim), dtype=float)
        self.sigma = np.ones((self.SIDE_COUNT, self.AXIS_COUNT), dtype=float)
        # Keep weights in [side, center, axis] form for straightforward output
        # and for easier compatibility with model-generation scripts.
        self.w0 = np.zeros((self.SIDE_COUNT, 0, self.AXIS_COUNT), dtype=float)
        self.delta_w = np.zeros_like(self.w0)
        self.model_ready = False
        if self.model_path:
            self.load_model(self.model_path)
        self.reset_learning()

    @property
    def num_centers(self):
        return int(self.centers.shape[2]) if self.centers.ndim == 4 else 0

    @property
    def learning_enabled(self):
        if not self.enabled:
            return False
        # Baseline and payload modes both learn continuously.  In payload
        # mode a completed support step updates only that support side, and
        # the resulting delta weights are used online on its next occurrence.
        return self.mode in ('baseline', 'payload')

    def load_model(self, path):
        data = np.load(path, allow_pickle=False)
        feature_names = tuple(str(x) for x in data['feature_names'].tolist())
        if feature_names != self.FEATURE_NAMES:
            raise ValueError(
                'Posture RBF-FEL feature schema mismatch: {}. '
                'This X/Y 3-feature controller requires {}'.format(
                    feature_names, self.FEATURE_NAMES))

        expected_dim = len(self.FEATURE_NAMES)
        mean = np.asarray(data['feature_mean'], dtype=float)
        std = np.asarray(data['feature_std'], dtype=float)
        centers = np.asarray(data['centers'], dtype=float)
        sigma = np.asarray(data['sigma'], dtype=float)

        # New preferred schema:
        #   mean/std : [side, axis, feature]
        #   centers  : [side, axis, center, feature]
        #   sigma    : [side, axis]
        # For convenience, a shared 3-feature geometry [side, ...] is also
        # accepted and duplicated to X/Y.  Old 9-feature models are rejected
        # above because their weights were fitted against a different basis.
        if mean.shape == (self.SIDE_COUNT, expected_dim):
            mean = np.repeat(mean[:, None, :], self.AXIS_COUNT, axis=1)
        if mean.shape != (self.SIDE_COUNT, self.AXIS_COUNT, expected_dim):
            raise ValueError('Invalid Posture RBF-FEL feature_mean shape')

        if std.shape == (self.SIDE_COUNT, expected_dim):
            std = np.repeat(std[:, None, :], self.AXIS_COUNT, axis=1)
        if std.shape != mean.shape:
            raise ValueError('Invalid Posture RBF-FEL feature_std shape')

        if (centers.ndim == 3
                and centers.shape[0] == self.SIDE_COUNT
                and centers.shape[2] == expected_dim
                and centers.shape[1] > 0):
            centers = np.repeat(centers[:, None, :, :],
                                self.AXIS_COUNT, axis=1)
        if (centers.ndim != 4
                or centers.shape[0] != self.SIDE_COUNT
                or centers.shape[1] != self.AXIS_COUNT
                or centers.shape[3] != expected_dim
                or centers.shape[2] <= 0):
            raise ValueError('Invalid Posture RBF-FEL centers shape')

        if sigma.size == 1:
            sigma = np.full((self.SIDE_COUNT, self.AXIS_COUNT),
                            float(sigma.reshape(-1)[0]), dtype=float)
        elif sigma.shape == (self.SIDE_COUNT,):
            sigma = np.repeat(sigma[:, None], self.AXIS_COUNT, axis=1)
        elif sigma.shape != (self.SIDE_COUNT, self.AXIS_COUNT):
            raise ValueError('Invalid Posture RBF-FEL sigma shape')

        if np.any(~np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError('Posture RBF-FEL feature_std must be positive')
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError('Posture RBF-FEL sigma must be positive')

        self.feature_mean = mean.copy()
        self.feature_std = std.copy()
        self.centers = centers.copy()
        self.sigma = sigma.copy()

        k = centers.shape[2]
        weight_shape = (self.SIDE_COUNT, k, self.AXIS_COUNT)

        if 'w0' in data:
            w0 = np.asarray(data['w0'], dtype=float)
            # Also accept [side, axis, center].
            if w0.shape == (self.SIDE_COUNT, self.AXIS_COUNT, k):
                w0 = np.transpose(w0, (0, 2, 1))
            if w0.shape != weight_shape:
                raise ValueError('Invalid Posture RBF-FEL w0 shape')
            self.w0 = w0.copy()
        else:
            self.w0 = np.zeros(weight_shape, dtype=float)

        if 'delta_w' in data:
            delta_w = np.asarray(data['delta_w'], dtype=float)
            if delta_w.shape == (self.SIDE_COUNT, self.AXIS_COUNT, k):
                delta_w = np.transpose(delta_w, (0, 2, 1))
            if delta_w.shape != weight_shape:
                raise ValueError('Invalid Posture RBF-FEL delta_w shape')
            self.delta_w = delta_w.copy()
        else:
            self.delta_w = np.zeros(weight_shape, dtype=float)

        self.model_ready = True

    def save_model(self, path=None):
        target = str(path or self.save_path or '')
        if not target or not self.model_ready:
            return False
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(
            target,
            feature_names=np.asarray(self.FEATURE_NAMES),
            axis_names=np.asarray(self.AXIS_NAMES),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            centers=self.centers,
            sigma=self.sigma,
            w0=self.w0,
            delta_w=self.delta_w,
        )
        return True

    def reset_learning(self):
        if getattr(self, 'model_ready', False):
            if self.mode == 'baseline':
                self.w0[:] = 0.0
                self.delta_w[:] = 0.0
            elif self.mode == 'payload':
                self.delta_w[:] = 0.0
        k = self.num_centers
        self._xtx = np.zeros(
            (self.SIDE_COUNT, self.AXIS_COUNT, k, k), dtype=float)
        self._xty = np.zeros(
            (self.SIDE_COUNT, self.AXIS_COUNT, k), dtype=float)
        self._active_key = None
        self._active_side = -1
        self._step_xtx = np.zeros((self.AXIS_COUNT, k, k), dtype=float)
        self._step_xty = np.zeros((self.AXIS_COUNT, k), dtype=float)
        self._step_samples_axis = np.zeros(self.AXIS_COUNT, dtype=int)
        self._step_was_actually_walking = False
        self.update_count = 0
        self.total_training_samples = 0
        self.training_samples_by_side = np.zeros(
            (self.SIDE_COUNT, self.AXIS_COUNT), dtype=int)
        self.completed_valid_steps = np.zeros(
            (self.SIDE_COUNT, self.AXIS_COUNT), dtype=int)
        self.payload_calibration_steps = np.zeros(
            (self.SIDE_COUNT, self.AXIS_COUNT), dtype=int)
        self.payload_ready_axis = np.zeros(
            (self.SIDE_COUNT, self.AXIS_COUNT), dtype=bool)
        # Aggregate compatibility flag: True once every side/axis has reached
        # the configured warm-up count.  Unlike the old implementation, this
        # flag no longer stops learning.
        self.payload_ready = False
        self.last_payload_weight_update_norm = np.zeros(
            self.AXIS_COUNT, dtype=float)
        self.last_feature = np.full(len(self.FEATURE_NAMES), np.nan)
        self.last_feature_normalized = np.full(
            (self.AXIS_COUNT, len(self.FEATURE_NAMES)), np.nan)
        self.last_phase = np.nan
        self.last_time_to_landing = np.nan
        self.last_step_side = -1
        self.last_step_tbegin = np.nan
        self.last_guarded = True
        self.last_ood_axis = np.ones(self.AXIS_COUNT, dtype=bool)
        self.last_max_phi_axis = np.zeros(self.AXIS_COUNT, dtype=float)
        self.last_sum_phi_axis = np.zeros(self.AXIS_COUNT, dtype=float)
        # Aggregate fields are retained for existing CSV consumers.
        self.last_ood = True
        self.last_max_phi = 0.0
        self.last_sum_phi = 0.0
        self.last_teacher = np.zeros(2, dtype=float)
        self.last_raw_output = np.zeros(2, dtype=float)
        self.last_unclipped_output = np.zeros(2, dtype=float)
        self.last_output = np.zeros(2, dtype=float)
        self.last_w0_output = np.zeros(2, dtype=float)
        self.last_delta_output = np.zeros(2, dtype=float)
        self.last_target = np.zeros(2, dtype=float)
        self.last_output_saturated_axis = np.zeros(
            self.AXIS_COUNT, dtype=bool)
        self.last_output_saturation_amount = np.zeros(
            self.AXIS_COUNT, dtype=float)
        self.last_anti_windup_blocked_axis = np.zeros(
            self.AXIS_COUNT, dtype=bool)
        self.anti_windup_blocked_samples = np.zeros(
            self.AXIS_COUNT, dtype=int)
        self._step_anti_windup_blocked_samples = np.zeros(
            self.AXIS_COUNT, dtype=int)
        self.last_update_anti_windup_blocked_samples = np.zeros(
            self.AXIS_COUNT, dtype=int)
        self.last_update_side = -1
        self.last_update_samples = 0
        self.last_walking_active = False
        self.last_support_transition = False
        self.last_output_jump_norm = 0.0
        self.last_w0_jump_norm = 0.0
        self.last_delta_jump_norm = 0.0
        self.last_transition_from_side = -1
        self.last_transition_to_side = -1
        self._transition_log_pending = False
        self._previous_walking_side = -1
        self._previous_walking_effective_output = np.zeros(2, dtype=float)
        self._previous_walking_w0_output = np.zeros(2, dtype=float)
        self._previous_walking_delta_output = np.zeros(2, dtype=float)
        self._current_psi = [None] * self.AXIS_COUNT
        self._current_train_valid = np.zeros(self.AXIS_COUNT, dtype=bool)
        self._saved_recovery_moment_ff_local = None

    @staticmethod
    def _active_steps(footstep_buffer, footstep):
        if footstep_buffer is not None and len(footstep_buffer.steps) > 0:
            current = footstep_buffer.steps[0]
            nxt = (footstep_buffer.steps[1]
                   if len(footstep_buffer.steps) > 1 else current)
            return current, nxt
        if footstep is not None and len(footstep.steps) > 0:
            current = footstep.steps[0]
            nxt = footstep.steps[1] if len(footstep.steps) > 1 else current
            return current, nxt
        return None, None

    @staticmethod
    def _inverse_rotate(orientation, vector):
        vector = np.asarray(vector, dtype=float)
        if hasattr(orientation, 'apply'):
            return np.asarray(orientation.apply(vector, inverse=True),
                              dtype=float)
        matrix = np.asarray(orientation, dtype=float)
        if matrix.shape == (3, 3):
            return matrix.T.dot(vector)
        return vector.copy()

    def _reference_feature(self, timer, stepping_controller,
                           footstep_buffer, footstep):
        step, _ = self._active_steps(footstep_buffer, footstep)
        if step is None:
            return None
        side = int(step.side)
        duration = float(step.duration)
        tbegin = float(step.tbegin)
        if side not in (0, 1) or not np.isfinite(duration) or duration <= 0.0:
            return None

        phase = float(np.clip((float(timer.time) - tbegin) / duration,
                              0.0, 1.0))
        ttl = float(getattr(stepping_controller, 'time_to_landing',
                            duration * (1.0 - phase)))
        ttl_norm = float(np.clip(ttl / duration, 0.0, 1.0))

        # Deliberately compact input: no DCM/ZMP/swing target.  Those nearly
        # constant dimensions were making the Gaussian distance brittle across
        # repeated nominal walks.
        x = np.array([
            np.sin(2.0 * np.pi * phase),
            np.cos(2.0 * np.pi * phase),
            ttl_norm,
        ], dtype=float)

        if (not np.isfinite(phase) or not np.isfinite(ttl)
                or not np.all(np.isfinite(x))):
            return None

        key = (side, round(tbegin, 9))
        guard = bool(
            (float(timer.time) - tbegin) < self.contact_guard_time
            or ttl < self.contact_guard_time)
        walking_active = bool(getattr(step, 'stepping', False))
        return key, side, phase, ttl, guard, walking_active, x

    def _basis(self, side, axis, x):
        if not self.model_ready or self.num_centers <= 0:
            return None, None, 0.0, 0.0, True

        xz = ((x - self.feature_mean[side, axis])
              / self.feature_std[side, axis])
        diff = self.centers[side, axis] - xz[None, :]
        d2 = np.sum(diff * diff, axis=1)
        sig = max(float(self.sigma[side, axis]), 1.0e-9)
        phi = np.exp(-0.5 * d2 / (sig * sig))
        sum_phi = float(np.sum(phi))
        max_phi = float(np.max(phi)) if phi.size else 0.0
        ood = bool(max_phi < self.coverage_threshold
                   or sum_phi < self.coverage_sum_threshold
                   or not np.isfinite(sum_phi))
        if ood:
            return xz, None, max_phi, sum_phi, True

        psi = phi / max(sum_phi, 1.0e-12)
        return xz, psi, max_phi, sum_phi, False

    def _project_weights(self, weights):
        # Weight clipping is intentionally disabled.  The RBF weights are
        # allowed to represent the learned residual freely; only the actual
        # feedforward moment injected into Stabilizer is saturated.
        return np.asarray(weights, dtype=float).copy()

    def _commit_pending(self):
        """Commit one completed support step.

        Baseline mode keeps the previous cumulative ridge-regression behavior.

        Payload mode uses a step-to-step FEL / ILC-style update.  During one
        support step the current delta weights are frozen.  The remaining
        stabilizer feedback is projected onto the RBF basis at the end of the
        step, producing a residual correction in weight space:

            dw_step = argmin ||Psi dw - tau_FB||^2 + lambda ||dw||^2

        Then the learned payload weights are updated as

            delta_w <- k_f * delta_w + k_l * dw_step

        where k_f is a retention/forgetting factor and k_l is an axis-specific
        learning gain.  The new weights are never injected halfway through a
        step; they become effective on the next occurrence of that support
        side.  This preserves a fixed feedforward predictor within each step.
        """
        has_samples = bool(np.any(self._step_samples_axis > 0))
        if (not self.learning_enabled or self._active_side not in (0, 1)
                or not self._step_was_actually_walking
                or not has_samples or self.num_centers <= 0):
            self._clear_step_accumulator()
            self._step_was_actually_walking = False
            return

        side = self._active_side
        if self._active_key is not None:
            assert int(self._active_key[0]) == int(side), (
                'Completed-step side mismatch: key side {} != active side {}'
                .format(self._active_key[0], side))

        # Bookkeeping is updated for every valid completed step.
        for axis in range(self.AXIS_COUNT):
            samples = int(self._step_samples_axis[axis])
            if samples <= 0:
                continue
            self.completed_valid_steps[side, axis] += 1
            self.training_samples_by_side[side, axis] += samples

        self.total_training_samples += int(np.max(self._step_samples_axis))
        eye = np.eye(self.num_centers, dtype=float)

        if self.mode == 'payload':
            updated = False
            self.last_payload_weight_update_norm[:] = 0.0

            for axis in range(self.AXIS_COUNT):
                samples = int(self._step_samples_axis[axis])
                if samples < self.payload_min_samples_per_step:
                    continue

                # Fit only the residual feedback observed during THIS step.
                # Because delta_w was held fixed throughout the step, this is
                # a correction to the currently active feedforward model.
                matrix = self._step_xtx[axis] + self.ridge_lambda * eye
                rhs = self._step_xty[axis]
                try:
                    correction = np.linalg.solve(matrix, rhs)
                except np.linalg.LinAlgError:
                    correction = np.linalg.lstsq(
                        matrix, rhs, rcond=None)[0]

                old = self.delta_w[side, :, axis].copy()
                proposed = (
                    self.payload_forgetting_factor * old
                    + self.payload_learning_gain[axis] * correction
                )
                proposed = self._project_weights(proposed)

                self.delta_w[side, :, axis] = proposed
                self.last_payload_weight_update_norm[axis] = float(
                    np.linalg.norm(proposed - old))

                self.payload_calibration_steps[side, axis] += 1
                if (self.payload_calibration_steps[side, axis]
                        >= self.payload_calibration_steps_per_side):
                    self.payload_ready_axis[side, axis] = True
                updated = True

            # This is only an aggregate status field.  Learning continues
            # after it becomes True.
            self.payload_ready = bool(np.all(self.payload_ready_axis))

            if updated:
                self.update_count += 1
                self.last_update_side = int(side)
                self.last_update_samples = int(
                    np.max(self._step_samples_axis))
                self.last_update_anti_windup_blocked_samples[:] = (
                    self._step_anti_windup_blocked_samples)

        else:
            # Baseline W0 learning: cumulative ridge regression as before.
            for axis in range(self.AXIS_COUNT):
                samples = int(self._step_samples_axis[axis])
                if samples <= 0:
                    continue
                self._xtx[side, axis] += self._step_xtx[axis]
                self._xty[side, axis] += self._step_xty[axis]

            updated = False
            for axis in range(self.AXIS_COUNT):
                if self._step_samples_axis[axis] <= 0:
                    continue
                matrix = self._xtx[side, axis] + self.ridge_lambda * eye
                rhs = self._xty[side, axis]
                try:
                    solution = np.linalg.solve(matrix, rhs)
                except np.linalg.LinAlgError:
                    solution = np.linalg.lstsq(
                        matrix, rhs, rcond=None)[0]
                self.w0[side, :, axis] = self._project_weights(solution)
                updated = True

            if updated:
                self.update_count += 1
                self.last_update_side = int(side)
                self.last_update_samples = int(
                    np.max(self.training_samples_by_side[side]))

        self._clear_step_accumulator()
        self._step_was_actually_walking = False
        if self.save_path:
            self.save_model(self.save_path)

    def _clear_step_accumulator(self):
        k = self.num_centers
        self._step_xtx = np.zeros(
            (self.AXIS_COUNT, k, k), dtype=float)
        self._step_xty = np.zeros((self.AXIS_COUNT, k), dtype=float)
        self._step_samples_axis = np.zeros(self.AXIS_COUNT, dtype=int)
        self._step_anti_windup_blocked_samples = np.zeros(
            self.AXIS_COUNT, dtype=int)

    def before_stabilizer(self, timer, param, centroid, base, feet,
                          stepping_controller, footstep_buffer, footstep,
                          stabilizer):
        del param, centroid, base, feet
        if hasattr(stabilizer, 'recovery_moment_ff_local'):
            self._saved_recovery_moment_ff_local = np.asarray(
                stabilizer.recovery_moment_ff_local, dtype=float).copy()
        else:
            self._saved_recovery_moment_ff_local = np.zeros(3, dtype=float)

        ref = self._reference_feature(
            timer, stepping_controller, footstep_buffer, footstep)
        self._current_psi = [None] * self.AXIS_COUNT
        self._current_train_valid[:] = False
        self.last_walking_active = False
        self.last_raw_output[:] = 0.0
        self.last_unclipped_output[:] = 0.0
        self.last_output[:] = 0.0
        self.last_w0_output[:] = 0.0
        self.last_delta_output[:] = 0.0
        self.last_teacher[:] = 0.0
        self.last_target[:] = 0.0
        self.last_output_saturated_axis[:] = False
        self.last_output_saturation_amount[:] = 0.0
        self.last_anti_windup_blocked_axis[:] = False
        if ref is None:
            self.last_feature[:] = np.nan
            self.last_feature_normalized[:] = np.nan
            self.last_phase = np.nan
            self.last_time_to_landing = np.nan
            self.last_step_side = -1
            self.last_step_tbegin = np.nan
            self.last_guarded = True
            self.last_ood_axis[:] = True
            self.last_max_phi_axis[:] = 0.0
            self.last_sum_phi_axis[:] = 0.0
            self.last_ood = True
            self.last_max_phi = 0.0
            self.last_sum_phi = 0.0
            if hasattr(stabilizer, 'recovery_moment_ff_local'):
                stabilizer.recovery_moment_ff_local[:] = (
                    self._saved_recovery_moment_ff_local)
            return

        key, side, phase, ttl, guard, walking_active, x = ref
        if key != self._active_key:
            if self._active_key is not None:
                self._commit_pending()
            self._active_key = key
            self._active_side = side
            self._clear_step_accumulator()
            self._step_was_actually_walking = False
        elif self._step_was_actually_walking and not walking_active:
            # A real walking step can end without the buffered Step key
            # changing (for example when entering the final hold).  Finalize
            # it exactly once on the true stepping -> inactive transition.
            self._commit_pending()

        if walking_active:
            self._step_was_actually_walking = True

        self.last_feature[:] = x
        self.last_feature_normalized[:] = np.nan

        psi_by_axis = [None] * self.AXIS_COUNT
        for axis in range(self.AXIS_COUNT):
            xz, psi, max_phi, sum_phi, ood = self._basis(side, axis, x)
            if xz is not None:
                self.last_feature_normalized[axis] = xz
            psi_by_axis[axis] = psi
            self.last_ood_axis[axis] = ood
            self.last_max_phi_axis[axis] = max_phi
            self.last_sum_phi_axis[axis] = sum_phi

        self.last_phase = phase
        self.last_time_to_landing = ttl
        self.last_step_side = side
        self.last_step_tbegin = float(key[1])
        self.last_guarded = guard
        self.last_walking_active = bool(walking_active)

        # Backward-compatible aggregate coverage values describe the worst
        # axis, while axis-specific fields are logged separately.
        self.last_ood = bool(np.any(self.last_ood_axis))
        self.last_max_phi = float(np.min(self.last_max_phi_axis))
        self.last_sum_phi = float(np.min(self.last_sum_phi_axis))

        if self.enabled and any(psi is not None for psi in psi_by_axis):
            w0_output = np.zeros(self.AXIS_COUNT, dtype=float)
            delta_output = np.zeros(self.AXIS_COUNT, dtype=float)

            for axis, psi in enumerate(psi_by_axis):
                if psi is None or self.last_ood_axis[axis]:
                    continue
                w0_output[axis] = psi.dot(self.w0[side, :, axis])
                if self.mode == 'payload':
                    # Each side/axis becomes usable independently after its
                    # first (or configured number of) completed learning step.
                    # Learning continues after activation.
                    if self.payload_ready_axis[side, axis]:
                        delta_output[axis] = psi.dot(
                            self.delta_w[side, :, axis])
                else:
                    delta_output[axis] = psi.dot(
                        self.delta_w[side, :, axis])

            unclipped = w0_output + delta_output
            total = np.clip(
                unclipped, -self.output_limit_nm, self.output_limit_nm)
            self.last_w0_output[:] = w0_output
            self.last_delta_output[:] = delta_output
            self.last_unclipped_output[:] = unclipped
            self.last_raw_output[:] = total
            self.last_output_saturated_axis[:] = (
                np.abs(unclipped) >= self.output_limit_nm - 1.0e-9)
            self.last_output_saturation_amount[:] = np.abs(
                unclipped - total)

            # Raw prediction is still logged while inactive, but only a true
            # VNOID walking step may inject effective FEL into Stabilizer.
            effective = total if walking_active else np.zeros(2, dtype=float)
            self.last_output[:] = effective

            # Detect the actual active support-side change at controller rate,
            # but latch the event until the next CSV row is written.  The CSV
            # is sampled more slowly than this callback, so a one-cycle pulse
            # here would otherwise miss almost every support transition.
            # Compare the last effective output of the completed support side
            # against the first effective output of the newly active side.
            if walking_active:
                previous_side = int(self._previous_walking_side)
                if previous_side in (0, 1) and side != previous_side:
                    self.last_support_transition = True
                    self.last_transition_from_side = previous_side
                    self.last_transition_to_side = int(side)
                    self.last_output_jump_norm = float(np.linalg.norm(
                        effective - self._previous_walking_effective_output))
                    self.last_w0_jump_norm = float(np.linalg.norm(
                        w0_output - self._previous_walking_w0_output))
                    self.last_delta_jump_norm = float(np.linalg.norm(
                        delta_output - self._previous_walking_delta_output))
                    self._transition_log_pending = True
                self._previous_walking_side = int(side)
                self._previous_walking_effective_output[:] = effective
                self._previous_walking_w0_output[:] = w0_output
                self._previous_walking_delta_output[:] = delta_output

            for axis, psi in enumerate(psi_by_axis):
                self._current_psi[axis] = (
                    psi.copy() if walking_active and psi is not None else None)
                self._current_train_valid[axis] = bool(
                    self.learning_enabled
                    and walking_active
                    and side in (0, 1)
                    and np.isfinite(phase)
                    and not guard
                    and not self.last_ood_axis[axis]
                    and psi is not None)
            if hasattr(stabilizer, 'recovery_moment_ff_local'):
                ff = self._saved_recovery_moment_ff_local.copy()
                ff[:2] += effective
                stabilizer.recovery_moment_ff_local[:] = ff
        elif hasattr(stabilizer, 'recovery_moment_ff_local'):
            stabilizer.recovery_moment_ff_local[:] = (
                self._saved_recovery_moment_ff_local)

    def after_stabilizer(self, centroid, stabilizer=None):
        del centroid
        if stabilizer is None:
            return
        if hasattr(stabilizer, 'last_recovery_moment_fb_local'):
            teacher = np.asarray(
                stabilizer.last_recovery_moment_fb_local,
                dtype=float)[:2].copy()
            teacher = np.clip(teacher, -self.teacher_limit_nm,
                              self.teacher_limit_nm)
            self.last_teacher[:] = teacher
        else:
            teacher = np.zeros(2, dtype=float)
            self.last_teacher[:] = 0.0

        if np.all(np.isfinite(teacher)):
            if self.mode == 'baseline':
                # W0 is refitted as the total feedforward command.
                target = self.last_w0_output + teacher
            else:
                # Online payload mode performs an incremental FEL/ILC update:
                # the current delta model is already active and frozen during
                # this step, so the remaining FB torque is exactly the
                # correction that should be learned for the next occurrence.
                target = teacher

            any_sample = False
            for axis in range(self.AXIS_COUNT):
                psi = self._current_psi[axis]
                if not self._current_train_valid[axis] or psi is None:
                    continue

                # Conditional-integration anti-windup:
                # if the actual FEL output is already at its +/- output limit
                # and the remaining FB requests still more moment in the same
                # direction, do not integrate that sample into the next weight
                # update.  Opposite-sign FB is still learned because it drives
                # the saturated output back toward the admissible region.
                raw_output = self.last_unclipped_output[axis]
                same_direction = (
                    raw_output * teacher[axis] > 0.0)
                if (self.last_output_saturated_axis[axis]
                        and same_direction):
                    self.last_anti_windup_blocked_axis[axis] = True
                    self.anti_windup_blocked_samples[axis] += 1
                    self._step_anti_windup_blocked_samples[axis] += 1
                    continue

                self._step_xtx[axis] += np.outer(psi, psi)
                self._step_xty[axis] += psi * target[axis]
                self._step_samples_axis[axis] += 1
                any_sample = True

            if any_sample:
                self.last_target[:] = target

        if hasattr(stabilizer, 'recovery_moment_ff_local'):
            saved = self._saved_recovery_moment_ff_local
            if saved is None:
                stabilizer.recovery_moment_ff_local[:] = 0.0
            else:
                stabilizer.recovery_moment_ff_local[:] = saved
        self._saved_recovery_moment_ff_local = None

    def finalize(self):
        if self._active_key is not None:
            # Simulation termination is not evidence that the current support
            # step completed.  Any partial step is discarded; completed steps
            # are committed on key change or stepping -> inactive transition.
            self._clear_step_accumulator()
            self._step_was_actually_walking = False
            self._active_key = None
            self._active_side = -1
        if self.save_path:
            self.save_model(self.save_path)

    def global_log_fields(self):
        if self.model_ready and self.w0.size:
            w0_l2 = float(np.linalg.norm(self.w0))
            w0_abs = float(np.max(np.abs(self.w0)))
            delta_l2 = float(np.linalg.norm(self.delta_w))
            delta_abs = float(np.max(np.abs(self.delta_w)))
        else:
            w0_l2 = w0_abs = delta_l2 = delta_abs = 0.0
        header = [
            'posture_rbf_enabled', 'posture_rbf_mode',
            'posture_rbf_model_ready', 'posture_rbf_num_centers',
            'posture_rbf_step_side', 'posture_rbf_step_tbegin_s',
            'posture_rbf_phase', 'posture_rbf_time_to_landing_s',
            'posture_rbf_walking_active',
            'posture_rbf_guarded', 'posture_rbf_ood',
            'posture_rbf_max_phi', 'posture_rbf_sum_phi',
            'posture_rbf_ood_x', 'posture_rbf_ood_y',
            'posture_rbf_max_phi_x', 'posture_rbf_max_phi_y',
            'posture_rbf_sum_phi_x', 'posture_rbf_sum_phi_y',
        ]
        values = [
            int(self.enabled), self.mode, int(self.model_ready),
            self.num_centers, self.last_step_side, self.last_step_tbegin,
            self.last_phase, self.last_time_to_landing,
            int(self.last_walking_active),
            int(self.last_guarded), int(self.last_ood),
            self.last_max_phi, self.last_sum_phi,
            int(self.last_ood_axis[0]), int(self.last_ood_axis[1]),
            self.last_max_phi_axis[0], self.last_max_phi_axis[1],
            self.last_sum_phi_axis[0], self.last_sum_phi_axis[1],
        ]
        for name, value in zip(self.FEATURE_NAMES, self.last_feature):
            header.append('posture_rbf_feat_' + name)
            values.append(value)
        header.extend([
            'posture_rbf_teacher_x_Nm', 'posture_rbf_teacher_y_Nm',
            'posture_rbf_w0_output_x_Nm', 'posture_rbf_w0_output_y_Nm',
            'posture_rbf_delta_output_x_Nm',
            'posture_rbf_delta_output_y_Nm',
            'posture_rbf_unclipped_output_x_Nm',
            'posture_rbf_unclipped_output_y_Nm',
            'posture_rbf_raw_output_x_Nm',
            'posture_rbf_raw_output_y_Nm',
            'posture_rbf_output_x_Nm', 'posture_rbf_output_y_Nm',
            'posture_rbf_output_saturated_x',
            'posture_rbf_output_saturated_y',
            'posture_rbf_output_saturation_amount_x_Nm',
            'posture_rbf_output_saturation_amount_y_Nm',
            'posture_rbf_anti_windup_blocked_x',
            'posture_rbf_anti_windup_blocked_y',
            'posture_rbf_anti_windup_blocked_samples_x',
            'posture_rbf_anti_windup_blocked_samples_y',
            'posture_rbf_last_update_anti_windup_blocked_samples_x',
            'posture_rbf_last_update_anti_windup_blocked_samples_y',
            'posture_rbf_target_x_Nm', 'posture_rbf_target_y_Nm',
            'posture_rbf_updates', 'posture_rbf_step_train_samples',
            'posture_rbf_total_train_samples',
            'posture_rbf_training_samples_right',
            'posture_rbf_training_samples_left',
            'posture_rbf_completed_valid_steps_total',
            'posture_rbf_completed_valid_steps_right',
            'posture_rbf_completed_valid_steps_left',
            'posture_rbf_regression_updates',
            'posture_rbf_last_update_side',
            'posture_rbf_last_update_samples',
            'posture_rbf_w0_l2_Nm', 'posture_rbf_w0_absmax_Nm',
            'posture_rbf_delta_l2_Nm', 'posture_rbf_delta_absmax_Nm',
            'posture_rbf_payload_ready',
            'posture_rbf_payload_ready_side0_x',
            'posture_rbf_payload_ready_side0_y',
            'posture_rbf_payload_ready_side1_x',
            'posture_rbf_payload_ready_side1_y',
            'posture_rbf_payload_learning_gain_x',
            'posture_rbf_payload_learning_gain_y',
            'posture_rbf_payload_forgetting_factor',
            'posture_rbf_payload_weight_update_norm_x',
            'posture_rbf_payload_weight_update_norm_y',
            'posture_rbf_payload_calib_steps_side0',
            'posture_rbf_payload_calib_steps_side1',
            'posture_rbf_calibration_ready',
            'posture_rbf_support_transition',
            'posture_rbf_transition_from_side',
            'posture_rbf_transition_to_side',
            'posture_rbf_output_jump_norm_Nm',
            'posture_rbf_w0_jump_norm_Nm',
            'posture_rbf_delta_jump_norm_Nm',
        ])
        values.extend([
            self.last_teacher[0], self.last_teacher[1],
            self.last_w0_output[0], self.last_w0_output[1],
            self.last_delta_output[0], self.last_delta_output[1],
            self.last_unclipped_output[0], self.last_unclipped_output[1],
            self.last_raw_output[0], self.last_raw_output[1],
            self.last_output[0], self.last_output[1],
            int(self.last_output_saturated_axis[0]),
            int(self.last_output_saturated_axis[1]),
            self.last_output_saturation_amount[0],
            self.last_output_saturation_amount[1],
            int(self.last_anti_windup_blocked_axis[0]),
            int(self.last_anti_windup_blocked_axis[1]),
            int(self.anti_windup_blocked_samples[0]),
            int(self.anti_windup_blocked_samples[1]),
            int(self.last_update_anti_windup_blocked_samples[0]),
            int(self.last_update_anti_windup_blocked_samples[1]),
            self.last_target[0], self.last_target[1],
            self.update_count, int(np.max(self._step_samples_axis)),
            self.total_training_samples,
            int(np.max(self.training_samples_by_side[0])),
            int(np.max(self.training_samples_by_side[1])),
            int(np.sum(np.min(self.completed_valid_steps, axis=1))),
            int(np.min(self.completed_valid_steps[0])),
            int(np.min(self.completed_valid_steps[1])),
            self.update_count,
            self.last_update_side, self.last_update_samples,
            w0_l2, w0_abs, delta_l2, delta_abs,
            int(self.payload_ready),
            int(self.payload_ready_axis[0, 0]),
            int(self.payload_ready_axis[0, 1]),
            int(self.payload_ready_axis[1, 0]),
            int(self.payload_ready_axis[1, 1]),
            self.payload_learning_gain[0],
            self.payload_learning_gain[1],
            self.payload_forgetting_factor,
            self.last_payload_weight_update_norm[0],
            self.last_payload_weight_update_norm[1],
            int(np.min(self.payload_calibration_steps[0])),
            int(np.min(self.payload_calibration_steps[1])),
            int(self.payload_ready),
            int(self._transition_log_pending),
            self.last_transition_from_side,
            self.last_transition_to_side,
            self.last_output_jump_norm,
            self.last_w0_jump_norm,
            self.last_delta_jump_norm,
        ])
        # Consume the latched transition only after its values have been
        # copied into this CSV row.  This keeps a controller-rate event alive
        # until the slower logger observes it exactly once.
        self._transition_log_pending = False
        self.last_support_transition = False
        self.last_transition_from_side = -1
        self.last_transition_to_side = -1
        self.last_output_jump_norm = 0.0
        self.last_w0_jump_norm = 0.0
        self.last_delta_jump_norm = 0.0
        return header, values