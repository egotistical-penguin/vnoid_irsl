import os

import numpy as np


class PostureGaussianRBFFELController:
    """Posture-PD FEL for stabilizer recovery moments using Gaussian RBFs.

    The predictor only sees quantities from the planned walking reference.
    The teacher is collected *after* Stabilizer.Update() from the remaining
    orientation-feedback recovery moment. Weights are updated only when a
    real VNOID support step changes, so the predictor is fixed within a step.
    """

    FEATURE_NAMES = (
        'phase_sin',
        'phase_cos',
        'time_to_landing_norm',
        'plan_dcm_local_x_m',
        'plan_dcm_local_y_m',
        'plan_zmp_local_x_m',
        'plan_zmp_local_y_m',
        'next_swing_local_x_m',
        'next_swing_local_y_m',
    )
    SIDE_COUNT = 2

    def __init__(self):
        self.configure()

    def configure(self, enabled=False, mode='frozen', model_path='',
                  save_path='', ridge_lambda=1.0e-2,
                  output_limit_nm=10.0, teacher_limit_nm=100.0,
                  contact_guard_time=0.05,
                  coverage_threshold=1.0e-4,
                  coverage_sum_threshold=1.0e-6,
                  payload_calibration_steps_per_side=2):
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
        self.payload_calibration_steps_per_side = max(
            1, int(payload_calibration_steps_per_side))

        self.feature_mean = np.zeros(
            (self.SIDE_COUNT, len(self.FEATURE_NAMES)), dtype=float)
        self.feature_std = np.ones_like(self.feature_mean)
        self.centers = np.zeros(
            (self.SIDE_COUNT, 0, len(self.FEATURE_NAMES)), dtype=float)
        self.sigma = np.ones(self.SIDE_COUNT, dtype=float)
        self.w0 = np.zeros((self.SIDE_COUNT, 0, 2), dtype=float)
        self.delta_w = np.zeros_like(self.w0)
        self.model_ready = False
        if self.model_path:
            self.load_model(self.model_path)
        self.reset_learning()

    @property
    def num_centers(self):
        return int(self.centers.shape[1]) if self.centers.ndim == 3 else 0

    @property
    def learning_enabled(self):
        if not self.enabled:
            return False
        if self.mode == 'payload':
            return not getattr(self, 'payload_ready', False)
        return self.mode == 'baseline'

    def load_model(self, path):
        data = np.load(path, allow_pickle=False)
        feature_names = tuple(str(x) for x in data['feature_names'].tolist())
        if feature_names != self.FEATURE_NAMES:
            raise ValueError(
                'Posture RBF-FEL feature schema mismatch: {}'.format(
                    feature_names))
        mean = np.asarray(data['feature_mean'], dtype=float)
        std = np.asarray(data['feature_std'], dtype=float)
        centers = np.asarray(data['centers'], dtype=float)
        sigma = np.asarray(data['sigma'], dtype=float).reshape(-1)
        expected_dim = len(self.FEATURE_NAMES)
        if mean.shape != (self.SIDE_COUNT, expected_dim):
            raise ValueError('Invalid Posture RBF-FEL feature_mean shape')
        if std.shape != mean.shape:
            raise ValueError('Invalid Posture RBF-FEL feature_std shape')
        if (centers.ndim != 3 or centers.shape[0] != self.SIDE_COUNT
                or centers.shape[2] != expected_dim
                or centers.shape[1] <= 0):
            raise ValueError('Invalid Posture RBF-FEL centers shape')
        if sigma.size == 1:
            sigma = np.repeat(sigma, self.SIDE_COUNT)
        if sigma.shape != (self.SIDE_COUNT,):
            raise ValueError('Invalid Posture RBF-FEL sigma shape')
        if np.any(~np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError('Posture RBF-FEL feature_std must be positive')
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError('Posture RBF-FEL sigma must be positive')

        self.feature_mean = mean
        self.feature_std = std
        self.centers = centers
        self.sigma = sigma
        shape = (self.SIDE_COUNT, centers.shape[1], 2)
        if 'w0' in data:
            w0 = np.asarray(data['w0'], dtype=float)
            if w0.shape != shape:
                raise ValueError('Invalid Posture RBF-FEL w0 shape')
            self.w0 = w0.copy()
        else:
            self.w0 = np.zeros(shape, dtype=float)
        if 'delta_w' in data:
            delta_w = np.asarray(data['delta_w'], dtype=float)
            if delta_w.shape != shape:
                raise ValueError('Invalid Posture RBF-FEL delta_w shape')
            self.delta_w = delta_w.copy()
        else:
            self.delta_w = np.zeros(shape, dtype=float)
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
        self._xtx = np.zeros((self.SIDE_COUNT, k, k), dtype=float)
        self._xty = np.zeros((self.SIDE_COUNT, k, 2), dtype=float)
        self._active_key = None
        self._active_side = -1
        self._step_xtx = np.zeros((k, k), dtype=float)
        self._step_xty = np.zeros((k, 2), dtype=float)
        self._step_samples = 0
        self._step_was_actually_walking = False
        self.update_count = 0
        self.total_training_samples = 0
        self.training_samples_by_side = np.zeros(
            self.SIDE_COUNT, dtype=int)
        self.completed_valid_steps = np.zeros(
            self.SIDE_COUNT, dtype=int)
        self.payload_calibration_steps = np.zeros(
            self.SIDE_COUNT, dtype=int)
        self.payload_ready = False
        self.last_feature = np.full(len(self.FEATURE_NAMES), np.nan)
        self.last_feature_normalized = np.full(
            len(self.FEATURE_NAMES), np.nan)
        self.last_phase = np.nan
        self.last_time_to_landing = np.nan
        self.last_step_side = -1
        self.last_step_tbegin = np.nan
        self.last_guarded = True
        self.last_ood = True
        self.last_max_phi = 0.0
        self.last_sum_phi = 0.0
        self.last_teacher = np.zeros(2, dtype=float)
        self.last_raw_output = np.zeros(2, dtype=float)
        self.last_output = np.zeros(2, dtype=float)
        self.last_w0_output = np.zeros(2, dtype=float)
        self.last_delta_output = np.zeros(2, dtype=float)
        self.last_target = np.zeros(2, dtype=float)
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
        self._current_psi = None
        self._current_train_valid = False
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
        step, next_step = self._active_steps(footstep_buffer, footstep)
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
        support = np.asarray(step.foot_pos[side], dtype=float)
        orientation = step.foot_ori[side]
        dcm_local = self._inverse_rotate(
            orientation, np.asarray(step.dcm, dtype=float) - support)
        zmp_local = self._inverse_rotate(
            orientation, np.asarray(step.zmp, dtype=float) - support)
        swing = 1 - side
        swing_target = np.asarray(next_step.foot_pos[swing], dtype=float)
        swing_local = self._inverse_rotate(
            orientation, swing_target - support)
        x = np.array([
            np.sin(2.0 * np.pi * phase),
            np.cos(2.0 * np.pi * phase),
            ttl_norm,
            dcm_local[0], dcm_local[1],
            zmp_local[0], zmp_local[1],
            swing_local[0], swing_local[1],
        ], dtype=float)
        if (not np.isfinite(phase) or not np.isfinite(ttl)
                or not np.all(np.isfinite(x))):
            return None
        key = (side, round(tbegin, 9))
        guard = bool(
            (float(timer.time) - tbegin) < self.contact_guard_time
            or ttl < self.contact_guard_time)
        # This is the exact flag logged as active_step_stepping in
        # walk_sim_vnoid.py for the same active buffered step.  A Step object
        # can exist before/after real walking, so Step existence alone is not
        # a valid lifecycle gate.
        walking_active = bool(getattr(step, 'stepping', False))
        return key, side, phase, ttl, guard, walking_active, x

    def _basis(self, side, x):
        if not self.model_ready or self.num_centers <= 0:
            return None, None, 0.0, 0.0, True
        xz = (x - self.feature_mean[side]) / self.feature_std[side]
        diff = self.centers[side] - xz[None, :]
        d2 = np.sum(diff * diff, axis=1)
        sig = max(float(self.sigma[side]), 1.0e-9)
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
        # Normalized RBF activations form a convex combination.  Bounding
        # every center output therefore bounds all in-distribution outputs,
        # rather than relying only on a runtime saturation.
        return np.clip(weights, -self.output_limit_nm,
                       self.output_limit_nm)

    def _commit_pending(self):
        if (not self.learning_enabled or self._active_side not in (0, 1)
                or not self._step_was_actually_walking
                or self._step_samples <= 0 or self.num_centers <= 0):
            self._clear_step_accumulator()
            self._step_was_actually_walking = False
            return
        side = self._active_side
        if self._active_key is not None:
            assert int(self._active_key[0]) == int(side), (
                'Completed-step side mismatch: key side {} != active side {}'
                .format(self._active_key[0], side))
        step_samples = int(self._step_samples)
        self._xtx[side] += self._step_xtx
        self._xty[side] += self._step_xty
        self.completed_valid_steps[side] += 1
        self.training_samples_by_side[side] += step_samples
        self.total_training_samples += step_samples

        # Payload adaptation is deliberately two-stage.  While calibrating,
        # delta_w is not injected at all; the remaining stabilizer feedback is
        # therefore measured under the same frozen W0 controller on every
        # calibration step.  Once enough left/right support steps have been
        # observed, fit the payload residual once and freeze it.  This avoids
        # the self-referential "current FF + remaining FB" recursion that can
        # move the closed-loop gait and make the learner chase its own effect.
        if self.mode == 'payload':
            self.payload_calibration_steps[side] += 1
            ready = bool(np.all(
                self.payload_calibration_steps
                >= self.payload_calibration_steps_per_side))
            if ready and not self.payload_ready:
                eye = np.eye(self.num_centers, dtype=float)
                for fit_side in range(self.SIDE_COUNT):
                    matrix = (self._xtx[fit_side]
                              + self.ridge_lambda * eye)
                    try:
                        solution = np.linalg.solve(
                            matrix, self._xty[fit_side])
                    except np.linalg.LinAlgError:
                        solution = np.linalg.lstsq(
                            matrix, self._xty[fit_side], rcond=None)[0]
                    self.delta_w[fit_side] = self._project_weights(solution)
                self.payload_ready = True
                self.update_count += 1
                self.last_update_side = int(side)
                self.last_update_samples = int(
                    np.sum(self.training_samples_by_side))
        else:
            matrix = self._xtx[side] + self.ridge_lambda * np.eye(
                self.num_centers, dtype=float)
            try:
                solution = np.linalg.solve(matrix, self._xty[side])
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(
                    matrix, self._xty[side], rcond=None)[0]
            self.w0[side] = self._project_weights(solution)
            self.update_count += 1
            self.last_update_side = int(side)
            self.last_update_samples = int(
                self.training_samples_by_side[side])
        self._clear_step_accumulator()
        self._step_was_actually_walking = False
        if self.save_path:
            self.save_model(self.save_path)

    def _clear_step_accumulator(self):
        k = self.num_centers
        self._step_xtx = np.zeros((k, k), dtype=float)
        self._step_xty = np.zeros((k, 2), dtype=float)
        self._step_samples = 0

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
        self._current_psi = None
        self._current_train_valid = False
        self.last_walking_active = False
        self.last_raw_output[:] = 0.0
        self.last_output[:] = 0.0
        self.last_w0_output[:] = 0.0
        self.last_delta_output[:] = 0.0
        self.last_teacher[:] = 0.0
        self.last_target[:] = 0.0
        if ref is None:
            self.last_feature[:] = np.nan
            self.last_feature_normalized[:] = np.nan
            self.last_phase = np.nan
            self.last_time_to_landing = np.nan
            self.last_step_side = -1
            self.last_step_tbegin = np.nan
            self.last_guarded = True
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

        xz, psi, max_phi, sum_phi, ood = self._basis(side, x)
        self.last_feature[:] = x
        if xz is None:
            self.last_feature_normalized[:] = np.nan
        else:
            self.last_feature_normalized[:] = xz
        self.last_phase = phase
        self.last_time_to_landing = ttl
        self.last_step_side = side
        self.last_step_tbegin = float(key[1])
        self.last_guarded = guard
        self.last_ood = ood
        self.last_walking_active = bool(walking_active)
        self.last_max_phi = max_phi
        self.last_sum_phi = sum_phi

        if self.enabled and psi is not None and not ood:
            w0_output = psi.dot(self.w0[side])
            if self.mode != 'payload' or self.payload_ready:
                delta_output = psi.dot(self.delta_w[side])
            else:
                delta_output = np.zeros(2, dtype=float)
            total = np.clip(w0_output + delta_output,
                            -self.output_limit_nm, self.output_limit_nm)
            self.last_w0_output[:] = w0_output
            self.last_delta_output[:] = delta_output
            self.last_raw_output[:] = total

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

            self._current_psi = psi.copy() if walking_active else None
            self._current_train_valid = bool(
                self.learning_enabled
                and walking_active
                and side in (0, 1)
                and np.isfinite(phase)
                and not guard
                and not ood)
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

        if (self._current_train_valid and self._current_psi is not None
                and np.all(np.isfinite(teacher))):
            if self.mode == 'baseline':
                target = self.last_w0_output + teacher
            else:
                # During payload calibration delta output is held at zero, so
                # the remaining feedback itself is the residual FF to learn.
                target = teacher
            target = np.clip(target, -self.output_limit_nm,
                             self.output_limit_nm)
            psi = self._current_psi
            self._step_xtx += np.outer(psi, psi)
            self._step_xty += np.outer(psi, target)
            self._step_samples += 1
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
        ]
        values = [
            int(self.enabled), self.mode, int(self.model_ready),
            self.num_centers, self.last_step_side, self.last_step_tbegin,
            self.last_phase, self.last_time_to_landing,
            int(self.last_walking_active),
            int(self.last_guarded), int(self.last_ood),
            self.last_max_phi, self.last_sum_phi,
        ]
        for name, value in zip(self.FEATURE_NAMES, self.last_feature):
            header.append('posture_rbf_feat_' + name)
            values.append(value)
        header.extend([
            'posture_rbf_teacher_x_Nm', 'posture_rbf_teacher_y_Nm',
            'posture_rbf_w0_output_x_Nm', 'posture_rbf_w0_output_y_Nm',
            'posture_rbf_delta_output_x_Nm',
            'posture_rbf_delta_output_y_Nm',
            'posture_rbf_raw_output_x_Nm',
            'posture_rbf_raw_output_y_Nm',
            'posture_rbf_output_x_Nm', 'posture_rbf_output_y_Nm',
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
            self.last_raw_output[0], self.last_raw_output[1],
            self.last_output[0], self.last_output[1],
            self.last_target[0], self.last_target[1],
            self.update_count, self._step_samples,
            self.total_training_samples,
            int(self.training_samples_by_side[0]),
            int(self.training_samples_by_side[1]),
            int(np.sum(self.completed_valid_steps)),
            int(self.completed_valid_steps[0]),
            int(self.completed_valid_steps[1]),
            self.update_count,
            self.last_update_side, self.last_update_samples,
            w0_l2, w0_abs, delta_l2, delta_abs,
            int(self.payload_ready),
            int(self.payload_calibration_steps[0]),
            int(self.payload_calibration_steps[1]),
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
