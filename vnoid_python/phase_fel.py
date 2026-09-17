import numpy as np


class PhaseFELController:
    """Phase-bin feedback-error learning for periodic walking torque."""

    LEG_SUFFIXES = (
        'hip_yaw_link',
        'hip_roll_link',
        'hip_pitch_link',
        'knee_link',
        'ankle_pitch_link',
        'ankle_roll_link',
    )
    PITCH_SUFFIXES = (
        'hip_pitch_link',
        'knee_link',
        'ankle_pitch_link',
    )

    def __init__(self):
        self.enabled = False
        self.num_bins = 280
        self.alpha = 0.1
        self.output_limit_ratio = 0.1
        self.joint_group = 'legs'

        self.num_joints = 0
        self.active_mask = None
        self.torque_scale = None
        self.fel_limit = None
        self.table = None
        self.pd_sum = None
        self.sample_count = 0
        self.accumulator_key = None

        self.support_side = -1
        self.phase = np.nan
        self.phase_bin = -1
        self.valid_active_step = False
        self.walking_active = False
        self.time_to_landing = np.nan
        self.last_torque = None
        self.last_table = None
        self.last_teacher_pd = None
        self.learning_gate = False
        self.update_count = 0

    def configure(self, enabled=True, num_bins=280, alpha=0.1,
                  output_limit_ratio=0.1, joint_group='legs'):
        self.enabled = bool(enabled)
        self.num_bins = int(num_bins)
        self.alpha = float(alpha)
        self.output_limit_ratio = float(output_limit_ratio)
        self.joint_group = str(joint_group).lower()
        if self.num_bins <= 0:
            raise ValueError('num_bins must be positive')
        if not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError('alpha must be finite and in [0, 1]')
        if (not np.isfinite(self.output_limit_ratio) or
                not 0.0 < self.output_limit_ratio <= 1.0):
            raise ValueError(
                'output_limit_ratio must be finite and in (0, 1]')
        if self.joint_group not in ('all', 'legs', 'pitch'):
            raise ValueError("joint_group must be 'all', 'legs', or 'pitch'")

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
        if self.joint_group == 'all':
            self.active_mask = np.ones(self.num_joints, dtype=bool)
        else:
            suffixes = (
                self.LEG_SUFFIXES if self.joint_group == 'legs'
                else self.PITCH_SUFFIXES)
            self.active_mask = np.asarray([
                (name.startswith('left_') or name.startswith('right_')) and
                any(name.endswith(suffix) for suffix in suffixes)
                for name in joint_names
            ], dtype=bool)
            expected = 12 if self.joint_group == 'legs' else 6
            if np.count_nonzero(self.active_mask) != expected:
                raise ValueError(
                    'G1 {} mask expected {} joints, found {}: {}'.format(
                        self.joint_group, expected,
                        int(np.count_nonzero(self.active_mask)),
                        ', '.join(np.asarray(joint_names)[self.active_mask])))

        torque_lower = self._joint_vector(body, 'u_lower')
        torque_upper = self._joint_vector(body, 'u_upper')
        self.torque_scale = np.maximum(
            np.abs(torque_lower), np.abs(torque_upper))
        if (not np.all(np.isfinite(self.torque_scale)) or
                np.any(self.torque_scale <= 0.0)):
            raise ValueError(
                'phase FEL requires finite positive joint torque limits')
        self.fel_limit = self.output_limit_ratio * self.torque_scale
        self.table = np.zeros(
            (2, self.num_bins, self.num_joints), dtype=float)
        self.pd_sum = np.zeros(self.num_joints, dtype=float)
        self.last_torque = np.zeros(self.num_joints, dtype=float)
        self.last_table = np.zeros(self.num_joints, dtype=float)
        self.last_teacher_pd = np.zeros(self.num_joints, dtype=float)
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
        active_step = self._active_step(walking_control)
        self.valid_active_step = False
        self.walking_active = False
        self.support_side = -1
        self.phase = np.nan
        self.phase_bin = -1
        self.time_to_landing = np.nan
        if active_step is None:
            return

        try:
            side = int(active_step.side)
            duration = float(active_step.duration)
            time_to_landing = float(
                walking_control.stepping_controller.time_to_landing)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        if (side not in (0, 1) or not np.isfinite(duration) or
                duration <= 0.0 or not np.isfinite(time_to_landing)):
            return

        phase = np.clip(
            1.0 - time_to_landing / duration,
            0.0,
            np.nextafter(1.0, 0.0),
        )
        self.support_side = side
        self.phase = float(phase)
        self.phase_bin = min(
            int(self.phase * self.num_bins), self.num_bins - 1)
        self.time_to_landing = time_to_landing
        self.valid_active_step = True
        self.walking_active = bool(getattr(active_step, 'stepping', False))

    def predict(self, walking_control):
        if self.table is None:
            raise RuntimeError('PhaseFELController.initialize() was not called')
        self._read_context(walking_control)
        self.last_torque.fill(0.0)
        self.last_table.fill(0.0)
        if self.valid_active_step:
            self.last_table[:] = self.table[
                self.support_side, self.phase_bin]
        if self.enabled and self.walking_active and self.valid_active_step:
            self.last_torque[:] = np.clip(
                self.last_table, -self.fel_limit, self.fel_limit)
            self.last_torque[~self.active_mask] = 0.0
        return self.last_torque.copy()

    def _flush_accumulator(self):
        if self.accumulator_key is None or self.sample_count <= 0:
            return
        side, phase_bin = self.accumulator_key
        mean_pd = self.pd_sum / float(self.sample_count)
        mean_pd[~self.active_mask] = 0.0
        self.table[side, phase_bin] += self.alpha * mean_pd
        self.table[side, phase_bin] = np.clip(
            self.table[side, phase_bin], -self.fel_limit, self.fel_limit)
        self.table[side, phase_bin, ~self.active_mask] = 0.0
        self.update_count += 1

    def update(self, tau_pd, ik_solver_ok, ik_command_ok):
        if self.table is None:
            raise RuntimeError('PhaseFELController.initialize() was not called')
        tau_pd = np.asarray(tau_pd, dtype=float)
        if tau_pd.shape != (self.num_joints,):
            raise ValueError(
                'tau_pd must have {} entries'.format(self.num_joints))
        if not np.all(np.isfinite(tau_pd)):
            raise ValueError('tau_pd must be finite')
        self.last_teacher_pd[:] = tau_pd
        self.last_teacher_pd[~self.active_mask] = 0.0

        key = None
        if self.valid_active_step:
            key = (self.support_side, self.phase_bin)
        if key != self.accumulator_key:
            self._flush_accumulator()
            self.pd_sum.fill(0.0)
            self.sample_count = 0
            self.accumulator_key = key

        self.learning_gate = bool(
            self.enabled and
            self.walking_active and
            self.valid_active_step and
            np.isfinite(self.phase) and
            bool(ik_solver_ok) and
            bool(ik_command_ok) and
            self.time_to_landing >= 0.0
        )
        if self.learning_gate:
            self.pd_sum += self.last_teacher_pd
            self.sample_count += 1
        return self.last_teacher_pd.copy()

    def reset_learning(self):
        if self.table is None:
            raise RuntimeError('PhaseFELController.initialize() was not called')
        self.table.fill(0.0)
        self.pd_sum.fill(0.0)
        self.sample_count = 0
        self.accumulator_key = None
        self.last_torque.fill(0.0)
        self.last_table.fill(0.0)
        self.last_teacher_pd.fill(0.0)
        self.learning_gate = False
        self.update_count = 0

    def global_log_fields(self):
        header = [
            'phase_fel_enabled',
            'phase_fel_num_bins',
            'phase_fel_alpha',
            'phase_fel_output_limit_ratio',
            'phase_fel_joint_group',
            'phase_fel_active_joint_count',
            'phase_fel_support_side',
            'phase_fel_phase',
            'phase_fel_bin',
            'phase_fel_update_count',
            'phase_fel_table_l2_norm',
            'phase_fel_learning_gate',
        ]
        values = [
            int(self.enabled),
            self.num_bins,
            self.alpha,
            self.output_limit_ratio,
            self.joint_group,
            int(np.count_nonzero(self.active_mask)),
            self.support_side,
            self.phase,
            self.phase_bin,
            self.update_count,
            float(np.linalg.norm(self.table)),
            int(self.learning_gate),
        ]
        return header, values

    def joint_log_fields(self, joint_name, index):
        header = [
            '{}_tau_phase_fel_Nm'.format(joint_name),
            '{}_phase_fel_table_Nm'.format(joint_name),
            '{}_phase_fel_teacher_pd_Nm'.format(joint_name),
            '{}_phase_fel_joint_enabled'.format(joint_name),
        ]
        values = [
            self.last_torque[index],
            self.last_table[index],
            self.last_teacher_pd[index],
            int(self.active_mask[index]),
        ]
        return header, values
