import numpy as np


class PhaseFELController:
    """Joint-PD FEL using gait phase bins for periodic torque residuals."""

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

        # Contact/step-transition guard.
        #
        # Do not LEARN immediately before or after a step transition.
        # Prediction/output remains enabled.
        self.contact_guard_time = 0.05  # [s], default 50 ms

        self.num_joints = 0
        self.active_mask = None
        self.torque_scale = None
        self.fel_limit = None
        self.table = None

        # One-step-delayed learning accumulator. The table used for
        # prediction is never modified while the active step is running.
        # PD residuals are collected for every phase bin and committed only
        # when the step changes, so step n teaches the next gait cycle.
        self.pd_sum = None
        self.sample_count = None
        self.accumulator_step_key = None
        self.accumulator_step_side = -1
        self.step_key = None
        self.step_tbegin = np.nan

        self.support_side = -1
        self.phase = np.nan
        self.phase_bin = -1
        self.valid_active_step = False
        self.walking_active = False

        self.step_duration = np.nan
        self.time_to_landing = np.nan
        self.time_from_step_start = np.nan

        self.last_torque = None
        self.last_table = None
        self.last_teacher_pd = None

        self.learning_gate = False
        self.contact_guard_active = False

        self.update_count = 0
        self.commit_count = 0
        self.last_commit_bins = 0
        self.last_commit_samples = 0
        self.last_commit_delta_l2_norm = 0.0

    def configure(
            self,
            enabled=True,
            num_bins=280,
            alpha=0.1,
            output_limit_ratio=0.1,
            joint_group='legs',
            contact_guard_time=0.05):

        self.enabled = bool(enabled)
        self.num_bins = int(num_bins)
        self.alpha = float(alpha)
        self.output_limit_ratio = float(output_limit_ratio)
        self.joint_group = str(joint_group).lower()
        self.contact_guard_time = float(contact_guard_time)

        if self.num_bins <= 0:
            raise ValueError('num_bins must be positive')

        if not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError('alpha must be finite and in [0, 1]')

        if (not np.isfinite(self.output_limit_ratio) or
                not 0.0 < self.output_limit_ratio <= 1.0):
            raise ValueError(
                'output_limit_ratio must be finite and in (0, 1]')

        if self.joint_group not in ('all', 'legs', 'pitch'):
            raise ValueError(
                "joint_group must be 'all', 'legs', or 'pitch'")

        if (not np.isfinite(self.contact_guard_time) or
                self.contact_guard_time < 0.0):
            raise ValueError(
                'contact_guard_time must be finite and >= 0')

    @staticmethod
    def _joint_vector(body, attribute):
        return np.asarray([
            float(getattr(body.joint(index), attribute))
            for index in range(body.numJoints)
        ], dtype=float)

    def initialize(self, body):
        self.num_joints = int(body.numJoints)

        joint_names = tuple(
            body.joint(index).name
            for index in range(self.num_joints)
        )

        if self.joint_group == 'all':
            self.active_mask = np.ones(
                self.num_joints, dtype=bool)

        else:
            suffixes = (
                self.LEG_SUFFIXES
                if self.joint_group == 'legs'
                else self.PITCH_SUFFIXES
            )

            self.active_mask = np.asarray([
                (name.startswith('left_') or
                 name.startswith('right_')) and
                any(name.endswith(suffix) for suffix in suffixes)
                for name in joint_names
            ], dtype=bool)

            expected = 12 if self.joint_group == 'legs' else 6

            if np.count_nonzero(self.active_mask) != expected:
                raise ValueError(
                    'G1 {} mask expected {} joints, found {}: {}'.format(
                        self.joint_group,
                        expected,
                        int(np.count_nonzero(self.active_mask)),
                        ', '.join(
                            np.asarray(joint_names)[self.active_mask]
                        )
                    )
                )

        torque_lower = self._joint_vector(body, 'u_lower')
        torque_upper = self._joint_vector(body, 'u_upper')

        self.torque_scale = np.maximum(
            np.abs(torque_lower),
            np.abs(torque_upper)
        )

        if (not np.all(np.isfinite(self.torque_scale)) or
                np.any(self.torque_scale <= 0.0)):
            raise ValueError(
                'phase FEL requires finite positive joint torque limits')

        self.fel_limit = (
            self.output_limit_ratio * self.torque_scale
        )

        self.table = np.zeros(
            (2, self.num_bins, self.num_joints),
            dtype=float
        )

        self.pd_sum = np.zeros(
            (self.num_bins, self.num_joints),
            dtype=float
        )

        self.sample_count = np.zeros(
            self.num_bins,
            dtype=np.int64
        )

        self.last_torque = np.zeros(
            self.num_joints,
            dtype=float
        )

        self.last_table = np.zeros(
            self.num_joints,
            dtype=float
        )

        self.last_teacher_pd = np.zeros(
            self.num_joints,
            dtype=float
        )

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
        self.step_key = None
        self.step_tbegin = np.nan

        self.step_duration = np.nan
        self.time_to_landing = np.nan
        self.time_from_step_start = np.nan

        self.contact_guard_active = False

        if active_step is None:
            return

        try:
            side = int(active_step.side)
            duration = float(active_step.duration)
            step_tbegin = float(active_step.tbegin)

            time_to_landing = float(
                walking_control.stepping_controller.time_to_landing
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            return

        if (side not in (0, 1) or
                not np.isfinite(duration) or
                duration <= 0.0 or
                not np.isfinite(step_tbegin) or
                not np.isfinite(time_to_landing)):
            return

        phase = np.clip(
            1.0 - time_to_landing / duration,
            0.0,
            np.nextafter(1.0, 0.0),
        )

        # Time elapsed from start of this step.
        time_from_step_start = duration - time_to_landing

        self.support_side = side
        self.phase = float(phase)
        self.phase_bin = min(
            int(self.phase * self.num_bins),
            self.num_bins - 1
        )
        self.step_tbegin = step_tbegin
        self.step_key = (side, step_tbegin)

        self.step_duration = duration
        self.time_to_landing = time_to_landing
        self.time_from_step_start = time_from_step_start

        self.valid_active_step = True
        self.walking_active = bool(
            getattr(active_step, 'stepping', False)
        )

        # --------------------------------------------------------
        # Contact guard
        #
        # Disable LEARNING:
        #
        #   1. immediately after a new step starts
        #   2. immediately before the next landing
        #
        # Example with 50 ms:
        #
        #   0.00 -- 0.05 s      : no learning
        #   0.05 -- T-0.05 s    : learning
        #   T-0.05 -- T         : no learning
        #
        # FEL prediction/output is NOT disabled here.
        # --------------------------------------------------------

        guard = self.contact_guard_time

        self.contact_guard_active = bool(
            time_from_step_start < guard or
            time_to_landing < guard
        )

    def _commit_step_accumulator(self):
        """Commit the completed step's phase-bin teacher residuals."""
        if (self.accumulator_step_key is None or
                self.accumulator_step_side not in (0, 1)):
            return

        valid_bins = self.sample_count > 0
        if not np.any(valid_bins):
            self.last_commit_bins = 0
            self.last_commit_samples = 0
            self.last_commit_delta_l2_norm = 0.0
            return

        side = self.accumulator_step_side
        mean_pd = np.zeros_like(self.pd_sum)
        mean_pd[valid_bins] = (
            self.pd_sum[valid_bins] /
            self.sample_count[valid_bins, None]
        )
        mean_pd[:, ~self.active_mask] = 0.0

        old_table = self.table[side].copy()
        self.table[side] += self.alpha * mean_pd
        self.table[side] = np.clip(
            self.table[side],
            -self.fel_limit,
            self.fel_limit
        )
        self.table[side, :, ~self.active_mask] = 0.0

        delta = self.table[side] - old_table
        self.last_commit_bins = int(np.count_nonzero(valid_bins))
        self.last_commit_samples = int(np.sum(self.sample_count))
        self.last_commit_delta_l2_norm = float(np.linalg.norm(delta))
        self.update_count += self.last_commit_bins
        self.commit_count += 1

    def _handle_step_transition(self):
        """Commit the previous step before using the next step's table."""
        current_key = self.step_key if self.valid_active_step else None

        if current_key == self.accumulator_step_key:
            return

        self._commit_step_accumulator()
        self.pd_sum.fill(0.0)
        self.sample_count.fill(0)

        self.accumulator_step_key = current_key
        self.accumulator_step_side = (
            self.support_side if current_key is not None else -1
        )

    def predict(self, walking_control):
        if self.table is None:
            raise RuntimeError(
                'PhaseFELController.initialize() was not called'
            )

        self._read_context(walking_control)
        self._handle_step_transition()

        self.last_torque.fill(0.0)
        self.last_table.fill(0.0)

        if self.valid_active_step:
            self.last_table[:] = self.table[
                self.support_side,
                self.phase_bin
            ]

        # IMPORTANT:
        # Contact guard does NOT stop prediction.
        #
        # We only stop learning near contact.
        #
        # Previously learned feed-forward torque remains active.
        if (self.enabled and
                self.walking_active and
                self.valid_active_step):

            self.last_torque[:] = np.clip(
                self.last_table,
                -self.fel_limit,
                self.fel_limit
            )

            self.last_torque[~self.active_mask] = 0.0

        return self.last_torque.copy()

    def update(
            self,
            tau_pd,
            ik_solver_ok,
            ik_command_ok):

        if self.table is None:
            raise RuntimeError(
                'PhaseFELController.initialize() was not called'
            )

        tau_pd = np.asarray(
            tau_pd,
            dtype=float
        )

        if tau_pd.shape != (self.num_joints,):
            raise ValueError(
                'tau_pd must have {} entries'.format(
                    self.num_joints
                )
            )

        if not np.all(np.isfinite(tau_pd)):
            raise ValueError(
                'tau_pd must be finite'
            )

        self.last_teacher_pd[:] = tau_pd
        self.last_teacher_pd[
            ~self.active_mask
        ] = 0.0

        # --------------------------------------------------------
        # Learning gate
        # --------------------------------------------------------

        self.learning_gate = bool(
            self.enabled and
            self.walking_active and
            self.valid_active_step and
            np.isfinite(self.phase) and
            bool(ik_solver_ok) and
            bool(ik_command_ok) and
            self.time_to_landing >= 0.0 and

            # NEW:
            # Do not learn close to contact / step transition.
            not self.contact_guard_active
        )

        if self.learning_gate:
            self.pd_sum[self.phase_bin] += self.last_teacher_pd
            self.sample_count[self.phase_bin] += 1

        return self.last_teacher_pd.copy()

    def reset_learning(self):
        if self.table is None:
            raise RuntimeError(
                'PhaseFELController.initialize() was not called'
            )

        self.table.fill(0.0)
        self.pd_sum.fill(0.0)
        self.sample_count.fill(0)

        self.accumulator_step_key = None
        self.accumulator_step_side = -1
        self.step_key = None
        self.step_tbegin = np.nan

        self.last_torque.fill(0.0)
        self.last_table.fill(0.0)
        self.last_teacher_pd.fill(0.0)

        self.learning_gate = False
        self.contact_guard_active = False

        self.update_count = 0
        self.commit_count = 0
        self.last_commit_bins = 0
        self.last_commit_samples = 0
        self.last_commit_delta_l2_norm = 0.0

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
            'phase_fel_commit_count',
            'phase_fel_table_l2_norm',
            'phase_fel_last_commit_bins',
            'phase_fel_last_commit_samples',
            'phase_fel_last_commit_delta_l2_norm',
            'phase_fel_learning_gate',

            # NEW
            'phase_fel_contact_guard_time_s',
            'phase_fel_contact_guard_active',
            'phase_fel_time_from_step_start_s',
            'phase_fel_time_to_landing_s',
            'phase_fel_step_tbegin_s',
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
            self.commit_count,
            float(np.linalg.norm(self.table)),
            self.last_commit_bins,
            self.last_commit_samples,
            self.last_commit_delta_l2_norm,
            int(self.learning_gate),

            # NEW
            self.contact_guard_time,
            int(self.contact_guard_active),
            self.time_from_step_start,
            self.time_to_landing,
            self.step_tbegin,
        ]

        return header, values

    def joint_log_fields(
            self,
            joint_name,
            index):

        header = [
            '{}_tau_phase_fel_Nm'.format(
                joint_name
            ),
            '{}_phase_fel_table_Nm'.format(
                joint_name
            ),
            '{}_phase_fel_teacher_pd_Nm'.format(
                joint_name
            ),
            '{}_phase_fel_joint_enabled'.format(
                joint_name
            ),
        ]

        values = [
            self.last_torque[index],
            self.last_table[index],
            self.last_teacher_pd[index],
            int(self.active_mask[index]),
        ]

        return header, values
