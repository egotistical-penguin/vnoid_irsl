import numpy as np
from scipy.spatial.transform import Rotation as R


def from_roll_pitch_yaw(rpy):
    return R.from_euler("xyz", rpy)


def rotate(ori, vec):
    return ori.apply(vec)


def inv_rotate(ori, vec):
    return ori.inv().apply(vec)


class Stabilizer:
    def __init__(self):
        self.min_contact_force = 1.0
        self.force_ctrl_damping = 0.0
        self.force_ctrl_gain = 0.0
        self.force_ctrl_limit = 0.0
        self.moment_ctrl_damping = 0.0
        self.moment_ctrl_gain = 0.0
        self.moment_ctrl_limit = 0.0

        # default gain setting
        self.orientation_ctrl_gain_p = 10.0
        self.orientation_ctrl_gain_d = 10.0
        self.dcm_ctrl_gain = 2.0

        self.base_tilt_rate = 0.0
        self.base_tilt_damping_p = 0.1
        self.base_tilt_damping_d = 0.1

        self.recovery_moment_limit = 100.0
        self.dcm_deviation_limit = 0.3

        # dpos, drot を numpy array (Vector3) として初期化
        self.dpos = [np.zeros(3, dtype=float) for _ in range(2)]
        self.drot = [np.zeros(3, dtype=float) for _ in range(2)]

    def CalcZmp(self, param, centroid, foot):
        # get actual force from the sensor
        for i in range(2):
            # set contact state
            foot[i].contact = (foot[i].force[2] >= self.min_contact_force)

            # measure continuous contact duration
            if foot[i].contact:
                foot[i].zmp = np.array([
                    -foot[i].moment[1] / foot[i].force[2],
                     foot[i].moment[0] / foot[i].force[2],
                     0.0
                ])
            else:
                foot[i].zmp = np.zeros(3, dtype=float)

        # both feet not in contact
        if not foot[0].contact and not foot[1].contact:
            foot[0].balance = 0.5
            foot[1].balance = 0.5
            centroid.zmp = np.zeros(3, dtype=float)
        else:
            f0 = max(0.0, foot[0].force[2])
            f1 = max(0.0, foot[1].force[2])
            foot[0].balance = f0 / (f0 + f1)
            foot[1].balance = f1 / (f0 + f1)

            # 注: ori_ref がSciPy等のRotationオブジェクトの場合、
            # `ori_ref.apply(vector)` のように書き換える必要があるかもしれません。
            centroid.zmp = (
                foot[0].balance * (foot[0].pos_ref + rotate(foot[0].ori_ref, foot[0].zmp))
                + foot[1].balance * (foot[1].pos_ref + rotate(foot[1].ori_ref, foot[1].zmp))
            )           
            
    def CalcForceDistribution(self, param, centroid, foot):
        # switch based on contact state
        if not foot[0].contact_ref and not foot[1].contact_ref:
            foot[0].balance_ref = 0.5
            foot[1].balance_ref = 0.5
            foot[0].zmp_ref = np.zeros(3, dtype=float)
            foot[1].zmp_ref = np.zeros(3, dtype=float)

        if foot[0].contact_ref and not foot[1].contact_ref:
            foot[0].balance_ref = 1.0
            foot[1].balance_ref = 0.0
            foot[0].zmp_ref = inv_rotate(foot[0].ori_ref, centroid.zmp_ref - foot[0].pos_ref)
            foot[1].zmp_ref = np.zeros(3, dtype=float)

        if not foot[0].contact_ref and foot[1].contact_ref:
            foot[0].balance_ref = 0.0
            foot[1].balance_ref = 1.0
            foot[0].zmp_ref = np.zeros(3, dtype=float)
            foot[1].zmp_ref = inv_rotate(foot[1].ori_ref, centroid.zmp_ref - foot[1].pos_ref)

        if foot[0].contact_ref and foot[1].contact_ref:
            b = np.zeros(2, dtype=float)
            pdiff = foot[1].pos_ref - foot[0].pos_ref
            pdiff2 = np.dot(pdiff, pdiff)
            eps = 1.0e-10

            if pdiff2 < eps:
                b[0] = b[1] = 0.5
            else:
                b[0] = np.dot(pdiff, foot[1].pos_ref - centroid.zmp_ref) / pdiff2
                b[0] = np.clip(b[0], 0.0, 1.0)
                b[1] = 1.0 - b[0]

            foot[0].balance_ref = b[0]
            foot[1].balance_ref = b[1]

            zmp_proj = b[0] * foot[0].pos_ref + b[1] * foot[1].pos_ref
            b2 = np.dot(b, b)

            foot[0].zmp_ref = (b[0] / b2) * (inv_rotate(foot[0].ori_ref, centroid.zmp_ref - zmp_proj))
            foot[1].zmp_ref = (b[1] / b2) * (inv_rotate(foot[1].ori_ref, centroid.zmp_ref - zmp_proj))

        # limit zmp
        for i in range(2):
            foot[i].zmp_ref = np.clip(foot[i].zmp_ref, param.zmp_min, param.zmp_max)
            foot[i].force_ref = inv_rotate(foot[i].ori_ref, foot[i].balance_ref * centroid.force_ref)
            foot[i].moment_ref[0] = foot[i].force_ref[2] * foot[i].zmp_ref[1]
            foot[i].moment_ref[1] = -foot[i].force_ref[2] * foot[i].zmp_ref[0]
            foot[i].moment_ref[2] = foot[i].balance_ref * centroid.moment_ref[2]

    def CalcBaseTilt(self, timer, param, base, theta, omega):
        # desired angular acceleration for regulating orientation (in local coordinate)
        omegadd_local = np.array([
            -(self.orientation_ctrl_gain_p * theta[0] + self.orientation_ctrl_gain_d * omega[0]),
            -(self.orientation_ctrl_gain_p * theta[1] + self.orientation_ctrl_gain_d * omega[1]),
            0.0
        ])

        omegadd_base = np.array([
            -self.base_tilt_rate * omegadd_local[0] - self.base_tilt_damping_p * base.angle_ref[0] - self.base_tilt_damping_d * base.angvel_ref[0],
            -self.base_tilt_rate * omegadd_local[1] - self.base_tilt_damping_p * base.angle_ref[1] - self.base_tilt_damping_d * base.angvel_ref[1],
             0.0
        ])

        base.angle_ref += base.angvel_ref * timer.dt
        base.ori_ref = from_roll_pitch_yaw(base.angle_ref)
        base.angvel_ref += omegadd_base * timer.dt

    def CalcDcmDynamics(self, timer, param, base, foot, theta, omega, centroid):
        T = param.T
        m = param.total_mass
        h = param.com_height

        offset = np.array([0.0, 0.0, param.com_height])

        # desired angular acceleration for regulating orientation (in local coordinate)
        omegadd_local = np.array([
            -(self.orientation_ctrl_gain_p * theta[0] + self.orientation_ctrl_gain_d * omega[0]),
            -(self.orientation_ctrl_gain_p * theta[1] + self.orientation_ctrl_gain_d * omega[1]),
            0.0
        ])

        # desired moment (in local coordinate)
        Ld_local = np.array([
            param.nominal_inertia[0] * omegadd_local[0],
            param.nominal_inertia[1] * omegadd_local[1],
            param.nominal_inertia[2] * omegadd_local[2]
        ])

        # limit recovery moment for safety
        Ld_local = np.clip(Ld_local, -self.recovery_moment_limit, self.recovery_moment_limit)
        Ld = rotate(base.ori_ref, Ld_local)
        delta = np.array([-(1.0 / (m * h)) * Ld[1], (1.0 / (m * h)) * Ld[0], 0.0], dtype=float)

        centroid.zmp_ref = centroid.zmp_target + self.dcm_ctrl_gain * (
            centroid.dcm_ref - centroid.dcm_target
        )

        if (foot[0].contact_ref and not foot[1].contact_ref) or (
            not foot[0].contact_ref and foot[1].contact_ref
        ):
            sup = 0 if foot[0].contact_ref else 1
            zmp_local = inv_rotate(foot[sup].ori_ref, centroid.zmp_ref - foot[sup].pos_ref)
            zmp_local = np.clip(zmp_local, param.zmp_min, param.zmp_max)
            centroid.zmp_ref = foot[sup].pos_ref + rotate(foot[sup].ori_ref, zmp_local)

        # calc DCM derivative
        dcm_d = (1.0 / T) * (centroid.dcm_ref - (centroid.zmp_ref + np.array([0.0, 0.0, h]))) + T * delta

        # calc CoM acceleration
        centroid.com_acc_ref = (1.0 / T) * (dcm_d - centroid.com_vel_ref)

        # update DCM
        centroid.dcm_ref += dcm_d * timer.dt

        # limit deviation from reference dcm
        centroid.dcm_ref = np.clip(centroid.dcm_ref, 
                                   centroid.dcm_target - self.dcm_deviation_limit, 
                                   centroid.dcm_target + self.dcm_deviation_limit)

        # calc CoM velocity from dcm
        centroid.com_vel_ref = (1.0 / T) * (centroid.dcm_ref - centroid.com_pos_ref)

        # update CoM position
        centroid.com_pos_ref += centroid.com_vel_ref * timer.dt

    #CalcDcmDynamicsSimple(timer, param, None, foot, None, None, None)
    def CalcDcmDynamicsSimple(self, timer, param, centroid, no_dcm_gain=True, no_dcm_derivative=True):
        T = param.T
        # m = param.total_mass
        h = param.com_height

        if no_dcm_gain:
            centroid.zmp_ref = centroid.zmp_target.copy()
        else:
            centroid.zmp_ref = centroid.zmp_target + self.dcm_ctrl_gain * (centroid.dcm_ref - centroid.dcm_target)

        # calc DCM derivative
        dcm_d = (1.0 / T) * (centroid.dcm_ref - (centroid.zmp_ref + np.array([0.0, 0.0, h])))

        # calc CoM acceleration
        centroid.com_acc_ref = (1.0 / T) * (dcm_d - centroid.com_vel_ref)

        #print("0:centroid.dcm_ref: ", centroid.dcm_ref)
        #print("dcm_d: ", dcm_d)
        #print("centroid.dcm_target: ", centroid.dcm_target)
        if no_dcm_derivative:
            centroid.dcm_ref = centroid.dcm_target.copy()
        else:
            centroid.dcm_ref += dcm_d * timer.dt
        #print("1:centroid.dcm_ref: ", centroid.dcm_ref)

        # calc CoM velocity from dcm
        centroid.com_vel_ref = (1.0 / T) * (centroid.dcm_ref - centroid.com_pos_ref)

        # update CoM position
        centroid.com_pos_ref += centroid.com_vel_ref * timer.dt

    def Update(self, timer, param, centroid, base, foot):
        # calc zmp from forces
        self.CalcZmp(param, centroid, foot)

        # error between desired and actual base link orientation
        theta = base.angle  - base.angle_ref
        omega = base.angvel - base.angvel_ref

        # calc base link tilt
        self.CalcBaseTilt(timer, param, base, theta, omega)

        # calc dcm and zmp 
        self.CalcDcmDynamics(timer, param, base, foot, theta, omega, centroid)

        # calc desired force applied to CoM
        centroid.force_ref  = param.total_mass * (centroid.com_acc_ref + np.array([0.0, 0.0, param.gravity]))
        centroid.moment_ref = np.zeros(3, dtype=float)

        # calculate desired forces from desired zmp
        self.CalcForceDistribution(param, centroid, foot)

        for i in range(2):
            # ground reaction force control
            if foot[i].contact:
                # Numpyベクトルの操作を活用し、for j in range(3): をベクトル計算で置換しています
                self.dpos[i] += (-self.force_ctrl_damping * self.dpos[i] + 
                                 self.force_ctrl_gain * (foot[i].force_ref - foot[i].force)) * timer.dt
                self.dpos[i] = np.clip(self.dpos[i], -self.force_ctrl_limit, self.force_ctrl_limit)

                self.drot[i] += (-self.moment_ctrl_damping * self.drot[i] + 
                                 self.moment_ctrl_gain * (foot[i].moment_ref - foot[i].moment)) * timer.dt
                self.drot[i] = np.clip(self.drot[i], -self.moment_ctrl_limit, self.moment_ctrl_limit)

                # feedback to desired foot pose
                foot[i].pos_ref -= self.dpos[i]
                foot[i].angle_ref -= self.drot[i]
                foot[i].ori_ref = from_roll_pitch_yaw(foot[i].angle_ref)