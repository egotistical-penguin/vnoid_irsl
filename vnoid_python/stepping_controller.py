"""
Stepping Controller - C++ の stepping_controller.cpp を Python に変換
リアルタイムの足の軌跡制御と DCM ベースのステップ調整
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.spatial.transform import Rotation as R
# TODO R => coordinates
from footstep_planner import Step, Footstep, Param, Ground, rotate_vector, printStep, printVec3

# TODO pos + angle => coordinates

pi = 3.14159265358979
eps = 1.0e-10

@dataclass
class Timer:
    """タイマー情報"""
    time: float = 0.0
    count: int = 0
    dt: float = 0.001
    def CountUp(self):
        self.count += 1
        self.time += self.dt

@dataclass
class Centroid:
    """重心情報"""
    dcm_ref: np.ndarray = None       # 参考 DCM
    dcm_target: np.ndarray = None    # 目標 DCM
    zmp_ref: np.ndarray = None       # 参考 ZMP
    zmp_target: np.ndarray = None    # 目標 ZMP
    ####
    force_ref:  np.ndarray = None  #/< reference force
    moment_ref: np.ndarray = None  #/< reference moment
    zmp:        np.ndarray = None  #/< current ZMP
    dcm:        np.ndarray = None  #/< DCM (divergent component of motion)
    com_pos: np.ndarray = None      # 目標? CoM
    com_pos_ref: np.ndarray = None # 参考 CoM
    com_vel_ref: np.ndarray = None #/< reference velocity of CoM
    com_acc_ref: np.ndarray = None #/< reference acceleration of CoMxo

    def __post_init__(self):
        if self.dcm_ref is None:
            self.dcm_ref = np.array([0.0, 0.0, 0.0])
        if self.dcm_target is None:
            self.dcm_target = np.array([0.0, 0.0, 0.0])
        if self.zmp_ref is None:
            self.zmp_ref = np.array([0.0, 0.0, 0.0])
        if self.zmp_target is None:
            self.zmp_target = np.array([0.0, 0.0, 0.0])
        if self.force_ref is None:
            self.force_ref = np.array([0.0, 0.0, 0.0])
        if self.moment_ref is None:
            self.moment_ref = np.array([0.0, 0.0, 0.0])
        if self.zmp is None:
            self.zmp = np.array([0.0, 0.0, 0.0])
        if self.dcm is None:
            self.dcm = np.array([0.0, 0.0, 0.0])
        if self.com_pos_ref is None:
            self.com_pos_ref = np.array([0.0, 0.0, 0.0])
        if self.com_pos is None:
            self.com_pos = np.array([0.0, 0.0, 0.0])
        if self.com_vel_ref is None:
            self.com_vel_ref = np.array([0.0, 0.0, 0.0])
        if self.com_acc_ref is None:
            self.com_acc_ref = np.array([0.0, 0.0, 0.0])

@dataclass
class Base:
    """ベース（ボディ）情報"""
    angle: np.ndarray = None         # 現在の角度 [roll, pitch, yaw]
    angle_ref: np.ndarray = None     # 参考角度
    ori: R = None                    # 現在の向き
    ori_ref: R = None                # 参考向き

    pos:        np.ndarray = None #/< position
    pos_ref:    np.ndarray = None #/< reference position
    vel:        np.ndarray = None #/< velocity
    vel_ref:    np.ndarray = None #/< reference velocity
    angvel:     np.ndarray = None #/< current angular velocity
    angvel_ref: np.ndarray = None #/< reference angular velocity
    acc:        np.ndarray = None
    acc_ref:    np.ndarray = None #/< reference acceleration
    angacc:     np.ndarray = None
    angacc_ref: np.ndarray = None #/< reference angular acceleration

    def __post_init__(self):
        if self.pos is None:
            self.pos = np.array([0.0, 0.0, 0.0])
        if self.pos_ref is None:
            self.pos_ref = np.array([0.0, 0.0, 0.0])
        if self.angle is None:
            self.angle = np.array([0.0, 0.0, 0.0])
        if self.angle_ref is None:
            self.angle_ref = np.array([0.0, 0.0, 0.0])
        if self.ori is None:
            self.ori = R.from_euler('xyz', [0, 0, 0])
        if self.ori_ref is None:
            self.ori_ref = R.from_euler('xyz', [0, 0, 0])
        if self.vel_ref is None:
            self.vel_ref = np.array([0.0, 0.0, 0.0])
        if self.angvel is None:
            self.angvel = np.array([0.0, 0.0, 0.0])
        if self.angvel_ref is None:
            self.angvel_ref = np.array([0.0, 0.0, 0.0])
        if self.acc is None:
            self.acc = np.array([0.0, 0.0, 0.0])
        if self.acc_ref is None:
            self.acc_ref = np.array([0.0, 0.0, 0.0])
        if self.angacc is None:
            self.angacc = np.array([0.0, 0.0, 0.0])
        if self.angacc_ref is None:
            self.angacc_ref = np.array([0.0, 0.0, 0.0])


@dataclass
class Foot:
    """足の情報"""
    pos_ref: np.ndarray = None       # 参考位置
    angle_ref: np.ndarray = None     # 参考角度 [roll, pitch, yaw]
    ori_ref: R = None                # 参考向き
    contact_ref: bool = False        # 接触フラグ
    pos: np.ndarray = None
    angle: np.ndarray = None
    ori: R = None
    vel_ref: np.ndarray = None
    angvel_ref: np.ndarray = None
    acc_ref: np.ndarray = None
    angacc_ref: np.ndarray = None
    force: np.ndarray = None
    force_ref: np.ndarray = None
    moment: np.ndarray = None
    moment_ref: np.ndarray = None
    zmp: np.ndarray = None
    zmp_ref: np.ndarray = None
    contact: bool = False
    balance: float = 0.0
    balance_ref: float = 0.0
#>bool        contact;      ///< current contact state (true if foot is in contact with the ground)
#>bool        contact_ref;  ///< reference contact state
#>double      balance;      ///< current balance ratio [0.0, 1.0].  indicates the ratio of vertical reaction force applied to this foot
#>double      balance_ref;  ///< reference balance ratio [0.0, 1.0]
#>Vector3     pos;          ///< position
#>Vector3     pos_ref;      ///< reference position
#>Quaternion  ori;          ///< orientation in quaternion
#>Quaternion  ori_ref;      ///< reference orientation in quaternion
#>Vector3     angle;        ///< orientation in roll-pitch-yaw
#>Vector3     angle_ref;    ///< reference orientation in roll-pitch-yaw
#>Vector3     vel_ref;      ///< reference velocity
#>Vector3     angvel_ref;   ///< reference angular velocity
#>Vector3     acc_ref;      ///< reference acceleration
#>Vector3     angacc_ref;   ///< reference angular acceleration
#>Vector3     force;        ///< ground reaction force acting on this foot
#>Vector3     force_ref;    ///< reference ground reaction force
#>Vector3     moment;       ///< ground reaction moment acting on this foot
#>Vector3     moment_ref;   ///< reference ground reaction moment
#>Vector3     zmp;          ///< ZMP (i.e., center-of-pressure) of this foot
#>Vector3     zmp_ref;      ///< reference ZMP of this foot

    def __post_init__(self):
        if self.pos_ref is None:
            self.pos_ref = np.array([0.0, 0.0, 0.0])
        if self.pos is None:
            self.pos = np.array([0.0, 0.0, 0.0])
        if self.angle_ref is None:
            self.angle_ref = np.array([0.0, 0.0, 0.0])
        if self.angle is None:
            self.angle = np.array([0.0, 0.0, 0.0])
        if self.ori_ref is None:
            self.ori_ref = R.from_euler("xyz", [0, 0, 0]).as_quat(scalar_first=True)
        if self.ori is None:
            self.ori = R.from_euler("xyz", [0, 0, 0]).as_quat(scalar_first=True)
        if self.vel_ref is None:
            self.vel_ref = np.array([0.0, 0.0, 0.0])
        if self.angvel_ref is None:
            self.angvel_ref = np.array([0.0, 0.0, 0.0])
        if self.acc_ref is None:
            self.acc_ref = np.array([0.0, 0.0, 0.0])
        if self.angacc_ref is None:
            self.angacc_ref = np.array([0.0, 0.0, 0.0])
        if self.force is None:
            self.force = np.array([0.0, 0.0, 0.0])
        if self.force_ref is None:
            self.force_ref = np.array([0.0, 0.0, 0.0])
        if self.moment is None:
            self.moment = np.array([0.0, 0.0, 0.0])
        if self.moment_ref is None:
            self.moment_ref = np.array([0.0, 0.0, 0.0])
        if self.zmp is None:
            self.zmp = np.array([0.0, 0.0, 0.0])
        if self.zmp_ref is None:
            self.zmp_ref = np.array([0.0, 0.0, 0.0])

def fmtVec3(vec3):
    return f'({vec3[0]:.6f}, {vec3[1]:.6f}, {vec3[2]:.6f} )'

def printFoot(foot, prefix=""):
    if foot.contact_ref:
        print(f'{prefix}contact_ref:\t{1}')
    else:
        print(f'{prefix}contact_ref:\t{0}')
    print(f'{prefix}pos_ref:\t' + fmtVec3(foot.pos_ref))
    print(f'{prefix}angle_ref:\t' + fmtVec3(foot.angle_ref))

class SteppingController:
    """リアルタイム足の軌跡制御"""

    def __init__(self):
        self.swing_height = 0.05              # スウィング足の高さ
        self.swing_tilt = 0.0                 # スウィング足の傾き
        self.dsp_duration = 0.1               # ダブルサポート期間
        self.descend_duration = 0.0           # 降下期間
        self.descend_depth = 0.0              # 降下深さ
        self.timing_adaptation_weight = 1.0   # タイミング適応の重み

        self.buffer_ready = False
        self.time_to_landing = 0.0

        self.debug = 4 #
        self.use_land_estimation = True
        self.use_reference_x_reanchor = False
        self.use_reference_y_reanchor = False
        self.reference_x_offset = 0.0
        self.reference_x_offset_target = 0.0
        self.reference_x_offset_start = 0.0
        self.reference_x_blend_elapsed = 0.0
        self.reference_x_blend_duration = 0.0
        self.reference_y_offset = 0.0
        self.reference_y_offset_target = 0.0
        self.reference_y_offset_start = 0.0
        self.reference_y_blend_elapsed = 0.0
        self.reference_y_blend_duration = 0.0
        self.reference_x_reanchor_event_index = 0
        self.reanchor_time_s = np.nan
        self.reanchor_side = -1
        self.reanchor_actual_x = np.nan
        self.reanchor_nominal_x = np.nan
        self.reanchor_error_x = np.nan
        self.reanchor_actual_stride_x = np.nan
        self.reanchor_nominal_stride_x = np.nan
        self.reanchor_prev_actual_x = np.nan
        self.reanchor_prev_nominal_x = np.nan
        self.reanchor_actual_y = np.nan
        self.reanchor_nominal_y = np.nan
        self.reanchor_error_y = np.nan
        self.reanchor_actual_stride_y = np.nan
        self.reanchor_nominal_stride_y = np.nan
        self.reanchor_prev_actual_y = np.nan
        self.reanchor_prev_nominal_y = np.nan

    def _shift_x(self, dx, footstep, footstep_buffer, centroid, foot):
        """Translate the live walking reference frame along world X.

        Only reference quantities are moved.  Actual robot state (``foot.pos``,
        ``centroid.dcm`` / ``centroid.zmp`` / ``centroid.com_pos``) is left
        untouched.  Applying one common translation preserves all relative
        step geometry and therefore does not change nominal stride.
        """
        if abs(dx) <= eps:
            return

        for sequence in (footstep.steps, footstep_buffer.steps):
            for step in sequence:
                step.dcm[0] += dx
                step.zmp[0] += dx
                for side in range(2):
                    step.foot_pos[side][0] += dx

        centroid.dcm_ref[0] += dx
        centroid.dcm_target[0] += dx
        centroid.zmp_ref[0] += dx
        centroid.zmp_target[0] += dx
        centroid.com_pos_ref[0] += dx

        for item in foot:
            item.pos_ref[0] += dx

    def _shift_y(self, dy, footstep, footstep_buffer, centroid, foot):
        """Translate the live walking reference frame along world Y."""
        if abs(dy) <= eps:
            return

        for sequence in (footstep.steps, footstep_buffer.steps):
            for step in sequence:
                step.dcm[1] += dy
                step.zmp[1] += dy
                for side in range(2):
                    step.foot_pos[side][1] += dy

        centroid.dcm_ref[1] += dy
        centroid.dcm_target[1] += dy
        centroid.zmp_ref[1] += dy
        centroid.zmp_target[1] += dy
        centroid.com_pos_ref[1] += dy

        for item in foot:
            item.pos_ref[1] += dy

    def _advance_reference_x_reanchor(self, timer, footstep,
                                      footstep_buffer, centroid, foot):
        if not self.use_reference_x_reanchor:
            return
        if self.reference_x_blend_duration <= 0.0:
            return
        if self.reference_x_blend_elapsed >= self.reference_x_blend_duration:
            return

        self.reference_x_blend_elapsed = min(
            self.reference_x_blend_duration,
            self.reference_x_blend_elapsed + timer.dt)
        alpha = self.reference_x_blend_elapsed / self.reference_x_blend_duration
        # Smoothstep avoids a velocity discontinuity at both ends of the DSP.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        new_offset = (
            self.reference_x_offset_start
            + alpha * (self.reference_x_offset_target
                       - self.reference_x_offset_start))
        dx = new_offset - self.reference_x_offset
        self._shift_x(dx, footstep, footstep_buffer, centroid, foot)
        self.reference_x_offset = new_offset

    def _advance_reference_y_reanchor(self, timer, footstep,
                                      footstep_buffer, centroid, foot):
        if not self.use_reference_y_reanchor:
            return
        if self.reference_y_blend_duration <= 0.0:
            return
        if self.reference_y_blend_elapsed >= self.reference_y_blend_duration:
            return

        self.reference_y_blend_elapsed = min(
            self.reference_y_blend_duration,
            self.reference_y_blend_elapsed + timer.dt)
        alpha = self.reference_y_blend_elapsed / self.reference_y_blend_duration
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        new_offset = (
            self.reference_y_offset_start
            + alpha * (self.reference_y_offset_target
                       - self.reference_y_offset_start))
        dy = new_offset - self.reference_y_offset
        self._shift_y(dy, footstep, footstep_buffer, centroid, foot)
        self.reference_y_offset = new_offset

    def _start_reference_x_reanchor(self, timer, st0, sup, foot):
        """Latch an absolute X-frame target from the newly landed support foot."""
        if not self.use_reference_x_reanchor:
            return

        actual_x = float(foot[sup].pos[0])
        shifted_planned_x = float(st0.foot_pos[sup][0])
        if not np.isfinite(actual_x) or not np.isfinite(shifted_planned_x):
            return

        # The planned queue already contains reference_x_offset.  Remove it to
        # recover the immutable nominal world-frame support position, then form
        # an absolute target.  This prevents re-adding the same error every step.
        nominal_x = shifted_planned_x - self.reference_x_offset
        target = actual_x - nominal_x

        if np.isfinite(self.reanchor_prev_actual_x):
            actual_stride = actual_x - self.reanchor_prev_actual_x
        else:
            actual_stride = np.nan
        if np.isfinite(self.reanchor_prev_nominal_x):
            nominal_stride = nominal_x - self.reanchor_prev_nominal_x
        else:
            nominal_stride = np.nan

        self.reference_x_offset_start = self.reference_x_offset
        self.reference_x_offset_target = target
        self.reference_x_blend_elapsed = 0.0
        self.reference_x_blend_duration = max(self.dsp_duration, timer.dt)

        self.reference_x_reanchor_event_index += 1
        self.reanchor_time_s = float(timer.time)
        self.reanchor_side = int(sup)
        self.reanchor_actual_x = actual_x
        self.reanchor_nominal_x = nominal_x
        self.reanchor_error_x = actual_x - shifted_planned_x
        self.reanchor_actual_stride_x = actual_stride
        self.reanchor_nominal_stride_x = nominal_stride
        self.reanchor_prev_actual_x = actual_x
        self.reanchor_prev_nominal_x = nominal_x

    def _start_reference_y_reanchor(self, timer, st0, sup, foot):
        """Latch an absolute Y-frame target from the newly landed support foot."""
        if not self.use_reference_y_reanchor:
            return

        actual_y = float(foot[sup].pos[1])
        shifted_planned_y = float(st0.foot_pos[sup][1])
        if not np.isfinite(actual_y) or not np.isfinite(shifted_planned_y):
            return

        nominal_y = shifted_planned_y - self.reference_y_offset
        target = actual_y - nominal_y

        if np.isfinite(self.reanchor_prev_actual_y):
            actual_stride = actual_y - self.reanchor_prev_actual_y
        else:
            actual_stride = np.nan
        if np.isfinite(self.reanchor_prev_nominal_y):
            nominal_stride = nominal_y - self.reanchor_prev_nominal_y
        else:
            nominal_stride = np.nan

        self.reference_y_offset_start = self.reference_y_offset
        self.reference_y_offset_target = target
        self.reference_y_blend_elapsed = 0.0
        self.reference_y_blend_duration = max(self.dsp_duration, timer.dt)

        self.reanchor_actual_y = actual_y
        self.reanchor_nominal_y = nominal_y
        self.reanchor_error_y = actual_y - shifted_planned_y
        self.reanchor_actual_stride_y = actual_stride
        self.reanchor_nominal_stride_y = nominal_stride
        self.reanchor_prev_actual_y = actual_y
        self.reanchor_prev_nominal_y = nominal_y

    def update(self, timer: Timer, param: Param, footstep: Footstep, footstep_buffer: Footstep, centroid: Centroid, base: Base, foot: List[Foot]):
        T = param.T
        offset = np.array([0.0, 0.0, param.com_height])
        support_switched = False

        # Apply only the incremental part of a previously latched re-anchor.
        # This is O(number of remaining steps), runs for one DSP window after a
        # landing event, and never regenerates the full trajectory at 1 kHz.
        self._advance_reference_x_reanchor(
            timer, footstep, footstep_buffer, centroid, foot)
        self._advance_reference_y_reanchor(
            timer, footstep, footstep_buffer, centroid, foot)

        ## *A*
        if self.debug > 2:
            print("enter *A*")
            printVec3(centroid.dcm_ref, "centroid.dcm_ref:\t")
            printVec3(centroid.dcm_target, "centroid.dcm_target:\t")
            printVec3(centroid.zmp_ref, "centroid.zmp_ref:\t")
            printVec3(centroid.zmp_target, "centroid.zmp_target:\t")
            for idx, step in enumerate(footstep.steps):
                print(f'steps[{idx}]')
                printStep(step, 'footstep.steps[i].')
            for idx, step in enumerate(footstep_buffer.steps):
                print(f'bsteps[{idx}]')
                printStep(step, 'footstep_buffer.steps[i].')
            for idx, ft in enumerate(foot):
                print(f'foot[{idx}]')
                printFoot(ft, 'foot[i].')

        if self.buffer_ready:
            if self.debug > 1:
                print("buffer_ready")
            #*001*st0 = footstep.steps[0]
            #*001*st1 = footstep.steps[1]
            stb0 = footstep_buffer.steps[0]
            stb1 = footstep_buffer.steps[1]

            t_ref = timer.time - stb0.tbegin
            alpha_ref = math.exp(t_ref / T)

            xi0 = stb0.dcm[:2] - stb0.zmp[:2]
            xi  = centroid.dcm_ref[:2] - stb0.zmp[:2]

            w = self.timing_adaptation_weight
            alpha = ((w * w * alpha_ref) + (np.linalg.norm(xi0) * np.linalg.norm(xi))) / ((w * w) + np.dot(xi0, xi0))
            t_dcm = T * math.log(alpha)

            self.time_to_landing = stb0.duration - t_dcm

            if self.time_to_landing <= 0.0:
                if len(footstep.steps) > 1:
                    footstep.steps.pop(0)
                    if self.debug > 1:
                        print("#1# pop footstep.steps")
                    if len(footstep.steps) == 1:
                        if self.debug > 1:
                            print("#3# end of footstep reached ###")
                        return False
                if self.debug > 1:
                   print("#2# pop/push footstep_buffer.steps")
                footstep_buffer.steps[1].dcm = footstep_buffer.steps[0].dcm.copy()
                footstep_buffer.steps.pop(0)
                footstep_buffer.steps.append(Step(stride=0.0, sway=0.0, spacing=0.0, turn=0.0, climb=0.0, duration=0.5, side=0))

                self.buffer_ready = False
                support_switched = True
            else:
                centroid.dcm_target = (stb0.zmp + offset) + alpha_ref * (stb0.dcm - (stb0.zmp + offset))

        ## *B*
        if self.debug > 2:
            print(f'enter *B* : {len(footstep.steps)}')
            for idx, step in enumerate(footstep.steps):
                print(f'steps[{idx}]')
                printStep(step, 'footstep.steps[i].')
            for idx, step in enumerate(footstep_buffer.steps):
                print(f'bsteps[{idx}]')
                printStep(step, 'footstep_buffer.steps[i].')
            for idx, ft in enumerate(foot):
                print(f'foot[{idx}]')
                printFoot(ft, 'foot[i].')

        #if len(footstep.steps) < 2 or len(footstep_buffer.steps) < 2:
        if len(footstep.steps) < 2:
            return False

        ## *C*
        if self.debug > 2:
            print("enter *C*")
        st0 = footstep.steps[0]
        st1 = footstep.steps[1]
        stb0 = footstep_buffer.steps[0]
        stb1 = footstep_buffer.steps[1]
        sup = st0.side
        swg = 1 - st0.side

        if not self.buffer_ready:
            if support_switched:
                self._start_reference_x_reanchor(timer, st0, sup, foot)
                self._start_reference_y_reanchor(timer, st0, sup, foot)
            if self.debug > 1:
                print("!buffer_ready")
            stb0.side = st0.side
            stb1.side = st1.side

            stb0.stepping = st0.stepping
            stb0.duration = st0.duration

            stb0.foot_pos  [sup] = foot[sup].pos_ref.copy()
            stb0.foot_pos  [sup][2] = st0.foot_pos[sup][2]
            stb0.foot_angle[sup] = np.array([0.0, 0.0, foot[sup].angle_ref[2]])
            stb0.foot_ori  [sup] = R.from_euler('xyz', stb0.foot_angle[sup])

            stb0.foot_pos  [swg] = foot[swg].pos_ref.copy()
            stb0.foot_pos  [swg][2] = st0.foot_pos[swg][2]
            stb0.foot_angle[swg] = np.array([0.0, 0.0, foot[swg].angle_ref[2]])
            stb0.foot_ori  [swg] = R.from_euler('xyz', stb0.foot_angle[swg])

            stb0.dcm = centroid.dcm_ref.copy()

            ori_rel_inv = st0.foot_ori[sup].inv()
            ori_rel = ori_rel_inv * st1.foot_ori[swg]
            pos_rel = ori_rel_inv.apply(st1.foot_pos[swg] - st0.foot_pos[sup])
            dcm_rel = ori_rel_inv.apply(st1.dcm - st0.foot_pos[sup])
            if self.debug > 3:
                printVec3(dcm_rel, "dcm_rel:\t")
            stb1.foot_pos  [sup] = stb0.foot_pos  [sup].copy()
            stb1.foot_ori  [sup] = R.from_quat(stb0.foot_ori  [sup].as_quat()) ## = stb0.foot_ori  [sup]
            stb1.foot_angle[sup] = stb0.foot_angle[sup].copy()
            stb1.foot_pos  [swg] = stb0.foot_pos[sup] + stb0.foot_ori[sup].apply(pos_rel)
            stb1.foot_ori  [swg] = stb0.foot_ori[sup] * ori_rel
            stb1.foot_angle[swg] = stb1.foot_ori[swg].as_euler('xyz')
            stb1.dcm = stb0.foot_pos[sup] + stb0.foot_ori[sup].apply(dcm_rel)
            if self.debug > 3:
                printVec3(stb1.dcm, "stb1.dcm:\t")
            ## calc zmp
            alpha = np.exp(stb0.duration / T)
            if abs(alpha - 1.0) > eps:
                stb0.zmp = (1.0 / (alpha - 1.0)) * (alpha * stb0.dcm - stb1.dcm) - offset
            else:
                stb0.zmp = stb0.dcm.copy()

            centroid.zmp_target = stb0.zmp.copy()
            ## store current time
            stb0.tbegin = timer.time
            ## default time-to-landing
            self.time_to_landing = stb0.duration
            self.buffer_ready = True

        if self.debug > 2:
            print("enter *D*")
            for idx, step in enumerate(footstep.steps):
                print(f'steps[{idx}]')
                printStep(step, 'footstep.steps[i].')
            for idx, step in enumerate(footstep_buffer.steps):
                print(f'bsteps[{idx}]')
                printStep(step, 'footstep_buffer.steps[i].')
            for idx, ft in enumerate(foot):
                print(f'foot[{idx}]')
                printFoot(ft, 'foot[i].')

        if self.use_land_estimation:
            # 着地時の DCM を予測
            if self.debug > 3:
                printVec3(stb0.zmp, "stb0.zmp:\t")
                printVec3(offset, "offset:\t")
                printVec3(centroid.dcm_ref, "centroid.dcm_ref:\t")
            ##
            land_dcm = (stb0.zmp + offset) + math.exp(self.time_to_landing / T) * (centroid.dcm_ref - (stb0.zmp + offset))
            ##
            if self.debug > 3:
                print(f'T:\t{T:.6f}')
                print(f'time_to_landing:\t{self.time_to_landing:.6f}')
                printVec3(land_dcm, "land_dcm:\t")
            # 着地調整（DCM ベース） - ここで st1 が本来の footstep.steps[1] を正しく参照するようになります
            stb1.foot_pos[swg][0] = land_dcm[0] - (st1.dcm[0] - st1.foot_pos[swg][0])
            stb1.foot_pos[swg][1] = land_dcm[1] - (st1.dcm[1] - st1.foot_pos[swg][1])
        else:
            # No 着地調整
            stb1.foot_pos[swg][0] = st1.foot_pos[swg][0]
            stb1.foot_pos[swg][1] = st1.foot_pos[swg][1]

        # ベース向きは足の向きの中点
        angle_diff = foot[1].angle_ref[2] - foot[0].angle_ref[2]
        while angle_diff > pi: angle_diff -= 2.0 * pi
        while angle_diff < -pi: angle_diff += 2.0 * pi
        base.angle_ref[2] = foot[0].angle_ref[2] + angle_diff / 2.0
        base.ori_ref = R.from_euler('xyz', base.angle_ref)

        # サポート足の位置を設定
        foot[sup].pos_ref   = stb0.foot_pos[sup].copy()
        foot[sup].angle_ref = stb0.foot_angle[sup].copy()
        foot[sup].ori_ref   = R.from_euler('xyz', foot[sup].angle_ref)
        foot[sup].contact_ref = True

        if self.debug > 2:
            print("before *E*")
            for idx, step in enumerate(footstep.steps):
                print(f'steps[{idx}]')
                printStep(step, 'footstep.steps[i].')
            for idx, step in enumerate(footstep_buffer.steps):
                print(f'bsteps[{idx}]')
                printStep(step, 'footstep_buffer.steps[i].')
            for idx, ft in enumerate(foot):
                print(f'foot[{idx}]')
                printFoot(ft, 'foot[i].')

        # スウィング足の位置を設定
        if not stb0.stepping or self.time_to_landing > (stb0.duration - self.dsp_duration):
            if self.debug > 1:
                print("enter *E-1*")
            foot[swg].pos_ref   = stb0.foot_pos[swg].copy()
            foot[swg].angle_ref = stb0.foot_angle[swg].copy()
            foot[swg].ori_ref   = stb0.foot_ori[swg]
            foot[swg].contact_ref = True
        else:
            if self.debug > 1:
                print("enter *E-2*")
            ts = (stb0.duration - self.dsp_duration) - self.time_to_landing
            tauv = stb0.duration - self.dsp_duration
            tauh = tauv - self.descend_duration

            sv = ts / tauv if tauv > 0 else 0
            sh = ts / tauh if tauh > 0 else 0
            thetav = 2.0 * pi * sv
            thetah = 2.0 * pi * sh

            ch = (thetah - np.sin(thetah)) / (2.0 * pi) if sh < 1.0 else 1.0
            cv = (1.0 - np.cos(thetav)) / 2.0
            cv2 = (1.0 - np.cos(thetav / 2.0)) / 2.0
            cw = np.sin(thetah)

            turn = stb1.foot_angle[swg] - stb0.foot_angle[swg]
            while turn[2] > pi: turn[2] -= 2.0 * pi
            while turn[2] < -pi: turn[2] += 2.0 * pi

            tilt = stb0.foot_ori[swg].apply(np.array([0.0, self.swing_tilt, 0.0]))

            foot[swg].pos_ref = (1.0 - ch) * stb0.foot_pos[swg] + ch * stb1.foot_pos[swg]
            foot[swg].pos_ref[2] += (cv * (self.swing_height + 0.5 * self.descend_depth) - cv2 * self.descend_depth)
            foot[swg].angle_ref = stb0.foot_angle[swg] + ch * turn + cw * tilt
            foot[swg].ori_ref = R.from_euler('xyz', foot[swg].angle_ref)
            foot[swg].contact_ref = False

            # 【バグ③の修正】SciPyの乗算規則（左が先、右が後）に合わせて C++ (Q_act.inv() が先、Q_ref が後) を表現
            qrel = R.from_euler('xyz', base.angle_ref) * R.from_euler('xyz', [base.angle[0], base.angle[1], base.angle_ref[2]]).inv()
            pivot = centroid.zmp_ref

            foot[swg].pos_ref = qrel.apply(foot[swg].pos_ref - pivot) + pivot
            foot[swg].ori_ref = qrel * foot[swg].ori_ref
            foot[swg].angle_ref = foot[swg].ori_ref.as_euler('xyz')
        if self.debug > 2:
            for idx, step in enumerate(footstep.steps):
                print(f'steps[{idx}]')
                printStep(step, 'footstep.steps[i].')
            for idx, step in enumerate(footstep_buffer.steps):
                print(f'bsteps[{idx}]')
                printStep(step, 'footstep_buffer.steps[i].')
            print(f'sup:\t{sup}')
            print(f'swg:\t{swg}')
            print(f'T:\t{T:.6f}')
            print(f'time_to_landing:\t{self.time_to_landing:.6f}')
            for idx, ft in enumerate(foot):
                print(f'foot[{idx}]')
                printFoot(ft, 'foot[i].')
            printVec3(centroid.dcm_ref,     "centroid.dcm_ref:\t")
            printVec3(centroid.dcm_target,  "centroid.dcm_target:\t")
            printVec3(centroid.zmp_ref,     "centroid.zmp_ref:\t")
            printVec3(centroid.zmp_target,  "centroid.zmp_target:\t")
            printVec3(centroid.com_pos_ref, "centroid.com_pos_ref:\t")
            printVec3(centroid.com_vel_ref, "centroid.com_vel_ref:\t")
            print("End Of Update")
        return True
