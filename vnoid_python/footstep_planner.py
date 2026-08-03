"""
Footstep Planner - C++ の footstep_planner.cpp を Python に変換
歩行ステップの計画と生成
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List
from scipy.spatial.transform import Rotation as R
## TODO  R => coordinates

eps = 1.0e-10

@dataclass
class Step:
    """1ステップの情報"""
    # 足の位置と姿勢（左足[0]、右足[1]）
    foot_pos: np.ndarray = None      # (2, 3) - 足の位置 [left, right]
    foot_angle: np.ndarray = None    # (2, 3) - roll, pitch, yaw [left, right]
    foot_ori: list = None            # (2,) - 回転行列 [left, right]

    # 足の側（0: 左, 1: 右）
    side: int = 0

    # ステップパラメータ
    stride: float  = 0.2      # 前進距離
    sway: float    = 0.0      # 横揺れ
    turn: float    = 0.0      # 回転角度
    spacing: float = 0.2      # 足の間隔
    climb: float   = 0.0      # 階段登り

    # タイミング情報
    duration: float = 0.8            # ステップの継続時間
    tbegin: float = 0.0              # ステップ開始時刻

    # DCM と ZMP
    dcm: np.ndarray = None           # Divergent Component of Motion
    zmp: np.ndarray = None           # Zero Moment Point

    # その他
    stepping: bool = True            # ステップするか（サポート交換）

    def __post_init__(self):
        if self.foot_pos is None:
            self.foot_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # (2, 3)
        if self.foot_angle is None:
            self.foot_angle = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # (2, 3)
        if self.foot_ori is None:
            self.foot_ori = [R.from_euler('xyz', [0, 0, 0]), R.from_euler('xyz', [0, 0, 0])]
        if self.dcm is None:
            self.dcm = np.array([0.0, 0.0, 0.0])
        if self.zmp is None:
            self.zmp = np.array([0.0, 0.0, 0.0])

    def copy(self):
        return Step(side=self.side, stride=self.stride, sway=self.sway, turn=self.turn, spacing=self.spacing,
                    climb=self.climb, duration=self.duration, tbegin=self.tbegin, stepping=self.stepping,
                    foot_pos=np.array(self.foot_pos), foot_angle=np.array(self.foot_angle),
                    foot_ori=[R.from_quat(ori.as_quat()) for ori in self.foot_ori],
                    dcm=np.array(self.dcm), zmp=np.array(self.zmp) )

def fmtVec3(vec3):
    return f'({vec3[0]:.6f}, {vec3[1]:.6f}, {vec3[2]:.6f} )'

def printVec3(vec3, prefix=''):
    print(prefix + fmtVec3(vec3))

def printStep(step, prefix='step.'):
    print(f'{prefix}stride:\t{step.stride:.6f}')
    print(f'{prefix}sway:\t{step.sway:.6f}')
    print(f'{prefix}spacing:\t{step.spacing:.6f}')
    print(f'{prefix}turn:\t{step.turn:.6f}')
    print(f'{prefix}climb:\t{step.climb:.6f}')
    print(f'{prefix}duration:\t{step.duration:.6f}')
    print(f'{prefix}side:\t{step.side}')
    #print(f'{prefix}stepping:\t{step.stepping}')
    if step.stepping:
        print(f'{prefix}stepping:\t{1}')
    else:
        print(f'{prefix}stepping:\t{0}')
    print(f'{prefix}tbegin:\t{step.tbegin:.6f}')
    print(f'{prefix}foot_pos[0]:\t' + fmtVec3(step.foot_pos[0]))
    print(f'{prefix}foot_pos[1]:\t' + fmtVec3(step.foot_pos[1]))
    print(f'{prefix}foot_angle[0]:\t' + fmtVec3(step.foot_angle[0]))
    print(f'{prefix}foot_angle[1]:\t' + fmtVec3(step.foot_angle[1]))
    print(f'{prefix}zmp:\t'+ fmtVec3(step.zmp))
    print(f'{prefix}dcm:\t'+ fmtVec3(step.dcm))

## TODO using coordinates
@dataclass
class StepCoords:
    """1ステップの情報"""
    # 足の位置と姿勢（左足[0]、右足[1]）
    #foot_pos: np.ndarray = None      # (2, 3) - 足の位置 [left, right]
    #foot_angle: np.ndarray = None    # (2, 3) - roll, pitch, yaw [left, right]
    #foot_ori: list = None            # (2,) - 回転行列 [left, right]
    foot_coords = None # (coordinates, coordinates)
    # 足の側（0: 左, 1: 右）
    side: int = 0
    # ステップパラメータ
    stride: float = 0.2              # 前進距離
    sway: float = 0.0                # 横揺れ
    turn: float = 0.0                # 回転角度
    spacing: float = 0.2             # 足の間隔
    climb: float = 0.0               # 階段登り
    # タイミング情報
    duration: float = 0.8            # ステップの継続時間
    tbegin: float = 0.0              # ステップ開始時刻
    # DCM と ZMP
    dcm: np.ndarray = None           # Divergent Component of Motion
    zmp: np.ndarray = None           # Zero Moment Point
    # その他
    stepping: bool = True            # ステップするか（サポート交換）

@dataclass
class Footstep:
    """複数ステップの歩行計画"""
    steps: List[Step] = None
    def __post_init__(self):
        if self.steps is None:
            self.steps = []

@dataclass
class Param:
    """歩行パラメータ (robot_base.h::Param の移植。移動制御に必要なフィールドのみ)"""
    total_mass: float = 50.0
    nominal_inertia: np.ndarray = field(
        default_factory=lambda: np.array([20.0, 20.0, 5.0]))
    com_height: float = 1.0  # CoM の高さ (C++デフォルトは1.0。歩行時は呼び出し側で上書きする)
    gravity: float = 9.8
    T: float = 1.0           # 時定数 T = sqrt(com_height/gravity)
    trunk_mass: float = 1.0
    trunk_com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    zmp_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    zmp_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.Init()

    def Init(self):
        """Param::Init() の移植"""
        self.T = math.sqrt(self.com_height / self.gravity)

@dataclass
class Ground:
    """地面情報"""
    ori: R = None                    # 地面の姿勢

    def __post_init__(self):
        if self.ori is None:
            self.ori = R.from_euler('xyz', [0, 0, 0])


def quat_to_rpy(quat):
    """クォータニオンから RPY を取得"""
    r = R.from_quat(quat)
    rpy = r.as_euler('xyz')
    return rpy


def rpy_to_quat(rpy):
    """RPY からクォータニオンを生成"""
    r = R.from_euler('xyz', rpy)
    return r.as_quat()


def rotate_vector(ori, vec):
    """ベクトルを回転行列で回転"""
    return ori.apply(vec)


class FootstepPlanner:
    """歩行ステップの計画"""

    def __init__(self):
        pass

    def plan(self, param: Param, footstep: Footstep):
        """
        足の配置とサポート足フラグを決定

        Args:
            param: 歩行パラメータ
            footstep: 歩行計画（step[0]の足の配置とDCMは外部で指定される）
        """
        nstep = len(footstep.steps)

        for i in range(nstep - 1):
            st0 = footstep.steps[i]
            st1 = footstep.steps[i + 1]

            sup = st0.side          # サポート足
            swg = 1 - st0.side      # スウィング足

            dtheta = st0.turn
            l = st0.stride
            d = st0.sway
            w = (1.0 if sup == 0 else -1.0) * st0.spacing
            dz = st0.climb

            # スウィング足の相対位置を計算
            if abs(dtheta) < eps:
                dprel = np.array([l, w + d, dz])
            else:
                r = l / dtheta
                dprel = np.array([
                    (r - w / 2.0 - d) * np.sin(dtheta),
                    (r + w / 2.0) - (r - w / 2.0 - d) * np.cos(dtheta),
                    dz
                ])

            # サポート足と スウィング足を交換
            st1.side = 1 - st0.side

            # サポート足の位置は変わらない
            st1.foot_pos[sup] = st0.foot_pos[sup].copy()
            st1.foot_angle[sup] = st0.foot_angle[sup].copy()
            st1.foot_ori[sup] = st0.foot_ori[sup]

            # スウィング足の位置が変わる
            st1.foot_pos[swg] = st0.foot_pos[sup] + rotate_vector(st0.foot_ori[sup], dprel)
            st1.foot_angle[swg] = st0.foot_angle[sup] + np.array([0.0, 0.0, dtheta])
            st1.foot_ori[swg] = R.from_euler('xyz', st1.foot_angle[swg])

    def align_to_ground(self, ground: Ground, footstep: Footstep):
        """
        地面に足を合わせる

        Args:
            ground: 地面情報
            footstep: 歩行計画
        """
        # 初期サポート足の中心を基準に回転
        pivot = footstep.steps[0].foot_pos[footstep.steps[0].side]

        # 地面の法線ベクトル
        normal = rotate_vector(ground.ori, np.array([0.0, 0.0, 1.0]))

        for k in range(len(footstep.steps)):
            st = footstep.steps[k]

            for i in range(2):
                # Z座標を修正
                dp = st.foot_pos[i] - pivot
                if abs(normal[2]) > eps:
                    dp[2] = -(normal[0] * dp[0] + normal[1] * dp[1]) / normal[2]

                st.foot_pos[i][2] = pivot[2] + dp[2]

                # 地面法線を足の yaw ローカル座標に変換
                yaw = st.foot_angle[i][2]
                rot_z_neg = R.from_euler('z', -yaw)
                nl = rotate_vector(rot_z_neg, normal)

                st.foot_angle[i][0] = np.arcsin(-nl[1])
                st.foot_angle[i][1] = np.arctan2(nl[0], nl[2])

                # クォータニオンに変換
                st.foot_ori[i] = R.from_euler('xyz', st.foot_angle[i])

    def generate_dcm(self, param: Param, footstep: Footstep):
        """
        参考 DCM と ZMP を生成

        Args:
            param: 歩行パラメータ
            footstep: 歩行計画
        """
        nstep = len(footstep.steps)
        offset = np.array([0.0, 0.0, param.com_height])

        # 最後のステップの状態を設定
        i = nstep - 1
        # ZMP は足の中点
        footstep.steps[i].zmp = (footstep.steps[i].foot_pos[0] + footstep.steps[i].foot_pos[1]) / 2.0

        # DCM は ZMP から com_height 上
        footstep.steps[i].dcm = (footstep.steps[i].foot_pos[0] + footstep.steps[i].foot_pos[1]) / 2.0 + offset

        i -= 1

        # N-1 から 0 ステップの状態を計算
        while i >= 0:
            st0 = footstep.steps[i]
            st1 = footstep.steps[i + 1]

            sup = st0.side
            swg = 1 - st0.side

            a = np.exp(-st0.duration / param.T)

            # 初期ステップ: DCM は外部で指定済み、ZMP を決定
            if i == 0:
                st0.zmp = (st0.dcm - a * st1.dcm) / (1.0 - a) - offset
            else:
                # その他のステップ
                eps_local = 1.0e-3

                # スウィング足の位置が変わらない場合はダブルサポート
                if (np.linalg.norm(st0.foot_pos[swg] - st1.foot_pos[swg]) < eps_local and
                    np.linalg.norm(st0.foot_angle[swg] - st1.foot_angle[swg]) < eps_local):
                    st0.zmp = (st0.foot_pos[sup] + st0.foot_pos[swg]) / 2.0
                else:
                    # 그렇지 않으면 ZMP をサポート足に設定
                    st0.zmp = st0.foot_pos[sup].copy()

                # DCM を ZMP から決定
                st0.dcm = (1.0 - a) * (st0.zmp + offset) + a * st1.dcm

            # ステップフラグを設定
            eps_local = 1.0e-3
            if (np.linalg.norm(st0.foot_pos[swg] - st1.foot_pos[swg]) < eps_local and
                np.linalg.norm(st0.foot_angle[swg] - st1.foot_angle[swg]) < eps_local):
                st0.stepping = False
            else:
                st0.stepping = True

            i -= 1
