import numpy as np

from footstep_planner import FootstepPlanner, Step, Footstep, Param, Ground, printStep
from stepping_controller import SteppingController, Timer, Centroid, Base, Foot
# from hrp2_footplane import generate_5step_walking_plan
from stabilizer import Stabilizer

class WalkingControl:

    def __init__(self, dt=0.01, render=True):
        """
        初期化

        Args:
            dt: シミュレーション時間ステップ [s]
            render: ビューアーを表示するか
        """
        self.dt = dt
        # self.render = render
        # self.scene = None
        # self.robot = None
        # self.viewer = None
        # self.time = 0.0

        # ロボット固有のパラメータ
        #self.joint_names = []
        #self.dofs_idx_local = None
        #self.num_motors = 0
        #self.joint_indices = {}

        # 歩行計画
        self.footstep = None
        self.param = None
        self.current_step = 0
        self.step_progress = 0.0  # 0 ~ 1

        # SteppingController
        self.stepping_controller = None
        self.timer = None
        self.centroid = None
        self.base = None
        self.feet = None

        # 目標位置・姿勢
        self.target_joint_angles = None

        self.no_dcm_gain=True
        self.no_dcm_derivative=True
        self.use_cpp_stabilizer = False
        self.state_callback = None
        self._missing_contact_force_warned = False
#>        self._init_genesis()
#>
#>    def _init_genesis(self):
#>        """Genesis を初期化"""
#>        print("Initializing Genesis...")
#>        gs.init(backend=gs.cuda)
#>
#>        # Scene 作成オプション
#>        scene_kwargs = {
#>            "sim_options": gs.options.SimOptions(
#>                dt=self.dt,
#>                gravity=[0, 0, -9.81],
#>            ),
#>        }
#>
#>        if self.render:
#>            scene_kwargs["show_viewer"] = True
#>
#>        self.scene = gs.Scene(**scene_kwargs)
#>
#>        # 地面の追加
#>        self.ground = self.scene.add_entity(gs.morphs.Plane())
#>
#>        # HRP2 ロボットを追加
#>        robot_path = os.path.join(os.path.dirname(__file__), "hrp2_description/HRP2_genesis.urdf")
#>        print(f"Loading URDF from: {robot_path}")
#>
#>        try:
#>            self.robot = self.scene.add_entity(
#>                gs.morphs.URDF(file=robot_path,
#>                               pos=(0.0, 0.0, 0.71),  # ハーフシッティングに合わせた初期高さ
#>                               fixed=False
#>                               ),
#>            )
#>            print(f"✓ Successfully loaded HRP2 robot")
#>        except Exception as e:
#>            print(f"✗ Error loading URDF: {e}")
#>            raise
#>
#>        # scene.build() を実行
#>        self.scene.build()
#>
#>        # ロボット情報を取得
#>        self._setup_robot_info()
#>
#>        # PD制御パラメータを設定
#>        self._setup_pd_control()
#>
#>        # 初期姿勢を設定
#>        self._set_initial_pose()
#>
#>        print(f"✓ Genesis initialized with {self.num_motors} DOFs")
#>
#>        if self.render and hasattr(self.scene, 'viewer'):
#>            self.viewer = self.scene.viewer

#>    def _setup_robot_info(self):
#>        """ロボット情報をセットアップ（hrp2_train.py の並び順と完全同期）"""
#>        self.joint_names = [
#>            # 右脚 (6 DOF)
#>            "RLEG_JOINT0", "RLEG_JOINT1", "RLEG_JOINT2", "RLEG_JOINT3", "RLEG_JOINT4", "RLEG_JOINT5",
#>            # 左脚 (6 DOF)
#>            "LLEG_JOINT0", "LLEG_JOINT1", "LLEG_JOINT2", "LLEG_JOINT3", "LLEG_JOINT4", "LLEG_JOINT5",
#>            # 体幹 (2 DOF)
#>            "CHEST_JOINT0", "CHEST_JOINT1",
#>            # 頭部 (2 DOF)
#>            "HEAD_JOINT0", "HEAD_JOINT1",
#>            # 右腕 (7 DOF)
#>            "RARM_JOINT0", "RARM_JOINT1", "RARM_JOINT2", "RARM_JOINT3", "RARM_JOINT4", "RARM_JOINT5", "RARM_JOINT6",
#>            # 左腕 (7 DOF)
#>            "LARM_JOINT0", "LARM_JOINT1", "LARM_JOINT2", "LARM_JOINT3", "LARM_JOINT4", "LARM_JOINT5", "LARM_JOINT6",
#>        ]
#>
#>        self.dofs_idx_local = []
#>        self.joint_indices = {}
#>
#>        for joint_name in self.joint_names:
#>            try:
#>                dof_idx = self.robot.get_joint(joint_name).dof_idx_local
#>                self.dofs_idx_local.append(dof_idx)
#>                self.joint_indices[joint_name] = dof_idx
#>            except Exception as e:
#>                print(f"  Warning: Could not get joint {joint_name}: {e}")
#>
#>        self.dofs_idx_local = np.array(self.dofs_idx_local)
#>        self.num_motors = len(self.dofs_idx_local)
#>        print(f"✓ Found {self.num_motors} motor DOFs")

#>    def _setup_pd_control(self):
#>        """PD制御パラメータを設定（hrp2_train.py の強固なゲインに設定）"""
#>        try:
#>            if not self.robot or not hasattr(self, 'dofs_idx_local'):
#>                return
#>
#>            # 全身一律で高ゲインを適用
#>            kp_values = np.ones(self.num_motors, dtype=np.float32) * 2000.0
#>            kv_values = np.ones(self.num_motors, dtype=np.float32) * 100.0
#>
#>            self.robot.set_dofs_kp(kp_values, self.dofs_idx_local)
#>            self.robot.set_dofs_kv(kv_values, self.dofs_idx_local)
#>
#>            # トルクリミットの設定
#>            force_upper = np.ones(self.num_motors, dtype=np.float32) * 300.0
#>            force_lower = -force_upper
#>            self.robot.set_dofs_force_range(force_lower, force_upper, self.dofs_idx_local)
#>
#>            print(f"✓ PD control configured (kp=2000.0, kd=100.0)")
#>        except Exception as e:
#>            print(f"Warning: Could not setup PD control: {e}")

#>    def _set_initial_pose(self):
#>        """初期姿勢を設定（hrp2_train.py のハーフシッティング姿勢）"""
#>        try:
#>            if not self.robot or not hasattr(self, 'dofs_idx_local'):
#>                return
#>
#>            # 股ピッチ=-0.4, 膝=0.8, 足首ピッチ=-0.4
#>            initial_angles = np.array([
#>                # 右脚: 姿勢
#>                0.0, 0.0, -0.4, 0.8, -0.4, 0.0,
#>                # 左脚: 姿勢
#>                0.0, 0.0, -0.4, 0.8, -0.4, 0.0,
#>                # 腰・頭・両腕はニュートラル
#>                0.0, 0.0,
#>                0.0, 0.0,
#>                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#>                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#>            ], dtype=np.float32)
#>
#>            print("Setting initial pose (Half-Sitting)...")
#>            for step in range(100):
#>                self.robot.set_dofs_position(initial_angles, self.dofs_idx_local)
#>                self.scene.step()
#>
#>            for settle_step in range(200):
#>                self.robot.control_dofs_position(initial_angles, self.dofs_idx_local)
#>                self.scene.step()
#>
#>            print("✓ Initial pose set and settled")
#>        except Exception as e:
#>            print(f"Warning: Could not set initial pose: {e}")

    def _setup_stepping_controller(self, spacing=0.1):
        """SteppingController を初期化"""
        self.stepping_controller = SteppingController()
        ##// init stepping controller
        self.stepping_controller.swing_height = 0.05;
        self.stepping_controller.swing_tilt   = 0.0;
        self.stepping_controller.dsp_duration = 0.05;
        self.stepping_controller.timing_adaptation_weight = 0.1;

        self.timer    = Timer()
        self.timer.dt = self.dt
        self.centroid = Centroid()
        #self.centroid.com_pos     = np.array([0., 0., self.param.com_height])
        self.centroid.com_pos_ref = np.array([0., 0., self.param.com_height])
        self.centroid.dcm_ref     = np.array([0., 0., self.param.com_height])
        self.centroid.dcm_target  = np.array([0., 0., self.param.com_height]) ## not equal to original
        self.base     = Base()
        self.feet     = [Foot(), Foot()]  # [左足, 右足]

        # 初期位置を設定（ワールド座標系における足底の初期位置）
        self.feet[0].pos_ref = np.array([0.0, -spacing, 0.0])  # 右足 (spacing 0.2)
        self.feet[1].pos_ref = np.array([0.0,  spacing, 0.0])  # 左足

        ## 重要: コントローラ内部の仕様に合わせて2ステップ先まで空のインスタンスを予約
        #self.footstep_buffer = Footstep(steps=[Step(), Step()])

        self.current_step  = 0
        self.step_progress = 0.0

        self.stabilizer = Stabilizer()

        print("SteppingController initialized")

    def setup_controller(self, param=None, stride=0.1, spacing=0.2, duration=0.5, stepSize=4):
        self.param = param if param is not None else Param()

        self.footstep_planner = FootstepPlanner()
        self._setup_stepping_controller()

        ## vnoid/controller/sample_controller/myrobot.cpp // init footsteps
        #>footstep.steps.push_back(Step(0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0));
        #>footstep.steps.push_back(Step(0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 1));
        #>// foot placement and DCM of the initial step must be specified
        #>footstep.steps[0].foot_pos[0] = foot[0].pos_ref;
        #>footstep.steps[0].foot_pos[1] = foot[1].pos_ref;
        #>footstep.steps[0].dcm = centroid.dcm_ref;
        #>footstep_planner.Plan(param, footstep);
        #>footstep_planner.GenerateDCM(param, footstep);
        #>
        #>footstep_buffer.steps.push_back(footstep.steps[0]);
        #>footstep_buffer.steps.push_back(footstep.steps[1]);
        lst = []
        lst.append( Step(stride=0.0, sway=0.0, spacing=spacing, turn=0.0, climb=0.0, duration=duration, side=0) ) #0
        lst.append( Step(stride=0.0, sway=0.0, spacing=spacing, turn=0.0, climb=0.0, duration=duration, side=1) ) #1
        lst[0].foot_pos[0] = self.feet[0].pos_ref;
        lst[0].foot_pos[1] = self.feet[1].pos_ref;
        lst[0].dcm = self.centroid.dcm_ref;
        self.footstep = Footstep(steps=lst)
        print('00 footstep(pre)')
        for idx, step in enumerate(self.footstep.steps):
            print(f'steps[{idx}]')
            printStep(step, 'footstep.steps[i].')
        self.footstep_planner.plan(self.param, self.footstep)
        print('00 footstep(after Plan)')
        for idx, step in enumerate(self.footstep.steps):
            print(f'steps[{idx}]')
            printStep(step, 'footstep.steps[i].')
        self.footstep_planner.generate_dcm(self.param, self.footstep)
        print('00 footstep(after GenDCM)')
        for idx, step in enumerate(self.footstep.steps):
            print(f'steps[{idx}]')
            printStep(step, 'footstep.steps[i].')
        print("")

        self.footstep_buffer = Footstep(steps=[self.footstep.steps[0].copy(), self.footstep.steps[1].copy()])

        ## vnoid/controller/sample_controller/myrobot.cpp // inside Control
        #>Step step;
        #>step.stride   = 0.1; //-max_stride*joystick.getPosition(Joystick::L_STICK_V_AXIS);
        #>step.turn     = 0.0; //-max_turn  *joystick.getPosition(Joystick::L_STICK_H_AXIS);
        #>step.spacing  = 0.20;
        #>step.climb    = 0.0;
        #>step.duration = 0.5;
        #>footstep.steps.push_back(step);
        #>footstep.steps.push_back(step);
        #>footstep.steps.push_back(step);
        #>step.stride = 0.0;
        #>step.turn   = 0.0;
        #>footstep.steps.push_back(step);
        for i in range(stepSize):
            self.footstep.steps.append( Step(stride=stride, sway=0.0, spacing=spacing, turn=0.0, climb=0.0, duration=duration, side=0) )
        self.footstep.steps.append( Step(stride=0.0, sway=0.0, spacing=spacing, turn=0.0, climb=0.0, duration=duration, side=0) ) ## finishing
        self.footstep.steps.append( Step(stride=0.0, sway=0.0, spacing=spacing, turn=0.0, climb=0.0, duration=duration, side=0) ) ## finishing

        print('footstep(pre)')
        for idx, step in enumerate(self.footstep.steps):
            print(f'steps[{idx}]')
            printStep(step, 'footstep.steps[i].')

        self.footstep_planner.plan(self.param, self.footstep);

        print('footstep(after Plan)')
        for idx, step in enumerate(self.footstep.steps):
            print(f'steps[{idx}]')
            printStep(step, 'footstep.steps[i].')

        self.footstep_planner.generate_dcm(self.param, self.footstep);

        print('footstep(after GenDCM)')
        for idx, step in enumerate(self.footstep.steps):
            print(f'steps[{idx}]')
            printStep(step, 'footstep.steps[i].')
        print("")

    def set_state_callback(self, callback):
        self.state_callback = callback

    def _apply_external_state(self):
        if self.state_callback is None:
            return False
        state = self.state_callback()
        if state is None:
            return False

        if 'base_angle' in state:
            self.base.angle = np.asarray(state['base_angle'], dtype=float)
        if 'base_angvel' in state:
            self.base.angvel = np.asarray(state['base_angvel'], dtype=float)
        if 'com_pos' in state:
            com_pos = np.asarray(state['com_pos'], dtype=float)
            self.centroid.com_vel = (com_pos - self.centroid.com_pos) / self.timer.dt
            self.centroid.com_pos = com_pos
            self.centroid.dcm = self.centroid.com_pos + self.param.T * self.centroid.com_vel
        if 'com_vel' in state:
            self.centroid.com_vel = np.asarray(state['com_vel'], dtype=float)
            self.centroid.dcm = self.centroid.com_pos + self.param.T * self.centroid.com_vel
        if 'foot_force' in state:
            for i, force in enumerate(state['foot_force'][:2]):
                self.feet[i].force = np.asarray(force, dtype=float)
        if 'foot_moment' in state:
            for i, moment in enumerate(state['foot_moment'][:2]):
                self.feet[i].moment = np.asarray(moment, dtype=float)
        return True

    def _has_required_contact_feedback(self):
        has_contact_ref = False
        for foot in self.feet:
            if not foot.contact_ref:
                continue
            has_contact_ref = True
            if foot.force is None:
                return False
            if np.linalg.norm(foot.force) <= 1.0e-12:
                return False
        return has_contact_ref
        

    def _warn_missing_contact_forces(self):
        if self._missing_contact_force_warned:
            return
        print("Warning: no foot force feedback. Check Choreonoid force sensor connection.")
        self._missing_contact_force_warned = True

    def step_simulation(self):
        """シミュレーションを1ステップ進める"""
        try:
            if hasattr(self, 'stepping_controller') and self.stepping_controller:
                ##self.timer.time = self.time

                if self.use_cpp_stabilizer:
                    self._apply_external_state()
                    if not self._has_required_contact_feedback():
                        self._warn_missing_contact_forces()

                # 軌道更新
                stepping = self.stepping_controller.update(
                    self.timer,
                    self.param,
                    self.footstep,
                    self.footstep_buffer,
                    self.centroid,
                    self.base,
                    self.feet
                )

                # 逆運動学の計算と指令
                #> self._update_joint_targets_from_feet()

                # 決定論的（オープンループ）に歩行させるため、前回の出力を今回の参照としてフィードバック
                #self.centroid.dcm_ref = self.centroid.dcm_target.copy()
                #self.centroid.zmp_ref = self.centroid.zmp_target.copy()
                if self.use_cpp_stabilizer:
                    self.stabilizer.Update(self.timer, self.param, self.centroid, self.base, self.feet)
                else:
                    self.stabilizer.CalcDcmDynamicsSimple(self.timer, self.param, self.centroid,
                                                          self.no_dcm_gain, self.no_dcm_derivative)

        except Exception as e:
            print(f"Warning in stepping controller: {e}")

        #> self.scene.step()
        #self.time += self.dt
        self.timer.CountUp()

#>    def _update_joint_targets_from_feet(self):
#>        """足の目標位置・姿勢から逆運動学(IK)を計算して関節角を一括制御"""
#>        try:
#>            if not self.robot or not hasattr(self, 'dofs_idx_local'):
#>                return
#>
#>            # 各足のワールド座標目標
#>            left_pos = self.feet[0].pos_ref
#>            right_pos = self.feet[1].pos_ref
#>
#>            # DCMベースの目標重心位置から腰の推定目標位置を算出
#>            base_pos_ref = self.centroid.dcm_target - np.array([0.0, 0.0, self.param.com_height])
#>
#>            # 腰から見た各足の相対位置
#>            left_rel_pos = left_pos - base_pos_ref
#>            right_rel_pos = right_pos - base_pos_ref
#>
#>            # ----------------- 左脚の幾何IK -----------------
#>            left_H = -left_rel_pos[2]
#>            # 初期高さ 0.71m のときに膝がちょうど 0.8rad になるリンク幾何モデル
#>            left_knee = 2.0 * np.arccos(np.clip(left_H / 0.77, -1.0, 1.0))
#>
#>            # 姿勢維持（平行リンク近似: 股ピッチと足首ピッチは膝の半分ずつ受け持つ）
#>            left_hip_pitch = -0.5 * left_knee
#>            left_ankle_pitch = -0.5 * left_knee
#>
#>            # X方向（前後移動）によるピッチ補正
#>            left_hip_pitch += -left_rel_pos[0] / 0.71
#>            left_ankle_pitch += -left_rel_pos[0] / 0.71
#>
#>            # Y方向（左右揺れ）によるロール補正 (デフォルトの足幅 0.1m からの変位)
#>            left_hip_roll = (left_rel_pos[1] - 0.1) / 0.71
#>            left_ankle_roll = -(left_rel_pos[1] - 0.1) / 0.71
#>            left_hip_yaw = self.feet[0].angle_ref[2]
#>
#>            # ----------------- 右脚の幾何IK -----------------
#>            right_H = -right_rel_pos[2]
#>            right_knee = 2.0 * np.arccos(np.clip(right_H / 0.77, -1.0, 1.0))
#>
#>            right_hip_pitch = -0.5 * right_knee
#>            right_ankle_pitch = -0.5 * right_knee
#>
#>            right_hip_pitch += -right_rel_pos[0] / 0.71
#>            right_ankle_pitch += -right_rel_pos[0] / 0.71
#>
#>            right_hip_roll = (right_rel_pos[1] - (-0.1)) / 0.71
#>            right_ankle_roll = -(right_rel_pos[1] - (-0.1)) / 0.71
#>            right_hip_yaw = self.feet[1].angle_ref[2]
#>
#>            # -----------------------------------------------
#>            # 関節角度配列の構築 (hrp2_train.py の joint_names の順序と完全同期)
#>            target_angles = np.array([
#>                # 右脚 (6 DOF)
#>                right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll,
#>                # 左脚 (6 DOF)
#>                left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll,
#>                # 腰 (2 DOF)
#>                0.0, 0.0,
#>                # 頭 (2 DOF)
#>                0.0, 0.0,
#>                # 右腕 (7 DOF)
#>                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#>                # 左腕 (7 DOF)
#>                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#>            ], dtype=np.float32)
#>
#>            self.set_all_joint_targets(target_angles)
#>
#>        except Exception as e:
#>            print(f"IK Error: {e}")
#>
#>    def set_all_joint_targets(self, angles):
#>        """すべての制御対象関節の目標角度を一括設定"""
#>        try:
#>            if self.robot and len(angles) == self.num_motors:
#>                angles = np.asarray(angles, dtype=np.float32)
#>                self.robot.control_dofs_position(angles, self.dofs_idx_local)
#>        except Exception as e:
#>            print(f"Warning: Could not set joint targets: {e}")

#>    def render_step(self):
#>        """画面を描画"""
#>        try:
#>            if self.render and self.viewer:
#>                self.viewer.render()
#>        except:
#>            pass
#>
#>    def run(self, duration=10.0):
#>        """シミュレーションを実行"""
#>        print(f"\nRunning simulation for {duration} seconds...")
#>
#>        num_steps = int(duration / self.dt)
#>        for i in range(num_steps):
#>            self.step_simulation()
#>            self.render_step()
#>
#>            if i % 100 == 0:
#>                if self.feet:
#>                    print(f"Time: {self.time:.2f}s | Left Foot Z: {self.feet[0].pos_ref[2]:.3f} | Right Foot Z: {self.feet[1].pos_ref[2]:.3f}")
#>
#>            if self.time > duration:
#>                break
#>
#>        print(f"\n✓ Simulation completed! Total time: {self.time:.2f} s")