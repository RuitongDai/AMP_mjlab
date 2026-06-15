import time
import numpy as np
import onnxruntime
import mujoco
import mujoco_viewer

# 保证 28 自由度顺序与 XML 和 ONNX 输出严格一致
mujoco_joint_index = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]

# 【默认初始姿态】与 x3_constants.py 中的 HOME_KEY_FRAME 完全一致
DEFAULT_ANGLES = np.array([
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # 左腿
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # 右腿
    0.0, 0.0,  # 腰部 (roll, yaw)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 左臂
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 右臂
], dtype=np.float32)

# 【动作缩放比例】与 x3_constants.py 中的 X3_ACTION_SCALE 完全一致
ACTION_SCALES = np.array([
    0.1875, 0.1875, 0.1875, 0.1875, 0.625, 0.625,  # 左腿
    0.1875, 0.1875, 0.1875, 0.1875, 0.625, 0.625,  # 右腿
    0.125, 0.125,  # 腰部
    0.2083, 0.2083, 0.2083, 0.2083, 0.2083, 0.0625, 0.0625,  # 左臂
    0.2083, 0.2083, 0.2083, 0.2083, 0.2083, 0.0625, 0.0625  # 右臂
], dtype=np.float32)

# ==========================================================
# 【PD 控制器参数】
# ==========================================================
KP = np.array([
    100.0, 100.0, 100.0, 100.0, 30.0, 30.0,  # 左腿 (Hip to Knee: 100, Ankle: 30)
    100.0, 100.0, 100.0, 100.0, 30.0, 30.0,  # 右腿
    100.0, 100.0,  # 腰部 (Waist: 100)
    30.0, 30.0, 30.0, 30.0, 30.0, 20.0, 20.0,  # 左臂 (Shoulder/Forearm: 30, Hand: 20)
    30.0, 30.0, 30.0, 30.0, 30.0, 20.0, 20.0  # 右臂
], dtype=np.float32)

KD = np.array([2.0] * 28, dtype=np.float32)  # 所有的 damping 都是 2.0


def get_projected_gravity(quat):
    """计算投影重力（世界 Z 轴向下的向量在局部坐标系中的表示）"""
    w, x, y, z = quat
    proj_g = np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y))
    ], dtype=np.float32)
    return proj_g


class Sim2Sim():
    def __init__(self, xml_path, policy_path):
        self.xml_path = xml_path
        self.policy_path = policy_path

        self.num_actions = 28

        # 物理引擎时间步 (200Hz)，这也是 PD 控制器的运行频率
        self.simulation_dt = 0.005

        # 策略网络时间步抽取 (4 * 0.005 = 0.02s，即 50Hz)
        self.control_decimation = 4

        self.target_q = np.zeros(self.num_actions, dtype=np.double)
        self.last_actions = np.zeros(self.num_actions, dtype=np.float32)

        # 历史缓冲：93维 * 4帧 = 372维
        self.obs_history = np.zeros(372, dtype=np.float32)

        # 速度指令：[X线速度, Y线速度, Z角速度]
        self.command = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    def run(self):
        # 1. 加载 ONNX 模型
        session = onnxruntime.InferenceSession(self.policy_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name

        # 2. 初始化 Mujoco 环境
        m = mujoco.MjModel.from_xml_path(self.xml_path)
        d = mujoco.MjData(m)
        m.opt.timestep = self.simulation_dt

        # 获取传感器 ID
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        sensor_gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")

        # 初始姿态赋值并设为目标角度
        d.qpos[7:7 + self.num_actions] = DEFAULT_ANGLES
        self.target_q = DEFAULT_ANGLES.copy()
        mujoco.mj_step(m, d)

        viewer = mujoco_viewer.MujocoViewer(m, d, width=1600, height=900)
        viewer.cam.distance = 3.0

        counter = 0

        # 4. 控制主循环
        while viewer.is_alive:
            step_start = time.time()

            # ==================================================
            # 【策略网络层】频率: 50Hz (每 4 个物理步执行一次)
            # ==================================================
            if counter % self.control_decimation == 0:
                # 读取状态
                robot_quat = d.xquat[body_id].copy()
                dof_pos = d.qpos[7:7 + self.num_actions].copy()
                dof_vel = d.qvel[6:6 + self.num_actions].copy()

                # 读取陀螺仪
                adr_gyro = m.sensor_adr[sensor_gyro_id]
                gyro = d.sensordata[adr_gyro: adr_gyro + 3].copy()

                # 拼装 93 维观测
                obs_t = np.concatenate([
                    gyro,  # 3: 基座角速度
                    get_projected_gravity(robot_quat),  # 3: 投影重力
                    self.command,  # 3: 目标速度指令
                    dof_pos - DEFAULT_ANGLES,  # 28: 相对关节角度
                    dof_vel,  # 28: 关节速度
                    self.last_actions  # 28: 上一帧网络输出的原始动作
                ]).astype(np.float32)

                # 滑动更新历史 Buffer
                self.obs_history[:-93] = self.obs_history[93:]
                self.obs_history[-93:] = obs_t

                # 推理动作 [-1, 1]
                ort_inputs = {input_name: self.obs_history.reshape(1, -1)}
                action_raw = session.run(None, ort_inputs)[0].flatten()

                self.last_actions = action_raw.copy()

                # 解算出物理世界的绝对目标角度
                self.target_q = DEFAULT_ANGLES + (action_raw * ACTION_SCALES)

            # ==================================================
            # 【底层控制层】频率: 200Hz (每次物理循环都执行)
            # ==================================================
            # 实时获取当前物理角度和速度
            current_q = d.qpos[7:7 + self.num_actions].copy()
            current_dq = d.qvel[6:6 + self.num_actions].copy()

            # 【核心公式】：手动计算 PD 扭矩
            # 目标速度默认我们设为 0，所以阻尼项直接是 - KD * current_dq
            torques = KP * (self.target_q - current_q) - KD * current_dq

            # 将计算出的力矩直接打给 XML 里定义的 <motor>
            d.ctrl[:] = torques

            # 步进物理引擎
            mujoco.mj_step(m, d)
            counter += 1

            # 画面渲染与时间同步
            viewer.render()
            time_until_next_step = self.simulation_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == '__main__':
    XML_PATH = "/home/dai/github/AMP_mjlab/src/assets/robots/moya/xmls/Moya01_V2.xml"
    POLICY_PATH = "/home/dai/x3_amp_locomotion/2026-06-13_10-55-45/policy.onnx"

    bot = Sim2Sim(XML_PATH, POLICY_PATH)
    bot.run()