from __future__ import annotations
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING


class MotionLoader:
    """
    AMP 运动数据加载器。
    负责将离线的 .npz 动作数据集读取到内存/显存中，并在训练期间为 RL 环境和 AMP 判别器提供数据查询和随机采样。
    """

    def __init__(
            self,
            motion_dir: str,  # 正常动作数据的文件夹路径 (如 WalkandRun)
            tgt_body_indexes: Sequence[int],  # 需要重点追踪的目标连杆的索引 (对应 env_cfgs 里的 body_names)
            tgt_anchor_indexes: int,  # 锚点连杆的索引 (通常是躯干 torso_link，用于计算局部坐标系)
            feet_indexes: int,  # 脚部连杆的索引
            device: str = "cpu",  # 数据加载到的设备 (通常是 cuda:0)
            recovery_dir: str | None = None,  # 恢复动作(跌倒爬起)数据的文件夹路径
    ):
        # 存储所有运动数据的列表，每个元素是一个字典，代表一段完整的动作
        self.motion_data: list[dict] = []
        # 存储所有恢复数据的列表
        self.motion_data_recovery: list[dict] = []

        # 1. 加载正常运动数据 (走、跑等)
        self.motion_data = self._load_dir(motion_dir, device)
        assert len(self.motion_data) > 0, f"No npz files found in: {motion_dir}"

        # 2. 加载恢复运动数据 (如果有的话)
        # 这就是为什么你之前把它设为 None 或者指向 motion_dir 后，代码依然能正常跑的原因
        if recovery_dir is not None and os.path.isdir(recovery_dir):
            self.motion_data_recovery = self._load_dir(recovery_dir, device)

        # 提取所有动作的名称，方便调试
        self.motion_names = [m["motion_name"] for m in self.motion_data + self.motion_data_recovery]

        if not self.motion_data and not self.motion_data_recovery:
            raise ValueError(f"No motion data loaded from: {motion_dir}")

        # 取第一个动作作为“默认动作”，提取基础属性
        default_motion = self.motion_data[0] if self.motion_data else self.motion_data_recovery[0]

        # 记录默认动作的全局属性
        self.fps = default_motion["fps"]
        self._dof_pos = default_motion["dof_pos"]
        self._dof_vel = default_motion["dof_vel"]
        self._body_pos_w = default_motion["body_pos_w"]
        self._body_quat_w = default_motion["body_quat_w"]
        self._body_lin_vel_w = default_motion["body_lin_vel_w"]
        self._body_ang_vel_w = default_motion["body_ang_vel_w"]

        # 记录关键部位的索引，供外部快速查询
        self._body_indexes = tgt_body_indexes
        self._anchor_indexes = tgt_anchor_indexes
        self._feet_indexes = feet_indexes
        self.time_step_total = self._dof_pos.shape[0]
        self.motion_total_time = self.time_step_total / self.fps

    @staticmethod
    def _load_dir(dir_path: str, device: str) -> list[dict]:
        """
        核心数据加载函数：从指定目录遍历读取所有 .npz 文件。
        并将 Numpy 数组转换为 PyTorch Tensor，直接推送到 GPU 上，加速后续训练。
        """
        assert os.path.isdir(dir_path), f"Not a directory: {dir_path}"
        result = []
        for filename in sorted(os.listdir(dir_path)):
            if not filename.endswith(".npz"):
                continue
            motion_name = os.path.splitext(filename)[0]
            data = np.load(os.path.join(dir_path, filename))

            # 将所有观测数据转换为 float32 的 Tensor 并存入 GPU
            result.append({
                "motion_name": motion_name,
                "fps": data["fps"],
                "dof_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
                "dof_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
                "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
                "body_quat_w": torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device),
                "body_lin_vel_w": torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
                "body_ang_vel_w": torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
            })
        return result

    def _get_motion_data(self, motion_index: int = None):
        """内部工具函数：根据索引获取具体的某一个动作数据。如果不传，则返回默认动作。"""
        if motion_index is None:
            return {
                "body_pos_w": self._body_pos_w,
                "body_quat_w": self._body_quat_w,
                "body_lin_vel_w": self._body_lin_vel_w,
                "body_ang_vel_w": self._body_ang_vel_w,
                "dof_pos": self._dof_pos,
                "dof_vel": self._dof_vel,
            }
        else:
            assert 0 <= motion_index < len(
                self.motion_data), f"Motion index {motion_index} out of range [0, {len(self.motion_data)})"
            return self.motion_data[motion_index]

    # =====================================================================
    # 以下所有的 tgt_xxx 函数都是为 RL 的 Reward Function 服务的。
    # 当 RL 环境需要计算“机器人当前的姿势和参考数据的差距”时，就会调用这些接口获取目标值(Target)。
    # =====================================================================
    def tgt_body_pos_w(self, motion_index: int = None) -> torch.Tensor:
        """获取目标追踪连杆（如手腕、膝盖）在世界坐标系下的位置。"""
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._body_indexes, :]

    def tgt_body_quat_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, self._body_indexes, :]

    def tgt_body_lin_vel_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, self._body_indexes, :]

    def tgt_body_ang_vel_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, self._body_indexes, :]

    def tgt_anchor_pos_w(self, motion_index: int = None) -> torch.Tensor:
        """获取锚点（躯干）的位置。"""
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._anchor_indexes]

    def tgt_anchor_quat_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, self._anchor_indexes]

    def tgt_anchor_lin_vel_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, self._anchor_indexes]

    def tgt_anchor_ang_vel_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, self._anchor_indexes]

    def tgt_dof_pos(self, motion_index: int = None) -> torch.Tensor:
        """获取所有关节的角度。"""
        data = self._get_motion_data(motion_index)
        return data["dof_pos"]

    def tgt_dof_vel(self, motion_index: int = None) -> torch.Tensor:
        """获取所有关节的角速度。"""
        data = self._get_motion_data(motion_index)
        return data["dof_vel"]

    def tgt_feet_pos_w(self, motion_index: int = None) -> torch.Tensor:
        """获取脚部在世界坐标系下的位置（用于足端轨迹追踪奖励）。"""
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._feet_indexes]

    # 第 0 个索引通常固定为 Pelvis (基座)
    def tgt_root_pos(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, 0, :]

    def tgt_root_quat(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, 0, :]

    def tgt_root_lin_vel(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, 0, :]

    def tgt_root_ang_vel(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, 0, :]

    # =====================================================================
    # 极其重要的函数：为 AMP 判别器采样
    # =====================================================================
    def sample_random_frames(self, num_samples: int) -> dict[str, torch.Tensor]:
        """
        从所有动作数据集中，随机抽取 num_samples 帧的数据。

        为什么需要这个？
        AMP 算法的判别器（Discriminator）像一个警察，它的任务是区分“真动作”和“机器人的假动作”。
        这个函数就是用来给警察提供“真动作”样本的。
        每次策略网络更新时，都会调用这个函数抓取一批真实的 Root 状态和 Joint 状态喂给判别器。

        Returns:
            包含根节点状态和关节状态的字典字典。
        """
        all_motions = self.motion_data + self.motion_data_recovery

        # 1. 先随机选出要抽取的 motion 的索引
        motion_indices = torch.randint(0, len(all_motions), (num_samples,))

        result_root_pos = []
        result_root_quat = []
        result_root_lin_vel = []
        result_root_ang_vel = []
        result_joint_pos = []
        result_joint_vel = []

        for i in range(num_samples):
            motion = all_motions[motion_indices[i].item()]
            num_frames = motion["dof_pos"].shape[0]

            # 2. 在选定的 motion 中，随机选出某一帧 (相当于在电影里随机暂停)
            frame_idx = torch.randint(0, num_frames, (1,)).item()

            # 3. 提取这一帧的物理数据
            result_root_pos.append(motion["body_pos_w"][frame_idx, 0, :])
            result_root_quat.append(motion["body_quat_w"][frame_idx, 0, :])
            result_root_lin_vel.append(motion["body_lin_vel_w"][frame_idx, 0, :])
            result_root_ang_vel.append(motion["body_ang_vel_w"][frame_idx, 0, :])
            result_joint_pos.append(motion["dof_pos"][frame_idx])
            result_joint_vel.append(motion["dof_vel"][frame_idx])

        # 把列表堆叠为一整块的 PyTorch Tensor，送给神经网络进行前向传播
        return {
            "root_pos": torch.stack(result_root_pos),
            "root_quat": torch.stack(result_root_quat),
            "root_lin_vel": torch.stack(result_root_lin_vel),
            "root_ang_vel": torch.stack(result_root_ang_vel),
            "joint_pos": torch.stack(result_joint_pos),
            "joint_vel": torch.stack(result_joint_vel),
        }