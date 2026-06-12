"""RL configuration for Moya X3 AMP locomotion task."""

import os
from dataclasses import dataclass, field
from typing import List

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


_MOTION_DATA_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
  "src", "assets", "motions", "x3", "amp",
)

@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Extended runner config with AMP-specific parameters."""
  amp_reward_coef: float = 0.1
  amp_motion_files: str = ""
  amp_num_preload_transitions: int = 200000
  amp_task_reward_lerp: float = 0.75
  amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])

  # 【核心防爆改动】min_normalized_std 用于设定策略网络输出动作的最小方差。
  # G1 是 29 个自由度，而你的 X3 是 28 个自由度。如果不改这里，底层 C++ 或 PyTorch 运算时会直接报维度越界错误。
  min_normalized_std: List[float] = field(default_factory=lambda: [0.05] * 28)
  amp_body_names: tuple = ()
  amp_anchor_name: str = ""


def x3_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create RL runner configuration for Moya X3 AMP locomotion task."""
  return RslRlAmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMPPPO",
    ),
    experiment_name="x3_amp_locomotion",
    logger="wandb",
    wandb_project="X3_AMP_Locomotion_Project",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100001,

    # AMP 参数配置
    amp_reward_coef=0.1,
    amp_motion_files=os.path.normpath(_MOTION_DATA_DIR),
    amp_num_preload_transitions=200000, # 训练开始前，从你的 AMASS 转成的 NPZ 数据里预加载多少步
    amp_task_reward_lerp=0.75,
    amp_discr_hidden_dims=[1024, 512, 256],
    min_normalized_std=[0.05] * 28,
    amp_body_names=(
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ),
    amp_anchor_name="torso_link",
  )