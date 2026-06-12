#!/usr/bin/env python3
"""Analyze base height and base-frame velocities from a motion NPZ file.

这个脚本用于读取运动数据 npz 文件，并提取/计算基座（Base/Pelvis）的关键状态：
- base height: 基座在世界坐标系下的高度 (Z轴)
- base linear velocity in world frame: 世界坐标系下的线速度
- base linear velocity transformed to base frame: 转换到基座局部坐标系下的线速度 (RL 常用)
- base angular velocity in world frame: 世界坐标系下的角速度
- base angular velocity transformed to base frame: 转换到基座局部坐标系下的角速度

默认读取路径指向:
/home/crp/wbc_mjlab/motion_data_npz/amp/WalkandRun/walk_sideway_right_loop_001__A022.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ==========================================
# 四元数与坐标系转换工具函数
# 强化学习中，策略网络（Actor）通常依赖机器人的局部坐标系（Base Frame）来做决策。
# 因为机器人不需要知道自己在世界里的绝对位置，只需要知道“我正在以多少速度向我的正前方移动”。
# ==========================================

def quat_normalize(quat: np.ndarray) -> np.ndarray:
    """归一化四元数，确保其模长为1。形状为 [..., 4]。
    如果四元数不归一化，旋转计算会导致向量被意外缩放。"""
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)  # 防止除以0
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """求四元数的共轭 (Conjugate)。
    输入输出格式均为 [w, x, y, z]。
    对于单位四元数，共轭就等于它的逆 (Inverse)。
    物理意义：如果 q 表示从 A 旋转到 B，那么它的共轭就表示从 B 旋转回 A。"""
    out = quat.copy()
    out[..., 1:] *= -1.0  # 将虚部 (x, y, z) 取反
    return out


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """计算两个四元数的哈密顿乘积 (Hamilton product)。格式 [w, x, y, z]。
    相当于连续进行两次旋转。"""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def rotate_vector_world_to_body(q_bw: np.ndarray, v_w: np.ndarray) -> np.ndarray:
    """将向量从世界坐标系 (World Frame) 转换到机体坐标系 (Body Frame)。

    数学原理：v_body = q_inverse * v_world * q
    这里 q_bw 是机体在世界坐标系下的姿态。我们需要用它的逆（共轭）来把世界速度转到局部速度。

    参数:
        q_bw: 机体在世界坐标系下的姿态，形状 [N, 4]，格式 [w, x, y, z]。
        v_w: 世界坐标系下的向量（如线速度/角速度），形状 [N, 3]。

    返回:
        机体坐标系下的向量，形状 [N, 3]。
    """
    q_bw = quat_normalize(q_bw)
    q_wb = quat_conjugate(q_bw)  # 求逆，表示从世界坐标系转回机体坐标系

    # 将 3D 向量 (x, y, z) 填充为一个纯四元数 (0, x, y, z) 以便参与四元数乘法
    zeros = np.zeros((v_w.shape[0], 1), dtype=v_w.dtype)
    v_quat = np.concatenate([zeros, v_w], axis=-1)

    # 执行旋转: q_inverse * v * q
    v_b_quat = quat_multiply(quat_multiply(q_wb, v_quat), q_bw)
    return v_b_quat[..., 1:]  # 抛弃实部，取出计算后的 (x, y, z)


def xyz_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """格式转换：从 [x, y, z, w] 转换为 [w, x, y, z]。
    MuJoCo 默认使用 wxyz，而某些动捕软件默认使用 xyzw。"""
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1)


def print_stats(name: str, values: np.ndarray) -> None:
    """打印统计信息：最小值、最大值、平均值、标准差。
    用于快速检查数据是否出现 NaN 或者数值爆炸。"""
    print(f"\n{name}")
    if values.ndim == 1:
        print(
            "  min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                float(np.min(values)),
                float(np.max(values)),
                float(np.mean(values)),
                float(np.std(values)),
            )
        )
    elif values.ndim == 2:
        labels = ["x", "y", "z"]
        for i in range(values.shape[1]):
            label = labels[i] if i < len(labels) else f"dim{i}"
            c = values[:, i]
            print(
                "  {}: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                    label,
                    float(np.min(c)),
                    float(np.max(c)),
                    float(np.mean(c)),
                    float(np.std(c)),
                )
            )


# ==========================================
# 核心分析流程
# ==========================================
def analyze_motion(npz_file: Path, base_index: int, quat_format: str, preview: int) -> None:
    """读取并分析 NPZ 运动文件中的基座高度和线/角速度。"""
    data = np.load(npz_file)

    # 检查 NPZ 文件是否包含必要的 AMP 训练字段
    required_keys = ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        raise KeyError(f"NPZ 文件中缺失必要的键: {missing}")

    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    body_lin_vel_w = data["body_lin_vel_w"]
    body_ang_vel_w = data["body_ang_vel_w"]

    # 维度形状校验：[时间帧数(T), 连杆数量(B), 空间维度(3或4)]
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w 形状必须是 [T, B, 3], 但得到 {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(f"body_quat_w 形状必须是 [T, B, 4], 但得到 {body_quat_w.shape}")
    if body_lin_vel_w.ndim != 3 or body_lin_vel_w.shape[-1] != 3:
        raise ValueError(f"body_lin_vel_w 形状必须是 [T, B, 3], 但得到 {body_lin_vel_w.shape}")
    if body_ang_vel_w.ndim != 3 or body_ang_vel_w.shape[-1] != 3:
        raise ValueError(f"body_ang_vel_w 形状必须是 [T, B, 3], 但得到 {body_ang_vel_w.shape}")

    num_frames, num_bodies, _ = body_pos_w.shape
    if not (0 <= base_index < num_bodies):
        raise IndexError(f"base_index {base_index} 超出范围 [0, {num_bodies - 1}]")

    # 提取第 0 个连杆（通常是 Base / Pelvis）的数据
    base_pos_w = body_pos_w[:, base_index, :]  # [T, 3]
    base_quat_w = body_quat_w[:, base_index, :]  # [T, 4]
    base_vel_w = body_lin_vel_w[:, base_index, :]  # [T, 3]
    base_ang_vel_w = body_ang_vel_w[:, base_index, :]  # [T, 3]

    if quat_format == "xyzw":
        base_quat_w = xyz_to_wxyz(base_quat_w)

    # 提取高度，并将速度从世界坐标系旋转到机体局部坐标系
    base_height = base_pos_w[:, 2]  # 世界坐标系下的 Z 轴即为高度
    base_vel_b = rotate_vector_world_to_body(base_quat_w, base_vel_w)
    base_ang_vel_b = rotate_vector_world_to_body(base_quat_w, base_ang_vel_w)

    fps = None
    if "fps" in data.files:
        fps_arr = np.asarray(data["fps"]).reshape(-1)
        if fps_arr.size > 0:
            fps = float(fps_arr[0])

    print("=" * 80)
    print(f"分析文件: {npz_file}")
    print(f"总帧数={num_frames}, 连杆总数={num_bodies}, 基座索引={base_index}")
    if fps is not None:
        print(f"控制频率(fps)={fps:.3f}, 动作总时长={num_frames / fps:.3f}秒")
    print("四元数解析格式:", quat_format)
    print("=" * 80)

    # 打印各项指标的统计信息，用于检验数据合理性
    print_stats("Base Height (world z) [基座高度]", base_height)
    print_stats("Base Linear Velocity (world frame) [世界坐标系线速度]", base_vel_w)
    print_stats("Base Linear Velocity (base frame) [局部坐标系线速度]", base_vel_b)
    print_stats("Base Angular Velocity (world frame) [世界坐标系角速度]", base_ang_vel_w)
    print_stats("Base Angular Velocity (base frame) [局部坐标系角速度]", base_ang_vel_b)

    # 校验：无论是世界坐标系还是局部坐标系，速度的标量大小（模长）必须是一致的
    speed_w = np.linalg.norm(base_vel_w, axis=-1)
    speed_b = np.linalg.norm(base_vel_b, axis=-1)
    print_stats("Speed Magnitude (world/base, should match) [线速度模长检验]", np.stack([speed_w, speed_b], axis=-1))
    ang_speed_w = np.linalg.norm(base_ang_vel_w, axis=-1)
    ang_speed_b = np.linalg.norm(base_ang_vel_b, axis=-1)
    print_stats(
        "Angular Speed Magnitude (world/base, should match) [角速度模长检验]",
        np.stack([ang_speed_w, ang_speed_b], axis=-1),
    )

    preview = max(0, int(preview))
    if preview > 0:
        show_n = min(preview, num_frames)
        print("\nPreview (前 {} 帧预览):".format(show_n))
        print(
            "frame | height | v_w(x,y,z) 世界线速度 | v_b(x,y,z) 局部线速度 | w_w(x,y,z) 世界角速度 | w_b(x,y,z) 局部角速度")
        for i in range(show_n):
            vw = base_vel_w[i]
            vb = base_vel_b[i]
            ww = base_ang_vel_w[i]
            wb = base_ang_vel_b[i]
            print(
                f"{i:5d} | {base_height[i]: .5f} | "
                f"({vw[0]: .5f},{vw[1]: .5f},{vw[2]: .5f}) | "
                f"({vb[0]: .5f},{vb[1]: .5f},{vb[2]: .5f}) | "
                f"({ww[0]: .5f},{ww[1]: .5f},{ww[2]: .5f}) | "
                f"({wb[0]: .5f},{wb[1]: .5f},{wb[2]: .5f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析运动 NPZ 文件中的基座高度和速度，并进行世界坐标系到机体坐标系的转换"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        # 默认路径已指向你的 x3 数据目录下的某个文件
        default=Path("/home/crp/wbc_mjlab/motion_data_npz/amp/WalkandRun/walk_sideway_right_loop_001__A022.npz"),
        help="输入运动 npz 文件的路径",
    )
    parser.add_argument(
        "--base-index",
        type=int,
        default=0,
        help="基座连杆在数据数组中的索引 (通常 0 就是 Pelvis)",
    )
    parser.add_argument(
        "--quat-format",
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="npz中 body_quat_w 存储的四元数格式 (默认: wxyz)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="打印前 N 帧的详细数据 (默认: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_motion(args.npz, args.base_index, args.quat_format, args.preview)


if __name__ == "__main__":
    main()