#!/usr/bin/env python3
"""
ZeroGrasp end-to-end demo that uses the native Orbbec SDK for RGB + Depth capture,
VLM for target detection, and TargetLocalizer + ZeroGrasp for grasp pose prediction.

Usage:
    python examples/zerograsp_vlm_demo.py --target-name "红色水杯"

Optional:
    --serial <DEVICE_SERIAL>   # Lock to a specific Orbbec device
    --print-json               # Dump the full ZeroGrasp JSON response
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tempfile
import subprocess
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from pyorbbecsdk import (  # type: ignore
            AlignFilter,
            Config,
            FormatConvertFilter,
            OBConvertFormat,
            OBFormat,
            OBSensorType,
            OBStreamType,
            Pipeline,
        )
except Exception as exc:  # pragma: no cover
    print(f"[ZeroGraspDemo] pyorbbecsdk 不可用: {exc}")
    sys.exit(1)

from orbbec_vlm_tracker import VLMDetector  # type: ignore
from voice.perception.localize_target import TargetLocalizer, DepthSnapshot


def _frame_to_bgr_image(frame) -> Optional[np.ndarray]:
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if color_format == OBFormat.NV12:
        y = data[0:height, :]
        uv = data[height:height + height // 2].reshape(height // 2, width)
        yuv = cv2.merge([y, uv])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        y = data[0:height, :]
        uv = data[height:height + height // 2].reshape(height // 2, width)
        yuv = cv2.merge([y, uv])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if color_format == OBFormat.I420:
        y = data[0:height, :]
        u = data[height:height + height // 4].reshape(height // 2, width // 2)
        v = data[height + height // 4:].reshape(height // 2, width // 2)
        yuv = cv2.merge([y, u, v])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    convert_format = None
    if color_format == OBFormat.YUYV:
        convert_format = OBConvertFormat.YUYV_TO_RGB888
    elif color_format == OBFormat.NV12:
        convert_format = OBConvertFormat.NV12_TO_RGB888
    elif color_format == OBFormat.NV21:
        convert_format = OBConvertFormat.NV21_TO_RGB888
    elif color_format == OBFormat.I420:
        convert_format = OBConvertFormat.I420_TO_RGB888
    if convert_format is None:
        return None
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        return None
    return _frame_to_bgr_image(rgb_frame)


def _depth_to_array(depth_frame) -> np.ndarray:
    height = depth_frame.get_height()
    width = depth_frame.get_width()
    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    return data.reshape(height, width).copy()


class OrbbecRGBDCapture:
    def __init__(
        self,
        serial_filter: Optional[str] = None,
        warmup: int = 30,
        align_to_color: bool = False,
    ) -> None:
        self.pipeline = Pipeline()
        self.config = Config()
        self.align_filter: Optional[AlignFilter] = None
        self._configure_streams(serial_filter)
        if align_to_color:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.pipeline.start(self.config)
        self._warmup(warmup)

    def _configure_streams(self, serial_filter: Optional[str]) -> None:
        if serial_filter:
            device_list = self.pipeline.get_connected_devices()
            if device_list:
                for idx in range(device_list.get_count()):
                    device = device_list.get_device(idx)
                    info = device.get_device_info()
                    if info and info.serial_number == serial_filter:
                        cfg = Config()
                        cfg.enable_streams_from_device(device, self.config)
                        self.config = cfg
                        break
        for sensor in (OBSensorType.COLOR_SENSOR, OBSensorType.DEPTH_SENSOR):
            profile_list = self.pipeline.get_stream_profile_list(sensor)
            if profile_list is None:
                raise RuntimeError(f"无法获取 {sensor} 流配置")
            profile = profile_list.get_default_video_stream_profile()
            if profile is None:
                raise RuntimeError(f"未找到 {sensor} 默认配置")
            self.config.enable_stream(profile)

    def _warmup(self, warmup: int) -> None:
        for _ in range(max(5, warmup)):
            frames = self.pipeline.wait_for_frames(1000)
            if frames and frames.get_depth_frame() and frames.get_color_frame():
                break
            time.sleep(0.03)

    def capture(self) -> Tuple[np.ndarray, DepthSnapshot]:
        frames = None
        for _ in range(100):
            frames = self.pipeline.wait_for_frames(1000)
            if frames and self.align_filter is not None:
                aligned = self.align_filter.process(frames)
                if not aligned:
                    continue
                try:
                    frames = aligned.as_frame_set()
                except AttributeError:
                    frames = aligned
            if frames and frames.get_depth_frame() and frames.get_color_frame():
                break
        if frames is None:
            raise RuntimeError("无法获取同步的 RGBD 帧")
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError("帧不完整，缺少颜色或深度帧")

        color_bgr = _frame_to_bgr_image(color_frame)
        if color_bgr is None:
            raise RuntimeError("颜色帧格式转换失败")
        depth_arr = _depth_to_array(depth_frame)

        depth_profile = depth_frame.get_stream_profile().as_video_stream_profile()
        color_profile = color_frame.get_stream_profile().as_video_stream_profile()
        depth_intr = depth_profile.get_intrinsic()
        extrinsic = depth_profile.get_extrinsic_to(color_profile)

        snapshot = DepthSnapshot(
            depth=depth_arr,
            intrinsics=depth_intr,
            extrinsic=extrinsic,
            dtype=str(depth_arr.dtype),
        )
        return color_bgr, snapshot

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def _save_temp_image(frame_bgr: np.ndarray) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="zerograsp_", suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, frame_bgr)
    return Path(tmp.name)


def _print_best_grasp(grasp_result: Dict[str, Any]) -> None:
    # ZeroGrasp 的返回格式在不同版本/配置下可能不同：
    # - grasp_result["objects"] 可能是 dict（单对象）或 list（多个对象）
    # - 或者顶层直接是一个对象列表
    objs = None
    if isinstance(grasp_result, dict):
        objs = grasp_result.get("objects")
    else:
        objs = grasp_result

    # 统一把所有对象的 grasps 合并到一个列表
    all_grasps = []
    if objs is None:
        # 尝试将顶层作为单个对象处理
        if isinstance(grasp_result, list):
            objs = grasp_result
        else:
            objs = []

    if isinstance(objs, dict):
        # 单个对象，可能包含 "grasps"
        g = objs.get("grasps")
        if isinstance(g, list):
            all_grasps.extend(g)
    elif isinstance(objs, list):
        for o in objs:
            if not isinstance(o, dict):
                continue
            g = o.get("grasps")
            if isinstance(g, list):
                all_grasps.extend(g)

    print(f"ZeroGrasp 返回 {len(all_grasps)} 个抓取候选")
    if not all_grasps:
        return

    # 选择得分最高的抓取候选
    best = max(all_grasps, key=lambda g: float(g.get("score", 0.0)))
    tcp = best.get("translation_mm") or best.get("position_mm")
    approach = best.get("approach_vector") or best.get("approach_dir")
    width = best.get("width_mm")
    score = float(best.get("score", 0.0))
    print(f"最佳评分: {score:.3f}, tcp(mm): {tcp}, approach: {approach}, width_mm: {width}")


# 相机 -> 机器人坐标的外参
R_R_C = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
P_R_C = np.array([50.0, 180.0, 0.0])
DEFAULT_TOOL_ROT_OFFSET_RAD = math.pi / 4  # 45° mechanical偏差


import numpy as np

################################################################################
# 旋转矩阵 → 欧拉角：六种顺序
################################################################################

def _euler_from_matrix_general(R: np.ndarray, order: str) -> tuple[float, float, float]:
    """
    通用转换函数，支持 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'。
    参考自 https://www.euclideanspace.com/ ，默认采用右手系、内旋定义。
    """
    assert R.shape == (3, 3), "R 必须是 3x3 矩阵"
    eps = 1e-6
    m = R
    if order == "xyz":
        sy = np.sqrt(m[0, 0] * m[0, 0] + m[0, 1] * m[0, 1])
        if sy > eps:
            x = np.arctan2(m[1, 2], m[2, 2])
            y = np.arctan2(-m[0, 2], sy)
            z = np.arctan2(m[0, 1], m[0, 0])
        else:
            x = np.arctan2(-m[2, 1], m[1, 1])
            y = np.arctan2(-m[0, 2], sy)
            z = 0.0
    elif order == "xzy":
        sz = np.sqrt(m[0, 0] * m[0, 0] + m[0, 2] * m[0, 2])
        if sz > eps:
            x = np.arctan2(-m[1, 2], m[1, 1])
            z = np.arctan2(-m[0, 1], sz)
            y = np.arctan2(m[0, 2], m[0, 0])
        else:
            x = np.arctan2(m[2, 0], m[2, 2])
            z = np.arctan2(-m[0, 1], sz)
            y = 0.0
    elif order == "yxz":
        sx = np.sqrt(m[1, 1] * m[1, 1] + m[1, 2] * m[1, 2])
        if sx > eps:
            y = np.arctan2(m[0, 2], m[2, 2])
            x = np.arctan2(-m[1, 2], sx)
            z = np.arctan2(m[1, 0], m[1, 1])
        else:
            y = np.arctan2(-m[2, 0], m[0, 0])
            x = np.arctan2(-m[1, 2], sx)
            z = 0.0
    elif order == "yzx":
        sz = np.sqrt(m[1, 1] * m[1, 1] + m[1, 0] * m[1, 0])
        if sz > eps:
            y = np.arctan2(-m[0, 1], m[0, 0])
            z = np.arctan2(-m[1, 0], sz)
            x = np.arctan2(m[1, 2], m[1, 1])
        else:
            y = np.arctan2(m[2, 0], m[2, 2])
            z = np.arctan2(-m[1, 0], sz)
            x = 0.0
    elif order == "zxy":
        sx = np.sqrt(m[2, 2] * m[2, 2] + m[2, 0] * m[2, 0])
        if sx > eps:
            z = np.arctan2(m[0, 1], m[1, 1])
            x = np.arctan2(-m[2, 1], sx)
            y = np.arctan2(m[2, 0], m[2, 2])
        else:
            z = np.arctan2(-m[1, 0], m[0, 0])
            x = np.arctan2(-m[2, 1], sx)
            y = 0.0
    elif order == "zyx":
        sy = np.sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0])
        if sy > eps:
            z = np.arctan2(m[1, 0], m[0, 0])
            y = np.arctan2(-m[2, 0], sy)
            x = np.arctan2(m[2, 1], m[2, 2])
        else:
            z = np.arctan2(-m[0, 1], m[1, 1])
            y = np.arctan2(-m[2, 0], sy)
            x = 0.0
    else:
        raise ValueError(f"不支持的欧拉顺序: {order}")
    return float(x), float(y), float(z)


def rot_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    return _euler_from_matrix_general(R, "xyz")


def rot_to_euler_xzy(R: np.ndarray) -> tuple[float, float, float]:
    return _euler_from_matrix_general(R, "xzy")


def rot_to_euler_yxz(R: np.ndarray) -> tuple[float, float, float]:
    return _euler_from_matrix_general(R, "yxz")


def rot_to_euler_yzx(R: np.ndarray) -> tuple[float, float, float]:
    return _euler_from_matrix_general(R, "yzx")


def rot_to_euler_zxy(R: np.ndarray) -> tuple[float, float, float]:
    return _euler_from_matrix_general(R, "zxy")


def rot_to_euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    return _euler_from_matrix_general(R, "zyx")


EULER_ORDER_FUNCS: Dict[str, Callable[[np.ndarray], tuple[float, float, float]]] = {
    "xyz": rot_to_euler_xyz,
    "xzy": rot_to_euler_xzy,
    "yxz": rot_to_euler_yxz,
    "yzx": rot_to_euler_yzx,
    "zxy": rot_to_euler_zxy,
    "zyx": rot_to_euler_zyx,
}

SELECTED_EULER_ORDER = "zyx"


def set_selected_euler_order(order: str) -> None:
    global SELECTED_EULER_ORDER
    if order not in EULER_ORDER_FUNCS:
        raise ValueError(f"不支持的欧拉顺序: {order}")
    SELECTED_EULER_ORDER = order


def get_selected_euler_order() -> str:
    return SELECTED_EULER_ORDER


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为轴角向量（长度=旋转弧度，方向=旋转轴）。
    """
    assert R.shape == (3, 3), "R 必须是 3x3 矩阵"
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3, dtype=float)
    if abs(theta - np.pi) < 1e-4:
        # 180 度附近单独处理，避免除以极小的 sin(theta)
        diag = np.diag(R)
        axis = np.zeros(3, dtype=float)
        idx = int(np.argmax(diag))
        axis[idx] = np.sqrt(max(diag[idx] + 1.0, 0.0) / 2.0)
        if axis[idx] < 1e-6:
            axis[idx] = 1.0
        j = (idx + 1) % 3
        k = (idx + 2) % 3
        axis[j] = R[idx, j] / (2.0 * axis[idx])
        axis[k] = R[idx, k] / (2.0 * axis[idx])
        axis = axis / np.linalg.norm(axis)
        return axis * theta
    axis_vec = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )
    axis = axis_vec / (2.0 * np.sin(theta))
    return axis * theta


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        raise ValueError("向量模长过小，无法归一化")
    return vec / norm


def _fallback_open_direction(approach: np.ndarray) -> np.ndarray:
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(basis, approach)) > 0.99:
        basis = np.array([0.0, 1.0, 0.0])
    return _normalize_vec(np.cross(approach, np.cross(basis, approach)))


def _gripper_axes_from_matrix(rotation_matrix: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ZeroGrasp 的矩阵列向量含义：
    - b1: -gnormal，夹爪深入方向（ZeroGrasp 坐标 +x）
    - b2: tangent，夹爪张开方向（ZeroGrasp 坐标 +y）
    - b3: b1 × b2，指厚方向（ZeroGrasp 坐标 +z）

    实际夹爪坐标采用:
    - +Z: 夹爪指向（沿 b1）
    - +Y: 张开方向（沿 b2）
    - +X: 指厚向下（取 -b3，保持右手系）
    """
    rot = np.array(rotation_matrix, dtype=float)
    approach = _normalize_vec(rot[:, 0])
    open_dir = rot[:, 1]
    if np.linalg.norm(open_dir) < 1e-6:
        open_dir = _fallback_open_direction(approach)
    else:
        open_dir = _normalize_vec(open_dir - np.dot(open_dir, approach) * approach)
    thick = np.cross(approach, open_dir)
    if np.linalg.norm(thick) < 1e-6:
        open_dir = _fallback_open_direction(approach)
        thick = np.cross(approach, open_dir)
    thick = _normalize_vec(thick)
    open_dir = _normalize_vec(np.cross(thick, approach))
    x_axis = _normalize_vec(-thick)
    y_axis = open_dir
    z_axis = approach
    return x_axis, y_axis, z_axis


def _axes_to_rotation_matrix(axes: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.column_stack(axes)


def _rotation_about_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _apply_tool_rotation_offset(
    rotation_matrix: np.ndarray,
    offset_rad: float,
    mode: str,
) -> np.ndarray:
    if abs(offset_rad) < 1e-8:
        return rotation_matrix
    offset_matrix = _rotation_about_z(offset_rad)
    if mode == "robot":
        return offset_matrix @ rotation_matrix
    return rotation_matrix @ offset_matrix


def _extract_gripper_pose(
    grasp: Dict[str, Any]
) -> Optional[
    Tuple[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]
]:
    """
    将 ZeroGrasp 返回的 rotation_matrix 拆解为三个物理含义明确的轴，
    并转换到机器人坐标系。
    """
    rotation = grasp.get("rotation_matrix")
    translation = grasp.get("translation_mm") or grasp.get("position_mm")
    if rotation is None or translation is None:
        print("⚠️  抓取结果缺少 rotation_matrix 或 translation_mm")
        return None
    try:
        axes_c = _gripper_axes_from_matrix(rotation)
    except ValueError as exc:
        print(f"⚠️  夹爪姿态解析失败: {exc}")
        return None
    position_c = np.array(translation, dtype=float)
    position_r = R_R_C @ position_c + P_R_C
    axes_r = tuple(_normalize_vec(R_R_C @ axis) for axis in axes_c)
    return position_r, axes_r, axes_c


def _offset_gripper_to_tcp(
    position_mm: np.ndarray,
    approach_axis: np.ndarray,
    offset_mm: float = 150.0,
) -> np.ndarray:
    """
    ZeroGrasp 返回的位置视为夹爪中心，实际夹爪沿局部 +Z（指向方向）后退 offset_mm。
    """
    approach = _normalize_vec(approach_axis)
    return position_mm - approach * offset_mm


def _call_ros2_linear_move(
    position_mm: np.ndarray,
    axes_robot: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tool_rotation_offset: float,
    tool_rotation_frame: str,
) -> bool:
    """
    调用 ROS2 服务 /jaka_driver/linear_move 来移动机器人。
    """
    try:
        tcp_position = _offset_gripper_to_tcp(position_mm, axes_robot[2], offset_mm=150.0)
        print(
            "TCP (after local +Z backoff 150mm): "
            f"[{tcp_position[0]:.1f}, {tcp_position[1]:.1f}, {tcp_position[2]:.1f}] mm"
        )
        rotation_matrix = _axes_to_rotation_matrix(axes_robot)
        rotation_matrix_cmd = _apply_tool_rotation_offset(
            rotation_matrix,
            tool_rotation_offset,
            tool_rotation_frame,
        )
        rot_vec = rotation_matrix_to_axis_angle(rotation_matrix_cmd)
        order = get_selected_euler_order()
        roll, pitch, yaw = EULER_ORDER_FUNCS[order](rotation_matrix_cmd)
        pose_list = list(tcp_position) + list(rot_vec)
        pose_str = ", ".join(f"{v:.3f}" for v in pose_list)

        # 打印机器人最终末端位姿
        print("=" * 60)
        print("机器人最终末端位姿 (Robot End-Effector Pose)")
        print("=" * 60)
        print(f"  位置 (mm):")
        print(f"    X: {tcp_position[0]:>10.3f}")
        print(f"    Y: {tcp_position[1]:>10.3f}")
        print(f"    Z: {tcp_position[2]:>10.3f}")
        print(f"  轴角 (rad):")
        print(f"    Rx: {rot_vec[0]:>10.4f}")
        print(f"    Ry: {rot_vec[1]:>10.4f}")
        print(f"    Rz: {rot_vec[2]:>10.4f}")
        print(f"  欧拉角[{order.upper()}] (rad):")
        print(f"    Roll:  {roll:>10.4f} ({math.degrees(roll):>8.2f}°)")
        print(f"    Pitch: {pitch:>10.4f} ({math.degrees(pitch):>8.2f}°)")
        print(f"    Yaw:   {yaw:>10.4f} ({math.degrees(yaw):>8.2f}°)")
        print(f"  完整 TCP Pose: [{pose_str}]")
        print("=" * 60)

        ros2_cmd = [
            "ros2",
            "service",
            "call",
            "/jaka_driver/linear_move",
            "jaka_msgs/srv/Move",
            f"""{{
    pose: [{pose_str}],
    mvvelo: 10.0,
    mvacc: 10.0,
    has_ref: false,
    ref_joint: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    mvtime: 0.0,
    mvradii: 0.0,
    coord_mode: 0,
    index: 0
}}""",
        ]

        print(f"rot_vec: [{rot_vec[0]:.3f}, {rot_vec[1]:.3f}, {rot_vec[2]:.3f}]")
        print(f"euler[{order.upper()}] (rad): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")

        # result = subprocess.run(
        #     ros2_cmd,
        #     capture_output=True,
        #     text=True,
        #     timeout=10,
        # )

        # if result.returncode == 0:
        #     print("[ROS2] ✓ 服务调用成功")
        #     if result.stdout:
        #         print(f"  输出: {result.stdout[:200]}")
        #     return True
        # print(f"[ROS2] ✗ 服务调用失败 (返回码: {result.returncode})")
        if result.stderr:
            print(f"  错误: {result.stderr[:200]}")
        return False
    except FileNotFoundError:
        print("[ROS2] ✗ 找不到 ros2 命令，请检查 ROS2 环境")
        return False
    except subprocess.TimeoutExpired:
        print("[ROS2] ✗ 服务调用超时")
        return False
    except Exception as e:
        print(f"[ROS2] ✗ 服务调用异常: {e}")
        return False


def _get_best_grasp(grasp_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从 ZeroGrasp 结果中提取最优抓取候选。
    
    Returns:
        最优抓取字典或 None
    """
    objs = None
    if isinstance(grasp_result, dict):
        objs = grasp_result.get("objects")
    else:
        objs = grasp_result

    all_grasps = []
    if objs is None:
        if isinstance(grasp_result, list):
            objs = grasp_result
        else:
            objs = []

    if isinstance(objs, dict):
        g = objs.get("grasps")
        if isinstance(g, list):
            all_grasps.extend(g)
    elif isinstance(objs, list):
        for o in objs:
            if not isinstance(o, dict):
                continue
            g = o.get("grasps")
            if isinstance(g, list):
                all_grasps.extend(g)

    if not all_grasps:
        return None
    
    return max(all_grasps, key=lambda g: float(g.get("score", 0.0)))


def _print_timings(timings: Dict[str, float]) -> None:
    if not timings:
        return
    print("[Timing]")
    for name, duration in timings.items():
        if name.endswith("payload_size_mb"):
            print(f"  - {name}: {duration:.2f} MB")
        elif name.endswith("bandwidth_mb_s"):
            print(f"  - {name}: {duration:.2f} MB/s")
        elif name.endswith("bandwidth_mbit_s"):
            print(f"  - {name}: {duration:.2f} Mbit/s")
        else:
            print(f"  - {name}: {duration * 1000:.1f} ms")


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _record_event(events: list[tuple[str, str]], label: str) -> None:
    events.append((_timestamp(), label))


def _print_event_log(events: list[tuple[str, str]]) -> None:
    if not events:
        return
    print("[Timestamps]")
    for ts, label in events:
        print(f"  - {ts} {label}")


def _scale_bbox(
    bbox: list[int],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> list[int]:
    if src_size == dst_size:
        return [int(v) for v in bbox]
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = dst_w / max(1, src_w)
    sy = dst_h / max(1, src_h)
    x1 = int(round(bbox[0] * sx))
    y1 = int(round(bbox[1] * sy))
    x2 = int(round(bbox[2] * sx))
    y2 = int(round(bbox[3] * sy))
    x1 = max(0, min(dst_w - 1, x1))
    x2 = max(0, min(dst_w - 1, x2))
    y1 = max(0, min(dst_h - 1, y1))
    y2 = max(0, min(dst_h - 1, y2))
    if x2 <= x1:
        x2 = min(dst_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(dst_h - 1, y1 + 1)
    return [x1, y1, x2, y2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbbec + VLM + ZeroGrasp demo")
    parser.add_argument("--target-name", required=True, help="VLM 要查找的目标名称")
    parser.add_argument("--serial", default=os.getenv("ORBBEC_SERIAL"), help="可选：指定 Orbbec 设备序列号")
    parser.add_argument("--print-json", action="store_true", help="输出完整 ZeroGrasp JSON")
    parser.add_argument("--disable-align", action="store_true", help="禁用 Orbbec 深度对齐到彩色（默认开启）")
    parser.add_argument("--move-robot", action="store_true", help="调用 ROS2 服务移动机器人到抓取位姿")
    parser.add_argument(
        "--euler-order",
        choices=sorted(EULER_ORDER_FUNCS.keys()),
        default=os.getenv("ZEROGRASP_EULER_ORDER", "zyx"),
        help="零抓姿态转换为机器人欧拉角时使用的顺序（默认: zyx）",
    )
    parser.add_argument(
        "--tool-rotation-offset",
        type=float,
        default=float(os.getenv("ZEROGRASP_TOOL_ROT_OFFSET", f"{DEFAULT_TOOL_ROT_OFFSET_RAD}")),
        help="夹爪与法兰的固定偏转量，单位：弧度（默认 0.785 ≈ 45°，可用环境变量 ZEROGRASP_TOOL_ROT_OFFSET 覆盖）",
    )
    parser.add_argument(
        "--tool-rotation-offset-mode",
        choices=("local", "robot"),
        default=os.getenv("ZEROGRASP_TOOL_ROT_MODE", "local"),
        help="偏转量作用的坐标系：local=绕夹爪局部 +Z；robot=绕机器人 Z",
    )
    args = parser.parse_args()

    timings: OrderedDict[str, float] = OrderedDict()
    events: list[tuple[str, str]] = []

    set_selected_euler_order(args.euler_order)
    print(f"[ZeroGraspDemo] Euler order for robot output: {get_selected_euler_order().upper()}")
    print(
        "[ZeroGraspDemo] Tool rotation offset: "
        f"{args.tool_rotation_offset:.3f} rad ({math.degrees(args.tool_rotation_offset):.1f}°) "
        f"about {args.tool_rotation_offset_mode} frame"
    )

    align_to_color = not args.disable_align
    print(f"[ZeroGraspDemo] Depth-to-color alignment: {'ON' if align_to_color else 'OFF'}")

    capture = OrbbecRGBDCapture(serial_filter=args.serial, align_to_color=align_to_color)
    try:
        _record_event(events, "capture.start")
        stage_start = time.perf_counter()
        color_bgr, depth_snapshot = capture.capture()
        timings["capture"] = time.perf_counter() - stage_start
    finally:
        capture.stop()
        _record_event(events, "capture.end")

    zerograsp_size = (1280, 800)  # width, height
    vlm_input_size = (1000, 1000)  # width, height

    color_bgr_zg = color_bgr
    if (color_bgr.shape[1], color_bgr.shape[0]) != zerograsp_size:
        color_bgr_zg = cv2.resize(color_bgr, zerograsp_size, interpolation=cv2.INTER_LINEAR)
    color_bgr_vlm = cv2.resize(color_bgr_zg, vlm_input_size, interpolation=cv2.INTER_LINEAR)
    color_bgr = color_bgr_zg

    image_path = _save_temp_image(color_bgr_vlm)
    vlm = VLMDetector()
    _record_event(events, "vlm.detect.start")
    stage_start = time.perf_counter()
    detection = vlm.detect(image_path, args.target_name)
    timings["vlm.detect"] = time.perf_counter() - stage_start
    _record_event(events, "vlm.detect.end")
    bbox_color = _scale_bbox(list(detection.bbox_xyxy), vlm_input_size, zerograsp_size)
    print(f"VLM 检测: bbox={bbox_color}, confidence={detection.confidence:.3f}")

    rgb_frame = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_h, depth_w = depth_snapshot.depth.shape
    color_h, color_w = rgb_frame.shape[:2]
    bbox_depth = _scale_bbox(bbox_color, (color_w, color_h), (depth_w, depth_h))
    if (color_w, color_h) != (depth_w, depth_h):
        rgb_frame = cv2.resize(rgb_frame, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)

    localizer = TargetLocalizer()
    _record_event(events, "localizer.localize.start")
    stage_start = time.perf_counter()
    transform = localizer.localize_object(
        bbox=bbox_depth,
        snapshot=depth_snapshot,
        surface_points_hint=None,
        range_estimate=None,
        rgb_frame=rgb_frame,
        include_transform=True,
    )
    timings["localizer.localize"] = time.perf_counter() - stage_start
    _record_event(events, "localizer.localize.end")
    if not transform:
        raise RuntimeError("localize_object 失败，无法运行 ZeroGrasp")

    transform_result = transform.get("transform_result", transform)
    zg_timings: Dict[str, float] = {}
    _record_event(events, "zerograsp.call.start")
    stage_start = time.perf_counter()
    grasp_result = localizer.run_zero_grasp_inference(
        transform_result=transform_result,
        bbox=transform["bbox"],
        rgb_frame=rgb_frame,
        timings=zg_timings,
        events=events,
    )
    timings["zerograsp.call"] = time.perf_counter() - stage_start
    _record_event(events, "zerograsp.call.end")
    for key, value in zg_timings.items():
        timings[f"zerograsp.{key}"] = value
    if grasp_result is None:
        raise RuntimeError("ZeroGrasp 返回空结果")

    _print_best_grasp(grasp_result)
    if args.print_json:
        print(json.dumps(grasp_result, ensure_ascii=False, indent=2))

    # 提取最优抓取并调用 ROS2 服务
    if args.move_robot:
        best_grasp = _get_best_grasp(grasp_result)
        if best_grasp:
            pose = _extract_gripper_pose(best_grasp)
            if pose:
                position_mm, axes_robot, _ = pose
                _call_ros2_linear_move(
                    position_mm,
                    axes_robot,
                    args.tool_rotation_offset,
                    args.tool_rotation_offset_mode,
                )
            else:
                print("⚠️  无法提取机器人位姿，跳过 ROS2 服务调用")
        else:
            print("⚠️  未找到有效的抓取候选")

    _print_timings(timings)
    _print_event_log(events)


if __name__ == "__main__":
    main()
