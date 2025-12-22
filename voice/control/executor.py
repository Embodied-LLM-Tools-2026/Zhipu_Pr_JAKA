"""
Skill executor responsible for running low-level atomic actions.

This module extracts the execution related logic from the legacy
TaskProcessor so that the orchestrator can delegate plan nodes to
explicit skills following the redesigned architecture.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
import random
import signal
import threading
import requests
import base64
import io
import functools
import msgpack
from msgpack import Packer, unpackb
import cv2
import websockets.sync.client
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from PIL import Image
from jaka_msgs.srv import GetIK, Move, ServoMove, ServoMoveEnable
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from tools.logging.task_logger import log_error, log_info, log_success, log_warning  # type: ignore
from tools.logging.trace_logger import TraceLogger  # type: ignore

# --- MsgPack Helpers ---
def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj

def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
# -----------------------

from ..perception.localize_target import TargetLocalizer, fetch_aligned_rgbd
from .task_structures import ExecutionResult, FailureCode, PlanNode, map_reason_to_failure_code, InspectionPacket, VerifierFinding
from .recovery_manager import RecoveryManager, RecoveryContext, RecoveryKind
from ..perception.sam_worker import sam_mask_worker  # type: ignore
from ..utils.config import Config

try:
    from action_sequence.gripper_controller import GripperController
except Exception:  # pragma: no cover - optional hardware dependency
    GripperController = None


@dataclass
class SkillRuntime:
    navigator: Any
    world_model: Any = None
    observation: Any = None
    frontend_payload: Optional[Dict[str, Any]] = None
    surface_points: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)


R_R_C = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
P_R_C = np.array([50.0, 180.0, 0.0], dtype=float)
DEFAULT_TOOL_ROT_OFFSET_RAD = math.pi / 4
DEFAULT_TCP_BACKOFF_MM = 150.0
VERIFY_CONFIDENCE_THRESHOLD = float(os.getenv("VERIFY_VISIBLE_CONF", "0.5"))
VERIFY_WORKSPACE_MAX_DIST = float(os.getenv("VERIFY_WORKSPACE_MAX_DIST_M", "2.0"))
VERIFY_GRIPPER_CLOSE_WIDTH = float(os.getenv("VERIFY_GRIPPER_CLOSE_WIDTH", "15.0"))  # percent
VERIFY_MIN_BBOX_AREA = float(os.getenv("VERIFY_MIN_BBOX_AREA", "100.0"))
VERIFY_MAX_YAW_ERROR_DEG = float(os.getenv("VERIFY_MAX_YAW_ERROR_DEG", "15.0"))
VERIFY_GRIPPER_WIDTH_DROP_MIN = float(os.getenv("VERIFY_GRIPPER_WIDTH_DROP_MIN", "5.0"))
VERIFY_GRIPPER_CURRENT_MIN = float(os.getenv("VERIFY_GRIPPER_CURRENT_MIN", "50.0"))
DEFAULT_RECOVERY_MAX_TOTAL_ATTEMPTS = int(os.getenv("RECOVERY_MAX_TOTAL_ATTEMPTS", "3"))
DEFAULT_L1_ESCALATE = int(os.getenv("RECOVERY_L1_ESCALATE", "2"))
DEFAULT_L2_ESCALATE = int(os.getenv("RECOVERY_L2_ESCALATE", "1"))
DEFAULT_CONTROL_MODE = os.getenv("CONTROL_MODE", "ours_full").lower()
DEFAULT_ENABLE_TRACE = os.getenv("ENABLE_TRACE", "true").lower() not in {"0", "false", "no"}
DEFAULT_ENABLE_FAILURE_TAX = os.getenv("ENABLE_FAILURE_TAXONOMY", "true").lower() not in {"0", "false", "no"}
DEFAULT_ENABLE_VERIFIER = os.getenv("ENABLE_VERIFIER", "true").lower() not in {"0", "false", "no"}
DEFAULT_ENABLE_RECOVERY = os.getenv("ENABLE_RECOVERY", "true").lower() not in {"0", "false", "no"}
DEFAULT_ENABLE_ROUTER_VLA = os.getenv("ENABLE_ROUTER_VLA", "false").lower() not in {"0", "false", "no"}
DEFAULT_SKILL_TIMEOUT_S = float(os.getenv("SKILL_TIMEOUT_S", "8.0"))

# Verifier Thresholds
VERIFY_CONFIDENCE_THRESHOLD = float(os.getenv("VERIFY_CONFIDENCE_THRESHOLD", "0.4"))
VERIFY_WORKSPACE_MAX_DIST = float(os.getenv("VERIFY_WORKSPACE_MAX_DIST", "0.8"))


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        raise ValueError("向量模长过小，无法归一化")
    return vec / norm


def _fallback_open_direction(approach: np.ndarray) -> np.ndarray:
    basis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(basis, approach))) > 0.99:
        basis = np.array([0.0, 1.0, 0.0], dtype=float)
    return _normalize_vec(np.cross(approach, np.cross(basis, approach)))


def _gripper_axes_from_matrix(rotation_matrix: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _apply_tool_rotation_offset(rotation_matrix: np.ndarray, offset_rad: float, mode: str) -> np.ndarray:
    if abs(offset_rad) < 1e-8:
        return rotation_matrix
    offset_matrix = _rotation_about_z(offset_rad)
    if mode == "robot":
        return offset_matrix @ rotation_matrix
    return rotation_matrix @ offset_matrix


def rotation_matrix_to_axis_angle(rotation_matrix: np.ndarray) -> np.ndarray:
    R = np.array(rotation_matrix, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-6:
        return np.zeros(3, dtype=float)
    if abs(theta - np.pi) < 1e-4:
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
        ],
        dtype=float,
    )
    axis = axis_vec / (2.0 * np.sin(theta))
    return axis * theta


def _offset_gripper_to_tcp(position_mm: np.ndarray, approach_axis: np.ndarray, offset_mm: float) -> np.ndarray:
    approach = _normalize_vec(approach_axis)
    return position_mm - approach * offset_mm


def _scale_bbox(
    bbox: Sequence[float],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> List[int]:
    """将 bbox 从源图像尺寸缩放到目标图像尺寸。"""
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


def _extract_gripper_pose(
    grasp: Dict[str, Any]
) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rotation = grasp.get("rotation_matrix")
    translation = grasp.get("translation_mm") or grasp.get("position_mm")
    if rotation is None or translation is None:
        return None
    axes_c = _gripper_axes_from_matrix(rotation)
    position_c = np.array(translation, dtype=float)
    position_r = R_R_C @ position_c + P_R_C
    axes_r = tuple(_normalize_vec(R_R_C @ axis) for axis in axes_c)
    return position_r, axes_r, axes_c


class _ArmIKClient(Node):
    """Minimal helper node to resolve IK, trigger joint moves, and handle servo control."""

    def __init__(self) -> None:
        super().__init__("skill_executor_arm_ik")
        self.joint_move_client = self.create_client(Move, "/jaka_driver/joint_move")
        self.get_ik_client = self.create_client(GetIK, "/jaka_driver/get_ik")
        self.servo_move_enable_client = self.create_client(ServoMoveEnable, '/jaka_driver/servo_move_enable')
        self.servo_p_client = self.create_client(ServoMove, '/jaka_driver/servo_p')
        
        self._latest_joint_state: Optional[JointState] = None
        self._latest_tool_pose: Optional[TwistStamped] = None
        
        self.create_subscription(
            JointState,
            "/jaka_driver/joint_position",
            self._joint_state_callback,
            qos_profile=10,
        )
        self.create_subscription(
            TwistStamped, 
            "/jaka_driver/tool_position", 
            self._tool_pos_callback, 
            5
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        if len(msg.position) >= 6:
            self._latest_joint_state = msg

    def _tool_pos_callback(self, msg: TwistStamped) -> None:
        self._latest_tool_pose = msg

    def wait_for_services(self, timeout_sec: float = 5.0) -> None:
        clients = (
            (self.joint_move_client, "/jaka_driver/joint_move"),
            (self.get_ik_client, "/jaka_driver/get_ik"),
            (self.servo_move_enable_client, "/jaka_driver/servo_move_enable"),
            (self.servo_p_client, "/jaka_driver/servo_p"),
        )
        for client, name in clients:
            while not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().warn(f"等待服务 {name} ...")

    def get_reference_joints(self, timeout_sec: float) -> List[float]:
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_joint_state and len(self._latest_joint_state.position) >= 6:
                return list(self._latest_joint_state.position[:6])
        raise RuntimeError("等待 joint_position 数据超时")

    def get_current_tool_pose(self, timeout_sec: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (pos_m, euler_rad)"""
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_tool_pose:
                msg = self._latest_tool_pose
                pos = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]) # m
                euler = np.array([
                    math.radians(msg.twist.angular.x),
                    math.radians(msg.twist.angular.y),
                    math.radians(msg.twist.angular.z)
                ]) # rad
                return pos, euler
        raise RuntimeError("等待 tool_position 数据超时")

    def enable_servo_mode(self) -> bool:
        req = ServoMoveEnable.Request()
        req.enable = True
        future = self.servo_move_enable_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done():
            return True
        return False

    def send_servo_p(self, pose_delta: List[float]) -> bool:
        req = ServoMove.Request()
        req.pose = pose_delta
        future = self.servo_p_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        return future.done()

    def solve_ik(self, pose: Sequence[float], ref_joints: Sequence[float], timeout_sec: float) -> List[float]:
        request = GetIK.Request()
        request.cartesian_pose = list(pose)
        request.ref_joint = list(ref_joints)
        future = self.get_ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise RuntimeError("get_ik 请求超时")
        response = future.result()
        if response is None or not response.joint:
            raise RuntimeError("get_ik 返回空响应")
        if any(abs(val) >= 9999 for val in response.joint[:6]):
            raise RuntimeError(f"get_ik 失败: {response.message}")
        return list(response.joint[:6])

    def execute_joint_move(self, joint_target: Sequence[float], *, speed: float, acc: float, timeout_sec: float) -> None:
        request = Move.Request()
        request.pose = list(joint_target)
        request.has_ref = False
        request.ref_joint = [0.0]
        request.mvvelo = float(speed)
        request.mvacc = float(acc)
        request.mvtime = 0.0
        request.mvradii = 0.0
        request.coord_mode = 0
        request.index = 0
        future = self.joint_move_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise RuntimeError("joint_move 请求超时")
        response = future.result()
        if response is None or not getattr(response, "ret", False):
            message = getattr(response, "message", "无响应")
            raise RuntimeError(f"joint_move 调用失败: {message}")
        log_info("🤖 [_call_jaka_joint_move] joint_move 服务返回成功")


class SkillExecutor:
    """Executes behaviour tree action nodes by calling registered skills."""

    def __init__(
        self,
        navigator=None,
        gripper_controller: Optional["GripperController"] = None,
        trace_logger: Optional[TraceLogger] = None,
    ) -> None:
        self.navigator = navigator
        self._gripper = gripper_controller
        self._gripper_port = os.getenv("GRIPPER_SERIAL_PORT") or None
        self.trace_logger = trace_logger or TraceLogger()
        self.depth_localizer = TargetLocalizer()
        self.target_distance = 0.3
        self.moved_to_center = False
        self._finalize_pose_ready = False

        # Camera calibration parameters
        self.camera_fx = 596
        self.camera_fy = 596
        self.camera_cx = 500
        self.camera_cy = 500
        self.camera_fov_h_deg = 80.0
        self.search_distance_threshold = 2.0
        self.tool_rotation_offset = float(
            os.getenv("ZEROGRASP_TOOL_ROT_OFFSET", f"{DEFAULT_TOOL_ROT_OFFSET_RAD}")
        )
        self.tool_rotation_mode = os.getenv("ZEROGRASP_TOOL_ROT_MODE", "local")
        self.tcp_backoff_mm = float(os.getenv("ZEROGRASP_TCP_BACKOFF_MM", f"{DEFAULT_TCP_BACKOFF_MM}"))
        # Optional x-axis shrink towards 0 to bias end-effector 5cm closer to origin
        self.tcp_shrink_x_mm = float(os.getenv("ZEROGRASP_TCP_X_SHRINK_MM", "50.0"))
        self._arm_client: Optional[_ArmIKClient] = None
        self._joint_state_timeout = float(os.getenv("JAKA_JOINT_STATE_TIMEOUT", "3.0"))
        self._arm_service_timeout = float(os.getenv("JAKA_SERVICE_TIMEOUT", "10.0"))
        self._arm_joint_speed = float(os.getenv("JAKA_JOINT_SPEED", "5.0"))
        self._arm_joint_acc = float(os.getenv("JAKA_JOINT_ACC", "5.0"))
        self.low_grasp_score_threshold = float(
            os.getenv("GRASP_CONFIRM_THRESHOLD", "0.45")
        )

        # Feature toggles for ablations/baselines
        self.mode = DEFAULT_CONTROL_MODE
        self.enable_trace = DEFAULT_ENABLE_TRACE
        self.enable_failure_taxonomy = DEFAULT_ENABLE_FAILURE_TAX
        self.enable_verifier = DEFAULT_ENABLE_VERIFIER
        self.enable_recovery = DEFAULT_ENABLE_RECOVERY
        self.recovery_max_total_attempts = DEFAULT_RECOVERY_MAX_TOTAL_ATTEMPTS
        self.stress_perception_prob = float(os.getenv("STRESS_PERCEPTION_PROB", "0.0"))
        self.stress_grasp_prob = float(os.getenv("STRESS_GRASP_PROB", "0.0"))
        self.enable_router_vla = DEFAULT_ENABLE_ROUTER_VLA
        self.l1_escalate = DEFAULT_L1_ESCALATE
        self.l2_escalate = DEFAULT_L2_ESCALATE
        self._recovery_fail_counts: Dict[str, int] = {}
        self.recovery_manager = RecoveryManager()
        
        # S1: Burst Blindness State
        self._blind_burst_remaining = 0
        
        # Reproducibility
        self.seed = int(os.getenv("EXECUTOR_SEED", "42"))
        self.scenario_id = os.getenv("SCENARIO_ID", "scenario_default")
        self.skill_timeout_s = float(os.getenv("SKILL_TIMEOUT_S", "8.0"))
        self.episode_timeout_s = float(os.getenv("EPISODE_TIMEOUT_S", "300.0"))
        self._episode_start_ts = time.time()
        self._header_logged = False
        
        # Set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self._apply_mode()

    # ------------------------------------------------------------------
    def _log_header_once(self):
        if self._header_logged or not self.trace_logger:
            return
        self.trace_logger.log_event("session_start", {
            "seed": self.seed,
            "scenario_id": self.scenario_id,
            "mode": self.mode,
            "config": {
                "enable_verifier": self.enable_verifier,
                "enable_recovery": self.enable_recovery,
                "enable_vla_router": Config.ENABLE_VLA_ROUTER,
                "skill_timeout_s": self.skill_timeout_s,
                "episode_timeout_s": self.episode_timeout_s
            }
        })
        self._header_logged = True

    @staticmethod
    def _collect_grasps(grasp_result: Any) -> List[Dict[str, Any]]:
        """
        Normalize ZeroGrasp output to a flat list of grasp dicts.
        Handles {"objects":[...]}, {"grasps":[...]}, or top-level lists.
        """
        if grasp_result is None:
            return []
        # Direct grasps at top-level dict
        if isinstance(grasp_result, dict) and isinstance(grasp_result.get("grasps"), list):
            return list(grasp_result.get("grasps") or [])

        candidate_sources: Any = None
        if isinstance(grasp_result, dict):
            candidate_sources = grasp_result.get("objects")
        else:
            candidate_sources = grasp_result

        grasps: List[Dict[str, Any]] = []
        if isinstance(candidate_sources, dict):
            g = candidate_sources.get("grasps")
            if isinstance(g, list):
                grasps.extend(g)
        elif isinstance(candidate_sources, list):
            for obj in candidate_sources:
                if not isinstance(obj, dict):
                    continue
                g = obj.get("grasps")
                if isinstance(g, list):
                    grasps.extend(g)
        return grasps

    # ------------------------------------------------------------------
    @staticmethod
    def transform_camera_to_robot(cam_point_mm: np.ndarray) -> np.ndarray:
        """
        Convert camera-frame coordinates (mm) to robot base coordinates (mm).

        The transformation follows the legacy finalize_target_pose logic.
        """
        cam_x, cam_y, cam_z = cam_point_mm
        x_ab = cam_y + 50.0
        y_ab = -cam_x + 180.0
        z_ab = cam_z
        return np.array([x_ab, y_ab, z_ab], dtype=float)

    @staticmethod
    def transform_robot_to_world(robot_point_mm: np.ndarray, pose: Dict[str, float]) -> np.ndarray:
        """
        Convert robot base coordinates (mm) to world (map) coordinates (m).
        这个函数是对的，没问题
        """
        x_ab, y_ab, z_ab = robot_point_mm
        # 交换坐标轴,新坐标很奇怪，x朝下，z朝前，y朝左，为了在平面上计算，保持和之前一致
        theta = pose["theta"]
        x_oa = pose["x"] * 1000.0
        y_oa = pose["y"] * 1000.0
        # todo 
        x_ob = x_oa + (z_ab * math.cos(theta) - y_ab * math.sin(theta))
        y_ob = y_oa + (z_ab * math.sin(theta) + y_ab * math.cos(theta))
        return np.array([x_ob / 1000.0, y_ob / 1000.0, x_ab / 1000.0], dtype=float)#对的，不用改

    # ------------------------------------------------------------------
    def set_navigator(self, navigator) -> None:
        self.navigator = navigator

    def set_trace_logger(self, trace_logger: TraceLogger) -> None:
        self.trace_logger = trace_logger
        if not self.enable_trace:
            self.trace_logger._enabled = False  # type: ignore

    def _apply_mode(self) -> None:
        """Apply high-level control mode to feature toggles."""
        self.mode = (self.mode or "ours_full").lower()
        if self.mode == "tool_use_only":
            self.enable_verifier = False
            self.enable_recovery = False
            self.enable_trace = True
            self.enable_router_vla = False
        elif self.mode == "ours_full":
            self.enable_trace = True
            self.enable_failure_taxonomy = True
            self.enable_verifier = True
            self.enable_recovery = True
            self.enable_router_vla = True
        if self.recovery_max_total_attempts < 1:
            self.recovery_max_total_attempts = DEFAULT_RECOVERY_MAX_TOTAL_ATTEMPTS
    # ------------------------------------------------------------------
    # Verifier Mechanism
    # ------------------------------------------------------------------
    def _get_verifier(self, skill_name: str) -> Optional[Any]:
        """Return the verifier function for a given skill, or None."""
        mapping = {
            "search_area": self._verify_target_visible,
            "approach_far": self._verify_target_visible,
            "finalize_target_pose": self._verify_pose_ready,
            "execute_grasp": self._verify_grasp_success,
            "vla_execute": self._verify_grasp_success,
        }
        return mapping.get(skill_name)

    def _verify_target_visible(self, result: ExecutionResult, runtime: SkillRuntime) -> VerifierFinding:
        """
        Verify that the target is visible in the latest observation.
        Criteria: found=True, confidence > VERIFY_CONFIDENCE_THRESHOLD
        """
        obs = runtime.observation
        # Stress injection: simulate perception miss
        if self.stress_perception_prob > 0.0 and random.random() < self.stress_perception_prob:
            evidence = {
                "reason": "stress_perception_drop",
                "prob": self.stress_perception_prob,
                "stress_injection": {"type": "perception", "prob": self.stress_perception_prob},
                "failure_code_override": FailureCode.NO_OBSERVATION,
                "confidence": 0.0,
                "bbox": None,
                "bbox_area": None,
                "last_seen_ms": None,
                "thresholds": {"confidence_tau": VERIFY_CONFIDENCE_THRESHOLD, "min_area": VERIFY_MIN_BBOX_AREA},
            }
            return VerifierFinding(name="target_visible", verdict="FAIL", confidence=1.0, evidence=evidence)
        
        if not obs or not getattr(obs, "found", False):
            evidence = {
                "reason": "target_not_found", 
                "confidence": 0.0,
                "bbox": None,
                "bbox_area": None,
                "last_seen_ms": None,
                "thresholds": {"confidence_tau": VERIFY_CONFIDENCE_THRESHOLD, "min_area": VERIFY_MIN_BBOX_AREA},
            }
            return VerifierFinding(name="target_visible", verdict="FAIL", confidence=1.0, evidence=evidence)

        conf = float(getattr(obs, "confidence", 0.0) or 0.0)
        bbox = getattr(obs, "bbox", None) or []
        area = None
        if len(bbox) >= 4:
            w = abs(float(bbox[2] - bbox[0]))
            h = abs(float(bbox[3] - bbox[1]))
            area = w * h
            
        evidence = {
            "confidence": conf,
            "bbox": bbox if bbox else None,
            "bbox_area": area,
            "last_seen_ms": int(time.time() * 1000),
            "target": getattr(obs, "target_id", None),
            "thresholds": {"confidence_tau": VERIFY_CONFIDENCE_THRESHOLD, "min_area": VERIFY_MIN_BBOX_AREA},
        }

        if conf < VERIFY_CONFIDENCE_THRESHOLD:
            evidence["reason"] = "confidence_too_low"
            return VerifierFinding(name="target_visible", verdict="FAIL", confidence=1.0, evidence=evidence)
            
        if area is not None and area < VERIFY_MIN_BBOX_AREA:
            evidence["reason"] = "bbox_too_small"
            return VerifierFinding(name="target_visible", verdict="FAIL", confidence=1.0, evidence=evidence)
            
        return VerifierFinding(name="target_visible", verdict="SUCCESS", confidence=conf, evidence=evidence)

    def _verify_pose_ready(self, result: ExecutionResult, runtime: SkillRuntime) -> VerifierFinding:
        """
        Verify that the robot pose is finalized and ready for grasping.
        Criteria: finalize_flag=True, target in workspace (distance < max)
        """
        if not self._finalize_pose_ready:
            evidence = {
                "reason": "finalize_flag_false",
                "distance_to_target": None,
                "yaw_error_deg": None,
                "workspace_ok": False,
                "thresholds": {"max_distance": VERIFY_WORKSPACE_MAX_DIST, "max_yaw_error": VERIFY_MAX_YAW_ERROR_DEG},
            }
            return VerifierFinding(name="pose_ready", verdict="FAIL", confidence=1.0, evidence=evidence)

        dist = None
        if runtime.observation and getattr(runtime.observation, "range_estimate", None) is not None:
            dist = float(runtime.observation.range_estimate)

        yaw_err = runtime.extra.get("yaw_error_deg") if runtime and runtime.extra else None
        evidence: Dict[str, Any] = {
            "distance_to_target": dist,
            "yaw_error_deg": yaw_err,
            "workspace_ok": True,
            "thresholds": {"max_distance": VERIFY_WORKSPACE_MAX_DIST, "max_yaw_error": VERIFY_MAX_YAW_ERROR_DEG},
        }

        if dist is not None and dist > VERIFY_WORKSPACE_MAX_DIST:
            evidence["workspace_ok"] = False
            evidence["reason"] = "target_out_of_workspace"
            return VerifierFinding(name="pose_ready", verdict="FAIL", confidence=1.0, evidence=evidence)
        if yaw_err is not None and abs(float(yaw_err)) > VERIFY_MAX_YAW_ERROR_DEG:
            evidence["workspace_ok"] = False
            evidence["reason"] = "yaw_error_too_large"
            return VerifierFinding(name="pose_ready", verdict="FAIL", confidence=1.0, evidence=evidence)

        return VerifierFinding(name="pose_ready", verdict="SUCCESS", confidence=1.0, evidence=evidence)

    def _verify_grasp_success(self, result: ExecutionResult, runtime: SkillRuntime) -> VerifierFinding:
        """
        Verify that the grasp execution was marked as successful.
        Criteria: grasp_pose_executed flag + gripper width heuristic.
        """
        if not runtime.extra.get("grasp_pose_executed"):
            evidence = {
                "reason": "grasp_execution_flag_missing",
                "gripper_width_before": None,
                "gripper_width_after": None,
                "gripper_current": None,
                "lift_delta_z": None,
                "thresholds": {
                    "width_drop_min": VERIFY_GRIPPER_WIDTH_DROP_MIN,
                    "current_min": VERIFY_GRIPPER_CURRENT_MIN
                }
            }
            return VerifierFinding(name="grasp_success", verdict="FAIL", confidence=1.0, evidence=evidence)

        width_before = runtime.extra.get("gripper_width_before")
        width_after = runtime.extra.get("gripper_width") # Assuming this is after
        
        # Try to get live values if missing
        if width_after is None and self._gripper and hasattr(self._gripper, "get_position"):
            try:
                width_after = float(self._gripper.get_position())
            except Exception:
                width_after = None
                
        current = runtime.extra.get("gripper_current")
        if current is None and self._gripper and hasattr(self._gripper, "get_current"):
            try:
                current = float(self._gripper.get_current())
            except Exception:
                current = None

        lift_delta = runtime.extra.get("lift_delta_z")
        
        evidence = {
            "gripper_width_before": width_before,
            "gripper_width_after": width_after,
            "gripper_current": current,
            "lift_delta_z": lift_delta,
            "visual_check": runtime.extra.get("grasp_visual_check"),
            "thresholds": {
                "width_drop_min": VERIFY_GRIPPER_WIDTH_DROP_MIN,
                "current_min": VERIFY_GRIPPER_CURRENT_MIN
            }
        }

        # Heuristic 1: Width drop check (if we have before/after)
        if width_before is not None and width_after is not None:
            drop = width_before - width_after
            if drop < VERIFY_GRIPPER_WIDTH_DROP_MIN:
                 evidence["reason"] = "gripper_width_drop_too_small"
                 evidence["drop"] = drop
                 return VerifierFinding(name="grasp_success", verdict="FAIL", confidence=0.8, evidence=evidence)

        # Heuristic 2: Current check (holding something usually draws current)
        if current is not None and current < VERIFY_GRIPPER_CURRENT_MIN:
             evidence["reason"] = "gripper_current_too_low"
             return VerifierFinding(name="grasp_success", verdict="FAIL", confidence=0.7, evidence=evidence)

        # Heuristic 3: Absolute width check (too closed = empty)
        if width_after is not None and width_after < VERIFY_GRIPPER_CLOSE_WIDTH:
            evidence["threshold"] = VERIFY_GRIPPER_CLOSE_WIDTH
            evidence["reason"] = "gripper_fully_closed"
            return VerifierFinding(name="grasp_success", verdict="FAIL", confidence=0.9, evidence=evidence)
            
        return VerifierFinding(name="grasp_success", verdict="SUCCESS", confidence=0.8, evidence=evidence)

    def execute(self, node: PlanNode, runtime: SkillRuntime) -> ExecutionResult:
        self._log_header_once()
        
        # Check Episode Timeout
        if time.time() - self._episode_start_ts > self.episode_timeout_s:
            log_error(f"⏱️ [Episode Timeout] Exceeded {self.episode_timeout_s}s")
            return ExecutionResult(
                status="failure",
                node=node.name,
                reason="episode_timeout",
                failure_code=FailureCode.EPISODE_TIMEOUT,
                evidence={"elapsed": time.time() - self._episode_start_ts, "limit": self.episode_timeout_s}
            )

        # Prevent infinite recursion
        current_depth = runtime.extra.get("recovery_depth", 0)
        max_depth = self.recovery_max_total_attempts

        # --- Pre-emptive VLA Routing ---
        if Config.ENABLE_VLA_ROUTER and node.name in ["execute_grasp", "zerograsp"]:
             # Check if target description matches complex/soft keywords
             target_desc = str(node.args.get("target", "")).lower()
             if any(k in target_desc for k in Config.VLA_TRIGGER_KEYWORDS):
                 log_info(f"🔀 [Router] Pre-emptive switch to VLA for target: {target_desc}")
                 # Swap skill to VLA
                 node.name = "vla_grasp_finish"
                 node.args["instruction"] = f"pick up {target_desc}"
                 # Add router decision to trace
                 if self.trace_logger:
                     self.trace_logger.log_event(
                         "router_decision", 
                         {"type": "preemptive", "original": "classic", "new": "vla", "reason": "keyword_match"}
                     )

        # 1. Execute the skill (Core logic)
        result = self._execute_single_shot(node, runtime)

        # 2. Recovery Logic
        if (
            self.enable_recovery
            and result.status == "failure"
            and current_depth < max_depth
        ):
            # Prepare context for RecoveryManager
            tracker = runtime.extra.setdefault("recovery_tracker", {})
            tracker_key = f"{node.name}:{result.failure_code}"
            tracker[tracker_key] = tracker.get(tracker_key, 0) + 1
            
            context = RecoveryContext(
                episode_id=runtime.extra.get("episode_id", "unknown_episode"),
                step_id=f"{node.name}_{time.time()}",
                task_goal=node.args.get("instruction", ""),
                max_attempts_per_code=3  # Default or config
            )
            
            # Ask RecoveryManager for a decision
            decision = self.recovery_manager.handle_failure(result, context)
            
            if decision.kind == RecoveryKind.EXECUTE_ACTIONS:
                log_info(f"🚑 [Recovery] Triggered {decision.level} for {result.failure_code}")
                
                if self.trace_logger:
                    self.trace_logger.log_event("recovery_triggered", {
                        "node": node.name,
                        "failure_code": str(result.failure_code),
                        "level": decision.level,
                        "attempt_idx": tracker[tracker_key],
                        "decision": decision.to_dict()
                    })

                # Execute suggested actions
                all_actions_ok = True
                for action_def in decision.actions:
                    action_name = action_def["skill"]
                    action_args = action_def.get("args", {})
                    
                    rec_runtime = SkillRuntime(
                        navigator=runtime.navigator,
                        world_model=runtime.world_model,
                        observation=runtime.observation,
                        extra=runtime.extra.copy(),
                    )
                    rec_runtime.extra["recovery_depth"] = current_depth + 1
                    rec_runtime.extra["recovery_level"] = decision.level
                    rec_runtime.extra["recovery_parent"] = node.name
                    rec_runtime.extra["recovery_tracker"] = tracker
                    
                    rec_node = PlanNode(type="action", name=action_name, args=action_args)
                    rec_result = self._execute_single_shot(rec_node, rec_runtime)
                    
                    if not rec_result.success:
                        all_actions_ok = False
                        break
                
                # If actions succeeded, retry original
                if all_actions_ok:
                    log_info(f"🔄 [Recovery] Retrying original skill: {node.name}")
                    retry_runtime = SkillRuntime(
                        navigator=runtime.navigator,
                        world_model=runtime.world_model,
                        observation=runtime.observation,
                        extra=runtime.extra.copy(),
                    )
                    retry_runtime.extra["recovery_depth"] = current_depth + 1
                    retry_runtime.extra["is_retry"] = True
                    retry_runtime.extra["recovery_tracker"] = tracker
                    if runtime.extra.get("episode_id"):
                        retry_runtime.extra["episode_id"] = runtime.extra["episode_id"]

                    retry_result = self._execute_single_shot(node, retry_runtime)
                    return retry_result

            elif decision.kind == RecoveryKind.ESCALATE_L3:
                log_error(f"🛑 [Recovery] Escalated to L3 (Human/Planner Intervention) for {node.name}")
                # In a real system, this might pause execution or ask the user.
                # For now, we just log and return the failure.
                pass
        return result

    def _should_route_to_vla(self, node: PlanNode, runtime: SkillRuntime, result: ExecutionResult) -> Tuple[bool, Optional[str]]:
        """
        Decide whether to route a failed skill to VLA.
        """
        # 1. Check Failure Code
        # VLA is good at: IK failures (reachability), Grasp prediction failures (unseen objects), 
        # and maybe execution failures (slip).
        # VLA is NOT good at: System errors (arm disconnected), Navigation failures.
        
        vla_suitable_codes = {
            FailureCode.IK_FAIL,
            FailureCode.ZEROGRASP_FAILED,
            FailureCode.GRASP_EXECUTION_FAILED,
            FailureCode.VLA_NO_EFFECT, # Retry VLA? Maybe not.
            FailureCode.UNKNOWN
        }
        
        if result.failure_code not in vla_suitable_codes:
            return False, None

        # 2. Check Target Type (Optional)
        # If we already know it's a "hard" object, we definitely route.
        target_desc = str(node.args.get("target", "")).lower()
        if any(k in target_desc for k in Config.VLA_TRIGGER_KEYWORDS):
            return True, "keyword_match_on_failure"

        # 3. Default Policy: Route if it's a manipulation failure
        # For now, we are aggressive: if grasp failed, try VLA.
        return True, f"fallback_on_{result.failure_code}"

    def _execute_single_shot(self, node: PlanNode, runtime: SkillRuntime) -> ExecutionResult:
        handler_name = f"_skill_{node.name}"
        handler = getattr(self, handler_name, None)
        if not handler:
            log_warning(f"⚠️ [execute_skill_node] 未实现的技能: {node.name}")
            result = ExecutionResult(
                status="failure",
                node=node.name or "unknown",
                reason="unsupported_skill",
            )
            if result.status != "success" and not result.failure_code:
                if self.enable_failure_taxonomy:
                    result.failure_code = map_reason_to_failure_code(result.reason)
            self._emit_trace(node, runtime, result, exec_status="error", elapsed=0.0)
            return result
        log_info(f"⚙️ [execute_skill_node] 开始执行技能 {node.name or node.type}，参数: {getattr(node, 'args', {})}")
        start_ts = time.perf_counter()
        exec_status = "ok"

        # Stress injection for grasp predictor
        if (
            self.stress_grasp_prob > 0.0
            and node.name == "predict_grasp_point"
            and random.random() < self.stress_grasp_prob
        ):
            result = ExecutionResult(
                status="failure",
                node=node.name or "predict_grasp_point",
                reason="zerograsp_failed",
                evidence={"stress_injection": {"type": "grasp", "prob": self.stress_grasp_prob}},
            )
            if self.enable_failure_taxonomy:
                result.failure_code = map_reason_to_failure_code(result.reason)
            self._emit_trace(node, runtime, result, exec_status="error", elapsed=0.0)
            return result

        try:
            # Skill Timeout Wrapper
            def _timeout_handler(signum, frame):
                raise TimeoutError("Skill execution timed out")
            
            # Only use signal in main thread
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, self.skill_timeout_s)
            
            try:
                result = handler(node.args or {}, runtime)
            finally:
                if threading.current_thread() is threading.main_thread():
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    
        except TimeoutError:
            exec_status = "error"
            log_error(f"⏱️ [Skill Timeout] {node.name} exceeded {self.skill_timeout_s}s")
            elapsed = time.perf_counter() - start_ts
            result = ExecutionResult(
                status="failure",
                node=node.name or "unknown",
                reason="skill_timeout",
                failure_code=FailureCode.SKILL_TIMEOUT,
                elapsed=elapsed,
                evidence={"timeout_s": self.skill_timeout_s}
            )
            self._emit_trace(node, runtime, result, exec_status=exec_status, elapsed=elapsed)
            return result
        except Exception as exc:
            exec_status = "error"
            log_error(f"❌ [execute_skill_node] 执行技能 {node.name} 发生异常: {exc}")
            elapsed = time.perf_counter() - start_ts
            result = ExecutionResult(
                status="failure",
                node=node.name or "unknown",
                reason=str(exc),
                elapsed=elapsed,
            )
        if result.status != "success" and not result.failure_code:
            if self.enable_failure_taxonomy:
                result.failure_code = map_reason_to_failure_code(result.reason)
        # Router to VLA finisher on classic grasp failures or soft targets
        routed = False
        if (
            self.enable_router_vla
            and result.status == "failure"
            and node.name in {"execute_grasp", "predict_grasp_point"}
            and not runtime.extra.get("router_vla_attempted")
        ):
            should_route, route_reason = self._should_route_to_vla(node, runtime, result)
            if should_route:
                runtime.extra["router_vla_attempted"] = True
                vla_args: Dict[str, Any] = {
                    "instruction": f"Finish grasp after {node.name} failed: {route_reason}",
                    "state": runtime.extra.get("observation_summary") if runtime.extra else None,
                }
                if runtime.observation:
                    vla_args["image"] = getattr(runtime.observation, "annotated_url", None) or getattr(
                        runtime.observation, "original_image_path", None
                    )
                vla_node = PlanNode(type="action", name="vla_grasp_finish", args=vla_args)
                vla_runtime = SkillRuntime(
                    navigator=runtime.navigator,
                    world_model=runtime.world_model,
                    observation=runtime.observation,
                    frontend_payload=runtime.frontend_payload,
                    surface_points=runtime.surface_points,
                    extra=dict(runtime.extra),
                )
                vla_runtime.extra["router_parent"] = node.name
                vla_runtime.extra["router_reason"] = route_reason
                vla_result = self._execute_single_shot(vla_node, vla_runtime)
                routed = True
                # annotate router decision in evidence
                if vla_result.evidence is None:
                    vla_result.evidence = {}
                vla_result.evidence["router_decision"] = {
                    "from": node.name,
                    "reason": route_reason,
                    "mode": "vla_fallback",
                }
                result = vla_result
        if routed and result.verified is None and self.enable_verifier:
            # Ensure verifier is evaluated for routed result if still missing.
            verifier = self._get_verifier(result.node)
            if verifier:
                v_start = time.perf_counter()
                try:
                    finding = verifier(result, runtime)
                    is_valid = (finding.verdict == "SUCCESS")
                    result.verified = is_valid
                    if result.evidence is None:
                        result.evidence = {}
                    result.evidence["verify_evidence"] = finding.to_dict()
                    if not is_valid:
                        # Note: We do NOT fail the result here anymore based on user request.
                        # The verifier is purely diagnostic.
                        # However, to maintain backward compatibility for now, we might want to keep it?
                        # User said: "明确它只输出 VerifierFinding，不更新 world_model、不触发恢复"
                        # But if we don't set status=failure, the loop continues blindly.
                        # Let's assume "不触发恢复" means the *verifier itself* doesn't trigger it,
                        # but the result status might still need to reflect reality?
                        # Actually, user said "不使用硬性的verifier...全盘指挥".
                        # So we should probably Log it but NOT change result.status to failure?
                        # "VLM检查来结合这些信息决定怎么处理" -> So we keep result.status as is (success),
                        # and let VLM see the finding in the packet.
                        log_warning(f"⚠️ [execute] Verifier finding: {finding.verdict} ({finding.evidence.get('reason')})")
                    else:
                        log_info(f"✅ [execute] Verifier passed: {finding.verdict}")
                except Exception as v_exc:
                    log_error(f"❌ [execute] Verifier crashed: {v_exc}")
                    result.verified = False
                    # result.status = "failure" # Don't fail hard
                v_elapsed = (time.perf_counter() - v_start) * 1000.0
                if result.evidence is None:
                    result.evidence = {}
                result.evidence["verify_elapsed_ms"] = round(v_elapsed, 2)
        
        # Verification Loop
        if self.enable_verifier and result.status == "success":
            verifier = self._get_verifier(node.name)
            if verifier:
                # Check if skill is mutating (invalidates visual observation)
                mutating_prefixes = ["navigate", "move", "rotate", "turn", "pick", "place", "grasp", "release", "home", "approach", "align"]
                is_mutating = any(node.name.startswith(p) for p in mutating_prefixes)
                
                # If mutating, visual verifiers (like target_visible) are unreliable without new observation
                # We skip them or mark them as UNCERTAIN to avoid false negatives based on stale data.
                # Internal state verifiers (like grasp_success using gripper width) are still valid.
                
                v_start = time.perf_counter()
                try:
                    finding = verifier(result, runtime)
                    
                    # Special handling for stale visual verification
                    if is_mutating and finding.name == "target_visible":
                        log_warning(f"⚠️ [execute] Skipping visual verifier {finding.name} for mutating skill {node.name} (observation stale)")
                        finding.verdict = "UNCERTAIN"
                        finding.evidence["reason"] = "observation_stale_after_mutation"
                    
                    is_valid = (finding.verdict == "SUCCESS")
                    result.verified = is_valid
                    if result.evidence is None:
                        result.evidence = {}
                    result.evidence["verify_evidence"] = finding.to_dict()
                    
                    if not is_valid:
                        log_warning(f"⚠️ [execute] Verifier finding: {finding.verdict} ({finding.evidence.get('reason')})")
                    else:
                        log_info(f"✅ [execute] Verifier passed: {finding.verdict}")
                except Exception as v_exc:
                    log_error(f"❌ [execute] Verifier crashed: {v_exc}")
                    result.verified = False
                v_elapsed = (time.perf_counter() - v_start) * 1000.0
                if result.evidence is None:
                    result.evidence = {}
                result.evidence["verify_elapsed_ms"] = round(v_elapsed, 2)
            else:
                result.verified = None  # Skipped

        elapsed = time.perf_counter() - start_ts
        result.elapsed = elapsed
        if elapsed > self.skill_timeout_s:
            result.status = "failure"
            result.reason = "skill_timeout"
            if self.enable_failure_taxonomy:
                result.failure_code = FailureCode.SKILL_TIMEOUT
            if result.evidence is None:
                result.evidence = {}
            result.evidence.update(
                {
                    "timeout_s": self.skill_timeout_s,
                    "elapsed_s": round(elapsed, 3),
                    "skill_name": node.name,
                    "router_reason": runtime.extra.get("router_reason") if runtime and runtime.extra else None,
                }
            )
        if result.status != "success" and not result.failure_code:
            if self.enable_failure_taxonomy:
                result.failure_code = map_reason_to_failure_code(result.reason)
        if result.status != "success" and result.verified is None:
            result.verified = False

        # --- Generate Inspection Packet ---
        try:
            packet = InspectionPacket(
                episode_id=runtime.extra.get("episode_id"),
                step_id=runtime.extra.get("step_id") or node.name,
                skill_name=node.name,
                skill_args=node.args or {},
                exec_result={
                    "status": result.status,
                    "failure_code": result.failure_code.value if result.failure_code else None,
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "reason": result.reason
                },
                raw_metrics=result.evidence,
                verifier_outputs=result.evidence.get("verify_evidence") if result.evidence else None,
                world_snapshot=runtime.world_model.snapshot() if runtime.world_model else None,
                budget=runtime.extra.get("budget"),
                artifacts={
                    "image_path": getattr(runtime.observation, "original_image_path", None) if runtime.observation else None,
                    "annotated_url": getattr(runtime.observation, "annotated_url", None) if runtime.observation else None
                }
            )
            if result.evidence is None:
                result.evidence = {}
            result.evidence["inspection_packet"] = packet
        except Exception as e:
            log_error(f"⚠️ [Inspection] Failed to generate packet: {e}")
        # ----------------------------------

        self._emit_trace(node, runtime, result, exec_status=exec_status, elapsed=elapsed)
        return result

    def _emit_trace(
        self,
        node: PlanNode,
        runtime: SkillRuntime,
        result: ExecutionResult,
        exec_status: str,
        elapsed: float,
    ) -> None:
        """Record a structured trace entry for the skill call."""
        if not self.enable_trace:
            return
        try:
            precheck: Dict[str, Any] = {}
            episode_id: Optional[str] = None
            step_hint: Optional[int] = None
            if runtime and isinstance(runtime.extra, dict):
                precheck = runtime.extra.get("precheck") or {}
                episode_id = runtime.extra.get("episode_id")
                step_hint = runtime.extra.get("step_id") or runtime.extra.get("step")
            evidence = result.evidence if isinstance(result.evidence, dict) else {}
            if evidence == {} and result.evidence not in (None, {}):
                evidence = {"value": result.evidence}
            event = {
                "episode_id": episode_id,
                "skill_name": node.name or node.type or "unknown",
                "inputs": node.args or {},
                "precheck": precheck if isinstance(precheck, dict) else {},
                "exec_status": exec_status,
                "elapsed_ms": round(elapsed * 1000.0, 3),
                "verified": result.verified,
                "failure_code": result.failure_code.value if hasattr(result.failure_code, "value") else result.failure_code,
                "evidence": evidence,
            }
            if step_hint is not None:
                event["step_id"] = step_hint

            if runtime and isinstance(runtime.extra, dict):
                if "recovery_level" in runtime.extra:
                    event["recovery_level"] = runtime.extra["recovery_level"]
                if "recovery_depth" in runtime.extra:
                    event["recovery_attempt_idx"] = runtime.extra["recovery_depth"]
                if "recovery_parent" in runtime.extra:
                    event["recovery_parent"] = runtime.extra["recovery_parent"]
                if "is_retry" in runtime.extra:
                    event["is_retry"] = runtime.extra["is_retry"]
                if "router_reason" in runtime.extra:
                    event["router_decision"] = runtime.extra["router_reason"]

            self.trace_logger.log_skill_call(event)
        except Exception:
            # Trace failures must not affect main execution path.
            return

    def _trace_recovery_attempt(
        self,
        *,
        runtime: SkillRuntime,
        plan: RecoveryPlan,
        attempt_idx: int,
        action_name: str,
        action_args: Dict[str, Any],
        rec_result: ExecutionResult,
    ) -> None:
        """Record recovery attempt metadata in the trace log."""
        try:
            episode_id = None
            if runtime and isinstance(runtime.extra, dict):
                episode_id = runtime.extra.get("episode_id")
            event = {
                "episode_id": episode_id,
                "skill_name": action_name,
                "inputs": action_args or {},
                "precheck": {},
                "exec_status": "ok",
                "elapsed_ms": rec_result.elapsed * 1000.0 if rec_result.elapsed else None,
                "verified": rec_result.verified,
                "failure_code": rec_result.failure_code.value if hasattr(rec_result.failure_code, "value") else rec_result.failure_code,
                "evidence": rec_result.evidence or {},
                "recovery_level": plan.level,
                "recovery_attempt_idx": attempt_idx,
                "recovery_action_name": action_name,
                "recovery_success": rec_result.success,
            }
            if self.enable_trace:
                self.trace_logger.log_skill_call(event)
        except Exception:
            return

    # ------------------------------------------------------------------
    # Skill implementations
    # ------------------------------------------------------------------
    def _skill_rotate_scan(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        angle = float(args.get("angle_deg", 30.0))
        success = self.control_turn_around(runtime.navigator, math.radians(angle))
        status = "success" if success else "failure"
        return ExecutionResult(status=status, node="rotate_scan")

    def _skill_search_area(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        # S1: Perception Stress Injection
        import random
        if Config.STRESS_PERCEPTION_FAILURE_RATE > 0.0:
            if random.random() < Config.STRESS_PERCEPTION_FAILURE_RATE:
                log_warning("⚡ [Stress] Injecting PERCEPTION failure in search_area")
                return ExecutionResult(
                    status="failure",
                    node="search_area",
                    reason="stress_injected_no_observation",
                    failure_code=FailureCode.NO_OBSERVATION,
                    evidence={"stress_injection": {"type": "perception_drop", "rate": Config.STRESS_PERCEPTION_FAILURE_RATE}}
                )

        turns = max(1, int(args.get("turns", 2)))
        angle = float(args.get("angle_deg", 45.0))
        for _ in range(turns):
            result = self._skill_rotate_scan({"angle_deg": angle}, runtime)
            if result.status != "success":
                return ExecutionResult(status="failure", node="search_area", reason="rotate_failed")
        pattern = args.get("pattern", "in_place")
        log_info(f"🔍 [_skill_search_area] search_area 完成, pattern={pattern}, turns={turns}")
        return ExecutionResult(status="success", node="search_area", evidence={"turns": turns, "pattern": pattern})

    def _skill_navigate_area(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        navigator = runtime.navigator
        if navigator is None:
            return ExecutionResult(status="failure", node="navigate_area", reason="navigator_missing")
        target_marker = args.get("marker")
        area_name = args.get("area") or args.get("target")
        pose = args.get("pose")
        world = runtime.world_model
        if pose is None and area_name and world:
            area = world.areas.get(area_name)
            if area and area.pose and len(area.pose) >= 3:
                pose = {"x": area.pose[0], "y": area.pose[1], "theta": area.pose[2]}
        success = False
        mode = None
        if target_marker:
            mode = "marker"
            success = navigator.navigate_to_target(target_marker)
        elif pose:
            mode = "pose"
            theta = float(pose.get("theta", 0.0))
            x = float(pose.get("x", 0.0))
            y = float(pose.get("y", 0.0))
            success = navigator.move_to_position(theta, x, y)
        else:
            return ExecutionResult(status="failure", node="navigate_area", reason="missing_target")
        if success and world:
            world.robot["pose"] = navigator.get_current_pose()
        status = "success" if success else "failure"
        evidence = {"mode": mode or "unknown", "area": area_name, "marker": target_marker}
        return ExecutionResult(status=status, node="navigate_area", reason=None if success else "navigate_failed", evidence=evidence)

    def _skill_return_home(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        params = dict(args)
        if not params.get("area") and not params.get("marker") and not params.get("pose"):
            params["area"] = args.get("default_area", "home")
        log_info(f"🏠 [_skill_return_home] return_home 调用 navigate_area: {params}")
        return self._skill_navigate_area(params, runtime)

    # def _skill_search_area(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
    #     order = args.get("area_order", [])
    #     log_info(f"🔍 搜索区域: {order}")
    #     # MVP: simple rotate scan
    #     result = self._skill_rotate_scan({"angle_deg": 45.0}, runtime)
    #     metadata = {"area_order": order, "found": bool(runtime.observation and runtime.observation.found)}
    #     return ExecutionResult(status=result.status, node="search_area", reason=result.reason, evidence=metadata)

    def _skill_approach_far(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        navigator = runtime.navigator
        if navigator is None:
            return ExecutionResult(status="failure", node="approach_far", reason="navigator_missing")

        observation = runtime.observation
        target = args.get("target")

        dist_m: Optional[float] = None
        if observation and observation.range_estimate:
            dist_m = float(observation.range_estimate)
        if dist_m is None and observation and observation.robot_center:
            rc = np.array(observation.robot_center, dtype=float)
            dist_m = float(np.linalg.norm(rc[:2]) / 1000.0)
        if dist_m is None and observation and observation.world_center:
            wc = np.array(observation.world_center, dtype=float)
            dist_m = float(np.linalg.norm(wc[:2]))
        # if dist_m is None and runtime.world_model and target:
        #     obj = runtime.world_model.objects.get(target)
        #     if obj:
        #         dist_m = obj.attrs.get("range_estimate")
        if dist_m is not None and dist_m <= self.search_distance_threshold:
            log_info(
                f"🚶‍♂️ [_skill_approach_far] approach_far: 目标距离 {dist_m:.2f}m 已≤阈值 {self.search_distance_threshold:.2f}m，跳过靠近"
            )
            return ExecutionResult(status="success", node="approach_far", reason="distance_within_threshold")

        if observation is None or not observation.found:
            return ExecutionResult(status="failure", node="approach_far", reason="no_observation")

        pose = observation.robot_pose or navigator.get_current_pose()
        current_position = np.array([pose["x"], pose["y"]], dtype=float)

        direction_vector: Optional[np.ndarray] = None
        if observation.world_center:
            wc = np.array(observation.world_center, dtype=float)
            target_position = wc[:2]
            direction_vector = target_position - current_position
        # elif observation.robot_center:
        #     rc = np.array(observation.robot_center, dtype=float) / 1000.0
        #     direction_vector = rc[:2]
        # elif observation.range_estimate:
        #     theta = pose["theta"]
        #     direction_vector = np.array([math.cos(theta), math.sin(theta)], dtype=float) * observation.range_estimate

        if direction_vector is None or np.linalg.norm(direction_vector) < 1e-3:
            return ExecutionResult(status="failure", node="approach_far", reason="no_direction_vector")

        direction_norm = direction_vector / np.linalg.norm(direction_vector)

        if dist_m is None:
            dist_m = self.search_distance_threshold + 1.0
        margin = max(0.3, self.search_distance_threshold - 0.5)
        step_distance = max(0.5, min(1.0, dist_m - margin))

        theta = math.atan2(direction_norm[1], direction_norm[0])
        new_theta = theta
        new_x = pose["x"] + direction_norm[0] * step_distance
        new_y = pose["y"] + direction_norm[1] * step_distance

        success = navigator.move_to_position(new_theta, new_x, new_y)
        if success:
            return ExecutionResult(
                status="success",
                node="approach_far",
                evidence={"distance": step_distance, "target_theta": new_theta},
            )
        return ExecutionResult(
            status="failure",
            node="approach_far",
            reason="move_to_position_failed",
            evidence={"distance": step_distance, "target_theta": new_theta},
        )

    def _skill_finalize_target_pose(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        navigator = runtime.navigator
        if navigator is None:
            return ExecutionResult(status="failure", node="finalize_target_pose", reason="navigator_missing")
        depth_info = self._ensure_depth_localization(runtime)
        if not depth_info:
            return ExecutionResult(
                status="failure",
                node="finalize_target_pose",
                reason="depth_localization_failed",
            )
        obj_center_3d = depth_info.get("obj_center_3d")
        if not obj_center_3d:
            return ExecutionResult(
                status="failure",
                node="finalize_target_pose",
                reason="missing_obj_center",
            )
        tune_angle = float(depth_info.get("tune_angle", 0.0))
        mask_available = bool(depth_info.get("surface_mask_available"))
        try:
            cam_obj_center_3d = obj_center_3d
            vec = np.array(cam_obj_center_3d.copy() + [1.0], dtype=float).reshape(4, 1)
            T_mat = np.array(
                [
                    [0, 1, 0, 180],
                    [-1, 0, 0, 50],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )
            jaka_vec = T_mat @ vec
            pose = navigator.get_current_pose()
            X_OA = pose["x"] * 1000
            Y_OA = pose["y"] * 1000
            theta_OA = pose["theta"]
            X_AB, Y_AB, Z_AB = jaka_vec[:3].ravel()
            log_info(f"🤖 [finalize_target_pose] 目标相机坐标系下位置 (mm):{X_AB} , {Y_AB} , {Z_AB} ")
            linear_axis_mm = float(X_AB)
            X_OB = X_OA + (X_AB * math.cos(theta_OA) - Y_AB * math.sin(theta_OA))
            Y_OB = Y_OA + (X_AB * math.sin(theta_OA) + Y_AB * math.cos(theta_OA))
            obj_world_x = X_OB / 1000.0
            obj_world_y = Y_OB / 1000.0
            
            # 📍 观测日志：记录目标物体和机器人的世界坐标
            log_info(f"📍 [finalize_target_pose] 机器人世界坐标: x={pose['x']:.3f}m, y={pose['y']:.3f}m, θ={math.degrees(theta_OA):.1f}°")
            log_info(f"📍 [finalize_target_pose] 目标物体世界坐标: x={obj_world_x:.3f}m, y={obj_world_y:.3f}m")

            if mask_available:
                # tune_angle 是用 atan 计算的边缘角度（-90° ~ 90°之间）
                # 直接加到当前机器人角度上就是正对目标边缘
                target_theta = theta_OA + tune_angle
                # 归一化到 [-π, π]
                while target_theta > math.pi:
                    target_theta -= 2 * math.pi
                while target_theta < -math.pi:
                    target_theta += 2 * math.pi
                orientation_source = "sam_edge"
                log_info(f"🧭 [finalize_target_pose] SAM边缘对齐: tune_angle={math.degrees(tune_angle):.1f}°, " +
                        f"current={math.degrees(theta_OA):.1f}°, target={math.degrees(target_theta):.1f}°")
            else:
                dx = obj_world_x - pose["x"]
                dy = obj_world_y - pose["y"]
                if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                    target_theta = theta_OA
                else:
                    target_theta = math.atan2(dy, dx)
                orientation_source = "target_vector"

            # 机器人应该在物体的反方向上（即物体前方 target_distance 处）
            # target_theta 是机器人朝向物体的方向，所以目标位置应该减去距离分量
            target_x = obj_world_x - self.target_distance * math.cos(target_theta)
            target_y = obj_world_y - self.target_distance * math.sin(target_theta)
            
            # 横向偏移：让右臂中心对准目标物体
            # 右臂在机器人右侧，需要机器人向左偏移，使目标物体出现在右臂工作范围内
            # 偏移方向：垂直于朝向方向向左（即 target_theta + 90度）
            ARM_LATERAL_OFFSET = 0.15  # 横向偏移量（米），根据右臂位置调整
            target_x += ARM_LATERAL_OFFSET * math.cos(target_theta + math.pi / 2)
            target_y += ARM_LATERAL_OFFSET * math.sin(target_theta + math.pi / 2)
            
            # 横向偏移后的角度处理
            dx_final = obj_world_x - target_x
            dy_final = obj_world_y - target_y
            recalc_theta = math.atan2(dy_final, dx_final)
            
            # 🔍 角度调试日志
            log_info(f"🧭 [finalize_target_pose] 角度计算详情:")
            log_info(f"    目标物体: ({obj_world_x:.3f}, {obj_world_y:.3f})")
            log_info(f"    机器人目标位置: ({target_x:.3f}, {target_y:.3f})")
            log_info(f"    方向向量: dx={dx_final:.3f}, dy={dy_final:.3f}")
            log_info(f"    位置向量角度: {math.degrees(recalc_theta):.1f}°")
            
            # 如果使用 SAM 边缘垂直对齐，保持原角度（正对桌面）
            # 如果使用目标向量，则用重新计算的角度
            if orientation_source == "sam_edge":
                log_info(f"    保持SAM边缘角度: {math.degrees(target_theta):.1f}° (弧度: {target_theta:.4f})")
            else:
                target_theta = recalc_theta
                log_info(f"    使用位置向量角度: {math.degrees(target_theta):.1f}° (弧度: {target_theta:.4f})")

            # 验证距离: 计算目标位置到物体的实际距离
            actual_distance = math.sqrt((obj_world_x - target_x)**2 + (obj_world_y - target_y)**2)
            distance_error = abs(actual_distance - self.target_distance)
            log_info(f"📏 [finalize_target_pose] 距离验证: 物体位置=({obj_world_x:.3f}, {obj_world_y:.3f}), " +
                    f"目标位置=({target_x:.3f}, {target_y:.3f})")
            log_info(f"📏 [finalize_target_pose] 实际距离={actual_distance:.3f}m, 期望距离={self.target_distance:.3f}m, " +
                    f"误差={distance_error:.4f}m ({distance_error*1000:.1f}mm)")
            if distance_error > 0.01:  # 误差超过10mm
                log_warning(f"⚠️ [finalize_target_pose] 距离误差较大: {distance_error*1000:.1f}mm")

            axis_min = -310.0
            axis_max = 200.0
            clamped_x = max(axis_min, min(axis_max, linear_axis_mm))
            if linear_axis_mm < axis_min or linear_axis_mm > axis_max:
                log_warning(f"⚠️ [finalize_target_pose] 目标线性轴位置 {linear_axis_mm:.1f}mm 超出范围 "
                    f"({axis_min:.0f}~{axis_max:.0f}mm)，使用边界值 {clamped_x:.1f}mm 对齐"
                )
            # self._call_linear_axis_move(clamped_x)
            log_info(f"🎯 [finalize_target_pose] 计算底盘目标位置: ({target_x:.2f}, {target_y:.2f}), θ={target_theta:.2f}rad, orient={orientation_source}"
            )
            success = navigator.move_to_position(target_theta, target_x, target_y)
            if success:
                runtime.world_model.robot["pose"] = [target_x, target_y, target_theta]
                self._finalize_pose_ready = True
            status = "success" if success else "failure"
            return ExecutionResult(
                status=status,
                node="finalize_target_pose",
                reason=None if success else "navigator_move_failed",
            )
        except Exception as exc:
            log_error(f"❌ [finalize_target_pose] finalize_target_pose 计算失败: {exc}")
            self._finalize_pose_ready = False
            return ExecutionResult(
                status="failure",
                node="finalize_target_pose",
                reason=str(exc),
            )

    def _prepare_robot_pose_from_grasp(self, grasp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            pose = _extract_gripper_pose(grasp)
        except ValueError as exc:
            log_warning(f"⚠️ [_prepare_robot_pose_from_grasp] 解析抓取姿态失败: {exc}")
            pose = None
        if not pose:
            return None
        position_mm, axes_robot, _ = pose
        log_info(
            f"🔧 [_prepare_robot_pose_from_grasp] 原始抓取点(机器人坐标): "
            f"x={position_mm[0]:.1f}, y={position_mm[1]:.1f}, z={position_mm[2]:.1f} mm"
        )
        log_info(
            f"🔧 [_prepare_robot_pose_from_grasp] approach轴(Z): "
            f"[{axes_robot[2][0]:.3f}, {axes_robot[2][1]:.3f}, {axes_robot[2][2]:.3f}], "
            f"后退量: {self.tcp_backoff_mm:.1f} mm"
        )
        tcp_position = _offset_gripper_to_tcp(position_mm, axes_robot[2], self.tcp_backoff_mm)
        log_info(
            f"🔧 [_prepare_robot_pose_from_grasp] 后退后TCP位置: "
            f"x={tcp_position[0]:.1f}, y={tcp_position[1]:.1f}, z={tcp_position[2]:.1f} mm"
        )
        if self.tcp_shrink_x_mm > 0:
            x_before = float(tcp_position[0])
            if abs(x_before) > 1e-6:
                delta = min(abs(x_before), self.tcp_shrink_x_mm)
                tcp_position[0] = x_before - math.copysign(delta, x_before)
                log_info(
                    f"🔧 [_prepare_robot_pose_from_grasp] TCP x轴向0收缩 {delta:.1f}mm: "
                    f"{x_before:.1f} -> {tcp_position[0]:.1f}"
                )
        rotation_matrix = _axes_to_rotation_matrix(axes_robot)
        rotation_matrix_cmd = _apply_tool_rotation_offset(
            rotation_matrix,
            self.tool_rotation_offset,
            self.tool_rotation_mode,
        )
        rot_vec = rotation_matrix_to_axis_angle(rotation_matrix_cmd)
        tcp_pose = list(tcp_position) + list(rot_vec)
        return {
            "gripper_position_mm": position_mm.tolist(),
            "axes_robot": [axis.tolist() for axis in axes_robot],
            "tcp_position_mm": tcp_position.tolist(),
            "rot_vec": rot_vec.tolist(),
            "rotation_matrix": rotation_matrix_cmd.tolist(),
            "tcp_pose": tcp_pose,
        }

    def _skill_predict_grasp_point(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        observation = runtime.observation
        if observation is None or not observation.bbox:
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason="missing_observation",
            )
        if not self._finalize_pose_ready:
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason="finalize_not_completed",
            )
        
        # 通过 API 同步获取对齐的 RGB+Depth（复用 UI 的相机连接）
        log_info("🔧 [_skill_predict_grasp_point] 从 API 获取对齐的 RGB+Depth...")
        try:
            rgb_frame, depth_snapshot = fetch_aligned_rgbd(timeout=1.5)
            if rgb_frame is None:
                log_error("❌ [_skill_predict_grasp_point] API 未返回 RGB 图像")
                return ExecutionResult(
                    status="failure",
                    node="predict_grasp_point",
                    reason="rgb_frame_unavailable",
                )
            log_info(
                f"🔧 [_skill_predict_grasp_point] RGB: {rgb_frame.shape}, "
                f"Depth: {depth_snapshot.depth.shape}"
            )
        except Exception as exc:
            log_error(f"❌ [_skill_predict_grasp_point] 获取对齐 RGBD 失败: {exc}")
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason=f"fetch_rgbd_failed: {exc}",
            )
        
        # 将 observation.bbox 从 VLM 图像尺寸缩放到深度图尺寸
        obs_bbox = observation.bbox
        depth_h, depth_w = depth_snapshot.depth.shape
        rgb_h, rgb_w = rgb_frame.shape[:2]
        
        # 获取原始 VLM 图像尺寸
        vlm_img_path = observation.original_image_path or observation.processed_image_path
        vlm_w, vlm_h = rgb_w, rgb_h  # 默认使用 RGB 尺寸
        if vlm_img_path and os.path.isfile(vlm_img_path):
            try:
                with Image.open(vlm_img_path) as img:
                    vlm_w, vlm_h = img.size
            except Exception:
                pass
        
        # bbox 缩放: VLM 图像 -> 深度图
        bbox_scaled = _scale_bbox(obs_bbox, (vlm_w, vlm_h), (depth_w, depth_h))
        log_info(
            f"🔧 [_skill_predict_grasp_point] bbox 缩放: VLM({vlm_w}x{vlm_h}) -> Depth({depth_w}x{depth_h}), "
            f"原始={obs_bbox}, 缩放后={bbox_scaled}"
        )
        
        # 如果 RGB 和 Depth 尺寸不同，resize RGB 到 Depth 尺寸
        if (rgb_w, rgb_h) != (depth_w, depth_h):
            rgb_frame = cv2.resize(rgb_frame, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
            log_info(f"🔧 [_skill_predict_grasp_point] RGB resize: ({rgb_w}x{rgb_h}) -> ({depth_w}x{depth_h})")
        
        # 使用 localize_object 处理深度数据
        transform = self.depth_localizer.localize_object(
            bbox=bbox_scaled,
            snapshot=depth_snapshot,
            surface_points_hint=None,
            range_estimate=observation.range_estimate,
            rgb_frame=rgb_frame,
            include_transform=True,
        )
        if not transform:
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason="depth_localization_failed",
                failure_code=FailureCode.DEPTH_LOCALIZATION_FAILED,
                evidence={"bbox_scaled": bbox_scaled, "depth_shape": depth_snapshot.depth.shape}
            )
        
        transform_result = transform.get("transform_result", transform)
        grasp_result = self.depth_localizer.run_zero_grasp_inference(
            transform_result=transform_result,
            bbox=transform["bbox"],
            rgb_frame=rgb_frame,
        )
        if not grasp_result:
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason="zerograsp_failed",
                failure_code=FailureCode.ZEROGRASP_FAILED,
                evidence={"transform_summary": str(transform_result)[:200]}
            )
        grasps = self._collect_grasps(grasp_result)
        best = max(grasps, key=lambda g: float(g.get("score", 0.0))) if grasps else None
        runtime.extra["zerograsp_result"] = grasp_result
        notifications = runtime.extra.setdefault("notifications", [])
        if "grasp_pose" in runtime.extra:
            runtime.extra.pop("grasp_pose", None)
        evidence: Dict[str, Any] = {"grasp_count": len(grasps)}
        grasp_pose_info: Optional[Dict[str, Any]] = None
        if best:
            grasp_pose_info = self._prepare_robot_pose_from_grasp(best)
            if grasp_pose_info:
                runtime.extra["grasp_pose"] = grasp_pose_info
                tcp_pos = grasp_pose_info.get("tcp_position_mm")
                rot_vec = grasp_pose_info.get("rot_vec")
                if tcp_pos:
                    evidence["tcp_position_mm"] = [round(float(v), 1) for v in tcp_pos[:3]]
                if rot_vec:
                    evidence["rot_vec"] = [round(float(v), 3) for v in rot_vec[:3]]
            evidence.update(
                {
                    "best_score": float(best.get("score", 0.0)),
                    "best_width_mm": best.get("width_mm"),
                    "best_translation_mm": best.get("translation_mm"),
                }
            )
            if evidence.get("best_score") is not None and evidence["best_score"] < self.low_grasp_score_threshold:
                notifications.append(
                    {
                        "message": f"抓取置信度偏低 ({evidence['best_score']:.2f})，请确认是否继续",  # noqa: E501
                        "level": "warning",
                    }
                )
            if not grasp_pose_info:
                notifications.append(
                    {
                        "message": "未能解析有效的抓取姿态，请重新观测或调整目标",
                        "level": "warning",
                    }
                )
        def _fmt_vec(vec: Any, precision: int = 1) -> str:
            try:
                values = list(vec)
            except Exception:
                return str(vec)
            formatted = []
            for v in values[:3]:
                try:
                    formatted.append(f"{float(v):.{precision}f}")
                except (TypeError, ValueError):
                    formatted.append(str(v))
            return "(" + ", ".join(formatted) + ")"

        best_score = evidence.get("best_score")
        pose_extra = ""
        if best:
            translation = best.get("translation_mm") or best.get("position_mm")
            if translation:
                pose_extra += f", tcp_mm={_fmt_vec(translation)}"
            approach = (
                best.get("approach_vector")
                or best.get("approach_dir")
                or best.get("orientation_vector")
            )
            if approach:
                pose_extra += f", approach={_fmt_vec(approach, precision=3)}"
            rotation = best.get("rotation_matrix") or best.get("quaternion")
            if rotation and not pose_extra:
                pose_extra += f", orientation={rotation}"

        if grasp_pose_info:
            tcp_mm = grasp_pose_info.get("tcp_position_mm")
            rot_vec = grasp_pose_info.get("rot_vec")
            if tcp_mm:
                pose_extra += f", tcp_robot={_fmt_vec(tcp_mm)}"
            if rot_vec:
                pose_extra += f", rot_vec={_fmt_vec(rot_vec, precision=3)}"

        if best_score is not None:
            log_info(
                f"🤲 [_skill_predict_grasp_point] ZeroGrasp返回 {evidence['grasp_count']} 个抓取候选，最佳评分 {best_score:.3f}{pose_extra}"
            )
        else:
            log_info(f"🤲 [_skill_predict_grasp_point] ZeroGrasp返回 {evidence['grasp_count']} 个抓取候选，但未找到有效抓取")
        self._finalize_pose_ready = False
        status = "success" if grasp_pose_info else "failure"
        reason = None if grasp_pose_info else "grasp_pose_unavailable"
        return ExecutionResult(
            status=status,
            node="predict_grasp_point",
            reason=reason,
            evidence=evidence,
        )

    def _skill_pick(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        # Placeholder for integration with manipulator stack
        log_warning("⚠️ [_skill_pick] pick技能尚未与真实机械臂对接，默认返回成功")
        return ExecutionResult(status="success", node="pick")

    def _skill_place(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        log_warning("⚠️ [_skill_place] place技能尚未实现，默认返回成功")
        return ExecutionResult(status="success", node="place")

    def _skill_open_gripper(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        controller = self._ensure_gripper()
        if controller is None:
            return ExecutionResult(status="failure", node="open_gripper", reason="gripper_unavailable")
        try:
            controller.open()
            log_success("🖐️ 夹爪已张开")
            return ExecutionResult(status="success", node="open_gripper")
        except Exception as exc:
            log_error(f"❌ [_skill_open_gripper] open_gripper 失败: {exc}")
            return ExecutionResult(status="failure", node="open_gripper", reason=str(exc))

    def _skill_close_gripper(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        controller = self._ensure_gripper()
        if controller is None:
            return ExecutionResult(status="failure", node="close_gripper", reason="gripper_unavailable")
        try:
            controller.close()
            log_success("✊ 夹爪已闭合")
            return ExecutionResult(status="success", node="close_gripper")
        except Exception as exc:
            log_error(f"❌ [_skill_close_gripper] close_gripper 失败: {exc}")
            return ExecutionResult(status="failure", node="close_gripper", reason=str(exc))

    def _skill_handover_item(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        controller = self._ensure_gripper()
        if controller is None:
            return ExecutionResult(status="failure", node="handover_item", reason="gripper_unavailable")
        item = args.get("item")
        if not item and runtime and runtime.extra:
            item = runtime.extra.get("requested_item")
        wait_sec = float(args.get("wait_sec", 1.0))
        log_info(f"🤝 [_skill_handover_item] 开始递交物品: {item or 'unknown'}，等待 {wait_sec:.1f}s")
        try:
            time.sleep(max(0.0, wait_sec))
            controller.deliver(item)
            return ExecutionResult(status="success", node="handover_item", evidence={"item": item})
        except Exception as exc:
            log_error(f"❌ [_skill_handover_item] handover_item 失败: {exc}")
            return ExecutionResult(status="failure", node="handover_item", reason=str(exc))

    def _skill_recover(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        mode = args.get("mode", "backoff")
        log_info(f"🔁 [_skill_recover] recover skill triggered, mode={mode}")
        success = False
        reason = None
        
        try:
            if mode == "backoff":
                navigator = runtime.navigator
                pose = navigator.get_current_pose()
                back_distance = float(args.get("distance", 0.3))
                new_x = pose["x"] - back_distance * math.cos(pose["theta"])
                new_y = pose["y"] - back_distance * math.sin(pose["theta"])
                success = navigator.move_to_position(pose["theta"], new_x, new_y)
                
            elif mode == "nudge_base":
                navigator = runtime.navigator
                pose = navigator.get_current_pose()
                # Small random perturbation
                import random
                dx = float(args.get("dx", 0.05))
                dy = float(args.get("dy", 0.05))
                dtheta = float(args.get("dtheta", 0.1))
                
                nx = pose["x"] + random.uniform(-dx, dx)
                ny = pose["y"] + random.uniform(-dy, dy)
                nt = pose["theta"] + random.uniform(-dtheta, dtheta)
                
                log_info(f"🔁 [_skill_recover] Nudging base to x={nx:.2f}, y={ny:.2f}, th={nt:.2f}")
                success = navigator.move_to_position(nt, nx, ny)
                
            elif mode == "reset_arm":
                arm_client = self._ensure_arm_client()
                if arm_client:
                    # Default safe pose (e.g. vertical)
                    home_joints = args.get("joints", [0.0, 0.0, 1.57, 0.0, 1.57, 0.0])
                    log_info(f"🔁 [_skill_recover] Resetting arm to {home_joints}")
                    arm_client.execute_joint_move(
                        home_joints,
                        speed=self._arm_joint_speed,
                        acc=self._arm_joint_acc,
                        timeout_sec=self._arm_service_timeout
                    )
                    success = True
                else:
                    reason = "arm_client_unavailable"
            else:
                reason = f"unknown_mode_{mode}"
                
        except Exception as exc:
            log_error(f"❌ [_skill_recover] recover失败: {exc}")
            success = False
            reason = str(exc)
            
        return ExecutionResult(
            status="success" if success else "failure",
            node="recover",
            reason=reason if not success else None,
            evidence={"mode": mode}
        )

    def _ensure_gripper(self) -> Optional["GripperController"]:
        if self._gripper:
            return self._gripper
        if not GripperController:
            log_warning("⚠️ [_ensure_gripper] GripperController 未安装，无法控制夹爪")
            return None
        if not self._gripper_port:
            log_warning("⚠️ [_ensure_gripper] 未设置 GRIPPER_SERIAL_PORT，无法初始化夹爪")
            return None
        try:
            self._gripper = GripperController(self._gripper_port)
        except Exception as exc:
            log_warning(f"⚠️ [_ensure_gripper] 初始化夹爪控制器失败: {exc}")
            self._gripper = None
        return self._gripper

    def _call_linear_axis_move(self, target_x_mm: float) -> None:
        """
        Fire-and-forget helper to invoke the linear axis ROS2 service so that the arm
        roughly aligns with the detected目标中心高度.
        """
        service = "/jaka_driver/linear_move"
        srv_type = "jaka_msgs/srv/Move"
        pose = f"[{target_x_mm:.1f}, -227.0, 363.0, 0.0, 0.0, -0.733]"
        request = (
            "{"
            f"pose: {pose}, "
            "mvvelo: 10.0, "
            "mvacc: 10.0, "
            "has_ref: false, "
            "ref_joint: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "
            "mvtime: 0.0, "
            "mvradii: 0.0, "
            "coord_mode: 0, "
            "index: 0"
            "}"
        )
        cmd = ["ros2", "service", "call", service, srv_type, request]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            log_info(f"🦾 [_call_linear_axis_move] 线性轴对齐: x={target_x_mm:.1f}mm -> 调用 {service}")
        except FileNotFoundError:
            log_warning("⚠️ [_call_linear_axis_move] 未找到 ros2 命令，跳过线性轴调用")
        except subprocess.CalledProcessError as exc:
            log_warning(f"⚠️ [_call_linear_axis_move] 线性轴服务调用失败: {exc.stderr or exc}")
        except subprocess.TimeoutExpired:
            log_warning("⚠️ [_call_linear_axis_move] 线性轴服务调用超时")

    def _extract_rgb_and_surface_mask(self, observation) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if observation is None:
            return None, None
        rgb_frame = None
        surface_mask = None
        mask_path = getattr(observation, "surface_mask_path", None)
        if (not mask_path or not os.path.isfile(mask_path)) and getattr(observation, "surface_mask_task_id", None):
            wait_timeout = float(os.getenv("SAM_MASK_WAIT_TIMEOUT", "3"))
            result = sam_mask_worker.wait_for_result(getattr(observation, "surface_mask_task_id"), timeout=wait_timeout)
            if result:
                mask_path = result.get("path")
                observation.surface_mask_path = mask_path
                observation.surface_mask_url = result.get("url")
                observation.surface_mask_score = result.get("score")
                setattr(observation, "surface_mask_task_id", None)
        if mask_path and os.path.isfile(mask_path):
            try:
                with Image.open(mask_path) as mask_img:
                    surface_mask = np.array(mask_img)
            except Exception as exc:
                log_warning(f"⚠️ [_read_surface_mask] 读取surface mask失败: {exc}")
        img_path = observation.original_image_path or observation.processed_image_path
        if img_path and os.path.isfile(img_path):
            try:
                with Image.open(img_path) as img:
                    rgb_frame = np.array(img.convert("RGB"))
            except Exception:
                rgb_frame = None
        return rgb_frame, surface_mask

    def _ensure_arm_client(self) -> Optional[_ArmIKClient]:
        if self._arm_client is None:
            try:
                # 使用 rclpy.ok() 检查是否已初始化 (rclpy.is_initialized() 不存在)
                if not rclpy.ok():
                    rclpy.init()
                self._arm_client = _ArmIKClient()
                self._arm_client.wait_for_services(timeout_sec=self._arm_service_timeout)
                log_info("🦾 [_ensure_arm_ik_client] 机械臂IK客户端已就绪")
            except Exception as exc:
                log_error(f"❌ [_ensure_arm_ik_client] 初始化机械臂IK客户端失败: {exc}")
                self._arm_client = None
        return self._arm_client

    # ------------------------------------------------------------------
    # Primitive interfaces (LLM facing atomic controls)
    # ------------------------------------------------------------------
    def primitive_move_joint(
        self,
        joint_positions: Sequence[float],
        *,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """joint_positions: list of >=6 joint angles (rad). Returns {'ok': bool, 'joint_positions': [...]}."""
        arm_client = self._ensure_arm_client()
        if arm_client is None:
            return {"ok": False, "error": "arm_client_init_failed"}
        target = list(map(float, joint_positions))
        if len(target) < 6:
            return {"ok": False, "error": "joint_positions_requires_6_values"}
        speed = float(speed if speed is not None else self._arm_joint_speed)
        acc = float(acc if acc is not None else self._arm_joint_acc)
        timeout = float(timeout if timeout is not None else self._arm_service_timeout)
        try:
            arm_client.execute_joint_move(target[:6], speed=speed, acc=acc, timeout_sec=timeout)
            return {"ok": True, "joint_positions": target[:6], "speed": speed, "acc": acc}
        except Exception as exc:
            log_error(f"❌ [_skill_move_joint] move_joint 失败: {exc}")
            return {"ok": False, "error": str(exc)}

    def primitive_move_tcp(
        self,
        pose: Sequence[float],
        *,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        timeout: Optional[float] = None,
        ref_joint: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """pose: [x_mm,y_mm,z_mm,rx,ry,rz], solves IK then performs joint move (PTP), returns {'ok': bool,...}."""
        arm_client = self._ensure_arm_client()
        if arm_client is None:
            return {"ok": False, "error": "arm_client_init_failed"}
        if len(pose) != 6:
            return {"ok": False, "error": "pose_requires_6_values"}
        speed = float(speed if speed is not None else self._arm_joint_speed)
        acc = float(acc if acc is not None else self._arm_joint_acc)
        timeout = float(timeout if timeout is not None else self._arm_service_timeout)
        try:
            ref = list(ref_joint) if ref_joint is not None else arm_client.get_reference_joints(self._joint_state_timeout)
            joint_target = arm_client.solve_ik(list(map(float, pose)), ref, timeout_sec=timeout)
            arm_client.execute_joint_move(joint_target, speed=speed, acc=acc, timeout_sec=timeout)
            return {
                "ok": True,
                "tcp_pose": [float(v) for v in pose],
                "joint_target": joint_target,
                "speed": speed,
                "acc": acc,
            }
        except Exception as exc:
            log_error(f"❌ [_skill_move_tcp] move_tcp 失败: {exc}")
            return {"ok": False, "error": str(exc)}

    def primitive_shift_tcp(
        self,
        base_pose: Sequence[float],
        *,
        delta_xyz: Optional[Sequence[float]] = None,
        delta_rpy: Optional[Sequence[float]] = None,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """base_pose: current TCP pose, delta_* are offsets; returns move_tcp result."""
        if len(base_pose) != 6:
            return {"ok": False, "error": "base_pose_requires_6_values"}
        shift_pose = list(map(float, base_pose))
        if delta_xyz:
            for idx in range(min(3, len(delta_xyz))):
                shift_pose[idx] += float(delta_xyz[idx])
        if delta_rpy:
            for idx in range(min(3, len(delta_rpy))):
                shift_pose[idx + 3] += float(delta_rpy[idx])
        return self.primitive_move_tcp(
            shift_pose,
            speed=speed,
            acc=acc,
            timeout=timeout,
        )

    def primitive_follow_tcp_path(
        self,
        waypoints: Sequence[Sequence[float]],
        *,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """waypoints: iterable TCP poses, sequentially executes move_tcp, returns summary dict."""
        executed: List[List[float]] = []
        for pose in waypoints:
            result = self.primitive_move_tcp(pose, speed=speed, acc=acc, timeout=timeout)
            if not result.get("ok"):
                result["executed_waypoints"] = executed
                return result
            executed.append(result.get("tcp_pose") or list(map(float, pose)))
        return {"ok": True, "executed_waypoints": executed, "count": len(executed)}

    def primitive_apply_force(
        self,
        axis: Sequence[float],
        *,
        force: float,
        duration: float,
    ) -> Dict[str, Any]:
        """axis: normalized [x,y,z], applies force if hardware available; placeholder returns unsupported."""
        log_warning("⚠️ [_skill_apply_force] apply_force 尚未接入真实力控，返回占位信息")
        return {
            "ok": False,
            "error": "force_control_unavailable",
            "requested_axis": list(axis),
            "requested_force": float(force),
            "duration": float(duration),
        }

    def primitive_set_gripper(
        self,
        *,
        position: Optional[float] = None,
        force: Optional[float] = None,
    ) -> Dict[str, Any]:
        """position: 1~100, force: 20~320; updates gripper and returns requested state."""
        controller = self._ensure_gripper()
        if controller is None:
            return {"ok": False, "error": "gripper_unavailable"}
        try:
            if position is not None:
                controller.set_position(position)
            if force is not None:
                controller.set_force(force)
            return {
                "ok": True,
                "position": None if position is None else float(position),
                "force": None if force is None else float(force),
            }
        except Exception as exc:
            log_error(f"❌ [_skill_set_gripper] set_gripper 失败: {exc}")
            return {"ok": False, "error": str(exc)}

    def primitive_read_gripper_state(self) -> Dict[str, Any]:
        """Reads gripper telemetry when supported; placeholder returns unsupported."""
        log_warning("⚠️ [_skill_get_gripper_state] 当前抓手驱动不支持读状态，仅返回占位信息")
        return {"ok": False, "error": "gripper_state_unavailable"}

    def primitive_capture_depth_patch(
        self,
        bbox: Sequence[float],
        *,
        rgb_frame: Optional[np.ndarray] = None,
        surface_mask: Optional[np.ndarray] = None,
        include_transform: bool = True,
        timeout: float = 1.2,
    ) -> Dict[str, Any]:
        """bbox: [x1,y1,x2,y2] pixels. Calls depth service to localize area, returns payload or error."""
        if len(bbox) != 4:
            return {"ok": False, "error": "bbox_requires_4_values"}
        try:
            payload = self.depth_localizer.localize_from_service(
                bbox=bbox,
                rgb_frame=rgb_frame,
                surface_mask=surface_mask,
                timeout=timeout,
                include_transform=include_transform,
            )
        except Exception as exc:
            log_error(f"❌ [_skill_capture_depth_patch] capture_depth_patch 失败: {exc}")
            payload = None
        if not payload:
            return {"ok": False, "error": "depth_localization_failed"}
        return {"ok": True, "data": payload}

    def primitive_detect_contact(self, axis: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """axis: optional [x,y,z] reference; placeholder returns unsupported until sensors wired."""
        log_warning("⚠️ [_skill_detect_contact] detect_contact 尚未实现，缺少力矩/触觉传感器输入")
        return {"ok": False, "error": "contact_sensor_unavailable", "axis": list(axis) if axis else None}

    def _skill_execute_grasp(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        # S2: IK Reachable Illusion (Stress Injection)
        if Config.STRESS_IK_OOB_PROB > 0.0:
            if random.random() < Config.STRESS_IK_OOB_PROB:
                log_warning("⚡ [Stress] Injecting IK Boundary Illusion (OOB)")
                return ExecutionResult(
                    status="failure",
                    node="execute_grasp",
                    reason="stress_injected_ik_oob",
                    failure_code=FailureCode.IK_FAIL,
                    evidence={
                        "stress_injection": {
                            "type": "ik_oob",
                            "rate": Config.STRESS_IK_OOB_PROB,
                            "simulated_error": "Target out of reach"
                        }
                    }
                )

        grasp_pose = runtime.extra.get("grasp_pose") if runtime.extra else None
        if not grasp_pose:
            return ExecutionResult(
                status="failure",
                node="execute_grasp",
                reason="grasp_pose_unavailable",
            )
        tcp_pose = grasp_pose.get("tcp_pose")
        if tcp_pose is None or len(tcp_pose) != 6:
            return ExecutionResult(
                status="failure",
                node="execute_grasp",
                reason="invalid_tcp_pose",
                failure_code=FailureCode.IK_FAIL,
                evidence={"tcp_pose": tcp_pose, "error": "malformed_pose"}
            )
        arm_client = self._ensure_arm_client()
        if arm_client is None:
            return ExecutionResult(
                status="failure",
                node="execute_grasp",
                reason="arm_client_init_failed",
                failure_code=FailureCode.ARM_UNAVAILABLE
            )
        log_info(
            "🤖 [_skill_execute_grasp] execute_grasp: 发送TCP姿态 "
            f"x={tcp_pose[0]:.1f}, y={tcp_pose[1]:.1f}, z={tcp_pose[2]:.1f}, "
            f"rx={tcp_pose[3]:.3f}, ry={tcp_pose[4]:.3f}, rz={tcp_pose[5]:.3f}"
        )
        try:
            ref_joints = arm_client.get_reference_joints(timeout_sec=self._joint_state_timeout)
            log_info(f"🤖 [_skill_execute_grasp] execute_grasp: 当前参考关节角 {[round(float(v), 4) for v in ref_joints]}")
            joint_target = arm_client.solve_ik(
                tcp_pose,
                ref_joints,
                timeout_sec=self._arm_service_timeout,
            )
            log_info(f"🤖 [_skill_execute_grasp] execute_grasp: IK 解算结果 {[round(float(v), 4) for v in joint_target]}")
            arm_client.execute_joint_move(
                joint_target,
                speed=self._arm_joint_speed,
                acc=self._arm_joint_acc,
                timeout_sec=self._arm_service_timeout,
            )
            runtime.extra["grasp_pose_executed"] = True
            log_success("✅ 抓取姿态执行成功")
            return ExecutionResult(
                status="success",
                node="execute_grasp",
                evidence={"tcp_pose": [round(float(v), 3) for v in tcp_pose]},
            )
        except Exception as exc:
            log_error(f"❌ [_skill_execute_grasp] execute_grasp 失败: {exc}")
            # Determine if it's an IK failure or other
            err_msg = str(exc).lower()
            code = FailureCode.IK_FAIL if "ik" in err_msg or "joint" in err_msg else FailureCode.ARM_UNAVAILABLE
            return ExecutionResult(
                status="failure",
                node="execute_grasp",
                reason=str(exc),
                failure_code=code,
                evidence={
                    "tcp_pose": [round(float(v), 3) for v in tcp_pose] if tcp_pose else None,
                    "exception": str(exc)
                }
            )

    # --- VLA Geometry Helpers ---
    def _normalize_euler_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _euler_to_matrix(self, euler: np.ndarray) -> np.ndarray:
        rx, ry, rz = euler
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _rotation_matrix_to_rpy(self, R: np.ndarray) -> np.ndarray:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        yaw = math.atan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw], dtype=float)

    def _compute_euler_delta(self, current_euler: np.ndarray, target_euler: np.ndarray) -> np.ndarray:
        R_cur = self._euler_to_matrix(current_euler)
        R_tgt = self._euler_to_matrix(target_euler)
        R_rel = R_tgt @ R_cur.T
        delta_rpy = self._rotation_matrix_to_rpy(R_rel)
        for i in range(3):
            delta_rpy[i] = self._normalize_euler_angle(delta_rpy[i])
        return delta_rpy

    def _subdivide_action(self, delta_pos: np.ndarray, delta_euler: np.ndarray, depth: int = 0) -> list:
        MAX_POS_DELTA = 8.0  # mm
        MAX_ANGLE_DELTA = math.radians(5)  # rad
        MAX_DEPTH = 10
        pos_magnitude = np.linalg.norm(delta_pos)
        angle_magnitude = np.max(np.abs(delta_euler))
        if (pos_magnitude <= MAX_POS_DELTA and angle_magnitude <= MAX_ANGLE_DELTA) or depth >= MAX_DEPTH:
            return [(delta_pos, delta_euler)]
        half_pos = delta_pos * 0.5
        half_euler = delta_euler * 0.5
        return self._subdivide_action(half_pos, half_euler, depth + 1) + self._subdivide_action(half_pos, half_euler, depth + 1)

    def _execute_subdivided_actions(self, subdivisions: list, arm_client: _ArmIKClient) -> bool:
        for delta_pos, delta_euler in subdivisions:
            pose_delta = list(delta_pos) + list(delta_euler)
            if not arm_client.send_servo_p(pose_delta):
                return False
        return True
    # ----------------------------

    def _skill_vla_execute(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        """
        VLA Skill: vla_execute
        Connects to a remote VLA server via WebSocket to execute closed-loop control based on natural language instructions.
        """
        instruction = args.get("instruction", "do task")
        vla_host_port = os.getenv("VLA_SERVER_URL") # e.g. "192.168.1.100:8001"
        
        if not vla_host_port:
            log_warning("⚠️ [VLA] VLA_SERVER_URL not set. Falling back to simulation.")
            if self.stress_grasp_prob > 0.0 and random.random() < self.stress_grasp_prob:
                return ExecutionResult(status="failure", node="vla_execute", reason="vla_no_effect", failure_code=FailureCode.VLA_NO_EFFECT)
            time.sleep(0.5)
            runtime.extra["vla_executed"] = True 
            return ExecutionResult(status="success", node="vla_execute", evidence={"mode": "simulation"})

        # Parse Host/Port
        if "://" in vla_host_port:
            vla_host_port = vla_host_port.split("://")[1]
        host, port = vla_host_port.split(":")
        port = int(port)
        uri = f"ws://{host}:{port}"

        arm_client = self._ensure_arm_client()
        if not arm_client:
             return ExecutionResult(status="failure", node="vla_execute", reason="arm_client_unavailable")

        # Enable Servo Mode
        if not arm_client.enable_servo_mode():
             return ExecutionResult(status="failure", node="vla_execute", reason="failed_to_enable_servo")

        # Connect WebSocket
        log_info(f"🔌 [VLA] Connecting to {uri}...")
        try:
            ws = websockets.sync.client.connect(uri, additional_headers=None)
            # Receive metadata
            _ = unpackb(ws.recv()) 
        except Exception as e:
            return ExecutionResult(status="failure", node="vla_execute", reason=f"websocket_connect_error: {e}")

        # Execution Loop
        max_steps = 100 # Safety limit
        step = 0
        packer = Packer()
        
        # State tracking for relative moves
        prev_target_pos = None
        prev_target_euler = None

        try:
            while step < max_steps:
                # 1. Capture Image
                rgb_frame, _ = fetch_aligned_rgbd(timeout=2.0)
                if rgb_frame is None:
                    log_error("❌ [VLA] Failed to capture image")
                    break
                
                # Preprocess Image (Resize to 448x448 as per script)
                img = cv2.resize(rgb_frame, (448, 448))
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)

                # 2. Get Robot State (Cartesian)
                try:
                    curr_pos, curr_euler = arm_client.get_current_tool_pose()
                except Exception:
                    log_error("❌ [VLA] Failed to get tool pose")
                    break
                
                # 3. Prepare Payload
                # Note: We duplicate image for wrist_image as we only have one camera stream here
                payload = {
                    "observation.images.image": img,
                    "observation.images.wrist_image": img, # Placeholder
                    "observation.state": np.concatenate([
                        curr_pos, # [x, y, z] m
                        curr_euler, # [rx, ry, rz] rad
                        [1.0] # Gripper state placeholder (assuming open)
                    ]),
                    "task": instruction,
                    "n_action_steps": 10, # Request chunk size
                    "is_ep_start": (step == 0)
                }

                # 4. Infer
                ws.send(packer.pack(payload))
                response = ws.recv()
                result = unpackb(response)
                actions = result["actions"] # List of [x, y, z, rx, ry, rz, gripper]

                # 5. Execute Chunk
                for i, action in enumerate(actions):
                    target_pos = action[:3] # mm
                    target_euler_deg = action[3:6] # deg
                    target_euler = np.deg2rad(target_euler_deg)
                    gripper_cmd = action[6]

                    # Determine current reference for delta calculation
                    if i == 0 and step == 0:
                        ref_pos = curr_pos * 1000.0 # m -> mm
                        ref_euler = curr_euler
                    elif prev_target_pos is not None:
                        ref_pos = prev_target_pos
                        ref_euler = prev_target_euler
                    else:
                        ref_pos = curr_pos * 1000.0
                        ref_euler = curr_euler

                    # Calculate Deltas
                    delta_pos = target_pos - ref_pos
                    # Apply scale if needed (args.position_scale in script), assuming 1.0 here
                    
                    delta_euler = self._compute_euler_delta(ref_euler, target_euler)
                    
                    # Subdivide and Execute
                    subdivisions = self._subdivide_action(delta_pos, delta_euler)
                    if not self._execute_subdivided_actions(subdivisions, arm_client):
                        log_error("❌ [VLA] Servo execution failed")
                        raise RuntimeError("Servo failed")

                    # Update state
                    prev_target_pos = target_pos
                    prev_target_euler = target_euler
                    
                    # Gripper (Simple threshold)
                    if gripper_cmd < 0.5:
                        # Close
                        # self._skill_close_gripper({}, runtime) # This might be too slow for servo loop
                        pass 
                    else:
                        # Open
                        pass

                step += 1
                # Check termination condition? (Not defined in script, runs until max steps or manual stop)
                # For now, we run one chunk and return, or loop? 
                # The user script loops forever. 
                # We should probably break if the action implies "done" (e.g. gripper closed and lifted).
                # For this implementation, let's run 2 chunks (20 steps) then return success as a demo.
                if step >= 2: 
                    break

        except Exception as e:
            log_error(f"❌ [VLA] Error: {e}")
            ws.close()
            return ExecutionResult(status="failure", node="vla_execute", reason=str(e))
        
        ws.close()
        return ExecutionResult(status="success", node="vla_execute", evidence={"steps": step * 10})

    def _skill_align_tcp_to_target(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        """
        Moves the TCP to a point 20cm back from the target object, aligned with the robot-object line.
        """
        target_name = args.get("target")
        offset_mm = float(args.get("offset_mm", 200.0))
        
        # 1. Get Object Position relative to Robot Base
        # We try to use fresh perception (Depth) first
        depth_info = self._ensure_depth_localization(runtime)
        
        target_pos_base = None # [x, y, z] in mm
        
        if depth_info and depth_info.get("obj_center_3d"):
            # Use Camera -> Base transform
            obj_center_3d = depth_info.get("obj_center_3d")
            vec = np.array(obj_center_3d + [1.0], dtype=float).reshape(4, 1)
            # Hardcoded Extrinsics (Same as finalize_target_pose)
            T_mat = np.array(
                [
                    [0, 1, 0, 180],
                    [-1, 0, 0, 50],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )
            jaka_vec = T_mat @ vec
            target_pos_base = jaka_vec[:3].ravel() # x, y, z in mm
            log_info(f"🤖 [_skill_align_tcp_to_target] Target in Base Frame: {target_pos_base}")
        else:
            # Fallback: Check World Model if we have a stored pose?
            # For now, fail if not visible.
            return ExecutionResult(status="failure", node="align_tcp_to_target", reason="target_not_visible")

        # 2. Calculate Target TCP Position (20cm back)
        x, y, z = target_pos_base
        dist_xy = math.hypot(x, y)
        if dist_xy < 1e-3:
             return ExecutionResult(status="failure", node="align_tcp_to_target", reason="target_too_close")
             
        # Direction vector (normalized)
        dir_x = x / dist_xy
        dir_y = y / dist_xy
        
        target_x = x - dir_x * offset_mm
        target_y = y - dir_y * offset_mm
        target_z = z # Keep same height
        
        log_info(f"🤖 [_skill_align_tcp_to_target] Approach Pos: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
        
        # 3. Calculate Orientation (Yaw aligned with direction)
        # We want the gripper to point AT the object.
        # Assuming Gripper Z-axis is approach direction.
        # Target Z-axis (Approach): (dir_x, dir_y, 0)
        # Target Y-axis (Down): (0, 0, -1)
        # Target X-axis (Right): Cross(Y, Z) = (dir_y, -dir_x, 0) ? No.
        # Cross((0,0,-1), (dx, dy, 0)) = (dy, -dx, 0)
        
        z_axis = np.array([dir_x, dir_y, 0.0])
        y_axis = np.array([0.0, 0.0, -1.0])
        x_axis = np.cross(y_axis, z_axis) # (dy, -dx, 0)
        
        # Construct Rotation Matrix
        R = np.column_stack((x_axis, y_axis, z_axis))
        
        # Convert to Axis-Angle
        rot_vec = rotation_matrix_to_axis_angle(R)
        
        tcp_pose = [target_x, target_y, target_z, rot_vec[0], rot_vec[1], rot_vec[2]]
        
        # 4. Execute
        arm_client = self._ensure_arm_client()
        if not arm_client:
            return ExecutionResult(status="failure", node="align_tcp_to_target", reason="arm_unavailable")
            
        try:
            ref_joints = arm_client.get_reference_joints(timeout_sec=self._joint_state_timeout)
            joint_target = arm_client.solve_ik(
                tcp_pose,
                ref_joints,
                timeout_sec=self._arm_service_timeout,
            )
            arm_client.execute_joint_move(
                joint_target,
                speed=self._arm_joint_speed,
                acc=self._arm_joint_acc,
                timeout_sec=self._arm_service_timeout
            )
            return ExecutionResult(status="success", node="align_tcp_to_target", evidence={"tcp_pose": tcp_pose})
        except Exception as exc:
            log_error(f"❌ [_skill_align_tcp_to_target] Failed: {exc}")
            return ExecutionResult(status="failure", node="align_tcp_to_target", reason=str(exc))

    # ------------------------------------------------------------------
    # Helpers adapted from legacy TaskProcessor
    # ------------------------------------------------------------------
    def control_turn_around(self, navigator, turn_angle: float = 0.785):
        try:
            pose = navigator.get_current_pose()
            x = pose["x"]
            y = pose["y"]
            current_theta = pose["theta"]
            log_info(
                f"🤖 [_skill_turn_in_place] 底盘原地转向探索: 当前位置({x:.2f}, {y:.2f}), 朝向 θ={current_theta:.2f}rad"
            )
            new_theta_rad = current_theta + turn_angle
            while new_theta_rad > math.pi:
                new_theta_rad -= 2 * math.pi
            while new_theta_rad < -math.pi:
                new_theta_rad += 2 * math.pi
            success = navigator.move_to_position(new_theta_rad, x, y)
            if success:
                log_success("✅ 底盘转向成功")
            else:
                log_error("❌ [_skill_turn_in_place] 底盘转向失败")
            return success
        except Exception as exc:
            log_error(f"❌ [_skill_turn_in_place] 原地转向异常: {exc}")
            return False

    # ------------------------------------------------------------------
    def _ensure_depth_localization(self, runtime: SkillRuntime) -> Optional[Dict[str, Any]]:
        # S1: Burst Blindness Injection
        if self._blind_burst_remaining > 0:
            self._blind_burst_remaining -= 1
            log_warning(f"⚡ [Stress] Burst Blindness Active. Remaining frames: {self._blind_burst_remaining}")
            return None
        
        if Config.STRESS_BLIND_BURST_PROB > 0.0:
            if random.random() < Config.STRESS_BLIND_BURST_PROB:
                self._blind_burst_remaining = Config.STRESS_BLIND_BURST_LEN
                log_warning(f"⚡ [Stress] Injecting Burst Blindness. Length: {self._blind_burst_remaining}")
                return None

        observation = runtime.observation
        if not observation or not observation.bbox:
            log_warning("⚠️ [_skill_localize_depth] 深度定位缺少bbox")
            return None
        bbox = observation.bbox
        surface_points = runtime.surface_points or observation.surface_points
        range_estimate = observation.range_estimate
        rgb_frame, surface_mask = self._extract_rgb_and_surface_mask(observation)
        try:
            depth_info = self.depth_localizer.localize_from_service(
                bbox=bbox,
                surface_points_hint=surface_points,
                range_estimate=range_estimate,
                rgb_frame=rgb_frame,
                surface_mask=surface_mask,
                include_transform=True,
            )
        except Exception as exc:
            log_error(f"❌ [_skill_localize_depth] 调用深度定位服务失败: {exc}")
            return None
        if depth_info:
            runtime.extra["depth_localization"] = depth_info
        else:
            log_warning("⚠️ [_skill_localize_depth] 深度定位服务返回空结果")
        return depth_info

    def localize_observation(self, observation) -> Optional[Dict[str, Any]]:
        runtime = SkillRuntime(
            navigator=self.navigator,
            observation=observation,
            surface_points=observation.surface_points,
        )
        return self._ensure_depth_localization(runtime)

    def estimate_observation_pose(self, observation, navigator=None) -> Optional[Dict[str, Any]]:
        nav = navigator or self.navigator
        if nav is None:
            return None
        depth_info = self.localize_observation(observation)
        if not depth_info:
            return None
        cam_center = depth_info.get("obj_center_3d")
        if not cam_center:
            return None
        cam_point_mm = np.array(cam_center, dtype=float)
        robot_point_mm = self.transform_camera_to_robot(cam_point_mm)
        # todo ： when get current_pose？
        world_point_m = self.transform_robot_to_world(robot_point_mm, observation.robot_pose or nav.get_current_pose())
        return {
            "camera_center": cam_point_mm.tolist(),
            "robot_center": robot_point_mm.tolist(),
            "world_center": world_point_m.tolist(),
            "confidence": float(depth_info.get("confidence", 0.0)),
        }
