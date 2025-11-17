"""
Skill executor responsible for running low-level atomic actions.

This module extracts the execution related logic from the legacy
TaskProcessor so that the orchestrator can delegate plan nodes to
explicit skills following the redesigned architecture.
"""

from __future__ import annotations

import math
import os
import sys
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

import numpy as np
import rclpy
from PIL import Image
from jaka_msgs.srv import GetIK, Move
from rclpy.node import Node
from sensor_msgs.msg import JointState
from task_logger import log_error, log_info, log_success, log_warning  # type: ignore

from localize_target import TargetLocalizer
from task_structures import ExecutionResult, PlanNode
from .sam_worker import sam_mask_worker  # type: ignore


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
    """Minimal helper node to resolve IK and trigger joint moves."""

    def __init__(self) -> None:
        super().__init__("skill_executor_arm_ik")
        self.joint_move_client = self.create_client(Move, "/jaka_driver/joint_move")
        self.get_ik_client = self.create_client(GetIK, "/jaka_driver/get_ik")
        self._latest_joint_state: Optional[JointState] = None
        self.create_subscription(
            JointState,
            "/jaka_driver/joint_position",
            self._joint_state_callback,
            qos_profile=10,
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        if len(msg.position) >= 6:
            self._latest_joint_state = msg

    def wait_for_services(self, timeout_sec: float = 5.0) -> None:
        clients = (
            (self.joint_move_client, "/jaka_driver/joint_move"),
            (self.get_ik_client, "/jaka_driver/get_ik"),
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


class SkillExecutor:
    """Executes behaviour tree action nodes by calling registered skills."""

    def __init__(self, navigator=None) -> None:
        self.navigator = navigator
        self.depth_localizer = TargetLocalizer()
        self.target_distance = 0.5
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
        self._arm_client: Optional[_ArmIKClient] = None
        self._joint_state_timeout = float(os.getenv("JAKA_JOINT_STATE_TIMEOUT", "3.0"))
        self._arm_service_timeout = float(os.getenv("JAKA_SERVICE_TIMEOUT", "10.0"))
        self._arm_joint_speed = float(os.getenv("JAKA_JOINT_SPEED", "5.0"))
        self._arm_joint_acc = float(os.getenv("JAKA_JOINT_ACC", "5.0"))

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

    def execute(self, node: PlanNode, runtime: SkillRuntime) -> ExecutionResult:
        handler_name = f"_skill_{node.name}"
        handler = getattr(self, handler_name, None)
        if not handler:
            log_warning(f"⚠️ 未实现的技能: {node.name}")
            return ExecutionResult(
                status="failure",
                node=node.name or "unknown",
                reason="unsupported_skill",
            )
        try:
            return handler(node.args or {}, runtime)
        except Exception as exc:
            log_error(f"❌ 执行技能 {node.name} 发生异常: {exc}")
            return ExecutionResult(
                status="failure",
                node=node.name or "unknown",
                reason=str(exc),
            )

    # ------------------------------------------------------------------
    # Skill implementations
    # ------------------------------------------------------------------
    def _skill_rotate_scan(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        angle = float(args.get("angle_deg", 30.0))
        success = self.control_turn_around(runtime.navigator, math.radians(angle))
        status = "success" if success else "failure"
        return ExecutionResult(status=status, node="rotate_scan")

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
            vec = np.array([cam_obj_center_3d.copy() + [1.0]])
            T_mat = np.array(
                [
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                    [50, 180, 0],
                ]
            )
            jaka_obj_center_3d = vec @ T_mat
            pose = navigator.get_current_pose()
            X_OA = pose["x"] * 1000
            Y_OA = pose["y"] * 1000
            theta_OA = pose["theta"]
            X_AB, Y_AB, Z_AB = jaka_obj_center_3d.ravel()
            robot_x_mm = float(X_AB)
            X_OB = X_OA + (X_AB * math.cos(theta_OA) - Y_AB * math.sin(theta_OA))
            Y_OB = Y_OA + (X_AB * math.sin(theta_OA) + Y_AB * math.cos(theta_OA))
            obj_world_x = X_OB / 1000.0
            obj_world_y = Y_OB / 1000.0

            if mask_available:
                target_theta = theta_OA + tune_angle
                orientation_source = "sam_edge"
            else:
                dx = obj_world_x - pose["x"]
                dy = obj_world_y - pose["y"]
                if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                    target_theta = theta_OA
                else:
                    target_theta = math.atan2(dy, dx)
                orientation_source = "target_vector"

            target_x = obj_world_x - self.target_distance * math.cos(target_theta)
            target_y = obj_world_y - self.target_distance * math.sin(target_theta)

            if robot_x_mm > -310 and robot_x_mm < 200:
                self._call_linear_axis_move(robot_x_mm)
            else :
                log_warning(f"⚠️ 目标X位置超出线性轴范围: {robot_x_mm:.1f}mm，跳过线性轴对齐")
            log_info(
                f"🎯 计算底盘目标位置: ({target_x:.2f}, {target_y:.2f}), θ={target_theta:.2f}rad, orient={orientation_source}"
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
            log_error(f"❌ finalize_target_pose 计算失败: {exc}")
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
            log_warning(f"⚠️ 解析抓取姿态失败: {exc}")
            pose = None
        if not pose:
            return None
        position_mm, axes_robot, _ = pose
        tcp_position = _offset_gripper_to_tcp(position_mm, axes_robot[2], self.tcp_backoff_mm)
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
        # range_threshold = float(args.get("range_threshold", 0.5))
        # range_est = observation.range_estimate
        # if range_est is None or range_est > range_threshold:
        #     return ExecutionResult(
        #         status="failure",
        #         node="predict_grasp_point",
        #         reason="target_too_far",
        #         evidence={"range_estimate": range_est},
        #     )
        if not self._finalize_pose_ready:
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason="finalize_not_completed",
            )
        depth_info = runtime.extra.get("depth_localization") or self._ensure_depth_localization(runtime)
        if not depth_info:
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason="depth_localization_unavailable",
            )
        rgb_frame, _ = self._extract_rgb_and_surface_mask(observation)
        if rgb_frame is None:
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason="missing_rgb_frame",
            )
        bbox = depth_info.get("bbox") or observation.bbox
        transform_result = depth_info.get("transform_result", depth_info)
        grasp_result = self.depth_localizer.run_zero_grasp_inference(
            transform_result=transform_result,
            bbox=bbox,
            rgb_frame=rgb_frame,
        )
        if not grasp_result:
            return ExecutionResult(
                status="failure",
                node="predict_grasp_point",
                reason="zerograsp_failed",
            )
        grasps = grasp_result.get("grasps") if isinstance(grasp_result, dict) else None
        best = None
        if isinstance(grasps, list) and grasps:
            best = max(grasps, key=lambda g: g.get("score", 0.0))
        runtime.extra["zerograsp_result"] = grasp_result
        if "grasp_pose" in runtime.extra:
            runtime.extra.pop("grasp_pose", None)
        evidence: Dict[str, Any] = {"grasp_count": len(grasps) if isinstance(grasps, list) else 0}
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
                f"🤲 ZeroGrasp返回 {evidence['grasp_count']} 个抓取候选，最佳评分 {best_score:.3f}{pose_extra}"
            )
        else:
            log_info(f"🤲 ZeroGrasp返回 {evidence['grasp_count']} 个抓取候选，但未找到有效抓取")
        self._finalize_pose_ready = False
        return ExecutionResult(
            status="success",
            node="predict_grasp_point",
            evidence=evidence,
        )

    def _skill_pick(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        # Placeholder for integration with manipulator stack
        log_warning("⚠️ pick技能尚未与真实机械臂对接，默认返回成功")
        return ExecutionResult(status="success", node="pick")

    def _skill_place(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        log_warning("⚠️ place技能尚未实现，默认返回成功")
        return ExecutionResult(status="success", node="place")

    def _skill_recover(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        mode = args.get("mode", "backoff")
        log_info(f"🔁 recover skill triggered, mode={mode}")
        try:
            navigator = runtime.navigator
            pose = navigator.get_current_pose()
            back_distance = float(args.get("distance", 0.3))
            new_x = pose["x"] - back_distance * math.cos(pose["theta"])
            new_y = pose["y"] - back_distance * math.sin(pose["theta"])
            success = navigator.move_to_position(pose["theta"], new_x, new_y)
        except Exception as exc:
            log_error(f"❌ recover失败: {exc}")
            success = False
        return ExecutionResult(
            status="success" if success else "failure",
            node="recover",
            reason=None if success else "recover_failed",
        )

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
            log_info(f"🦾 线性轴对齐: x={target_x_mm:.1f}mm -> 调用 {service}")
        except FileNotFoundError:
            log_warning("⚠️ 未找到 ros2 命令，跳过线性轴调用")
        except subprocess.CalledProcessError as exc:
            log_warning(f"⚠️ 线性轴服务调用失败: {exc.stderr or exc}")
        except subprocess.TimeoutExpired:
            log_warning("⚠️ 线性轴服务调用超时")

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
                log_warning(f"⚠️ 读取surface mask失败: {exc}")
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
                if not rclpy.is_initialized():
                    rclpy.init()
                self._arm_client = _ArmIKClient()
                self._arm_client.wait_for_services(timeout_sec=self._arm_service_timeout)
                log_info("🦾 机械臂IK客户端已就绪")
            except Exception as exc:
                log_error(f"❌ 初始化机械臂IK客户端失败: {exc}")
                self._arm_client = None
        return self._arm_client

    def _skill_execute_grasp(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
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
            )
        arm_client = self._ensure_arm_client()
        if arm_client is None:
            return ExecutionResult(
                status="failure",
                node="execute_grasp",
                reason="arm_client_init_failed",
            )
        log_info(
            "🤖 execute_grasp: 发送TCP姿态 "
            f"x={tcp_pose[0]:.1f}, y={tcp_pose[1]:.1f}, z={tcp_pose[2]:.1f}, "
            f"rx={tcp_pose[3]:.3f}, ry={tcp_pose[4]:.3f}, rz={tcp_pose[5]:.3f}"
        )
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
            log_error(f"❌ execute_grasp 失败: {exc}")
            return ExecutionResult(
                status="failure",
                node="execute_grasp",
                reason=str(exc),
            )

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
                f"🤖 底盘原地转向探索: 当前位置({x:.2f}, {y:.2f}), 朝向 θ={current_theta:.2f}rad"
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
                log_error("❌ 底盘转向失败")
            return success
        except Exception as exc:
            log_error(f"❌ 原地转向异常: {exc}")
            return False

    # ------------------------------------------------------------------
    def _ensure_depth_localization(self, runtime: SkillRuntime) -> Optional[Dict[str, Any]]:
        observation = runtime.observation
        if not observation or not observation.bbox:
            log_warning("⚠️ 深度定位缺少bbox")
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
            log_error(f"❌ 调用深度定位服务失败: {exc}")
            return None
        if depth_info:
            runtime.extra["depth_localization"] = depth_info
        else:
            log_warning("⚠️ 深度定位服务返回空结果")
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
