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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

import numpy as np
from PIL import Image
from task_logger import log_error, log_info, log_success, log_warning  # type: ignore

from localize_target import TargetLocalizer
from task_structures import ExecutionResult, PlanNode


@dataclass
class SkillRuntime:
    navigator: Any
    world_model: Any = None
    observation: Any = None
    frontend_payload: Optional[Dict[str, Any]] = None
    surface_points: Optional[Any] = None
    surface_region: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)


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
        grasp_result = self.depth_localizer.run_zero_grasp_inference(
            transform_result=depth_info,
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
        evidence: Dict[str, Any] = {"grasp_count": len(grasps) if isinstance(grasps, list) else 0}
        if best:
            evidence.update(
                {
                    "best_score": float(best.get("score", 0.0)),
                    "best_width_mm": best.get("width_mm"),
                    "best_translation_mm": best.get("translation_mm"),
                }
            )
        best_score = evidence.get("best_score")
        if best_score is not None:
            log_info(f"🤲 ZeroGrasp返回 {evidence['grasp_count']} 个抓取候选，最佳评分 {best_score:.3f}")
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

    def _skill_execute_grasp(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        """
        Placeholder for真实抓取策略接口。？？
        """
        log_info("🤖 execute_grasp: 预留抓取执行接口（当前返回成功）")
        return ExecutionResult(status="success", node="execute_grasp")

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
