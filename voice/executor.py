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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

import numpy as np
from PIL import Image
from task_logger import log_error, log_info, log_success, log_warning  # type: ignore

from .localize_target import TargetLocalizer
from .task_structures import ExecutionResult, PlanNode


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

        # Camera calibration parameters
        self.camera_fx = 596
        self.camera_fy = 596
        self.camera_cx = 500
        self.camera_cy = 500
        self.camera_fov_h_deg = 80.0
        self.search_distance_threshold = 5.0

    # ------------------------------------------------------------------
    @staticmethod
    def transform_camera_to_robot(cam_point_mm: np.ndarray) -> np.ndarray:
        """
        Convert camera-frame coordinates (mm) to robot base coordinates (mm).

        The transformation follows the legacy finalize_target_pose logic.
        """
        cam_x, cam_y, cam_z = cam_point_mm
        x_ab = -cam_y
        y_ab = -cam_z + 180.0
        z_ab = cam_x - 50.0
        return np.array([x_ab, y_ab, z_ab], dtype=float)

    @staticmethod
    def transform_robot_to_world(robot_point_mm: np.ndarray, pose: Dict[str, float]) -> np.ndarray:
        """
        Convert robot base coordinates (mm) to world (map) coordinates (m).
        """
        x_ab, y_ab, z_ab = robot_point_mm
        theta = pose["theta"]
        x_oa = pose["x"] * 1000.0
        y_oa = pose["y"] * 1000.0
        # todo 
        x_ob = x_oa + (x_ab * math.cos(theta) - y_ab * math.sin(theta))
        y_ob = y_oa + (x_ab * math.sin(theta) + y_ab * math.cos(theta))
        return np.array([x_ob / 1000.0, y_ob / 1000.0, z_ab / 1000.0], dtype=float)

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
        if dist_m is not None and dist_m <= 5.0:
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
            dist_m = 6.0
        step_distance = max(0.5, min(1.0, dist_m - 4.5))

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

    # def _skill_approach_bbox(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
    #     observation = runtime.observation
    #     if observation is None or not observation.found or not observation.bbox:
    #         return ExecutionResult(
    #             status="failure",
    #             node="approach_bbox",
    #             reason="target_not_visible",
    #         )
    #     navigator = runtime.navigator
    #     bbox = observation.bbox
    #     image_size = observation.image_size
    #     centered = self.control_chassis_to_center(
    #         bbox,
    #         image_size,
    #         navigator=navigator,
    #         tolerance_px=args.get("tolerance_px"),
    #     )
    #     if not centered:
    #         return ExecutionResult(
    #             status="failure",
    #             node="approach_bbox",
    #             reason="center_alignment_failed",
    #             evidence={"bbox": bbox},
    #         )
    #     range_estimate = observation.range_estimate
    #     distance = float(args.get("distance") or 0.0)
    #     if distance == 0.0 and range_estimate:
    #         step = max(0.2, min(0.6, range_estimate - self.search_distance_threshold))
    #         distance = step if step > 0 else 0.3
    #     if distance:
    #         moved = self.control_chassis_forward(distance, navigator=navigator)
    #         status = "success" if moved else "failure"
    #         return ExecutionResult(
    #             status=status,
    #             node="approach_bbox",
    #             reason=None if moved else "move_forward_failed",
    #             evidence={"distance": distance},
    #         )
    #     return ExecutionResult(status="success", node="approach_bbox")

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
        tune_angle = depth_info.get("tune_angle", 0.0)
        try:
            cam_obj_center_3d = obj_center_3d
            vec = np.array([cam_obj_center_3d.copy() + [1.0]])
            T_mat = np.array(
                [
                    [0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, 180, -50],
                ]
            )
            jaka_obj_center_3d = vec @ T_mat
            pose = navigator.get_current_pose()
            X_OA = pose["x"] * 1000
            Y_OA = pose["y"] * 1000
            theta_OA = pose["theta"]
            X_AB, Y_AB, Z_AB = jaka_obj_center_3d.ravel()
            X_OB = X_OA + (X_AB * math.cos(theta_OA) - Y_AB * math.sin(theta_OA))
            Y_OB = Y_OA + (X_AB * math.sin(theta_OA) + Y_AB * math.cos(theta_OA))
            target_theta = theta_OA + tune_angle
            target_x = (X_OB - self.target_distance * 1000 * math.cos(target_theta)) / 1000
            target_y = (Y_OB - self.target_distance * 1000 * math.sin(target_theta)) / 1000

            log_info(
                f"🎯 计算底盘目标位置: ({target_x:.2f}, {target_y:.2f}), θ={target_theta:.2f}rad"
            )
            success = navigator.move_to_position(target_theta, target_x, target_y)
            if success:
                runtime.world_model.robot["pose"] = [target_x, target_y, target_theta]
            status = "success" if success else "failure"
            return ExecutionResult(
                status=status,
                node="finalize_target_pose",
                reason=None if success else "navigator_move_failed",
            )
        except Exception as exc:
            log_error(f"❌ finalize_target_pose 计算失败: {exc}")
            return ExecutionResult(
                status="failure",
                node="finalize_target_pose",
                reason=str(exc),
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

    def _skill_predict_grasp_point(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        """
        Placeholder for ZeroGrasp/抓取点预测接口。
        """
        log_info("🤖 predict_grasp_point: 预留ZeroGrasp接口（当前返回成功）")
        return ExecutionResult(status="success", node="predict_grasp_point")

    def _skill_execute_grasp(self, args: Dict[str, Any], runtime: SkillRuntime) -> ExecutionResult:
        """
        Placeholder for真实抓取策略接口。
        """
        log_info("🤖 execute_grasp: 预留抓取执行接口（当前返回成功）")
        return ExecutionResult(status="success", node="execute_grasp")

    # ------------------------------------------------------------------
    # Helpers adapted from legacy TaskProcessor
    # ------------------------------------------------------------------
    # def control_chassis_to_center(
    #     self,
    #     bbox,
    #     image_size,
    #     navigator=None,
    #     tolerance_px: Optional[float] = None,
    # ):
    #     navigator = navigator or self.navigator
    #     if navigator is None:
    #         log_error("❌ 底盘中心对齐失败：缺少导航控制器")
    #         return False
    #     try:
    #         if not bbox or len(bbox) < 4:
    #             log_warning("⚠️ 底盘中心对齐: 边界框无效")
    #             return False
    #         if not image_size or len(image_size) < 2:
    #             log_warning("⚠️ 底盘中心对齐: 图像尺寸无效")
    #             return False

    #         img_center_x = image_size[0] / 2.0
    #         img_center_y = image_size[1] / 2.0

    #         x1, y1, x2, y2 = bbox
    #         bbox_center_x = (x1 + x2) / 2.0
    #         bbox_center_y = (y1 + y2) / 2.0

    #         dx_pixels = img_center_x - bbox_center_x
    #         dy_pixels = img_center_y - bbox_center_y

    #         log_info(f"🎯 底盘中心对齐: 像素偏差({dx_pixels:.1f}, {dy_pixels:.1f})")

    #         tolerance = tolerance_px if tolerance_px is not None else 50.0
    #         if abs(dx_pixels) < tolerance and abs(dy_pixels) < tolerance:
    #             log_success("✅ 目标已在视野中心")
    #             self.moved_to_center = False
    #             return True

    #         turn_angle_rad = math.atan(dx_pixels / self.camera_fx) * 0.7
    #         turn_angle_deg = -math.degrees(turn_angle_rad)
    #         log_info(f"💡 像素转角度: {dx_pixels:.1f}px → {turn_angle_deg:.2f}°")

    #         min_turn_angle_deg = 0.5
    #         if abs(turn_angle_deg) < min_turn_angle_deg:
    #             log_success("✅ 转向角度不足，无需调整")
    #             self.moved_to_center = False
    #             return True

    #         current_pose = navigator.get_current_pose()
    #         current_x = current_pose["x"]
    #         current_y = current_pose["y"]
    #         current_theta = current_pose["theta"]
    #         new_theta = current_theta + turn_angle_rad
    #         success = navigator.move_to_position(new_theta, current_x, current_y)
    #         if success:
    #             log_success("✅ 底盘转向成功")
    #             self.moved_to_center = True
    #         else:
    #             log_error("❌ 底盘转向失败")
    #             self.moved_to_center = False
    #         return success
    #     except Exception as exc:
    #         log_error(f"❌ 中心对齐异常: {exc}")
    #         self.moved_to_center = False
    #         return False

    # # def control_chassis_forward(self, distance: float = 1.0, navigator=None):
    #     navigator = navigator or self.navigator
    #     if navigator is None:
    #         log_error("❌ 底盘前进失败: Navigator 实例不可用")
    #         return False
    #     try:
    #         current_pose = navigator.get_current_pose()
    #         x0 = current_pose["x"]
    #         y0 = current_pose["y"]
    #         theta = current_pose["theta"]
    #     except Exception as exc:
    #         log_error(f"❌ 获取当前位置失败: {exc}")
    #         return False
    #     x_new = x0 + distance * math.cos(theta)
    #     y_new = y0 + distance * math.sin(theta)
    #     log_info(
    #         f"🤖 底盘前进: 当前({x0:.2f}, {y0:.2f}) → 目标({x_new:.2f}, {y_new:.2f}), 距离={distance:.2f}m"
    #     )
    #     try:
    #         success = navigator.move_to_position(theta, x_new, y_new)
    #         if success:
    #             log_success("✅ 底盘前进成功")
    #         else:
    #             log_error("❌ 底盘前进失败")
    #         return success
    #     except Exception as exc:
    #         log_error(f"❌ 底盘前进异常: {exc}")
    #         return False

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
        rgb_frame = None
        img_path = observation.processed_image_path or observation.original_image_path
        if img_path and os.path.isfile(img_path):
            try:
                with Image.open(img_path) as img:
                    rgb_frame = np.array(img.convert("RGB"))
            except Exception:
                rgb_frame = None
        try:
            depth_info = self.depth_localizer.localize_from_service(
                bbox=bbox,
                surface_points_hint=surface_points,
                range_estimate=range_estimate,
                rgb_frame=rgb_frame,
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
