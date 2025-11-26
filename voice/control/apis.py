"""
High level API wrappers that expose robot capabilities as modular interfaces.

These adapters wrap the existing Navigate/VLMObserver/SkillExecutor/Planner
components so that other modules (or future LLM agents) can discover and
compose capabilities without直接操作底层实现。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from tools.logging.task_logger import log_info, log_warning  # type: ignore

from action_sequence.navigate import Navigate
from .executor import SkillExecutor, SkillRuntime
from .world_model import WorldModel
from .task_structures import (
    ObservationPhase,
    PlanContextEntry,
    PlanNode,
    ExecutionTurn,
)

if TYPE_CHECKING:
    # Only import for type checking to avoid circular imports at runtime.
    from ..perception.observer import VLMObserver, ObservationContext

try:  # pragma: no cover - optional hardware dependency
    from action_sequence.gripper_controller import GripperController
except Exception:  # pragma: no cover
    GripperController = None


class NavigationAPI:
    """Wraps Navigate to provide a stable interface for LLM/agents."""

    def __init__(self, navigator: Optional[Navigate]) -> None:
        self._navigator = navigator

    def update_navigator(self, navigator: Optional[Navigate]) -> None:
        self._navigator = navigator

    def _require(self) -> Navigate:
        if self._navigator is None:
            raise RuntimeError("导航模块尚未初始化")
        return self._navigator

    def goto_marker(self, marker: str) -> bool:
        """Navigate to a named marker (调用底层 /api/move?marker=...)."""
        return self._require().navigate_to_target(marker)

    def goto_pose(self, x: float, y: float, theta: float, timeout: float = 60.0) -> bool:
        """Move to an explicit pose."""
        navigator = self._require()
        return navigator.move_to_position(theta, x, y, timeout=timeout)

    def wait_until_idle(self, timeout: float = 60.0) -> bool:
        """Block until the current导航指令结束."""
        return self._require().wait_until_navigation_complete(timeout=timeout)

    def current_pose(self) -> Dict[str, float]:
        return self._require().get_current_pose()

    def navigation_state(self) -> Dict[str, Any]:
        return self._require().get_navigation_state()


class PerceptionAPI:
    """Provides utility methods around VLMObserver + 深度定位."""

    def __init__(
        self,
        observer: VLMObserver,
        executor: SkillExecutor,
        world: WorldModel,
        navigator: Optional[Navigate],
    ) -> None:
        self._observer = observer
        self._executor = executor
        self._world = world
        self._navigator = navigator
        self._step_counter = 0

    def update_navigator(self, navigator: Optional[Navigate]) -> None:
        self._navigator = navigator

    def observe(
        self,
        target: str,
        *,
        phase: ObservationPhase = ObservationPhase.SEARCH,
        force_vlm: bool = False,
        max_steps: int = 1,
        analysis_request: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        # Lazy import to avoid circular dependency during module import.
        from ..perception.observer import ObservationContext

        if self._navigator is None:
            raise RuntimeError("Navigator 未设置，无法执行观察")
        self._step_counter += 1
        context = ObservationContext(step=self._step_counter, max_steps=max_steps)
        observation, payload = self._observer.observe(
            target,
            phase,
            context,
            self._navigator,
            force_vlm=force_vlm,
            analysis_request=analysis_request,
        )
        pose_info = self._executor.estimate_observation_pose(observation, self._navigator)
        if pose_info:
            observation.camera_center = pose_info.get("camera_center")
            observation.robot_center = pose_info.get("robot_center")
            observation.world_center = pose_info.get("world_center")
            observation.range_estimate = pose_info.get("range_estimate") or observation.range_estimate
        self._world.update_from_observation(target, observation)
        if observation.robot_center:
            try:
                rx, ry, rz = observation.robot_center
                log_info(
                    f"📍 观测目标 {target}: 机器人系坐标 ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm"
                )
            except Exception:
                log_info(f"📍 观测目标 {target}: 机器人系坐标 {observation.robot_center}")
        else:
            log_info(f"📍 观测目标 {target}: 未获取到机器人系坐标")
        return observation, payload

    def get_object_state(self, object_id: str) -> Optional[Dict[str, Any]]:
        obj = self._world.objects.get(object_id)
        if not obj:
            return None
        return {
            "cls": obj.cls,
            "visible": obj.visible,
            "attrs": obj.attrs,
            "world_center": obj.world_center,
            "camera_center": obj.camera_center,
            "robot_center": obj.robot_center,
            "confidence": obj.confidence,
        }

    def capture_depth_patch(
        self,
        bbox: Sequence[float],
        *,
        rgb_frame: Optional[Any] = None,
        surface_mask: Optional[Any] = None,
        include_transform: bool = True,
        timeout: float = 1.2,
    ) -> Dict[str, Any]:
        return self._executor.primitive_capture_depth_patch(
            bbox,
            rgb_frame=rgb_frame,
            surface_mask=surface_mask,
            include_transform=include_transform,
            timeout=timeout,
        )


class ManipulationAPI:
    """Thin wrapper around SkillExecutor for ad-hoc技能调用."""

    def __init__(
        self,
        executor: SkillExecutor,
        world: WorldModel,
        navigator: Optional[Navigate],
    ) -> None:
        self._executor = executor
        self._world = world
        self._navigator = navigator

    def update_navigator(self, navigator: Optional[Navigate]) -> None:
        self._navigator = navigator

    def execute_skill(
        self,
        name: str,
        *,
        args: Optional[Dict[str, Any]] = None,
        observation: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        runtime = SkillRuntime(
            navigator=self._navigator,
            world_model=self._world,
            observation=observation,
            extra=extra or {},
        )
        plan_node = PlanNode(type="action", name=name, args=args or {})
        return self._executor.execute(plan_node, runtime)

    def move_joint(
        self,
        joint_positions: Sequence[float],
        *,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._executor.primitive_move_joint(
            joint_positions,
            speed=speed,
            acc=acc,
            timeout=timeout,
        )

    def move_tcp_linear(
        self,
        pose: Sequence[float],
        *,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        timeout: Optional[float] = None,
        ref_joint: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        return self._executor.primitive_move_tcp_linear(
            pose,
            speed=speed,
            acc=acc,
            timeout=timeout,
            ref_joint=ref_joint,
        )

    def shift_tcp(
        self,
        base_pose: Sequence[float],
        *,
        delta_xyz: Optional[Sequence[float]] = None,
        delta_rpy: Optional[Sequence[float]] = None,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._executor.primitive_shift_tcp(
            base_pose,
            delta_xyz=delta_xyz,
            delta_rpy=delta_rpy,
            speed=speed,
            acc=acc,
            timeout=timeout,
        )

    def follow_tcp_path(
        self,
        waypoints: Sequence[Sequence[float]],
        *,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._executor.primitive_follow_tcp_path(
            waypoints,
            speed=speed,
            acc=acc,
            timeout=timeout,
        )

    def apply_force(
        self,
        axis: Sequence[float],
        *,
        force: float,
        duration: float,
    ) -> Dict[str, Any]:
        return self._executor.primitive_apply_force(axis, force=force, duration=duration)

    def set_gripper(
        self,
        *,
        position: Optional[float] = None,
        force: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._executor.primitive_set_gripper(position=position, force=force)

    def read_gripper_state(self) -> Dict[str, Any]:
        return self._executor.primitive_read_gripper_state()

    def detect_contact(self, axis: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        return self._executor.primitive_detect_contact(axis)


class GripperAPI:
    """Dedicated controller for夹爪动作，便于LLM直接调用."""

    def __init__(self, controller: Optional[GripperController] = None, port: Optional[str] = None) -> None:
        self._controller = controller
        self._port = port or os.getenv("GRIPPER_SERIAL_PORT")

    def _ensure(self) -> GripperController:
        if self._controller:
            return self._controller
        if GripperController is None:
            raise RuntimeError("未找到 GripperController 依赖")
        if not self._port:
            raise RuntimeError("未配置 GRIPPER_SERIAL_PORT，无法控制夹爪")
        self._controller = GripperController(self._port)
        return self._controller

    def open(self) -> None:
        self._ensure().open()

    def close(self) -> None:
        self._ensure().close()

    def deliver(self, item: Optional[str] = None) -> None:
        controller = self._ensure()
        controller.deliver(item)


class PlanningAPI:
    """Expose planning与反思接口，供Agent自助调用。"""

    def __init__(
        self,
        planner: BehaviorPlanner,
        reflection: Optional[ReflectionAdvisor],
        world: WorldModel,
    ) -> None:
        self._planner = planner
        self._reflection = reflection
        self._world = world

    def plan(self, goal: str, history: Optional[List[Dict[str, Any]]] = None):
        return self._planner.make_plan(goal, self._world, plan_context=history)

    def reflect(
        self,
        plan_entry: PlanContextEntry,
        execution_turns: List[ExecutionTurn],
        goal: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._reflection:
            return None
        return self._reflection.reflect(goal or plan_entry.goal, plan_entry, execution_turns)


@dataclass
class RobotAPI:
    """Aggregates all API surfaces for easier discovery."""

    navigation: NavigationAPI
    perception: PerceptionAPI
    manipulation: ManipulationAPI
    planning: PlanningAPI
    gripper: GripperAPI
    registry: Optional[Any] = None

    @classmethod
    def build(
        cls,
        *,
        navigator: Optional[Navigate],
        observer: VLMObserver,
        executor: SkillExecutor,
        planner: BehaviorPlanner,
        world: WorldModel,
        reflection: Optional[ReflectionAdvisor] = None,
        gripper_controller: Optional[GripperController] = None,
        registry: Optional[Any] = None,
    ) -> "RobotAPI":
        navigation = NavigationAPI(navigator)
        perception = PerceptionAPI(observer, executor, world, navigator)
        manipulation = ManipulationAPI(executor, world, navigator)
        planning = PlanningAPI(planner, reflection, world)
        gripper = GripperAPI(controller=gripper_controller)
        return cls(
            navigation=navigation,
            perception=perception,
            manipulation=manipulation,
            planning=planning,
            gripper=gripper,
            registry=registry,
        )

    def update_navigator(self, navigator: Optional[Navigate]) -> None:
        self.navigation.update_navigator(navigator)
        self.perception.update_navigator(navigator)
        self.manipulation.update_navigator(navigator)
        log_info("🤖 RobotAPI 已更新 navigator 引用")
