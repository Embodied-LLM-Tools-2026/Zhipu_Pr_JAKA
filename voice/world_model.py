"""
Lightweight world model for the modular TaskProcessor pipeline.

This version follows the "no semantic map" MVP from the redesign
proposal: it tracks named areas, recently observed objects, robot
state and minimal task memory required for behaviour tree planning
and replanning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import threading

from .task_structures import ObservationResult


@dataclass
class AreaState:
    name: str
    pose: Optional[List[float]] = None  # [x, y, theta]
    vantage_points: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ObjectState:
    object_id: str
    cls: str = "unknown"
    attrs: Dict[str, Any] = field(default_factory=dict)
    visible: bool = False
    camera_center: Optional[List[float]] = None
    robot_center: Optional[List[float]] = None
    world_center: Optional[List[float]] = None
    confidence: float = 0.0
    last_seen: float = field(default_factory=time.time)
    seen_in: Optional[str] = None
    annotated_url: Optional[str] = None


class WorldModel:
    """Task-level state store shared across planner/observer/executor."""

    def __init__(self) -> None:
        self.areas: Dict[str, AreaState] = {}
        self.objects: Dict[str, ObjectState] = {}
        self.robot: Dict[str, Any] = {
            "pose": [0.0, 0.0, 0.0],
            "gripper": "open",
            "holding": None,
        }
        self.task_memory: Dict[str, Any] = {
            "tried_areas": [],
            "fail_counts": {},
        }
        self.goal: Optional[str] = None
        self.version: int = 0
        self._last_execution: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()
        self._id_counters: Dict[str, int] = {}
        self.catalog_merge_threshold = 0.2  # metres

    # ------------------------------------------------------------------
    # Goal & snapshot helpers
    # ------------------------------------------------------------------
    def set_goal(self, goal: str) -> None:
        with self._lock:
            self.goal = goal
    # todo : how to make a info-enough snapshot for LLM planner?
    def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable summary for planner consumption."""
        with self._lock:
            areas_snapshot = {name: area.__dict__.copy() for name, area in self.areas.items()}
            objects_snapshot: Dict[str, Any] = {}
            for obj_id, obj in self.objects.items():
                objects_snapshot[obj_id] = {
                    "cls": obj.cls,
                    "attrs": dict(obj.attrs),
                    "visible": obj.visible,
                    "camera_center": obj.camera_center,
                    "robot_center": obj.robot_center,
                    "world_center": obj.world_center,
                    "confidence": obj.confidence,
                    "last_seen": obj.last_seen,
                    "seen_in": obj.seen_in,
                    "annotated_url": obj.annotated_url,
                }
            return {
                "version": self.version,
                "goal": self.goal,
                "areas": areas_snapshot,
                "objects": objects_snapshot,
                "robot": self.robot.copy(),
                "task_memory": dict(self.task_memory),
            }

    # ------------------------------------------------------------------
    # Area/object management
    # ------------------------------------------------------------------
    def upsert_area(
        self, name: str, pose: Optional[List[float]] = None, vantage: Optional[List[Dict[str, float]]] = None
    ) -> None:
        with self._lock:
            state = self.areas.get(name) or AreaState(name=name)
            if pose is not None:
                state.pose = pose
            if vantage:
                state.vantage_points = vantage
            self.areas[name] = state
    # todo : remove some info , something needed in ObservationResult but not in world model
    def update_from_observation(
        self, target_id: str, observation: ObservationResult, current_area: Optional[str] = None
    ) -> None:
        """Fuse the latest observation into the world model."""
        with self._lock:
            obj = self.objects.get(target_id) or ObjectState(object_id=target_id, cls=target_id)
            obj.visible = observation.found
            obj.confidence = max(obj.confidence, observation.confidence)
            obj.last_seen = time.time()
            obj.annotated_url = observation.annotated_url or obj.annotated_url
            obj.seen_in = current_area or obj.seen_in
            obj.attrs["analysis"] = observation.analysis
            if observation.range_estimate is not None:
                obj.attrs["range_estimate"] = observation.range_estimate
            if observation.surface_points:
                obj.attrs["surface_points"] = observation.surface_points
            if observation.surface_roi:
                obj.attrs["surface_region"] = observation.surface_roi
            if observation.camera_center:
                obj.camera_center = observation.camera_center
            if observation.robot_center:
                obj.robot_center = observation.robot_center
            if observation.world_center:
                obj.world_center = observation.world_center
            self.objects[target_id] = obj
            self.version += 1

    def update_pose_estimate(
        self,
        object_id: str,
        *,
        camera_center: Optional[List[float]] = None,
        robot_center: Optional[List[float]] = None,
        world_center: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            obj = self.objects.get(object_id) or ObjectState(object_id=object_id, cls=object_id)
            if camera_center is not None:
                obj.camera_center = list(camera_center)
            if robot_center is not None:
                obj.robot_center = list(robot_center)
            if world_center is not None:
                obj.world_center = list(world_center)
            if confidence is not None:
                obj.confidence = max(obj.confidence, confidence)
            if attrs:
                obj.attrs.update(attrs)
            obj.last_seen = time.time()
            self.objects[object_id] = obj
            self.version += 1

    def register_catalog_detection(
        self,
        label: str,
        world_center: List[float],
        confidence: float,
        *,
        camera_center: Optional[List[float]] = None,
        robot_center: Optional[List[float]] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            existing_id = self._find_similar_object(label, world_center)
            if existing_id:
                obj = self.objects[existing_id]
                if confidence >= obj.confidence:
                    if camera_center is not None:
                        obj.camera_center = list(camera_center)
                    if robot_center is not None:
                        obj.robot_center = list(robot_center)
                    obj.world_center = list(world_center)
                    obj.confidence = confidence
                    if attrs:
                        obj.attrs.update(attrs)
                obj.last_seen = time.time()
                self.objects[existing_id] = obj
            else:
                object_id = self._generate_object_id(label)
                obj = ObjectState(
                    object_id=object_id,
                    cls=label,
                    attrs=dict(attrs or {}),
                    visible=False,
                    camera_center=list(camera_center) if camera_center else None,
                    robot_center=list(robot_center) if robot_center else None,
                    world_center=list(world_center),
                    confidence=confidence,
                )
                obj.last_seen = time.time()
                self.objects[object_id] = obj
            self.version += 1

    # ------------------------------------------------------------------
    # Execution feedback & replanning heuristics
    # ------------------------------------------------------------------
    def record_execution_result(self, result: Dict[str, Any]) -> None:
        with self._lock:
            self._last_execution = result
            node = result.get("node")
            status = result.get("status")
            if node and status == "failure":
                fail_counts = self.task_memory.setdefault("fail_counts", {})
                fail_counts[node] = fail_counts.get(node, 0) + 1
            self.version += 1

    def should_replan(self, exec_result: Dict[str, Any]) -> bool:
        """Simple heuristics for MVP replanning triggers."""
        status = exec_result.get("status")
        node = exec_result.get("node")
        if status == "failure":
            return True
        if node in {"search_area", "rotate_scan"} and exec_result.get("metadata", {}).get("found") is False:
            return True
        return False

    # ------------------------------------------------------------------
    # Condition evaluation (for BT check nodes)
    # ------------------------------------------------------------------
    def _generate_object_id(self, label: str) -> str:
        safe_label = label.replace(" ", "_")
        count = self._id_counters.get(safe_label, 0) + 1
        self._id_counters[safe_label] = count
        return f"{safe_label}#{count}"

    def _find_similar_object(self, label: str, world_center: List[float]) -> Optional[str]:
        best_id: Optional[str] = None
        best_distance = float("inf")
        candidate = np.array(world_center, dtype=float)
        for obj_id, obj in self.objects.items():
            if obj.cls != label or obj.world_center is None:
                continue
            dist = np.linalg.norm(candidate - np.array(obj.world_center, dtype=float))
            if dist <= self.catalog_merge_threshold and dist < best_distance:
                best_id = obj_id
                best_distance = dist
        return best_id
    # todo : what is this
    def evaluate_condition(self, expression: str) -> bool:
        """
        Evaluate a limited condition language used by planner check nodes.

        Supported patterns: "<path> <op> <value>", where op ∈ {==, !=, <, <=, >, >=}
        and <path> may reference dotted attributes inside the snapshot dictionary.
        """
        try:
            tokens = None
            for op in ["==", "!=", "<=", ">=", "<", ">"]:
                if op in expression:
                    lhs, rhs = expression.split(op, 1)
                    tokens = (lhs.strip(), op, rhs.strip())
                    break
            if not tokens:
                return False
            lhs, op, rhs = tokens
            value = self._resolve_path(lhs)
            rhs_value = self._coerce_rhs(rhs, value)
            if value is None:
                if op == "==" and rhs_value is None:
                    return True
                if op == "!=" and rhs_value is not None:
                    return True
                return False
            if op == "==":
                return value == rhs_value
            if op == "!=":
                return value != rhs_value
            if op == "<":
                return value < rhs_value
            if op == "<=":
                return value <= rhs_value
            if op == ">":
                return value > rhs_value
            if op == ">=":
                return value >= rhs_value
        except Exception:
            return False
        return False

    def _resolve_path(self, path: str) -> Any:
        snapshot = self.snapshot()
        parts = path.split(".")
        current: Any = snapshot
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                break
        return current

    @staticmethod
    def _coerce_rhs(rhs: str, lhs_value: Any) -> Any:
        if rhs.lower() in {"true", "false"}:
            return rhs.lower() == "true"
        if rhs.lower() == "null":
            return None
        try:
            if isinstance(lhs_value, float):
                return float(rhs)
            if isinstance(lhs_value, int):
                return int(rhs)
        except ValueError:
            pass
        return rhs.strip('"').strip("'")
