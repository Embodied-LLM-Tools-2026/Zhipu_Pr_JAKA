"""
Shared data structures for the task execution pipeline.

This module defines lightweight dataclasses that are exchanged between
the planner, observer, executor and world-model components introduced
in the modular TaskProcessor redesign.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from localize_target import DepthSnapshot

class ObservationPhase(enum.Enum):
    """High-level observation stage for prompt selection."""

    SEARCH = "search"
    APPROACH = "approach"


@dataclass
class ObservationResult:
    """Structured output returned by the VLM observer."""

    found: bool
    bbox: List[float]
    confidence: float
    range_estimate: Optional[float]
    surface_points: Optional[List[List[int]]] = None
    annotated_url: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    processed_image_path: Optional[str] = None
    original_image_path: Optional[str] = None
    surface_mask_path: Optional[str] = None
    surface_mask_url: Optional[str] = None
    surface_mask_score: Optional[float] = None
    surface_mask_task_id: Optional[str] = None
    camera_center: Optional[List[float]] = None
    robot_center: Optional[List[float]] = None
    world_center: Optional[List[float]] = None
    depth_snapshot: Optional[DepthSnapshot] = None
    robot_pose: Optional[Dict[str, float]] = None
    source: str = "vlm"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "range_estimate": self.range_estimate,
            "surface_points": self.surface_points,
            "annotated_url": self.annotated_url,
            "raw_response": self.raw_response,
            "surface_mask_url": self.surface_mask_url,
            "surface_mask_score": self.surface_mask_score,
            "surface_mask_task_id": self.surface_mask_task_id,
            "camera_center": self.camera_center,
            "robot_center": self.robot_center,
            "world_center": self.world_center,
            "source": self.source,
        }


@dataclass
class ExecutionResult:
    """Outcome returned by a skill execution."""

    status: str  # success|failure|running
    node: str
    reason: Optional[str] = None
    elapsed: Optional[float] = None
    evidence: Optional[Dict[str, Any]] = None

    @property
    def success(self) -> bool:
        return self.status == "success"


@dataclass
class PlanNode:
    """Node in a behaviour tree returned by the planner."""

    type: str
    name: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    children: List["PlanNode"] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PlanNode":
        node_type = data.get("type") or data.get("bt")
        children_data = data.get("children") or []
        children = [PlanNode.from_dict(child) for child in children_data]
        return PlanNode(
            type=node_type,
            name=data.get("name"),
            args=data.get("args", {}),
            children=children,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the node (and its children) back to a JSON-friendly dict."""
        return {
            "type": self.type,
            "name": self.name,
            "args": self.args or {},
            "children": [child.to_dict() for child in self.children] if self.children else [],
        }


@dataclass
class CompiledPlan:
    """Flattened representation of the active plan."""

    root: PlanNode
    steps: List[PlanNode]
    metadata: Dict[str, Any] = field(default_factory=dict)
