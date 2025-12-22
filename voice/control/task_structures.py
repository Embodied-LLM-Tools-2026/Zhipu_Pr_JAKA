"""
Shared data structures for the task execution pipeline.

This module defines lightweight dataclasses that are exchanged between
the planner, observer, executor and world-model components introduced
in the modular TaskProcessor redesign.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ..perception.localize_target import DepthSnapshot


class FailureCode(enum.Enum):
    """Canonical failure codes for skill execution and recovery."""

    # Perception
    NO_OBSERVATION = "perception.no_observation"
    RGB_UNAVAILABLE = "perception.rgb_unavailable"
    DEPTH_LOCALIZATION_FAILED = "perception.depth_localization_failed"

    # Navigation
    NAV_BLOCKED = "nav.nav_blocked"
    ROTATE_FAILED = "nav.rotate_failed"
    MISSING_TARGET = "nav.missing_target"

    # Manipulation
    ZEROGRASP_FAILED = "manip.zerograsp_failed"
    IK_FAIL = "manip.ik_fail"
    GRASP_FAIL = "manip.grasp_fail"
    GRASP_SLIP = "manip.grasp_slip"

    # VLA / Learned Policy
    VLA_NO_EFFECT = "vla.no_effect"
    VLA_POLICY_OOB = "vla.policy_oob"
    VLA_MODEL_ERROR = "vla.model_error"

    # Infrastructure / dependencies
    GRIPPER_UNAVAILABLE = "infra.gripper_unavailable"
    NAVIGATOR_UNAVAILABLE = "infra.navigator_unavailable"
    ARM_UNAVAILABLE = "infra.arm_unavailable"
    IMPORT_ERROR = "infra.import_error"
    MISSING_RUN = "infra.missing_run"
    SKILL_TIMEOUT = "infra.skill_timeout"
    EPISODE_TIMEOUT = "infra.episode_timeout"

    # Contract / orchestration
    VERIFICATION_FAILED = "contract.verification_failed"
    UNSUPPORTED_SKILL = "contract.unsupported_skill"

    UNKNOWN = "unknown"


_REASON_MAP = {
    # Perception
    "no_observation": FailureCode.NO_OBSERVATION,
    "missing_observation": FailureCode.NO_OBSERVATION,
    "no_direction_vector": FailureCode.NO_OBSERVATION,
    "rgb_unavailable": FailureCode.RGB_UNAVAILABLE,
    "rgb_frame_unavailable": FailureCode.RGB_UNAVAILABLE,
    "depth_localization_failed": FailureCode.DEPTH_LOCALIZATION_FAILED,
    "missing_obj_center": FailureCode.DEPTH_LOCALIZATION_FAILED,
    "finalize_not_completed": FailureCode.VERIFICATION_FAILED,
    # Navigation
    "navigator_missing": FailureCode.NAVIGATOR_UNAVAILABLE,
    "navigator_unavailable": FailureCode.NAVIGATOR_UNAVAILABLE,
    "missing_target": FailureCode.MISSING_TARGET,
    "move_to_position_failed": FailureCode.NAV_BLOCKED,
    "rotate_failed": FailureCode.ROTATE_FAILED,
    # Manipulation
    "zerograsp_failed": FailureCode.ZEROGRASP_FAILED,
    "ik_fail": FailureCode.IK_FAIL,
    "invalid_tcp_pose": FailureCode.IK_FAIL,
    "grasp_pose_unavailable": FailureCode.GRASP_FAIL,
    "grasp_failed": FailureCode.GRASP_FAIL,
    "grasp_slip": FailureCode.GRASP_SLIP,
    # Infrastructure
    "gripper_unavailable": FailureCode.GRIPPER_UNAVAILABLE,
    "arm_client_init_failed": FailureCode.ARM_UNAVAILABLE,
    "arm_unavailable": FailureCode.ARM_UNAVAILABLE,
    "skill_timeout": FailureCode.SKILL_TIMEOUT,
    "episode_timeout": FailureCode.EPISODE_TIMEOUT,
    # Contract / system
    "unsupported_skill": FailureCode.UNSUPPORTED_SKILL,
    "verification_failed": FailureCode.VERIFICATION_FAILED,
    "vla_no_effect": FailureCode.VLA_NO_EFFECT,
    "vla_policy_oob": FailureCode.VLA_POLICY_OOB,
}


@dataclass
class VerifierFinding:
    """Structured output from a skill verifier."""
    name: str
    verdict: str  # SUCCESS|FAIL|UNCERTAIN
    confidence: float
    evidence: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


@dataclass
class InspectionPacket:
    """
    Structured diagnostic packet generated after every skill execution.
    Used by VLM Verifiers and Recovery Managers to analyze failures.
    """
    episode_id: Optional[str]
    step_id: Optional[str]
    skill_name: str
    skill_args: Dict[str, Any]
    exec_result: Dict[str, Any]  # status, failure_code, elapsed_ms
    raw_metrics: Optional[Dict[str, Any]] = None
    verifier_outputs: Optional[Dict[str, Any]] = None
    world_snapshot: Optional[Dict[str, Any]] = None
    post_execution_observation: Optional[Dict[str, Any]] = None  # New: Fresh observation AFTER action
    budget: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


def map_reason_to_failure_code(reason: Optional[str]) -> FailureCode:
    """
    Map a free-text reason string to a structured FailureCode.

    Unmapped strings fall back to IMPORT_ERROR/MISSING_RUN prefixes,
    otherwise UNKNOWN.
    """
    if not reason:
        return FailureCode.UNKNOWN
    normalized = str(reason).lower()
    if normalized in _REASON_MAP:
        return _REASON_MAP[normalized]
    if normalized.startswith("import_error"):
        return FailureCode.IMPORT_ERROR
    if normalized.startswith("missing_run"):
        return FailureCode.MISSING_RUN
    if normalized.startswith("navigator"):
        return FailureCode.NAVIGATOR_UNAVAILABLE
    if normalized.startswith("gripper"):
        return FailureCode.GRIPPER_UNAVAILABLE
    return FailureCode.UNKNOWN

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
    analysis: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    processed_image_path: Optional[str] = None
    original_image_path: Optional[str] = None
    surface_mask_path: Optional[str] = None
    surface_mask_url: Optional[str] = None
    surface_mask_score: Optional[float] = None
    surface_mask_task_id: Optional[str] = None
    description: Optional[str] = None
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
            "analysis": self.analysis,
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
    failure_code: Optional[FailureCode] = None
    verified: Optional[bool] = None

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


@dataclass
class PlanContextEntry:
    """High-level summary of a single planning attempt."""

    plan_id: str
    goal: str
    planner_thought: Optional[str] = None
    planned_steps: List[str] = field(default_factory=list)
    status: str = "running"
    failure_reason: Optional[str] = None
    executed: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Compact representation for planner prompts."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "status": self.status,
            "planned_steps": self.planned_steps,
            "failure_reason": self.failure_reason,
            "executed": self.executed,
            "planner_thought": self.planner_thought,
            "timestamp": self.timestamp,
        }


@dataclass
class ExecutionTurn:
    """Fact log for an atomic observation or skill execution."""

    plan_id: str
    stage: str
    node: str
    status: str
    observation: Optional[str] = None
    action: Optional[str] = None
    detail: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "stage": self.stage,
            "node": self.node,
            "status": self.status,
            "observation": self.observation,
            "action": self.action,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


@dataclass
class ReflectionEntry:
    """Structured reflection outcome after a failed attempt."""

    plan_id: str
    goal: str
    trigger: str
    diagnosis: str
    adjustment_hint: Optional[str] = None
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "trigger": self.trigger,
            "diagnosis": self.diagnosis,
            "adjustment_hint": self.adjustment_hint,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
