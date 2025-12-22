"""
Deterministic, chain-agnostic recovery planner.

This module is intentionally free of BT/FC wiring. Callers provide
FailureCode + a RecoveryContext and receive a RecoveryDecision that can
be executed by any orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .task_structures import FailureCode, InspectionPacket


@dataclass
class RecoverySuggestion:
    action: str  # skill_name
    args: Dict[str, Any]
    why: str
    expected_effect: str
    cost: str  # "low", "med", "high"
    priority: int  # 1 (highest) to 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "args": self.args,
            "why": self.why,
            "expected_effect": self.expected_effect,
            "cost": self.cost,
            "priority": self.priority,
        }


@dataclass
class RecoveryContext:
    skill_name: str
    failure_code: FailureCode
    history: List[Any]
    world_state: Any
    inspection_report: Optional[Any] = None # Added for VLM report
    episode_id: Optional[str] = None
    step_id: Optional[int] = None
    task_goal: Optional[str] = None
    world_snapshot: Optional[Dict[str, Any]] = None
    history_tail: List[Dict[str, Any]] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)  # e.g., {"total":3,"per_code":{"ik_fail":1},"elapsed_s":1.2}


@dataclass
class RecoveryDecision:
    kind: str  # EXECUTE_ACTIONS | ESCALATE_L3 | ABORT
    level: str  # L1 | L2 | L3 | NONE
    actions: List[Dict[str, Any]]
    reason: str
    evidence: Dict[str, Any]


class RecoveryManager:
    """
    Table-driven recovery router with deterministic output for a given
    (failure_code, counters).
    """

    def __init__(
        self,
        *,
        max_total_attempts: int = 5,
        max_attempts_per_failure_code: int = 3,
        max_time_in_recovery_s: float = 120.0,
    ) -> None:
        self.max_total_attempts = max_total_attempts
        self.max_attempts_per_failure_code = max_attempts_per_failure_code
        self.max_time_in_recovery_s = max_time_in_recovery_s

        # Deterministic policies
        self.l1_policy: Dict[FailureCode, List[Dict[str, Any]]] = {
            FailureCode.IK_FAIL: [{"skill_name": "recover", "args": {"mode": "nudge_base", "dx": 0.05, "dy": 0.05}}],
            FailureCode.NAV_BLOCKED: [{"skill_name": "recover", "args": {"mode": "backoff", "distance": 0.2}}],
            FailureCode.GRASP_FAIL: [
                {"skill_name": "open_gripper", "args": {}},
                {"skill_name": "recover", "args": {"mode": "reset_arm"}},
            ],
            FailureCode.GRASP_SLIP: [
                {"skill_name": "open_gripper", "args": {}},
                {"skill_name": "recover", "args": {"mode": "reset_arm"}},
            ],
        }
        self.l2_policy: Dict[FailureCode, List[Dict[str, Any]]] = {
            FailureCode.ZEROGRASP_FAILED: [{"skill_name": "rotate_scan", "args": {"angle_deg": 30.0}}],
            FailureCode.NO_OBSERVATION: [{"skill_name": "search_area", "args": {"turns": 1, "angle_deg": 45.0}}],
            FailureCode.VERIFICATION_FAILED: [{"skill_name": "recover", "args": {"mode": "backoff", "distance": 0.1}}],
            FailureCode.IK_FAIL: [{"skill_name": "rotate_scan", "args": {"angle_deg": 15.0}}],
        }

    def _budget_state(self, ctx: RecoveryContext, failure_code: FailureCode) -> Dict[str, Any]:
        per_code = ctx.budget.get("per_code", {}) if ctx.budget else {}
        total_used = ctx.budget.get("total", 0) if ctx.budget else 0
        elapsed = ctx.budget.get("elapsed_s", 0.0) if ctx.budget else 0.0
        fc_key = failure_code.value
        fc_used = per_code.get(fc_key, 0)
        return {"total_used": total_used, "per_code_used": fc_used, "elapsed_s": elapsed}

    def _over_budget(self, state: Dict[str, Any]) -> bool:
        if state["total_used"] >= self.max_total_attempts:
            return True
        if state["per_code_used"] >= self.max_attempts_per_failure_code:
            return True
        if state["elapsed_s"] >= self.max_time_in_recovery_s:
            return True
        return False

    def handle_failure(self, failure_code: Optional[FailureCode], context: RecoveryContext) -> RecoveryDecision:
        fc = failure_code or FailureCode.UNKNOWN
        state = self._budget_state(context, fc)
        now_ts = time.time()

        if self._over_budget(state):
            return RecoveryDecision(
                kind="ABORT",
                level="L3",
                actions=[],
                reason="recovery_budget_exhausted",
                evidence={
                    "failure_code": fc.value,
                    "budget": {
                        "total_used": state["total_used"],
                        "per_code_used": state["per_code_used"],
                        "elapsed_s": state["elapsed_s"],
                        "max_total_attempts": self.max_total_attempts,
                        "max_attempts_per_failure_code": self.max_attempts_per_failure_code,
                        "max_time_in_recovery_s": self.max_time_in_recovery_s,
                    },
                    "episode_id": context.episode_id,
                    "step_id": context.step_id,
                    "timestamp": now_ts,
                },
            )

        # L1 reflex
        if fc in self.l1_policy:
            return RecoveryDecision(
                kind="EXECUTE_ACTIONS",
                level="L1",
                actions=self.l1_policy[fc],
                reason="l1_reflex",
                evidence={
                    "failure_code": fc.value,
                    "policy": "l1_policy",
                    "budget": state,
                    "episode_id": context.episode_id,
                    "step_id": context.step_id,
                    "timestamp": now_ts,
                },
            )

        # L2 local replanning
        if fc in self.l2_policy:
            return RecoveryDecision(
                kind="EXECUTE_ACTIONS",
                level="L2",
                actions=self.l2_policy[fc],
                reason="l2_local_replan",
                evidence={
                    "failure_code": fc.value,
                    "policy": "l2_policy",
                    "budget": state,
                    "episode_id": context.episode_id,
                    "step_id": context.step_id,
                    "timestamp": now_ts,
                },
            )

        # Fallback to L3 escalate
        return RecoveryDecision(
            kind="ESCALATE_L3",
            level="L3",
            actions=[],
            reason="no_policy_available",
            evidence={
                "failure_code": fc.value,
                "budget": state,
                "episode_id": context.episode_id,
                "step_id": context.step_id,
                "timestamp": now_ts,
            },
        )

    def suggest_recovery(self, ctx: RecoveryContext) -> List[RecoverySuggestion]:
        """
        Generate a list of recovery suggestions based on the recovery context.
        Does NOT execute any actions.
        """
        suggestions: List[RecoverySuggestion] = []
        
        # Extract failure code from context
        # Note: ctx.failure_code is already a FailureCode enum
        fc = ctx.failure_code or FailureCode.UNKNOWN
        
        # Extract inspection report if available
        report = getattr(ctx, "inspection_report", None)

        # 1. VLM-based Suggestions (Highest Priority)
        if report and report.optional_extra_suggestions:
            for idx, sugg in enumerate(report.optional_extra_suggestions):
                suggestions.append(RecoverySuggestion(
                    action=sugg.get("action", "unknown"),
                    args=sugg.get("args", {}),
                    why=sugg.get("reason", "VLM suggestion"),
                    expected_effect="fix_diagnosed_issue",
                    cost="med",
                    priority=1 + idx # 1, 2, 3...
                ))

        # 2. Heuristic Policy Suggestions (Fallback)
        # L1 Reflex
        if fc in self.l1_policy:
            for action in self.l1_policy[fc]:
                suggestions.append(RecoverySuggestion(
                    action=action["skill_name"],
                    args=action.get("args", {}),
                    why=f"L1 reflex for {fc.value}",
                    expected_effect="quick_fix",
                    cost="low",
                    priority=5
                ))
        
        # L2 Local Replan
        if fc in self.l2_policy:
            for action in self.l2_policy[fc]:
                suggestions.append(RecoverySuggestion(
                    action=action["skill_name"],
                    args=action.get("args", {}),
                    why=f"L2 replan for {fc.value}",
                    expected_effect="retry_with_change",
                    cost="med",
                    priority=6
                ))

        # 3. Generic Fallbacks
        if not suggestions:
             suggestions.append(RecoverySuggestion(
                action="recover",
                args={"mode": "backoff", "distance": 0.1},
                why="generic_fallback",
                expected_effect="reset_state",
                cost="low",
                priority=10
            ))

        return suggestions

        # Extract metrics/findings
        verifier_outputs = packet.verifier_outputs or {}

        # 1. NO_OBSERVATION / MISSING_TARGET
        if fc in (FailureCode.NO_OBSERVATION, FailureCode.MISSING_TARGET):
            suggestions.append(RecoverySuggestion(
                action="search_area",
                args={"turns": 1, "angle_deg": 45.0},
                why="Target not visible in current view",
                expected_effect="Rotate base to find target",
                cost="med",
                priority=1
            ))
            suggestions.append(RecoverySuggestion(
                action="recover",
                args={"mode": "backoff", "distance": 0.2},
                why="Might be too close to see target",
                expected_effect="Move back to widen FOV",
                cost="low",
                priority=2
            ))

        # 2. DEPTH_LOCALIZATION_FAILED
        elif fc == FailureCode.DEPTH_LOCALIZATION_FAILED:
            suggestions.append(RecoverySuggestion(
                action="recover",
                args={"mode": "nudge_base", "dx": 0.05, "dy": 0.05},
                why="Depth sensor noise or occlusion",
                expected_effect="Shift perspective slightly",
                cost="low",
                priority=1
            ))

        # 3. NAV_BLOCKED
        elif fc == FailureCode.NAV_BLOCKED:
            suggestions.append(RecoverySuggestion(
                action="recover",
                args={"mode": "backoff", "distance": 0.1},
                why="Navigation path blocked",
                expected_effect="Clear obstacle",
                cost="low",
                priority=1
            ))

        # 4. ZEROGRASP_FAILED
        elif fc == FailureCode.ZEROGRASP_FAILED:
            suggestions.append(RecoverySuggestion(
                action="rotate_scan",
                args={"angle_deg": 15.0},
                why="No valid grasp detected from current angle",
                expected_effect="Find better grasp approach",
                cost="med",
                priority=1
            ))

        # 5. IK_FAIL
        elif fc == FailureCode.IK_FAIL:
            suggestions.append(RecoverySuggestion(
                action="recover",
                args={"mode": "nudge_base", "dx": 0.05, "dy": 0.0},
                why="Target out of reach or kinematic singularity",
                expected_effect="Move base to reachable range",
                cost="low",
                priority=1
            ))

        # 6. GRASP_FAIL / GRASP_SLIP
        elif fc in (FailureCode.GRASP_FAIL, FailureCode.GRASP_SLIP):
            suggestions.append(RecoverySuggestion(
                action="open_gripper",
                args={},
                why="Grasp failed, ensure gripper is open",
                expected_effect="Reset gripper state",
                cost="low",
                priority=1
            ))
            suggestions.append(RecoverySuggestion(
                action="recover",
                args={"mode": "reset_arm"},
                why="Arm might be in collision state",
                expected_effect="Retract arm to safe pose",
                cost="med",
                priority=2
            ))

        # 7. VERIFICATION_FAILED
        elif fc == FailureCode.VERIFICATION_FAILED:
            # Check specific verifier output if available
            reason = str(verifier_outputs.get("evidence", {}).get("reason", "unknown"))
            if "distance" in reason or "workspace" in reason:
                suggestions.append(RecoverySuggestion(
                    action="recover",
                    args={"mode": "backoff", "distance": 0.15},
                    why=f"Verification failed: {reason}",
                    expected_effect="Adjust distance to target",
                    cost="low",
                    priority=1
                ))
            elif "gripper" in reason:
                suggestions.append(RecoverySuggestion(
                    action="execute_grasp",
                    args={},  # Retry grasp? Or maybe predict again?
                    why=f"Grasp verification failed: {reason}",
                    expected_effect="Retry grasp execution",
                    cost="high",
                    priority=2
                ))
            else:
                suggestions.append(RecoverySuggestion(
                    action="observe_scene",
                    args={"force_vlm": True},
                    why="Verification failed with unclear reason",
                    expected_effect="Re-analyze scene",
                    cost="med",
                    priority=1
                ))

        # 8. VLA_NO_EFFECT / VLA_POLICY_OOB
        elif fc in (FailureCode.VLA_NO_EFFECT, FailureCode.VLA_POLICY_OOB):
            suggestions.append(RecoverySuggestion(
                action="recover",
                args={"mode": "reset_arm"},
                why="VLA policy failed or had no effect",
                expected_effect="Reset to known state",
                cost="med",
                priority=1
            ))

        # Default fallback
        if not suggestions:
            suggestions.append(RecoverySuggestion(
                action="observe_scene",
                args={"force_vlm": True},
                why="Unknown failure, need fresh observation",
                expected_effect="Update world model",
                cost="med",
                priority=99
            ))

        return suggestions
