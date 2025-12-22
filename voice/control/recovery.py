from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .task_structures import FailureCode
from ..utils.config import Config

@dataclass
class RecoveryPlan:
    level: str  # "L1", "L2", "L3"
    actions: List[Dict[str, Any]]  # [{"skill": "name", "args": {}}]
    retry_original: bool = False
    max_retries: int = 1

def get_recovery_plan(failure_code: Optional[FailureCode], attempt_idx: int = 1) -> Optional[RecoveryPlan]:
    """
    Maps a FailureCode to a specific RecoveryPlan with escalation logic.
    attempt_idx: 1-based index of consecutive failures for this node.
    """
    if not failure_code:
        return None
        
    # Escalation Thresholds
    L1_LIMIT = 2
    L2_LIMIT = 4 # Total attempts (L1 + L2)
    
    # Dynamic Policy based on Router Config
    vla_fallback_plan = RecoveryPlan(
        level="L2",
        actions=[
            {"skill": "open_gripper", "args": {}},
            {"skill": "recover", "args": {"mode": "reset_arm"}},
            {"skill": "vla_execute", "args": {"instruction": "grasp object"}}
        ],
        retry_original=False, # Don't retry classic, switch to VLA
        max_retries=1
    )

    # --- L3: Global / Human Intervention ---
    l3_plan = RecoveryPlan(
        level="L3",
        actions=[], # No local actions, requires planner intervention or human
        retry_original=False
    )

    # --- L2: Local Re-planning ---
    l2_scan_plan = RecoveryPlan(
        level="L2",
        actions=[{"skill": "rotate_scan", "args": {"angle_deg": 30.0}}],
        retry_original=True,
        max_retries=1
    )

    # Table driven policy
    # Structure: FailureCode -> { "L1": Plan, "L2": Plan }
    # If a level is missing, it might fallback or escalate immediately.
    
    # Helper to select plan based on attempt
    def select_plan(l1: Optional[RecoveryPlan], l2: Optional[RecoveryPlan]) -> Optional[RecoveryPlan]:
        if attempt_idx <= L1_LIMIT:
            return l1 or l2 or l3_plan
        elif attempt_idx <= L2_LIMIT:
            return l2 or l3_plan
        else:
            return l3_plan

    # 1. Navigation Blocked
    if failure_code == FailureCode.NAV_BLOCKED:
        return select_plan(
            l1=RecoveryPlan(level="L1", actions=[{"skill": "recover", "args": {"mode": "backoff", "distance": 0.2}}], retry_original=True),
            l2=RecoveryPlan(level="L2", actions=[{"skill": "navigate_area", "args": {"target": "random_nearby"}}], retry_original=True)
        )

    # 2. IK Failure
    if failure_code == FailureCode.IK_FAIL:
        return select_plan(
            l1=RecoveryPlan(level="L1", actions=[{"skill": "recover", "args": {"mode": "nudge_base", "dx": 0.05, "dy": 0.05}}], retry_original=True),
            l2=l2_scan_plan # Try scanning from new angle
        )

    # 3. Grasp Failure
    if failure_code in [FailureCode.GRASP_FAIL, FailureCode.GRASP_SLIP]:
        if Config.ENABLE_VLA_ROUTER:
            return vla_fallback_plan
        return select_plan(
            l1=RecoveryPlan(level="L1", actions=[{"skill": "open_gripper", "args": {}}, {"skill": "recover", "args": {"mode": "reset_arm"}}], retry_original=True),
            l2=vla_fallback_plan # Escalate to VLA if L1 fails even if router disabled? Or just L2 scan. Let's use VLA as L2 if available, else scan.
               if Config.ENABLE_VLA_ROUTER else l2_scan_plan
        )

    # 4. Perception / ZeroGrasp
    if failure_code == FailureCode.ZEROGRASP_FAILED:
        if Config.ENABLE_VLA_ROUTER:
            return vla_fallback_plan
        return select_plan(
            l1=RecoveryPlan(level="L2", actions=[{"skill": "rotate_scan", "args": {"angle_deg": 15.0}}], retry_original=True), # Start at L2 immediately
            l2=l3_plan
        )
        
    if failure_code == FailureCode.NO_OBSERVATION:
        return select_plan(
            l1=RecoveryPlan(level="L2", actions=[{"skill": "rotate_scan", "args": {"angle_deg": 45.0}}], retry_original=True),
            l2=l3_plan
        )

    # Default / Fallback
    return l3_plan
