"""
Lightweight Plan runner for FunctionCall mode.

Takes a validated Plan (task_plan.schema.Plan), expands macros to skill
sequences, executes via SkillExecutor + RecoveryManager, and enforces
budgets (tool calls / wall time).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..control.executor import SkillExecutor, SkillRuntime
from ..control.recovery_manager import RecoveryManager, RecoveryContext
from ..control.task_structures import ExecutionResult, FailureCode, PlanNode
from ..control.world_model import WorldModel
from .task_plan.schema import Plan, Subtask


@dataclass
class MacroStep:
    skill_name: str
    args: Dict[str, Any]


class MacroRegistry:
    """Table-driven macro expander."""

    def __init__(self) -> None:
        self._registry: Dict[str, Any] = {}
        self.register_default()

    def register_default(self) -> None:
        self._registry["fetch_place"] = self._macro_fetch_place
        self._registry["fetch_only"] = self._macro_fetch_only
        self._registry["place_only"] = self._macro_place_only

    def expand(self, subtask: Subtask) -> List[MacroStep]:
        fn = self._registry.get(subtask.type)
        if not fn:
            return [MacroStep(skill_name=subtask.type, args=subtask.params or {})]
        return fn(subtask)

    def _macro_fetch_place(self, subtask: Subtask) -> List[MacroStep]:
        obj = subtask.params.get("object") or subtask.params.get("target")
        src = subtask.params.get("from")
        dst = subtask.params.get("to")
        steps: List[MacroStep] = []
        if src:
            steps.append(MacroStep("navigate_area", {"area": src}))
        steps.extend(
            [
                MacroStep("search_area", {"target": obj}),
                MacroStep("approach_far", {"target": obj}),
                MacroStep("finalize_target_pose", {"target": obj}),
                MacroStep("predict_grasp_point", {"target": obj}),
                MacroStep("execute_grasp", {"target": obj}),
            ]
        )
        if dst:
            steps.append(MacroStep("navigate_area", {"area": dst}))
            steps.append(MacroStep("place", {"target": obj}))
        return steps

    def _macro_fetch_only(self, subtask: Subtask) -> List[MacroStep]:
        obj = subtask.params.get("object") or subtask.params.get("target")
        src = subtask.params.get("from")
        steps: List[MacroStep] = []
        if src:
            steps.append(MacroStep("navigate_area", {"area": src}))
        steps.extend(
            [
                MacroStep("search_area", {"target": obj}),
                MacroStep("approach_far", {"target": obj}),
                MacroStep("finalize_target_pose", {"target": obj}),
                MacroStep("predict_grasp_point", {"target": obj}),
                MacroStep("execute_grasp", {"target": obj}),
            ]
        )
        return steps

    def _macro_place_only(self, subtask: Subtask) -> List[MacroStep]:
        obj = subtask.params.get("object") or subtask.params.get("target")
        dst = subtask.params.get("to")
        steps: List[MacroStep] = []
        if dst:
            steps.append(MacroStep("navigate_area", {"area": dst}))
        steps.append(MacroStep("place", {"target": obj}))
        return steps


@dataclass
class PlanRunnerResult:
    ok: bool
    plan_id: str
    subtasks_status: List[Dict[str, Any]]
    final_outcome: str
    last_failure: Optional[Dict[str, Any]] = None
    trace_pointer: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class PlanRunner:
    def __init__(
        self,
        executor: SkillExecutor,
        recovery_manager: RecoveryManager,
        world: WorldModel,
        *,
        max_tool_calls: int = 50,
        max_time_s: float = 300.0,
    ) -> None:
        self.executor = executor
        self.recovery_manager = recovery_manager
        self.world = world
        self.max_tool_calls = max_tool_calls
        self.max_time_s = max_time_s
        self.macro_registry = MacroRegistry()
        self._plan_progress: Dict[str, Dict[str, str]] = {}

    def run(self, plan: Plan, runtime: SkillRuntime) -> PlanRunnerResult:
        start_ts = time.time()
        tool_calls = 0
        plan_state = self._plan_progress.get(plan.plan_id, {})
        subtasks_status: List[Dict[str, Any]] = []
        last_failure: Optional[Dict[str, Any]] = None
        trace_ptr = {
            "episode_id": runtime.extra.get("episode_id") if runtime and runtime.extra else None,
            "trace_path": getattr(self.executor.trace_logger, "file_path", None),
        }

        for st in plan.subtasks:
            status = plan_state.get(st.id, "PENDING")
            if status == "DONE":
                subtasks_status.append({"id": st.id, "status": "DONE"})
                continue
            # Dependencies
            if any(plan_state.get(dep) != "DONE" for dep in st.depends_on):
                plan_state[st.id] = "SKIPPED"
                subtasks_status.append({"id": st.id, "status": "SKIPPED"})
                continue
            plan_state[st.id] = "RUNNING"
            steps = self.macro_registry.expand(st)
            step_failed = False
            for step in steps:
                if tool_calls >= self.max_tool_calls or (time.time() - start_ts) > self.max_time_s:
                    plan_state[st.id] = "FAILED"
                    reason = "plan_budget_exceeded"
                    last_failure = {"failure_code": FailureCode.EPISODE_TIMEOUT.value, "evidence": {"tool_calls": tool_calls}}
                    subtasks_status.append({"id": st.id, "status": "FAILED", "reason": reason})
                    self._plan_progress[plan.plan_id] = plan_state
                    return PlanRunnerResult(
                        ok=False,
                        plan_id=plan.plan_id,
                        subtasks_status=subtasks_status,
                        final_outcome="ABORT",
                        last_failure=last_failure,
                        trace_pointer=trace_ptr,
                        reason=reason,
                    )
                tool_calls += 1
                node = PlanNode(type="action", name=step.skill_name, args=step.args)
                result = self.executor.execute(node, runtime)
                if result.status != "success":
                    last_failure = {
                        "failure_code": result.failure_code.value if result.failure_code else None,
                        "evidence": result.evidence,
                    }
                    step_failed = True
                    break
            if step_failed:
                plan_state[st.id] = "FAILED"
                subtasks_status.append({"id": st.id, "status": "FAILED", "last_failure": last_failure})
                break
            plan_state[st.id] = "DONE"
            subtasks_status.append({"id": st.id, "status": "DONE"})

        self._plan_progress[plan.plan_id] = plan_state
        final_success = all(s.get("status") == "DONE" or s.get("status") == "SKIPPED" for s in subtasks_status)
        outcome = "SUCCESS" if final_success else "FAILED"
        return PlanRunnerResult(
            ok=final_success,
            plan_id=plan.plan_id,
            subtasks_status=subtasks_status,
            final_outcome=outcome,
            last_failure=last_failure,
            trace_pointer=trace_ptr,
            reason=None if final_success else "subtask_failed",
        )
