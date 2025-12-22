"""
TaskExecutive: Executes long-horizon plans by orchestrating subtasks,
handling state transitions, and managing recovery via RecoveryManager.
"""

from __future__ import annotations

import time
import enum
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.logging.task_logger import log_error, log_info, log_success, log_warning
from tools.logging.trace_logger import TraceLogger

from ..control.executor import SkillExecutor, SkillRuntime
from ..control.world_model import WorldModel
from ..control.recovery_manager import RecoveryManager, RecoveryContext, RecoveryKind
from ..control.task_structures import ExecutionResult, FailureCode, PlanNode
from .task_plan.schema import Plan, Subtask
from .task_plan.macros import MacroRegistry, MacroStep


class SubtaskState(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class PlanExecutionResult:
    success: bool
    plan_id: str
    final_state: Dict[str, SubtaskState]
    failure_reason: Optional[str] = None
    failed_subtask_id: Optional[str] = None
    trace_id: Optional[str] = None


class TaskExecutive:
    def __init__(
        self,
        executor: SkillExecutor,
        world_model: WorldModel,
        recovery_manager: RecoveryManager,
    ) -> None:
        self.executor = executor
        self.world_model = world_model
        self.recovery_manager = recovery_manager
        self.trace_logger = executor.trace_logger or TraceLogger()

    def run_plan(self, plan: Plan, runtime: SkillRuntime) -> PlanExecutionResult:
        """
        Executes a plan sequentially.
        """
        log_info(f"📋 [TaskExecutive] Starting plan: {plan.plan_id} (Goal: {plan.goal})")
        
        subtask_states: Dict[str, SubtaskState] = {t.id: SubtaskState.PENDING for t in plan.subtasks}
        
        # Topological sort or just sequential? 
        # The schema validation ensures no cycles, but for now we assume the list order is the execution order
        # or we respect dependencies.
        # Simple implementation: Iterate through list, check dependencies.
        
        # Since the user asked for "sequential execution" of a list, we will iterate.
        # But we should check if dependencies are met.
        
        for subtask in plan.subtasks:
            # 1. Check Dependencies
            if not self._check_dependencies(subtask, subtask_states):
                log_warning(f"⏭️ [TaskExecutive] Skipping {subtask.id} due to failed dependencies.")
                subtask_states[subtask.id] = SubtaskState.SKIPPED
                continue

            # 2. Check Done Condition (Pre-check)
            if subtask.done_if and self._check_done_condition(subtask.done_if):
                log_info(f"✅ [TaskExecutive] Subtask {subtask.id} already satisfied: {subtask.done_if}")
                subtask_states[subtask.id] = SubtaskState.DONE
                continue

            # 3. Execute
            subtask_states[subtask.id] = SubtaskState.RUNNING
            success = self._execute_subtask_with_recovery(subtask, runtime)
            
            if success:
                subtask_states[subtask.id] = SubtaskState.DONE
            else:
                subtask_states[subtask.id] = SubtaskState.FAILED
                log_error(f"❌ [TaskExecutive] Plan failed at subtask {subtask.id}")
                return PlanExecutionResult(
                    success=False,
                    plan_id=plan.plan_id,
                    final_state=subtask_states,
                    failure_reason=f"Subtask {subtask.id} failed",
                    failed_subtask_id=subtask.id
                )

        log_success(f"🎉 [TaskExecutive] Plan {plan.plan_id} completed successfully.")
        return PlanExecutionResult(
            success=True,
            plan_id=plan.plan_id,
            final_state=subtask_states
        )

    def _check_dependencies(self, subtask: Subtask, states: Dict[str, SubtaskState]) -> bool:
        for dep_id in subtask.depends_on:
            if states.get(dep_id) != SubtaskState.DONE:
                return False
        return True

    def _execute_subtask_with_recovery(self, subtask: Subtask, runtime: SkillRuntime) -> bool:
        """
        Executes a single subtask, handling retries and recovery via RecoveryManager.
        Uses MacroRegistry to expand high-level tasks into skill sequences.
        """
        log_info(f"▶️ [TaskExecutive] Executing subtask {subtask.id}: {subtask.type} {subtask.params}")
        
        # 1. Expand Macro
        steps = MacroRegistry.expand(subtask.type, subtask.params)
        log_info(f"   [Macro] Expanded to {len(steps)} steps: {[s.skill_name for s in steps]}")
        
        # Update runtime with subtask context
        runtime.extra["subtask_id"] = subtask.id
        
        # 2. Execute Steps
        for i, step in enumerate(steps):
            log_info(f"   [Step {i+1}/{len(steps)}] {step.skill_name} {step.args}")
            
            node = PlanNode(type="action", name=step.skill_name, args=step.args)
            
            # Execute with built-in L1/L2 recovery
            result = self.executor.execute(node, runtime)
            
            if result.status != "success":
                if step.optional:
                    log_warning(f"⚠️ [TaskExecutive] Optional step {step.skill_name} failed. Continuing.")
                    continue
                else:
                    log_error(f"❌ [TaskExecutive] Step {step.skill_name} failed. Aborting subtask.")
                    return False
        
        # 3. Post-condition check
        if subtask.done_if:
            if not self._check_done_condition(subtask.done_if):
                log_warning(f"⚠️ [TaskExecutive] Subtask {subtask.id} action succeeded but done_if '{subtask.done_if}' not met.")
                return False
                
        return True

    def _map_subtask_to_node(self, subtask: Subtask) -> PlanNode:
        """
        Deprecated: Use MacroRegistry.expand instead.
        """
        pass

    def _check_done_condition(self, condition: str) -> bool:
        """
        Evaluates predicates like 'at(kitchen)', 'holding(cup)'.
        """
        # 1. Parse
        match = re.match(r"(\w+)\((.*)\)", condition)
        if not match:
            log_warning(f"Invalid done_if format: {condition}")
            return False
        
        predicate, args_str = match.groups()
        args = [a.strip() for a in args_str.split(",")]
        
        # 2. Evaluate
        if predicate == "at":
            # Check robot location vs area
            target_area = args[0]
            # In a real system, check self.world_model.robot["pose"] vs self.world_model.areas[target_area]
            # For now, check task_memory or last status?
            # Or assume if we just navigated there successfully, we are there?
            # But done_if is for skipping.
            # Let's check if robot is close to area center.
            area = self.world_model.areas.get(target_area)
            if not area or not area.pose:
                return False
            # Simple distance check
            robot_pose = self.world_model.robot.get("pose", [0,0,0])
            dist = ((robot_pose[0]-area.pose[0])**2 + (robot_pose[1]-area.pose[1])**2)**0.5
            return dist < 0.5 # 0.5m tolerance

        elif predicate == "holding":
            target_obj = args[0]
            # Check gripper state
            holding = self.world_model.robot.get("holding")
            return holding == target_obj

        elif predicate == "on_table":
            obj_id = args[0]
            # Check object state
            obj = self.world_model.objects.get(obj_id)
            if not obj: return False
            # Check if z is table height? Or 'seen_in' attribute?
            return obj.seen_in == "table" # Example

        elif predicate == "area_clear":
            # Check if no objects in area
            return False # Hard to verify without perception

        return False
