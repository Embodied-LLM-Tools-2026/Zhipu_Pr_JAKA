"""
JSON schema + validation for long-horizon task plans used by TaskExecutive/LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_SUBTASK_TYPES: Set[str] = {
    "observe",
    "navigate",
    "pick",
    "place",
    "handover",
    "plan_subgoal",
    "wait",
    "operate",
    # Macro-style subtasks used by PlanRunner/execute_plan tool
    "fetch_place",
    "fetch_only",
    "place_only",
}

REQUIRED_PARAMS: Dict[str, Set[str]] = {
    "observe": {"target"},
    "navigate": {"target"},
    "pick": {"target"},
    "place": {"target"},
    "handover": {"target"},
    "plan_subgoal": {"goal"},
    "wait": {"duration_s"},
    "operate": {"action"},
    # Macro subtasks
    "fetch_place": {"object", "from", "to"},
    "fetch_only": {"object", "from"},
    "place_only": {"object", "to"},
}


@dataclass
class RetryPolicy:
    max_retries: int = 0
    backoff_s: float = 0.0


@dataclass
class Subtask:
    id: str
    type: str
    params: Dict[str, Any]
    done_if: Optional[str] = None
    timeout_s: Optional[float] = None
    retry_policy: Optional[RetryPolicy] = None
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Plan:
    goal: str
    plan_id: str
    subtasks: List[Subtask]
    constraints: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class PlanValidationError(Exception):
    """Raised when a plan fails validation constraints."""
    pass


def _validate_ids(subtasks: List[Dict[str, Any]]) -> None:
    ids = [t.get("id") for t in subtasks]
    if any(i is None or i == "" for i in ids):
        raise PlanValidationError("All subtasks must have a non-empty 'id'.")
    
    seen = set()
    duplicates = set()
    for i in ids:
        if i in seen:
            duplicates.add(i)
        seen.add(i)
    
    if duplicates:
        raise PlanValidationError(f"Duplicate subtask IDs found: {duplicates}")


def _validate_depends_on(subtasks: List[Dict[str, Any]]) -> None:
    ids = {t["id"] for t in subtasks}
    
    # 1. Check existence
    for t in subtasks:
        deps = t.get("depends_on")
        if deps:
            if not isinstance(deps, list):
                raise PlanValidationError(f"Subtask '{t['id']}' depends_on must be a list.")
            for dep in deps:
                if dep not in ids:
                    raise PlanValidationError(f"Subtask '{t['id']}' depends on unknown ID '{dep}'.")

    # 2. Cycle detection
    graph = {t["id"]: set(t.get("depends_on") or []) for t in subtasks}
    visited = {}  # None: unvisited, "visiting", "done"

    def dfs(node: str, path: List[str]) -> None:
        state = visited.get(node)
        if state == "visiting":
            cycle = " -> ".join(path + [node])
            raise PlanValidationError(f"Dependency cycle detected: {cycle}")
        if state == "done":
            return
        
        visited[node] = "visiting"
        path.append(node)
        
        for nei in graph.get(node, []):
            dfs(nei, path)
            
        path.pop()
        visited[node] = "done"

    for node in graph:
        if node not in visited:
            dfs(node, [])


def _validate_type_and_params(subtasks: List[Dict[str, Any]]) -> None:
    for t in subtasks:
        tid = t.get("id")
        ttype = t.get("type")
        
        if not ttype:
            raise PlanValidationError(f"Subtask '{tid}' missing 'type'.")
            
        if ttype not in ALLOWED_SUBTASK_TYPES:
            raise PlanValidationError(f"Subtask '{tid}' has invalid type '{ttype}'. Allowed: {sorted(list(ALLOWED_SUBTASK_TYPES))}")
        
        required = REQUIRED_PARAMS.get(ttype, set())
        params = t.get("params")
        if not isinstance(params, dict):
             raise PlanValidationError(f"Subtask '{tid}' params must be a dictionary.")
             
        missing = [p for p in required if p not in params]
        if missing:
            raise PlanValidationError(f"Subtask '{tid}' (type='{ttype}') missing required params: {missing}")
        
        # Validate timeout
        timeout = t.get("timeout_s")
        if timeout is not None:
            try:
                timeout_f = float(timeout)
            except (ValueError, TypeError):
                raise PlanValidationError(f"Subtask '{tid}' timeout_s must be numeric.")
            
            if timeout_f <= 0 or timeout_f > 3600:
                raise PlanValidationError(f"Subtask '{tid}' timeout_s {timeout_f} out of reasonable range (0, 3600].")


def validate_plan(plan_dict: Dict[str, Any]) -> Plan:
    """
    Validates a raw dictionary against the Plan schema.
    Returns a strongly-typed Plan object if valid.
    Raises PlanValidationError with clear messages if invalid.
    """
    if not isinstance(plan_dict, dict):
        raise PlanValidationError("Plan must be a JSON object (dict).")
        
    goal = plan_dict.get("goal")
    plan_id = plan_dict.get("plan_id")
    
    if not goal or not isinstance(goal, str):
        raise PlanValidationError("Plan must have a 'goal' string.")
    if not plan_id or not isinstance(plan_id, str):
        raise PlanValidationError("Plan must have a 'plan_id' string.")
        
    subtasks = plan_dict.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise PlanValidationError("Plan must have a non-empty 'subtasks' list.")
        
    # Run validations
    _validate_ids(subtasks)
    _validate_depends_on(subtasks)
    _validate_type_and_params(subtasks)

    # Construct Objects
    def build_retry(obj: Any) -> Optional[RetryPolicy]:
        if not obj:
            return None
        return RetryPolicy(
            max_retries=int(obj.get("max_retries", 0)),
            backoff_s=float(obj.get("backoff_s", 0.0)),
        )

    subtask_objs: List[Subtask] = []
    for t in subtasks:
        rp = build_retry(t.get("retry_policy"))
        subtask_objs.append(
            Subtask(
                id=t["id"],
                type=t["type"],
                params=t.get("params") or {},
                done_if=t.get("done_if"),
                timeout_s=float(t["timeout_s"]) if t.get("timeout_s") is not None else None,
                retry_policy=rp,
                depends_on=t.get("depends_on") or [],
            )
        )

    return Plan(
        goal=goal,
        plan_id=plan_id,
        subtasks=subtask_objs,
        constraints=plan_dict.get("constraints"),
        metadata=plan_dict.get("metadata"),
    )
