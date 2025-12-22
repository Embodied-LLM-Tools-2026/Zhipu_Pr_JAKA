"""Robot control primitives: executor, APIs, world model, registry."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

__all__ = [
    "SkillExecutor",
    "ActionRegistry",
    "ActionEntry",
    "ActionTicket",
    "PrimitiveEntry",
    "RobotAPI",
    "WorldModel",
    "ExecutionResult",
    "PlanNode",
    "PlanContextEntry",
    "ExecutionTurn",
    "ObservationPhase",
    "RecoveryManager",
    "RecoveryContext",
    "RecoveryDecision",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "SkillExecutor": "voice.control.executor",
    "ActionRegistry": "voice.control.action_registry",
    "ActionEntry": "voice.control.action_registry",
    "ActionTicket": "voice.control.action_registry",
    "PrimitiveEntry": "voice.control.action_registry",
    "RobotAPI": "voice.control.apis",
    "WorldModel": "voice.control.world_model",
    "ExecutionResult": "voice.control.task_structures",
    "PlanNode": "voice.control.task_structures",
    "PlanContextEntry": "voice.control.task_structures",
    "ExecutionTurn": "voice.control.task_structures",
    "ObservationPhase": "voice.control.task_structures",
    "RecoveryManager": "voice.control.recovery_manager",
    "RecoveryContext": "voice.control.recovery_manager",
    "RecoveryDecision": "voice.control.recovery_manager",
}


def __getattr__(name: str) -> Any:
    """
    Lazily import heavy submodules to avoid pulling in optional dependencies
    when only lightweight structures are needed.
    """
    module_path = _LAZY_IMPORTS.get(name)
    if not module_path:
        raise AttributeError(f"module 'voice.control' has no attribute '{name}'")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> Any:
    return sorted(list(__all__) + [key for key in globals() if not key.startswith("_")])
