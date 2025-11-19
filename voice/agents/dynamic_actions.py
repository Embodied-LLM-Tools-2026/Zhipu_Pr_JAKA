"""
Runtime loader for dynamically registered high-level actions.

Actions are stored as Python modules (e.g. actions/pour_drink.py) that expose
`run(api, runtime, **kwargs)` returning an ExecutionResult-compatible dict.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tools.logging.task_logger import log_error, log_info, log_warning  # type: ignore

from ..control.task_structures import ExecutionResult
from ..control.apis import RobotAPI
from ..control.action_registry import ActionRegistry, ActionEntry


class DynamicActionRunner:
    """Loads and executes custom actions from the actions/ directory."""

    def __init__(
        self,
        robot_api: RobotAPI,
        registry: ActionRegistry,
        *,
        base_package: str = "actions",
        search_path: Optional[str] = None,
    ) -> None:
        self.robot_api = robot_api
        self.registry = registry
        self.base_package = base_package
        self.search_path = search_path or os.getenv("ACTION_LIBRARY_PATH")
        if self.search_path:
            abs_path = Path(self.search_path).resolve()
            if str(abs_path) not in sys.path:
                sys.path.insert(0, str(abs_path))

    def has_action(self, name: str) -> bool:
        if self.registry.get_action(name):
            return True
        module_path = self._module_from_name(name)
        try:
            importlib.util.find_spec(module_path)
            return True
        except Exception:
            return False

    def execute(self, name: str, args: Dict[str, Any], runtime) -> ExecutionResult:
        entry = self.registry.get_action(name)
        module_name = self._module_from_entry(name, entry)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            log_error(f"❌ 动作 {name} 模块加载失败: {exc}")
            return ExecutionResult(status="failure", node=name, reason=f"import_error:{exc}")
        if not hasattr(module, "run"):
            log_error(f"❌ 动作 {name} 缺少 run(api, runtime, **kwargs) 方法")
            return ExecutionResult(status="failure", node=name, reason="missing_run")
        try:
            result = module.run(self.robot_api, runtime, **args)
        except Exception as exc:  # pragma: no cover
            log_error(f"❌ 动作 {name} 执行异常: {exc}")
            return ExecutionResult(status="failure", node=name, reason=str(exc))
        return self._normalize_result(name, result)

    def _module_from_entry(self, name: str, entry: Optional[ActionEntry]) -> str:
        if entry and entry.code_path:
            path = entry.code_path.replace("/", ".").replace("\\", ".")
            if path.endswith(".py"):
                path = path[:-3]
            return path
        return self._module_from_name(name)

    def _module_from_name(self, name: str) -> str:
        safe_name = name.replace("-", "_")
        if self.base_package:
            return f"{self.base_package}.{safe_name}"
        return safe_name

    @staticmethod
    def _normalize_result(name: str, result: Any) -> ExecutionResult:
        if isinstance(result, ExecutionResult):
            return result
        if isinstance(result, dict):
            status = result.get("status", "success")
            reason = result.get("reason")
            evidence = result.get("evidence")
            elapsed = result.get("elapsed")
            return ExecutionResult(
                status=status,
                node=name,
                reason=reason,
                evidence=evidence,
                elapsed=elapsed,
            )
        log_warning(f"⚠️ 动作 {name} 返回未知类型 {type(result)}, 将视为成功")
        return ExecutionResult(status="success", node=name)
