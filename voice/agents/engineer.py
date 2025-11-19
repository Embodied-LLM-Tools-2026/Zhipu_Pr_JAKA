"""
Engineer agent that translates Planner tickets into runnable actions.

如果配置了 DeepSeek API，将尝试自动生成 Python 模块；否则退化为占位注册。
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tools.logging.task_logger import log_error, log_info, log_success, log_warning  # type: ignore

from ..control.action_registry import ActionRegistry, ActionTicket, ActionEntry
from ..control.apis import RobotAPI


class EngineerAgent:
    """Receives ActionTicket and materialises actions."""

    def __init__(
        self,
        registry: ActionRegistry,
        robot_api: RobotAPI,
        *,
        actions_dir: str = "actions",
        tests_dir: str = "tests/actions",
    ) -> None:
        self.registry = registry
        self.robot_api = robot_api
        self.actions_dir = Path(actions_dir)
        self.tests_dir = Path(tests_dir)
        self.repo_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        init_file = self.actions_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Auto-generated actions package."""\n', encoding="utf-8")
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.llm_api_key = os.getenv("DEEPSEEK_ENGINEER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.llm_api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    def process_ticket(self, ticket: ActionTicket) -> Optional[ActionEntry]:
        action_name = self._sanitize_name(ticket.constraints.get("suggested_name") or f"custom_{ticket.ticket_id}")
        log_info(f"🛠️ EngineerAgent 接到需求单 {ticket.ticket_id}: {ticket.description}")
        if not self.llm_api_key:
            log_warning("⚠️ 未配置 DEEPSEEK_ENGINEER_API_KEY，使用占位动作")
            entry = self._register_placeholder(ticket, action_name)
            return entry

        try:
            code_text = self._generate_action_code(ticket, action_name)
            module_path = self._write_action_file(action_name, ticket.ticket_id, code_text)
            test_path = self._write_test_stub(action_name)
            entry = ActionEntry(
                name=action_name,
                description=ticket.description,
                inputs=ticket.inputs,
                outputs=ticket.outputs,
                code_path=module_path,
                tests=test_path,
                author="engineer_agent",
                version="0.1",
            )
            self.registry.register_action(entry)
            self.registry.close_ticket(ticket.ticket_id)
            log_success(f"✅ 已生成并注册动作 {action_name} -> {module_path}")
            return entry
        except Exception as exc:  # pragma: no cover - fallback path
            log_error(f"❌ EngineerAgent 自动生成失败: {exc}")
            return self._register_placeholder(ticket, action_name)

    def _register_placeholder(self, ticket: ActionTicket, action_name: str) -> ActionEntry:
        entry = ActionEntry(
            name=action_name,
            description=ticket.description,
            inputs=ticket.inputs,
            outputs=ticket.outputs,
            code_path=str(Path("actions") / f"{action_name}.py"),
            author="engineer_agent_stub",
            version="0.0",
        )
        self.registry.register_action(entry)
        log_warning(f"⚠️ 已登记占位action {entry.name}，需后续人工补全")
        return entry

    def _generate_action_code(self, ticket: ActionTicket, action_name: str) -> str:
        prompt = {
            "role": "engineer",
            "instructions": [
                "生成一个Python模块，用于actions目录，必须包含函数：run(api, runtime, **kwargs)。",
                "该函数需返回 voice.control.task_structures.ExecutionResult 或 dict(status/node/reason/evidence)。",
                "禁止使用未公开的库，只能通过 RobotAPI 提供的 navigation/perception/manipulation/planning/gripper 接口操作机器人。",
                "不要执行危险动作（如未授权的系统命令）。",
                "代码需包含必要的 import，例如 from voice.control.task_structures import ExecutionResult。",
                "如需记录日志，可使用 task_logger.log_info/log_warning。",
            ],
            "ticket": {
                "id": ticket.ticket_id,
                "goal": ticket.goal,
                "description": ticket.description,
                "inputs": ticket.inputs,
                "outputs": ticket.outputs,
            },
            "available_api": self._api_surface(),
            "action_name": action_name,
        }
        payload = {
            "model": os.getenv("DEEPSEEK_ENGINEER_MODEL", "deepseek-chat"),
            "messages": [
                {
                    "role": "system",
                    "content": "你是机器人应用的高级工程师，负责根据需求编写可靠的Python动作模块。",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False),
                },
            ],
            "temperature": 0.2,
            "top_p": 0.8,
        }
        url = f"{self.llm_api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("engineer_llm_no_choice")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if not isinstance(content, str):
            raise RuntimeError("engineer_llm_invalid_response")
        return self._extract_code(content)

    def _write_action_file(self, action_name: str, ticket_id: str, body: str) -> str:
        file_path = self.actions_dir / f"{action_name}.py"
        header = textwrap.dedent(
            f"""\
# Auto-generated by EngineerAgent for ticket {ticket_id}
# Edit with caution. Ensure any changes are reflected in ActionRegistry.
"""
        )
        file_path.write_text(header + body.rstrip() + "\n", encoding="utf-8")
        rel_path = self._relative_to_repo(file_path)
        return rel_path

    def _write_test_stub(self, action_name: str) -> Optional[str]:
        file_path = self.tests_dir / f"test_{action_name}.py"
        content = textwrap.dedent(
            f"""\
import importlib

def test_action_module_imports():
    module = importlib.import_module("actions.{action_name}")
    assert hasattr(module, "run")
"""
        )
        file_path.write_text(content, encoding="utf-8")
        return self._relative_to_repo(file_path)

    def _api_surface(self) -> Dict[str, List[str]]:
        surface = {}
        sections = {
            "navigation": self.robot_api.navigation,
            "perception": self.robot_api.perception,
            "manipulation": self.robot_api.manipulation,
            "gripper": self.robot_api.gripper,
        }
        for name, api in sections.items():
            methods = [
                attr
                for attr in dir(api)
                if not attr.startswith("_") and callable(getattr(api, attr, None))
            ]
            surface[name] = methods
        return surface

    @staticmethod
    def _sanitize_name(name: str) -> str:
        clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
        return clean.lower()

    @staticmethod
    def _extract_code(content: str) -> str:
        if "```" not in content:
            return content
        segments = []
        capture = False
        for line in content.splitlines():
            if line.strip().startswith("```"):
                capture = not capture
                continue
            if capture:
                segments.append(line)
        return "\n".join(segments) if segments else content

    def _relative_to_repo(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)
