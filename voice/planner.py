"""
Behaviour planner that produces a constrained behaviour tree plan.

This implementation follows the redesign guideline where the LLM is
responsible for high level planning. For robustness the planner falls
back to a deterministic sequence when the LLM is unavailable or the
response fails validation.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional
import time

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

from task_logger import log_error, log_info, log_success, log_warning  # type: ignore

from .task_structures import CompiledPlan, PlanNode, PlanContextEntry, ExecutionTurn


class PlanValidationError(Exception):
    """Raised when a plan does not conform to the required schema."""


class ConditionFailed(Exception):
    """Raised when a check node fails while compiling the plan."""


class BehaviorPlanner:
    """Planner that queries an LLM for a structured behaviour tree plan."""

    def __init__(self, llm_api_key: Optional[str] = None, llm_model: str = "deepseek-chat") -> None:
        self.llm_api_key = "sk-860c486b30454e0abb404b7aa3deb3dc"
        self.llm_model = llm_model
        self.llm_api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.max_retries = int(os.getenv("LLM_PLAN_MAX_RETRIES", "5"))

    # ------------------------------------------------------------------
    def make_plan(self, goal: str, world_model, plan_context: Optional[List[Dict[str, Any]]] = None) -> CompiledPlan:
        """Return a compiled behaviour tree for the current goal."""
        plan_dict: Optional[Dict[str, Any]] = None
        metadata: Dict[str, Any] = {}
        if self.llm_api_key:
            last_error: Optional[Exception] = None
            for attempt in range(1, self.max_retries + 1):
                start_time = time.time()
                log_info(f"🤖 正在调用LLM生成行为树 (尝试 {attempt}/{self.max_retries}): {goal}")
                try:
                    plan_dict = self._request_plan(goal, world_model.snapshot(), plan_context)
                    metadata["source"] = "llm"
                    duration = time.time() - start_time
                    log_success(f"✅ LLM 行为树生成成功，耗时 {duration:.2f} 秒")
                    break
                except Exception as exc:
                    duration = time.time() - start_time
                    metadata["source"] = "llm_error"
                    metadata["error"] = str(exc)
                    log_warning(
                        f"⚠️ LLM 生成行为树失败(第 {attempt} 次，耗时 {duration:.2f} 秒)，准备重试: {exc}"
                    )
                    last_error = exc
            if plan_dict is None:
                log_error("❌ LLM 多次生成行为树失败，终止规划流程")
                raise RuntimeError(f"llm_plan_failed: {last_error}")
        else:
            plan_dict = self._fallback_plan(goal)
            metadata["source"] = "fallback"

        root = PlanNode.from_dict(plan_dict)
        try:
            steps = self._compile_plan(root, world_model)
        except ConditionFailed as exc:
            metadata["compile_fallback"] = str(exc)
            fallback_dict = self._fallback_plan(goal)
            root = PlanNode.from_dict(fallback_dict)
            steps = self._compile_plan(root, world_model)
        return CompiledPlan(root=root, steps=steps, metadata=metadata)

    # ------------------------------------------------------------------
    # LLM interaction & validation
    # ------------------------------------------------------------------
    def _request_plan(
        self, goal: str, world_snapshot: Dict[str, Any], plan_context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        prompt: Dict[str, Any] = {
            "schema": "BT/1.0",
            "goal": goal,
            "world": world_snapshot,
            "allowed_nodes": ["sequence", "selector", "check", "action", "repeat_until"],
            "allowed_actions": [
                "observe_scene",
                "rotate_scan",
                "search_area",
                "navigate_area",
                "approach_far",
                "finalize_target_pose",
                "predict_grasp_point",
                "execute_grasp",
                "open_gripper",
                "close_gripper",
                "handover_item",
                "return_home",
                "recover",
            ],
            "action_docs": {
                "observe_scene": "触发 RGBD/VLM 观测，刷新对场景的认知，可选force_vlm=true强制调用VLM重新检测",
                "rotate_scan": "原地旋转或摆头扫描寻找目标",
                "search_area": "在当前位置执行多次旋转或扫描，用于重新搜寻目标",
                "navigate_area": "通过导航到达指定区域/marker/坐标，area字段对应world.areas中的名字",
                "approach_far": "当距离目标物体大于2米时沿机器人与目标连线迈大步靠近",
                "finalize_target_pose": "进行精确定位调整底盘姿态至抓取位",
                "predict_grasp_point": "在精确定位后，调用ZeroGrasp等接口，预测抓取点",
                "execute_grasp": "根据预测结果执行抓取策略",
                "open_gripper": "打开夹爪，为递交或重新抓取做准备",
                "close_gripper": "闭合夹爪，将目标夹紧",
                "handover_item": "面向用户递交物品，可配合open_gripper使用",
                "return_home": "导航回home区域或指定marker，常用于任务结束重置",
                # "recover": "执行回退/重置动作以从失败状态恢复",
            },
            "rules": [
                "仅输出合法JSON，禁止多余解释",
                "check节点的参数必须是cond字段",
                "action节点的name必须来自词表",
                "在固定场景内：先 observe_scene，如未看到目标则 rotate_scan 后再次 observe_scene",
                f"当 objects.{goal}.attrs.range_estimate 大于 2 时，必须重复调用 approach_far，并在每次移动前后 observe_scene，直到距离<=2米",
                "距离足够近(<=2米)后执行 finalize_target_pose，随后 predict_grasp_point → execute_grasp 完成抓取",
                "在 finalize_target_pose 前必须执行 observe_scene(force_vlm=true) 以刷新桌面与目标bbox",
                "如需在局部重复执行动作直到条件满足，可使用 repeat_until 节点，其children为需要循环的子树，cond为退出条件",
                "当需要切换到另一个区域/工作站时先使用 navigate_area，再继续观察与操作",
                "需要向用户递交物品时：close_gripper→(移动/导航)→handover_item→open_gripper",
            ],
            "example_plan": self._example_plan(),
            "notes": [
                "example_plan 仅作为示例，展示完整行为树结构和节点写法。",
                "输出时必须结合当前 goal/world，不得直接复用 example_plan。",
            ],
        }
        if plan_context:
            prompt["recent_history"] = plan_context[-5:]

        if not self.llm_api_key:
            raise RuntimeError("DeepSeek API key is missing")

        url = f"{self.llm_api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是服务机器人任务规划器，返回符合要求的JSON行为树。",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False),
                },
            ],
            "temperature": 0.1,
            "top_p": 0.85,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise PlanValidationError("LLM 没有返回任何结果")
        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            if content and isinstance(content[0], dict):
                payload_text = content[0].get("text", "")
            else:
                payload_text = "".join(str(item) for item in content)
        elif isinstance(content, dict):
            payload_text = content.get("text", "")
        else:
            payload_text = str(content)

        plan_dict = json.loads(payload_text)
        self._validate_plan(plan_dict)
        return plan_dict

    def _validate_plan(self, plan_dict: Dict[str, Any]) -> None:
        def _walk(node: Dict[str, Any]) -> None:
            node_type = node.get("type") or node.get("bt")
            if node_type not in {"sequence", "selector", "check", "action", "repeat_until"}:
                raise PlanValidationError(f"非法节点类型: {node_type}")
            if node_type == "action":
                name = node.get("name")
                if name not in {
                    "observe_scene",
                    "rotate_scan",
                    "approach_far",
                    "finalize_target_pose",
                    "predict_grasp_point",
                    "execute_grasp",
                    "recover",
                }:
                    raise PlanValidationError(f"非法动作: {name}")
            if node_type == "check":
                if "args" not in node or "cond" not in node["args"]:
                    raise PlanValidationError("check节点缺少cond")
            if node_type == "repeat_until":
                if "args" not in node or "cond" not in node["args"]:
                    raise PlanValidationError("repeat_until节点缺少cond参数")
            for child in node.get("children", []):
                _walk(child)

        _walk(plan_dict)

    # ------------------------------------------------------------------
    # Compilation helpers
    # ------------------------------------------------------------------
    def _compile_plan(self, root: PlanNode, world_model) -> List[PlanNode]:
        """Collect action nodes for UI display (execution使用完整行为树)."""
        steps: List[PlanNode] = []

        def traverse(node: PlanNode) -> None:
            if node.type == "action":
                steps.append(node)
            for child in node.children:
                traverse(child)

        traverse(root)
        return steps

    # ------------------------------------------------------------------
    @staticmethod
    def _example_plan() -> Dict[str, Any]:
        return BehaviorPlanner._fallback_plan("__example_target__")

    # ------------------------------------------------------------------
    # Fallback plan
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_plan(goal: str) -> Dict[str, Any]:
        """Deterministic sequence that approximates the previous behaviour."""
        return {
            "type": "sequence",
            "children": [
                {"type": "action", "name": "observe_scene", "args": {"target": goal, "force_vlm": True}},
                {
                    "type": "repeat_until",
                    "args": {"cond": f"objects.{goal}.visible==true"},
                    "children": [
                        {"type": "action", "name": "rotate_scan", "args": {}},
                        {"type": "action", "name": "observe_scene", "args": {"target": goal}},
                    ],
                },
                {
                    "type": "repeat_until",
                    "args": {"cond": f"objects.{goal}.attrs.range_estimate<=2"},
                    "children": [
                        {"type": "action", "name": "approach_far", "args": {"target": goal}},
                        {"type": "action", "name": "observe_scene", "args": {"target": goal}},
                    ],
                },
                {"type": "action", "name": "observe_scene", "args": {"target": goal, "force_vlm": True}},
                {"type": "action", "name": "finalize_target_pose", "args": {"target": goal}},
                {"type": "action", "name": "observe_scene", "args": {"target": goal, "force_vlm": True}},
                {"type": "action", "name": "predict_grasp_point", "args": {"target": goal}},
                {"type": "action", "name": "execute_grasp", "args": {"target": goal}},
            ],
        }


class ReflectionAdvisor:
    """Helper that generates diagnosis/hint pairs after a failed attempt."""

    def __init__(self, llm_api_key: Optional[str] = None, llm_model: str = "deepseek-chat") -> None:
        self.llm_api_key = llm_api_key or os.getenv("DEEPSEEK_REFLECT_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.llm_model = llm_model
        self.llm_api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    def reflect(
        self, goal: str, plan_entry: PlanContextEntry, execution_turns: List[ExecutionTurn]
    ) -> Optional[Dict[str, Any]]:
        history = [turn.to_prompt_dict() for turn in execution_turns[-5:]] if execution_turns else []
        prompt = {
            "goal": goal,
            "plan": plan_entry.to_prompt_dict(),
            "recent_execution": history,
            "instruction": (
                "根据计划与执行日志，分析失败原因并给出下一次规划应注意的事项。"
                "输出JSON，字段包括 diagnosis(中文)、adjustment_hint(中文)和confidence(0-1)。"
            ),
        }
        if not self.llm_api_key:
            return self._fallback_reflection(plan_entry, execution_turns)
        try:
            url = f"{self.llm_api_base.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.llm_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是机器人任务专家，负责分析失败原因并提出改进建议，务必输出JSON。",
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                "temperature": 0.2,
                "top_p": 0.8,
                "response_format": {"type": "json_object"},
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("no_reflection_choices")
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, list):
                if content and isinstance(content[0], dict):
                    payload_text = content[0].get("text", "")
                else:
                    payload_text = "".join(str(item) for item in content)
            elif isinstance(content, dict):
                payload_text = content.get("text", "")
            else:
                payload_text = str(content)
            return json.loads(payload_text)
        except Exception as exc:
            log_warning(f"⚠️ 反思阶段调用LLM失败: {exc}")
            return self._fallback_reflection(plan_entry, execution_turns)

    @staticmethod
    def _fallback_reflection(
        plan_entry: PlanContextEntry, execution_turns: List[ExecutionTurn]
    ) -> Dict[str, Any]:
        diagnosis = plan_entry.failure_reason or "执行失败，具体原因未知"
        if execution_turns:
            tail = execution_turns[-1]
            diagnosis = tail.detail or tail.status or diagnosis
        if plan_entry.planned_steps:
            hint = f"重新审视步骤「{plan_entry.planned_steps[-1]}」，确保感知和位姿准确。"
        else:
            hint = "重新进行完整观测，确认目标位置后再执行动作。"
        return {
            "diagnosis": diagnosis,
            "adjustment_hint": hint,
            "confidence": 0.2,
        }
