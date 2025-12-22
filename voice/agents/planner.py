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
from typing import Any, Dict, List, Optional
import time

import requests

from tools.logging.task_logger import log_error, log_info, log_success, log_warning  # type: ignore

from ..control.task_structures import CompiledPlan, PlanNode, PlanContextEntry, ExecutionTurn
from .task_plan.schema import Plan, validate_plan, PlanValidationError as SchemaValidationError


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
        if not self.llm_api_key:
            raise RuntimeError("LLM API key is required for planning; fallback has been disabled")

        plan_dict: Optional[Dict[str, Any]] = None
        metadata: Dict[str, Any] = {}
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

        root = PlanNode.from_dict(plan_dict)
        steps = self._compile_plan(root, world_model)
        return CompiledPlan(root=root, steps=steps, metadata=metadata)

    # ------------------------------------------------------------------
    # Long-Horizon Planning (TaskExecutive Mode)
    # ------------------------------------------------------------------
    def make_long_horizon_plan(self, goal: str, world_model, plan_context: Optional[List[Dict[str, Any]]] = None) -> Plan:
        """
        Generates a structured long-horizon Plan (list of subtasks) for TaskExecutive.
        """
        if not self.llm_api_key:
            raise RuntimeError("LLM API key is required for planning")

        last_error: Optional[Exception] = None
        validation_feedback: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            start_time = time.time()
            log_info(f"🧠 [Planner] Generating Long-Horizon Plan (Attempt {attempt}/{self.max_retries}): {goal}")
            
            try:
                plan_dict = self._request_long_horizon_plan(goal, world_model.snapshot(), plan_context, validation_feedback)
                
                # Validate using strict schema
                plan_obj = validate_plan(plan_dict)
                
                duration = time.time() - start_time
                log_success(f"✅ [Planner] Plan generated & validated in {duration:.2f}s")
                return plan_obj

            except SchemaValidationError as exc:
                duration = time.time() - start_time
                log_warning(f"⚠️ [Planner] Validation Failed (Attempt {attempt}): {exc}")
                last_error = exc
                validation_feedback = f"Previous plan failed validation: {str(exc)}. Please fix and retry."
            except Exception as exc:
                duration = time.time() - start_time
                log_error(f"❌ [Planner] LLM Error (Attempt {attempt}): {exc}")
                last_error = exc
                # For generic errors, maybe just retry without specific feedback or simple feedback
                validation_feedback = f"Previous attempt failed with error: {str(exc)}"

        raise RuntimeError(f"Failed to generate valid plan after {self.max_retries} attempts. Last error: {last_error}")

    def _request_long_horizon_plan(
        self, 
        goal: str, 
        world_snapshot: Dict[str, Any], 
        plan_context: Optional[List[Dict[str, Any]]] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        
        prompt = {
            "role": "You are a high-level robot task planner. Output a JSON plan.",
            "goal": goal,
            "world_context": world_snapshot,
            "schema_requirements": {
                "root": "Must be a JSON object with 'goal', 'plan_id', 'subtasks' (list).",
                "subtask_fields": ["id (unique string)", "type (enum)", "params (dict)", "depends_on (list of ids)", "done_if (predicate string)"],
                "allowed_types": [
                    "fetch_place (params: object, from, to)",
                    "fetch_only (params: object, from)",
                    "place_only (params: object, to)",
                    "navigate (params: target)",
                    "observe (params: target)",
                    "wait (params: duration_s)",
                    "operate (params: action)"
                ],
                "predicates": ["at(location)", "holding(object)", "on_table(object)", "area_clear(table)"]
            },
            "examples": [
                {
                    "goal": "Move apple from table to desk",
                    "plan": {
                        "goal": "Move apple from table to desk",
                        "plan_id": "p_001",
                        "subtasks": [
                            {"id": "1", "type": "fetch_place", "params": {"object": "apple", "from": "table", "to": "desk"}, "done_if": "on_table(apple)"}
                        ]
                    }
                },
                {
                    "goal": "Go to kitchen and wait",
                    "plan": {
                        "goal": "Go to kitchen and wait",
                        "plan_id": "p_002",
                        "subtasks": [
                            {"id": "t1", "type": "navigate", "params": {"target": "kitchen"}, "done_if": "at(kitchen)"},
                            {"id": "t2", "type": "wait", "params": {"duration_s": 5}, "depends_on": ["t1"]}
                        ]
                    }
                }
            ]
        }
        
        if feedback:
            prompt["feedback_from_previous_attempt"] = feedback
            
        messages = [
            {"role": "system", "content": "You are a robot task planner. Output ONLY valid JSON conforming to the schema."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
        ]

        url = f"{self.llm_api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

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
                # "recover",
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
                "在固定场景内：通常先 observe_scene 如未看到目标则结合 rotate_scan/search_area 重试",
                f"建议通过 approach_far 等动作逐步拉近 objects.{goal} 的 range_estimate，必要场景下自行安排 observe_scene 的时机以确保定位精准。",
                "当距离足够近时再执行 finalize_target_pose → predict_grasp_point → execute_grasp。",
                "在 finalize_target_pose 以及 predict_grasp_point 之前，应确保最近一次 observe_scene 使用参数 force_vlm=true 以刷新桌面与目标信息。",
                "如需在局部重复执行动作直到条件满足，可使用 repeat_until 节点，其children为需要循环的子树，cond为退出条件",
                "当需要切换到另一个区域/工作站时先使用 navigate_area，再继续观察与操作",
                "需要向用户递交物品时：close_gripper→(移动/导航)→handover_item→open_gripper",
                "在填写 action 的 args['target'] 时，如果已知目标物体的材质或物理属性（如 soft, sponge, plush, rigid 等），请务必包含在 target 描述中（例如 'soft toy' 而非仅 'toy'），以便底层控制器选择正确的抓取策略。",
            ],
            "notes": [
                "行为树只接受 type ∈ {sequence, selector, check, action, repeat_until}，且每个节点必须显式写出 type。",
                "action 节点的 name 必须来自 allowed_actions；不要创造 observe_force_vlm 之类的新动作。若要强制 VLM，直接在 observe_scene 的 args 中写 force_vlm:true。",
                "check/repeat_until 节点的 args 必须包含 cond，例如 {\"type\":\"repeat_until\",\"args\":{\"cond\":\"objects.饮料.visible==true\"},\"children\":[...]}。",
                "Plan JSON 示例：{\"type\":\"sequence\",\"children\":[{\"type\":\"action\",\"name\":\"observe_scene\",\"args\":{\"target\":\"饮料\",\"force_vlm\":true}},{\"type\":\"repeat_until\",\"args\":{\"cond\":\"objects.饮料.visible==true\"},\"children\":[{\"type\":\"action\",\"name\":\"rotate_scan\",\"args\":{}},{\"type\":\"action\",\"name\":\"observe_scene\",\"args\":{\"target\":\"饮料\"}}]},{\"type\":\"action\",\"name\":\"finalize_target_pose\",\"args\":{\"target\":\"饮料\"}}]}。此示例仅供格式参考，可根据当前场景调整节点与动作。",
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
                    "open_gripper",
                    "close_gripper",
                    "handover_item",
                    "return_home",
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
