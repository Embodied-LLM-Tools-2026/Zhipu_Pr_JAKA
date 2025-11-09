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

from .task_structures import CompiledPlan, PlanNode


class PlanValidationError(Exception):
    """Raised when a plan does not conform to the required schema."""


class ConditionFailed(Exception):
    """Raised when a check node fails while compiling the plan."""


class BehaviorPlanner:
    """Planner that queries an LLM for a structured behaviour tree plan."""

    def __init__(self, llm_api_key: Optional[str] = None, llm_model: str = "GLM-4.5-Flash") -> None:
        self.llm_api_key = llm_api_key or os.getenv("ZHIPUAI_API_KEY")
        self.llm_model = llm_model
        self._client = None
        if self.llm_api_key:
            try:
                from zhipuai import ZhipuAI  # type: ignore

                self._client = ZhipuAI(api_key=self.llm_api_key)
            except Exception:
                self._client = None

    # ------------------------------------------------------------------
    def make_plan(self, goal: str, world_model) -> CompiledPlan:
        """Return a compiled behaviour tree for the current goal."""
        plan_dict: Optional[Dict[str, Any]] = None
        metadata: Dict[str, Any] = {}
        if self._client:
            try:
                plan_dict = self._request_plan(goal, world_model.snapshot())
                metadata["source"] = "llm"
            except Exception as exc:
                metadata["source"] = "llm_error"
                metadata["error"] = str(exc)

        if not plan_dict:
            plan_dict = self._fallback_plan(goal)
            metadata["source"] = metadata.get("source") or "fallback"

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
    def _request_plan(self, goal: str, world_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        prompt = {
            "schema": "BT/1.0",
            "goal": goal,
            "world": world_snapshot,
            "allowed_nodes": ["sequence", "selector", "check", "action"],
            "allowed_actions": [
                "observe_scene",
                "rotate_scan",
                "approach_far",
                "finalize_target_pose",
                "predict_grasp_point",
                "execute_grasp",
                "recover",
            ],
            "action_docs": {
                "observe_scene": "触发 RGBD/VLM 观测，刷新对场景的认知",
                "rotate_scan": "原地旋转或摆头扫描寻找目标",
                "approach_far": "当距离目标物体大于5米时沿机器人与目标连线迈大步靠近",
                "finalize_target_pose": "进行精确定位调整底盘姿态至抓取位",
                "predict_grasp_point": "在精确定位后，调用ZeroGrasp等接口，预测抓取点",
                "execute_grasp": "根据预测结果执行抓取策略",
                # "recover": "执行回退/重置动作以从失败状态恢复",
            },
            "rules": [
                "仅输出合法JSON，禁止多余解释",
                "check节点的参数必须是cond字段",
                "action节点的name必须来自词表",
                "在固定场景内：先 observe_scene，如未看到目标则 rotate_scan 后再次 observe_scene",
                f"当 objects.{goal}.attrs.range_estimate 大于 5 时，必须重复调用 approach_far，并在每次移动前后 observe_scene，直到距离<=5米",
                "距离足够近后执行 finalize_target_pose，随后 predict_grasp_point → execute_grasp 完成抓取",
            ],
        }

        response = self._client.chat.completions.create(  # type: ignore[operator]
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是服务机器人任务规划器，返回符合要求的JSON行为树。",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False),
                },
            ],
            temperature=0.1,
            top_p=0.85,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content  # type: ignore[index]
        if isinstance(content, list):
            payload = content[0].get("text") if content else ""
        elif isinstance(content, dict):
            payload = content.get("text", "")
        else:
            payload = str(content)
        plan_dict = json.loads(payload)
        self._validate_plan(plan_dict)
        return plan_dict

    def _validate_plan(self, plan_dict: Dict[str, Any]) -> None:
        def _walk(node: Dict[str, Any]) -> None:
            node_type = node.get("type") or node.get("bt")
            if node_type not in {"sequence", "selector", "check", "action"}:
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
            for child in node.get("children", []):
                _walk(child)

        _walk(plan_dict)

    # ------------------------------------------------------------------
    # Compilation helpers
    # ------------------------------------------------------------------
    def _compile_plan(self, root: PlanNode, world_model) -> List[PlanNode]:
        """Flatten the behaviour tree into a sequence of actions."""
        steps: List[PlanNode] = []

        def traverse(node: PlanNode) -> None:
            if node.type == "sequence":
                for child in node.children:
                    traverse(child)
            elif node.type == "selector":
                for child in node.children:
                    if child.type == "check":
                        cond = child.args.get("cond", "")
                        if world_model.evaluate_condition(cond):
                            return
                        continue
                    traverse(child)
                    return
            elif node.type == "check":
                cond = node.args.get("cond", "")
                if not world_model.evaluate_condition(cond):
                    raise ConditionFailed(cond)
            elif node.type == "action":
                steps.append(node)

        traverse(root)
        return steps

    # ------------------------------------------------------------------
    # Fallback plan
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_plan(goal: str) -> Dict[str, Any]:
        """Deterministic sequence that approximates the previous behaviour."""
        return {
            "type": "sequence",
            "children": [
                {"type": "action", "name": "observe_scene", "args": {"target": goal}},
                {
                    "type": "selector",
                    "children": [
                        {
                            "type": "check",
                            "args": {"cond": f"objects.{goal}.visible==true"},
                        },
                        {
                            "type": "sequence",
                            "children": [
                                {"type": "action", "name": "rotate_scan", "args": {}},
                                {"type": "action", "name": "observe_scene", "args": {"target": goal}},
                            ],
                        },
                    ],
                },
                {
                    "type": "selector",
                    "children": [
                        {
                            "type": "check",
                            "args": {"cond": f"objects.{goal}.attrs.range_estimate<=5"},
                        },
                        {
                            "type": "sequence",
                            "children": [
                                {"type": "action", "name": "approach_far", "args": {"target": goal}},
                                {"type": "action", "name": "observe_scene", "args": {"target": goal}},
                                {"type": "action", "name": "approach_far", "args": {"target": goal}},
                                {"type": "action", "name": "observe_scene", "args": {"target": goal}},
                            ],
                        },
                    ],
                },
                {"type": "action", "name": "observe_scene", "args": {"target": goal}},
                {"type": "action", "name": "finalize_target_pose", "args": {"target": goal}},
                {"type": "action", "name": "observe_scene", "args": {"target": goal}},
                {"type": "action", "name": "predict_grasp_point", "args": {"target": goal}},
                {"type": "action", "name": "execute_grasp", "args": {"target": goal}},
            ],
        }
