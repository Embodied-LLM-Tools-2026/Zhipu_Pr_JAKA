"""
Function-call driven TaskProcessor.

This module introduces an alternative orchestration loop where an LLM
controls the robot purely via structured function calls. It runs
separately from the legacy behaviour tree pipeline so callers can select
either mode via configuration.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import base64
from zhipuai import ZhipuAI

try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    dashscope = None
    MultiModalConversation = None

from tools.vision.upload_image import upload_file_and_get_url

from tools.logging.task_logger import log_error, log_info, log_success, log_warning  # type: ignore
from tools.ui.ui_state_bridge import UIStateBridge  # type: ignore

from ...control.apis import RobotAPI
from ...control.world_model import WorldModel
from ...control.executor import SkillExecutor, SkillRuntime
from ...control.task_structures import ExecutionResult, ObservationPhase, PlanNode, ExecutionTurn, PlanContextEntry
from ...perception.observer import VLMObserver
from ...agents.planner import BehaviorPlanner, ReflectionAdvisor


class FunctionCallToolset:
    """Wraps RobotAPI primitives as callable functions for the LLM."""

    def __init__(self, api: RobotAPI, processor: "FunctionCallTaskProcessor") -> None:
        self.api = api
        self.processor = processor

    # ------------------------------------------------------------------
    def specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "observe_scene",
                "description": "Capture a new observation of the target object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Object identifier or description."},
                        "phase": {
                            "type": "string",
                            "enum": ["search", "approach"],
                            "description": "Observation phase hint.",
                        },
                        "force_vlm": {"type": "boolean", "description": "Force a full VLM refresh."},
                        "query": {
                            "type": "string",
                            "description": "Optional natural-language request (e.g. describe scene relations).",
                        },
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "rotate_scan",
                "description": "Rotate the base in place to search the environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "angle_deg": {"type": "number", "description": "Rotation angle in degrees (default 30)."}
                    },
                },
            },
            {
                "name": "search_area",
                "description": "Perform repeated rotations/scans at the current spot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "turns": {"type": "integer", "description": "Number of rotations."},
                        "angle_deg": {"type": "number", "description": "Per-rotation angle in degrees."},
                    },
                },
            },
            {
                "name": "navigate_area",
                "description": "Navigate the base to a named area, marker, or explicit pose.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "string"},
                        "marker": {"type": "string"},
                        "pose": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "theta": {"type": "number"},
                            },
                        },
                    },
                },
            },
            {
                "name": "get_object_state",
                "description": "Return the latest known state of an object from the world model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string", "description": "Object identifier (target name)."},
                    },
                    "required": ["object_id"],
                },
            },
            {
                "name": "navigate_to_pose",
                "description": "Command the base to an explicit pose (meters + radians).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "theta": {"type": "number", "description": "Rotation in radians."},
                    },
                    "required": ["x", "y", "theta"],
                },
            },
            {
                "name": "move_tcp_linear",
                "description": "Perform a linear TCP motion using millimetre cartesian pose + axis-angle rotation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pose": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "[x_mm, y_mm, z_mm, rx, ry, rz]",
                        },
                        "speed": {"type": "number", "description": "Optional joint speed override."},
                        "acc": {"type": "number", "description": "Optional joint acceleration override."},
                    },
                    "required": ["pose"],
                },
            },
            {
                "name": "set_gripper",
                "description": "Update the gripper position/force.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "number", "description": "1-100 opening percentage."},
                        "force": {"type": "number", "description": "20-320 force value."},
                    },
                    "required": [],
                },
            },
            {
                "name": "execute_skill",
                "description": "Invoke a legacy skill from the SkillExecutor (e.g. open_gripper, finalize_target_pose).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name."},
                        "args": {"type": "object", "description": "Arguments dict for the skill."},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "approach_far",
                "description": "Approach the target object when it is further than two metres away.",
                "parameters": {
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                },
            },
            {
                "name": "finalize_target_pose",
                "description": "Use depth localization + base alignment before grasping.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "predict_grasp_point",
                "description": "Run ZeroGrasp inference to obtain a grasp pose.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "execute_grasp",
                "description": "Execute the previously predicted grasp TCP pose.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "open_gripper",
                "description": "Open the gripper via the legacy skill.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "close_gripper",
                "description": "Close the gripper via the legacy skill.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "handover_item",
                "description": "Perform a handover to a human once an item is grasped.",
                "parameters": {
                    "type": "object",
                    "properties": {"item": {"type": "string", "description": "Optional item name."}},
                },
            },
            # {
            #     "name": "recover",
            #     "description": "Trigger the recovery skill (typically back off).",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "mode": {"type": "string"},
            #             "distance": {"type": "number"},
            #         },
            #     },
            # },
        ]

    # ------------------------------------------------------------------
    def dispatch(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, f"_fn_{name}", None)
        if not handler:
            return {"ok": False, "error": f"unknown_function:{name}"}
        try:
            return handler(**params)
        except Exception as exc:  # pragma: no cover - hardware path
            log_error(f"❌ Function {name} 调用失败: {exc}")
            return {"ok": False, "error": str(exc)}

    # -- individual function handlers ---------------------------------
    def _fn_observe_scene(
        self,
        target: str,
        phase: str = "search",
        force_vlm: bool = False,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        observation, payload = self.api.perception.observe(
            target,
            phase=ObservationPhase.SEARCH if phase != "approach" else ObservationPhase.APPROACH,
            force_vlm=force_vlm,
            analysis_request=query,
        )
        self.processor.update_observation(observation)
        return {
            "ok": True,
            "observation": self._serialize_observation(observation),
            "vlm_payload": payload,
        }

    def _fn_rotate_scan(self, angle_deg: Optional[float] = None) -> Dict[str, Any]:
        args = {"angle_deg": float(angle_deg)} if angle_deg is not None else {}
        return self._call_skill("rotate_scan", args)

    def _fn_search_area(self, turns: Optional[int] = None, angle_deg: Optional[float] = None) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        if turns is not None:
            args["turns"] = int(turns)
        if angle_deg is not None:
            args["angle_deg"] = float(angle_deg)
        return self._call_skill("search_area", args)

    def _fn_navigate_area(
        self,
        area: Optional[str] = None,
        marker: Optional[str] = None,
        pose: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        if area:
            args["area"] = area
        if marker:
            args["marker"] = marker
        if pose:
            args["pose"] = pose
        return self._call_skill("navigate_area", args)

    def _fn_get_object_state(self, object_id: str) -> Dict[str, Any]:
        state = self.api.perception.get_object_state(object_id)
        return {"ok": bool(state), "state": state}

    def _fn_navigate_to_pose(self, x: float, y: float, theta: float) -> Dict[str, Any]:
        obs_result = self._ensure_target_observation()
        success = self.api.navigation.goto_pose(x, y, theta)
        response = {
            "ok": success,
            "pose": {"x": x, "y": y, "theta": theta},
        }
        if obs_result and "observation" in obs_result:
            response["observation"] = obs_result["observation"]
        return response

    def _fn_move_tcp_linear(
        self,
        pose: List[float],
        speed: Optional[float] = None,
        acc: Optional[float] = None,
    ) -> Dict[str, Any]:
        obs_result = self._ensure_target_observation(phase=ObservationPhase.APPROACH)
        result = self.api.manipulation.move_tcp_linear(pose, speed=speed, acc=acc)
        if obs_result and "observation" in obs_result and isinstance(result, dict):
            result = dict(result)
            result["observation"] = obs_result["observation"]
        return result

    def _fn_set_gripper(
        self,
        position: Optional[float] = None,
        force: Optional[float] = None,
    ) -> Dict[str, Any]:
        obs_result = self._ensure_target_observation()
        result = self.api.manipulation.set_gripper(position=position, force=force)
        if obs_result and "observation" in obs_result and isinstance(result, dict):
            result = dict(result)
            result["observation"] = obs_result["observation"]
        return result

    def _fn_execute_skill(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._call_skill(name, args or {})

    def _fn_approach_far(self, target: Optional[str] = None) -> Dict[str, Any]:
        args = {"target": target} if target else {}
        return self._call_skill("approach_far", args)

    def _fn_finalize_target_pose(self) -> Dict[str, Any]:
        return self._call_skill("finalize_target_pose", {})

    def _fn_predict_grasp_point(self) -> Dict[str, Any]:
        return self._call_skill("predict_grasp_point", {})

    def _fn_execute_grasp(self) -> Dict[str, Any]:
        return self._call_skill("execute_grasp", {})

    def _fn_open_gripper(self) -> Dict[str, Any]:
        return self._call_skill("open_gripper", {})

    def _fn_close_gripper(self) -> Dict[str, Any]:
        return self._call_skill("close_gripper", {})

    def _fn_handover_item(self, item: Optional[str] = None) -> Dict[str, Any]:
        args = {"item": item} if item else {}
        return self._call_skill("handover_item", args)

    def _fn_recover(self, mode: Optional[str] = None, distance: Optional[float] = None) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        if mode:
            args["mode"] = mode
        if distance is not None:
            args["distance"] = float(distance)
        return self._call_skill("recover", args)

    @staticmethod
    def _serialize_observation(observation: Any) -> Dict[str, Any]:
        if observation is None:
            return {}
        return {
            "found": bool(getattr(observation, "found", False)),
            "confidence": getattr(observation, "confidence", None),
            "bbox": getattr(observation, "bbox", None),
            "range_estimate": getattr(observation, "range_estimate", None),
            "camera_center": getattr(observation, "camera_center", None),
            "robot_center": getattr(observation, "robot_center", None),
            "world_center": getattr(observation, "world_center", None),
            "source": getattr(observation, "source", None),
            "analysis": getattr(observation, "analysis", None),
        }

    def _call_skill(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Always refresh observation before executing a skill to keep state current.
        obs_result = self._ensure_target_observation()
        result = self.processor.invoke_skill(name, args)
        response = {
            "ok": result.status == "success",
            "status": result.status,
            "reason": result.reason,
            "evidence": result.evidence,
        }
        if obs_result and "observation" in obs_result:
            response["observation"] = obs_result["observation"]
        return response

    def _ensure_target_observation(
        self, phase: ObservationPhase = ObservationPhase.SEARCH
    ) -> Optional[Dict[str, Any]]:
        """Refresh VLM observation for the current goal, if available."""
        target = self.processor.world.goal
        if not target:
            return None
        try:
            observation, payload = self.api.perception.observe(
                target, phase=phase, force_vlm=False, max_steps=1
            )
            self.processor.update_observation(observation, reset_extra=False)
            return {
                "observation": self._serialize_observation(observation),
                "vlm_payload": payload,
            }
        except Exception as exc:
            log_warning(f"[FunctionCall] 自动观测目标 {target} 失败: {exc}")
            return None


class FunctionCallTaskProcessor:
    """
    Alternative orchestration pipeline driven by LLM function calls.

    The processor exposes RobotAPI primitives to the LLM. The agent must
    call these functions (observe/navigate/manipulate) to achieve the
    requested goal. Final responses are delivered once the LLM returns a
    normal assistant message.
    """

    def __init__(self, navigator=None) -> None:
        self.navigator = navigator
        self.world = WorldModel()
        self.observer = VLMObserver()
        self.executor = SkillExecutor(navigator=navigator)
        # Planner/Reflection are still constructed to keep RobotAPI happy,
        # but are unused in this chain.
        self.planner = BehaviorPlanner()
        self.reflection = ReflectionAdvisor(
            llm_api_key=getattr(self.planner, "llm_api_key", None),
            llm_model=getattr(self.planner, "llm_model", "deepseek-chat"),
        )
        self.api = RobotAPI.build(
            navigator=self.navigator,
            observer=self.observer,
            executor=self.executor,
            planner=self.planner,
            world=self.world,
            reflection=self.reflection,
        )
        self.tools = FunctionCallToolset(self.api, self)
        self.max_rounds = int(os.getenv("FUNCTION_CALL_MAX_ROUNDS", "16"))
        self.temperature = float(os.getenv("FUNCTION_CALL_TEMPERATURE", "0.1"))
        self._finalize_calls = 0
        
        # 模型提供商选择: "zhipu" (GLM-4.5v) 或 "dashscope" (Qwen3-VL)
        self.provider = os.getenv("FUNCTION_CALL_PROVIDER", "zhipu").lower()
        
        if self.provider == "dashscope":
            # Qwen3-VL via DashScope
            self.model = os.getenv("FUNCTION_CALL_MODEL", "qwen-vl-max")
            self.api_key = (
                os.getenv("FUNCTION_CALL_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
                or os.getenv("Zhipu_real_demo_API_KEY")  # 复用 VLM 的 key
            )
            if not self.api_key:
                raise RuntimeError("FunctionCallTaskProcessor (dashscope) 需要配置 DASHSCOPE_API_KEY")
            if not DASHSCOPE_AVAILABLE:
                raise RuntimeError("dashscope 库未安装，请运行: pip install dashscope")
            dashscope.api_key = self.api_key
            self.client = None  # dashscope 不需要 client 对象
            log_info(f"[FunctionCall] 使用 DashScope 模型: {self.model}")
        else:
            # GLM-4.5v via ZhipuAI (default)
            self.provider = "zhipu"
            self.model = os.getenv("FUNCTION_CALL_MODEL", "glm-4.5v")
            self.api_key = (
                os.getenv("FUNCTION_CALL_API_KEY")
                or os.getenv("ZHIPUAI_API_KEY")
                or getattr(self.planner, "llm_api_key", None)
            )
            if not self.api_key:
                raise RuntimeError("FunctionCallTaskProcessor (zhipu) 需要配置 ZHIPUAI_API_KEY")
            self.client = ZhipuAI(api_key=self.api_key)
            log_info(f"[FunctionCall] 使用 ZhipuAI 模型: {self.model}")
        
        self.system_prompt = os.getenv(
            "FUNCTION_CALL_SYSTEM_PROMPT",
            (
                "You are a robotics planner controlling a real robot. "
                "CRITICAL RULES:\n"
                "1. You MUST use tool_calls (function calls) to execute any action. NEVER describe actions in text.\n"
                "2. Do NOT say 'I will call predict_grasp_point' - instead, actually call the function.\n"
                "3. Do NOT include function names in your text response. Use the tools directly.\n"
                "4. Only respond with a final JSON summary AFTER the grasp is complete or truly impossible.\n"
                "5. Never fabricate execution results - always call the actual function and wait for results.\n"
                "If you need to predict a grasp point, call predict_grasp_point(). If you need to execute grasp, call execute_grasp()."
            ),
        )
        self._latest_observation: Optional[Any] = None
        self._shared_runtime_extra: Dict[str, Any] = {}
        disable_bridge = os.getenv("DISABLE_ROBOT_UI_BRIDGE", "").lower() in {"1", "true", "yes"}
        self.ui_bridge = None if disable_bridge else UIStateBridge(os.getenv("ROBOT_UI_URL"))
        self._timeline: List[Dict[str, Any]] = []
        self._timeline_limit = int(os.getenv("EXEC_TIMELINE_LIMIT", "30"))
        self.execution_history: List[ExecutionTurn] = []
        self._execution_history_limit = int(os.getenv("EXEC_HISTORY_LIMIT", "60"))
        self._current_plan_entry: Optional[PlanContextEntry] = None
        self._current_plan_id: Optional[str] = None
        self._last_status_message: Optional[str] = None
        self._mission_text = self._build_mission_instruction()
        self._tool_instruction = self._build_tool_instruction()
        self._failure_threshold = int(os.getenv("FUNCTION_CALL_FAILURE_THRESHOLD", "3"))
        self._failure_count = 0

    def _get_current_image_base64(self, cam_name: str = "front") -> Optional[str]:
        """获取当前相机图像，根据 provider 返回不同格式"""
        try:
            # Capture image from local API
            resp = requests.get(
                f"http://127.0.0.1:8000/api/capture?cam={cam_name}",
                timeout=3,
            )
            if resp.status_code != 200:
                log_warning(f"Failed to capture image: {resp.status_code}")
                return None
            
            data = resp.json()
            image_path = data.get("url")
            if not image_path or not os.path.exists(image_path):
                log_warning(f"Image path invalid: {image_path}")
                return None
            
            # 根据 provider 返回不同格式
            if self.provider == "dashscope":
                # DashScope 需要上传图像到 OSS 获取 oss:// URL
                try:
                    oss_url = upload_file_and_get_url(
                        api_key=self.api_key,
                        model_name=self.model,
                        file_path=image_path,
                    )
                    log_info(f"[FunctionCall] 图像已上传到 OSS: {oss_url[:50]}...")
                    return oss_url
                except Exception as e:
                    log_error(f"上传图像到 OSS 失败: {e}")
                    return None
            else:
                # ZhipuAI 使用 base64 格式
                with open(image_path, "rb") as img_file:
                    img_base = base64.b64encode(img_file.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_base}"
        except Exception as e:
            log_error(f"Error getting camera image: {e}")
            return None

    # ------------------------------------------------------------------
    def set_navigator(self, navigator) -> None:
        self.navigator = navigator
        self.executor.set_navigator(navigator)
        self.api.update_navigator(navigator)

    def update_observation(self, observation: Any, reset_extra: bool = True) -> None:
        self._latest_observation = observation
        if reset_extra:
            self._shared_runtime_extra = {}

    def invoke_skill(self, name: str, args: Dict[str, Any]) -> ExecutionResult:
        runtime = SkillRuntime(
            navigator=self.navigator,
            world_model=self.world,
            observation=self._latest_observation,
            extra=self._shared_runtime_extra,
        )
        node = PlanNode(type="action", name=name, args=args or {})
        result = self.executor.execute(node, runtime)
        # Runtime.extra may be mutated by skills (e.g. predict_grasp_point), reuse it for future calls.
        self._shared_runtime_extra = runtime.extra or self._shared_runtime_extra
        return result

    def _publish_world_snapshot(self) -> None:
        if not self.ui_bridge:
            return
        try:
            snapshot = self.world.snapshot()
        except Exception:
            return
        self.ui_bridge.post_world_model(snapshot)

    def _emit_ui_log(self, message: str, level: str = "info") -> None:
        if not self.ui_bridge:
            return
        try:
            self.ui_bridge.post_task_log(message, level=level)
        except Exception:
            pass

    def _record_timeline_event(
        self,
        *,
        stage: str,
        node: str,
        status: str,
        detail: Optional[str] = None,
        elapsed: Optional[float] = None,
    ) -> None:
        entry = {
            "ts": int(time.time() * 1000),
            "time": time.strftime("%H:%M:%S"),
            "stage": stage,
            "node": node,
            "status": status,
            "detail": detail or "",
        }
        if elapsed is not None:
            entry["elapsed"] = float(elapsed)
        self._timeline.append(entry)
        if len(self._timeline) > self._timeline_limit:
            self._timeline = self._timeline[-self._timeline_limit :]

    def _timeline_payload(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._timeline]

    def _append_execution_turn(
        self,
        *,
        stage: str,
        node: str,
        status: str,
        detail: Optional[str] = None,
    ) -> None:
        if not self._current_plan_id:
            return
        turn = ExecutionTurn(
            plan_id=self._current_plan_id,
            stage=stage,
            node=node,
            status=status,
            detail=detail,
        )
        self.execution_history.append(turn)
        if len(self.execution_history) > self._execution_history_limit:
            self.execution_history = self.execution_history[-self._execution_history_limit :]

    # ------------------------------------------------------------------
    def _call_llm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.provider == "dashscope":
            return self._call_llm_dashscope(messages)
        else:
            return self._call_llm_zhipu(messages)
    
    def _call_llm_zhipu(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用 ZhipuAI GLM-4.5v"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": spec} for spec in self.tools.specs()],
                tool_choice="auto",
                temperature=self.temperature,
            )
            return response.choices[0].message.model_dump()
        except Exception as e:
            log_error(f"ZhipuAI LLM call failed: {e}")
            raise RuntimeError(f"function_call_llm_failed: {e}")
    
    def _call_llm_dashscope(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用 DashScope Qwen3-VL (支持 function call)"""
        try:
            if not MultiModalConversation:
                raise RuntimeError("dashscope.MultiModalConversation 未安装")
            # 转换消息格式为 DashScope 格式
            ds_messages = self._convert_messages_for_dashscope(messages)
            
            # 构建 tools 参数
            tools = [{"type": "function", "function": spec} for spec in self.tools.specs()]
            
            response = MultiModalConversation.call(
                model=self.model,
                messages=ds_messages,
                tools=tools,
                result_format="message",
                temperature=self.temperature,
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"DashScope API error: {response.code} - {response.message}")
            
            # 解析响应
            output = response.output
            choice = output.choices[0] if output.choices else None
            if not choice:
                raise RuntimeError("DashScope 返回空响应")
            
            message = choice.message
            result = {
                "role": message.role,
                "content": message.content or "",
            }
            
            # 处理 tool_calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                result["tool_calls"] = []
                for tc in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.get("id") or tc.get("function", {}).get("name", "call_0"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        }
                    })
            
            return result
        except Exception as e:
            log_error(f"DashScope LLM call failed: {e}")
            raise RuntimeError(f"function_call_llm_failed: {e}")
    
    def _convert_messages_for_dashscope(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将消息格式转换为 DashScope 兼容格式"""
        ds_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            # 处理 tool 消息
            if role == "tool":
                ds_messages.append({
                    "role": "tool",
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                    "tool_call_id": msg.get("tool_call_id", ""),
                })
                continue
            
            # 处理 assistant 消息 (可能包含 tool_calls)
            if role == "assistant":
                ds_msg = {"role": "assistant", "content": msg.get("content") or ""}
                if "tool_calls" in msg and msg["tool_calls"]:
                    ds_msg["tool_calls"] = msg["tool_calls"]
                ds_messages.append(ds_msg)
                continue
            
            # 处理 user/system 消息 (可能包含图像)
            if isinstance(content, list):
                # 多模态内容: DashScope 需要 {"image": "..."} / {"text": "..."} 形式
                ds_content = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type == "text" or ("text" in part and part_type is None):
                        ds_content.append({"text": part.get("text", "")})
                        continue
                    # 兼容 OpenAI 风格的 image_url 结构
                    if part_type == "image_url" or "image_url" in part or "image" in part:
                        image_url = ""
                        if part_type == "image_url" or "image_url" in part:
                            image_field = part.get("image_url", "")
                            if isinstance(image_field, dict):
                                image_url = image_field.get("url", "")
                            elif isinstance(image_field, str):
                                image_url = image_field
                        else:
                            image_url = part.get("image", "")
                        # DashScope 仅接受 oss:// 或 http(s)://
                        if image_url.startswith(("oss://", "http://", "https://")):
                            ds_content.append({"image": image_url})
                        else:
                            log_warning(f"[FunctionCall] DashScope 不支持此图像格式，跳过")
                ds_messages.append({"role": role, "content": ds_content or [{"text": ""}]})
            else:
                ds_messages.append({"role": role, "content": content or ""})
        
        return ds_messages

    def _initial_messages(self, goal: str) -> List[Dict[str, Any]]:
        snapshot = self.world.snapshot()
        user_payload = {
            "goal": goal,
            "world": snapshot,
            "instructions": self._mission_text,
        }
        
        content_parts = [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
            {"type": "text", "text": self._tool_instruction}
        ]
        
        # Add initial image
        img_base = self._get_current_image_base64()
        if img_base:
            content_parts.insert(0, {
                "type": "image_url",
                "image_url": {
                    "url": img_base
                }
            })
            
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content_parts},
        ]

    def _parse_arguments(self, arguments: Optional[str]) -> Dict[str, Any]:
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    def process_grasp_task(self, target_name: str, navigator, cam_name: str = "front") -> Dict[str, Any]:
        if navigator:
            self.set_navigator(navigator)
        elif self.navigator is None:
            raise ValueError("FunctionCallTaskProcessor 需要有效的导航控制器")
        self.observer.cam_name = cam_name
        self.world.set_goal(target_name)
        self._current_plan_id = uuid.uuid4().hex[:8]
        self._current_plan_entry = PlanContextEntry(
            plan_id=self._current_plan_id,
            goal=target_name,
            planned_steps=[],
        )
        self._timeline = []
        self.execution_history = []
        self._shared_runtime_extra = {}
        self.update_observation(None)
        self._publish_world_snapshot()
        self._failure_count = 0
        self._finalize_calls = 0

        messages = self._initial_messages(target_name)
        trace: List[Dict[str, Any]] = []
        final_result: Dict[str, Any] = {}

        for round_id in range(1, self.max_rounds + 1):
            log_info(f"[FunctionCall] 回合 {round_id}")
            message = self._call_llm(messages)
            messages.append(message)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # Handle multiple tool calls if present, though usually one
                for tool_call in tool_calls:
                    fn_name = tool_call.get("function", {}).get("name")
                    raw_args = tool_call.get("function", {}).get("arguments")
                    call_id = tool_call.get("id")
                    
                    args = self._parse_arguments(raw_args)
                    start_ts = time.time()
                    if fn_name == "finalize_target_pose":
                        self._finalize_calls += 1
                    result = self.tools.dispatch(fn_name, args)
                    if fn_name == "finalize_target_pose" and isinstance(result, dict):
                        result = dict(result)
                        result["finalize_attempt"] = self._finalize_calls
                    elapsed = time.time() - start_ts
                    
                    trace.append(
                        {
                            "round": round_id,
                            "function": fn_name,
                            "args": args,
                            "result": result,
                        }
                    )
                    status = "success" if result.get("ok") else "failure"
                    detail = result.get("reason") or result.get("error")
                    self._record_timeline_event(
                        stage="function_call",
                        node=fn_name or "unknown",
                        status=status,
                        detail=detail,
                        elapsed=elapsed,
                    )
                    self._append_execution_turn(
                        stage="function_call",
                        node=fn_name or "unknown",
                        status=status,
                        detail=json.dumps({"args": args, "result": result}, ensure_ascii=False),
                    )
                    
                    if status == "success":
                        log_success(f"[FunctionCall] {fn_name} 成功")
                        self._publish_world_snapshot()
                        self._failure_count = 0
                    else:
                        msg = f"[FunctionCall] {fn_name} 失败: {detail}"
                        log_warning(msg)
                        self._emit_ui_log(msg, "warning")
                        self._failure_count += 1
                        if self._failure_threshold > 0 and self._failure_count >= self._failure_threshold:
                            self._auto_recover(messages, trace, target_name)
                            self._failure_count = 0
                            
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    
                    # If finalize_target_pose has been called many times, inject a steering hint.
                    if fn_name == "finalize_target_pose" and self._finalize_calls >= 3:
                        messages.append({
                            "role": "user",
                            "content": (
                                "You have already aligned (finalize_target_pose) 3+ times. "
                                "Stop repeating finalize_target_pose; proceed to predict_grasp_point then execute_grasp."
                            ),
                        })
                    
                    if result.get("terminate"):
                        final_result = result
                        break
                
                if final_result:
                    break

                # Inject new image after tool execution for the next round
                img_base = self._get_current_image_base64()
                if img_base:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_base
                                }
                            },
                            {
                                "type": "text",
                                "text": "Current scene state after action."
                            }
                        ]
                    })
                
                continue

            # No tool call -> check if this is a valid final response or model hallucination
            content = message.get("content") or ""
            
            # Detect invalid/hallucinated responses (model outputting garbage instead of function calls)
            invalid_patterns = [
                "<|begin_of_box|>",
                "<|end_of_box|>",
                "```html",
                "```json",  # code block instead of function call
                "observe_scene\n",  # function name in text instead of call
                "predict_grasp_point\n",
                "execute_grasp\n",
                "finalize_target_pose\n",
            ]
            is_invalid_response = any(pat in content for pat in invalid_patterns)
            
            # Also check: if we haven't done key steps yet, this shouldn't be a final response
            has_grasp_result = any(
                t.get("function") in ("execute_grasp", "close_gripper") and t.get("result", {}).get("ok")
                for t in trace
            )
            
            # If response looks invalid or task isn't done, prompt model to retry with function call
            if is_invalid_response or (not has_grasp_result and "status" not in content.lower()):
                log_warning(f"[FunctionCall] 模型返回了无效响应，要求重试: {content[:100]}...")
                messages.append({
                    "role": "user",
                    "content": (
                        "ERROR: Your response was invalid. You MUST use function calls (tool_calls) to execute actions. "
                        "Do NOT output text descriptions of actions. "
                        "Please call the appropriate function now. What is the next function you need to call?"
                    )
                })
                continue
            
            try:
                final_result = json.loads(content)
            except Exception:
                # If can't parse as JSON and no grasp completed, treat as failure
                if not has_grasp_result:
                    final_result = {"status": "failure", "summary": f"Invalid model response: {content[:200]}"}
                else:
                    final_result = {"status": "success", "summary": content}
            self._record_timeline_event(
                stage="llm_response",
                node="final_response",
                status=str(final_result.get("status", "success")),
                detail=final_result.get("summary") or content,
            )
            self._append_execution_turn(
                stage="llm_response",
                node="final_response",
                status=str(final_result.get("status", "success")),
                detail=final_result.get("summary") or content,
            )
            break
        else:
            final_result = {"status": "failure", "summary": "max_rounds_exceeded"}

        success = str(final_result.get("status", "")).lower() == "success"
        if not success:
            log_warning(f"[FunctionCall] 任务失败: {final_result}")
            self._emit_ui_log(f"❌ 任务失败: {final_result.get('summary')}", "error")
        else:
            log_info(f"[FunctionCall] 任务完成: {final_result}")
            self._emit_ui_log(f"✅ 任务完成: {final_result.get('summary')}", "success")
        self._last_status_message = final_result.get("summary")
        self._finalize_plan_entry(success, final_result.get("summary") or final_result.get("reason"))
        return {
            "success": success,
            "reason": final_result.get("summary") or final_result.get("reason"),
            "final_response": final_result,
            "trace": trace,
            "timeline": self._timeline_payload(),
            "execution_history": [turn.__dict__ for turn in self.execution_history],
        }

    def _build_mission_instruction(self) -> str:
        text = textwrap.dedent(
            """
            Workflow guidance:

            General grounding rules:
            1) Always start with observe_scene(target, force_vlm=true).
            2) Any time the robot base moves (approach_far / finalize_target_pose / recover / navigation), you MUST call
            observe_scene(target, force_vlm=true) again before planning the next grasp.

            Locate:
            3) If the target is not visible, use rotate_scan and/or search_area, then observe_scene again. Repeat until visible
            or until scan attempts exceed a reasonable limit.

            Approach:
            4) When estimated_range > 2m, call approach_far, then observe_scene again; repeat until within range or limit reached.

            Align + Grasp (default: finalize once, then try grasp):
            5) Run finalize_target_pose ONCE (default behavior).
            6) After finalize_target_pose, MUST call observe_scene(target, force_vlm=true) again (camera view changed).
            7) Then run predict_grasp_point → execute_grasp → close_gripper.

            Failure handling (only on grasp execution failure):
            8) If execute_grasp OR close_gripper fails:
            - call recover(distance≈0.3)
            - observe_scene(target, force_vlm=true)
            - run finalize_target_pose ONCE as correction
            - observe_scene(target, force_vlm=true)
            - retry predict_grasp_point → execute_grasp → close_gripper
            - repeat this failure-retry loop up to grasp_attempts_max times (e.g., 3 total grasp attempts).
            If still failing after grasp_attempts_max, stop and return impossible.

            Handover:
            9) To hand the item to a user: navigate as needed, then handover_item → open_gripper.
            If navigation moved the base, re-observe before the handover step when appropriate.

            You may call observe_scene at any time when you think the situation has changed.
            Provide a final JSON summary {"status": "success|impossible", "summary": "..."} once done.
            """
        ).strip()
        return text

    def _build_tool_instruction(self) -> str:
        docs = textwrap.dedent(
            """
            Tool reference (units: meters for base poses, millimetres for TCP xyz, radians for angles):
            - observe_scene(target, phase, force_vlm, query): capture RGBD+VLM observation; optional query asks for textual analysis (adds "analysis" field).
            - rotate_scan(angle_deg) / search_area(turns, angle_deg): spin in place to look for target.
            - navigate_area(area/marker/pose) or navigate_to_pose(x,y,theta): move the base.
            - approach_far(target): coarse approach step when target is far away.
            - finalize_target_pose(): align the base using depth localization before grasping.
            - predict_grasp_point(): run ZeroGrasp; stores grasp pose inside runtime state.
            - execute_grasp(): execute the grasp pose computed previously.
            - open_gripper() / close_gripper().
            - handover_item(item): extend the item to a human after grasping.
            - get_object_state(object_id): returns the last known world model entry for that object.
            Always re-run observe_scene whenever the situation changes or after a recover.
            """
        ).strip()
        return docs

    def _finalize_plan_entry(self, success: bool, reason: Optional[str]) -> None:
        if not self._current_plan_entry:
            return
        entry = self._current_plan_entry
        entry.status = "completed" if success else "failed"
        entry.failure_reason = None if success else reason
        entry.executed = [{"node": turn.node, "status": turn.status} for turn in self.execution_history]
        entry.timestamp = time.time()
        self._current_plan_entry = None
        self._current_plan_id = None

    def _execute_internal_tool(
        self,
        fn_name: str,
        args: Dict[str, Any],
        trace: List[Dict[str, Any]],
        stage: str,
    ) -> Dict[str, Any]:
        start_ts = time.time()
        result = self.tools.dispatch(fn_name, args)
        elapsed = time.time() - start_ts
        trace.append(
            {
                "round": "auto",
                "function": fn_name,
                "args": args,
                "result": result,
                "stage": stage,
            }
        )
        status = "success" if result.get("ok") else "failure"
        detail = result.get("reason") or result.get("error")
        self._record_timeline_event(stage=stage, node=fn_name, status=status, detail=detail, elapsed=elapsed)
        self._append_execution_turn(
            stage=stage,
            node=fn_name,
            status=status,
            detail=json.dumps({"args": args, "result": result}, ensure_ascii=False),
        )
        if status == "success":
            self._publish_world_snapshot()
        else:
            self._emit_ui_log(f"[Auto] {fn_name} 失败: {detail}", "warning")
        return result

    def _auto_recover(self, messages: List[Dict[str, Any]], trace: List[Dict[str, Any]], target: str) -> None:
        log_warning("[FunctionCall] 连续失败，自动触发 recover + observe")
        recover_result = self._execute_internal_tool(
            "recover",
            {"distance": 0.3},
            trace,
            stage="auto_recover",
        )
        observe_result: Optional[Dict[str, Any]] = None
        if target:
            observe_result = self._execute_internal_tool(
                "observe_scene",
                {"target": target, "force_vlm": True},
                trace,
                stage="auto_observe",
            )
        summary = {
            "auto_recover": recover_result,
            "auto_observe": observe_result,
        }
        messages.append(
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "auto_recover": "recover + observe executed",
                        "details": summary,
                        "hint": "Please continue planning with the updated observation.",
                    },
                    ensure_ascii=False,
                ),
            }
        )
