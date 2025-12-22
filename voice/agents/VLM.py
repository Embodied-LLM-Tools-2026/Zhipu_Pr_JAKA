import base64
import json
import os
import time
import uuid
from typing import Optional, Dict, Any, List, Tuple

import dashscope
import numpy as np
import requests

from ..utils.config import Config
from tools.vision.upload_image import upload_file_and_get_url  # type: ignore
from tools.ui.ui_state_bridge import UIStateBridge  # type: ignore
from tools.logging.task_logger import log_info, log_success, log_warning, log_error  # type: ignore

from ..perception.observer import VLMObserver, ObservationContext
from ..control.executor import SkillExecutor, SkillRuntime
from .planner import BehaviorPlanner, ReflectionAdvisor
from ..control.world_model import WorldModel
from ..control.task_structures import (
    ObservationPhase,
    PlanNode,
    PlanContextEntry,
    ExecutionTurn,
    ReflectionEntry,
    ExecutionResult,
)
from ..control.recovery_manager import RecoveryManager, RecoveryContext
from ..perception.catalog_worker import SceneCatalogWorker
from ..control.apis import RobotAPI
from ..control.action_registry import ActionRegistry, ActionEntry, ActionTicket
from .engineer import EngineerAgent
from .dynamic_actions import DynamicActionRunner


# ================================
# 机器人指令处理模块
# ================================


class RobotCommandProcessor:
    """机器人指令处理器"""

    def __init__(self, api_key: Optional[str] = "", model: str = "qwen3-vl-plus"):
        """初始化指令处理器"""
        self.vlm_api_key = os.getenv("Zhipu_real_demo_API_KEY")
        self.llm_api_key = os.getenv("ZHIPUAI_API_KEY")
        self.vlm_model = Config.VLM_NAME
        self.llm_model = "GLM-4.5-Flash"
        if not self.llm_api_key:
            raise ValueError("请提供智谱AI API密钥")
        if not self.vlm_api_key:
            raise ValueError("请提供Qwen3-VL-Plus API密钥")

        from zhipuai import ZhipuAI  # type: ignore

        self.llm_client = ZhipuAI(api_key=self.llm_api_key)

        self.action_map = Config.ACTION_MAP
        self.drink_list = Config.drink_list

    def needs_web_search(self, text: str) -> bool:
        """判断用户输入是否需要联网搜索"""
        search_keywords = [
            "天气",
            "新闻",
            "股票",
            "价格",
            "汇率",
            "时间",
            "日期",
            "今天",
            "明天",
            "昨天",
            "现在",
            "实时",
            "最新",
            "当前",
            "今天天气",
            "明天天气",
            "天气预报",
            "股市",
            "基金",
            "黄金",
            "油价",
            "房价",
            "机票",
            "火车票",
            "电影票",
            "演唱会",
            "比赛",
            "比分",
            "赛程",
            "排名",
            "热搜",
            "热门",
            "趋势",
        ]

        for keyword in search_keywords:
            if keyword in text:
                return True

        time_patterns = [
            "几点",
            "什么时候",
            "多久",
            "多长时间",
            "什么时候开始",
            "什么时候结束",
        ]
        for pattern in time_patterns:
            if pattern in text:
                return True

        return False

    def web_search(self, query: str) -> str:
        """执行联网搜索"""
        try:
            response = self.llm_client.web_search.web_search(
                search_engine="search_std",
                search_query=query,
                count=1,
                search_domain_filter="www.sohu.com",
                search_recency_filter="noLimit",
                content_size="medium",
            )
            return response.search_result[0].content

        except Exception as e:
            print(f"联网搜索出错: {e}")
            return f"搜索时出现错误: {str(e)}"

    def capture_image(self, cam_name: str) -> str:
        try:
            resp = requests.get(
                f"http://127.0.0.1:8000/api/capture?cam={cam_name}&w=960&h=540",
                timeout=3,
            )
            data = resp.json()
            image_path = data["url"]
            image_url = upload_file_and_get_url(
                api_key=self.vlm_api_key, model_name=self.vlm_model, file_path=image_path
            )
        except Exception as e:
            print(f"获取图片失败: {e}")
            image_url = (
                "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
            )  # 兜底
        return image_url

    def process_command(self, text: str) -> Dict[str, Any]:
        """处理语音识别的文本，返回机器人动作指令"""

        image_url = self.capture_image("front")
        prompt = f"""
        请你扮演一台语音交互机器人上的决策模块，以下中文语音文本是用户对你说的话，请你判断用户的意图，并将判断结果以JSON格式返回。
        特别注意：若用户文本明显不是在与你沟通，如文本不完整、无意义、内容杂乱，判断结果的confidence必须设为0。
        在与用户交互时，你可以根据传入的图片辅助理解，可以在回复时加上礼貌得体的问候等,如果你不需要图片信息，可以忽略图片部分。
        
        语音文本："{text}"

        可能的意图包括：
        1. 聊天 - 普通对话内容（关键词：你好、天气、新闻、笑话等）
        2. 指令 - 控制机器人执行具体的动作（支持的动作类型：打招呼/摆手、摇头、点头、鞠躬、其他）
        请先判断用户的意图（特别地，如果用户说的话询问机器人是否能执行某个具体的动作，也请判断为对应的指令；但如果用户是询问能否做一些动作或者表演几个动作，请判断为聊天）
        如果你判断意图为聊天，intent设置为chat，action设置为unknown，confidence设置为1(特别地，如果用户的文本不完整、无意义或内容杂乱，则confidence设为0)，description设置为识别到的意图或动作
        如果你判断意图为指令，intent设置为command，action设置为识别到的动作，obj_name设置为识别到的饮料，num设置为要拿的饮料数量，confidence设置为1，description设置为识别到的意图或动作，指令对应的动作类型只包括greet, shake_head, nod, bow, get_drink, others这六种动作，前面五种就是具体的动作，而others就是除了前面五种动作之外的所有动作，比如握手就属于others。
        但是请注意让你讲一些东西不算动作，比如背诵、讲一个xx、介绍一个xx这种，这些要判断为chat。
        另外打招呼这个动作和摆摆手这个动作是等价的，所以打招呼和摆摆手都对应greet动作类型。
        输出的标准格式如下：
        {{
            "intent": "command"或"chat",
            "action": "动作类型（仅当intent为command时有效，使用英文描述，动作类型只有可能是：greet, shake_head, nod, bow, get_drink, others）",
            "obj_name": "饮料类型（仅当action为get_drink时有效，使用中文描述，饮料类型只有可能是：饮料类型只有可能是：{self.drink_list}这{len(self.drink_list)}种，其中“阿萨姆”也算“奶茶”）",
            "num": "数量（仅当action为get_drink时有效，使用数字描述，如1，2，3，4，5，6，7，8，9，10）",
            "confidence": 0或1,
            "description": "意图或动作描述"
            “response” : "当intent为command时，给出对用户的简短回应，仅当intent为command时有效"
        }}

        注意：
        1. 只返回JSON，不要其他内容
        2. 如果无法识别明确意图，intent设为"chat"
        3. confidence=1的条件：用户文本明显是在与机器人沟通，且意图明确，内容完整，无杂乱信息；否则将confidence设为0
        4. description用中文描述识别到的意图或动作
        """
        try:
            decition_start = time.time()
            response = dashscope.MultiModalConversation.call(
                api_key=self.vlm_api_key,
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"image": image_url},
                            {"text": prompt},
                        ],
                    }
                ],
                max_tokens=200,
                temperature=0.01,
                response_format={"type": "json_object"},
            )
            content = response.output.choices[0].message.content  # type: ignore[index]
            result_text = None
            if isinstance(content, list):
                if content and isinstance(content[0], dict) and "text" in content[0]:
                    result_text = content[0]["text"]
                else:
                    result_text = str(content)
            elif isinstance(content, dict):
                if "text" in content:
                    result_text = content["text"]
                else:
                    result_text = str(content)
            else:
                result_text = str(content)
            decition_end = time.time()
            print(f"大模型响应原始内容: {result_text}")
            decition_duration = decition_end - decition_start
            log_info(f"🤖 大模型决策耗时: {decition_duration:.2f} 秒")
            try:
                if isinstance(result_text, dict):
                    result = result_text
                else:
                    result = json.loads(result_text)
                print(f"大模型响应: {result}")
                if result.get("intent") == "chat" and result.get("confidence") == 1:
                    needs_search = self.needs_web_search(text)
                    search_result = ""
                    if needs_search:
                        print(f"检测到需要搜索的内容: {text}")
                        search_result = self.web_search(text)
                        print(f"搜索结果: {search_result}")
                    chat_prompt = f"""
                    用户的输入是："{text}"
                    你的名字是“家卡”，你是中国人民大学机器人创新实践基地研发的一台人型机器人，你目前会执行的动作包括：打招呼、摇头、点头、鞠躬。
                    如果用户用类似的称呼比如节卡,可能也是在叫你。请根据用户的输入，生成一个自然的回答，简单问题回答可以简短一点，复杂问题回答可以长一点，不过必须在200字以内，不然会超出token限制。注意，回答会被TTS语音系统朗读：
                 【默认行为】
                    - 大多数情况下，用户只是和你聊天或随口提问，并不是在下达任务命令。请直接进行自然的对话，不用说“我会根据你的要求回答”之类的话。也不要重复用户的输入。
                    例如：
                    用户说：1+1等于几？
                    你应该回答：1+1等于2。

                    【语气风格要求】
                    - 不要使用“好的”、“当然可以”等作为开头。
                    - 不要使用“嘿”、“哎呀”这类拟声词，也不要加表情符号。
                    - 不要在回答前加“你说得是”、“家卡同学说”等固定前缀。
                    - 不要在回答中使用表情或者任何用声音念出来会让人难以理解的词语,因为你的输出会交给TTS语音系统朗读。

                    【身份规则】
                    - 你是一个带语音交互功能的有单手且装有轮子的人形机器人。
                    - 用户叫你“家卡”时，是在叫你，不是说公司。
                    - 你底层的大模型用的是智谱的GLM-4.5-Flash
                    - 你现在所在的地方是北京市海淀区中国人民大学机器人创新实践基地。
                    - 在与用户交互时，你可以根据传入的图片辅助理解，可以在回复时加上礼貌得体的问候等,如果你不需要图片信息，可以忽略图片部分。

                    【你会的东西】
                    你目前会执行的动作包括：打招呼、摇头、点头、鞠躬。但每次只能做一个，比如你不能一边点头一边鞠躬。
                    其他例如收拾垃圾，拿取物品，叠放衣服等动作正在加紧学习中，很快就会学会。
                    然后你还能提供智能对话、信息整理等服务。

                    【能力边界】
                    - 如果用户提出你目前无法完成的请求（例如让你炒饭），你可以委婉拒绝，并说你以后会继续学习改进，但不要说你没有实体，也不要说你只是个语音助手，因为我是把你部署到一个人形机器人上的，就说你还没学会。

                    【关于联网搜索】
                    为了帮助你回答问题，如果用户的问题是需要联网找的，我已经帮你得到了搜索结果如下，你可以作为参考，如果没得到搜索结果，代表这个问题不需要联网搜索，那你不用管。
                    联网搜索结果："{search_result}"
                """
                    try:
                        chat_start = time.time()
                        chat_response = dashscope.MultiModalConversation.call(
                            api_key=self.vlm_api_key,
                            model=self.vlm_model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [{"text": chat_prompt}],
                                }
                            ],
                            max_tokens=300,
                            temperature=0.2,
                        )
                        chat_content = chat_response.output.choices[0].message.content  # type: ignore[index]
                        if isinstance(chat_content, list):
                            if chat_content and isinstance(chat_content[0], dict):
                                chat_text = chat_content[0].get("text", "")
                            else:
                                chat_text = "".join(str(item) for item in chat_content)
                        elif isinstance(chat_content, dict):
                            chat_text = chat_content.get("text", "")
                        else:
                            chat_text = str(chat_content)
                        chat_duration = time.time() - chat_start
                        log_info(f"💬 聊天回复耗时: {chat_duration:.2f} 秒")
                        result["response"] = chat_text.strip()
                    except Exception as chat_exc:
                        log_error(f"❌ 生成聊天回复失败: {chat_exc}")
                        result["response"] = "抱歉，我暂时无法回答这个问题。"
                    return result
                return result
            except (json.JSONDecodeError, TypeError) as parse_err:
                log_error(f"❌ 解析大模型输出失败: {parse_err}")
                return {
                    "intent": "chat",
                    "confidence": 0,
                    "description": "无法解析模型响应",
                    "response": "抱歉，我暂时没有听懂，可以再说一遍吗？",
                }
        except Exception as exc:
            log_error(f"❌ 处理指令失败: {exc}")
            return {
                "intent": "chat",
                "confidence": 0,
                "description": "系统出现异常",
                "response": "抱歉，系统出现了一些问题，请稍后再试。",
            }


# ================================
# 行为树运行器
# ================================


class BehaviorTreeRunner:
    """轻量级行为树解释器，支持运行时check/selector以及repeat_until循环。"""

    def __init__(self, root: Optional[PlanNode] = None, world_model=None) -> None:
        self.root = root
        self.world_model = world_model
        self.pending_node: Optional[PlanNode] = None
        self.pending_node_id: Optional[int] = None
        self.action_results: Dict[int, str] = {}
        self.last_status: str = "idle"
        self.last_failure_reason: Optional[str] = None
        self._node_states: Dict[int, Dict[str, Any]] = {}

    def reset(self, root: Optional[PlanNode]) -> None:
        self.root = root
        self.pending_node = None
        self.pending_node_id = None
        self.action_results.clear()
        self.last_status = "idle"
        self.last_failure_reason = None
        self._node_states.clear()

    def tick(self) -> Tuple[Optional[PlanNode], str]:
        if not self.root:
            self.last_status = "success"
            return None, "success"
        if self.pending_node is not None:
            # 尚未提交上一个动作的结果
            return self.pending_node, "running"
        status, action = self._tick_node(self.root)
        self.last_status = status
        if action is not None:
            self.pending_node = action
            self.pending_node_id = id(action)
        return action, status

    def apply_action_result(self, node: PlanNode, success: bool) -> None:
        node_id = id(node)
        if self.pending_node_id != node_id:
            # 忽略不匹配的节点，避免状态错乱
            return
        self.pending_node = None
        self.pending_node_id = None
        self.action_results[node_id] = "success" if success else "failure"

    def _tick_node(self, node: PlanNode) -> Tuple[str, Optional[PlanNode]]:
        node_type = node.type
        if node_type == "check":
            cond = node.args.get("cond", "")
            result = (
                "success"
                if self.world_model and self.world_model.evaluate_condition(cond)
                else "failure"
            )
            if result == "failure":
                self.last_failure_reason = f"check '{cond}' evaluated False"
            return result, None
        if node_type == "action":
            node_id = id(node)
            if node_id in self.action_results:
                result = self.action_results.pop(node_id)
                if result == "failure":
                    self.last_failure_reason = f"action '{node.name or node.type}' reported failure"
                return result, None
            return "running", node
        if node_type == "sequence":
            return self._tick_sequence(node)
        if node_type == "selector":
            return self._tick_selector(node)
        if node_type == "repeat_until":
            return self._tick_repeat_until(node)
        self.last_failure_reason = f"unsupported node type: {node_type}"
        return "failure", None

    def _node_state(self, node: PlanNode) -> Dict[str, Any]:
        return self._node_states.setdefault(id(node), {})

    def _tick_sequence(self, node: PlanNode) -> Tuple[str, Optional[PlanNode]]:
        children = node.children
        if not children:
            return "success", None
        state = self._node_state(node)
        idx = int(state.get("index", 0))
        while idx < len(children):
            child = children[idx]
            status, action = self._tick_node(child)
            if action is not None:
                state["index"] = idx
                return "running", action
            if status == "failure":
                state["index"] = 0
                return "failure", None
            idx += 1
        state["index"] = 0
        return "success", None

    def _tick_selector(self, node: PlanNode) -> Tuple[str, Optional[PlanNode]]:
        children = node.children
        if not children:
            return "failure", None
        state = self._node_state(node)
        idx = int(state.get("index", 0))
        while idx < len(children):
            child = children[idx]
            status, action = self._tick_node(child)
            if action is not None:
                state["index"] = idx
                return "running", action
            if status == "success":
                state["index"] = 0
                return "success", None
            idx += 1
        state["index"] = 0
        return "failure", None

    def _tick_repeat_until(self, node: PlanNode) -> Tuple[str, Optional[PlanNode]]:
        cond = node.args.get("cond", "")
        if self.world_model and self.world_model.evaluate_condition(cond):
            self._node_state(node)["index"] = 0
            return "success", None
        children = node.children
        if not children:
            self.last_failure_reason = f"repeat_until missing children before '{cond}' became true"
            return "failure", None
        state = self._node_state(node)
        idx = int(state.get("index", 0))
        while idx < len(children):
            child = children[idx]
            status, action = self._tick_node(child)
            if action is not None:
                state["index"] = idx
                return "running", action
            if status == "failure":
                state["index"] = 0
                self.last_failure_reason = (
                    self.last_failure_reason or f"repeat_until children failed before '{cond}' became true"
                )
                return "failure", None
            idx += 1
        state["index"] = 0
        return "running", None


# ================================
# 视觉-执行管线 TaskProcessor
# ================================


class TaskProcessor:
    """
    新版任务处理器：Planner(LLM) → Executor(技能) → Observer(VLM) 闭环。
    """

    def __init__(self, navigator=None) -> None:
        self.navigator = navigator
        self.world = WorldModel()
        self.observer = VLMObserver()
        self.executor = SkillExecutor(navigator=navigator)
        self.planner = BehaviorPlanner()
        self.catalog_worker = SceneCatalogWorker(
            self.world, vlm_api_key=self.observer.vlm_api_key, vlm_model=self.observer.vlm_model
        )
        disable_bridge = os.getenv("DISABLE_ROBOT_UI_BRIDGE", "").lower() in {"1", "true", "yes"}
        self.ui_bridge = None if disable_bridge else UIStateBridge(os.getenv("ROBOT_UI_URL"))
        self.current_plan = None
        self.plan_runner = BehaviorTreeRunner(root=None, world_model=self.world)
        self.max_iter = 20
        self._observation_counter = 0
        self.plan_context: List[PlanContextEntry] = []
        self.execution_history: List[ExecutionTurn] = []
        self.reflection_log: List[ReflectionEntry] = []
        self._active_plan_log: List[Dict[str, Any]] = []
        self._plan_history_limit = int(os.getenv("PLAN_CONTEXT_LIMIT", "6"))
        self._execution_history_limit = int(os.getenv("EXEC_HISTORY_LIMIT", "60"))
        self._reflection_limit = int(os.getenv("REFLECTION_HISTORY_LIMIT", "8"))
        self._last_action_name: Optional[str] = None
        self._timeline: List[Dict[str, Any]] = []
        self._timeline_limit = int(os.getenv("EXEC_TIMELINE_LIMIT", "30"))
        self.recovery_manager = RecoveryManager()
        self._recovery_budget: Dict[str, Any] = {"total": 0, "per_code": {}, "start_time": time.time()}
        self._last_vlm_ts = 0.0
        self._last_tracker_ts = 0.0
        self._vlm_refresh_interval = float(os.getenv("VLM_REFRESH_INTERVAL", "3.0"))
        self._failure_notice_threshold = int(os.getenv("FAIL_NOTICE_THRESHOLD", "2"))
        self._last_status_message: Optional[str] = None
        self._current_plan_id: Optional[str] = None
        self._current_plan_entry: Optional[PlanContextEntry] = None
        self._next_plan_hint: Optional[str] = None
        self.action_registry = ActionRegistry()
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
            registry=self.action_registry,
        )
        self._engineer_enabled = os.getenv("ENABLE_CODE_ENGINEER", "false").lower() in {"1", "true", "yes"}
        self._dynamic_actions_enabled = os.getenv("ENABLE_DYNAMIC_ACTIONS", "false").lower() in {"1", "true", "yes"}
        self.dynamic_runner: Optional[DynamicActionRunner] = None
        if self._dynamic_actions_enabled:
            self.dynamic_runner = DynamicActionRunner(self.api, self.action_registry)
        self.engineer: Optional[EngineerAgent] = None
        if self._engineer_enabled:
            self.engineer = EngineerAgent(self.action_registry, self.api)

    # ------------------------------------------------------------------
    def set_navigator(self, navigator) -> None:
        self.navigator = navigator
        self.executor.set_navigator(navigator)
        if hasattr(self, "api") and self.api:
            self.api.update_navigator(navigator)

    def _publish_world_snapshot(self) -> None:
        if not self.ui_bridge:
            return
        try:
            snapshot = self.world.snapshot()
        except Exception:
            return
        self.ui_bridge.post_world_model(snapshot)

    def _publish_plan_state(self) -> None:
        if not self.ui_bridge:
            return
        timeline_payload = self._timeline_payload()
        if not self.current_plan:
            self.ui_bridge.post_plan_state(
                root=None,
                steps=[],
                metadata={},
                current_index=-1,
                current_node=None,
                timeline=timeline_payload,
            )
            return
        steps_payload = [
            {
                "type": node.type,
                "name": node.name,
                "args": node.args or {},
            }
            for node in self.current_plan.steps
        ]
        current_index = -1
        current_node = None
        if self._last_action_name and steps_payload:
            for idx, node in enumerate(self.current_plan.steps):
                if (node.name or node.type) == self._last_action_name:
                    current_index = idx
                    current_node = node.name or node.type
                    break
        try:
            root_dict = self.current_plan.root.to_dict()
        except Exception:
            root_dict = None
        metadata_payload = dict(self.current_plan.metadata or {})
        metadata_payload.setdefault("source", metadata_payload.get("source", "planner"))
        metadata_payload["timeline_size"] = len(self._timeline)
        self.ui_bridge.post_plan_state(
            root=root_dict,
            steps=steps_payload,
            metadata=metadata_payload,
            current_index=current_index,
            current_node=current_node,
            timeline=timeline_payload,
        )

    def _ensure_plan(self, target_name: str) -> None:
        recent_history = [entry.to_prompt_dict() for entry in self.plan_context[-self._plan_history_limit :]]
        if self._next_plan_hint:
            recent_history.append(
                {
                    "phase": "reflection_hint",
                    "hint": self._next_plan_hint,
                }
            )
        self.current_plan = self.planner.make_plan(target_name, self.world, plan_context=recent_history)
        self.plan_runner.reset(self.current_plan.root)
        self._active_plan_log = []
        self._last_action_name = None
        self._timeline.clear()
        self._last_status_message = None
        step_names = [node.name or node.type for node in self.current_plan.steps]
        self._current_plan_id = uuid.uuid4().hex[:8]
        self._current_plan_entry = PlanContextEntry(
            plan_id=self._current_plan_id,
            goal=target_name,
            planner_thought=self.current_plan.metadata.get("thought"),
            planned_steps=step_names,
        )
        log_info(f"🧠 当前计划: {step_names}")
        self._publish_plan_state()

    def _estimate_distance_from_state(self, obj_state: Optional[Any]) -> Optional[float]:
        if obj_state is None:
            return None
        range_est = obj_state.attrs.get("range_estimate") if hasattr(obj_state, "attrs") else None
        if range_est is not None:
            try:
                return float(range_est)
            except (TypeError, ValueError):
                pass
        if getattr(obj_state, "world_center", None):
            wc = np.array(obj_state.world_center, dtype=float)
            return float(np.linalg.norm(wc[:2]))
        if getattr(obj_state, "robot_center", None):
            rc = np.array(obj_state.robot_center, dtype=float)
            return float(np.linalg.norm(rc[:2]) / 1000.0)
        return None

    def _determine_phase(self, obj_state: Optional[Any]) -> ObservationPhase:
        if obj_state and getattr(obj_state, "visible", False):
            dist = self._estimate_distance_from_state(obj_state)
            if dist is None or dist <= 2.0:
                return ObservationPhase.APPROACH
        return ObservationPhase.SEARCH

    def _perform_observation(self, target_name: str, *, force_vlm: bool = False) -> tuple[Any, Dict[str, Any]]:
        self._observation_counter += 1
        obj_state = self.world.objects.get(target_name)
        phase = self._determine_phase(obj_state)
        context = ObservationContext(
            step=self._observation_counter,
            max_steps=self.max_iter,
        )
        should_force = self._should_force_vlm(force_vlm)
        observation, frontend_payload = self.observer.observe(
            target_name, phase, context, self.navigator, force_vlm=should_force
        )
        now_ts = time.time()
        source = getattr(observation, "source", "vlm")
        if source == "vlm":
            self._last_vlm_ts = now_ts
        else:
            self._last_tracker_ts = now_ts
        summary = self._summarize_observation(observation)
        setattr(observation, "summary", summary)
        announce_level = "success" if getattr(observation, "found", False) else "warning"
        self._announce_status(
            f"👀 观测[{source}] {summary}",
            announce_level,
            push_hint=announce_level != "success",
        )
        event_status = "success" if getattr(observation, "found", False) else "observe"
        self._record_timeline_event(
            stage="observe",
            node="observe_scene",
            status=event_status,
            detail=summary,
        )
        self._append_execution_turn(
            stage="observe",
            node="observe_scene",
            status=event_status,
            observation=summary,
        )

        pose_info = self.executor.estimate_observation_pose(observation, self.navigator)
        distance_attr: Optional[Dict[str, Any]] = None
        if pose_info:
            observation.camera_center = pose_info.get("camera_center")
            observation.robot_center = pose_info.get("robot_center")
            observation.world_center = pose_info.get("world_center")
            distance_attr = {}
            distance_m: Optional[float] = None
            depth_derived = False

            robot_center = observation.robot_center
            if robot_center and len(robot_center) >= 3:
                rc = np.asarray(robot_center[:3], dtype=float)
                distance_m = float(np.linalg.norm(rc[1:3]) / 1000.0)
                depth_derived = True
            elif observation.world_center and observation.robot_pose:
                target_xy = np.asarray(observation.world_center[:2], dtype=float)
                pose = observation.robot_pose
                robot_xy = np.array([float(pose.get("x", 0.0)), float(pose.get("y", 0.0))], dtype=float)
                distance_m = float(np.linalg.norm(target_xy - robot_xy))
                depth_derived = True
            elif observation.range_estimate is not None:
                distance_m = float(observation.range_estimate)

            if distance_m is not None:
                observation.range_estimate = distance_m
                distance_attr["range_estimate"] = distance_m
                if depth_derived:
                    distance_attr["range_source"] = "depth_localization"
            if not distance_attr:
                distance_attr = None

        self.world.update_from_observation(target_name, observation)
        if pose_info:
            # todo : no need to store every center,only world center is ok
            self.world.update_pose_estimate(
                target_name,
                camera_center=pose_info.get("camera_center"),
                robot_center=pose_info.get("robot_center"),
                world_center=pose_info.get("world_center"),
                confidence=pose_info.get("confidence"),
                attrs=distance_attr,
            )
            # todo : merge depth get from frontend
            depth_bundle = observation.depth_snapshot
            if depth_bundle and observation.original_image_path and self.navigator is not None:
                try:
                    job = {
                        "image_path": observation.original_image_path,
                        "depth_map": depth_bundle.depth,
                        "depth_intrinsics": depth_bundle.intrinsics,
                        "extrinsic": depth_bundle.extrinsic,
                        "robot_pose": observation.robot_pose,
                    }
                    self.catalog_worker.submit(job)
                except Exception as exc:
                    log_warning(f"⚠️ 提交场景建模任务失败: {exc}")

        self._publish_world_snapshot()
        return observation, frontend_payload

    # ------------------------------------------------------------------
    def _append_plan_context(self, entry: PlanContextEntry) -> None:
        self.plan_context.append(entry)
        if len(self.plan_context) > self._plan_history_limit:
            self.plan_context = self.plan_context[-self._plan_history_limit :]

    def _trigger_reflection(self, trigger: str, reason: Optional[str], plan_entry: PlanContextEntry) -> None:
        if not self.reflection:
            return
        relevant_turns = [turn for turn in self.execution_history if turn.plan_id == plan_entry.plan_id]
        reflection = self.reflection.reflect(plan_entry.goal, plan_entry, relevant_turns)
        if not reflection:
            return
        diagnosis = reflection.get("diagnosis") or reason or "执行失败"
        adjustment_hint = reflection.get("adjustment_hint")
        try:
            confidence = float(reflection.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        entry = ReflectionEntry(
            plan_id=plan_entry.plan_id,
            goal=plan_entry.goal,
            trigger=trigger,
            diagnosis=diagnosis,
            adjustment_hint=adjustment_hint,
            confidence=confidence,
        )
        self.reflection_log.append(entry)
        if len(self.reflection_log) > self._reflection_limit:
            self.reflection_log = self.reflection_log[-self._reflection_limit :]
        if adjustment_hint:
            self._next_plan_hint = adjustment_hint
        self._emit_ui_log(f"🧠 反思: {diagnosis}", "warning")

    def _request_action_ticket(self, action_name: str, action_args: Dict[str, Any], failure_reason: str) -> None:
        description = (
            f"需要实现新的动作 '{action_name}'，原因: {failure_reason}. "
            "输入参数参考 args 字段，输出应返回执行状态。"
        )
        ticket = self.action_registry.create_ticket(
            goal=self.world.goal or "unknown",
            description=description,
            inputs={"args": action_args},
            outputs={"status": "success|failure", "evidence": "dict"},
            constraints={"suggested_name": action_name, "failure_reason": failure_reason},
        )
        self._emit_ui_log(f"📮 已创建动作需求单 {ticket.ticket_id} -> {action_name}", "info")
        if self.engineer:
            self.engineer.process_ticket(ticket)

    def request_action(
        self,
        name: str,
        description: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ActionTicket:
        ticket = self.action_registry.create_ticket(
            goal=self.world.goal or "unknown",
            description=description,
            inputs=inputs or {},
            outputs=outputs or {},
            constraints=constraints or {"suggested_name": name},
        )
        self._emit_ui_log(f"📮 Planner 手动创建动作需求单 {ticket.ticket_id} -> {name}", "info")
        if self.engineer:
            self.engineer.process_ticket(ticket)
        return ticket

    def list_action_tickets(self) -> List[Dict[str, Any]]:
        return self.action_registry.list_tickets()

    def list_registered_actions(self) -> List[Dict[str, Any]]:
        return self.action_registry.list_actions()

    def list_primitives(self) -> List[Dict[str, Any]]:
        return self.action_registry.list_primitives()

    def _latest_reflection(self) -> Optional[Dict[str, Any]]:
        if not self.reflection_log:
            return None
        return self.reflection_log[-1].to_dict()

    def _maybe_recover_bt(
        self, action_node: PlanNode, result: ExecutionResult, runtime: SkillRuntime, step: int
    ) -> Optional[ExecutionResult]:
        """Invoke RecoveryManager for BT failures and optionally retry the original action."""
        failure_code = result.failure_code or FailureCode.UNKNOWN
        budget = {
            "total": self._recovery_budget.get("total", 0),
            "per_code": self._recovery_budget.get("per_code", {}),
            "elapsed_s": time.time() - self._recovery_budget.get("start_time", time.time()),
        }
        history_tail = [
            {"node": t.node, "status": t.status, "detail": t.detail}
            for t in self.execution_history[-5:]
        ]
        ctx = RecoveryContext(
            episode_id=self._current_plan_id,
            step_id=step,
            task_goal=self.world.goal,
            world_snapshot=None,
            history_tail=history_tail,
            budget=budget,
        )
        decision = self.recovery_manager.handle_failure(failure_code, ctx)
        # update budgets
        per_code = self._recovery_budget.setdefault("per_code", {})
        per_code[failure_code.value] = per_code.get(failure_code.value, 0) + 1
        self._recovery_budget["total"] = self._recovery_budget.get("total", 0) + 1

        if decision.kind == "EXECUTE_ACTIONS":
            all_ok = True
            last_rec_result: Optional[ExecutionResult] = None
            for idx, action in enumerate(decision.actions, start=1):
                rec_runtime = SkillRuntime(
                    navigator=runtime.navigator,
                    world_model=runtime.world_model,
                    observation=runtime.observation,
                    extra=dict(runtime.extra or {}),
                )
                rec_runtime.extra.update(
                    {
                        "recovery_level": decision.level,
                        "recovery_attempt_idx": idx,
                        "recovery_policy": decision.reason,
                        "recovery_triggered": True,
                    }
                )
                rec_node = PlanNode(type="action", name=action["skill_name"], args=action.get("args", {}))
                rec_result = self.executor.execute(rec_node, rec_runtime)
                last_rec_result = rec_result
                self._record_timeline_event(
                    stage="recovery",
                    node=action["skill_name"],
                    status=rec_result.status,
                    detail=f"{decision.level}:{decision.reason}",
                    elapsed=rec_result.elapsed,
                )
                self._append_execution_turn(
                    stage="recovery",
                    node=action["skill_name"],
                    status=rec_result.status,
                    detail=rec_result.reason or decision.reason,
                    evidence=rec_result.evidence,
                )
                if rec_result.status != "success":
                    all_ok = False
                    if result.evidence is None:
                        result.evidence = {}
                    result.evidence["recovery_side_effect_failure_code"] = (
                        rec_result.failure_code.value if rec_result.failure_code else None
                    )
                    break
            if all_ok:
                retry_runtime = SkillRuntime(
                    navigator=runtime.navigator,
                    world_model=runtime.world_model,
                    observation=runtime.observation,
                    extra=dict(runtime.extra or {}),
                )
                retry_runtime.extra["recovery_level"] = decision.level
                retry_runtime.extra["recovery_policy"] = decision.reason
                retry_result = self.executor.execute(action_node, retry_runtime)
                if retry_result.evidence is None:
                    retry_result.evidence = {}
                retry_result.evidence["recovery_decision"] = decision.evidence
                retry_result.evidence["recovery_success"] = True
                return retry_result
            if result.evidence is None:
                result.evidence = {}
            result.evidence["recovery_decision"] = decision.evidence
            result.evidence["recovery_success"] = False
            return last_rec_result or result

        if decision.kind == "ESCALATE_L3":
            if result.evidence is None:
                result.evidence = {}
            result.evidence["recovery_decision"] = decision.evidence
            result.reason = result.reason or "escalate_L3"
            return result

        # ABORT
        if result.evidence is None:
            result.evidence = {}
        result.evidence["recovery_decision"] = decision.evidence
        result.reason = result.reason or "recovery_abort"
        return result

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False, default=str)
            return value
        except TypeError:
            if isinstance(value, dict):
                return {k: TaskProcessor._json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [TaskProcessor._json_safe(v) for v in value]
            return str(value)

    def _emit_ui_log(self, message: str, level: str = "info") -> None:
        if not self.ui_bridge or not message:
            return
        try:
            self.ui_bridge.post_task_log(message=message, level=level)
        except Exception:
            pass

    def _push_suggestion(self, message: str, level: str = "info") -> None:
        if not self.ui_bridge or not message:
            return
        try:
            self.ui_bridge.post_suggestion(message=message, level=level)
        except Exception:
            pass

    def _announce_status(self, message: str, level: str = "info", *, push_hint: bool = False) -> None:
        if not message:
            return
        self._last_status_message = message
        status_entry = {
            "message": message,
            "level": level,
            "ts": time.time(),
        }
        self.world.task_memory["last_status"] = status_entry
        if level in {"error", "warning"}:
            log_warning(message)
        else:
            log_info(message)
        self._emit_ui_log(message, level)
        if push_hint or level in {"error", "warning"}:
            self._push_suggestion(message, level)

    def _handle_runtime_notifications(self, runtime: Optional[SkillRuntime]) -> None:
        if runtime is None or not runtime.extra:
            return
        notes = runtime.extra.get("notifications") or []
        if not isinstance(notes, list):
            return
        for note in notes:
            if isinstance(note, dict):
                msg = note.get("message")
                level = note.get("level", "info")
            else:
                msg = str(note)
                level = "info"
            if not msg:
                continue
            self._push_suggestion(msg, level)
            self._emit_ui_log(msg, level)
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
        observation: Optional[str] = None,
        action: Optional[str] = None,
        detail: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._current_plan_id:
            return
        turn = ExecutionTurn(
            plan_id=self._current_plan_id,
            stage=stage,
            node=node,
            status=status,
            observation=observation,
            action=action,
            detail=detail,
            evidence=self._json_safe(evidence) if evidence else None,
        )
        self.execution_history.append(turn)
        if len(self.execution_history) > self._execution_history_limit:
            self.execution_history = self.execution_history[-self._execution_history_limit :]

    def _should_force_vlm(self, requested: bool) -> bool:
        if requested:
            return True
        if not getattr(self.observer, "tracker", None):
            return True
        if not self.observer.tracker.is_active():
            return True
        now = time.time()
        if (now - self._last_vlm_ts) > self._vlm_refresh_interval:
            return True
        return False

    def _summarize_observation(self, observation: Optional[Any]) -> str:
        if observation is None:
            return "未获得观测结果"
        source = getattr(observation, "source", "vlm")
        found = bool(getattr(observation, "found", False))
        confidence = getattr(observation, "confidence", None)
        range_est = getattr(observation, "range_estimate", None)
        bbox = getattr(observation, "bbox", None)
        parts = [f"{source.upper()} {'FOUND' if found else 'MISS'}"]
        if confidence is not None:
            try:
                parts.append(f"conf={float(confidence):.2f}")
            except Exception:
                parts.append(f"conf={confidence}")
        if range_est is not None:
            try:
                parts.append(f"dist={float(range_est):.2f}m")
            except Exception:
                parts.append(f"dist={range_est}")
        if bbox:
            bbox_text = ",".join(str(int(v)) for v in bbox[:4])
            parts.append(f"bbox=[{bbox_text}]")
        return " | ".join(parts)

    def _format_skill_detail(
        self,
        node_name: str,
        result: ExecutionResult,
        runtime: Optional[SkillRuntime],
    ) -> str:
        evidence = result.evidence or {}
        pieces: List[str] = []
        if "best_score" in evidence:
            pieces.append(f"score={evidence['best_score']:.3f}")
        if "tcp_position_mm" in evidence:
            pieces.append(f"tcp={evidence['tcp_position_mm']}")
        if "rot_vec" in evidence:
            pieces.append(f"rot={evidence['rot_vec']}")
        if "grasp_count" in evidence:
            pieces.append(f"grasps={evidence['grasp_count']}")
        if runtime and runtime.extra:
            summary = runtime.extra.get("observation_summary")
            if summary:
                pieces.append(f"obs:{summary}")
        if result.reason and result.status != "success":
            pieces.append(f"reason={result.reason}")
        if not pieces:
            pieces.append(result.status)
        return " | ".join(str(p) for p in pieces)

    def _finalize_plan_iteration(self, status: str, reason: Optional[str] = None) -> None:
        entry = self._current_plan_entry
        if entry is None:
            entry = PlanContextEntry(
                plan_id=self._current_plan_id or uuid.uuid4().hex[:8],
                goal=self.world.goal or "unknown",
                planned_steps=[node.name or node.type for node in self.current_plan.steps] if self.current_plan else [],
            )
        entry.status = status
        entry.failure_reason = reason if status != "completed" else None
        entry.executed = [
            {"node": log.get("node"), "status": log.get("status")}
            for log in self._active_plan_log
        ]
        entry.timestamp = time.time()
        self._append_plan_context(entry)
        self._current_plan_entry = None
        self._current_plan_id = None
        if status == "completed":
            self._next_plan_hint = None
        else:
            self._trigger_reflection(status, reason, entry)
        self._active_plan_log = []

    # ------------------------------------------------------------------
    def process_grasp_task(self, target_name: str, navigator, cam_name: str = "front") -> Dict[str, Any]:
        """执行抓取任务主循环。"""
        if navigator:
            self.set_navigator(navigator)
        elif self.navigator is None:
            raise ValueError("TaskProcessor 需要有效的导航控制器")

        self.observer.cam_name = cam_name
        self.world.set_goal(target_name)
        self.current_plan = None
        self.plan_runner.reset(None)
        self._observation_counter = 0
        self._publish_world_snapshot()
        self._publish_plan_state()

        try:
            current_observation, current_frontend_payload = self._perform_observation(
                target_name, force_vlm=True
            )
        except Exception as exc:
            log_error(f"❌ 初始观测失败: {exc}")
            self._emit_ui_log(f"❌ 初始观测失败: {exc}", "error")
            self._record_timeline_event(
                stage="observe",
                node="observe_scene",
                status="failure",
                detail=str(exc),
            )
            return {
                "success": False,
                "reason": f"initial_observe_failed: {exc}",
                "timeline": self._timeline_payload(),
                "status": self._last_status_message,
                "reflection": self._latest_reflection(),
            }

        for step in range(1, self.max_iter + 1):
            if not self.current_plan:
                self._ensure_plan(target_name)
                if not self.current_plan or not self.current_plan.root:
                    log_error("❌ 规划结果为空")
                    return {
                        "success": False,
                        "reason": "planner_return_empty",
                        "timeline": self._timeline_payload(),
                        "status": self._last_status_message,
                        "reflection": self._latest_reflection(),
                    }

            action_node, bt_status = self.plan_runner.tick()
            if action_node is None:
                if bt_status == "success":
                    self._finalize_plan_iteration("completed", "plan_executed")
                    log_success("🎯 行为树计划执行完成")
                    return {
                        "success": True,
                        "result": current_observation.raw_response if current_observation else None,
                        "annotated_url": current_observation.annotated_url if current_observation else None,
                        "timeline": self._timeline_payload(),
                        "status": self._last_status_message,
                        "reflection": self._latest_reflection(),
                    }
                if bt_status == "failure":
                    fail_reason = self.plan_runner.last_failure_reason or "unknown"
                    log_warning(f"⚠️ 行为树失败（{fail_reason}），重新规划")
                    self._finalize_plan_iteration("failed", f"bt_failure:{fail_reason}")
                    self.current_plan = None
                    self.plan_runner.reset(None)
                    current_observation = None
                    current_frontend_payload = None
                    self._last_action_name = None
                    self._publish_plan_state()
                    continue
                # running但未返回动作，视为异常，触发重规划
                log_warning("⚠️ 行为树未返回动作，触发重规划")
                self._finalize_plan_iteration("replan", "no_action")
                self.current_plan = None
                self.plan_runner.reset(None)
                current_observation = None
                current_frontend_payload = None
                self._last_action_name = None
                self._publish_plan_state()
                continue

            node_name = action_node.name or action_node.type
            self._last_action_name = node_name

            if node_name == "observe_scene":
                try:
                    current_observation, current_frontend_payload = self._perform_observation(
                        target_name, force_vlm=bool(action_node.args.get("force_vlm"))
                    )
                    exec_record = {"node": "observe_scene", "status": "success"}
                except Exception as exc:
                    log_error(f"❌ 观测失败: {exc}")
                    self._emit_ui_log(f"❌ 观测失败: {exc}", "error")
                    self._record_timeline_event(
                        stage="observe",
                        node="observe_scene",
                        status="failure",
                        detail=str(exc),
                    )
                    self._append_execution_turn(
                        stage="observe",
                        node="observe_scene",
                        status="failure",
                        detail=str(exc),
                    )
                    self._publish_plan_state()
                    exec_record = {"node": "observe_scene", "status": "failure", "reason": str(exc)}
                    current_observation = None
                    current_frontend_payload = None
                self.world.record_execution_result(exec_record)
                self._active_plan_log.append(exec_record)
                self.plan_runner.apply_action_result(action_node, exec_record["status"] == "success")
                if exec_record["status"] != "success":
                    continue
                self._publish_world_snapshot()
                self._publish_plan_state()
                continue

            if current_observation is None:
                try:
                    current_observation, current_frontend_payload = self._perform_observation(
                        target_name, force_vlm=False
                    )
                except Exception as exc:
                    log_error(f"❌ 兜底观测失败: {exc}")
                    self._emit_ui_log(f"❌ 兜底观测失败: {exc}", "error")
                    self._record_timeline_event(
                        stage="observe",
                        node="observe_scene",
                        status="failure",
                        detail=str(exc),
                    )
                    self._append_execution_turn(
                        stage="observe",
                        node="observe_scene",
                        status="failure",
                        detail=str(exc),
                    )
                    self._publish_plan_state()
                    self.plan_runner.apply_action_result(action_node, False)
                    continue

            runtime_extra = {
                "step": step,
                "node": node_name,
            }
            if self._current_plan_id:
                runtime_extra["episode_id"] = self._current_plan_id
            if current_observation is not None:
                runtime_extra["observation_source"] = getattr(current_observation, "source", "unknown")
                runtime_extra["observation_found"] = bool(getattr(current_observation, "found", False))
                summary_text = getattr(current_observation, "summary", None)
                if summary_text:
                    runtime_extra["observation_summary"] = summary_text
            runtime = SkillRuntime(
                navigator=self.navigator,
                world_model=self.world,
                observation=current_observation,
                frontend_payload=current_frontend_payload,
                surface_points=current_observation.surface_points if current_observation else None,
                extra=runtime_extra,
            )
            handler_name = f"_skill_{node_name}"
            handler_exists = hasattr(self.executor, handler_name)
            if (
                not handler_exists
                and self.dynamic_runner
                and self.dynamic_runner.has_action(node_name)
            ):
                result = self.dynamic_runner.execute(node_name, action_node.args or {}, runtime)
            else:
                result = self.executor.execute(action_node, runtime)
            exec_record = {
                "node": node_name,
                "status": result.status,
                "reason": result.reason,
                "evidence": result.evidence,
            }
            detail = self._format_skill_detail(node_name, result, runtime)
            self._record_timeline_event(
                stage="skill",
                node=node_name,
                status=result.status,
                detail=detail,
                elapsed=result.elapsed,
            )
            observation_text = None
            if runtime and runtime.extra:
                observation_text = runtime.extra.get("observation_summary")
            self._append_execution_turn(
                stage="skill",
                node=node_name,
                status=result.status,
                observation=observation_text,
                action=node_name,
                detail=result.reason or detail,
                evidence=result.evidence,
            )
            self._handle_runtime_notifications(runtime)
            announce_level = "success" if result.success else "warning"
            self._announce_status(
                f"⚙️ {node_name}: {detail}",
                announce_level,
                push_hint=not result.success,
            )
            self.world.record_execution_result(exec_record)
            if result.reason == "unsupported_skill":
                self._request_action_ticket(node_name, action_node.args or {}, "unsupported_skill")
            if result.status != "success":
                fail_counts = self.world.task_memory.get("fail_counts", {})
                fail_total = fail_counts.get(node_name, 0)
                if fail_total >= self._failure_notice_threshold:
                    self._push_suggestion(
                        f"{node_name} 已连续失败 {fail_total} 次，请检查环境或尝试其他动作",
                        "error",
                    )
            log_entry = dict(exec_record)
            log_entry["evidence"] = self._json_safe(log_entry.get("evidence"))
            self._active_plan_log.append(log_entry)
            if not result.success:
                recovered = self._maybe_recover_bt(action_node, result, runtime, step)
                if recovered and recovered.status == "success":
                    result = recovered
                    exec_record = {
                        "node": node_name,
                        "status": result.status,
                        "reason": result.reason,
                        "evidence": result.evidence,
                    }
                    self._record_timeline_event(
                        stage="skill",
                        node=node_name,
                        status=result.status,
                        detail=detail,
                        elapsed=result.elapsed,
                    )
                    self._append_execution_turn(
                        stage="skill",
                        node=node_name,
                        status=result.status,
                        observation=observation_text,
                        action=node_name,
                        detail=result.reason or detail,
                        evidence=result.evidence,
                    )
                else:
                    self.plan_runner.apply_action_result(action_node, False)
                    self._publish_world_snapshot()
                    self._publish_plan_state()
                    continue
            self.plan_runner.apply_action_result(action_node, result.success)
            self._publish_world_snapshot()
            self._publish_plan_state()

        log_error("❌ 达到最大探索次数，未完成任务")
        self._finalize_plan_iteration("failed", "max_steps_exceeded")
        return {
            "success": False,
            "reason": "未能在限定步数内完成抓取任务",
            "timeline": self._timeline_payload(),
            "status": self._last_status_message,
            "reflection": self._latest_reflection(),
        }
