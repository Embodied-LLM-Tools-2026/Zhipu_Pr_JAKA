import base64
import json
import os
import sys
import time
from typing import Optional, Dict, Any, List, Tuple

import dashscope
import numpy as np
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

from config import Config
from upload_image import upload_file_and_get_url  # type: ignore
from ui_state_bridge import UIStateBridge  # type: ignore
from task_logger import log_info, log_success, log_warning, log_error  # type: ignore

from .observer import VLMObserver, ObservationContext
from .executor import SkillExecutor, SkillRuntime
from .planner import BehaviorPlanner
from .world_model import WorldModel
from .task_structures import ObservationPhase, PlanNode
from .catalog_worker import SceneCatalogWorker


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

    def reset(self, root: Optional[PlanNode]) -> None:
        self.root = root
        self.pending_node = None
        self.pending_node_id = None
        self.action_results.clear()
        self.last_status = "idle"

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
            return result, None
        if node_type == "action":
            node_id = id(node)
            if node_id in self.action_results:
                result = self.action_results.pop(node_id)
                return result, None
            return "running", node
        if node_type == "sequence":
            return self._tick_sequence(node.children)
        if node_type == "selector":
            return self._tick_selector(node.children)
        if node_type == "repeat_until":
            cond = node.args.get("cond", "")
            if self.world_model and self.world_model.evaluate_condition(cond):
                return "success", None
            status, action = self._tick_sequence(node.children)
            if action is not None:
                return "running", action
            if status == "failure":
                return "failure", None
            return "running", None
        return "failure", None

    def _tick_sequence(self, children: List[PlanNode]) -> Tuple[str, Optional[PlanNode]]:
        for child in children:
            status, action = self._tick_node(child)
            if action is not None:
                return "running", action
            if status == "failure":
                return "failure", None
        return "success", None

    def _tick_selector(self, children: List[PlanNode]) -> Tuple[str, Optional[PlanNode]]:
        for child in children:
            status, action = self._tick_node(child)
            if action is not None:
                return "running", action
            if status == "success":
                return "success", None
        return "failure", None


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
        self.plan_context: List[Dict[str, Any]] = []
        self._active_plan_log: List[Dict[str, Any]] = []
        self._plan_history_limit = int(os.getenv("PLAN_CONTEXT_LIMIT", "6"))
        self._last_action_name: Optional[str] = None

    # ------------------------------------------------------------------
    def set_navigator(self, navigator) -> None:
        self.navigator = navigator
        self.executor.set_navigator(navigator)

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
        if not self.current_plan:
            self.ui_bridge.post_plan_state(root=None, steps=[], metadata={}, current_index=-1, current_node=None)
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
        self.ui_bridge.post_plan_state(
            root=root_dict,
            steps=steps_payload,
            metadata=self.current_plan.metadata,
            current_index=current_index,
            current_node=current_node,
        )

    def _ensure_plan(self, target_name: str) -> None:
        history_context = self.plan_context[-self._plan_history_limit :]
        self.current_plan = self.planner.make_plan(target_name, self.world, plan_context=history_context)
        self.plan_runner.reset(self.current_plan.root)
        self._active_plan_log = []
        self._last_action_name = None
        step_names = [node.name or node.type for node in self.current_plan.steps]
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
        observation, frontend_payload = self.observer.observe(
            target_name, phase, context, self.navigator, force_vlm=force_vlm
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
    def _append_plan_context(self, entry: Dict[str, Any]) -> None:
        entry["timestamp"] = time.time()
        self.plan_context.append(entry)
        if len(self.plan_context) > self._plan_history_limit:
            self.plan_context = self.plan_context[-self._plan_history_limit :]

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

    def _finalize_plan_iteration(self, status: str, reason: Optional[str] = None) -> None:
        if self.current_plan:
            planned_steps = [node.name or node.type for node in self.current_plan.steps]
        else:
            planned_steps = []
        summary = {
            "phase": "llm_plan",
            "status": status,
            "reason": reason,
            "planned_steps": planned_steps,
            "executed": list(self._active_plan_log),
        }
        self._append_plan_context(summary)
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
            return {"success": False, "reason": f"initial_observe_failed: {exc}"}

        for step in range(1, self.max_iter + 1):
            if not self.current_plan:
                self._ensure_plan(target_name)
                if not self.current_plan or not self.current_plan.root:
                    log_error("❌ 规划结果为空")
                    return {"success": False, "reason": "planner_return_empty"}

            action_node, bt_status = self.plan_runner.tick()
            if action_node is None:
                if bt_status == "success":
                    self._finalize_plan_iteration("completed", "plan_executed")
                    log_success("🎯 行为树计划执行完成")
                    return {
                        "success": True,
                        "result": current_observation.raw_response if current_observation else None,
                        "annotated_url": current_observation.annotated_url if current_observation else None,
                    }
                if bt_status == "failure":
                    log_warning("⚠️ 行为树失败，重新规划")
                    self._finalize_plan_iteration("failed", "bt_failure")
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
                    self.plan_runner.apply_action_result(action_node, False)
                    continue

            runtime = SkillRuntime(
                navigator=self.navigator,
                world_model=self.world,
                observation=current_observation,
                frontend_payload=current_frontend_payload,
                surface_points=current_observation.surface_points if current_observation else None,
                extra={"step": step, "node": node_name},
            )
            result = self.executor.execute(action_node, runtime)
            exec_record = {
                "node": node_name,
                "status": result.status,
                "reason": result.reason,
                "evidence": result.evidence,
            }
            self.world.record_execution_result(exec_record)
            log_entry = dict(exec_record)
            log_entry["evidence"] = self._json_safe(log_entry.get("evidence"))
            self._active_plan_log.append(log_entry)
            self.plan_runner.apply_action_result(action_node, result.success)
            self._publish_world_snapshot()
            self._publish_plan_state()
            if not result.success:
                continue

        log_error("❌ 达到最大探索次数，未完成任务")
        self._finalize_plan_iteration("failed", "max_steps_exceeded")
        return {"success": False, "reason": "未能在限定步数内完成抓取任务"}
