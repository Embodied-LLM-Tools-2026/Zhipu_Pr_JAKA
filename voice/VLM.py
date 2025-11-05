import json
import os
import sys
import time
import math
from typing import Optional, Dict, Any, List, Tuple, Callable
import requests
from PIL import Image, ImageDraw
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))

from config import Config
from upload_image import upload_file_and_get_url  # type: ignore
from task_logger import log_info, log_success, log_warning, log_error  # type: ignore
from openai import OpenAI
import dashscope

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
            
        # if not deps.zhipuai_available:
        #     raise ImportError("请先安装智谱AI SDK: pip install zhipuai")
            
        from zhipuai import ZhipuAI
        self.llm_client = ZhipuAI(api_key=self.llm_api_key)
        
        self.action_map = Config.ACTION_MAP
        self.drink_list = Config.drink_list


    def needs_web_search(self, text: str) -> bool:
        """判断用户输入是否需要联网搜索"""
        search_keywords = [
            "天气", "新闻", "股票", "价格", "汇率", "时间", "日期", "今天", "明天", "昨天",
            "现在", "实时", "最新", "当前", "今天天气", "明天天气", "天气预报",
            "股市", "基金", "黄金", "油价", "房价", "机票", "火车票", "电影票",
            "演唱会", "比赛", "比分", "赛程", "排名", "热搜", "热门", "趋势"
        ]
        
        # 检查是否包含搜索关键词
        for keyword in search_keywords:
            if keyword in text:
                return True
        
        # 检查是否是时间相关的问题
        time_patterns = [
            "几点", "什么时候", "多久", "多长时间", "什么时候开始", "什么时候结束"
        ]
        for pattern in time_patterns:
            if pattern in text:
                return True 
                
        return False
    
    def web_search(self, query: str) -> str:
        """执行联网搜索"""
        try:
            # 使用智谱AI的web_search工具
            response = self.llm_client.web_search.web_search(
                search_engine="search_std",
                search_query=query,
                count=1,  # 返回结果的条数，范围1-50，默认10
                search_domain_filter="www.sohu.com",  # 只访问指定域名的内容
                search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
                content_size="medium"  # 控制网页摘要的字数，默认medium
            )
            return response.search_result[0].content
            
        except Exception as e:
            print(f"联网搜索出错: {e}")
            return f"搜索时出现错误: {str(e)}"

    def capture_image(self, cam_name: str) -> str:
        import requests
        # 调用UI的capture接口，获取图片url
        try:
            resp = requests.get(f"http://127.0.0.1:8000/api/capture?cam={cam_name}&w=960&h=540", timeout=3)
            data = resp.json()
            image_path = data["url"]
            image_url = upload_file_and_get_url(
                api_key=self.vlm_api_key,model_name=self.vlm_model,file_path=image_path)
        except Exception as e:
            print(f"获取图片失败: {e}")
            image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"  # 兜底
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
                    {"role": "user",
                     "content": [
                         {"image": image_url},
                         {"text": prompt},
                     ]}
                ],
                max_tokens=200,
                temperature=0.01,
                response_format={"type": "json_object"}
            )
            # 兼容dashscope返回内容为list[dict]、dict或str
            content = response.output.choices[0].message.content
            result_text = None
            if isinstance(content, list):
                # 可能是 [{'text': '{...json...}'}]
                if content and isinstance(content[0], dict) and "text" in content[0]:
                    result_text = content[0]["text"]
                else:
                    result_text = str(content)
            elif isinstance(content, dict):
                # 可能直接是 {'text': '{...json...}'}
                if "text" in content:
                    result_text = content["text"]
                else:
                    result_text = str(content)
            else:
                # 可能直接是字符串
                result_text = str(content)
            decition_end = time.time()
            print(f"大模型响应原始内容: {result_text}")
            decition_duration = decition_end - decition_start
            log_info(f"🤖 大模型决策耗时: {decition_duration:.2f} 秒")
            try:
                # 如果已经是dict则直接用，否则json.loads
                if isinstance(result_text, dict):
                    result = result_text
                else:
                    result = json.loads(result_text)
                print(f"大模型响应: {result}")
                # 优化：如果意图是聊天，直接在这里一次性生成响应
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
                                {"role": "user",
                                 "content": [
                                     {"image": image_url},
                                     {"text": chat_prompt},
                                 ]}
                            ],
                            max_tokens=250,
                            temperature=0.01
                        )
                        chat_content = chat_response.output.choices[0].message.content
                        chat_result_text = None
                        if isinstance(chat_content, list):
                            if chat_content and isinstance(chat_content[0], dict) and "text" in chat_content[0]:
                                chat_result_text = chat_content[0]["text"]
                            else:
                                chat_result_text = str(chat_content)
                        elif isinstance(chat_content, dict):
                            if "text" in chat_content:
                                chat_result_text = chat_content["text"]
                            else:
                                chat_result_text = str(chat_content)
                        else:
                            chat_result_text = str(chat_content)
                        # 这里chat_result_text理论上是字符串
                        chat_end = time.time()
                        chat_duration = chat_end - chat_start
                        log_info(f"🤖 大模型聊天响应耗时: {chat_duration:.2f} 秒")
                        if isinstance(chat_result_text, dict):
                            result = chat_result_text
                        else:
                            # 优雅处理空字符串或非JSON内容
                            chat_result_text_str = (chat_result_text or '').strip()
                            if chat_result_text_str.startswith('{') and chat_result_text_str.endswith('}'):
                                try:
                                    result = json.loads(chat_result_text_str)
                                except Exception as e2:
                                    print(f"chat响应JSON解析失败: {e2}, 内容: {chat_result_text_str}")
                                    result["description"] = "生成聊天响应时内容格式错误"
                            else:
                                # 直接作为description返回
                                result["description"] = chat_result_text_str or "生成聊天响应时内容为空"
                    except Exception as e:
                        print(f"生成聊天响应时出错: {e}")
                        result["description"] = "生成聊天响应时出错"
                return result
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                return {"intent": "chat", "action": "unknown", "confidence": 0, "description": "解析失败"}
        except Exception as e:
            print(f"大模型处理错误: {e}")
            return {"intent": "chat", "action": "unknown", "confidence": 0, "description": "处理失败"}

class TaskProcessor:
    """任务处理器"""
    
    def __init__(self, navigator=None):
        self.navigator = navigator
        self.vlm_api_key = os.getenv("Zhipu_real_demo_API_KEY")
        self.vlm_model = Config.VLM_NAME
        self.target_resolution = (1000, 1000)
        self.target_distance = 0.5  # 目标距离，单位米
        self.moved_to_center = False
        self.action_registry: Dict[str, Callable[..., bool]] = {}
        self._register_default_actions()
        # 阈值与策略配置
        self.search_distance_threshold = 3.0  # 阈值（米）：超过则继续探索靠近
        self.default_forward_step = 0.4       # 默认前进步长（米）
        self.max_forward_attempts = 8         # 最大前进次数，防止撞击
        self.max_total_rotation_deg = 720.0   # 最大搜索旋转角度
        # ========================================
        # 摄像头标定参数（2K 80° 摄像头）
        # ========================================
        # 原始分辨率（2560×1440）的焦距
        self.camera_fx_original = 1525
        # 缩放到 1000×1000 后的焦距参数
        self.camera_fx = 596   # 水平焦距（像素，1000×1000 分辨率）
        self.camera_fy = 596   # 竖直焦距（像素，1000×1000 分辨率）
        self.camera_cx = 500   # 主点 x 坐标（1000×1000 图像中心）
        self.camera_cy = 500   # 主点 y 坐标（1000×1000 图像中心）
        # 视场角参数
        self.camera_fov_h_deg = 80.0  # 水平视场角（度）
        self.camera_fov_h_rad = math.radians(self.camera_fov_h_deg)  # 转换为弧度

    def _register_default_actions(self):
        """注册默认的原子动作函数，供VLM按需组合调用。"""
        self.action_registry = {
            "rotate_chassis": self._action_rotate_chassis,
            "move_target_to_center": self._action_move_target_to_center,
            "move_forward": self._action_move_forward,
            "finalize_target_pose": self._action_finalize_target_pose,
            "refine_surface_alignment": self._action_refine_surface_alignment,
            "approach_via_plane": self._action_approach_via_plane,
        }

    def register_action(self, name: str, handler: Callable[..., bool]) -> None:
        """允许外部扩展动作集合。"""
        self.action_registry[name] = handler

    def execute_action(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        navigator=None,
    ) -> bool:
        """执行单个动作，失败时返回False并记录日志。"""
        navigator = navigator or self.navigator
        name = action.get("name")
        if not name:
            log_warning("⚠️ 动作缺少 name 字段，已跳过")
            return False

        handler = self.action_registry.get(name)
        if handler is None:
            log_warning(f"⚠️ 未注册的动作: {name}")
            return False

        params = action.get("params") or {}
        try:
            success = bool(handler(context=context, navigator=navigator, **params))
            if success:
                log_success(f"✅ 动作执行完成: {name}")
            else:
                log_warning(f"⚠️ 动作执行未成功: {name}")
            return success
        except TypeError as type_err:
            log_error(f"❌ 动作参数错误 {name}: {type_err}")
            return False
        except Exception as exc:
            log_error(f"❌ 执行动作 {name} 异常: {exc}")
            return False

    def _action_rotate_chassis(
        self,
        *,
        angle_deg: float = 30.0,
        context: Dict[str, Any],
        navigator=None,
    ) -> bool:
        """调用底盘旋转动作，默认每次旋转30度。"""
        navigator = navigator or self.navigator
        if navigator is None:
            log_error("❌ rotate_chassis 动作失败：缺少导航控制器")
            return False
        try:
            turn_angle = math.radians(float(angle_deg))
        except (TypeError, ValueError):
            log_warning(f"⚠️ 旋转角度非法: {angle_deg}")
            return False
        return self.control_turn_around(navigator, turn_angle)

    def _action_move_target_to_center(
        self,
        *,
        context: Dict[str, Any],
        navigator=None,
        bbox: Optional[List[float]] = None,
        image_size: Optional[List[int]] = None,
        tolerance_px: Optional[float] = None,
    ) -> bool:
        """根据当前检测结果将目标移至视野中央。"""
        navigator = navigator or self.navigator
        bbox = bbox or context.get("bbox")
        image_size = image_size or context.get("image_size")
        if not bbox or not image_size:
            log_warning("⚠️ move_target_to_center 需要 bbox 和 image_size")
            return False
        return self.control_chassis_to_center(
            bbox,
            image_size,
            navigator=navigator,
            tolerance_px=tolerance_px,
        )

    def _action_move_forward(
        self,
        *,
        context: Dict[str, Any],
        navigator=None,
        distance: Optional[float] = None,
    ) -> bool:
        """控制底盘直线前进或后退。"""
        navigator = navigator or self.navigator
        if distance is None:
            vlm_result = context.get("vlm_result") or {}
            distance = vlm_result.get("forward_distance", 0.0)
        try:
            distance_val = float(distance)
        except (TypeError, ValueError):
            log_warning(f"⚠️ move_forward 距离参数非法: {distance}")
            return False
        return self.control_chassis_forward(distance_val, navigator=navigator)

    def _action_finalize_target_pose(
        self,
        *,
        context: Dict[str, Any],
        navigator=None,
    ) -> bool:
        """根据前端返回的三维坐标调整底盘最终位姿。"""
        navigator = navigator or self.navigator
        if navigator is None:
            log_error("❌ finalize_target_pose 动作失败：缺少导航控制器")
            return False

        frontend_response = context.get("frontend_response")
        if frontend_response is None:
            log_warning("⚠️ finalize_target_pose 缺少三维识别结果，尝试重新推送检测信息")
            frontend_response = self._push_detection_to_frontend(context)
            if frontend_response is None:
                return False
            context["frontend_response"] = frontend_response

        response_data = frontend_response.get("data") if isinstance(frontend_response, dict) else {}
        if not response_data:
            log_error("❌ 前端返回数据为空，无法完成最终对位")
            return False

        try:
            cam_obj_center_3d = response_data.get("obj_center_3d", [0.0, 0.0, 0.0])
            tune_angle = response_data.get("tune_angle", 0.0)
            vec = np.array([cam_obj_center_3d.copy() + [1.0]])
            T_mat = np.array(
                [
                    [0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, 180, -50],
                ]
            )

            jaka_obj_center_3d = vec @ T_mat
            jaka_obj_center_3d_list = jaka_obj_center_3d.tolist()

            pose = navigator.get_current_pose()
            X_OA = pose["x"] * 1000
            Y_OA = pose["y"] * 1000
            theta_OA = pose["theta"]
            X_AB, Y_AB, Z_AB = jaka_obj_center_3d.ravel()
            X_OB = X_OA + (X_AB * math.cos(theta_OA) - Y_AB * math.sin(theta_OA))
            Y_OB = Y_OA + (X_AB * math.sin(theta_OA) + Y_AB * math.cos(theta_OA))
            target_theta = theta_OA + tune_angle
            target_x = (X_OB - self.target_distance * 1000 * math.cos(target_theta)) / 1000
            target_y = (Y_OB - self.target_distance * 1000 * math.sin(target_theta)) / 1000

            log_info(
                f"🎯 计算底盘目标位置: ({target_x:.2f}, {target_y:.2f}), 目标朝向 θ={target_theta:.2f}rad"
            )
            log_info(f"🎯 目标物体 3D 中心点(jaka坐标): {jaka_obj_center_3d_list}")
            log_info(f"🎯 垂直目标需要转向 : {tune_angle * 180 / np.pi} 度")

            success = navigator.move_to_position(target_theta, target_x, target_y)
            if success:
                context["task_completed"] = True
                return True
            return False
        except Exception as exc:
            log_error(f"❌ finalize_target_pose 计算失败: {exc}")
            return False

    def _push_detection_to_frontend(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将检测结果推送给前端，并返回响应数据。"""
        import requests

        payload = context.get("frontend_payload")
        if payload is None:
            log_warning("⚠️ _push_detection_to_frontend 缺少 payload")
            return None
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/vlm/result",
                json=payload,
                timeout=3,
            )
            data = response.json()
            edge_conf = data.get("edge_confidence")
            if edge_conf is not None:
                log_info(f"🧭 桌面/柜面夹角置信度: {edge_conf:.2f}")
            zerograsp_info = data.get("zerograsp")
            if zerograsp_info:
                summary = zerograsp_info
                if isinstance(zerograsp_info, dict):
                    summary = {
                        key: zerograsp_info[key]
                        for key in list(zerograsp_info.keys())[:5]
                    }
                log_info(f"🖐️ ZeroGrasp 输出: {summary}")
            result = {"payload": payload, "data": data}
            surface_pts = data.get("surface_points")
            if surface_pts:
                log_info(f"🖼️ 前端平面点: {surface_pts}")
            log_success("✓ 标注结果已推送前端")
            return result
        except Exception as exc:
            log_error(f"❌ 推送VLM结果至前端失败: {exc}")
            return None

    def _action_refine_surface_alignment(
        self,
        *,
        context: Dict[str, Any],
        navigator=None,
        mask_bbox: Optional[List[int]] = None,
    ) -> bool:
        frontend_response = context.get("frontend_response")
        if not frontend_response:
            frontend_response = self._push_detection_to_frontend(context)
            context["frontend_response"] = frontend_response
        if not frontend_response:
            return False
        data = frontend_response.get("data", {})
        zerograsp = data.get("zerograsp", {}) if isinstance(data, dict) else {}
        surface = zerograsp.get("surface_region")
        if isinstance(surface, list) and len(surface) == 4:
            try:
                surface = [int(float(v)) for v in surface]
            except (TypeError, ValueError):
                surface = None
        if (not surface) and mask_bbox:
            try:
                surface = [int(float(v)) for v in mask_bbox]
            except (TypeError, ValueError):
                surface = None
        surface_points = data.get("surface_points")
        if isinstance(surface_points, list):
            context["surface_points"] = surface_points
        if surface:
            context["surface_region"] = surface
            log_info(f"🪑 更新背景平面区域: {surface}")
        elif surface_points:
            log_info(f"🪑 更新背景引导点: {surface_points}")
        else:
            log_warning("⚠️ refine_surface_alignment 缺少平面引导信息")
            return False
        return True

    def _action_approach_via_plane(
        self,
        *,
        context: Dict[str, Any],
        navigator=None,
        tolerance_deg: float = 8.0,
    ) -> bool:
        navigator = navigator or self.navigator
        if navigator is None:
            log_error("❌ approach_via_plane 失败: 无导航器")
            return False
        frontend_response = context.get("frontend_response")
        if not frontend_response:
            return False
        data = frontend_response.get("data", {})
        if not isinstance(data, dict):
            return False
        tune_angle = data.get("tune_angle")
        edge_conf = data.get("edge_confidence", 0.0) or 0.0
        if tune_angle is None:
            log_warning("⚠️ 无法获取 tune_angle")
            return False
        angle_deg = math.degrees(float(tune_angle))
        if abs(angle_deg) < tolerance_deg:
            log_info(f"🪑 平面已对齐(误差 {angle_deg:.2f}°)，跳过调整")
            return True
        log_info(
            f"🪑 依据平面角度调整底盘: angle={angle_deg:.2f}°, confidence={edge_conf:.2f}"
        )
        try:
            pose = navigator.get_current_pose()
            new_theta = pose["theta"] + float(tune_angle)
            success = navigator.move_to_position(new_theta, pose["x"], pose["y"])
            if success:
                log_success("🪑 平面对齐完成")
            else:
                log_warning("⚠️ 平面对齐动作执行失败")
            return success
        except Exception as exc:
            log_error(f"❌ 平面对齐异常: {exc}")
            return False

    def capture_image(self, cam_name: str = "front") -> str:
        import requests
        try:
            resp = requests.get(f"http://127.0.0.1:8000/api/capture?cam={cam_name}", timeout=3)
            data = resp.json()
            image_path = data["url"]
            return image_path
        except Exception as e:
            print(f"采集图片失败: {e}")
            return ""

    def _prepare_image_for_vlm(self, image_path: str) -> Dict[str, Any]:
        """确保上传到VLM的图片满足指定分辨率，并返回处理信息。"""
        info: Dict[str, Any] = {
            "path": image_path,
            "original_size": [0, 0],
            "processed_size": [0, 0],
            "resized": False,
        }

        if not os.path.isfile(image_path):
            print(f"[处理] 图片路径不存在: {image_path}")
            return info

        try:
            with Image.open(image_path) as img:
                original_size = list(img.size)
                info["original_size"] = original_size

                target = self.target_resolution
                if target and tuple(original_size) != target:
                    resized_img = img.resize(target, Image.BILINEAR)
                    base_dir = os.path.dirname(image_path)
                    ts = int(time.time() * 1000)
                    new_name = f"vlm_{ts}_{target[0]}x{target[1]}.jpg"
                    processed_path = os.path.join(base_dir, new_name)
                    resized_img.save(processed_path, format="JPEG", quality=90)
                    info.update({
                        "path": processed_path,
                        "processed_size": [target[0], target[1]],
                        "resized": True,
                    })
                    print(f"[处理] 图片尺寸 {original_size} -> {info['processed_size']}")
                else:
                    info["processed_size"] = original_size
        except Exception as err:
            print(f"[处理] 图片尺寸调整失败: {err}")

        if info["processed_size"] == [0, 0]:
            try:
                with Image.open(info["path"]) as fallback_img:
                    info["processed_size"] = list(fallback_img.size)
            except Exception:
                pass

        if self.target_resolution and info["processed_size"] != list(self.target_resolution):
            print(f"[警告] 处理后的图片尺寸 {info['processed_size']} 未匹配目标 {self.target_resolution}")

        return info

    def _build_search_prompt(
        self,
        target_name: str,
        step: int,
        max_iter: int,
        last_action_feedback: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        surface_region: Optional[List[int]] = None,
        surface_points: Optional[List[List[int]]] = None,
    ) -> str:
        """构造探索阶段的VLM提示词。"""
        available_actions_lines = [
            "可调用的原子动作函数（一次最多返回两个动作，按顺序执行后会重新观测）：",
            "1. rotate_chassis(angle_deg): 控制底盘原地旋转，正角度左转，负角度右转。",
            "2. move_forward(distance): 底盘沿当前朝向前进(>0)或后退(<0)指定米数。",
            "3. move_target_to_center(tolerance_px): 将目标移至画面中心，tolerance默认50像素。",
            "4. finalize_target_pose(): 当目标已近在咫尺且抓取入口正面时调用，会触发深度定位。",
            "5. refine_surface_alignment(mask_bbox): 当需要重新拟合背景平面时调用，可提供背景区域bbox。",
            "6. approach_via_plane(tolerance_deg): 根据平面角度微调朝向，默认容差8°。",
        ]
        last_actions_str = json.dumps(last_action_feedback, ensure_ascii=False)
        prompt_parts = [
            f"你是一个服务机器人，任务是抓取“{target_name}”。当前处于探索阶段。",
            f"当前是第 {step} 步（最多 {max_iter} 步）。上一步动作反馈: {last_actions_str}",
            f"图片分辨率: {image_size[0]}x{image_size[1]}。",
            f"当前背景平面参考区域: {surface_region}" if surface_region else "当前暂无背景平面参考区域。",
            f"已知背景参考点: {surface_points}" if surface_points else "如需我定位背景平面，请提供一个或多个 surface_points。",
            "请分析图像目标情况并估计目标与机器人的距离range_estimate（米，允许近似）。",
            f"当 range_estimate > {self.search_distance_threshold}m 时，请优先调用 move_target_to_center 让目标居中，再通过 move_forward 小步靠近。",
            f"当 range_estimate ≤ {self.search_distance_threshold}m 时，请专注于提供 surface_points / surface_roi 以便我精准拟合背景平面。",
            *available_actions_lines,
            "返回严格的JSON：",
            "{",
            '  "found": true/false,',
            '  "bbox": [x_min, y_max, x_max, y_min],',
            '  "image_size": [width, height],',
            '  "range_estimate": number,  // 估计距离（米），若无法估计给出-1',
            '  "confidence": number,      // 0-1',
            '  "analysis": "<简短中文分析>",',
            '  "surface_roi": [x_min, y_min, x_max, y_max], // 可选，背景平面区域',
            '  "surface_points": [[x, y], ...], // 可选，背景平面上的像素点',
            '  "actions": [',
            '    {"name": "<动作名>", "params": {...}, "reason": "<原因描述>"}',
            "  ]",
            "}",
            "如果未找到目标，请至少给出一个 rotate_chassis 动作；动作数组最多两个。",
        ]
        return "\n".join(prompt_parts)

    def _build_approach_prompt(
        self,
        target_name: str,
        step: int,
        max_iter: int,
        last_action_feedback: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        surface_region: Optional[List[int]] = None,
        surface_points: Optional[List[List[int]]] = None,
    ) -> str:
        """构造靠近阶段的VLM提示词。"""
        available_actions_lines = [
            "可调用的原子动作函数：",
            "1. move_target_to_center(tolerance_px): 将目标调整到画面中心，默认容差50像素。",
            "2. move_forward(distance): 如果仍需微小前进，可设置0.1-0.4米的小步。",
            "3. finalize_target_pose(): 当姿态合适时调用，触发精确定位并结束探索。",
            "4. refine_surface_alignment(mask_bbox): 如果需要重新拟合目标背后的平面，可给出平面区域。",
            "5. approach_via_plane(tolerance_deg): 根据平面角度精调朝向，默认容差8°。",
        ]
        last_actions_str = json.dumps(last_action_feedback, ensure_ascii=False)
        prompt_parts = [
            f"你是一个服务机器人，任务是抓取“{target_name}”。当前已接近目标，处于精对位阶段。",
            f"当前是第 {step} 步（最多 {max_iter} 步）。上一步动作反馈: {last_actions_str}",
            f"图片分辨率: {image_size[0]}x{image_size[1]}。",
            f"当前背景平面参考区域: {surface_region}" if surface_region else "当前暂无背景平面参考区域。",
            f"已知背景参考点: {surface_points}" if surface_points else "如需更精准平面拟合，请在 surface_points 字段提供一个或多个背景平面像素点（例如[[x1,y1]]）。",
            "请分析目标是否居中、距离是否适合抓取，并估计range_estimate（米）。",
            "此阶段请避免继续 move_target_to_center；可以结合 approach_via_plane / move_forward 等动作配合平面角度微调。",
            *available_actions_lines,
            "返回严格JSON：",
            "{",
            '  "found": true/false,',
            '  "bbox": [x_min, y_max, x_max, y_min],',
            '  "image_size": [width, height],',
            '  "range_estimate": number,',
            '  "confidence": number,',
            '  "analysis": "<简短中文分析>",',
            '  "surface_roi": [x_min, y_min, x_max, y_max],',
            '  "surface_points": [[x, y], ...],',
            '  "actions": [',
            '    {"name": "<动作名>", "params": {...}, "reason": "<原因描述>"}',
            "  ]",
            "}",
            "动作最多两个，请按执行顺序给出。",
        ]
        return "\n".join(prompt_parts)

    def control_chassis_to_center(
        self,
        bbox,
        image_size,
        navigator=None,
        tolerance_px: Optional[float] = None,
    ):
        """
        控制底盘将目标移至视野中心
        
        使用摄像头标定参数计算从像素偏差到转向角度的转换
        
        参数:
            bbox: 边界框 [x1, y1, x2, y2]（左上到右下的像素坐标）
            image_size: 图像尺寸 [width, height]
            navigator: Navigate 实例（用于执行转向运动）
        
        返回:
            True: 目标已对齐或已成功对齐
            False: 对齐失败
        """
        try:
            if not bbox or len(bbox) < 4:
                print(f"⚠️ [底盘控制] 边界框无效: {bbox}")
                log_warning(f"⚠️ 底盘中心对齐: 边界框无效")
                return False
            
            if not image_size or len(image_size) < 2:
                print(f"⚠️ [底盘控制] 图像尺寸无效: {image_size}")
                return False
            
            # 获取图像中心坐标
            img_center_x = image_size[0] / 2.0
            img_center_y = image_size[1] / 2.0
            
            # 获取边界框中心坐标
            x1, y1, x2, y2 = bbox
            bbox_center_x = (x1 + x2) / 2.0
            bbox_center_y = (y1 + y2) / 2.0
            
            # 计算偏差（像素，相对于图像中心）
            dx_pixels = img_center_x - bbox_center_x
            dy_pixels = img_center_y - bbox_center_y
            
            print(f"🎯 [底盘控制] 目标中心对齐")
            print(f"   图像中心: ({img_center_x:.0f}, {img_center_y:.0f})")
            print(f"   目标中心: ({bbox_center_x:.0f}, {bbox_center_y:.0f})")
            print(f"   像素偏差: Δx={dx_pixels:.1f}px, Δy={dy_pixels:.1f}px")
            
            log_info(f"🎯 底盘中心对齐: 像素偏差({dx_pixels:.1f}, {dy_pixels:.1f})")
            
            # ================================================
            # 使用摄像头焦距计算转向角度
            # ================================================
            # 公式：θ = arctan(Δx / fx)
            # 其中 fx 是摄像头焦距（1000×1000 分辨率下）
            
            # 判断是否已对齐（容差 50 像素）
            tolerance = tolerance_px if tolerance_px is not None else 50.0
            if abs(dx_pixels) < tolerance and abs(dy_pixels) < tolerance:
                print(f"✅ [底盘控制] 目标已在视野中心，无需调整")
                log_success(f"✅ 目标已在视野中心")
                self.moved_to_center = False
                return True
            
            # 计算转向所需角度（仅水平转向）
            # 使用焦距反正切法（比 FOV 线性法更准确）
            turn_angle_rad = math.atan(dx_pixels / self.camera_fx)*0.7
            turn_angle_deg = -math.degrees(turn_angle_rad)
            
            print(f"💡 [底盘控制] 计算转向角度")
            print(f"   摄像头焦距 fx={self.camera_fx} px")
            print(f"   视场角 FOV={self.camera_fov_h_deg}°")
            print(f"   转向需求：{turn_angle_deg:.2f}° ({turn_angle_rad:.4f} rad)")
            
            log_info(f"💡 像素转角度: {dx_pixels:.1f}px → {turn_angle_deg:.2f}°")
            
            # 如果转向角很小（<0.5°），不需要转向
            min_turn_angle_deg = 0.5
            if abs(turn_angle_deg) < min_turn_angle_deg:
                print(f"✅ [底盘控制] 转向角度很小 ({turn_angle_deg:.2f}°)，无需转向")
                log_success(f"✅ 转向角度不足 {min_turn_angle_deg}°，无需调整")
                self.moved_to_center = False
                return True
            
            # 获取导航器实例
            if navigator is None:
                navigator = getattr(self, 'navigator', None)
            
            if navigator is None:
                print(f"⚠️ [底盘控制] Navigator 不可用，仅报告所需转向角度")
                log_warning(f"⚠️ Navigator 不可用，计算结果: 需转向 {turn_angle_deg:.2f}°")
                self.moved_to_center = False
                return False
            
            # 获取当前位置并执行转向
            try:
                current_pose = navigator.get_current_pose()
                current_x = current_pose['x']
                current_y = current_pose['y']
                current_theta = current_pose['theta']
                
                # 计算新朝向（加上转向角）
                new_theta = current_theta + turn_angle_rad
                
                print(f"🤖 [底盘控制] 执行转向")
                print(f"   当前朝向: {current_theta:.4f} rad")
                print(f"   转向增量: {turn_angle_rad:.4f} rad")
                print(f"   目标朝向: {new_theta:.4f} rad")
                
                log_info(f"🤖 底盘转向: θ {current_theta:.4f} → {new_theta:.4f}")
                
                # 调用底盘转向
                success = navigator.move_to_position(new_theta, current_x, current_y)
                
                if success:
                    print(f"✅ [底盘控制] 转向完成")
                    log_success(f"✅ 底盘转向成功: {turn_angle_deg:.2f}°")
                    self.moved_to_center = True
                    return True
                else:
                    print(f"❌ [底盘控制] 转向失败")
                    log_error(f"❌ 底盘转向失败")
                    self.moved_to_center = False
                    return False
                    
            except Exception as e:
                print(f"❌ [底盘控制] 获取位置或转向异常: {e}")
                log_error(f"❌ 转向执行异常: {e}")
                self.moved_to_center = False
                return False
            
        except Exception as e:
            print(f"❌ [底盘控制] 中心对齐异常: {e}")
            log_error(f"❌ 中心对齐异常: {e}")
            self.moved_to_center = False
            return False

    def control_chassis_forward(self, distance: float = 1.0, navigator=None):
        """
        底盘按给定距离直线前进
        
        坐标系说明:
        - X轴正方向：水平向右
        - Y轴正方向：竖直向上
        - theta: 朝向角（弧度），以X轴正方向为0rad，逆时针增长，范围[-π, π]
        
        前进原理:
        - 获取当前位置 (x0, y0) 和朝向 theta
        - 计算新位置: x_new = x0 + distance * cos(theta)
        -            y_new = y0 + distance * sin(theta)
        - 调用 move_to_position(theta, x_new, y_new) 移动
        
        参数:
            distance: 前进距离（米），正数表示向前，负数表示向后
            navigator: Navigate 实例（如果为None则从当前对象属性获取）
        
        返回:
            True: 前进成功
            False: 前进失败
        """
        import math
        
        # 获取 navigator 实例
        if navigator is None:
            navigator = getattr(self, 'navigator', None)
        
        if navigator is None:
            print(f"❌ [底盘控制] Navigator 实例不可用")
            log_error("❌ 底盘前进失败: Navigator 实例不可用")
            return False
        
        # 获取当前位置和朝向
        try:
            current_pose = navigator.get_current_pose()
            x0 = current_pose['x']
            y0 = current_pose['y']
            theta = current_pose['theta']
        except Exception as e:
            print(f"❌ [底盘控制] 获取当前位置失败: {e}")
            log_error(f"❌ 获取当前位置失败: {e}")
            return False
        
        
        # 计算新的目标位置
        x_new = x0 + distance * math.cos(theta)
        y_new = y0 + distance * math.sin(theta)
        
        print(f"🤖 [底盘控制] 底盘直线前进")
        print(f"   当前位置: x={x0:.2f}m, y={y0:.2f}m, θ={theta:.2f}°")
        print(f"   前进距离: {distance:.2f}m")
        print(f"   目标位置: x={x_new:.2f}m, y={y_new:.2f}m, θ={theta:.2f}°")
        
        log_info(f"🤖 底盘前进: 当前({x0:.2f}, {y0:.2f}) → 目标({x_new:.2f}, {y_new:.2f}), 距离={distance:.2f}m")
        
        # 调用 move_to_position 移动到新位置
        try:
            success = navigator.move_to_position(theta, x_new, y_new)
            if success:
                print(f"✅ [底盘控制] 底盘前进完成")
                log_success(f"✅ 底盘前进成功: 到达目标位置({x_new:.2f}, {y_new:.2f})")
            else:
                print(f"❌ [底盘控制] 底盘前进失败")
                log_error(f"❌ 底盘前进失败: 未能到达目标位置")
            return success
        except Exception as e:
            print(f"❌ [底盘控制] 底盘前进异常: {e}")
            log_error(f"❌ 底盘前进异常: {e}")
            return False

    def control_turn_around(self, navigator, turn_angle: float = 0.785):
        """
        控制底盘原地转圈探索（增量转向）
        
        参数:
            navigator: Navigate 实例
            turn_angle: 转向角度（弧度），默认30度(0.5236rad)，负数表示反向旋转
        
        返回:
            True: 转向成功
            False: 转向失败
        """
        try:
            # 获取当前位置和朝向
            pose = navigator.get_current_pose()
            x = pose['x']
            y = pose['y']
            current_theta = pose['theta']
            log_info(f"🤖 底盘原地转向探索: 当前位置 ({x:.2f}, {y:.2f}), 朝向 θ={current_theta:.2f}rad")
            # 计算新的朝向角
            new_theta_rad = current_theta + turn_angle
            
            # 将新 theta 限制在 [-π, π] 范围内
            while new_theta_rad > math.pi:
                new_theta_rad -= 2 * math.pi
            while new_theta_rad < -math.pi:
                new_theta_rad += 2 * math.pi
            

            
            print(f"🔄 [底盘控制] 原地转向探索")
            print(f"   当前位置: x={x:.2f}m, y={y:.2f}m")
            print(f"   当前朝向: θ={current_theta:.2f}°")
            print(f"   转向角度: {math.degrees(turn_angle):.1f}°")
            print(f"   新朝向: θ={new_theta_rad:.2f}rad")
            
            log_info(f"🔄 底盘原地转向: θ {current_theta:.2f}° → {math.degrees(new_theta_rad):.2f}°")
            
            # 调用 move_to_position 原地转向（x, y 不变，只改变 theta）
            success = navigator.move_to_position(new_theta_rad, x, y)
            
            if success:
                print(f"✅ [底盘控制] 原地转向完成")
                log_success(f"✅ 底盘转向成功: 新朝向 {math.degrees(new_theta_rad):.2f}°")
            else:
                print(f"❌ [底盘控制] 原地转向失败")
                log_error(f"❌ 底盘转向失败")
            
            return success
            
        except Exception as e:
            print(f"❌ [底盘控制] 原地转向异常: {e}")
            log_error(f"❌ 原地转向异常: {e}")
            return False
    
    def _path_to_static_url(self, abs_path: str) -> str:
        normalized = abs_path.replace("\\", "/")
        idx = normalized.find("/static/")
        return normalized[idx:] if idx != -1 else normalized

    def _save_annotated_image(
        self,
        image_path: str,
        boxes: List[List[int]],
        surface_region: Optional[List[int]] = None,
        surface_points: Optional[List[List[int]]] = None,
    ) -> Dict[str, str]:
        """保存带框的标注图片到原图所在目录，并可叠加平面提示。"""
        base_dir = os.path.dirname(image_path)
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            # boxes 中坐标已经是标准的 [x1, y1, x2, y2]（左上到右下），直接画即可
            for x1, y1, x2, y2 in boxes:
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
            if surface_region and len(surface_region) == 4:
                sx1, sy1, sx2, sy2 = surface_region
                draw.rectangle([(sx1, sy1), (sx2, sy2)], outline="green", width=3)
            if surface_points:
                for pt in surface_points:
                    if not pt or len(pt) < 2:
                        continue
                    px, py = pt[:2]
                    px = int(px)
                    py = int(py)
                    r = 6
                    draw.ellipse(
                        [(px - r, py - r), (px + r, py + r)],
                        outline="blue",
                        fill="blue",
                    )
            ts = int(time.time() * 1000)
            annotated_name = f"annotated_{ts}.jpg"
            annotated_path = os.path.join(base_dir, annotated_name)
            img.save(annotated_path, format="JPEG", quality=90)

        return {
            "path": annotated_path,
            "url": self._path_to_static_url(annotated_path),
        }



    def process_grasp_task(self, target_name: str, navigator, cam_name: str = "front") -> dict:
        """执行抓取任务视觉-控制-反馈循环，由VLM在探索/精对位两阶段组合原子动作。"""
        max_iter = 20
        log_info(f"🤖 开始抓取任务: 目标={target_name}")
        last_action_feedback: List[Dict[str, Any]] = []
        phase = "search"  # search 或 approach
        forward_attempts = 0
        total_rotation_deg = 0.0
        known_surface_region: Optional[List[int]] = None
        known_surface_points: Optional[List[List[int]]] = None

        def compute_forward_step(range_estimate: Optional[float]) -> float:
            if range_estimate is None or range_estimate <= 0:
                return self.default_forward_step
            gap = max(0.0, range_estimate - self.search_distance_threshold)
            step = min(max(gap, 0.2), 0.6)
            return max(0.2, step)

        for step in range(1, max_iter + 1):
            print(f"[Step {step}] 采集图片并VLM推理...")
            log_info(f"[Step {step}/{max_iter}] 采集图片...")

            image_path = self.capture_image(cam_name)
            if not image_path:
                log_error("❌ 采集图片失败")
                return {"success": False, "reason": "采集图片失败"}

            log_success("✓ 图片采集成功")

            prep_info = self._prepare_image_for_vlm(image_path)
            processed_image_path = prep_info["path"]
            original_size = prep_info["original_size"]
            actual_size = prep_info["processed_size"]

            if prep_info["resized"]:
                log_info(f"📐 图片尺寸调整: {original_size} → {actual_size}")

            if actual_size == [0, 0]:
                try:
                    with Image.open(processed_image_path) as img_check:
                        actual_size = list(img_check.size)
                except Exception as open_err:
                    log_error(f"❌ 读取图片失败: {open_err}")
                    return {"success": False, "reason": "读取图片失败"}

            prompt = (
                self._build_search_prompt(
                    target_name,
                    step,
                    max_iter,
                    last_action_feedback,
                    tuple(actual_size),
                    surface_region=known_surface_region,
                    surface_points=known_surface_points,
                )
                if phase == "search"
                else self._build_approach_prompt(
                    target_name,
                    step,
                    max_iter,
                    last_action_feedback,
                    tuple(actual_size),
                    surface_region=known_surface_region,
                    surface_points=known_surface_points,
                )
            )

            log_info("🧠 VLM推理中...")
            decision_start = time.time()
            try:
                image_url = upload_file_and_get_url(
                    api_key=self.vlm_api_key,
                    model_name=self.vlm_model,
                    file_path=processed_image_path,
                )
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
                    max_tokens=400,
                    temperature=0.01,
                    response_format={"type": "json_object"},
                )
                content = response.output.choices[0].message.content
                print(f"[VLM响应] {content}")
                decision_duration = time.time() - decision_start
                log_info(f"🧠 决策耗时: {decision_duration:.2f} 秒")
                if isinstance(content, list):
                    if content and isinstance(content[0], dict) and "text" in content[0]:
                        result_text = content[0]["text"]
                    else:
                        raise ValueError("VLM返回格式异常: 非预期的列表结构")
                elif isinstance(content, dict):
                    if "text" in content:
                        result_text = content["text"]
                    else:
                        raise ValueError("VLM返回格式异常: 字典缺少'text'")
                else:
                    result_text = str(content)

                try:
                    result = json.loads(result_text)
                except json.JSONDecodeError:
                    if isinstance(result_text, dict):
                        result = result_text
                    else:
                        raise ValueError("无法解析VLM返回内容")
            except Exception as exc:
                log_error(f"❌ VLM推理失败: {exc}")
                return {"success": False, "reason": "VLM推理失败"}

            found = bool(result.get("found", False))
            bbox = result.get("bbox", [])
            confidence = float(result.get("confidence", 0))
        range_estimate_raw = result.get("range_estimate")
        if range_estimate_raw in (None, "", "-1"):
            range_estimate = None
        else:
            try:
                range_estimate = float(range_estimate_raw)
                if range_estimate <= 0:
                    range_estimate = None
            except (TypeError, ValueError):
                range_estimate = None
        if range_estimate is None:
            try:
                fallback_distance = float(result.get("forward_distance", 0.0) or 0.0)
                if fallback_distance > 0:
                    range_estimate = fallback_distance
            except (TypeError, ValueError):
                range_estimate = None

        surface_roi_raw = result.get("surface_roi")
        surface_region_processed: Optional[List[float]] = None
        if isinstance(surface_roi_raw, list) and len(surface_roi_raw) == 4:
            try:
                surface_region_processed = [float(v) for v in surface_roi_raw]
            except (TypeError, ValueError):
                surface_region_processed = None

        surface_points_raw = result.get("surface_points")
        surface_points_processed: Optional[List[List[float]]] = None
        if isinstance(surface_points_raw, list):
            temp_points: List[List[float]] = []
            for pt in surface_points_raw:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        temp_points.append([float(pt[0]), float(pt[1])])
                    except (TypeError, ValueError):
                        continue
            if temp_points:
                surface_points_processed = temp_points

            image_size = actual_size
            result["image_size"] = image_size

            if found:
                log_success(f"✓ 目标已检测到: bbox={bbox}")
            else:
                log_warning("⚠️ 未检测到目标物品")

        context: Dict[str, Any] = {
            "step": step,
            "phase": phase,
            "target_name": target_name,
            "image_path": image_path,
            "processed_image_path": processed_image_path,
            "original_size": original_size,
            "image_size": image_size,
            "vlm_result": result,
            "bbox": bbox,
            "last_actions": last_action_feedback,
            "range_estimate": range_estimate,
            "confidence": confidence,
            "surface_region": known_surface_region,
            "surface_points": surface_points_processed,
        }

        def remap_bbox_to_original(
            bbox_local: List[float],
            processed_size: Tuple[int, int],
            original_size_local: Tuple[int, int],
        ) -> List[int]:
            if not bbox_local or len(bbox_local) != 4:
                return []

            proc_w, proc_h = processed_size
            orig_w, orig_h = original_size_local

            sx = orig_w / proc_w
            sy = orig_h / proc_h

            x_min, y_max, x_max, y_min = bbox_local
            x_min *= sx
            x_max *= sx
            y_min *= sy
            y_max *= sy

            top = y_max
            bottom = y_min

            return [int(x_min), int(top), int(x_max), int(bottom)]

        def remap_roi_to_original(
            roi_local: List[float],
            processed_size: Tuple[int, int],
            original_size_local: Tuple[int, int],
        ) -> Optional[List[int]]:
            if not roi_local or len(roi_local) != 4:
                return None
            proc_w, proc_h = processed_size
            orig_w, orig_h = original_size_local
            if proc_w == 0 or proc_h == 0:
                return None
            sx = orig_w / proc_w
            sy = orig_h / proc_h
            x_min, y_min, x_max, y_max = roi_local
            x_min *= sx
            x_max *= sx
            y_min *= sy
            y_max *= sy
            x_min = max(0, min(orig_w, x_min))
            x_max = max(0, min(orig_w, x_max))
            y_min = max(0, min(orig_h, y_min))
            y_max = max(0, min(orig_h, y_max))
            if x_min >= x_max or y_min >= y_max:
                return None
            return [int(x_min), int(y_min), int(x_max), int(y_max)]

        def convert_bbox(box_local, size_local):
            if not box_local or len(box_local) != 4:
                return None
            w, h = size_local
            x_min, y_max, x_max, y_min = box_local
            x_min = max(0, min(w, x_min))
            x_max = max(0, min(w, x_max))
            y1 = max(0, min(h, y_max))
            y2 = max(0, min(h, y_min))
            return [int(x_min), int(y1), int(x_max), int(y2)]

        boxes_for_ui: List[List[int]] = []
        mapped_bbox: List[int] = []
        annotated_url = self._path_to_static_url(image_path)
        mapped_surface_region: Optional[List[int]] = None

        if isinstance(bbox, list) and found:
            if bbox and isinstance(bbox[0], (int, float)):
                mapped_bbox = remap_bbox_to_original(bbox, actual_size, original_size)
                converted = convert_bbox(mapped_bbox, original_size)
                if converted:
                    boxes_for_ui.append(converted)
            else:
                for candidate in bbox:
                    converted = convert_bbox(candidate, image_size)
                    if converted:
                        boxes_for_ui.append(converted)

        if boxes_for_ui:
            log_info("🎨 生成标注图片...")
            try:
                annotated_info = self._save_annotated_image(
                    image_path,
                    boxes_for_ui,
                    surface_region=mapped_surface_region,
                    surface_points=mapped_surface_points,
                )
                annotated_url = annotated_info["url"]
                log_success("✓ 标注图片已生成")
            except Exception as annotate_err:
                log_warning(f"⚠️ 生成标注图片失败: {annotate_err}")
                annotated_url = self._path_to_static_url(image_path)

        if surface_region_processed:
            mapped_surface_region = remap_roi_to_original(
                surface_region_processed, actual_size, original_size
            )
            if mapped_surface_region is None:
                surface_region_processed = None
            else:
                log_info(f"🖼️ VLM 建议的背景平面区域: {mapped_surface_region}")

        def remap_points_to_original(
            points: List[List[float]],
            processed_size: Tuple[int, int],
            original_size: Tuple[int, int],
        ) -> Optional[List[List[int]]]:
            if not points:
                return None
            proc_w, proc_h = processed_size
            orig_w, orig_h = original_size
            if proc_w == 0 or proc_h == 0:
                return None
            sx = orig_w / proc_w
            sy = orig_h / proc_h
            mapped: List[List[int]] = []
            for pt in points:
                if len(pt) < 2:
                    continue
                x = int(max(0, min(orig_w - 1, pt[0] * sx)))
                y = int(max(0, min(orig_h - 1, pt[1] * sy)))
                mapped.append([x, y])
            return mapped or None

        mapped_surface_points = None
        if surface_points_processed:
            mapped_surface_points = remap_points_to_original(
                surface_points_processed, actual_size, original_size
            )

        context.update(
            {
                "boxes_for_ui": boxes_for_ui,
                "mapped_bbox": mapped_bbox,
                "annotated_url": annotated_url,
                "surface_region": mapped_surface_region
                or context.get("surface_region")
                or None,
                "surface_points": mapped_surface_points
                or context.get("surface_points"),
            }
        )

        if context.get("surface_region"):
            known_surface_region = context["surface_region"]
        if context.get("surface_points"):
            known_surface_points = context["surface_points"]
            log_info(f"🖼️ 背景引导点: {context['surface_points']}")

        payload = {
            "found": found,
            "boxes": boxes_for_ui,
            "original_bbox": bbox,
            "mapped_bbox": mapped_bbox,
            "image_size": image_size,
            "original_size": original_size,
            "range_estimate": range_estimate,
            "step": step,
            "target": target_name,
            "annotated_url": annotated_url,
            "surface_region": context.get("surface_region"),
            "surface_points": context.get("surface_points"),
        }

        context["frontend_payload"] = payload
        frontend_result = self._push_detection_to_frontend(context)
        if frontend_result:
            context["frontend_response"] = frontend_result
            response_data = frontend_result.get("data")
            if isinstance(response_data, dict):
                context["zerograsp"] = response_data.get("zerograsp")
                if context["zerograsp"]:
                    log_info("🖐️ 已获取ZeroGrasp抓取候选")

            actions_raw = result.get("actions") or []
            if not isinstance(actions_raw, list):
                actions_raw = []
            actions: List[Dict[str, Any]] = [action for action in actions_raw if isinstance(action, dict)]

        if (
            phase == "approach"
            and surface_points_processed
            and not any(a.get("name") == "refine_surface_alignment" for a in actions)
        ):
            actions.insert(
                0,
                {
                    "name": "refine_surface_alignment",
                    "params": {},
                    "reason": "根据新背景点拟合平面",
                },
            )
        elif (
            phase == "approach"
            and mapped_surface_region
            and not any(a.get("name") == "refine_surface_alignment" for a in actions)
        ):
            actions.insert(
                0,
                {
                    "name": "refine_surface_alignment",
                    "params": {"mask_bbox": mapped_surface_region},
                        "reason": "根据背景区域拟合平面",
                    },
                )

            if not found:
                phase = "search"
                actions = [{"name": "rotate_chassis", "params": {"angle_deg": 30}, "reason": "未找到目标，继续扫描"}]
            else:
                if range_estimate is not None and range_estimate <= self.search_distance_threshold:
                    phase = "approach"
                    if not any(a.get("name") == "finalize_target_pose" for a in actions):
                        actions.insert(0, {"name": "finalize_target_pose", "params": {}, "reason": "距离合适，触发精确定位"})
                else:
                    phase = "search"
                    if not any(a.get("name") == "move_forward" for a in actions):
                        step_distance = compute_forward_step(range_estimate)
                        actions.insert(0, {"name": "move_forward", "params": {"distance": step_distance}, "reason": "继续靠近目标"})

                if phase == "search":
                    if not any(a.get("name") == "move_target_to_center" for a in actions) and mapped_bbox:
                        img_w, img_h = image_size
                        bbox_center_x = (mapped_bbox[0] + mapped_bbox[2]) / 2
                        offset_px = abs(bbox_center_x - img_w / 2)
                        if offset_px > 80:
                            actions.insert(
                                0,
                                {
                                    "name": "move_target_to_center",
                                    "params": {"tolerance_px": 40},
                                    "reason": "目标偏离中心",
                                },
                            )
                else:
                    actions = [
                        act
                        for act in actions
                        if act.get("name") != "move_target_to_center"
                    ]

            if not actions:
                if phase == "approach":
                    actions = [
                        {
                            "name": "approach_via_plane",
                            "params": {},
                            "reason": "默认根据平面角度精调",
                        }
                    ]
                else:
                    actions = [
                        {
                            "name": "rotate_chassis",
                            "params": {"angle_deg": 25},
                            "reason": "兜底搜索",
                        }
                    ]

            current_feedback: List[Dict[str, Any]] = []
            for action in actions[:2]:
                name = action.get("name")
                params = action.get("params") or {}

                if name == "move_forward" and "distance" not in params:
                    params["distance"] = compute_forward_step(range_estimate)
                action["params"] = params

                success = self.execute_action(action, context=context, navigator=navigator)
                feedback = {
                    "name": name,
                    "success": success,
                    "params": params,
                    "reason": action.get("reason", ""),
                }
                current_feedback.append(feedback)

                if name == "move_forward" and success:
                    forward_attempts += 1
                    if forward_attempts > self.max_forward_attempts:
                        log_error("❌ 前进次数超出安全阈值，终止任务")
                        return {"success": False, "reason": "前进次数过多，任务终止"}
                if name == "rotate_chassis" and success:
                    angle = abs(params.get("angle_deg", 0))
                    try:
                        angle = float(angle)
                    except (TypeError, ValueError):
                        angle = 0.0
                    total_rotation_deg += angle
                    if total_rotation_deg > self.max_total_rotation_deg:
                        log_error("❌ 旋转角度累计过大，终止任务")
                        return {"success": False, "reason": "旋转过多未找到目标"}

                if name == "finalize_target_pose" and success:
                    log_success("🎯 底盘定位完成，任务结束")
                    return {
                        "success": True,
                        "result": result,
                        "annotated_url": annotated_url,
                    }

            last_action_feedback = current_feedback

        log_error("❌ 达到最大探索次数，未完成任务")
        return {"success": False, "reason": "未能在限定步数内完成抓取任务"}
