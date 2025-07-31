import json
from typing import Optional, Dict, Any
from config import Config

# ================================
# 机器人指令处理模块
# ================================

class RobotCommandProcessor:
    """机器人指令处理器"""
    
    def __init__(self, api_key: Optional[str] = "", model: str = "glm-4-airx"):
        """初始化指令处理器"""
        self.api_key = api_key or Config.ZHIPUAI_API_KEY
        self.model = model
        
        if not self.api_key:
            raise ValueError("请提供智谱AI API密钥")
            
        # if not deps.zhipuai_available:
        #     raise ImportError("请先安装智谱AI SDK: pip install zhipuai")
            
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=self.api_key)
        
        self.action_map = Config.ACTION_MAP
    
    def process_command(self, text: str) -> Dict[str, Any]:
        """处理语音识别的文本，返回机器人动作指令"""
        prompt = f"""
        请分析以下中文语音文本，判断用户的意图。

        语音文本："{text}"

        可能的意图包括：
        1. 聊天 - 普通对话内容（关键词：你好、天气、新闻、笑话等）
        2. 指令 - 控制机器人执行动作（支持的动作类型：打招呼/摆手、摇头、点头、鞠躬、其他）
        请先判断用户的意图（特别地，如果用户说的话询问机器人是否能执行动作，也请判断为对应的指令）
        如果你判断意图为聊天，请正常输出回答，intent设置为chat，action设置为unknown，confidence设置为1.0，description设置为识别到的意图或动作
        如果你判断意图为指令，请输出JSON格式，intent设置为command，action设置为识别到的动作，confidence设置为1.0，description设置为识别到的意图或动作，指令对应的动作类型只包括greet, shake_head, nod, bow,others这五种动作，前面四种就是具体的动作，而others就是除了前面四种动作之外的所有动作。另外打招呼这个动作和摆摆手这个动作是等价的，所以打招呼和摆摆手都对应greet动作类型。
        如果你判断意图为指令，输出的标准格式如下：
        {{
            "intent": "command"或"chat",
            "action": "动作类型（仅当intent为command时有效，使用英文描述，动作类型只有可能是：greet, shake_head, nod, bow, others）",
            "confidence": 0.0到1.0之间的置信度,
            "description": "意图或动作描述"
        }}

        注意：
        1. 如果意图为指令，只返回JSON，不要其他内容
        2. 如果无法识别明确意图，intent设为"chat"
        3. confidence表示识别的置信度
        4. description用中文描述识别到的意图或动作
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.05,
                response_format={"type": "json_object"}
            )
            
            # 修复linter错误：正确处理响应内容
            result_text = ""
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    result_text = message.content.strip()

            try:
                result = json.loads(result_text)
                print(f"大模型响应: {result_text}")
                
                # 优化：如果意图是聊天，直接在这里一次性生成响应
                if result.get("intent") == "chat":
                    chat_prompt = f"""
                    用户说："{text}"
                    你的名字叫小智。请生成一个自然简洁的对话响应，不用特别简短但字数不要超过100字，你生成的回答会被TTS模型念出来，所以不要使用表情，也不要以"小智同学说："这样的东西开头。回答不要以"好的"或"当然可以"这类语句开头。
                    """
                    try:
                        chat_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "user", "content": chat_prompt}
                            ],
                            max_tokens=150,
                            temperature=0.3
                        )
                        
                        if hasattr(chat_response, 'choices') and chat_response.choices:
                            chat_message = chat_response.choices[0].message
                            if hasattr(chat_message, 'content') and chat_message.content:
                                result["description"] = chat_message.content.strip()
                            else:
                                result["description"] = "无法生成聊天响应"
                        else:
                            result["description"] = "无法生成聊天响应"
                    except Exception as e:
                        print(f"生成聊天响应时出错: {e}")
                        result["description"] = "生成聊天响应时出错"
                
                return result
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                return {"intent": "chat", "action": "unknown", "confidence": 0.0, "description": "解析失败"}
                
        except Exception as e:
            print(f"大模型处理错误: {e}")
            return {"intent": "chat", "action": "unknown", "confidence": 0.0, "description": "处理失败"}

    def generate_chat_response(self, user_input: str) -> str:
        """生成聊天响应"""
        # if not deps.zhipuai_available:
        #     return "智谱AI库未安装，无法生成聊天响应"
        
        prompt = f"用户说：{user_input}\n请生成一个自然简洁的对话响应（不超过100字）。"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    return message.content.strip()
            return "无法生成响应"
        except Exception as e:
            print(f"生成聊天响应时出错: {e}")
            return "生成聊天响应时出错"