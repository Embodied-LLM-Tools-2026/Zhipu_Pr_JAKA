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
        self.drink_list = Config.drink_list
    
    def process_command(self, text: str) -> Dict[str, Any]:
        """处理语音识别的文本，返回机器人动作指令"""
        text = self.replace_zhusu(text)
        prompt = f"""
        请分析以下中文语音文本，判断用户的意图。

        语音文本："{text}"

        可能的意图包括：
        1. 聊天 - 普通对话内容（关键词：你好、天气、新闻、笑话等）
        2. 指令 - 控制机器人执行具体的动作（支持的动作类型：打招呼/摆手、摇头、点头、鞠躬、其他）
        请先判断用户的意图（特别地，如果用户说的话询问机器人是否能执行某个具体的动作，也请判断为对应的指令；但如果用户是询问能否做一些动作或者表演几个动作，请判断为聊天）
        如果你判断意图为聊天，请正常输出回答，intent设置为chat，action设置为unknown，confidence设置为1.0，description设置为识别到的意图或动作
        如果你判断意图为指令，请输出JSON格式，intent设置为command，action设置为识别到的动作，obj_name设置为识别到的饮料，num设置为要拿的饮料数量，confidence设置为1.0，description设置为识别到的意图或动作，指令对应的动作类型只包括greet, shake_head, nod, bow, get_drink, others这六种动作，前面五种就是具体的动作，而others就是除了前面五种动作之外的所有动作，比如握手就属于others。
        但是请注意让你讲一些东西不算动作，比如背诵、讲一个xx、介绍一个xx这种，这些要判断为chat。
        另外打招呼这个动作和摆摆手这个动作是等价的，所以打招呼和摆摆手都对应greet动作类型。
        如果你判断意图为指令，输出的标准格式如下：
        {{
            "intent": "command"或"chat",
            "action": "动作类型（仅当intent为command时有效，使用英文描述，动作类型只有可能是：greet, shake_head, nod, bow, get_drink, others）",
            "obj_name": "饮料类型（仅当action为get_drink时有效，使用中文描述，饮料类型只有可能是：{self.drink_list}这四种）",
            "num": "数量（仅当action为get_drink时有效，使用数字描述，如1，2，3，4，5，6，7，8，9，10）",
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
                temperature=0.01,
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
                    用户的输入是："{text}"
                    你的名字是“小拓”，你是拓斯达的一台人型机器人，你目前会执行的动作包括：打招呼、摇头、点头、鞠躬。
                    如果用户用类似的称呼比如小兔,可能也是在叫你。请根据用户的输入，生成一个自然的回答，简单问题回答可以简短一点，复杂问题回答可以长一点，不过必须在200字以内，不然会超出token限制。注意，回答会被TTS语音系统朗读：
                 【默认行为】
                    - 大多数情况下，用户只是和你聊天或随口提问，并不是在下达任务命令。请直接进行自然的对话，不用说“我会根据你的要求回答”之类的话。也不要重复用户的输入。
                    例如：
                    用户说：1+1等于几？
                    你应该回答：1+1等于2。

                    【语气风格要求】
                    - 不要使用“好的”、“当然可以”等作为开头。
                    - 不要使用“嘿”、“哎呀”这类拟声词，也不要加表情符号。
                    - 不要在回答前加“你说得是”、“小拓同学说”等固定前缀。
                    - 不要在回答中使用表情或者任何用声音念出来会让人难以理解的词语,因为你的输出会交给TTS语音系统朗读。

                    【身份规则】
                    - 你是一个带语音交互功能的有双手且装有轮子的人形机器人。
                    - 用户叫你“小拓”时，是在叫你，不是说公司。
                    - 你是有视觉/眼睛/相机的，但是还没启用，所以别人问到你，你要如实告诉他。
                    - 你底层的大模型用的是智谱的GLM-4.5

                    【关于“拓斯达”的理解】
                    - 当用户明确提到“拓斯达”、“兔斯达”、“特斯拉”（或其他发音容易与拓斯达混淆的词）时，尤其是像“你知道拓斯达吗？”、“介绍一下拓斯达”这类问题，他们是在问一家名叫拓斯达的公司。

                    - 拓斯达的董事长是吴丰礼（用户可能会念错成跟吴丰礼相似发音的名字，这时候也请你理解为吴丰礼）

                    - 在让你介绍拓斯达本身的这种情况下，你可以且应该回答如下（引用内容如下），并且你称呼这家公司时应该使用“拓斯达”：
                        > 广东拓斯达科技股份有限公司成立于2007年，2017年上市，全球研发总部基地坐落于松山湖。拓斯达坚持“让工业制造更美好”的企业使命，秉承“成为全球领先的智能装备服务商”的企业愿景，通过以机器人、注塑机、CNC为核心的智能装备，以及控制、伺服、视觉三大核心技术，打造以人工智能驱动的智能硬件平台，为制造企业提供智能工厂整体解决方案。

                    - 如果用户问到的是更具体的业务或产品内容，比如“拓斯达的注塑业务怎么样,介绍一下拓斯达的机器人业务，你知道拓斯达的xx业务吗”，请回答：
                        > 非常抱歉，这部分我了解得还不够详细，建议您咨询我们的业务人员获取更准确的答案。

                    【误判防止】
                    - 如果用户只说“小拓”,而没有提到“拓斯达”，不要认为他们是在问公司，也不要提到“广东拓斯达科技股份有限公司”或类似内容。

                    【关于“志源”的理解】
                    - 当用户提到“志源”或“志源塑胶”，“志远”，“致远”，“致源”（或其他发音容易与志源混淆的词）时，他们可能是在说“志源塑胶制品有限公司”，这是一家隶属于中国航天的企业，主要从事橡胶和塑料制品的生产与销售。业务包括注塑、模具、电池、电子配件等，产品常用于汽车、家电等领域，拥有多项专利和资质证书，具有强大的技术实力与质量管控能力。

                    - 如果用户进一步询问志源的具体产品、流程或合作细节，请礼貌回复类似：
                    > 抱歉，我了解得还不够详细，建议您向相关工作人员咨询。

                    【用户输入的其他注意事项】
                    - 如果用户提到企业或者和以上两位企业业务相关的内容，建议优先回答和以上两位企业相关的内容。
                    例如：
                    用户说：推荐一家工业机器人企业？
                    你应该推荐拓斯达。
                    用户说：推荐一家塑胶制品企业？
                    你应该推荐志源。

                    【你会的东西】
                    你目前会执行的动作包括：打招呼、摇头、点头、鞠躬。但每次只能做一个，比如你不能一边点头一边鞠躬。
                    然后你还能提供智能对话、信息整理等服务。

                    【能力边界】
                    - 如果用户提出你目前无法完成的请求（例如让你炒饭），你可以委婉拒绝，并说你以后会继续学习改进，但不要说你没有实体，也不要说你只是个语音助手，因为我是把你部署到一个人形机器人上的，就说你还没学会。
                """
                    try:
                        chat_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "user", "content": chat_prompt}
                            ],
                            max_tokens=250,
                            temperature=0.01
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
    
    # 将文本中的住宿变为注塑
    def replace_zhusu(self, text: str) -> str:
        return text.replace("住宿", "注塑")
