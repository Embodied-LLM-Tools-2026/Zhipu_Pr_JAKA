import os
import platform
import subprocess
import warnings

# ================================
# 配置和常量管理
# ================================


class Config:
    """统一配置管理"""

    # 环境变量配置
    ZAI_API_KEY = os.getenv("ZAI_API_KEY")
    # VLM_NAME = "qwen3-vl-flash"
    VLM_NAME = "qwen3-vl-plus"
    ZERO_GRASP_CAMERA_CFG = os.getenv("ZEROGRASP_CAMERA_CFG")
    ZERO_GRASP_WS_URL = os.getenv("ZEROGRASP_WS_URL")
    # 机器人默认配置
    ROBOT_IP_LEFT = "192.168.1.9"
    ROBOT_IP_RIGHT = "192.168.1.10"

    # 音频配置
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK_SIZE = 1024

    # 文本转语音配置
    TTS_ENGINE = "edge_tts"  # "kokoro" or "edge_tts"

    # 缓存配置
    ENABLE_CACHE = True
    PRELOAD_COMMON_AUDIO = True

    # 唤醒词
    WAKE_WORDS = [ "节卡","家卡","加卡","加咖","佳卡","佳咖","嘉卡","嘉咖"]

    # 自我介绍关键词
    INTRO_KEYWORDS = [
        "介绍一下你自己",
        "你是谁",
        "自我介绍",
        "介绍你自己",
        "你能做什么",
        "你的功能",
        "你的作用",
        "你是做什么的",
    ]

    # 常用音频短语
    COMMON_PHRASES = [
        "你好，我是家卡同学，很高兴见到你！",
        "好的，我去休息了。需要时请叫我！",
        "抱歉，我不理解这个指令，请重新说一遍",
        # 新增自我介绍
        "我是人工智能助手，专为提供信息、解答疑问和协助解决问题而设计。我可以处理各种查询，并尽力提供准确、有用的回答。",
        # 快速响应短语
        "我在听，请说",
        "嗯，我在",
        "请说",
        "好的，请继续",
        # 未知动作指令
        "这个动作我还不会，不过我会抓紧学习的",
        # 未知饮料指令
        "我们这里没有这种饮料",
        # 饮料不够
        "不好意思，饮料不够了",
        # 检查饮料
        "好的，我去看看饮料还够不够",
        # 饮料充足
        "饮料还够，我这就拿给您，请您稍等",
        # 拿完一瓶饮料
        "这是您要的饮料",
        # 准备拿下一瓶
        "下一瓶我这就去拿",
        # 请您告诉我具体需要哪种饮料
        "请您告诉我具体需要哪种饮料",
        # 用户要多杯咖啡
        "抱歉，一次只能做一杯美式咖啡哦",
        # 用户要其它咖啡
        "抱歉，只能做美式咖啡哦",
        # 用户要一杯美式咖啡
        "好的，我这就去为您做咖啡",
        # 按下咖啡机按钮
        "咖啡正在制作中，请您稍等片刻",
        # 把咖啡放到传送带上后
        "请享用您的咖啡",
    ]

    # 动作映射
    ACTION_MAP = {
        "greet": "打招呼",
        "shake_head": "摇头",
        "nod": "点头",
        "bow": "鞠躬",
        "get_drink": "拿饮料",
        # "others": "其他",
    }

    # 饮料
    drink_list = ["可乐", "雪碧", "水", "奶茶", "冰红茶","绿茶","美式咖啡"]
    # drink_layer_mapping = {
    #     "水": [2,28,-4],
    #     "可乐": [3,0,100],
    #     "奶茶": [4,0,250],
    #     "雪碧": [5,0,450]
    # }

    # 实机还是模拟
    ROBOT_AVAILABLE = True  # 调试时要模拟执行动作就改为False

    @classmethod
    def setup_environment(cls):
        """设置环境变量，抑制ALSA错误"""
        if platform.system() == "Linux":
            env_vars = {
                "ALSA_PCM_CARD": "default",
                "ALSA_PCM_DEVICE": "0",
                "ALSA_CARD": "0",
                "ALSA_LOG_LEVEL": "0",
                "ALSA_SILENCE": "1",
                "ALSA_PCM_PLUGINS": "",
                "PULSE_LOG": "0",
            }

            for key, value in env_vars.items():
                os.environ[key] = value

            # 抑制系统警告
            warnings.filterwarnings("ignore")

            # 设置内核日志级别
            try:
                subprocess.run(
                    ["sysctl", "-w", "kernel.printk=1 1 1 1"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except:
                pass
        else:
            # For non-Linux systems, only suppress warnings
            warnings.filterwarnings("ignore")
