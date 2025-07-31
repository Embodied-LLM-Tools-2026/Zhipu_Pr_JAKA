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
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
    
    # 机器人默认配置
    ROBOT_IP_DEFAULT = "192.168.1.6"
    ROBOT_PORT_DEFAULT = 2000
    
    # 音频配置
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK_SIZE = 1024
    
    # 缓存配置
    ENABLE_CACHE = True
    PRELOAD_COMMON_AUDIO = True
    
    # 常用音频短语
    COMMON_PHRASES = [
        "你好，我是小智同学，很高兴见到你！",
        "好的，我去休息了。需要时请叫我！",
        "抱歉，我不理解这个指令，请重新说一遍",
        # 新增自我介绍
        "我是人工智能助手，专为提供信息、解答疑问和协助解决问题而设计。我可以处理各种查询，并尽力提供准确、有用的回答。",
        # 快速响应短语
        "我在听，请说",
        "嗯，我在",
        "请说",
        "好的，请继续"
    ]
    
    # 拖延语备选
    DELAY_PHRASES = [
        "嗯...让我想一想。",
        "嗯，这个问题有点意思。",
        "这是个好问题。",
        "我查一下。",
        "我想一想哦……",
    ]
    
    # 动作映射
    ACTION_MAP = {
        "waiting": "回到待机位置",
        "left_right": "左右摆动", 
        "rotate": "摇头动作",
        "coffee": "准备咖啡",
    }
    
    @classmethod
    def setup_environment(cls):
        """设置环境变量，抑制ALSA错误"""
        if platform.system() == 'Linux':
            env_vars = {
                'ALSA_PCM_CARD': 'default',
                'ALSA_PCM_DEVICE': '0',
                'ALSA_CARD': '0',
                'ALSA_LOG_LEVEL': '0',
                'ALSA_SILENCE': '1',
                'ALSA_PCM_PLUGINS': '',
                'PULSE_LOG': '0'
            }
            
            for key, value in env_vars.items():
                os.environ[key] = value
            
            # 抑制系统警告
            warnings.filterwarnings("ignore")
            
            # 设置内核日志级别
            try:
                subprocess.run(['sysctl', '-w', 'kernel.printk=1 1 1 1'], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            except:
                pass
        else:
            # For non-Linux systems, only suppress warnings
            warnings.filterwarnings("ignore")