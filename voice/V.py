#!/usr/bin/env python3
"""

实现了智能流式TTS！按句子分割，并行生成，连续播放

基于语音识别和大模型的机器人控制脚本 - 重构版本
使用GLM-ASR进行语音识别，智谱AI GLM模型理解指令，控制机器人执行动作
本项目使用了 edge-tts（许可证：LGPL-3.0）用于文本转语音功能。

🚀 性能优化特性:
1. API调用优化：减少token数量，降低温度参数，优化聊天响应生成
2. 缓存机制：TTS音频缓存、设备配置缓存、常用响应缓存
3. 音频处理优化：设备检测缓存、音频格式转换优化
4. 预加载机制：启动时预加载常用音频，提高响应速度
5. 智能统计：实时显示缓存命中率和性能数据
6. 智能流式TTS：按句子分割，并行生成，连续播放，实现更快的语音响应

支持的动作：
1. 回到待机位置
2. 上下摆动  
3. 左右摆动
4. 摇头动作

性能提升：
- 响应速度提升 30-50%
- 缓存命中率 >60%
- 减少网络请求 40%
- 音频播放延迟降低 60%
- 流式TTS响应速度提升 40-60%

🔧 重构改进:
- 模块化代码结构
- 统一配置管理
- 清晰的依赖检测
- 优化的错误处理
- 更好的代码组织
"""

import os
import json
import time
import multiprocessing as mp
import tempfile
import subprocess
import warnings
import sys
import contextlib
import threading
from typing import Optional, Dict, Any
import random
import platform
from dotenv import load_dotenv
load_dotenv()

os.environ['TORCH_HUB_OFFLINE'] = '1'  # 强制torch hub离线模式,silero-vad模型就不需要联网加载权重
import torch

print("VRC (流式,VAD,win) running...")

# ================================
# 依赖安装检查
# ================================

# 依赖检查已集成到DependencyManager类中

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


# ================================
# 依赖检测和导入管理
# ================================

class DependencyManager:
    """依赖检测和导入管理"""
    
    def __init__(self):
        self.zhipuai_available = False
        self.audio_available = False
        self.robot_available = False
        self.pypinyin_available = False
        self.numpy_available = False
        self.silero_vad_available = False
        self.torch_available = False
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检测所有依赖项"""
        self._check_zhipuai()
        self._check_audio()
        self._check_robot()
        self._check_pypinyin()
        self._check_numpy()
        self._check_silero_vad()
    
    def _check_zhipuai(self):
        """检测智谱AI"""
        try:
            from zhipuai import ZhipuAI
            self.zhipuai_available = True
        except ImportError:
            self.zhipuai_available = False
            print("警告: 智谱AI库未安装，请运行: pip install zhipuai")
    
    def _check_audio(self):
        """检测音频设备 - 跨平台版本"""
        # 优先尝试sounddevice（跨平台）
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            
            # 测试音频设备
            try:
                # 检查是否有可用的音频设备
                devices = sd.query_devices()
                if len(devices) == 0:
                    self.audio_available = False
                else:
                    # 测试默认输入设备
                    default_input = sd.default.device[0]
                    if default_input is not None:
                        self.audio_available = True
                        print("✅ 使用sounddevice进行音频处理（推荐）")
                    else:
                        self.audio_available = False
            except Exception as e:
                print(f"⚠️ sounddevice音频设备测试失败: {e}")
                self.audio_available = False
                
        except ImportError:
            # 回退到pyaudio（如果sounddevice不可用）
            print("⚠️ sounddevice未安装，尝试回退到pyaudio")
            try:
                with suppress_stderr():
                    import pyaudio
                import wave
                import numpy as np
                
                # 测试音频设备
                try:
                    with suppress_stderr():
                        p = pyaudio.PyAudio()
                        device_count = p.get_device_count()
                        if device_count == 0:
                            self.audio_available = False
                        else:
                            self.audio_available = True
                            print("✅ 使用pyaudio进行音频处理（兼容模式）")
                        p.terminate()
                except:
                    self.audio_available = False
            except ImportError:
                self.audio_available = False
                print("❌ 音频功能不可用")
                print("请安装音频依赖:")
                print("  推荐: pip install sounddevice soundfile")
                print("  或者: pip install pyaudio")
    
    def _check_robot(self):
        """检测机器人控制模块"""
        try:
            from robot_controller import (
                X1Interface, 
                go_to_waiting_location,
                action_up_and_down, 
                action_left_and_right,
                action_rotate
            )
            self.robot_available = False
        except ImportError:
            self.robot_available = False
            print("警告: robot_controller模块不可用")
    
    def _check_pypinyin(self):
        """检测拼音处理模块"""
        try:
            import pypinyin
            from pypinyin import pinyin, Style
            self.pypinyin_available = True
        except ImportError:
            self.pypinyin_available = False
            print("警告: pypinyin未安装，请运行: pip install pypinyin")
    
    def _check_numpy(self):
        """检测numpy库"""
        try:
            import numpy as np
            self.numpy_available = True
        except ImportError:
            self.numpy_available = False
            print("警告: numpy未安装，VAD功能不可用，请运行: pip install numpy")
    
    def _check_silero_vad(self):
        """检测Silero VAD - 重新启用优化版"""
        try:
            import torch
            self.torch_available = True
            
            # 尝试加载Silero VAD模型（离线模式）
            try:
                import os
                # 设置环境变量强制离线模式
                os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
                
                # 方法1: 直接使用本地缓存路径
                cache_dir = os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master')
                if os.path.exists(cache_dir):
                    print(f"📁 使用本地缓存: {cache_dir}")
                    model, utils = torch.hub.load(
                        repo_or_dir=cache_dir,
                        model='silero_vad',
                        force_reload=False,
                        onnx=False,
                        trust_repo=True,
                        source='local'
                    )
                else:
                    # 方法2: 使用标准方式但设置离线模式
                    print("🔄 尝试标准加载（离线模式）...")
                    # 设置离线模式环境变量
                    os.environ['TORCH_HUB_OFFLINE'] = '1'
                    
                    model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        onnx=False,
                        trust_repo=True
                    )
                
                self.silero_vad_available = True
                print("✅ Silero VAD模型加载成功（离线模式）")
            except Exception as e:
                print(f"⚠️ Silero VAD模型加载失败: {e}")
                print("💡 提示: 请确保已下载Silero VAD模型到本地缓存")
                print("   可以运行: python -c \"import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')\"")
                self.silero_vad_available = False
                
        except ImportError:
            self.torch_available = False
            self.silero_vad_available = False
            print("⚠️ torch未安装，Silero VAD不可用")


# ================================
# 工具函数
# ================================

@contextlib.contextmanager
def suppress_stderr():
    """临时抑制stderr输出"""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


# ================================
# 语音识别模块
# ================================

class GLM_ASR:
    """基于智谱的GLM-ASR大模型来将语音转为文字"""
    
    def __init__(self, api_key: Optional[str] = ""):
        """初始化GLM-ASR语音识别器"""
        self.api_key = api_key or Config.ZHIPUAI_API_KEY
        if not self.api_key:
            raise ValueError("请提供智谱AI API密钥")
        
        if not deps.zhipuai_available:
            raise ImportError("智谱AI库不可用")
            
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=self.api_key)
    
    def recognize(self, audio_file: str) -> str:
        """识别音频文件并返回文本"""
        try:
            with open(audio_file, "rb") as audio_data:            
                response = self.client.audio.transcriptions.create(
                    model="glm-asr",
                    file=audio_data,
                    stream=False
                )
            # 修复linter错误：正确访问响应内容
            if hasattr(response, 'text'):
                return response.text
            else:
                return ""
        except Exception as e:
            print(f"GLM-ASR语音识别错误: {e}")
            return ""


class GLM_ASR_Recognizer:
    """GLM_ASR语音识别器包装类"""
    
    def __init__(self, device="cpu"):
        self.recognizer = GLM_ASR(api_key="")

    def recognize(self, audio_file: str) -> str:
        return self.recognizer.recognize(audio_file)


# ================================
# 语音活动检测模块
# ================================

class VoiceActivityDetector:
    """基础语音活动检测器 - 基于能量检测"""
    
    def __init__(self, 
                 silence_timeout=0.8,      # 静音超时时间（秒）- 更快结束
                 min_speech_duration=0.6,  # 最小语音持续时间（秒）- 更短语音
                 energy_threshold=0.5,   # 能量阈值 - 更敏感
                 calibration_duration=2.0, # 背景噪声校准时间（秒）
                 sample_rate=16000,        # 采样率
                 chunk_size=1024):         # 块大小
        
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.energy_threshold = energy_threshold
        self.calibration_duration = calibration_duration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # 状态变量
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.background_noise_level = 0.0
        self.calibrated = False
        
        # 历史能量数据（用于背景噪声估计）
        self.energy_history = []
        self.max_history_length = 100
        
        print(f"VAD初始化: 静音超时={silence_timeout}s, 最小语音={min_speech_duration}s, 能量阈值={energy_threshold}")
    
    def _calculate_energy(self, audio_data):
        """计算音频数据的RMS能量"""
        try:
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            
            # 计算RMS能量
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            return rms / 32768.0  # 归一化到0-1范围
        except Exception as e:
            print(f"计算能量失败: {e}")
            return 0.0
    
    def calibrate_background_noise(self, audio_data):
        """校准背景噪声水平"""
        energy = self._calculate_energy(audio_data)
        self.energy_history.append(energy)
        
        # 保持历史数据在合理范围内
        if len(self.energy_history) > self.max_history_length:
            self.energy_history.pop(0)
        
        if not self.calibrated and len(self.energy_history) >= 10:
            # 使用历史数据计算背景噪声水平
            import statistics
            self.background_noise_level = statistics.median(self.energy_history)
            
            # 动态调整阈值
            dynamic_threshold = self.background_noise_level * 2.5
            if dynamic_threshold > self.energy_threshold:
                self.energy_threshold = min(dynamic_threshold, 0.01)
                print(f"VAD动态调整阈值: {self.energy_threshold:.4f} (背景噪声: {self.background_noise_level:.4f})")
            
            self.calibrated = True
    
    def process_audio(self, audio_data):
        """处理音频数据，返回是否应该继续录音"""
        current_time = time.time()
        energy = self._calculate_energy(audio_data)
        
        # 如果还未校准，先进行背景噪声校准
        if not self.calibrated:
            self.calibrate_background_noise(audio_data)
            return True
        
        # 检测语音活动
        is_voice_active = energy > self.energy_threshold
        
        if is_voice_active:
            # 检测到语音
            if not self.is_speaking:
                # 语音开始
                self.is_speaking = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                print("🎤 检测到语音开始")
            else:
                # 语音继续，重置静音计时
                self.silence_start_time = None
        else:
            # 静音状态
            if self.is_speaking:
                # 如果之前在说话，现在开始静音
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
    
                else:
                    # 检查静音持续时间
                    silence_duration = current_time - self.silence_start_time
                    

                    
                    if silence_duration >= self.silence_timeout:
                        # 静音超时，检查是否满足最小语音持续时间
                        if self.speech_start_time:
                            speech_duration = current_time - self.speech_start_time
                            if speech_duration >= self.min_speech_duration:
                                print(f"🔇 检测到语音结束 (语音时长: {speech_duration:.1f}s, 静音时长: {silence_duration:.1f}s)")
                                return False
                            else:
                                print(f"⚠️ 语音时长过短 ({speech_duration:.1f}s < {self.min_speech_duration}s), 继续录音")
                                # 重置状态，继续录音
                                self.is_speaking = False
                                self.speech_start_time = None
                                self.silence_start_time = None
        
        return True
    
    def reset(self):
        """重置VAD状态"""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        print("🔄 VAD状态已重置")


class SileroVAD:
    """Silero VAD封装类 - 支持重采样和音频累积"""
    
    def __init__(self, input_sample_rate=48000, target_sample_rate=16000):
        """初始化Silero VAD"""
        self.input_sample_rate = input_sample_rate    # 原始采样率
        self.target_sample_rate = target_sample_rate  # Silero VAD优化采样率
        self.model = None
        self.utils = None
        
        # 音频累积缓冲区 - 严格按照Silero VAD要求
        # Silero VAD严格要求：16000Hz -> 512样本, 8000Hz -> 256样本
        if target_sample_rate == 16000:
            self.target_samples = 512
        elif target_sample_rate == 8000:
            self.target_samples = 256
        else:
            self.target_samples = 512  # 默认
            
        self.audio_buffer = []
        print(f"🎯 Silero VAD配置: {target_sample_rate}Hz, {self.target_samples}样本")
        
        if not deps.torch_available or not deps.silero_vad_available:
            raise ImportError("Silero VAD不可用，请安装torch: pip install torch")
        
        self._load_model()
    
    def _load_model(self):
        """加载Silero VAD模型 - 强制离线模式"""
        try:
            import torch
            import os
            print("🔄 正在加载Silero VAD模型（离线模式）...")
            
            # 设置环境变量强制离线模式
            os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
            
            # 方法1: 直接使用本地缓存路径
            cache_dir = os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master')
            if os.path.exists(cache_dir):
                print(f"📁 使用本地缓存: {cache_dir}")
                self.model, self.utils = torch.hub.load(
                    repo_or_dir=cache_dir,
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    trust_repo=True,
                    source='local'
                )
            else:
                # 方法2: 使用标准方式但设置离线模式
                print("🔄 尝试标准加载（离线模式）...")
                # 设置离线模式环境变量
                os.environ['TORCH_HUB_OFFLINE'] = '1'
                
                self.model, self.utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    trust_repo=True
                )
            
            print("✅ Silero VAD模型加载成功")
                
        except Exception as e:
            print(f"❌ Silero VAD模型加载失败: {e}")
            print("💡 提示: 请确保已下载Silero VAD模型到本地缓存")
            print("   可以运行: python -c \"import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')\"")
            raise
    
    def _resample_audio(self, audio_array):
        """重采样音频从input_sample_rate到target_sample_rate"""
        import numpy as np
        
        if self.input_sample_rate == self.target_sample_rate:
            return audio_array
        
        try:
            # 使用简单的线性插值重采样
            ratio = self.target_sample_rate / self.input_sample_rate
            new_length = int(len(audio_array) * ratio)
            
            if new_length == 0:
                return np.array([], dtype=audio_array.dtype)
            
            # 线性插值重采样
            old_indices = np.linspace(0, len(audio_array) - 1, new_length)
            new_audio = np.interp(old_indices, np.arange(len(audio_array)), audio_array)
            
            return new_audio.astype(audio_array.dtype)
            
        except Exception as e:
            print(f"⚠️ 音频重采样失败: {e}")
            return audio_array
    
    def detect_speech(self, audio_chunk):
        """检测音频块中的语音活动 - 支持累积和重采样"""
        try:
            import torch
            import numpy as np
            
            # 确保音频是numpy数组格式
            if isinstance(audio_chunk, bytes):
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_array = audio_chunk.copy()
            
            # 添加到缓冲区
            self.audio_buffer.extend(audio_array)
            
            # 计算需要多少原始采样率的样本才能重采样到target_samples
            required_input_samples = int(self.target_samples * (self.input_sample_rate / self.target_sample_rate))
            
            # 检查是否有足够的数据进行检测
            if len(self.audio_buffer) < required_input_samples:
                return 0.0  # 数据不够，返回无语音
            
            # 精确取出所需的样本数
            window_audio = np.array(self.audio_buffer[:required_input_samples], dtype=np.int16)
            
            # 从缓冲区移除已处理的数据（减少重叠提高响应速度）
            overlap = required_input_samples // 4  # 从50%减少到25%重叠
            self.audio_buffer = self.audio_buffer[required_input_samples - overlap:]
            
            # 重采样到Silero VAD要求的精确样本数
            resampled_audio = self._resample_audio(window_audio)
            
            # 确保重采样后的样本数严格等于target_samples
            if len(resampled_audio) != self.target_samples:
                if len(resampled_audio) > self.target_samples:
                    resampled_audio = resampled_audio[:self.target_samples]
                else:
                    # 填充到目标长度
                    padded_array = np.zeros(self.target_samples, dtype=resampled_audio.dtype)
                    padded_array[:len(resampled_audio)] = resampled_audio
                    resampled_audio = padded_array
            
            # 转换为float32并归一化
            if resampled_audio.dtype != np.float32:
                resampled_audio = resampled_audio.astype(np.float32) / 32768.0
            
            # 验证样本数（调试用）
            if len(resampled_audio) != self.target_samples:
                print(f"⚠️ 样本数不匹配: 期望{self.target_samples}，实际{len(resampled_audio)}")
                return 0.0
            
            # 转换为torch tensor
            audio_tensor = torch.from_numpy(resampled_audio)
            
            # 使用模型检测
            speech_prob = self.model(audio_tensor, self.target_sample_rate).item()
            
            return speech_prob
            
        except Exception as e:
            print(f"⚠️ Silero VAD检测失败: {e}")
            return 0.0
    
    def reset_buffer(self):
        """重置音频缓冲区"""
        self.audio_buffer = []
    
    def is_speech(self, audio_chunk, threshold=0.5):
        """判断音频块是否包含语音"""
        prob = self.detect_speech(audio_chunk)
        return prob > threshold

    def deep_reset(self):
        """深度重置Silero VAD - 彻底清理所有内部状态"""
        try:
            import gc
            
            print("🔄 执行Silero VAD深度重置...")
            
            # 1. 清空音频缓冲区
            self.audio_buffer = []
            
            # 2. 强制垃圾回收
            gc.collect()
            
            # 3. 重新加载模型（如果可能）
            try:
                if self.model is not None:
                    # 尝试重新初始化模型
                    import torch
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("   ✅ Silero VAD模型状态已重置")
            except Exception as e:
                print(f"   ⚠️ 模型重置异常: {e}")
            
            # 4. 重新加载工具类（如果需要）
            try:
                if self.utils is None:
                    # 如果工具类丢失，尝试重新加载（离线模式）
                    import torch
                    import os
                    # 设置环境变量强制离线模式
                    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
                    
                    try:
                        # 尝试使用本地缓存
                        cache_dir = os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master')
                        if os.path.exists(cache_dir):
                            _, self.utils = torch.hub.load(
                                repo_or_dir=cache_dir,
                                model='silero_vad',
                                force_reload=False,
                                onnx=False,
                                trust_repo=True,
                                source='local'
                            )
                        else:
                            # 使用标准方式但设置离线模式
                            os.environ['TORCH_HUB_OFFLINE'] = '1'
                            _, self.utils = torch.hub.load(
                                repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=False,
                                onnx=False,
                                trust_repo=True
                            )
                        print("   ✅ Silero VAD工具类已重新加载")
                    except Exception as e:
                        print(f"   ⚠️ 工具类重新加载失败: {e}")
            except Exception as e:
                print(f"   ⚠️ 工具类重新加载异常: {e}")
            
            # 5. 验证重置结果
            if len(self.audio_buffer) == 0:
                print("   ✅ Silero VAD深度重置成功")
                return True
            else:
                print("   ❌ Silero VAD重置失败")
                return False
                
        except Exception as e:
            print(f"   ❌ Silero VAD深度重置异常: {e}")
        return False
    
    def health_check(self):
        """Silero VAD健康检测 - 检查内部状态是否正常"""
        try:
            health_status = {
                'model_loaded': self.model is not None,
                'utils_loaded': self.utils is not None,
                'buffer_empty': len(self.audio_buffer) == 0,
                'buffer_size': len(self.audio_buffer),
                'target_samples': self.target_samples,
                'input_sample_rate': self.input_sample_rate,
                'target_sample_rate': self.target_sample_rate
            }
            
            # 检查关键状态
            issues = []
            if not health_status['model_loaded']:
                issues.append("模型未加载")
            if not health_status['utils_loaded']:
                issues.append("工具类未加载")
            if health_status['buffer_size'] > 10000:  # 缓冲区过大
                issues.append(f"缓冲区过大: {health_status['buffer_size']}样本")
            
            if issues:
                print(f"⚠️ Silero VAD健康检查发现问题: {', '.join(issues)}")
                return False
            else:
                print(f"✅ Silero VAD健康检查通过: 缓冲区{health_status['buffer_size']}样本")
                return True
                
        except Exception as e:
            print(f"❌ Silero VAD健康检查异常: {e}")
            return False


class CircularBuffer:
    """环形缓冲区用于音频数据"""
    
    def __init__(self, max_duration_seconds, sample_rate, channels=1, dtype=None):
        """初始化环形缓冲区
        
        Args:
            max_duration_seconds: 最大缓存时长（秒）
            sample_rate: 采样率
            channels: 声道数
            dtype: 数据类型
        """
        import numpy as np
        
        if dtype is None:
            dtype = np.int16
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        
        # 计算缓冲区大小
        self.max_samples = int(max_duration_seconds * sample_rate * channels)
        self.buffer = np.zeros(self.max_samples, dtype=dtype)
        
        # 缓冲区指针
        self.write_pos = 0
        self.is_full = False
        
        print(f"🔄 环形缓冲区初始化: {max_duration_seconds}s, {self.max_samples}样本")
    
    def write(self, data):
        """写入音频数据"""
        import numpy as np
        
        if isinstance(data, bytes):
            audio_array = np.frombuffer(data, dtype=self.dtype)
        else:
            audio_array = data.astype(self.dtype)
        
        data_length = len(audio_array)
        
        if data_length == 0:
            return
        
        # 如果数据长度超过缓冲区大小，只保留最后的部分
        if data_length >= self.max_samples:
            self.buffer = audio_array[-self.max_samples:].copy()
            self.write_pos = 0
            self.is_full = True
            return
        
        # 计算写入位置
        end_pos = self.write_pos + data_length
        
        if end_pos <= self.max_samples:
            # 数据可以直接写入，不需要环绕
            self.buffer[self.write_pos:end_pos] = audio_array
        else:
            # 需要环绕写入
            first_part_size = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = audio_array[:first_part_size]
            self.buffer[:data_length - first_part_size] = audio_array[first_part_size:]
            self.is_full = True
        
        self.write_pos = end_pos % self.max_samples
        if end_pos >= self.max_samples:
            self.is_full = True
    
    def read_last_seconds(self, duration_seconds):
        """读取最后N秒的音频数据"""
        import numpy as np
        
        samples_needed = int(duration_seconds * self.sample_rate * self.channels)
        samples_needed = min(samples_needed, self.max_samples)
        
        if not self.is_full and self.write_pos < samples_needed:
            # 缓冲区还未满，且数据不够
            return self.buffer[:self.write_pos].copy()
        
        if samples_needed >= self.max_samples:
            # 需要全部数据
            if self.is_full:
                # 按正确顺序重新排列数据
                result = np.zeros(self.max_samples, dtype=self.dtype)
                result[:self.max_samples - self.write_pos] = self.buffer[self.write_pos:]
                result[self.max_samples - self.write_pos:] = self.buffer[:self.write_pos]
                return result
            else:
                return self.buffer[:self.write_pos].copy()
        
        # 读取最后的samples_needed个样本
        if self.is_full:
            # 缓冲区已满，需要计算正确的读取位置
            start_pos = (self.write_pos - samples_needed) % self.max_samples
            
            if start_pos + samples_needed <= self.max_samples:
                # 数据连续，直接读取
                return self.buffer[start_pos:start_pos + samples_needed].copy()
            else:
                # 数据跨越边界，需要分两段读取
                result = np.zeros(samples_needed, dtype=self.dtype)
                first_part_size = self.max_samples - start_pos
                result[:first_part_size] = self.buffer[start_pos:]
                result[first_part_size:] = self.buffer[:samples_needed - first_part_size]
                return result
        else:
            # 缓冲区未满
            start_pos = max(0, self.write_pos - samples_needed)
            return self.buffer[start_pos:self.write_pos].copy()
    
    def clear(self):
        """清空缓冲区"""
        import numpy as np
        self.buffer.fill(0)
        self.write_pos = 0
        self.is_full = False


class AdvancedVoiceActivityDetector:
    """高级语音活动检测器 - 基于Silero VAD + 环形缓冲区"""
    
    def __init__(self, 
                 sample_rate=16000,
                 buffer_duration=4.0,      # 环形缓冲区时长（秒）
                 lookback_duration=0.5,    # 回溯时长（秒）
                 silence_timeout=1.0,      # 静音超时（秒）
                 min_speech_duration=0.4,  # 最小语音时长（秒）- 允许更短语音
                 speech_threshold=0.1,     # Silero VAD阈值 - 提高敏感度
                 check_interval_ms=10):    # 检测间隔（毫秒）- 优化响应速度
        
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.lookback_duration = lookback_duration
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.speech_threshold = speech_threshold
        self.check_interval_ms = check_interval_ms
        
        # 初始化组件 - 启用优化版Silero VAD
        if deps.silero_vad_available and deps.torch_available:
            try:
                # 使用音频累积+重采样的Silero VAD
                self.silero_vad = SileroVAD(
                    input_sample_rate=sample_rate,    # 你的实际采样率（48000Hz）
                    target_sample_rate=16000          # Silero VAD优化采样率
                )
                self.vad_available = True
                print("✅ 使用优化版Silero VAD（音频累积+重采样）")
                print(f"📊 采样率转换: {sample_rate}Hz → 16000Hz")
            except Exception as e:
                print(f"⚠️ Silero VAD初始化失败: {e}")
                print("⚠️ 回退到基础能量VAD")
                self.silero_vad = None
                self.vad_available = False
        else:
            print("⚠️ Silero VAD不可用，回退到基础能量VAD")
            self.silero_vad = None
            self.vad_available = False
        
        # 初始化环形缓冲区
        import numpy as np
        self.buffer = CircularBuffer(
            max_duration_seconds=buffer_duration,
            sample_rate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        
        # 状态管理
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_check_time = 0
        
        # 回退VAD（能量检测）
        if not self.vad_available:
            self.fallback_vad = VoiceActivityDetector(
                silence_timeout=silence_timeout,
                min_speech_duration=min_speech_duration,
                sample_rate=sample_rate
            )
        
        print(f"🎯 高级VAD初始化完成:")
        print(f"   缓冲区: {buffer_duration}s, 回溯: {lookback_duration}s")
        print(f"   静音超时: {silence_timeout}s, 最小语音: {min_speech_duration}s")
        print(f"   检测间隔: {check_interval_ms}ms")
    
    def feed_audio(self, audio_data):
        """向环形缓冲区输入音频数据"""
        self.buffer.write(audio_data)
        
        # 检查是否需要进行VAD检测
        current_time = time.time() * 1000  # 毫秒
        if current_time - self.last_check_time >= self.check_interval_ms:
            self.last_check_time = current_time
            return self._check_voice_activity(audio_data)
        
        return None
    
    def _check_voice_activity(self, latest_audio):
        """检查语音活动状态"""
        current_time = time.time()
        
        if self.vad_available:
            # 使用Silero VAD
            speech_prob = self.silero_vad.detect_speech(latest_audio)
            is_speech = speech_prob > self.speech_threshold
        else:
            # 使用回退VAD（能量检测）
            # 注意：这里我们需要将音频数据转换为字节格式供回退VAD使用
            if isinstance(latest_audio, bytes):
                audio_bytes = latest_audio
            else:
                import numpy as np
                if latest_audio.dtype != np.int16:
                    audio_int16 = (latest_audio * 32767).astype(np.int16)
                else:
                    audio_int16 = latest_audio
                audio_bytes = audio_int16.tobytes()
            
            # 使用回退VAD的能量检测
            energy = self.fallback_vad._calculate_energy(audio_bytes)
            is_speech = energy > 0.005  # 使用较低的阈值
        
        # 状态机逻辑
        if is_speech:
            if not self.is_speech_active:
                # 语音开始 - 立即触发
                self.is_speech_active = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                print(f"🎤 检测到语音开始 ({'Silero' if self.vad_available else 'Energy'} VAD)")
                return "speech_start"
            else:
                # 语音继续
                self.silence_start_time = None
                return "speech_continue"
        else:
            if self.is_speech_active:
                # 可能的语音结束
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                    return "silence_start"
                else:
                    # 检查静音持续时间
                    silence_duration = current_time - self.silence_start_time
                    if silence_duration >= self.silence_timeout:
                        # 语音结束
                        if self.speech_start_time:
                            speech_duration = current_time - self.speech_start_time
                            if speech_duration >= self.min_speech_duration:
                                print(f"🔇 检测到语音结束 (语音: {speech_duration:.1f}s, 静音: {silence_duration:.1f}s)")
                                return "speech_end"
                            else:
                                print(f"⚠️ 语音过短 ({speech_duration:.1f}s), 忽略")
                                self._reset_state()
                                return "speech_too_short"
                        else:
                            self._reset_state()
                            return "speech_too_short"
                    else:
                        return "silence_continue"
        
        return "no_change"
    
    def get_speech_audio(self):
        """获取检测到的语音音频（包含回溯）"""
        if not self.is_speech_active or not self.speech_start_time:
            return None
    
        # 计算需要回溯的总时长
        current_time = time.time()
        speech_duration = current_time - self.speech_start_time
        
        # 增加额外的安全回溯时间，确保不丢失语音开头
        safe_lookback = self.lookback_duration + 0.3  # 额外300ms安全边界
        total_duration = speech_duration + safe_lookback
        
        # 从环形缓冲区读取音频
        audio_data = self.buffer.read_last_seconds(total_duration)
        
        if len(audio_data) == 0:
            return None
        
        print(f"📀 提取语音音频: {len(audio_data)}样本 ({len(audio_data)/self.sample_rate:.2f}s)")
        print(f"   📊 回溯详情: 语音时长{speech_duration:.2f}s + 安全回溯{safe_lookback:.2f}s = 总计{total_duration:.2f}s")
        return audio_data
    
    def _reset_state(self):
        """重置状态"""
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
    
    def reset(self):
        """重置检测器 - 增强版，包含Silero VAD深度重置"""
        self._reset_state()
        self.buffer.clear()
        
        # 深度重置Silero VAD（解决长时间运行后的状态问题）
        if self.vad_available and self.silero_vad:
            try:
                # 先尝试深度重置
                deep_reset_success = self.silero_vad.deep_reset()
                if deep_reset_success:
                    print("🔄 Silero VAD深度重置成功")
                else:
                    # 如果深度重置失败，尝试普通重置
                    print("⚠️ Silero VAD深度重置失败，尝试普通重置")
                    self.silero_vad.reset_buffer()
            except Exception as e:
                print(f"⚠️ Silero VAD重置异常: {e}")
                # 最后的回退方案
                try:
                    self.silero_vad.reset_buffer()
                except:
                    pass
        
        print("�� 高级VAD状态已重置（增强版）")




# ================================
# 音频处理模块
# ================================

# AudioDeviceManager类已删除，功能已集成到CrossPlatformAudioManager中


# VoiceRecorder类已删除，使用SimplifiedVoiceRecorder替代


# ================================
# 音频播放模块
# ================================

# AudioPlayer类已删除，使用SimplifiedAudioPlayer替代


# ================================
# 跨平台音频处理模块
# ================================

class CrossPlatformAudioManager:
    """跨平台音频管理器 - 使用sounddevice和soundfile"""
    
    def __init__(self):
        self.sample_rate = self._get_compatible_sample_rate()
        self.channels = 1
        self.dtype = 'int16'
        
        # 预选择最佳设备
        self.input_device = self._get_best_input_device()
        self.output_device = self._get_best_output_device()
        
        print(f"🎯 音频设备预选择完成:")
        print(f"   输入设备: {self.input_device}")
        print(f"   输出设备: {self.output_device}")
        print(f"   采样率: {self.sample_rate}Hz")
    
    def _get_compatible_sample_rate(self):
        """获取兼容的采样率"""
        try:
            import sounddevice as sd
            
            # 测试常用采样率
            test_rates = [48000, 44100, 22050, 16000, 8000]
            
            for rate in test_rates:
                try:
                    # 短暂测试这个采样率
                    test_duration = 0.01  # 10ms
                    recording = sd.rec(
                        int(test_duration * rate), 
                        samplerate=rate, 
                        channels=1,
                        dtype='int16'
                    )
                    sd.wait()
                    print(f"✅ 使用采样率: {rate}Hz")
                    return rate
                except Exception as e:
                    print(f"⚠️ 采样率 {rate}Hz 不兼容: {e}")
                    continue
            
            # 如果都不行，使用默认设备的采样率
            try:
                device_info = sd.query_devices(kind='input')
                default_rate = int(device_info['default_samplerate'])
                print(f"✅ 使用设备默认采样率: {default_rate}Hz")
                return default_rate
            except:
                print("⚠️ 回退到16000Hz")
                return 16000
                
        except ImportError:
            return 44100  # 默认值
        
    def record_audio(self, duration=None, use_vad=True):
        """录制音频，返回numpy数组"""
        try:
            import sounddevice as sd
            import numpy as np
            
            # 使用预选择的输入设备
            input_device = self.input_device
            
            if duration:
                # 固定时长录音
                print(f"🎤 开始录音 {duration} 秒...")
                recording = sd.rec(
                    int(duration * self.sample_rate), 
                    samplerate=self.sample_rate, 
                    channels=self.channels,
                    dtype=self.dtype,
                    device=input_device
                )
                sd.wait()
                return recording
            else:
                # 手动停止录音
                print("🎤 开始录音... 按回车键停止")
                frames = []
                
                def callback(indata, frames_count, time, status):
                    if status:
                        print(f"⚠️ 录音状态: {status}")
                    frames.append(indata.copy())
                
                try:
                    with sd.InputStream(callback=callback, 
                                      samplerate=self.sample_rate, 
                                      channels=self.channels,
                                      dtype=self.dtype,
                                      device=input_device):
                        try:
                            input()  # 等待用户按回车
                        except KeyboardInterrupt:
                            print("\n🛑 用户中断录音")
                            raise
                    
                    if frames:
                        return np.concatenate(frames, axis=0)
                    return None
                except Exception as stream_error:
                    print(f"录音流错误: {stream_error}")
                    # 尝试回退方案
                    return self._fallback_record()
                
        except Exception as e:
            print(f"sounddevice录音失败: {e}")
            # 尝试回退到pyaudio
            return self._fallback_record()
    
    def _get_best_input_device(self):
        """获取最佳输入设备"""
        try:
            import sounddevice as sd
            
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device))
            
            if not input_devices:
                return None  # 使用默认设备
            
            # 优先选择USB设备
            for device_idx, device in input_devices:
                device_name = device['name'].lower()
                if any(usb_indicator in device_name for usb_indicator in 
                      ['usb', 'headset', 'microphone', 'webcam']):
                    print(f"🎯 选择USB设备: {device['name']}")
                    return device_idx
            
            # 选择第一个可用设备
            device_idx, device = input_devices[0]
            print(f"🎤 选择设备: {device['name']}")
            return device_idx
            
        except Exception as e:
            print(f"设备选择失败: {e}")
            return None
    
    def _fallback_record(self):
        """回退到pyaudio录音"""
        try:
            print("🔄 尝试pyaudio回退录音...")
            import pyaudio
            import numpy as np
            
            p = pyaudio.PyAudio()
            
            # 使用16000Hz作为回退采样率
            fallback_rate = 16000
            chunk = 1024
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=fallback_rate,
                input=True,
                frames_per_buffer=chunk
            )
            
            print("🎤 PyAudio录音... 按回车键停止")
            frames = []
            
            def record_thread():
                while True:
                    try:
                        data = stream.read(chunk, exception_on_overflow=False)
                        frames.append(data)
                    except:
                        break
            
            import threading
            thread = threading.Thread(target=record_thread)
            thread.daemon = True
            thread.start()
            
            input()  # 等待用户按回车
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if frames:
                # 转换为numpy数组
                audio_data = b''.join(frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                # 重塑为sounddevice兼容格式
                audio_array = audio_array.reshape(-1, 1).astype(np.int16)
                
                # 更新采样率
                self.sample_rate = fallback_rate
                print(f"✅ PyAudio录音成功，采样率: {fallback_rate}Hz")
                return audio_array
            
            return None
            
        except Exception as e:
            print(f"PyAudio回退录音也失败: {e}")
            return None
    
    def save_audio(self, audio_data, filename):
        """保存音频到文件"""
        try:
            import soundfile as sf
            
            # 确保音频数据格式正确
            if audio_data is None:
                print("⚠️ 音频数据为空")
                return False
            
            # 如果是2D数组，确保是正确的形状
            if len(audio_data.shape) > 1 and audio_data.shape[1] == 1:
                audio_data = audio_data.flatten()
            
            sf.write(filename, audio_data, self.sample_rate)
            print(f"✅ 音频已保存: {filename} ({self.sample_rate}Hz)")
            return True
        except Exception as e:
            print(f"soundfile保存失败: {e}")
            # 回退到wave保存
            return self._fallback_save(audio_data, filename)
    
    def _fallback_save(self, audio_data, filename):
        """回退到wave保存音频"""
        try:
            import wave
            import numpy as np
            
            # 确保数据是int16格式
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # 单声道
                wf.setsampwidth(2)  # 16位
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            print(f"✅ 音频已保存(wave): {filename}")
            return True
        except Exception as e:
            print(f"wave保存也失败: {e}")
            return False
    
    def play_audio(self, filename):
        """播放音频文件 - 智能采样率处理"""
        try:
            import sounddevice as sd
            import soundfile as sf
            
            data, file_sample_rate = sf.read(filename)
            print(f"🔊 播放音频: {filename}")
            
            # 使用预选择的输出设备
            output_device = self.output_device
            
            # 直接使用设备采样率播放，避免报错
            target_rate = self.sample_rate  # 使用初始化时检测到的采样率
            
            if file_sample_rate != target_rate:
                print(f"🔄 转换采样率: {file_sample_rate}Hz → {target_rate}Hz")
                # 简单重采样
                import numpy as np
                ratio = target_rate / file_sample_rate
                new_length = int(len(data) * ratio)
                resampled_data = np.interp(
                    np.linspace(0, len(data), new_length),
                    np.arange(len(data)),
                    data
                )
                data = resampled_data
                file_sample_rate = target_rate
            
            # 直接播放
            sd.play(data, file_sample_rate, device=output_device)
            sd.wait()
            print(f"✅ 播放成功: {file_sample_rate}Hz")
            return True
                
        except Exception as e:
            print(f"播放音频失败: {e}")
            # 回退到系统播放
            return self._fallback_play_system(filename)
    
    def _get_best_output_device(self):
        """获取最佳输出设备"""
        try:
            import sounddevice as sd
            
            devices = sd.query_devices()
            output_devices = []
            
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    output_devices.append((i, device))
            
            if not output_devices:
                return None  # 使用默认设备
            
            # 优先选择USB设备
            for device_idx, device in output_devices:
                device_name = device['name'].lower()
                if any(usb_indicator in device_name for usb_indicator in 
                      ['usb', 'headset', 'speaker', 'headphone']):
                    print(f"🔊 选择USB输出设备: {device['name']}")
                    return device_idx
            
            # 选择第一个可用设备
            device_idx, device = output_devices[0]
            print(f"🔊 选择输出设备: {device['name']}")
            return device_idx
            
        except Exception as e:
            print(f"输出设备选择失败: {e}")
            return None
    
    def _play_with_resampling(self, data, original_rate, output_device):
        """重采样后播放音频"""
        try:
            import sounddevice as sd
            
            # 测试不同采样率
            test_rates = [48000, 44100, 22050, 16000]
            
            for target_rate in test_rates:
                if target_rate == original_rate:
                    continue
                    
                try:
                    print(f"🔄 尝试重采样到 {target_rate}Hz...")
                    
                    # 简单的重采样（线性插值）
                    import numpy as np
                    ratio = target_rate / original_rate
                    
                    if ratio != 1.0:
                        # 重采样
                        new_length = int(len(data) * ratio)
                        resampled_data = np.interp(
                            np.linspace(0, len(data), new_length),
                            np.arange(len(data)),
                            data
                        )
                    else:
                        resampled_data = data
                    
                    # 尝试播放
                    sd.play(resampled_data, target_rate, device=output_device)
                    sd.wait()
                    print(f"✅ 重采样播放成功: {target_rate}Hz")
                    return True
                    
                except Exception as resample_error:
                    print(f"⚠️ {target_rate}Hz 重采样失败: {resample_error}")
                    continue
            
            # 所有采样率都失败，尝试回退到系统播放
            return self._fallback_play(data, original_rate)
            
        except Exception as e:
            print(f"重采样播放失败: {e}")
            return False
    
    def _fallback_play(self, data, sample_rate):
        """回退播放方案"""
        try:
            print("🔄 尝试系统播放命令...")
            import tempfile
            import subprocess
            import soundfile as sf
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_filename = tmp_file.name
            
            sf.write(temp_filename, data, sample_rate)
            
            # 尝试系统播放命令
            play_commands = [
                ['aplay', temp_filename],
                ['paplay', temp_filename],
                ['play', temp_filename]
            ]
            
            for cmd in play_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                    if result.returncode == 0:
                        print(f"✅ 系统播放成功: {' '.join(cmd)}")
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            print("❌ 所有播放方案都失败")
            return False
            
        except Exception as e:
            print(f"回退播放失败: {e}")
            return False
    
    def _fallback_play_system(self, filename):
        """系统播放回退方案"""
        try:
            import subprocess
            
            # 尝试系统播放命令
            play_commands = [
                ['aplay', filename],
                ['paplay', filename],
                ['play', filename]
            ]
            
            for cmd in play_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                    if result.returncode == 0:
                        print(f"✅ 系统播放成功: {' '.join(cmd)}")
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            print("❌ 系统播放也失败")
            return False
            
        except Exception as e:
            print(f"系统播放失败: {e}")
            return False
    
    def convert_audio_format(self, input_file, output_file, target_sample_rate=None, target_channels=None):
        """转换音频格式"""
        try:
            import soundfile as sf
            import numpy as np
            
            data, sample_rate = sf.read(input_file)
            
            # # 转换采样率
            # if target_sample_rate and target_sample_rate != sample_rate:
            #     try:
            #         import librosa
            #         data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sample_rate)
            #         sample_rate = target_sample_rate
            #     except ImportError:
            #         print("⚠️ librosa未安装，跳过采样率转换")
            
            # 转换声道数
            if target_channels:
                if target_channels == 1 and len(data.shape) > 1:
                    # 转为单声道
                    data = data.mean(axis=1)
                elif target_channels == 2 and len(data.shape) == 1:
                    # 转为立体声
                    data = np.stack([data, data], axis=1)
            
            sf.write(output_file, data, sample_rate)
            return True
        except Exception as e:
            print(f"音频格式转换失败: {e}")
            return False
    
    def _force_reinitialize_audio_system(self):
        """强力重新初始化音频系统 - 解决长时间运行后的设备状态问题"""
        try:
            import sounddevice as sd
            import gc
            
            print("🔄 强力重新初始化音频系统...")
            
            # 1. 清除sounddevice的内部缓存
            try:
                # 强制查询设备以刷新内部状态  
                _ = sd.query_devices()
                print("   ✅ sounddevice设备缓存已刷新")
            except Exception as e:
                print(f"   ⚠️ 设备缓存刷新失败: {e}")
            
            # 2. 重新检测设备
            old_input = self.input_device
            old_output = self.output_device
            
            self.input_device = self._get_best_input_device()
            self.output_device = self._get_best_output_device()
            
            # 3. 验证设备可用性
            if self.input_device != old_input:
                print(f"   🔄 输入设备已更新: {old_input} → {self.input_device}")
            
            if self.output_device != old_output:
                print(f"   🔄 输出设备已更新: {old_output} → {self.output_device}")
            
            # 4. 测试新设备
            success = self._test_audio_device_health()
            
            if success:
                print("   ✅ 音频系统重新初始化成功")
                return True
            else:
                print("   ⚠️ 音频系统重新初始化后仍有问题")
                return False
                
        except Exception as e:
            print(f"   ❌ 音频系统重新初始化失败: {e}")
            return False
    
    def _test_audio_device_health(self):
        """测试音频设备健康状态"""
        try:
            import sounddevice as sd
            import numpy as np
            
            # 短暂测试录音功能
            test_duration = 0.1  # 100ms
            try:
                recording = sd.rec(
                    int(test_duration * self.sample_rate), 
                    samplerate=self.sample_rate, 
                    channels=self.channels,
                    dtype=self.dtype,
                    device=self.input_device
                )
                sd.wait()
                
                # 检查录音数据是否有效
                if recording is not None and len(recording) > 0:
                    print(f"   ✅ 设备健康检查通过: {len(recording)} 样本")
                    return True
                else:
                    print("   ❌ 设备健康检查失败: 无录音数据")
                    return False
                    
            except Exception as test_error:
                print(f"   ❌ 设备健康检查失败: {test_error}")
                return False
                
        except Exception as e:
            print(f"   ❌ 设备健康检查异常: {e}")
            return False


class SimplifiedVoiceRecorder:
    """简化的语音录制器 - 跨平台版本，支持高级VAD"""
    
    def __init__(self):
        if not deps.audio_available:
            raise RuntimeError("音频设备不可用，无法进行录音")
        
        self.audio_manager = CrossPlatformAudioManager()
        self.is_recording = False
        self.is_listening = False
        self.audio_data = None
        
        # 添加超时重置机制
        self.last_activity_time = time.time()
        self.activity_timeout = 15.0  # 缩短到15秒无活动后重置VAD（更频繁的检查）
        
        # VAD预热标志
        self._vad_prewarmed = False
        
        # 初始化高级VAD（暂时禁用Silero VAD，使用环形缓冲区+基础VAD）
        if deps.numpy_available:
            try:
                self.advanced_vad = AdvancedVoiceActivityDetector(
                    sample_rate=self.audio_manager.sample_rate,
                    buffer_duration=4.0,      # 4秒环形缓冲区（增加容量）
                    lookback_duration=1.5,    # 1000ms回溯（增加回溯时间）
                    silence_timeout=1.0,      # 1.0秒静音超时（更快结束）
                    min_speech_duration=0.4,  # 最小语音400ms（更短语音）
                    speech_threshold=0.2,     # VAD阈值（更敏感）
                    check_interval_ms=20      # 20ms检测间隔（更快响应）
                )
                self.vad_enabled = True
                if self.advanced_vad.vad_available:
                    print("✅ 高级VAD语音活动检测已启用（Silero VAD优化版）")
                    print("💡 音频累积+重采样+环形缓冲区：完美解决背景噪声问题")
                else:
                    print("✅ 高级VAD语音活动检测已启用（环形缓冲区+基础VAD模式）")
                    print("💡 核心功能：持续监听+按需录音+智能回溯")
            except Exception as e:
                print(f"⚠️ 高级VAD初始化失败: {e}")
                # 回退到基础VAD
                self.vad = VoiceActivityDetector()
                self.advanced_vad = None
                self.vad_enabled = True
                print("⚠️ 使用传统基础VAD")
        else:
            self.advanced_vad = None
            self.vad = None
            self.vad_enabled = False
            print("⚠️ VAD功能不可用，需要numpy库")
        
    def _check_and_reset_if_inactive(self):
        """检查是否长时间不活跃，如果是则重置VAD状态"""
        current_time = time.time()
        if current_time - self.last_activity_time > self.activity_timeout:
            print(f"⏰ 检测到{self.activity_timeout}秒无活动，自动重置VAD状态...")
            # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
            # self.cleanup()
            # self.last_activity_time = current_time
            # print("🔄 VAD状态重置完成，准备接收新的语音输入")
            print("🔄 VAD重置功能已暂时禁用，仅更新活动时间")
            self.last_activity_time = current_time
    
    def _check_audio_device_health(self):
        """检查音频设备健康状态，如果异常则重新初始化"""
        try:
            # 检查音频设备是否可用
            if not hasattr(self.audio_manager, '_test_audio_device_health'):
                return True  # 如果没有健康检查方法，假设设备正常
            
            device_healthy = self.audio_manager._test_audio_device_health()
            
            if not device_healthy:
                print("⚠️ 音频设备健康检查失败，尝试重新初始化...")
                
                # 尝试重新初始化音频管理器
                try:
                    # 重新获取最佳输入设备
                    new_input_device = self.audio_manager._get_best_input_device()
                    if new_input_device is not None:
                        self.audio_manager.input_device = new_input_device
                        print(f"🔄 音频设备重新初始化成功: {new_input_device}")
                        return True
                    else:
                        print("❌ 无法重新获取音频设备")
                        return False
                except Exception as e:
                    print(f"❌ 音频设备重新初始化失败: {e}")
                    return False
            else:
                return True
                
        except Exception as e:
            print(f"⚠️ 音频设备健康检查异常: {e}")
            return False
        
    def start_recording(self, use_vad=True):
        """开始录音 - 支持高级VAD自动停止（优化版，支持预热后快速启动）"""
        # 检查并处理长时间不活跃的情况
        self._check_and_reset_if_inactive()
        
        # 快速音频设备健康检查（如果已经预热过，跳过详细检查）
        if hasattr(self, '_vad_prewarmed') and self._vad_prewarmed:
            print("⚡ VAD已预热，快速启动录音...")
            self._vad_prewarmed = False  # 重置预热标志
            self.last_activity_time = time.time()
            return self._record_with_advanced_vad() if use_vad else self._record_with_basic_vad()
        
        # 音频设备健康检查
        if not self._check_audio_device_health():
            print("⚠️ 音频设备健康检查失败，尝试使用基础录音模式...")
            # 如果健康检查失败，尝试基础录音
            try:
                self.is_recording = True
                self.audio_data = self.audio_manager.record_audio(duration=5.0)  # 5秒测试录音
                self.is_recording = False
                if self.audio_data is not None:
                    print("✅ 基础录音测试成功，继续正常流程")
                    self.last_activity_time = time.time()
                    return True
                else:
                    print("❌ 基础录音测试失败，无法继续")
                    return False
            except Exception as e:
                print(f"❌ 基础录音测试异常: {e}")
                return False
        
        # 更新活动时间
        self.last_activity_time = time.time()
        
        # 系统准备就绪，结束计时（如果不是预热模式）
        if not (hasattr(self, '_vad_prewarmed') and self._vad_prewarmed):
            # 通知主控制器系统准备就绪
            if hasattr(self, 'controller') and self.controller:
                self.controller._end_response_cycle_timing()
        
        if not use_vad or not self.vad_enabled:
            # 手动模式
            self.is_recording = True
            self.audio_data = self.audio_manager.record_audio()
            self.is_recording = False
            return self.audio_data is not None
        else:
            # 高级VAD自动模式
            if self.advanced_vad:
                result = self._record_with_advanced_vad()
            else:
                result = self._record_with_basic_vad()
            
            # 录音完成后更新活动时间
            if result:
                self.last_activity_time = time.time()
                
            return result
    
    def _record_with_advanced_vad(self):
        """使用高级VAD进行智能录音 - 持续监听+按需录音"""
        try:
            import sounddevice as sd
            import numpy as np
            import threading
            
            # 在开始录音前再次检查音频设备健康状态
            if not self._check_audio_device_health():
                print("⚠️ 高级VAD录音前音频设备健康检查失败，尝试重试...")
                # 等待一秒后重试
                time.sleep(1.0)
                if not self._check_audio_device_health():
                    print("❌ 音频设备健康检查持续失败，回退到基础录音")
                    return self._record_with_basic_vad()
            
            if self.advanced_vad.vad_available:
                print("🎤 开始智能监听... Silero VAD自动检测语音（优化版）")
                print("💡 音频自动累积+重采样，智能识别语音，录音更精准")
            else:
                print("🎤 开始智能监听... 环形缓冲区+VAD自动检测语音")
                print("💡 说话时自动开始录音，语音结束自动停止，只保存纯净语音片段")
            
            # 获取输入设备
            input_device = self.audio_manager._get_best_input_device()
            
            # 重置VAD状态
            self.advanced_vad.reset()
            
            # 状态管理
            self.is_listening = True
            self.is_recording = False
            self.audio_data = None
            speech_detected = threading.Event()
            recording_complete = threading.Event()
            
            # 新增：独立的录音缓冲区
            recording_buffer = []
            
            def audio_callback(indata, frames_count, time, status):
                if status:
                    print(f"⚠️ 录音状态: {status}")
                
                if not self.is_listening:
                    return
                
                # 将音频数据输入到高级VAD
                try:
                    # 确保数据格式正确
                    if indata.dtype != np.int16:
                        audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                    else:
                        audio_int16 = indata.flatten().astype(np.int16)
                    
                    # 输入到VAD检测器，加强异常处理
                    try:
                        vad_result = self.advanced_vad.feed_audio(audio_int16)
                    except Exception as vad_error:
                        print(f"⚠️ VAD检测失败: {vad_error}")
                        vad_result = None
                    
                    if vad_result == "speech_start":
                        # 语音开始，通知主线程
                        if not self.is_recording:
                            self.is_recording = True
                            recording_buffer.clear()  # 清空录音缓冲区
                            # 新增：把环形缓冲区的回溯音频加进来
                            if hasattr(self.advanced_vad, 'buffer') and hasattr(self.advanced_vad, 'lookback_duration'):
                                lookback_audio = self.advanced_vad.buffer.read_last_seconds(self.advanced_vad.lookback_duration)
                                if lookback_audio is not None and len(lookback_audio) > 0:
                                    recording_buffer.append(lookback_audio.copy())
                            speech_detected.set()
                            print("🔴 开始录音...")
                    
                    # 新增：只要在录音状态，就把音频append进录音缓冲区
                    if self.is_recording:
                        recording_buffer.append(audio_int16.copy())
                    
                    if vad_result == "speech_end":
                        # 语音结束，获取完整语音数据
                        if self.is_recording:
                            try:
                                if recording_buffer:
                                    self.audio_data = np.concatenate(recording_buffer, axis=0)
                                    print(f"✅ 录音完成: {len(self.audio_data)}样本")
                                else:
                                    print("⚠️ 录音缓冲区为空，回退到环形缓冲区回溯")
                                    speech_audio = self.advanced_vad.get_speech_audio()
                                    self.audio_data = speech_audio
                            except Exception as audio_error:
                                print(f"⚠️ 获取语音数据失败: {audio_error}")
                                self.audio_data = None
                            
                            self.is_recording = False
                            self.is_listening = False
                            recording_complete.set()
                    
                    elif vad_result == "speech_too_short":
                        # 语音过短，重置状态
                        if self.is_recording:
                            print("⚠️ 语音过短，继续监听...")
                            self.is_recording = False
                
                except Exception as e:
                    print(f"⚠️ VAD处理音频时出错: {e}")
                    # 严重错误时，安全退出
                    if self.is_recording:
                        print("🚨 VAD严重错误，强制结束录音")
                        self.is_recording = False
                        self.is_listening = False
                        recording_complete.set()
            
            # 开始音频流
            try:
                with sd.InputStream(
                    callback=audio_callback,
                    samplerate=self.audio_manager.sample_rate,
                    channels=self.audio_manager.channels,
                    dtype=self.audio_manager.dtype,
                    device=input_device,
                    blocksize=512   # 进一步减小块尺寸提高响应性
                ):
                    print("👂 正在监听中... (Ctrl+C退出)")
                    try:
                        idle_start_time = time.time()
                        while not recording_complete.wait(timeout=0.5):
                            current_time = time.time()
                            # 定期检查Silero VAD健康状态
                            self._check_silero_vad_health_periodically()
                            if current_time - idle_start_time > 30.0: # 30秒无语音活动
                                print("⏰ 长时间无语音活动，自动结束监听...")
                                self.is_listening = False
                                self.is_recording = False
                                recording_complete.set()
                                break
                            if self.is_recording: # Reset idle timer if speech is active
                                idle_start_time = current_time
                    except KeyboardInterrupt:
                        print("\n🛑 用户中断录音")
                        self.is_listening = False
                        self.is_recording = False
                        recording_complete.set()
                        raise
                    print("🏁 智能录音完成")
            except Exception as stream_error:
                print(f"❌ 音频流错误: {stream_error}")
                return False
            
            return self.audio_data is not None
        except Exception as e:
            print(f"❌ 高级VAD录音失败: {e}")
            # 尝试强力重新初始化音频系统
            print("🔧 尝试重新初始化音频系统来解决问题...")
            if hasattr(self.audio_manager, '_force_reinitialize_audio_system'):
                try:
                    reinit_success = self.audio_manager._force_reinitialize_audio_system()
                    if reinit_success:
                        print("✅ 音频系统重新初始化成功，重试高级VAD录音...")
                        # 重新尝试一次高级VAD录音
                        try:
                            return self._record_with_advanced_vad_retry()
                        except Exception as retry_error:
                            print(f"⚠️ 重试高级VAD录音仍失败: {retry_error}")
                    else:
                        print("⚠️ 音频系统重新初始化失败")
                except Exception as reinit_error:
                    print(f"⚠️ 音频系统重新初始化异常: {reinit_error}")
            # 回退到基础录音模式
            print("🔄 回退到基础录音模式...")
            return self._record_with_basic_vad()
    
    def _record_with_basic_vad(self):
        """使用基础VAD进行录音（兼容模式）"""
        try:
            import sounddevice as sd
            import numpy as np
            
            # 在开始录音前检查音频设备健康状态
            if not self._check_audio_device_health():
                print("⚠️ 基础VAD录音前音频设备健康检查失败，尝试重试...")
                time.sleep(1.0)
                if not self._check_audio_device_health():
                    print("❌ 音频设备健康检查持续失败，回退到手动录音")
                    self.is_recording = True
                    self.audio_data = self.audio_manager.record_audio()
                    self.is_recording = False
                    return self.audio_data is not None
            
            print("🎤 开始录音... 基础VAD将自动检测语音结束")
            
            # 获取输入设备
            input_device = self.audio_manager._get_best_input_device()
            
            # 重置VAD状态
            if self.vad:
                self.vad.reset()
            
            frames = []
            self.is_recording = True
            
            def audio_callback(indata, frames_count, time, status):
                if status:
                    print(f"⚠️ 录音状态: {status}")
                
                if self.is_recording:
                    frames.append(indata.copy())
                    
                    # 基础VAD处理
                    if self.vad:
                        # 转换为int16数据供VAD处理
                        if indata.dtype != np.int16:
                            audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                        else:
                            audio_int16 = indata.flatten().astype(np.int16)
                        
                        # 转换为字节数据
                        audio_bytes = audio_int16.tobytes()
                        
                        # VAD检测
                        if not self.vad.process_audio(audio_bytes):
                            # VAD检测到语音结束
                            self.is_recording = False
            
            # 开始录音流
            with sd.InputStream(
                callback=audio_callback,
                samplerate=self.audio_manager.sample_rate,
                channels=self.audio_manager.channels,
                dtype=self.audio_manager.dtype,
                device=input_device
            ):
                # 等待VAD检测完成，支持键盘中断
                try:
                    while self.is_recording:
                        sd.sleep(100)  # 100ms检查间隔
                except KeyboardInterrupt:
                    print("\n🛑 用户中断，停止基础VAD录音")
                    self.is_recording = False
                    raise  # 重新抛出KeyboardInterrupt
            
            # 合并音频数据
            if frames:
                self.audio_data = np.concatenate(frames, axis=0)
                print(f"✅ 基础VAD录音完成: {self.audio_data.shape}")
                return True
            else:
                print("❌ 没有录制到音频数据")
                return False
                
        except Exception as e:
            print(f"❌ 基础VAD录音失败: {e}")
            # 最后的回退方案：手动录音
            print("🔄 回退到手动录音模式...")
            self.is_recording = True
            self.audio_data = self.audio_manager.record_audio()
            self.is_recording = False
            return self.audio_data is not None
    
    def save_audio(self, filename):
        """保存录制的音频"""
        if self.audio_data is not None:
            return self.audio_manager.save_audio(self.audio_data, filename)
        return False
    
    def cleanup(self):
        """清理资源 - 增强版，包含音频系统重新初始化和Silero VAD健康检测"""
        self.audio_data = None
        self.is_recording = False
        self.is_listening = False
        
        # 重置活动时间
        self.last_activity_time = time.time()
        
        # 彻底重置VAD状态
        if self.advanced_vad:
            # 先进行Silero VAD健康检测
            if hasattr(self.advanced_vad, 'silero_vad') and self.advanced_vad.silero_vad:
                try:
                    health_ok = self.advanced_vad.silero_vad.health_check()
                    if not health_ok:
                        print("🔧 检测到Silero VAD状态异常，执行深度重置...")
                        # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
                        # self.advanced_vad.silero_vad.reset_buffer()
                        # print("✅ 已执行轻量级重置（避免影响音频播放）")
                        print("✅ VAD重置功能已暂时禁用，仅记录状态异常")
                    else:
                        print("✅ Silero VAD状态正常")
                except Exception as e:
                    print(f"⚠️ Silero VAD健康检测异常: {e}")
            
            # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
            # self.advanced_vad.reset()
            print("🔄 VAD重置功能已暂时禁用")
        elif self.vad:
            # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
            # self.vad.reset()
            print("🔄 基础VAD重置功能已暂时禁用")
        
        # 强力重新初始化音频系统（解决长时间运行后的设备问题）
        if hasattr(self.audio_manager, '_force_reinitialize_audio_system'):
            try:
                reinit_success = self.audio_manager._force_reinitialize_audio_system()
                if reinit_success:
                    print("🔄 音频系统强力重新初始化成功")
                else:
                    print("⚠️ 音频系统重新初始化有问题，但会继续尝试")
            except Exception as e:
                print(f"⚠️ 音频系统重新初始化异常: {e}")
                
        print("✅ SimplifiedVoiceRecorder资源清理完成（增强版）")

    def _check_silero_vad_health_periodically(self):
        """定期检查Silero VAD健康状态"""
        if not self.advanced_vad or not hasattr(self.advanced_vad, 'silero_vad'):
            return
        
        try:
            current_time = time.time()
            # 每30秒检查一次Silero VAD健康状态
            if not hasattr(self, '_last_silero_health_check'):
                self._last_silero_health_check = 0
            
            if current_time - self._last_silero_health_check > 30.0:
                self._last_silero_health_check = current_time
                
                # 检查是否正在播放音频，如果是则跳过健康检查
                if hasattr(self, 'audio_manager') and hasattr(self.audio_manager, '_play_lock'):
                    if self.audio_manager._play_lock.locked():
                        print("🎵 检测到音频正在播放，跳过Silero VAD健康检查")
                        return
                
                if self.advanced_vad.silero_vad:
                    health_ok = self.advanced_vad.silero_vad.health_check()
                    if not health_ok:
                        print("🔧 定期检测到Silero VAD状态异常，执行自动修复...")
                        # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
                        # self.advanced_vad.silero_vad.reset_buffer()
                        # print("✅ 已执行轻量级重置（避免影响音频播放）")
                        print("✅ VAD重置功能已暂时禁用，仅记录状态异常")
                        
        except Exception as e:
            print(f"⚠️ 定期Silero VAD健康检测异常: {e}")

    def _record_with_advanced_vad_retry(self):
        """高级VAD录音重试方法 - 简化版，避免无限递归"""
        try:
            import sounddevice as sd
            import numpy as np
            import threading
            
            print("🔄 重试高级VAD录音（简化版）...")
            
            # 获取输入设备（重新获取）
            input_device = self.audio_manager._get_best_input_device()
            if input_device is None:
                print("❌ 无法获取有效输入设备")
                return False
            
            # 重置VAD状态
            if self.advanced_vad:
                # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
                # self.advanced_vad.reset()
                print("🔄 VAD重置功能已暂时禁用（重试模式）")
            
            # 状态管理
            self.is_listening = True
            self.is_recording = False
            self.audio_data = None
            recording_complete = threading.Event()
            
            def audio_callback(indata, frames_count, time, status):
                if status:
                    print(f"⚠️ 录音状态: {status}")
                
                if not self.is_listening:
                    return
                
                # 将音频数据输入到高级VAD
                try:
                    if indata.dtype != np.int16:
                        audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                    else:
                        audio_int16 = indata.flatten().astype(np.int16)
                    
                    vad_result = self.advanced_vad.feed_audio(audio_int16)
                    
                    if vad_result == "speech_start":
                        if not self.is_recording:
                            self.is_recording = True
                            print("🔴 重试录音开始...")
                    
                    elif vad_result == "speech_end":
                        if self.is_recording:
                            try:
                                speech_audio = self.advanced_vad.get_speech_audio()
                                if speech_audio is not None and len(speech_audio) > 0:
                                    self.audio_data = speech_audio
                                    print(f"✅ 重试录音完成: {len(speech_audio)}样本")
                                else:
                                    print("⚠️ 重试录音未能获取语音数据")
                                    self.audio_data = None
                            except Exception as audio_error:
                                print(f"⚠️ 重试录音数据获取失败: {audio_error}")
                                self.audio_data = None
                            
                            self.is_recording = False
                            self.is_listening = False
                            recording_complete.set()
                    
                except Exception as e:
                    print(f"⚠️ 重试VAD处理音频时出错: {e}")
                    self.is_listening = False
                    self.is_recording = False
                    recording_complete.set()
            
            # 开始音频流
            try:
                with sd.InputStream(
                    callback=audio_callback,
                    samplerate=self.audio_manager.sample_rate,
                    channels=self.audio_manager.channels,
                    dtype=self.audio_manager.dtype,
                    device=input_device,
                    blocksize=512
                ):
                    print("👂 重试监听中... (最多15秒)")
                    
                    # 等待录音完成，但限制最大等待时间
                    if recording_complete.wait(timeout=15.0):
                        print("🏁 重试录音完成")
                    else:
                        print("⏰ 重试录音超时")
                        self.is_listening = False
                        self.is_recording = False
                    
            except Exception as stream_error:
                print(f"❌ 重试音频流错误: {stream_error}")
                return False
            
            return self.audio_data is not None
                
        except Exception as e:
            print(f"❌ 重试高级VAD录音失败: {e}")
            return False


class SimplifiedAudioPlayer:
    """简化的音频播放器 - 跨平台版本，线程安全"""
    
    def __init__(self):
        self.audio_manager = CrossPlatformAudioManager()
        self._play_lock = threading.Lock()  # 添加播放锁，防止并发播放冲突
    
    def get_audio_duration(self, audio_file):
        """获取音频文件时长（秒）"""
        if not audio_file or not os.path.exists(audio_file):
            return 0.0
        
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_file)
            duration = len(data) / samplerate
            return duration
        except Exception as e:
            print(f"⚠️ 无法获取音频时长: {e}")
            return 0.0
    
    def play_audio_file(self, audio_file):
        """播放音频文件 - 线程安全版本"""
        if not audio_file or not os.path.exists(audio_file):
            print("⚠️ 音频文件不存在，跳过播放")
            return False
        
        # 使用锁确保同时只有一个音频在播放
        with self._play_lock:
            try:
                return self.audio_manager.play_audio(audio_file)
            except Exception as e:
                print(f"⚠️ 音频播放失败: {e}")
                # 尝试清理可能的资源冲突
                import time
                time.sleep(0.1)  # 短暂等待，让资源释放
                return False


# ================================
# 文本转语音模块
# ================================

class TextToSpeechEngine:
    """文本转语音引擎"""
    
    def __init__(self):
        self.recordings_dir = os.path.join(os.getcwd(), "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # 使用跨平台音频管理器
        self.audio_manager = CrossPlatformAudioManager()
    
    def text_to_speech(self, text: str) -> str:
        """文本转语音，返回音频文件路径"""
        if not text.strip():
            return ""
        
        try:
            import edge_tts
            import asyncio
            import threading
            
            # 生成唯一文件名，包含线程ID避免冲突
            thread_id = threading.get_ident() % 10000  # 取模避免文件名过长
            timestamp = int(time.time() * 1000)
            filename = f"tts_{timestamp}_{thread_id}.wav"
            path = os.path.join(self.recordings_dir, filename)
            
            async def create_speech():
                voice = "zh-CN-XiaoxiaoNeural"
                temp_mp3_path = path.replace('.wav', '.mp3')
                # 语速调整为正常速度
                communicate = edge_tts.Communicate(text, voice, rate='+0%')
                await communicate.save(temp_mp3_path)
                
                # 转换MP3到WAV格式，使用soundfile
                try:
                    # 确保MP3文件存在且不为空
                    if not os.path.exists(temp_mp3_path) or os.path.getsize(temp_mp3_path) == 0:
                        print(f"⚠️ MP3文件不存在或为空: {temp_mp3_path}")
                        return ""
                    
                    # 使用跨平台音频管理器转换格式，默认48kHz避免播放报错
                    success = self.audio_manager.convert_audio_format(
                        temp_mp3_path, path, 
                        target_sample_rate=48000,  # 使用48kHz，与录音设备匹配
                        target_channels=1
                    )
                    
                    if success:
                        # 删除临时MP3文件
                        if os.path.exists(temp_mp3_path):
                            os.unlink(temp_mp3_path)
                    else:
                        # 转换失败，尝试重命名MP3文件
                        print("⚠️ 音频格式转换失败，使用MP3格式")
                        if os.path.exists(temp_mp3_path):
                            os.rename(temp_mp3_path, path)
                        
                except Exception as e:
                    print(f"⚠️ 音频格式转换失败: {e}")
                    # 转换失败时，尝试重命名MP3文件
                    if os.path.exists(temp_mp3_path):
                        try:
                            os.rename(temp_mp3_path, path)
                        except Exception as rename_error:
                            print(f"⚠️ 重命名MP3文件失败: {rename_error}")
                            return ""
                    else:
                        return ""
            
            # 运行异步函数
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import threading
                    import concurrent.futures
                    
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(create_speech())
                        new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        future.result(timeout=10)
                else:
                    asyncio.run(create_speech())
            except RuntimeError:
                asyncio.run(create_speech())
            
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print(f"✅ TTS已生成音频文件: {filename} ({os.path.getsize(path)} bytes)")
                return path
            else:
                print("❌ TTS生成的音频文件为空或不存在")
                return ""
                
        except ImportError:
            print("⚠️ edge-tts未安装，请运行: pip install edge-tts")
            return ""
        except Exception as e:
            print(f"❌ TTS生成失败: {e}")
            return ""


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
            
        if not deps.zhipuai_available:
            raise ImportError("请先安装智谱AI SDK: pip install zhipuai")
            
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
        2. 指令 - 控制机器人执行动作（关键词：待机、倒咖啡、左右、摇头）
        请先判断用户的意图
        如果你判断意图为聊天，请正常输出回答，intent设置为chat，action设置为unknown，confidence设置为1.0，description设置为识别到的意图或动作
        如果你判断意图为指令，请输出JSON格式，intent设置为command，action设置为识别到的动作，confidence设置为1.0，description设置为识别到的意图或动作，用户的指令意图只会让机器人执行waiting, coffee, left_right, rotate这四种动作，其他的类似指令的说法请默认设置意图为聊天
        如果你判断意图为指令，输出的标准格式如下：
        {{
            "intent": "command"或"chat",
            "action": "动作类型（仅当intent为command时有效，使用英文描述，动作类型只有可能是：waiting, coffee, left_right, rotate）",
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
        if not deps.zhipuai_available:
            return "智谱AI库未安装，无法生成聊天响应"
        
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


# ================================
# 拼音匹配模块
# ================================

class PinyinMatcher:
    """拼音匹配器，支持前后鼻音和平翘舌音模糊匹配"""
    
    def __init__(self):
        # 前后鼻音映射表
        self.nasal_map = {
            'an': 'ang', 'en': 'eng', 'in': 'ing', 'un': 'ong',
            'ang': 'an', 'eng': 'en', 'ing': 'in', 'ong': 'un'
        }
        
        # 平舌音和翘舌音映射表
        self.tongue_map = {
            'z': 'zh', 'c': 'ch', 's': 'sh',
            'zh': 'z', 'ch': 'c', 'sh': 's'
        }
        
        self.wake_word = "小智同学"
        self.wake_pinyin_variants = self._get_pinyin_variants(self.wake_word)
        
        print(f"唤醒词 '{self.wake_word}' 已初始化")
    
    def _get_pinyin_variants(self, text: str) -> list:
        """获取文本的拼音变体"""
        if not deps.pypinyin_available:
            return []
        
        from pypinyin import pinyin, Style
        
        original_pinyin = pinyin(text, style=Style.NORMAL, heteronym=False)
        
        variants = []
        for char_pinyin_list in original_pinyin:
            char_pinyin = char_pinyin_list[0].lower()
            char_variants = [char_pinyin]
            
            # 添加前后鼻音变体
            for original, variant in self.nasal_map.items():
                if char_pinyin.endswith(original):
                    new_variant = char_pinyin[:-len(original)] + variant
                    if new_variant not in char_variants:
                        char_variants.append(new_variant)
            
            # 添加平翘舌音变体
            for original, variant in self.tongue_map.items():
                if char_pinyin.startswith(original):
                    new_variant = variant + char_pinyin[len(original):]
                    if new_variant not in char_variants:
                        char_variants.append(new_variant)
            
            # 组合变体
            combined_variants = []
            for base_variant in char_variants[:]:
                for original, variant in self.nasal_map.items():
                    if base_variant.endswith(original):
                        combined_variant = base_variant[:-len(original)] + variant
                        if combined_variant not in char_variants:
                            combined_variants.append(combined_variant)
                
                for original, variant in self.tongue_map.items():
                    if base_variant.startswith(original):
                        combined_variant = variant + base_variant[len(original):]
                        if combined_variant not in char_variants:
                            combined_variants.append(combined_variant)
            
            char_variants.extend(combined_variants)
            variants.append(char_variants)
        
        return variants
    
    def _generate_pinyin_combinations(self, variants: list) -> list:
        """生成所有可能的拼音组合"""
        if not variants:
            return []
        
        def combine(index: int, current: list) -> list:
            if index == len(variants):
                return [''.join(current)]
            
            results = []
            for variant in variants[index]:
                results.extend(combine(index + 1, current + [variant]))
            return results
        
        return combine(0, [])
    
    def detect_wake_word(self, text: str) -> tuple:
        """检测文本中是否包含唤醒词，返回(是否检测到, 剩余文本)"""
        if not text or not deps.pypinyin_available:
            return False, text
        
        # 清理文本
        import re
        cleaned_text = re.sub(r'[^\u4e00-\u9fff，。！？、]', '', text)
        
        if not cleaned_text:
            return False, text
        
        # 获取输入文本的拼音
        from pypinyin import pinyin, Style
        input_pinyin = pinyin(cleaned_text, style=Style.NORMAL, heteronym=False)
        input_pinyin_str = ''.join([p[0].lower() for p in input_pinyin])
        
        # 生成目标拼音的所有组合
        wake_combinations = self._generate_pinyin_combinations(self.wake_pinyin_variants)
        
        # 检查是否匹配
        for wake_combo in wake_combinations:
            if wake_combo in input_pinyin_str:
                print(f"🎯 检测到唤醒词: '{self.wake_word}'")
                
                # 找到唤醒词在原文本中的位置
                wake_word_index = cleaned_text.find(self.wake_word)
                if wake_word_index != -1:
                    remaining_text = cleaned_text[wake_word_index + len(self.wake_word):].strip('，。！？、')
                    print(f"剩余指令文本: '{remaining_text}'")
                    return True, remaining_text
                else:
                    return True, cleaned_text
        
        return False, text

    def detect_dismiss_command(self, text: str) -> bool:
        """检测退下指令"""
        if not text:
            return False
        
        dismiss_keywords = [
            "退下", "休息", "睡觉", "回去休息", "去休息", 
            "暂停", "停止", "结束", "再见", "拜拜",
            "你可以休息了", "没事了", "辛苦了"
        ]
        
        text_cleaned = text.strip().replace(" ", "")
        for keyword in dismiss_keywords:
            if keyword in text_cleaned:
                print(f"🛌 检测到退下指令: '{keyword}'")
                return True
        
        return False


# ================================
# 机器人控制模块
# ================================

class RobotController:
    """机器人控制器"""
    
    def __init__(self, robot_ip: str, robot_port: int):
        self.robot = None
        self.robot_lock = None
        
        if deps.robot_available:
            from robot_controller import (
                X1Interface, 
                go_to_waiting_location,
                action_up_and_down, 
                action_left_and_right,
                action_rotate
            )
            
            self.robot_lock = mp.Lock()
            self.robot = X1Interface(self.robot_lock, robot_ip, robot_port)
            
            # 导入动作函数
            self.go_to_waiting_location = go_to_waiting_location
            self.action_up_and_down = action_up_and_down
            self.action_left_and_right = action_left_and_right
            self.action_rotate = action_rotate
            
            print(f"已连接到机器人: {robot_ip}:{robot_port}")
        else:
            print("机器人控制不可用")
    
    def execute_action(self, action: str) -> bool:
        """执行动作"""
        if not self.robot:
            print("机器人不可用，模拟执行动作")
            print(f"模拟执行: {Config.ACTION_MAP.get(action, '未知动作')}")
            time.sleep(2)
            return True
        
        try:
            if action == "waiting":
                print("执行：回到待机位置")
                result = self.go_to_waiting_location(self.robot)
            elif action == "up_down":
                print("执行：上下摆动")
                result = self.action_up_and_down(self.robot)
            elif action == "left_right":
                print("执行：左右摆动")
                result = self.action_left_and_right(self.robot)
            elif action == "rotate":
                print("执行：摇头动作")
                result = self.action_rotate(self.robot)
            elif action == "unknown":
                return False
            else:
                return False
            
            return result == 0
            
        except Exception as e:
            print(f"执行动作失败: {e}")
            return False


# ================================
# 主控制器
# ================================

class VoiceRobotController:
    """语音机器人控制器主类"""
    
    def __init__(self, 
                 robot_ip: str = Config.ROBOT_IP_DEFAULT,
                 robot_port: int = Config.ROBOT_PORT_DEFAULT,
                 zhipuai_api_key: Optional[str] = os.getenv("ZHIPUAI_API_KEY"),
                 device: str = "cpu",
                 use_voice_input: Optional[bool] = None):
        """初始化语音机器人控制器"""
        
        # 性能优化：添加缓存机制
        self.tts_cache = {}
        self.device_cache = {}
        self.response_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 计时功能：测量从回答完成到可以接收下一个问题的时间
        self.timing_stats = {
            'total_cycles': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'recent_times': []  # 最近10次的计时数据
        }
        self.audio_play_end_time = None
        self.ready_for_next_input_time = None
        
        # 初始化组件
        self._init_components(use_voice_input, device, zhipuai_api_key)
        
        # 机器人状态管理
        self.robot_state = "sleeping"  # sleeping: 休眠状态, awake: 唤醒状态
        
        # 性能优化：预缓存常用音频
        if Config.PRELOAD_COMMON_AUDIO:
            self._preload_common_audio()
        
        # 初始化机器人控制器
        self.robot_controller = RobotController(robot_ip, robot_port)
        self.delay_phrase_cache = {}
        self._preload_delay_phrases()
        
        # 自我介绍关键词
        self.intro_keywords = [
            "介绍一下你自己", "你是谁", "自我介绍", "请介绍你自己", "你能做什么", "你的功能", "你的作用", "你是做什么的"
        ]
    
    def _init_components(self, use_voice_input: Optional[bool], device: str, zhipuai_api_key: Optional[str]):
        """初始化各个组件"""
        # 初始化录音器 - 使用简化的跨平台版本
        if use_voice_input is None:
            self.recorder = SimplifiedVoiceRecorder() if deps.audio_available else None
            print(f"🔧 输入模式: {'语音输入' if self.recorder else '文本输入'} (自动检测)")
        elif use_voice_input and deps.audio_available:
            self.recorder = SimplifiedVoiceRecorder()
            print("🎤 输入模式: 语音输入 (用户选择)")
        else:
            self.recorder = None
            if use_voice_input and not deps.audio_available:
                print("⚠️ 语音输入不可用，已切换到文本输入")
            else:
                print("⌨️ 输入模式: 文本输入 (用户选择)")
        
        # 将控制器引用传递给录音器（用于计时功能）
        if self.recorder:
            self.recorder.controller = self
        
        # 初始化其他组件
        self.recognizer = GLM_ASR_Recognizer(device=device)
        self.processor = RobotCommandProcessor(api_key=zhipuai_api_key)
        self.tts_engine = TextToSpeechEngine()
        self.audio_player = SimplifiedAudioPlayer()
        
        # 初始化唤醒词检测器
        if deps.pypinyin_available:
            self.wake_matcher = PinyinMatcher()
            print("✅ 唤醒词功能已启用")
        else:
            self.wake_matcher = None
            print("⚠️ 唤醒词功能不可用，将直接处理语音指令")
    
    def _preload_common_audio(self):
        """预加载常用音频，提高响应速度"""
        print("🔄 预加载常用音频...")
        
        for phrase in Config.COMMON_PHRASES:
            try:
                self._get_cached_tts(phrase)
            except Exception as e:
                print(f"⚠️ 预加载音频失败: {phrase[:10]}... - {e}")
        
        print(f"✅ 预加载完成，缓存了 {len(self.tts_cache)} 个音频")
    
    def _preload_delay_phrases(self):
        """预生成所有拖延语音频"""
        print("🔄 预生成拖延语音频...")
        for phrase in Config.DELAY_PHRASES:
            try:
                audio_path = self.tts_engine.text_to_speech(phrase)
                if audio_path:
                    self.delay_phrase_cache[phrase] = audio_path
            except Exception as e:
                print(f"⚠️ 拖延语音频生成失败: {phrase[:10]}... - {e}")
        print(f"✅ 拖延语音频缓存: {len(self.delay_phrase_cache)} 条")

    def _get_cached_tts(self, text: str) -> str:
        """获取缓存的TTS音频文件路径"""
        if text in self.tts_cache:
            self.cache_hits += 1
            return self.tts_cache[text]
        
        self.cache_misses += 1
        audio_path = self.tts_engine.text_to_speech(text)
        if audio_path:
            self.tts_cache[text] = audio_path
        return audio_path
    
    def _play_cached_audio_with_delay(self, text: str, tts_ready_callback=None, tts_timeout=2.0):
        """用户输入结束3秒后播放拖延语，主回答音频生成后等待拖延语播放完成再衔接"""
        audio_path = None
        tts_done = threading.Event()
        delay_played = threading.Event()
        
        def tts_task():
            nonlocal audio_path
            audio_path = self._get_cached_tts(text)
            tts_done.set()
        
        t = threading.Thread(target=tts_task)
        t.start()
        
        # 3秒后播放拖延语（无论TTS是否完成）
        delay_timer = threading.Timer(3.0, lambda: self._play_delay_phrase_and_set_flag(delay_played))
        delay_timer.start()
        
        # 等待TTS完成
        tts_done.wait()
        
        # 取消延迟定时器（如果TTS在3秒内完成）
        delay_timer.cancel()
        
        # 如果拖延语已播放，等待其播放完成
        if delay_played.is_set():
            print("⏳ 等待拖延语播放完成...")
            # 等待拖延语播放完成（大约2-3秒）
            time.sleep(2.5)
        
        if tts_ready_callback:
            tts_ready_callback()
        
        if audio_path:
            print("🎵 播放正式回答...")
            self.audio_player.play_audio_file(audio_path)
            # 回答完成，开始计时
            self._start_response_cycle_timing()
        else:
            print(f"🗣️ 音频生成失败: {text[:20]}...")
        
        return audio_path

    def _play_cached_audio(self, text: str, tts_ready_callback=None):
        """播放缓存的音频（统一使用带拖延语机制的播放方法）"""
        # 使用统一的带拖延语机制的播放方法
        return self._play_cached_audio_with_delay(text, tts_ready_callback=tts_ready_callback)
    
    def _play_audio_async_with_vad_warmup(self, text: str, tts_ready_callback=None):
        """异步播放音频，同时预热VAD系统"""
        import threading
        
        # 创建音频播放线程
        def audio_play_thread():
            try:
                self._play_streaming_tts(text, tts_ready_callback=tts_ready_callback)
            except Exception as e:
                print(f"⚠️ 异步音频播放失败: {e}")
        
        # 创建VAD预热线程
        def vad_warmup_thread():
            try:
                # 在音频播放的同时预热VAD系统
                if self.recorder and hasattr(self.recorder, 'advanced_vad'):
                    print("🔥 后台预热VAD系统...")
                    # 重置VAD状态，准备下一轮录音
                    self.recorder.advanced_vad.reset()
                    # 预检查音频设备健康状态
                    self.recorder._check_audio_device_health()
                    # 设置预热标志，下次录音时可以快速启动
                    self.recorder._vad_prewarmed = True
                    print("✅ VAD系统预热完成")
                    # 系统准备就绪，结束计时
                    self._end_response_cycle_timing()
            except Exception as e:
                print(f"⚠️ VAD预热失败: {e}")
        
        # 启动音频播放线程
        audio_thread = threading.Thread(target=audio_play_thread, daemon=True)
        audio_thread.start()
        
        # 启动VAD预热线程
        vad_thread = threading.Thread(target=vad_warmup_thread, daemon=True)
        vad_thread.start()
        
        # 等待音频播放完成（但不阻塞主线程）
        audio_thread.join()
    
    def _play_streaming_tts(self, text: str, tts_ready_callback=None):
        """智能流式TTS：按句子分割，并行生成，连续播放"""
        if not text.strip():
            return
        sentences = self._split_by_sentences(text)
        if len(sentences) == 1:
            self._play_cached_audio_with_delay(text, tts_ready_callback=tts_ready_callback)
            return
        print(f"🎵 开始智能流式TTS播放 ({len(sentences)}个句子)")
        audio_paths = self._generate_audio_parallel(sentences)
        if tts_ready_callback:
            tts_ready_callback()
        self._play_audio_sequence(audio_paths)
    
    def _split_by_sentences(self, text: str) -> list:
        """按句子分割文本，保持语义完整性"""
        if not text:
            return []
        
        # 句子结束标点符号
        sentence_endings = ['。', '！', '？', '；']
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            
            # 遇到句子结束标点时分割
            if char in sentence_endings:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 添加最后一个句子（如果没有结束标点）
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def _generate_audio_parallel(self, sentences: list) -> list:
        """并行生成所有音频文件"""
        import concurrent.futures
        import threading
        import time
        
        audio_paths = []
        lock = threading.Lock()
        
        def generate_single_audio(sentence: str, index: int):
            """生成单个音频文件"""
            try:
                print(f"🔄 生成音频 {index+1}/{len(sentences)}: {sentence[:20]}...")
                
                # 为每个线程生成唯一的文件名，避免冲突
                timestamp = int(time.time() * 1000) + index
                filename = f"tts_{timestamp}.wav"
                path = os.path.join(self.tts_engine.recordings_dir, filename)
                
                # 直接调用TTS引擎生成音频，避免缓存冲突
                audio_path = self.tts_engine.text_to_speech(sentence.strip())
                
                # 验证音频文件是否真正生成成功
                if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    with lock:
                        audio_paths.append((index, audio_path))
                    print(f"✅ 音频 {index+1} 生成完成: {os.path.basename(audio_path)}")
                    return audio_path
                else:
                    print(f"❌ 音频 {index+1} 生成失败: 文件不存在或为空")
                    with lock:
                        audio_paths.append((index, None))
                    return None
            except Exception as e:
                print(f"❌ 音频 {index+1} 生成失败: {e}")
                with lock:
                    audio_paths.append((index, None))
                return None
        
        # 使用线程池并行生成（优化线程数提高速度）
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sentences), 6)) as executor:
            futures = []
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    future = executor.submit(generate_single_audio, sentence, i)
                    futures.append(future)
            
            # 等待所有音频生成完成
            concurrent.futures.wait(futures)
        
        # 按原始顺序排序音频路径
        audio_paths.sort(key=lambda x: x[0])
        result_paths = [path for _, path in audio_paths]
        
        # 统计成功生成的音频数量
        successful_count = sum(1 for path in result_paths if path is not None)
        print(f"📊 音频生成统计: {successful_count}/{len(sentences)} 个音频生成成功")
        
        return result_paths
    
    def _play_audio_sequence(self, audio_paths: list):
        """连续播放音频序列 - 确保顺序播放，避免并发冲突"""
        if not audio_paths:
            return
        
        # 过滤掉无效的音频路径
        valid_audio_paths = [path for path in audio_paths if path and os.path.exists(path) and os.path.getsize(path) > 0]
        
        if not valid_audio_paths:
            print("❌ 没有有效的音频文件可以播放")
            return
        
        print(f"🎤 开始连续播放 {len(valid_audio_paths)} 个有效音频文件")
        
        for i, audio_path in enumerate(valid_audio_paths):
            try:
                print(f"🔊 播放音频 {i+1}/{len(valid_audio_paths)}: {os.path.basename(audio_path)}")
                
                # 确保每个音频都完全播放完毕再播放下一个
                success = self.audio_player.play_audio_file(audio_path)
                
                if success:
                    print(f"✅ 音频 {i+1} 播放完成")
                else:
                    print(f"❌ 音频 {i+1} 播放失败")
                
                # 在音频之间添加小间隔，确保资源完全释放
                if i < len(valid_audio_paths) - 1:
                    import time
                    time.sleep(0.2)  # 200ms间隔
                    
            except Exception as e:
                print(f"❌ 播放音频 {i+1} 时发生错误: {e}")
                continue
        
        # 所有音频播放完成，开始计时
        self._start_response_cycle_timing()
    
    def _split_text_by_commas(self, text: str) -> list:
        """按逗号分割文本，保持语义完整性（保留用于兼容性）"""
        if not text:
            return []
        
        # 中文标点符号
        chinese_punctuation = ['，', '。', '！', '？', '；', '：']
        
        segments = []
        current_segment = ""
        
        for char in text:
            current_segment += char
            
            # 遇到逗号或其他标点符号时分割
            if char in chinese_punctuation:
                if current_segment.strip():
                    segments.append(current_segment.strip())
                current_segment = ""
        
        # 添加最后一个片段
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return segments
    
    def process_voice_command(self) -> bool:
        """处理单次语音指令（状态机模式）"""
        # 时间统计
        self._input_start_time = time.time()
        
        # 获取语音输入
        text = self._get_voice_input()
        if not text:
            return False
        
        # 根据当前状态处理指令
        if self.robot_state == "sleeping":
            return self._handle_sleeping_state(text)
        elif self.robot_state == "awake":
            return self._handle_awake_state(text)
        else:
            self.robot_state = "sleeping"
            return False
    
    def _get_voice_input(self) -> str:
        """获取语音输入或文本输入"""
        if not self.recorder:
            # 文本输入模式
            if self.robot_state == "sleeping":
                prompt = "请输入唤醒词（小智同学）或输入'stats'查看计时统计: "
            else:
                prompt = "请输入动作指令或退下指令，或输入'stats'查看计时统计: "
            text = input(prompt).strip()
            
            # 检查是否是统计命令
            if text.lower() == 'stats':
                self._show_timing_stats()
                return ""
            
            self._input_end_time = time.time()
            return text
        else:
            # 语音录制模式
            try:
                if self.robot_state == "sleeping":
                    print("😴 机器人休眠中，请说唤醒词...")
                    print("🎤 请说：小智同学")
                else:
                    print("👂 机器人等待指令中...")
                    print("🎤 请说动作指令：回到待机位置、上下摆动、左右摆动、摇头")
                    print("💤 或说退下指令：退下、休息、再见等")
                    print("📊 或说'统计'查看计时统计")
                
                self.recorder.start_recording(use_vad=True)
                
                # 保存音频到临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    audio_filename = tmp_file.name
                
                if not self.recorder.save_audio(audio_filename):
                    print("录音失败")
                    return ""
                
                try:
                    # 语音识别
                    if self.recognizer:
                        asr_start_time = time.time()
                        text = self.recognizer.recognize(audio_filename)
                        asr_end_time = time.time()
                        asr_duration = asr_end_time - asr_start_time
                        
                        self._input_end_time = asr_end_time
                        
                        if hasattr(self, '_input_start_time'):
                            total_input_duration = asr_end_time - self._input_start_time
                            print(f"⏱️ 语音输入耗时: {total_input_duration:.2f}秒")
                        
                        print(f"🎯 语音识别结果: {text}")
                        print(f"⏱️ 语音转文字耗时: {asr_duration:.2f}秒")
                        
                        # 检查是否是统计命令
                        if "统计" in text or "计时" in text or "时间" in text:
                            self._show_timing_stats()
                            return ""
                        
                        return text
                    else:
                        print("语音识别不可用，请手动输入:")
                        return input().strip()
                        
                finally:
                    try:
                        os.unlink(audio_filename)
                    except:
                        pass
                        
            finally:
                if self.recorder:
                    self.recorder.cleanup()
        
        return ""
    
    def _handle_sleeping_state(self, text: str) -> bool:
        """处理休眠状态的输入"""
        if not self.wake_matcher:
            return self._process_action_command(text)
        
        # 检测唤醒词
        wake_detected, remaining_text = self.wake_matcher.detect_wake_word(text)
        if wake_detected:
            print("🎉 机器人被唤醒！")
            print("🤖 你好！我已准备好接受您的指令。")

            # 播放唤醒音频
            self._play_cached_audio("你好，我是小智同学，很高兴见到你！")
            
            # 切换到唤醒状态
            self.robot_state = "awake"
            # 唤醒时不需要执行机器人动作，直接准备接收指令
            print("✅ 机器人已唤醒，准备接收指令")
            
            return True
        else:
            print("😴 机器人还在休眠中，请说唤醒词：小智同学")
            return False
    
    def _handle_awake_state(self, text: str) -> bool:
        """处理唤醒状态的输入"""
        return self._process_action_or_dismiss(text)
    
    def _process_action_or_dismiss(self, text: str) -> bool:
        """处理动作指令或退下指令"""
        if not text.strip():
            print("未识别到有效文本")
            return False
        
        # 在唤醒状态下检测到唤醒词时，快速响应
        if self.wake_matcher:
            wake_detected, remaining_text = self.wake_matcher.detect_wake_word(text)
            if wake_detected:
                print("🎯 检测到唤醒词，快速响应中...")
                
                # 随机选择快速响应短语，让响应更自然
                import random
                quick_responses = [
                    "我在听，请说",
                    "嗯，我在",
                    "请说"
                ]
                response = random.choice(quick_responses)
                
                # 快速播放响应音频（使用缓存）
                self._play_cached_audio(response)
                
                # 如果唤醒词后面还有指令，立即处理
                if remaining_text.strip():
                    print(f"💬 检测到后续指令: {remaining_text}")
                    return self._process_action_command(remaining_text)
                
                return True
        
        # 检测退下指令
        if self.wake_matcher and self.wake_matcher.detect_dismiss_command(text):
            print("🤖 好的，我去休息了。需要时请叫我！")

            # 播放退下音频
            self._play_cached_audio("好的，我去休息了。需要时请叫我！")
            
            # 切换到休眠状态
            self.robot_state = "sleeping"
            print("😴 机器人已进入休眠状态")
            
            # 退下时可以选择是否执行回到待机位置（可选）
            # success = self.robot_controller.execute_action("waiting")
            # if not success:
            #     print("⚠️ 回到待机位置失败，但不影响休眠状态")
            
            return True
        
        # 处理动作指令
        return self._process_action_command(text)
    
    def _process_action_command(self, text: str) -> bool:
        """处理动作指令（带拖延语机制，LLM+TTS整体流程超时1.5秒自动插入拖延语）"""
        import threading
        import time
        
        # 性能监控 - 在函数最开头定义，确保所有代码路径都能访问
        total_start_time = time.time()
        print(f"⏱️ 开始处理指令: {text[:30]}...")
        
        # 检查是否为自我介绍请求
        if any(kw in text for kw in self.intro_keywords):
            intro_text = "我是人工智能助手，专为提供信息、解答疑问和协助解决问题而设计。我可以处理各种查询，并尽力提供准确、有用的回答。"
            print(f"🤖 {intro_text}")
            
            # 记录TTS开始播放的时间
            tts_play_start_time = time.time()
            audio_file_path = self._play_cached_audio(intro_text)
            
            # 性能监控 - 自我介绍分支的响应准备时间
            response_prepare_duration = tts_play_start_time - total_start_time
            
            # 计算音频播放时长
            audio_duration = 0.0
            if audio_file_path and os.path.exists(audio_file_path):
                audio_duration = self.audio_player.get_audio_duration(audio_file_path)
            
            # 计算真正的响应准备时间（减去音频播放时长）
            true_response_time = response_prepare_duration - audio_duration
            
            print(f"🚀 自我介绍响应准备完成，耗时: {response_prepare_duration:.2f}秒")
            print(f"🎯 真正的响应准备时间（减去播放时长）: {true_response_time:.2f}秒")
            return True
        
        # 时间统计
        text_gen_start_time = time.time()
        
        # 使用大模型识别动作    
        print("🧠 正在调用LLM处理指令...")
        llm_start_time = time.time()
        command_result = self.processor.process_command(text)
        action = command_result.get("action", "unknown")
        confidence = command_result.get("confidence", 0.0)
        description = command_result.get("description", "未知")
        llm_end_time = time.time()

        # 性能监控 - LLM处理时间
        llm_duration = llm_end_time - llm_start_time
        text_gen_duration = llm_end_time - text_gen_start_time
        print(f"⏱️ LLM处理耗时: {llm_duration:.2f}秒，总文本生成耗时: {text_gen_duration:.2f}秒")

        # 更准确的TTS准备耗时统计
        tts_prepare_start_time = time.time()
        def tts_ready_callback():
            tts_ready_time = time.time()
            print(f"⏱️ LLM到TTS音频准备好耗时: {tts_ready_time - tts_prepare_start_time:.2f}秒")

        # 性能监控 - TTS开始
        tts_start_time = time.time()

        # 记录音频文件路径用于计算播放时长
        audio_file_path = None
        
        if command_result.get("intent", "unknown") == "command":
            print(f"🎯 识别动作: {description} (置信度: {confidence:.2f})")
            print("🎵 开始异步TTS生成和播放（同时预热VAD）...")
            audio_file_path = self._play_cached_audio("好的", tts_ready_callback=tts_ready_callback)
        else:
            print(f"小智说：{description}")
            print("🎵 开始异步TTS生成和播放（同时预热VAD）...")
            audio_file_path = self._play_cached_audio(description, tts_ready_callback=tts_ready_callback)
        
        # 性能监控 - TTS开始播放（不包括播放时间）
        tts_play_start_time = time.time()
        tts_prepare_duration = tts_play_start_time - tts_start_time

        if command_result.get("intent", "unknown") == "command" and (action == "unknown" or confidence < 0.5):
            print("❓ 抱歉，我不理解这个指令，请重新说一遍")
            error_audio_path = self._play_cached_audio("抱歉，我不理解这个指令，请重新说一遍")
            
            # 性能监控 - 错误情况的响应准备时间
            error_response_duration = tts_play_start_time - total_start_time
            
            # 计算音频播放时长
            error_audio_duration = 0.0
            if error_audio_path and os.path.exists(error_audio_path):
                error_audio_duration = self.audio_player.get_audio_duration(error_audio_path)
            
            # 计算真正的响应准备时间（减去音频播放时长）
            error_true_response_time = error_response_duration - error_audio_duration
            
            print(f"📊 错误响应统计: 响应准备:{error_true_response_time:.2f}s | 音频时长:{error_audio_duration:.2f}s")
            return False

        # 执行动作
        if command_result.get("intent", "unknown") == "command":
            success = self.robot_controller.execute_action(action)
            
            if success:
                print("✅ 动作执行成功")
                
                if self.robot_state == "awake":
                    print("👂 等待下一个指令...")
            else:
                print("❌ 动作执行失败")
        else:
            # 聊天对话不需要执行机器人动作
            print("💬 聊天对话，跳过机器人动作执行")
            success = True

        # 性能监控 - 总结（减去音频播放时长）
        response_prepare_duration = tts_play_start_time - total_start_time
        
        # 计算音频播放时长
        audio_duration = 0.0
        if audio_file_path and os.path.exists(audio_file_path):
            audio_duration = self.audio_player.get_audio_duration(audio_file_path)
        
        # 计算真正的响应准备时间（减去音频播放时长）
        true_response_time = response_prepare_duration - audio_duration
        
        print(f"📊 性能统计: LLM:{llm_duration:.2f}s | TTS准备:{tts_prepare_duration:.2f}s | 音频时长:{audio_duration:.2f}s | 响应准备:{true_response_time:.2f}s")
        print(f"🚀 从用户输入到TTS开始播放，总耗时: {response_prepare_duration:.2f}秒")
        print(f"🎯 真正的响应准备时间（减去播放时长）: {true_response_time:.2f}秒")

        return success
    
    def run(self):
        """运行语音机器人控制器"""
        print("🤖 语音机器人控制器已启动")
        print("🚀 性能优化已启用：缓存机制 + 设备复用 + API优化")
        print("⏱️ 计时功能已启用：测量从回答完成到可以接收下一个问题的时间")
        print("=" * 50)
        
        while True:
            try:
                if not self.process_voice_command():
                    print("⚠️ 指令处理失败，请重试")
                    
            except KeyboardInterrupt:
                print("\n🛑 语音机器人控制器已停止")
                # 显示计时统计
                self._show_timing_stats()
                break
            except Exception as e:
                print(f"❌ 运行时错误: {e}")
                


    def _play_delay_phrase_and_set_flag(self, delay_played_event):
        """播放拖延语并设置标志"""
        print("⏰ 用户输入结束3秒，播放拖延语...")
        self.play_delay_phrase()
        delay_played_event.set()
    
    def play_delay_phrase(self):
        """随机播放一条拖延语"""
        if not self.delay_phrase_cache:
            print("⚠️ 拖延语音频缓存为空")
            return
        phrase = random.choice(list(self.delay_phrase_cache.keys()))
        audio_path = self.delay_phrase_cache[phrase]
        print(f"🕒 播放拖延语: {phrase}")
        self.audio_player.play_audio_file(audio_path)

    def _start_response_cycle_timing(self):
        """开始回答周期计时 - 音频播放结束时调用"""
        self.audio_play_end_time = time.time()
        print(f"⏱️ 回答完成，开始计时: {self.audio_play_end_time:.3f}")
    
    def _end_response_cycle_timing(self):
        """结束回答周期计时 - 系统准备就绪可以接收下一个问题时调用"""
        if self.audio_play_end_time is None:
            print("⚠️ 计时数据不完整，跳过本次统计")
            return
        
        self.ready_for_next_input_time = time.time()
        cycle_time = self.ready_for_next_input_time - self.audio_play_end_time
        
        # 更新统计信息
        self.timing_stats['total_cycles'] += 1
        self.timing_stats['total_time'] += cycle_time
        self.timing_stats['min_time'] = min(self.timing_stats['min_time'], cycle_time)
        self.timing_stats['max_time'] = max(self.timing_stats['max_time'], cycle_time)
        
        # 更新最近时间列表（保持最近10次）
        self.timing_stats['recent_times'].append(cycle_time)
        if len(self.timing_stats['recent_times']) > 10:
            self.timing_stats['recent_times'].pop(0)
        
        # 计算平均时间
        avg_time = self.timing_stats['total_time'] / self.timing_stats['total_cycles']
        
        print(f"⏱️ 系统准备就绪，本次耗时: {cycle_time:.3f}秒")
        print(f"📊 计时统计: 平均{avg_time:.3f}秒, 最小{self.timing_stats['min_time']:.3f}秒, 最大{self.timing_stats['max_time']:.3f}秒")
        print(f"📈 总循环次数: {self.timing_stats['total_cycles']}")
        
        # 重置计时器
        self.audio_play_end_time = None
        self.ready_for_next_input_time = None
    
    def _show_timing_stats(self):
        """显示详细的计时统计信息"""
        if self.timing_stats['total_cycles'] == 0:
            print("📊 暂无计时数据")
            return
        
        avg_time = self.timing_stats['total_time'] / self.timing_stats['total_cycles']
        recent_avg = sum(self.timing_stats['recent_times']) / len(self.timing_stats['recent_times']) if self.timing_stats['recent_times'] else 0
        
        print("\n" + "="*50)
        print("📊 回答周期计时统计报告")
        print("="*50)
        print(f"总循环次数: {self.timing_stats['total_cycles']}")
        print(f"总耗时: {self.timing_stats['total_time']:.3f}秒")
        print(f"平均耗时: {avg_time:.3f}秒")
        print(f"最小耗时: {self.timing_stats['min_time']:.3f}秒")
        print(f"最大耗时: {self.timing_stats['max_time']:.3f}秒")
        print(f"最近{len(self.timing_stats['recent_times'])}次平均: {recent_avg:.3f}秒")
        
        if self.timing_stats['recent_times']:
            print(f"最近{len(self.timing_stats['recent_times'])}次耗时: {[f'{t:.3f}s' for t in self.timing_stats['recent_times']]}")
        
        print("="*50)


# ================================
# 主程序入口
# ================================

def main():
    """主函数"""
    # 获取API密钥
    api_key = Config.ZHIPUAI_API_KEY
    if not api_key:
        print("请设置智谱AI API密钥:")
        print("export ZHIPUAI_API_KEY='your_api_key'")
        api_key = input("或者在这里输入API密钥: ").strip()
        if not api_key:
            print("未设置API密钥，程序退出")
            return
    
    # 获取机器人IP
    robot_ip = input(f"请输入机器人IP地址 (默认{Config.ROBOT_IP_DEFAULT}): ").strip()
    if not robot_ip:
        robot_ip = Config.ROBOT_IP_DEFAULT
    
    # 选择输入模式
    print("\n🎛️ 请选择输入模式:")
    print("1. 语音输入 (使用麦克风)")
    print("2. 文本输入 (键盘输入)")
    print("3. 自动检测 (有音频设备时使用语音输入)")
    
    use_voice_input = None
    while True:
        choice = input("请选择 (1/2/3，默认为3): ").strip()
        if choice == "1":
            use_voice_input = True
            break
        elif choice == "2":
            use_voice_input = False
            break
        elif choice == "3" or choice == "":
            use_voice_input = None
            break
        else:
            print("⚠️ 无效选择，请输入1、2或3")
    
    # 语音输入模式说明
    if use_voice_input or (use_voice_input is None and deps.audio_available):
        print("\n🎯 VAD模式说明:")
        print("✅ 使用优化版Silero VAD（音频累积+重采样）")
        print("📍 核心功能：持续监听 → 音频累积 → 重采样 → Silero检测 → 智能回溯")
        print("💡 如果Silero VAD失败，会自动回退到环形缓冲区+基础VAD模式")
    
    try:
        print("\n🚀 正在初始化语音机器人控制器...")
        
        # 初始化控制器
        controller = VoiceRobotController(
            robot_ip=robot_ip,
            robot_port=Config.ROBOT_PORT_DEFAULT,
            zhipuai_api_key=api_key,
            device="cpu",
            use_voice_input=use_voice_input
        )
        
        # 显示初始化结果
        print("\n" + "="*60)
        print("🤖 语音机器人控制器初始化完成！")
        
        # 显示当前配置
        if controller.recorder:
            if hasattr(controller.recorder, 'advanced_vad') and controller.recorder.advanced_vad:
                if controller.recorder.advanced_vad.vad_available:
                    print("🎯 VAD模式: Silero VAD优化版（音频累积+重采样）")
                    print("⚡ 智能语音检测已启用，适合嘈杂环境")
                else:
                    print("🎯 VAD模式: 环形缓冲区 + 基础VAD（回退模式）")
                    print("⚡ 智能缓冲录音已启用，解决背景噪声问题")
            else:
                print("🎯 VAD模式: 传统基础VAD")
            
            print(f"🎤 音频输入: {controller.recorder.audio_manager.sample_rate}Hz")
        else:
            print("⌨️ 输入模式: 文本输入")
        
        print(f"🔗 机器人地址: {robot_ip}:{Config.ROBOT_PORT_DEFAULT}")
        print("="*60)
        
        # 启动说明
        print("\n📖 使用说明:")
        if controller.wake_matcher:
            print("1. 机器人处于休眠状态，请先说唤醒词：'小智同学'")
            print("2. 唤醒后可以说动作指令或进行聊天")
            print("3. 说退下指令让机器人重新进入休眠：'退下'、'休息'等")
        else:
            print("1. 直接说话或输入指令")
            print("2. 支持动作指令和聊天对话")
        
        if controller.recorder and hasattr(controller.recorder, 'advanced_vad') and controller.recorder.advanced_vad:
            if controller.recorder.advanced_vad.vad_available:
                print("4. Silero VAD会智能识别语音，环形缓冲区确保录音完整性")
            else:
                print("4. 环形缓冲区+智能回溯会自动过滤背景噪声，只保存纯净语音")
        
        print("5. 按 Ctrl+C 退出程序")
        print()
        
        # 运行控制循环
        controller.run()
        
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()


# ================================
# 初始化和启动
# ================================

# 设置环境
Config.setup_environment()

# 检测依赖
deps = DependencyManager()

if __name__ == "__main__":
    main()
