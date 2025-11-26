import os
import sys
import contextlib
# 将父目录临时注册为系统路径，方便python导入模块
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
        self.funasr_available = False
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检测所有依赖项"""
        self._check_zhipuai()
        self._check_audio()
        self._check_robot()
        self._check_pypinyin()
        self._check_numpy()
        self._check_silero_vad()
        self._check_funasr()
    
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
            from Pr.action_sequence.old_file.execute_action import init_robot, wave, bow, Nod, Shake_head
            self.robot_available = True
            print("✅ 机器人控制模块可用")
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
    
    def _check_funasr(self):
        """检测FunASR"""
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            self.funasr_available = True
            print("✅ FunASR已安装，SenseVoice语音识别可用")
        except ImportError:
            self.funasr_available = False
            print("警告: FunASR未安装，请运行: pip install funasr")

    # 如果有一项为False，则返回False
    def _get_check_result(self):
        # if not self.robot_available:
        #     return False
        if not self.audio_available:
            return False
        if not self.zhipuai_available:
            return False
        if not self.pypinyin_available:
            return False
        if not self.numpy_available:
            return False
        if not self.silero_vad_available:
            return False
        if not self.funasr_available:
            return False
        return True 
        