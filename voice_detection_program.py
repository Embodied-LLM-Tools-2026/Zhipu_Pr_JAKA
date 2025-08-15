#!/usr/bin/env python3
"""
语音检测程序 - 基于VAD自动检测语音，识别唤醒词并执行相应动作

功能流程：
1. VAD自动检测语音开始和结束
2. 检测到语音中包含"小拓"后，播放help.mp3，执行动作1
3. 再次检测语音，结束后播放ok.mp3，执行动作2
4. 循环检测
"""

import os
import sys
import time
import tempfile
import threading
import signal
from typing import Optional

# 添加voice模块路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'voice'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 环境设置
os.environ['TORCH_HUB_OFFLINE'] = '1'  # 强制torch hub离线模式

print("🔄 语音检测程序启动中...")

# 依赖检查
def check_dependencies():
    """检查必要的依赖项"""
    missing_deps = []
    
    # 检查语音识别
    try:
        from funasr import AutoModel
        print("✅ FunASR (语音识别) 可用")
    except ImportError:
        missing_deps.append("FunASR (pip install funasr)")
    
    # 检查拼音匹配
    try:
        import pypinyin
        print("✅ pypinyin (拼音匹配) 可用")
    except ImportError:
        missing_deps.append("pypinyin (pip install pypinyin)")
    
    # 检查音频处理
    try:
        import sounddevice as sd
        import soundfile as sf
        print("✅ sounddevice (音频处理) 可用")
    except ImportError:
        try:
            import pyaudio
            print("✅ pyaudio (音频处理) 可用")
        except ImportError:
            missing_deps.append("音频库 (pip install sounddevice soundfile 或 pip install pyaudio)")
    
    # 检查numpy
    try:
        import numpy as np
        print("✅ numpy 可用")
    except ImportError:
        missing_deps.append("numpy (pip install numpy)")
    
    if missing_deps:
        print("❌ 缺少以下依赖项:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n请安装缺少的依赖项后重新运行程序。")
        return False
    
    return True

# 检查依赖
if not check_dependencies():
    print("🛑 依赖检查失败，程序退出")
    sys.exit(1)

# 导入所需模块
try:
    from voice.ASR import SenseVoiceRecognizer
    from voice.PinyinMatcher import PinyinMatcher
    from voice.utils import SimplifiedVoiceRecorder, SimplifiedAudioPlayer
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保voice目录存在且包含必要的模块文件")
    sys.exit(1)


class VoiceDetectionController:
    """语音检测控制器"""
    
    def __init__(self, device="cpu"):
        """初始化语音检测控制器"""
        self.device = device
        self.running = False
        self.current_state = "waiting_for_wake"  # waiting_for_wake, waiting_for_response
        
        # 初始化组件
        print("🔄 初始化语音检测组件...")
        try:
            self._init_components()
        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise
        
        # 预设音频文件路径
        self.help_audio_path = "preset_audio/help.mp3"
        self.ok_audio_path = "preset_audio/ok.mp3"
        
        # 检查预设音频文件是否存在
        self._check_preset_audio()
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("✅ 语音检测控制器初始化完成")
    
    def _init_components(self):
        """初始化各个组件"""
        # 初始化录音器
        try:
            self.recorder = SimplifiedVoiceRecorder()
            if not self.recorder:
                raise RuntimeError("录音器初始化失败")
            print("✅ 语音录音器已启用")
        except Exception as e:
            print(f"❌ 录音器初始化失败: {e}")
            print("请检查音频设备是否正常工作")
            raise
        
        # 初始化语音识别器
        try:
            self.recognizer = SenseVoiceRecognizer(device=self.device)
            print("✅ SenseVoice语音识别已启用")
        except Exception as e:
            print(f"❌ 语音识别器初始化失败: {e}")
            raise
        
        # 初始化拼音匹配器
        try:
            self.matcher = PinyinMatcher()
            print("✅ 拼音匹配器已启用")
        except Exception as e:
            print(f"❌ 拼音匹配器初始化失败: {e}")
            raise
        
        # 初始化音频播放器
        try:
            self.audio_player = SimplifiedAudioPlayer()
            print("✅ 音频播放器已启用")
        except Exception as e:
            print(f"❌ 音频播放器初始化失败: {e}")
            raise
    
    def _check_preset_audio(self):
        """检查预设音频文件是否存在"""
        print("🔍 检查预设音频文件...")
        
        # 检查preset_audio目录
        preset_dir = "preset_audio"
        if not os.path.exists(preset_dir):
            print(f"⚠️ 创建预设音频目录: {preset_dir}")
            os.makedirs(preset_dir, exist_ok=True)
            print("💡 请将help.mp3和ok.mp3文件放入preset_audio目录")
        
        # 检查help.mp3
        if not os.path.exists(self.help_audio_path):
            print(f"⚠️ 预设音频文件不存在: {self.help_audio_path}")
            print("   程序将跳过音频播放，但会继续执行动作")
        else:
            print(f"✅ 找到预设音频: {self.help_audio_path}")
        
        # 检查ok.mp3
        if not os.path.exists(self.ok_audio_path):
            print(f"⚠️ 预设音频文件不存在: {self.ok_audio_path}")
            print("   程序将跳过音频播放，但会继续执行动作")
        else:
            print(f"✅ 找到预设音频: {self.ok_audio_path}")
    
    def _signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        print("\n🛑 收到退出信号，正在停止...")
        self.stop()
        exit(0)
    
    def action_1(self):
        """动作1 - 检测到唤醒词后执行的动作（待实现）"""
        print("🤖 执行动作1 - 响应唤醒词")
        print("💡 在这里添加您的具体动作实现")
        print("   例如：机器人动作、LED指示、其他响应等")
        # TODO: 在这里实现具体的动作逻辑
        time.sleep(1)  # 模拟动作执行时间
        print("✅ 动作1执行完成")
    
    def action_2(self):
        """动作2 - 检测到后续语音后执行的动作（待实现）"""
        print("🤖 执行动作2 - 响应用户语音")
        print("💡 在这里添加您的具体动作实现")
        print("   例如：机器人动作、处理用户请求等")
        # TODO: 在这里实现具体的动作逻辑
        time.sleep(1)  # 模拟动作执行时间
        print("✅ 动作2执行完成")
    
    def _play_audio_file(self, audio_path: str) -> bool:
        """播放音频文件"""
        if not os.path.exists(audio_path):
            print(f"⚠️ 音频文件不存在，跳过播放: {os.path.basename(audio_path)}")
            return False
        
        try:
            print(f"🎵 播放音频: {os.path.basename(audio_path)}")
            success = self.audio_player.play_audio_file(audio_path)
            if success:
                print(f"✅ 音频播放完成: {os.path.basename(audio_path)}")
            else:
                print(f"⚠️ 音频播放失败: {os.path.basename(audio_path)}")
            return success
        except Exception as e:
            print(f"⚠️ 播放音频时发生错误: {e}")
            return False
    
    def _detect_and_process_speech(self) -> Optional[str]:
        """检测并处理语音，返回识别的文本"""
        print("👂 开始语音检测...")
        
        try:
            # 使用SimplifiedVoiceRecorder的内置VAD功能
            success = self.recorder.start_recording(use_vad=True)
            
            if not success:
                print("❌ 录音失败")
                return None
            
            # 保存录音到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                audio_filename = tmp_file.name
            
            if not self.recorder.save_audio(audio_filename):
                print("❌ 录音保存失败")
                return None
            
            # 语音识别
            try:
                print("🧠 正在进行语音识别...")
                text = self.recognizer.recognize(audio_filename)
                if text:
                    print(f"🎯 识别结果: {text}")
                else:
                    print("⚠️ 语音识别结果为空")
                return text
            except Exception as e:
                print(f"❌ 语音识别失败: {e}")
                return None
            finally:
                # 清理临时文件
                try:
                    os.unlink(audio_filename)
                except:
                    pass
                    
        except KeyboardInterrupt:
            print("\n🛑 用户中断录音")
            return None
        except Exception as e:
            print(f"❌ 语音检测过程出错: {e}")
            return None
        finally:
            # 清理录音器资源
            if hasattr(self, 'recorder') and self.recorder:
                try:
                    self.recorder.cleanup()
                except:
                    pass
    
    def _process_wake_detection(self):
        """处理唤醒词检测阶段"""
        print("\n" + "="*50)
        print("😴 等待唤醒词检测...")
        print("🎤 请说包含'小拓'的语音")
        print("="*50)
        
        # 检测语音
        text = self._detect_and_process_speech()
        if not text:
            print("⚠️ 未检测到有效语音，继续等待...")
            return False
        
        # 检测是否包含唤醒词
        try:
            wake_detected, remaining_text = self.matcher.detect_wake_word(text)
        except Exception as e:
            print(f"⚠️ 唤醒词检测失败: {e}")
            return False
        
        if wake_detected:
            print("🎉 检测到唤醒词！")
            
            # 播放help.mp3
            self._play_audio_file(self.help_audio_path)
            
            # 执行动作1
            self.action_1()
            
            # 切换到等待响应状态
            self.current_state = "waiting_for_response"
            print("🔄 切换到等待用户响应状态")
            return True
        else:
            print("😴 未检测到唤醒词，继续等待...")
            return False
    
    def _process_response_detection(self):
        """处理用户响应检测阶段"""
        print("\n" + "="*50)
        print("👂 等待用户语音响应...")
        print("🎤 请说话，系统将自动检测语音结束")
        print("="*50)
        
        # 检测语音
        text = self._detect_and_process_speech()
        if not text:
            print("⚠️ 未检测到有效语音，继续等待...")
            return False
        
        print(f"📝 收到用户语音: {text}")
        
        # 播放ok.mp3
        self._play_audio_file(self.ok_audio_path)
        
        # 执行动作2
        self.action_2()
        
        # 切换回等待唤醒状态
        self.current_state = "waiting_for_wake"
        print("🔄 切换回等待唤醒词状态")
        return True
    
    def run(self):
        """运行语音检测程序"""
        print("\n🚀 语音检测程序启动")
        print("💡 程序将自动检测语音，识别唤醒词并执行相应动作")
        print("⌨️ 按 Ctrl+C 退出程序")
        print()
        
        self.running = True
        
        try:
            while self.running:
                try:
                    if self.current_state == "waiting_for_wake":
                        # 等待唤醒词检测
                        self._process_wake_detection()
                        
                    elif self.current_state == "waiting_for_response":
                        # 等待用户响应
                        self._process_response_detection()
                    
                    # 短暂休息，避免CPU占用过高
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"⚠️ 处理过程中出现错误: {e}")
                    print("🔄 程序将继续运行...")
                    time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 程序被用户中断")
        except Exception as e:
            print(f"❌ 程序运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """停止程序"""
        print("🛑 正在停止语音检测程序...")
        self.running = False
        
        # 清理资源
        if hasattr(self, 'recorder') and self.recorder:
            try:
                self.recorder.cleanup()
            except:
                pass
        
        print("✅ 程序已停止")


def main():
    """主函数"""
    print("🎤 语音检测程序")
    print("="*60)
    
    try:
        # 创建控制器
        controller = VoiceDetectionController(device="cuda:0")
        
        # 显示程序说明
        print("\n📖 程序说明:")
        print("1. 程序会持续监听语音输入")
        print("2. 当检测到包含'小拓'的语音时，会播放help.mp3并执行动作1")
        print("3. 然后等待用户说话，语音结束后播放ok.mp3并执行动作2")
        print("4. 循环进行以上流程")
        print("\n📁 音频文件:")
        print("- 请将help.mp3和ok.mp3放入preset_audio目录")
        print("- 如果音频文件不存在，程序会跳过播放但继续执行动作")
        print("\n🔧 自定义动作:")
        print("- 请在action_1()和action_2()方法中添加您的具体实现")
        print()
        
        # 运行程序
        controller.run()
        
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 