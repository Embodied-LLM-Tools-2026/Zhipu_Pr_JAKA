import os
import time
from utils import CrossPlatformAudioManager
# ================================
# 文本转语音模块
# ================================

class TextToSpeechEngine:
    """文本转语音引擎"""
    
    def __init__(self):
        # 确保recordings目录在正确的位置
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