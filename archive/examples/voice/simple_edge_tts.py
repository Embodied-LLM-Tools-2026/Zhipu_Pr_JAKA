import os
import time
import asyncio
import threading
import edge_tts

class SimpleTTS:
    """简单的Edge-TTS文本转语音引擎"""
    
    def __init__(self, output_dir: str = "audio_output"):
        """
        初始化TTS引擎
        
        Args:
            output_dir: 音频文件输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def text_to_speech(self, text: str, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%") -> str:
        """
        将文本转换为语音文件
        
        Args:
            text: 要转换的文本
            voice: 语音类型，默认为中文女声
            rate: 语速调节，如 "+0%", "+10%", "-10%"
            
        Returns:
            str: 生成的音频文件路径，如果失败返回空字符串
        """
        if not text.strip():
            print("⚠️ 输入文本为空")
            return ""
        
        try:
            # 生成唯一文件名
            thread_id = threading.get_ident() % 10000
            timestamp = int(time.time() * 1000)
            filename = f"tts_{timestamp}_{thread_id}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            async def create_audio():
                """异步生成音频文件"""
                communicate = edge_tts.Communicate(text, voice, rate=rate)
                await communicate.save(filepath)
            
            # 运行异步函数
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果当前线程已有运行的事件循环，在新线程中运行
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(create_audio())
                        new_loop.close()
                    
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        future.result(timeout=30)  # 30秒超时
                else:
                    # 当前线程没有事件循环，直接运行
                    loop.run_until_complete(create_audio())
            except RuntimeError:
                # 处理事件循环相关错误
                asyncio.run(create_audio())
            
            # 检查文件是否成功生成
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"✅ 语音文件已生成: {filename} ({os.path.getsize(filepath)} bytes)")
                return filepath
            else:
                print("❌ 语音文件生成失败或为空")
                return ""
                
        except Exception as e:
            print(f"❌ TTS生成错误: {e}")
            return ""
    
    def get_available_voices(self):
        """
        获取可用的语音列表
        
        Returns:
            list: 可用语音列表
        """
        async def get_voices():
            voices = await edge_tts.list_voices()
            return voices
        
        try:
            voices = asyncio.run(get_voices())
            # 筛选中文语音
            chinese_voices = [v for v in voices if 'zh-CN' in v['Locale']]
            return chinese_voices
        except Exception as e:
            print(f"❌ 获取语音列表失败: {e}")
            return []
    
    def print_chinese_voices(self):
        """打印可用的中文语音"""
        voices = self.get_available_voices()
        if voices:
            print("可用的中文语音:")
            for voice in voices:
                print(f"  {voice['ShortName']} - {voice['FriendlyName']}")
        else:
            print("❌ 无法获取语音列表")


# 示例使用
if __name__ == "__main__":
    # 创建TTS实例
    tts = SimpleTTS()
    
    # 测试文本转语音
    test_text = "好的"
    audio_file = tts.text_to_speech(test_text)
    
    if audio_file:
        print(f"音频文件已保存到: {audio_file}")
    else:
        print("语音生成失败")
    
    # 显示可用的中文语音
    print("\n" + "="*50)
    tts.print_chinese_voices() 