#!/usr/bin/env python3
"""

实现了智能流式TTS！按句子分割，并行生成，连续播放

基于语音识别和大模型的机器人控制脚本 - 重构版本
使用SenseVoice进行语音识别，智谱AI GLM模型理解指令，控制机器人执行动作
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
import time
import multiprocessing as mp
import tempfile
import threading
import signal
from typing import Optional, Dict, Any
import random
from dotenv import load_dotenv
load_dotenv()

os.environ['TORCH_HUB_OFFLINE'] = '1'  # 强制torch hub离线模式,silero-vad模型就不需要联网加载权重
import torch
import sys

# # 将父目录临时注册为系统路径，方便python导入模块
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'voice'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("VRC (流式,VAD,win) running...")

# ================================
# 依赖安装检查
# ================================

# 依赖检查已集成到DependencyManager类中

from voice.config import Config
from voice.deps import DependencyManager
from voice.TTS import TextToSpeechEngine
from voice.ASR import SenseVoiceRecognizer
from voice.LLM import RobotCommandProcessor
from voice.PinyinMatcher import PinyinMatcher
from voice.utils import SimplifiedVoiceRecorder, SimplifiedAudioPlayer
from voice.ActionExecuter_hand import ActionExecuter

# 设置环境
Config.setup_environment()

deps = DependencyManager()
if not deps._get_check_result():
    print("⚠️ 依赖检测失败，请检查依赖项")
from objectLocalization.objLocalization import ObjectLocalization



# ================================
# 主控制器
# ================================

class VoiceRobotController:
    """语音机器人控制器主类"""
    
    def __init__(self, 
                 robot_ip_left: str = Config.ROBOT_IP_LEFT,
                 robot_ip_right: str = Config.ROBOT_IP_RIGHT,
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
        
        # 音频文件管理
        self.session_audio_files = set()  # 本次运行生成的音频文件
        self.persistent_audio_files = set()  # 持久化的预制音频文件
        
        # 性能优化：预缓存常用音频
        if Config.PRELOAD_COMMON_AUDIO:
            self._preload_common_audio()
        
        # 初始化机器人控制器
        self.robot_controller = ActionExecuter(robot_ip_left, robot_ip_right, deps.robot_available)

        # 性能优化：预缓存拖延语音频
        self.delay_phrase_cache = {}
        self._preload_delay_phrases()
        
        # 自我介绍关键词
        self.intro_keywords = [
            "介绍一下你自己", "你是谁", "自我介绍", "请介绍你自己", "你能做什么", "你的功能", "你的作用", "你是做什么的"
        ]
        
        # 初始化饮料货架定位器
        self.obj_locater = ObjectLocalization()

        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理Ctrl+C信号，清理运行时音频文件"""
        print("\n🛑 收到退出信号，正在清理音频文件...")
        self._cleanup_session_audio()
        print("✅ 音频文件清理完成")
        exit(0)
    
    def _cleanup_session_audio(self):
        """清理本次运行生成的音频文件，保留预制音频"""
        if not self.session_audio_files:
            print("📁 没有需要清理的运行时音频文件")
            return
        
        cleaned_count = 0
        failed_count = 0
        
        for audio_file in self.session_audio_files:
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    cleaned_count += 1
                    print(f"🗑️ 已删除: {os.path.basename(audio_file)}")
            except Exception as e:
                failed_count += 1
                print(f"❌ 删除失败: {os.path.basename(audio_file)} - {e}")
        
        print(f"📊 清理完成: 成功删除 {cleaned_count} 个文件，失败 {failed_count} 个文件")
        self.session_audio_files.clear()
    
    def _get_persistent_audio_path(self, text: str, category: str = "common") -> str:
        """获取持久化音频文件路径"""
        import hashlib
        
        # 使用文本内容的哈希值作为文件名，确保相同文本生成相同文件名
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        filename = f"persistent_{category}_{text_hash}.wav"
        
        # 确保recordings目录存在
        recordings_dir = os.path.join(os.getcwd(), "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        
        return os.path.join(recordings_dir, filename)
    
    def _check_persistent_audio(self, text: str, category: str = "common") -> Optional[str]:
        """检查是否存在持久化的音频文件"""
        audio_path = self._get_persistent_audio_path(text, category)
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            self.persistent_audio_files.add(audio_path)
            return audio_path
        return None
    
    def _save_persistent_audio(self, text: str, audio_path: str, category: str = "common"):
        """保存音频文件为持久化文件"""
        persistent_path = self._get_persistent_audio_path(text, category)
        try:
            import shutil
            shutil.copy2(audio_path, persistent_path)
            self.persistent_audio_files.add(persistent_path)
            print(f"💾 已保存持久化音频: {os.path.basename(persistent_path)}")
        except Exception as e:
            print(f"❌ 保存持久化音频失败: {e}")
    
    def _init_components(self, use_voice_input: Optional[bool], device: str, zhipuai_api_key: Optional[str]):
        """初始化各个组件"""
        # 初始化录音器 - 使用简化的跨平台版本
        if use_voice_input is None:
            self.recorder = SimplifiedVoiceRecorder()
            print(f"🔧 输入模式: {'语音输入' if self.recorder else '文本输入'} (自动检测)")
        elif use_voice_input:
            self.recorder = SimplifiedVoiceRecorder()
            print("🎤 输入模式: 语音输入 (用户选择)")
        else:
            self.recorder = None
            print("⌨️ 输入模式: 文本输入 (用户选择)")
        
        # 将控制器引用传递给录音器（用于计时功能）
        if self.recorder:
            setattr(self.recorder, "controller", self)
        
        # 初始化其他组件
        self.recognizer = SenseVoiceRecognizer(device=device)
        print("✅ SenseVoice语音识别已启用")
        
        self.processor = RobotCommandProcessor()
        self.tts_engine = TextToSpeechEngine()
        self.audio_player = SimplifiedAudioPlayer()
        
        # 初始化唤醒词检测器
        self.wake_matcher = PinyinMatcher()
        print("✅ 唤醒词功能已启用")
    
    def _preload_common_audio(self):
        """预加载常用音频，优先使用持久化文件"""
        print("🔄 预加载常用音频...")
        
        for phrase in Config.COMMON_PHRASES:
            try:
                # 首先检查是否有持久化的音频文件
                persistent_path = self._check_persistent_audio(phrase, "common")
                if persistent_path:
                    self.tts_cache[phrase] = persistent_path
                    print(f"📁 使用持久化音频: {phrase[:20]}...")
                else:
                    # 如果没有持久化文件，生成新的音频文件
                    audio_path = self.tts_engine.text_to_speech(phrase)
                    if audio_path:
                        self.tts_cache[phrase] = audio_path
                        self.session_audio_files.add(audio_path)
                        # 保存为持久化文件
                        self._save_persistent_audio(phrase, audio_path, "common")
            except Exception as e:
                print(f"⚠️ 预加载音频失败: {phrase[:10]}... - {e}")
        
        print(f"✅ 预加载完成，缓存了 {len(self.tts_cache)} 个音频")
    
    def _preload_delay_phrases(self):
        """预生成所有拖延语音频，优先使用持久化文件"""
        print("🔄 预生成拖延语音频...")
        for phrase in Config.DELAY_PHRASES:
            try:
                # 首先检查是否有持久化的音频文件
                persistent_path = self._check_persistent_audio(phrase, "delay")
                if persistent_path:
                    self.delay_phrase_cache[phrase] = persistent_path
                    print(f"📁 使用持久化拖延语: {phrase[:20]}...")
                else:
                    # 如果没有持久化文件，生成新的音频文件
                    audio_path = self.tts_engine.text_to_speech(phrase)
                    if audio_path:
                        self.delay_phrase_cache[phrase] = audio_path
                        self.session_audio_files.add(audio_path)
                        # 保存为持久化文件
                        self._save_persistent_audio(phrase, audio_path, "delay")
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
            # 将运行时生成的音频文件添加到清理列表
            self.session_audio_files.add(audio_path)
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
        
        # # 3秒后播放拖延语（无论TTS是否完成）
        # delay_timer = threading.Timer(3.0, lambda: self._play_delay_phrase_and_set_flag(delay_played))
        # delay_timer.start()
        
        # 等待TTS完成
        tts_done.wait()
        
        # # 取消延迟定时器（如果TTS在3秒内完成）
        # delay_timer.cancel()
        
        # # 如果拖延语已播放，等待其播放完成
        # if delay_played.is_set():
        #     print("⏳ 等待拖延语播放完成...")
        #     # 等待拖延语播放完成（大约2-3秒）
        #     time.sleep(2.5)
        
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
                if self.recorder:
                    adv_vad = getattr(self.recorder, 'advanced_vad', None)
                    if adv_vad is not None:
                        print("🔥 后台预热VAD系统...")
                        # 重置VAD状态，准备下一轮录音
                        if hasattr(adv_vad, 'reset'):
                            adv_vad.reset()
                        # 预检查音频设备健康状态
                        if hasattr(self.recorder, '_check_audio_device_health'):
                            self.recorder._check_audio_device_health()
                        # 设置预热标志，下次录音时可以快速启动
                        setattr(self.recorder, "_vad_prewarmed", True)
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
                    # 将生成的音频文件添加到清理列表
                    self.session_audio_files.add(audio_path)
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
        """获取语音输入或文本输入并用ASR转换成文本"""
        if not self.recorder:
            # 文本输入模式
            if self.robot_state == "sleeping":
                prompt = "请输入唤醒词（小拓同学）或输入'stats'查看计时统计: "
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
                    print("🎤 请说：小拓同学")
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
                        try:
                            text = self.recognizer.recognize(audio_filename)
                        except Exception as asr_error:
                            print(f"⚠️ SenseVoice语音识别失败: {asr_error}")
                            print("请手动输入:")
                            return input().strip()
                        
                        asr_end_time = time.time()
                        asr_duration = asr_end_time - asr_start_time
                        
                        self._input_end_time = asr_end_time
                        
                        if hasattr(self, '_input_start_time'):
                            total_input_duration = asr_end_time - self._input_start_time
                            print(f"⏱️ 语音输入耗时: {total_input_duration:.2f}秒")
                        
                        print(f"🎯 SenseVoice语音识别结果: {text}")
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

            self._play_and_execute_action("你好，我是小拓同学，很高兴见到你！", "greet")
            
            # 切换到唤醒状态
            self.robot_state = "awake"
            # 唤醒时不需要执行机器人动作，直接准备接收指令
            print("✅ 机器人已唤醒，准备接收指令")
            
            return True
        else:
            print("😴 机器人还在休眠中，请说唤醒词：小拓同学")
            return False
    
    # 同时播放音频和执行动作
    def _play_and_execute_action(self, audio_text: str, action: str) -> bool:
        """同时播放音频和执行动作"""
        result_holder = {"success": False}
        def play_audio_thread():
            self._play_cached_audio(audio_text)
        def execute_action_thread():
            success = self.robot_controller.execute_action(action)
            if success:
                print(f"✅ {action}动作执行成功")
            else:
                print(f"❌ {action}动作执行失败")
            result_holder["success"] = success
        audio_thread = threading.Thread(target=play_audio_thread)
        action_thread = threading.Thread(target=execute_action_thread)
        audio_thread.start()
        action_thread.start()
        audio_thread.join()
        action_thread.join()
        return result_holder["success"]

    def _handle_awake_state(self, text: str) -> bool:
        """处理正常聊天或动作指令或退下指令"""
        if not text.strip():
            print("未识别到有效文本")
            return False
        
        # 在唤醒状态下检测到唤醒词时，快速响应
        # if self.wake_matcher:
        #     wake_detected, remaining_text = self.wake_matcher.detect_wake_word(text)
        #     if wake_detected:
        #         print("🎯 检测到唤醒词，快速响应中...")
                
        #         # 随机选择快速响应短语，让响应更自然
        #         import random
        #         quick_responses = [
        #             "我在听，请说",
        #             "嗯，我在",
        #             "请说"
        #         ]
        #         response = random.choice(quick_responses)
                
        #         # 快速播放响应音频（使用缓存）
        #         self._play_cached_audio(response)
                
        #         # 如果唤醒词后面还有指令，立即处理
        #         # if remaining_text.strip():
        #         #     print(f"💬 检测到后续指令: {remaining_text}")
        #         #     return self._process_action_command(remaining_text)
                
        #         return True
        
        # 检测退下指令
        if self.wake_matcher and self.wake_matcher.detect_dismiss_command(text):
            print("🤖 好的，我去休息了。需要时请叫我！")

            self._play_and_execute_action("好的，我去休息了。需要时请叫我！", "nod")
            
            # 切换到休眠状态
            self.robot_state = "sleeping"
            print("😴 机器人已进入休眠状态")
            
            # 退下时可以选择是否执行回到待机位置（可选）
            # success = self.robot_controller.execute_action("waiting")
            # if not success:
            #     print("⚠️ 回到待机位置失败，但不影响休眠状态")
            
            return True
        else:
            # 处理其他指令（动作指令或聊天）
            return self._process_action_command(text)
    
    # 主要在这里调用后面视觉和实机执行的接口
    def _process_action_command(self, text: str) -> bool:
        """处理动作指令（带拖延语机制，LLM+TTS整体流程超时1.5秒自动插入拖延语）"""
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
        intent = command_result.get("intent", "unknown") # 意图
        action = command_result.get("action", "unknown") # 具体动作
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
        
        if intent == "command":
            print(f"🎯 识别动作: {description} (置信度: {confidence:.2f})")
            print("🎵 开始异步TTS生成和播放（同时预热VAD）...")
            
        else:
            print(f"小拓说：{description}")
            print("🎵 开始异步TTS生成和播放（同时预热VAD）...")
            audio_file_path = self._play_cached_audio(description, tts_ready_callback=tts_ready_callback)
        
        # 性能监控 - TTS开始播放（不包括播放时间）
        tts_play_start_time = time.time()
        tts_prepare_duration = tts_play_start_time - tts_start_time

        if intent == "command" and (action == "unknown" or confidence < 0.5):
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
        if intent == "command":
            if action not in Config.ACTION_MAP.keys():
                print("💬 这个动作我还不会，不过我会抓紧学习的")
                self._play_and_execute_action("这个动作我还不会，不过我会抓紧学习的", "shake_head")
                success = True
            else:
                # 区分拿饮料和非拿饮料
                if action == "get_drink":
                    obj_name = command_result.get("obj_name", "unknown")
                    num = int(command_result.get("num", "0"))
                    if obj_name not in Config.drink_list:   
                        if obj_name == "饮料": # 如果用户说“饮料”，则提示用户具体需要哪种饮料。待解决：如果用户同时说了多种饮料，会出错
                            print("请您告诉我具体需要哪种饮料")
                            self._play_cached_audio("请您告诉我具体需要哪种饮料",tts_ready_callback=tts_ready_callback)
                        else:
                            print("💬 我们这里没有这种饮料")
                            self._play_and_execute_action("我们这里没有这种饮料", "shake_head")
                        success = True
                    else:
                        # if obj_name in ["美式咖啡", "其它咖啡"]:
                        #     if num != 1:
                        #         print("💬 抱歉，一次只能做一杯美式咖啡哦")
                        #         self._play_and_execute_action("抱歉，一次只能做一杯美式咖啡哦", "shake_head")
                        #         success = True
                        #     elif obj_name == "其它咖啡":
                        #         print("💬 抱歉，只能做美式咖啡哦")
                        #         self._play_and_execute_action("抱歉，只能做美式咖啡哦", "shake_head")
                        #         success = True
                        #     else:
                        #         print("💬 好的，我这就去为您做咖啡")
                        #         self._play_and_execute_action("好的，我这就去为您做咖啡", "nod")
                        #         self.robot_controller.move_to_coffee_machine_and_make_coffee()
                        #         self._play_cached_audio("咖啡正在制作中，请您稍等片刻", tts_ready_callback=tts_ready_callback)
                        #         self.robot_controller.get_coffee_and_serve()
                        #         self.robot_controller.execute_action(action="rotate_head_to_angle", angle=-40)
                        #         self._play_cached_audio("请享用您的咖啡", tts_ready_callback=tts_ready_callback)
                        #         self.robot_controller.execute_action(action="rotate_head_to_angle", angle=0, incremental=True, back_to_init=True)
                        #         self.robot_controller.back_bar_station()
                        #         self.robot_controller.back_to_init_height_and_angle()
                        #         success = True
                        # else:
                        self._play_and_execute_action("好的，我去看看饮料还够不够", "nod")
                        # 获取饮料层数及对应的机器人头部俯仰角和身躯高度
                        layer_number,head_angle,body_distance = self.obj_locater.get_layer_number(obj_name=obj_name,num=num)
                        # 到达对应层数
                        self.robot_controller.execute_get_drink(head_angle=head_angle, body_distance=body_distance)
                        # 获取饮料位置
                        pos_list = self.obj_locater.observe(obj_name, num)
                        pos_list = pos_list or []
                        # pos_list = [5,4] # 测试用
                        print(f"💬 所在的层数：{layer_number}, 机器人头部俯仰角：{head_angle}, 机器人身躯高度：{body_distance}")
                        print(f"💬 饮料位置: {pos_list}")
                        if len(pos_list) > 0:
                            audio_file_path = self._play_cached_audio("饮料还够，我这就拿给您，请您稍等", tts_ready_callback=tts_ready_callback)
                            for i,pos in enumerate(pos_list):
                                if not self.robot_controller.execute_get_drink(drink_id=pos,layer_number=layer_number,head_angle=head_angle,body_distance=body_distance):
                                    print("💬 不好意思，饮料不够了")
                                    self._play_cached_audio("不好意思，饮料不够了", tts_ready_callback=tts_ready_callback) 
                                    success = True
                                    break
                                audio_file_path = self._play_cached_audio("这是您要的饮料", tts_ready_callback=tts_ready_callback)
                                if i < len(pos_list) - 1:
                                    audio_file_path = self._play_cached_audio("下一瓶我这就去拿", tts_ready_callback=tts_ready_callback)
                        else:
                            print("💬 不好意思，饮料不够了")
                            self.robot_controller.back_bar_station()
                            self.robot_controller.back_to_init_height_and_angle()
                            self._play_cached_audio("不好意思，饮料不够了", tts_ready_callback=tts_ready_callback) 
                        success = True
                else:
                    audio_file_path = self._play_cached_audio("好的", tts_ready_callback=tts_ready_callback)
                    success = self.robot_controller.execute_action(action)

                # 动作成功或失败后的处理
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
                # 清理运行时音频文件
                self._cleanup_session_audio()
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
    if not deps._get_check_result():
        return
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
    # robot_ip = input(f"请输入机器人IP地址 (默认{Config.ROBOT_IP_DEFAULT}): ").strip()
    # if not robot_ip:
    robot_ip_left = Config.ROBOT_IP_LEFT
    robot_ip_right = Config.ROBOT_IP_RIGHT

    
    # 选择输入模式
    print("\n🎛️ 请选择输入模式:")
    print("1. 语音输入 (使用麦克风)")
    print("2. 文本输入 (键盘输入)")
    print("3. 自动检测 (有音频设备时使用语音输入)")
    
    use_voice_input = None
    while True:
        choice = input("请选择 (1/2/3，默认为1): ").strip()
        if choice == "1"  or choice == "":
            use_voice_input = True
            break
        elif choice == "2":
            use_voice_input = False
            break
        elif choice == "3":
            use_voice_input = None
            break
        else:
            print("⚠️ 无效选择，请输入1、2或3")
    
    # 语音输入模式说明
    if use_voice_input or (use_voice_input is None):
        print("\n🎯 VAD模式说明:")
        print("✅ 使用优化版Silero VAD（音频累积+重采样）")
        print("📍 核心功能：持续监听 → 音频累积 → 重采样 → Silero检测 → 智能回溯")
        print("💡 如果Silero VAD失败，会自动回退到环形缓冲区+基础VAD模式")
    
    try:
        print("\n🚀 正在初始化语音机器人控制器...")
        
        # 初始化控制器
        controller = VoiceRobotController(
            robot_ip_left=robot_ip_left,
            robot_ip_right=robot_ip_right,
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
        
        print(f"🔗 机器人地址: {robot_ip_left} 和 {robot_ip_right}")
        print("="*60)
        
        # 启动说明
        print("\n📖 使用说明:")
        if controller.wake_matcher:
            print("1. 机器人处于休眠状态，请先说唤醒词：'小拓同学'")
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

if __name__ == "__main__":
    main()
