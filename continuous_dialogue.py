#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持续性语音对话系统
结合VAD、ASR、LLM和TTS的完整对话循环
"""

import time
import threading
import queue
import logging
from pathlib import Path
import sys
import numpy as np
import os

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
# 添加voice模块路径
voice_dir = os.path.join(os.path.dirname(__file__), "voice")
sys.path.insert(0, voice_dir)

# 添加项目根目录到路径
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
# 导入语音模块
from voice.utils import SimplifiedVoiceRecorder
from voice.ASR import SenseVoiceRecognizer
from Pr.voice.VLM import RobotCommandProcessor
from voice.TTS import TextToSpeechEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("continuous_dialogue.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ASRAdapter:
    """ASR适配器类，用于统一不同ASR引擎的接口"""

    def __init__(self):
        self.recognizer = None

    def initialize(self):
        """初始化ASR引擎"""
        try:
            self.recognizer = SenseVoiceRecognizer()
            return True
        except Exception as e:
            logger.error(f"ASR初始化失败: {e}")
            return False

    def transcribe_audio_data(self, audio_data):
        """识别音频数据并返回文本"""
        if self.recognizer is None:
            return ""

        try:
            # SenseVoiceRecognizer需要音频文件路径，所以我们需要先保存音频数据
            import tempfile
            import wave
            import struct

            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file_path = tmp_file.name

            # 使用wave模块写入音频数据
            with wave.open(tmp_file_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位
                wav_file.setframerate(16000)  # 采样率16kHz

                # 将numpy数组转换为16位整数并写入
                if hasattr(audio_data, "dtype"):
                    # 如果是numpy数组
                    if audio_data.dtype == "float32" or audio_data.dtype == "float64":
                        # 浮点数数据，需要转换为16位整数
                        audio_data = (audio_data * 32767).astype("int16")
                    elif audio_data.dtype != "int16":
                        audio_data = audio_data.astype("int16")

                    # 写入音频数据
                    wav_file.writeframes(audio_data.tobytes())
                else:
                    # 如果是普通列表，转换为bytes
                    audio_bytes = struct.pack(f"{len(audio_data)}h", *audio_data)
                    wav_file.writeframes(audio_bytes)

            # 使用SenseVoiceRecognizer识别
            text = self.recognizer.recognize(tmp_file_path)

            # 删除临时文件
            import os

            try:
                os.unlink(tmp_file_path)
            except:
                pass

            return text

        except Exception as e:
            logger.error(f"音频识别失败: {e}")
            return ""


class TTSAdapter:
    """TTS适配器类，用于统一不同TTS引擎的接口"""

    def __init__(self):
        self.engine = None
        self.audio_manager = None

    def initialize(self):
        """初始化TTS引擎"""
        try:
            self.engine = TextToSpeechEngine()
            # 导入音频管理器用于播放
            from voice.utils import CrossPlatformAudioManager

            self.audio_manager = CrossPlatformAudioManager()
            return True
        except Exception as e:
            logger.error(f"TTS初始化失败: {e}")
            return False

    def speak(self, text):
        """合成并播放语音"""
        if self.engine is None or self.audio_manager is None:
            return False

        try:
            # 生成音频文件
            audio_file = self.engine.text_to_speech(text)

            if audio_file and os.path.exists(audio_file):
                # 播放音频文件
                success = self.audio_manager.play_audio(audio_file)

                # 播放完成后删除临时文件
                try:
                    os.unlink(audio_file)
                except:
                    pass

                return success
            else:
                logger.error("TTS生成音频文件失败")
                return False

        except Exception as e:
            logger.error(f"TTS播放失败: {e}")
            return False

    def speak_with_pause_control(self, text, dialogue_system):
        """合成并播放语音，同时控制录音暂停"""
        if self.engine is None or self.audio_manager is None:
            return False

        try:
            # 生成音频文件
            audio_file = self.engine.text_to_speech(text)

            if audio_file and os.path.exists(audio_file):
                # 暂停录音
                dialogue_system.pause_recording()

                # 播放音频文件
                success = self.audio_manager.play_audio(audio_file)

                # 播放完成后等待一小段时间，确保音频完全停止
                import time

                time.sleep(0.5)

                # 恢复录音
                dialogue_system.resume_recording()

                # 播放完成后删除临时文件
                try:
                    os.unlink(audio_file)
                except:
                    pass

                return success
            else:
                logger.error("TTS生成音频文件失败")
                return False

        except Exception as e:
            logger.error(f"TTS播放失败: {e}")
            # 确保在异常情况下也恢复录音
            dialogue_system.resume_recording()
            return False


class ContinuousDialogueSystem:
    """持续性对话系统"""

    def __init__(self):
        """初始化对话系统"""
        self.running = False
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # 初始化各个模块
        self.recorder = None
        self.asr = None
        self.llm = None
        self.tts = None

        # 线程锁
        self.lock = threading.Lock()

        # 录音控制
        self.recording_paused = False  # 录音暂停标志
        self.pause_lock = threading.Lock()  # 暂停控制锁

        # 配置参数
        self.silence_duration = 1.0  # 静音时长阈值(秒)
        self.min_recording_duration = 0.5  # 最小录音时长(秒)
        self.max_recording_duration = 10.0  # 最大录音时长(秒)

    def initialize_modules(self):
        """初始化所有语音模块"""
        try:
            logger.info("正在初始化语音模块...")

            # 初始化录音器
            logger.info("初始化录音器...")
            self.recorder = SimplifiedVoiceRecorder()

            # 初始化ASR
            logger.info("初始化ASR...")
            self.asr = ASRAdapter()
            if not self.asr.initialize():
                raise Exception("ASR初始化失败")

            # 初始化LLM
            logger.info("初始化LLM...")
            self.llm = RobotCommandProcessor()

            # 初始化TTS
            logger.info("初始化TTS...")
            self.tts = TTSAdapter()
            if not self.tts.initialize():
                raise Exception("TTS初始化失败")

            logger.info("所有模块初始化完成!")
            return True

        except Exception as e:
            logger.error(f"模块初始化失败: {e}")
            return False

    def pause_recording(self):
        """暂停录音"""
        with self.pause_lock:
            self.recording_paused = True
            logger.info("录音已暂停")

    def resume_recording(self):
        """恢复录音"""
        with self.pause_lock:
            self.recording_paused = False
            logger.info("录音已恢复")

    def is_recording_paused(self):
        """检查录音是否暂停"""
        with self.pause_lock:
            return self.recording_paused

    def audio_recording_thread(self):
        """音频录制线程"""
        logger.info("音频录制线程启动")

        while self.running:
            try:
                # 检查录音是否被暂停
                if self.is_recording_paused():
                    logger.debug("录音已暂停，等待恢复...")
                    time.sleep(0.1)
                    continue

                # 等待语音活动检测
                logger.info("等待语音输入...")

                # 开始录音 - 使用正确的方法名
                success = self.recorder.start_recording(use_vad=True)

                if success and self.recorder.audio_data is not None:
                    audio_data = self.recorder.audio_data

                    if len(audio_data) > 0:
                        # 检查录音时长
                        duration = (
                            len(audio_data) / self.recorder.advanced_vad.sample_rate
                        )

                        if duration >= self.min_recording_duration:
                            logger.info(f"录制到音频数据，时长: {duration:.2f}秒")
                            self.audio_queue.put(audio_data)
                        else:
                            logger.info(f"录音时长太短({duration:.2f}s)，忽略")
                    else:
                        logger.debug("录音数据为空")
                else:
                    logger.debug("未检测到有效语音")

                # 短暂休息
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"录音线程出错: {e}")
                time.sleep(1)

        logger.info("音频录制线程结束")

    def speech_processing_thread(self):
        """语音处理线程"""
        logger.info("语音处理线程启动")

        while self.running:
            try:
                # 获取音频数据
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                logger.info("开始处理语音...")

                # 语音识别
                text = self.asr.transcribe_audio_data(audio_data)
                if not text or text.strip() == "":
                    logger.info("未识别到有效文本")
                    continue

                logger.info(f"识别到文本: {text}")

                # LLM处理
                with self.lock:
                    result = self.llm.process_command(text)

                if result and result.get("confidence", 0) > 0:
                    response_text = result.get(
                        "description", "抱歉，我没有理解您的意思"
                    )
                    logger.info(f"LLM响应: {response_text}")

                    # 将响应放入队列
                    self.response_queue.put(
                        {
                            "text": response_text,
                            "intent": result.get("intent", "chat"),
                            "action": result.get("action", "unknown"),
                            "original_text": text,
                        }
                    )
                else:
                    logger.info("LLM判断为无效输入，跳过响应")

            except Exception as e:
                logger.error(f"语音处理线程出错: {e}")
                time.sleep(0.5)

        logger.info("语音处理线程结束")

    def speech_synthesis_thread(self):
        """语音合成线程"""
        logger.info("语音合成线程启动")

        while self.running:
            try:
                # 获取响应数据
                try:
                    response_data = self.response_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                response_text = response_data["text"]
                intent = response_data["intent"]
                action = response_data["action"]
                original_text = response_data["original_text"]

                logger.info(f"开始合成语音: {response_text}")

                # 语音合成 - 使用带暂停控制的方法
                with self.lock:
                    success = self.tts.speak_with_pause_control(response_text, self)

                if success:
                    logger.info("语音播放完成")

                    # 如果是指令，这里可以添加动作执行逻辑
                    if intent == "command" and action != "unknown":
                        logger.info(f"检测到指令: {action}")
                        # TODO: 在这里添加机器人动作执行代码
                        # self.execute_robot_action(action, response_data)

                else:
                    logger.error("语音合成失败")

            except Exception as e:
                logger.error(f"语音合成线程出错: {e}")
                time.sleep(0.5)

        logger.info("语音合成线程结束")

    def execute_robot_action(self, action, response_data):
        """执行机器人动作(预留接口)"""
        logger.info(f"执行机器人动作: {action}")

        # 这里可以根据action类型调用相应的机器人控制代码
        action_map = {
            "greet": "打招呼/摆手",
            "shake_head": "摇头",
            "nod": "点头",
            "bow": "鞠躬",
            "get_drink": "拿饮料",
            "others": "其他动作",
        }

        if action in action_map:
            logger.info(f"模拟执行动作: {action_map[action]}")

            # 获取饮料相关信息
            if action == "get_drink":
                obj_name = response_data.get("obj_name", "未知饮料")
                num = response_data.get("num", 1)
                logger.info(f"饮料类型: {obj_name}, 数量: {num}")
        else:
            logger.warning(f"未知动作类型: {action}")

    def start(self):
        """启动对话系统"""
        logger.info("启动持续性对话系统...")

        # 初始化模块
        if not self.initialize_modules():
            logger.error("模块初始化失败，无法启动系统")
            return False

        # 设置运行标志
        self.running = True

        # 启动处理线程
        threads = []

        # 录音线程
        recording_thread = threading.Thread(
            target=self.audio_recording_thread, daemon=True
        )
        recording_thread.start()
        threads.append(recording_thread)

        # 语音处理线程
        processing_thread = threading.Thread(
            target=self.speech_processing_thread, daemon=True
        )
        processing_thread.start()
        threads.append(processing_thread)

        # 语音合成线程
        synthesis_thread = threading.Thread(
            target=self.speech_synthesis_thread, daemon=True
        )
        synthesis_thread.start()
        threads.append(synthesis_thread)

        logger.info("对话系统启动成功!")
        logger.info("说话开始对话，按 Ctrl+C 结束程序")

        try:
            # 主线程保持运行
            while self.running:
                time.sleep(1)

                # 检查线程状态
                alive_threads = [t for t in threads if t.is_alive()]
                if len(alive_threads) < len(threads):
                    logger.warning("检测到线程异常退出")

        except KeyboardInterrupt:
            logger.info("接收到停止信号...")
            self.stop()

        # 等待线程结束
        for thread in threads:
            thread.join(timeout=2.0)

        logger.info("对话系统已停止")
        return True

    def stop(self):
        """停止对话系统"""
        logger.info("正在停止对话系统...")
        self.running = False

        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break

    def get_status(self):
        """获取系统状态"""
        return {
            "running": self.running,
            "audio_queue_size": self.audio_queue.qsize(),
            "response_queue_size": self.response_queue.qsize(),
            "modules_initialized": all(
                [
                    self.recorder is not None,
                    self.asr is not None,
                    self.llm is not None,
                    self.tts is not None,
                ]
            ),
        }


def main():
    """主函数"""
    print("=" * 60)
    print("持续性语音对话系统")
    print("=" * 60)
    print("功能:")
    print("- 持续监听语音输入")
    print("- 自动语音识别(ASR)")
    print("- 智能对话生成(LLM)")
    print("- 语音合成播放(TTS)")
    print("- 多线程并行处理")
    print("=" * 60)

    # 创建对话系统
    dialogue_system = ContinuousDialogueSystem()

    try:
        # 启动系统
        dialogue_system.start()

    except Exception as e:
        logger.error(f"系统运行错误: {e}")

    finally:
        # 确保系统停止
        dialogue_system.stop()
        print("\n对话系统已退出")


if __name__ == "__main__":
    main()
