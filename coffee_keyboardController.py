#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘控制器模块
用于监听小键盘输入并执行分级选择操作
"""

import os
import sys
import time
import logging
import queue
from queue import Queue
from threading import Thread
from typing import Dict, Callable, Optional
from dataclasses import dataclass
from pynput.keyboard import Key, Listener

# 添加父目录到路径，确保可以正确导入controller模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# 导入控制器模块
try:
    from controller.hand_controller import InspireHandR
    from controller.gripper_controller import GripperController
    from action_sequence.agv_client import AGVClient
    from action_sequence.pour_coffee import put_coffee_cup, press_button, get_coffee_cup_with_coffee
except ImportError as e:
    logging.warning(f"导入控制器模块失败: {e}")
    # 创建占位类以避免NameError
    class InspireHandR:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InspireHandR类导入失败")
    
    class AGVClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("AGVClient类导入失败")

@dataclass
class KeyboardConfig:
    """键盘配置类"""
    # 小键盘1~9的虚拟键码映射（Windows系统）
    NUMPAD_VK_MAPPING: Dict[int, int] = None
    
    def __post_init__(self):
        if self.NUMPAD_VK_MAPPING is None:
            self.NUMPAD_VK_MAPPING = {
                97: 1, 98: 2, 99: 3,      # 小键盘1-3
                100: 4, 101: 5, 102: 6,   # 小键盘4-6
                103: 7, 104: 8, 105: 9    # 小键盘7-9
            }
    
    # 退出键列表
    EXIT_KEYS = [Key.esc, Key.backspace]
    
    # 处理间隔时间（秒）
    PROCESSING_DELAY = 2.0

class WindowsKeyboardListener:
    """Windows专用键盘监听器，只监听特定按键"""
    
    def __init__(self, config: KeyboardConfig, key_queue: Queue, stop_callback: Callable):
        self.config = config
        self.key_queue = key_queue
        self.stop_callback = stop_callback
        self.is_running = False
        self.listener = None
    
    def on_press(self, key):
        """按键按下事件处理"""
        # 只处理小键盘数字和退出键
        if key in self.config.EXIT_KEYS:
            self.key_queue.put(key)
            return True
        if (hasattr(key, 'vk') and key.vk in self.config.NUMPAD_VK_MAPPING):
            self.key_queue.put(key)
            return True
        return True
    
    def on_release(self, key):
        return True
    
    def start(self):
        """启动监听器"""
        self.is_running = True
        self.listener = Listener(
            on_press=self.on_press, 
            on_release=self.on_release,
            suppress=True  # 尝试抑制按键的默认行为
        )
        self.listener.start()
    
    def stop(self):
        """停止监听器"""
        self.is_running = False
        if self.listener:
            self.listener.stop()
            self.listener = None

class KeyboardController:
    """分级选择键盘控制器类"""
    
    def __init__(self, config: Optional[KeyboardConfig] = None):
        self.config = config or KeyboardConfig()
        self.key_queue = Queue()
        self.is_running = False
        self.keyboard_listener = None
        self.processor_thread = None
        self.logger = None
        self._setup_logging()
        
        # 初始化动作映射
        self._setup_action_mapping()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('keyboard_controller.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_action_mapping(self):
        """设置动作映射"""
        self.action_mapping = {
            1: self._action_1,
            2: self._action_2,
            3: self._action_3,
            4: self._action_4,
            5: self._action_5,
            6: self._action_6,
            7: self._action_7,
            8: self._action_8,
            9: self._action_9
        }
    
    def _action_1(self):
        """动作1：模拟耗时任务"""
        self.is_processing = True
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作1：开始")
        time.sleep(self.config.PROCESSING_DELAY)
        self.is_processing = False
        self.logger.info("执行动作1：完成")
    
    def _action_2(self):
        """动作2"""
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作2")
    
    def _action_3(self):
        """动作3"""
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作3")
    
    def _action_4(self):
        """动作4"""
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作4")
    
    def _action_5(self):
        """动作5"""
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作5")
    
    def _action_6(self):
        """动作6"""
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作6")
    
    def _action_7(self):
        """动作7"""
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作7")
    
    def _action_8(self):
        """动作8"""
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作8")
    
    def _action_9(self):
        """动作9"""
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作9")
    
    def key_processor(self):
        self.logger.info("按键处理线程启动")
        state = "category"  # category/function/exit
        selected_category = None
        backspace_count = 0
        is_executing_function = False  # 标记是否正在执行函数

        self.print_category_menu()

        while self.is_running and not self.exit_flag:
            try:
                key = self.key_queue.get(timeout=1.0)
                
                # 如果正在执行函数，忽略所有按键输入（除了ESC用于紧急退出）
                if is_executing_function:
                    if key == Key.esc:
                        print("\n检测到ESC，程序退出。")
                        self.logger.info("检测到ESC，程序退出")
                        self.exit_flag = True
                        self.exit_event.set()
                        os._exit(0)
                    else:
                        # 忽略其他按键，继续等待函数执行完成
                        self.key_queue.task_done()
                        continue
                
                # 处理Backspace和Esc
                if key == Key.backspace:
                    backspace_count += 1
                    if state == "function":
                        # 返回到类别选择
                        print("\n已返回类别选择。")
                        self.logger.info("返回类别选择")
                        state = "category"
                        selected_category = None
                        self.print_category_menu()
                        backspace_count = 0
                    elif state == "category":
                        if backspace_count == 2:
                            print("\n检测到连续两次Backspace，程序退出。")
                            self.logger.info("检测到连续两次Backspace，程序退出")
                            self.exit_flag = True
                            self.exit_event.set()
                            # 立即退出整个进程
                            os._exit(0)
                        else:
                            print("\n再按一次Backspace退出程序。")
                            self.logger.info("检测到Backspace，等待第二次确认退出")
                    continue
                elif key == Key.esc:
                    print("\n检测到ESC，程序退出。")
                    self.logger.info("检测到ESC，程序退出")
                    self.exit_flag = True
                    self.exit_event.set()
                    # 立即退出整个进程
                    os._exit(0)
                    continue
                else:
                    backspace_count = 0  # 只要不是Backspace就重置

                # 只处理小键盘数字
                if hasattr(key, 'vk') and key.vk in self.config.NUMPAD_VK_MAPPING:
                    num = self.config.NUMPAD_VK_MAPPING[key.vk]
                    if state == "category":
                        if 1 <= num <= self.category_count:
                            selected_category = num
                            print(f"\n已选择类别：{self.category_names.get(num, f'类别{num}')}")
                            self.logger.info(f"选择类别: {num}")
                            state = "function"
                            self.print_function_menu(selected_category)
                        else:
                            print(f"无效类别编号: {num}，请重新选择。")
                    elif state == "function":
                        if 1 <= num <= self.function_count:
                            print(f"\n已选择函数：{self.function_names.get(num, f'函数{num}')}")
                            self.logger.info(f"选择函数: {num}")
                            # 执行对应函数
                            func = self.function_mapping.get(num)
                            if func:
                                # 设置执行状态，暂停按键监听器并清空队列
                                is_executing_function = True
                                self._pause_keyboard_listener()
                                self._clear_key_queue()
                                print("函数执行中，按键输入已被完全禁用...")
                                
                                # 执行函数
                                func()
                                
                                # 函数执行完成，恢复按键监听器
                                is_executing_function = False
                                self._resume_keyboard_listener()
                                print("函数执行完成，按键输入已恢复。")
                                self.print_function_menu(selected_category)
                            else:
                                print("未定义的函数。")
                        else:
                            print(f"无效函数编号: {num}，请重新选择。")
                    else:
                        pass
                self.key_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"按键处理错误: {e}")
                continue

    def start(self):
        if self.is_running:
            self.logger.warning("键盘控制器已在运行")
            return
        self.is_running = True
        self.exit_flag = False
        self.exit_event.clear()
        self.processor_thread = Thread(target=self.key_processor, daemon=True)
        self.processor_thread.start()
        self.keyboard_listener = WindowsKeyboardListener(
            self.config, 
            self.key_queue, 
            self.stop
        )
        self.keyboard_listener.start()
        self.logger.info("键盘控制器启动成功，开始监听小键盘输入...")

    def stop(self):
        self.is_running = False
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        self.logger.info("键盘控制器已停止")

    def wait(self):
        # 等待退出事件
        if self.keyboard_listener and self.keyboard_listener.listener:
            try:
                # 等待退出事件被设置
                self.exit_event.wait()
            except KeyboardInterrupt:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def main():
    try:
        config = KeyboardConfig()
        with KeyboardController(config) as controller:
            controller.wait()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        os._exit(0)
    except Exception as e:
        logging.error(f"程序运行错误: {e}")
        os._exit(1)

if __name__ == "__main__":
    main()