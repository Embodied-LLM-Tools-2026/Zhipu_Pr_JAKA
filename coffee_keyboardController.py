#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘控制器模块
用于监听小键盘输入并执行相应操作
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
import xapi.api as x5
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
        # 检查退出键
        if key in self.config.EXIT_KEYS:
            self.stop_callback()
            return False
        
        # 检查小键盘数字键
        if (hasattr(key, 'vk') and 
            key.vk in self.config.NUMPAD_VK_MAPPING):
            # 阻止按键的默认行为（显示字符）
            self.key_queue.put(key)
            # 不返回False，让监听器继续运行
            return True
        
        # 忽略其他所有按键
        return True
    
    def on_release(self, key):
        """按键释放事件处理"""
        # 只处理退出键
        if key in self.config.EXIT_KEYS:
            self.stop_callback()
            return False
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
    """键盘控制器类"""
    
    def __init__(self, config: Optional[KeyboardConfig] = None):
        """
        初始化键盘控制器
        
        Args:
            config: 键盘配置对象，如果为None则使用默认配置
        """
        self.config = config or KeyboardConfig()
        self.key_queue = Queue()
        self.is_processing = False
        self.is_running = False
        self.keyboard_listener = None
        self.processor_thread = None
        
        # 设置日志
        self._setup_logging()
        
        # 初始化动作映射
        self._setup_action_mapping()

        # 机械臂控制
        self.handle_l = x5.connect("192.168.1.9")
        self.handle_r = x5.connect("192.168.1.10")
        self.hand_l = InspireHandR(port="COM12", baudrate=115200, hand_id=1)
        self.hand_r = InspireHandR(port="COM14", baudrate=115200, hand_id=2)
        self.add_data_1 = x5.MovPointAdd(vel=100, acc=100)

    def _setup_logging(self):
        """设置日志配置"""
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
        put_coffee_cup(self.handle_l,self.handle_r,self.hand_l,self.hand_r,self.add_data_1)
        self.is_processing = False
        self.logger.info("执行动作1：完成")
    
    def _action_2(self):
        """动作2"""
        self.is_processing = True
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作2：开始")
        press_button(self.handle_l,self.handle_r,self.hand_l,self.hand_r,self.add_data_1)
        self.is_processing = False
        self.logger.info("执行动作2：完成")
    
    def _action_3(self):
        """动作3"""
        self.is_processing = True
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清理提示信息
        self.logger.info("执行动作3：开始")
        get_coffee_cup_with_coffee(self.handle_l,self.handle_r,self.hand_l,self.hand_r,self.add_data_1)
        self.is_processing = False
        self.logger.info("执行动作3：完成")
    
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
        """按键处理线程"""
        self.logger.info("按键处理线程启动")
        while self.is_running:
            try:
                # 使用超时机制，避免无限等待
                key = self.key_queue.get(timeout=1.0)  # 1秒超时
                if hasattr(key, 'vk') and key.vk in self.config.NUMPAD_VK_MAPPING:
                    num = self.config.NUMPAD_VK_MAPPING[key.vk]
                    # 显示按键反馈
                    print(f"\r[检测到按键 {num}] ", end="", flush=True)
                    self.logger.info(f"检测到小键盘按键: {num}")
                    
                    if num in self.action_mapping:
                        self.action_mapping[num]()
                    else:
                        self.logger.warning(f"未定义的动作: {num}")
                
                self.key_queue.task_done()
            except queue.Empty:
                # 超时是正常情况，不需要记录错误
                continue
            except Exception as e:
                self.logger.error(f"按键处理错误: {e}")
                continue
    
    def start(self):
        """启动键盘控制器"""
        if self.is_running:
            self.logger.warning("键盘控制器已在运行")
            return
        
        self.is_running = True
        
        # 启动处理线程
        self.processor_thread = Thread(target=self.key_processor, daemon=True)
        self.processor_thread.start()
        
        # 启动专门的键盘监听器
        self.keyboard_listener = WindowsKeyboardListener(
            self.config, 
            self.key_queue, 
            self.stop
        )
        self.keyboard_listener.start()
        
        self.logger.info("键盘控制器启动成功，开始监听小键盘输入...")
        self.logger.info(f"按 {', '.join(str(k) for k in self.config.EXIT_KEYS)} 键退出")
    
    def stop(self):
        """停止键盘控制器"""
        self.is_running = False
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        
        self.logger.info("键盘控制器已停止")
    
    def wait(self):
        """等待键盘控制器运行"""
        if self.keyboard_listener and self.keyboard_listener.listener:
            self.keyboard_listener.listener.join()
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()

def main():
    """主函数"""
    try:
        # 创建配置对象
        config = KeyboardConfig()
        
        # 使用上下文管理器启动键盘控制器
        with KeyboardController(config) as controller:
            controller.wait()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logging.error(f"程序运行错误: {e}")
        raise

if __name__ == "__main__":
    main()