import multiprocessing as mp
import copy
import os, sys
from voice.config import Config

# ================================
# 根据文本指令控制机器人执行固定动作（基础版本）
# ================================

# 将父目录临时注册为系统路径，方便python导入模块
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class ActionExecuter:
    """机器人控制器（基础版本 - 仅包含问好动作）"""

    def __init__(self, robot_ip_left: str, robot_ip_right: str, robot_available: bool):
        self.handle_l = None
        self.handle_r = None

        if Config.ROBOT_AVAILABLE:
            # 初始化机械臂
            import xapi.api as x5
            from action_sequence.execute_action import wave, bow, Nod, Shake_head, init_robot
            self.handle_l = x5.connect(robot_ip_left)
            self.handle_r = x5.connect(robot_ip_right)
            self.add_data_1 = x5.MovPointAdd(vel=100, acc=100)
            self.add_data_2 = x5.MovPointAdd(
                vel=100, cnt=100, acc=100, dec=100, offset=-1,
                offset_data=(10, 0, 0, 0, 0, 0, 0, 0, 0)
            )

            # 导入基础动作函数（仅问好动作）
            self.init_robot = init_robot
            self.greeting = wave
            self.shaking_head = Shake_head
            self.nodding = Nod
            self.bowing = bow

            print(f"已连接到机器人: {robot_ip_left} 和 {robot_ip_right}")
            self.init_robot(self.handle_l, self.handle_r, self.add_data_1)
            print("handle_l:", self.handle_l)
            print("handle_r:", self.handle_r)
        else:
            print("机器人控制不可用")

    def execute_action(self, action: str) -> bool:
        """执行动作"""
        print("handle_l:", self.handle_l)
        print("handle_r:", self.handle_r)
        if self.handle_l is None or self.handle_r is None:
            print("机器人不可用，模拟执行动作")
            if action == "greet":
                print("执行：打招呼")
            elif action == "shake_head":
                print("执行：摇头")
            elif action == "nod":
                print("执行：点头")
            elif action == "bow":
                print("执行：鞠躬")
            elif action == "get_drink":
                print("❌ 基础版本不支持拿取饮料功能")
                return False
            else:
                return False
            return True

        try:
            if action == "greet":
                print("执行：打招呼")
                self.greeting(self.handle_l, self.handle_r, self.add_data_1)
            elif action == "shake_head":
                print("执行：摇头")
                self.shaking_head(self.handle_l, self.handle_r, self.add_data_2)
            elif action == "nod":
                print("执行：点头")
                self.nodding(self.handle_l, self.handle_r, self.add_data_2)
            elif action == "bow":
                print("执行：鞠躬")
                self.bowing(self.handle_l, self.handle_r, self.add_data_1)
            elif action == "get_drink":
                print("❌ 基础版本不支持拿取饮料功能")
                return False
            else:
                return False
            
            return True
            
        except Exception as e:
            print(f"执行动作失败: {e}")
            return False 