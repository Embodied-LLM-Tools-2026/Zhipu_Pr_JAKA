import time
from config import Config
import multiprocessing as mp

# ================================
# 根据文本指令控制机器人执行固定动作
# ================================

class ActionExecuter:
    """机器人控制器"""
    
    def __init__(self, robot_ip: str, robot_port: int, robot_available: bool):
        self.robot = None
        self.robot_lock = None
        
        # if deps.robot_available:
        if robot_available: #调试时要模拟执行动作就改为False
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
