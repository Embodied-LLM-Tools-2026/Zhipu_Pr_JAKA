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
                greet,
                shake_head,
                nod,
                bow
            )
            
            self.robot_lock = mp.Lock()
            self.robot = X1Interface(self.robot_lock, robot_ip, robot_port)
            
            # 导入动作函数
            self.greeting = greet
            self.shaking_head = shake_head
            self.nodding = nod
            self.bowing = bow
            
            print(f"已连接到机器人: {robot_ip}:{robot_port}")
        else:
            print("机器人控制不可用")
    
    def execute_action(self, action: str) -> bool:
        """执行动作"""
        if not self.robot:
            print("机器人不可用，模拟执行动作")
            if action == "greet":
                print("执行：打招呼")
            elif action == "shake_head":
                print("执行：摇头")
            elif action == "nod":
                print("执行：点头")
            elif action == "bow":
                print("执行：鞠躬")
            # elif action == "others":
            #     return True
            else:
                return False
            return True
        
        try:
            if action == "greet":
                print("执行：打招呼")
                result = self.greet(self.robot)
            elif action == "shake_head":
                print("执行：摇头")
                result = self.shake_head(self.robot)
            elif action == "nod":
                print("执行：点头")
                result = self.nod(self.robot)
            elif action == "bow":
                print("执行：鞠躬")
                result = self.bow(self.robot)
            # elif action == "others":
            #     return True
            else:
                return False
            
            return result == 0
            
        except Exception as e:
            print(f"执行动作失败: {e}")
            return False
