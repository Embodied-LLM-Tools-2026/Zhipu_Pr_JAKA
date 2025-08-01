import multiprocessing as mp
import xapi.api as x5
import copy
import os, sys
# ================================
# 根据文本指令控制机器人执行固定动作
# ================================
# 将父目录临时注册为系统路径，方便python导入模块
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
class ActionExecuter:
    """机器人控制器"""
    
    def __init__(self, robot_ip_left: str, robot_ip_right: str, robot_available: bool):
        self.handle_l = None
        self.handle_r = None
        
        # if deps.robot_available:
        if robot_available: #调试时要模拟执行动作就改为False
            from action_sequence.execute_action import init_robot, wave, bow, Nod, Shake_head
            
            self.handle_l = x5.connect(robot_ip_left)
            self.handle_r = x5.connect(robot_ip_right)
            self.add_data_1 = x5.MovPointAdd(vel=100, acc=100)
            self.add_data_2 = x5.MovPointAdd(vel=100, cnt=100, acc=100, dec=100, offset =-1, offset_data=(10,0,0,0,0,0,0,0,0))
            
            # 导入动作函数
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
            # elif action == "others":
            #     return True
            else:
                return False
            return True
        
        try:
            if action == "greet":
                print("执行：打招呼")
                result = self.greeting(self.handle_l, self.handle_r,self.add_data_1)
            elif action == "shake_head":
                print("执行：摇头")
                result = self.shaking_head(self.handle_l, self.handle_r,self.add_data_2)
            elif action == "nod":
                print("执行：点头")
                result = self.nodding(self.handle_l, self.handle_r,self.add_data_2)
            elif action == "bow":
                print("执行：鞠躬")
                result = self.bowing(self.handle_l, self.handle_r,self.add_data_1)
            # elif action == "others":
            #     return True
            else:
                return False
            
            return result == 0
            
        except Exception as e:
            print(f"执行动作失败: {e}")
            return False
