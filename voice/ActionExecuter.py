import multiprocessing as mp
import copy
import os, sys
from config import Config
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
        if Config.ROBOT_AVAILABLE: 
            # 初始化机械臂
            import xapi.api as x5
            from action_sequence.execute_action import wave, bow, Nod, Shake_head
            self.handle_l = x5.connect(robot_ip_left)
            self.handle_r = x5.connect(robot_ip_right)
            self.add_data_1 = x5.MovPointAdd(vel=100, acc=100)
            self.add_data_2 = x5.MovPointAdd(vel=100, cnt=100, acc=100, dec=100, offset =-1, offset_data=(10,0,0,0,0,0,0,0,0))
            # 初始化手
            from controller.hand_controller import InspireHandR
            import time
            from action_sequence.PP import init_robot, pick_1_5
            self.hand_l = InspireHandR(port="COM11", baudrate=115200, hand_id=1)
            self.hand_r = InspireHandR(port="COM12", baudrate=115200, hand_id=2)
            self.hand_l.set_default_speed(100,100,100,100,100,100)
            self.hand_r.set_default_speed(200,200,200,200,200,200)

            # 导入动作函数
            self.init_robot = init_robot
            self.greeting = wave
            self.shaking_head = Shake_head
            self.nodding = Nod
            self.bowing = bow
            self.get_drink = pick_1_5
            
            print(f"已连接到机器人: {robot_ip_left} 和 {robot_ip_right}")
            self.init_robot(self.handle_l, self.handle_r, self.add_data_1, self.hand_l, self.hand_r)
            print("handle_l:", self.handle_l)
            print("handle_r:", self.handle_r)
        else:
            print("机器人控制不可用")
    
    def execute_action(self, action: str, pos_list: list = None) -> bool:
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
                print("执行：拿饮料")
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
            elif action == "get_drink":
                print("执行：拿饮料")
                # result = self.get_drink(self.handle_l, self.handle_r,self.add_data_1)
            # elif action == "others":
            #     return True
            else:
                return False
            
            return True
            
        except Exception as e:
            print(f"执行动作失败: {e}")
            return False

    def execute_get_drink(self, pos_list: list = None, layer_number: int = None, head_angle: float = None, body_distance: float = None) -> bool:
        if pos_list is None: # 到达对应层数
            pass
        else: # 到达对应位置
            for pos in pos_list:
                if pos == 5:
                    self.get_drink(self.handle_l, self.handle_r, self.add_data_1, self.hand_l, self.hand_r)
        return True