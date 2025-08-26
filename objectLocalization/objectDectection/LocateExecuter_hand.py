import multiprocessing as mp
import copy
import os, sys
from voice.config import Config

# ================================
# 根据文本指令控制机器人执行固定动作（灵巧手版本）
# ================================

# 将父目录临时注册为系统路径，方便python导入模块
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class ActionExecuter:
    """机器人控制器（灵巧手）"""

    def __init__(self, robot_ip_left: str, robot_ip_right: str, robot_available: bool):
        self.handle_l = None
        self.handle_r = None

        if Config.ROBOT_AVAILABLE:
            # 初始化机械臂
            import xapi.api as x5
            self.handle_l = x5.connect(robot_ip_left)
            self.handle_r = x5.connect(robot_ip_right)
            self.add_data_1 = x5.MovPointAdd(vel=100, acc=100)
            self.add_data_2 = x5.MovPointAdd(
                vel=100, cnt=100, acc=100, dec=100, offset=-1,
                offset_data=(10, 0, 0, 0, 0, 0, 0, 0, 0)
            )

            # 初始化灵巧手
            from controller.hand_controller import InspireHandR
            import time
            from action_sequence.PP_hand import (
                init_robot,
                move_to_pick_height_pitch_angle,
                move_to_shelf,
                back_bar_station,
            )
            
            self.hand_l = InspireHandR(port="COM12", baudrate=115200, hand_id=1)
            self.hand_r = InspireHandR(port="COM14", baudrate=115200, hand_id=2)
            self.hand_l.set_default_speed(100, 100, 100, 100, 100, 100)
            self.hand_r.set_default_speed(200, 200, 200, 200, 200, 200)

            # 导入动作函数
            self.init_robot = init_robot
            self.move_to_pick_height_pitch_angle = move_to_pick_height_pitch_angle
            self.move_to_shelf = move_to_shelf
            self.back_bar_station = back_bar_station
            
            print(f"已连接到机器人: {robot_ip_left} 和 {robot_ip_right}")
            self.init_robot(self.handle_l, self.handle_r, self.add_data_1, self.hand_l, self.hand_r)
            # self.back_bar_station()
            print("handle_l:", self.handle_l)
            print("handle_r:", self.handle_r)
        else:
            print("机器人控制不可用")


    def back_to_init_height_and_angle(self):
        self.move_to_pick_height_pitch_angle(self.handle_l, self.handle_r, self.add_data_1, 160, 0)

