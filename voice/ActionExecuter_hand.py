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
            from action_sequence.execute_action import wave, bow, Nod, Shake_head, rotate_head_to_angle
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
            
            # 尝试导入所有可能的pick函数
            import action_sequence.PP_hand as pp_hand_module
            for layer in range(1, 6):  # layer_number: 1~4
                for drink in range(1, 6):  # drink_id: 1~5
                    function_name = f"pick_{layer}_{drink}"
                    try:
                        if hasattr(pp_hand_module, function_name):
                            pick_function = getattr(pp_hand_module, function_name)
                            setattr(self, function_name, pick_function)
                            print(f"成功导入函数: {function_name}")
                        else:
                            print(f"跳过不存在的函数: {function_name}")
                    except Exception as e:
                        print(f"导入函数 {function_name} 时出错: {e}")
            from action_sequence.pour_coffee import (
                move_to_coffee_machine_and_make_coffee,
                get_coffee_and_serve,
            )
            self.hand_l = InspireHandR(port="COM12", baudrate=115200, hand_id=1)
            self.hand_r = InspireHandR(port="COM14", baudrate=115200, hand_id=2)
            self.hand_l.set_default_speed(100, 100, 100, 100, 100, 100)
            self.hand_r.set_default_speed(200, 200, 200, 200, 200, 200)

            # 导入动作函数
            self.init_robot = init_robot
            self.greeting = wave
            self.shaking_head = Shake_head
            self.nodding = Nod
            self.bowing = bow
            self.move_to_pick_height_pitch_angle = move_to_pick_height_pitch_angle
            self.move_to_shelf = move_to_shelf
            self.back_bar_station = back_bar_station
            self.move_to_coffee_machine_and_make_coffee = move_to_coffee_machine_and_make_coffee
            self.get_coffee_and_serve = get_coffee_and_serve
            self.rotate_head_to_angle = rotate_head_to_angle
            
            print(f"已连接到机器人: {robot_ip_left} 和 {robot_ip_right}")
            self.init_robot(self.handle_l, self.handle_r, self.add_data_1, self.hand_l, self.hand_r)
            self.back_bar_station()
            print("handle_l:", self.handle_l)
            print("handle_r:", self.handle_r)
        else:
            print("机器人控制不可用")

    def execute_action(self, action: str, angle=None, incremental=False, back_to_init=False) -> bool:
        """执行动作"""
        print("handle_l:", self.handle_l)
        print("handle_r:", self.handle_r)
        if self.handle_l is None or self.handle_r is None:
            print("机器人不可用，模拟执行动作")
            if action == "greet":
                print("执行：打招呼")
            elif action == "shake_head":
                print("执行：摇头")
            elif action == "rotate_head_to_angle":
                print("执行：转头")
            elif action == "nod":
                print("执行：点头")
            elif action == "bow":
                print("执行：鞠躬")
            elif action == "get_drink":
                print("执行：拿饮料")
            else:
                return False
            return True

        try:
            if action == "greet":
                print("执行：打招呼")
                _ = self.greeting(self.handle_l, self.handle_r, self.add_data_1)
            elif action == "shake_head":
                print("执行：摇头")
                _ = self.shaking_head(self.handle_l, self.handle_r, self.add_data_2)
            elif action == "rotate_head_to_angle":
                print("执行：转头")
                _ = self.rotate_head_to_angle(self.handle_l, self.handle_r, self.add_data_1, angle=angle, incremental=incremental, back_to_init=back_to_init)
            elif action == "nod":
                print("执行：点头")
                _ = self.nodding(self.handle_l, self.handle_r, self.add_data_2)
            elif action == "bow":
                print("执行：鞠躬")
                _ = self.bowing(self.handle_l, self.handle_r, self.add_data_1)
            elif action == "get_drink":
                print("执行：拿饮料")
            else:
                return False

            return True

        except Exception as e:
            print(f"执行动作失败: {e}")
            return False

    # 拿一瓶饮料
    def execute_get_drink(
        self,
        drink_id: int = None,
        layer_number: int = None,
        head_angle: float = None,
        body_distance: float = None,
    ) -> bool:
        try:
            if self.handle_l is None or self.handle_r is None:
                print("机器人不可用")
                return True

            if drink_id is None:  # 到达货架再到达对应层数
                self.move_to_shelf()
                self.move_to_pick_height_pitch_angle(
                    self.handle_l, self.handle_r, self.add_data_1, body_distance, head_angle
                )
                print("到达对应层数")
            else:  # 到达对应位置
                self.move_to_shelf()
                self.move_to_pick_height_pitch_angle(
                    self.handle_l, self.handle_r, self.add_data_1, body_distance, head_angle
                )
                # 检查layer_number和drink_id是否在有效范围内
                if isinstance(layer_number, int) and 1 <= layer_number <= 5 and isinstance(drink_id, int) and 1 <= drink_id <= 5:
                    # 动态调用pick函数
                    pick_function_name = f"pick_{layer_number}_{drink_id}"
                    if hasattr(self, pick_function_name):
                        pick_function = getattr(self, pick_function_name)
                        pick_function(
                            self.handle_l,
                            self.handle_r,
                            self.hand_l,
                            self.hand_r,
                            self.add_data_1,
                        )
                    else:
                        print(f"警告：函数 {pick_function_name} 不存在")
                        return False
                else:
                    print(f"无效的参数：layer_number={layer_number}, drink_id={drink_id}")
                    return False
                self.init_robot(self.handle_l, self.handle_r, self.add_data_1, self.hand_l, self.hand_r)
                print("到达对应位置")
            return True
        except Exception as e:
            print(f"执行失败: {e}")
            return False

    def back_to_init_height_and_angle(self):
        self.move_to_pick_height_pitch_angle(self.handle_l, self.handle_r, self.add_data_1, 160, 0)

