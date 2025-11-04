import xapi.api as x5
import copy
import time
import os
import sys

# 添加父目录到路径，确保可以正确导入controller模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from controller.hand_controller import InspireHandR

except ImportError as e:
    print(f"导入InspireHandR失败，请检查controller/hand_controller.py。错误信息: {e}")
    # 如果导入失败，创建占位类以避免NameError
    class InspireHandR:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InspireHandR类导入失败，请检查controller/hand_controller.py文件是否存在且无语法错误。")

def accept_letter(handle_l, handle_r, hand_l, hand_r, add_data):
    accept_joint_l = x5.Joint(j1 =-41.416,j2 = -44.899, j3 = -20.344, j4 = -87.096, 
     j5 = -4.514, j6 = -15.965, e1 = 43.839, e2=0, e3=0)

    x5.movj(handle_l, accept_joint_l, add_data)
    x5.wait_move_done(handle_l)
    
    accept_joint_r = x5.Joint(j1 =40.756,j2 = -46.892, j3 = 18.806, j4 = -83.806, 
     j5 = 25.688, j6 = -17.893, e1 = -75.856, e2=0, e3=123)

    x5.movj(handle_r, accept_joint_r, add_data)
    x5.wait_move_done(handle_r)

    hand_l.setpos(793, 857, 895, 902, 53, 0)
    hand_r.setpos(793, 857, 895, 902, 53, 0)
    

if __name__ == "__main__":
    hand_l = InspireHandR(port="COM12", baudrate=115200, hand_id=1)
    hand_r = InspireHandR(port="COM14", baudrate=115200, hand_id=2)
    hand_l.set_default_speed(100,100,100,100,100,100)
    hand_r.set_default_speed(200,200,200,200,200,200)
    hand_r.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)
    # hand_r.setpos(472,509,589,670,736,0)
    # time.sleep(1)
    add_data_1 = x5.MovPointAdd(vel=100, acc=100)
    add_data_2 = x5.MovPointAdd(vel=100, cnt=100, acc=100, dec=100, offset =-1,
    offset_data=(10,0,0,0,0,0,0,0,0))
    # # 连接机器人
    handle_l = x5.connect("192.168.1.9")
    handle_r = x5.connect("192.168.1.10")
    
    accept_letter(handle_l, handle_r, hand_l, hand_r, add_data_1)
