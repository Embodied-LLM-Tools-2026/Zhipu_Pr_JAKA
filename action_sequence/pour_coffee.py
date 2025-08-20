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
    from controller.gripper_controller import GripperController
    from action_sequence.agv_client import AGVClient
except ImportError as e:
    print(f"导入InspireHandR或AGVClient失败，请检查controller/hand_controller.py和controller/AGV_controller.py路径。错误信息: {e}")
    # 如果导入失败，创建占位类以避免NameError
    class InspireHandR:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InspireHandR类导入失败，请检查controller/hand_controller.py文件是否存在且无语法错误。")
    class AGVClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("AGVClient类导入失败，请检查controller/AGV_controller.py文件是否存在且无语法错误。")


# 初始化右臂
INIT_POS_R = x5.Pose(x=-200, y=-245, z=5, a=-120, b=-5, c=-115, e1=90, e2=0, e3=160)
INIT_POINT_R = x5.Point(pose=INIT_POS_R, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_R = x5.Joint(j1=-7.277, j2=-80.381, j3=97.428, j4=-91.135, j5=-24.051, j6=-74.093, e1=12.445-90, e2=0, e3=160)

# 初始化左臂
INIT_POS_L = x5.Pose(x=-200, y=245, z=5, a=120, b=5, c=115, e1=-90, e2=0, e3=0)
INIT_POINT_L = x5.Point(pose=INIT_POS_L, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_L = x5.Joint(j1=4.927, j2=-80.496, j3=-97.021, j4=-87.812, j5=22.909, j6=-79.830, e1=-20.340+90, e2=0, e3=0)

def init_robot(handle_l, handle_r, add_data, hand_l, hand_r):
    # 初始化左臂
    x5.movj(handle_l, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_l)
    # 初始化右臂
    x5.movj(handle_r, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_r)
    hand_r.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)

def move_to_coffee_bar(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    """
    pass

def press_cup_extractor_button(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    按取杯器按钮
    """

    # 握拳
    hand_l.setpos(0,0,0,0,0,0)
    time.sleep(2)
    # pre 按按钮动作1（上扬）
    pick_2 = x5.Joint(j1 =-39.069,j2 = -70.266, j3 = -86.151, j4 = -85.395, 
     j5 = 79.926, j6 = -23.361, e1 = 69.704, e2=0, e3=0)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # pre 按按钮动作2（靠近按钮）
    pick_3 = x5.Joint(j1 = -27.298, j2 = -73.639, j3 = -101.255, j4 = -97.217, 
                      j5 = 122.885, j6 = -13.852, e1 = 120.141, e2 = -0.002, e3 = -0.001)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 按按钮
    pick_2 = x5.Joint(j1 = -27.475, j2 = -74.293, j3 = -104.427, j4 = -96.909, 
                      j5 = 122.861, j6 = -13.777, e1 = 120.134, e2 = -0.003, e3 = -0.001)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 远离按钮动作
    pick_3 = x5.Joint(j1 = -27.298, j2 = -73.639, j3 = -101.255, j4 = -97.217, 
                      j5 = 122.885, j6 = -13.852, e1 = 120.141, e2 = -0.002, e3 = -0.001)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

def get_coffee_cup(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    从取杯器中拿出杯子（未写完）
    """

    # pre 按按钮
    pick_1 = x5.Joint(j1 = -27.298, j2 = -73.639, j3 = -101.255, j4 = -97.217, 
                      j5 = 122.885, j6 = -13.852, e1 = 120.141, e2 = -0.002, e3 = -0.001)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)


    hand_l.setpos(1000,1000,1000,1000,1000,200)
    time.sleep(2)

    # pre 拿杯子动作1（远离拿杯子的位置）
    pick_1 = x5.Joint(j1 = -14.389, j2 = -73.388, j3 = -105.649, j4 = -95.166, 
                      j5 = 124.675, j6 = -29.278, e1 = 120.14, e2 = -0.003, e3 = -0.003)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    # pre 拿杯子动作2（稍微远离拿杯子的位置）
    pick_2 = x5.Joint(j1 = 4.048, j2 = -75.777, j3 = -118.218, j4 = -98.083, 
                      j5 = 79.486, j6 = -13.469, e1 = 120.133, e2 = -0.004, e3 = -0.003)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # pre 拿杯子动作3（稍微靠近拿杯子的位置）
    pick_3 = x5.Joint(j1 = 10.475, j2 = -79.828, j3 = -127.527, j4 = -99.766,
                       j5 = 58.134, j6 = -10.202, e1 = 120.135, e2 = -0.005, e3 = -0.004)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # pre 拿杯子动作4（稍微靠近拿杯子的位置）
    pick_4 = x5.Joint(j1 = -12.068, j2 = -83.1, j3 = -130.269, j4 = -84.751, 
                      j5 = 55.787, j6 = -21.082, e1 = 120.133, e2 = -0.013, e3 = -0.008)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

    # 拿杯子
    pick_5 = x5.Joint(j1 = -19.188, j2 = -76.922, j3 = -124.378, j4 = -75.898, 
                      j5 = 62.795, j6 = -18.854, e1 = 120.136, e2 = -0.014, e3 = -0.003)
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)

    # 合爪子
    hand_l.setpos(723,729,474,477,917,0)
    time.sleep(2)

    # 拿出杯子
    pick_6 = x5.Joint(j1 = -17.606, j2 = -83.217, j3 = -113.746, j4 = -76.159, 
                      j5 = 62.809, j6 = -29.866, e1 = 120.135, e2 = -0.016, e3 = -0.007)
    x5.movj(handle_L, pick_6, add_data)
    x5.wait_move_done(handle_L)

    # 在杯托上方
    pick_7 = x5.Joint(j1 = -40.249, j2 = -77.03, j3 = -109.008, j4 = -55.659, 
                      j5 = 87.595, j6 = -32.462, e1 = 96.092, e2 = -0.014, e3 = -0.003)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

def put_coffee_cup(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    将杯子从桌面上拿到咖啡机接水处
    """

    # 回到初始化
    pick_1 = INIT_JOINT_L 
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)
    hand_l.setpos(1000, 1000, 1000, 1000, 1000, 0)

    # pre 拿杯子动作1（上扬）
    pick_2 = x5.Joint(j1 =-43.836,j2 = -40.784, j3 = -69.693, j4 = -75.668, 
     j5 = 43.75, j6 = -15.309, e1 = 87.207, e2=0, e3=0)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # pre 拿杯子动作2（临近杯子上方）
    pick_3 = x5.Joint(j1 =-39.853,j2 = -52.416, j3 = -62.818, j4 = -64.683, 
     j5 = 43.744, j6 = -1.954, e1 = 87.205, e2=0, e3=0)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 拿杯子
    pick_4 = x5.Joint(j1 =-47.123,j2 = -51.262, j3 = -52.629, j4 = -51.509, 
     j5 = 45.781, j6 = -14.476, e1 = 87.210, e2=0, e3=0)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

    # 合手
    hand_l.setpos(750, 700, 700, 650, 500, 0)
    time.sleep(1)

    # 举起杯子
    pick_5 = x5.Joint(j1 =-53.213,j2 = -49.375, j3 = -49.728, j4 = -50.686, 
     j5 = 28.693, j6 = -19.63, e1 = 87.212, e2=0, e3=0)
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)


    # pre 放杯子（咖啡机接水处）
    pick_6 = x5.Joint(j1 =-62.378,j2 = -58.391, j3 = -26.807, j4 = -50.763, 
     j5 = 23.412, j6 = -10.574, e1 = 87.205, e2=0, e3=0)
    x5.movj(handle_L, pick_6, add_data)
    x5.wait_move_done(handle_L)


    # 放杯子（咖啡机接水处）
    pick_7 = x5.Joint(j1 =-59.618,j2 = -75.203, j3 = -38.432, j4 = -43.136, 
     j5 = 35.257, j6 = 0.072, e1 = 87.208, e2=0, e3=0)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    # 张开手
    hand_l.setpos(1000, 1000, 1000, 1000, 1000, 0)
    time.sleep(2)

    # 收手动作1（调整手腕角度，稍微远离放杯处）
    pick_2 = x5.Joint(j1 =-58.955,j2 = -75.194, j3 = -38.530, j4 = -44.154, 
     j5 = 45.847, j6 = 2.131, e1 = 87.206, e2=0, e3=0)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 收手动作2（稍微远离放杯处）
    pick_3 = x5.Joint(j1 =-50.525,j2 = -63.152, j3 = -41.353, j4 = -60.378, 
     j5 = 34.733, j6 = 0.367, e1 = 87.205, e2=0, e3=0)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)


def press_button(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    点击做咖啡按钮
    """

    # 回到初始化
    pick_3 = INIT_JOINT_L 
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)
    hand_l.setpos(1000, 1000, 1000, 1000, 1000, 0)
    time.sleep(2)


    # pre 点击出咖啡按钮
    pick_1 = x5.Joint(j1 =-54.693,j2 = -57.803, j3 = -93.615, j4 = -71.047, 
     j5 = 70.896, j6 = -29.662, e1 = 87.206, e2=0, e3=0)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    hand_l.setpos(0,0,0,1000,0,0)
    time.sleep(2)

    # 点击按钮
    pick_2 = x5.Joint(j1 =-65.037,j2 = -63.601, j3 = -59.010, j4 = -71.290, 
     j5 = 110.391, j6 = -22.381, e1 = 88.551, e2=0, e3=0)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    time.sleep(1)

    # 远离点击按钮
    pick_3 = x5.Joint(j1 =-54.693,j2 = -57.803, j3 = -93.615, j4 = -71.047, 
     j5 = 70.896, j6 = -29.662, e1 = 87.206, e2=0, e3=0)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 回到初始化
    pick_4 = INIT_JOINT_L 
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)
    
    # 手部回到初始状态
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)
    # pass

def get_coffee_cup_with_coffee(handle_L,handle_R,hand_l,hand_r,add_data):   
    """
    将做好的咖啡从咖啡机接水处拿走
    """

    # 回到初始化
    pick_1 = INIT_JOINT_L 
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)
    hand_l.setpos(1000, 1000, 1000, 1000, 1000, 0)

    # 靠近动作1（远离放杯处） #TODO: 修改动作序号
    pick_3 = x5.Joint(j1 =-39.853,j2 = -52.416, j3 = -62.818, j4 = -64.683, 
     j5 = 43.744, j6 = -1.954, e1 = 87.205, e2=0, e3=0)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    hand_l.setpos(820, 820, 820, 820, 1000, 0)
    time.sleep(2)

    # # 靠近动作1（稍微远离放杯处）
    # pick_2 = x5.Joint(j1 =-50.525,j2 = -63.152, j3 = -41.353, j4 = -60.378, 
    #  j5 = 34.733, j6 = 0.367, e1 = 87.205, e2=0, e3=0)
    # x5.movj(handle_L, pick_2, add_data)
    # x5.wait_move_done(handle_L)


    # 靠近动作2（靠近放杯处）
    pick_3 = x5.Joint(j1 =-50.472,j2 = -67.304, j3 = -44.792, j4 = -56.411, 
     j5 = 32.597, j6 = 3.897, e1 = 87.189, e2=0, e3=0)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 抓杯子（咖啡机接水处）
    pick_4 = x5.Joint(j1 =-59.618,j2 = -75.203, j3 = -38.432, j4 = -43.136, 
     j5 = 35.257, j6 = 0.072, e1 = 87.208, e2=0, e3=0)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

    hand_l.setpos(750, 700, 700, 650, 500, 0)

    # 升起杯子（咖啡机接水处）
    pick_5 = x5.Joint(j1 =-62.378,j2 = -58.391, j3 = -26.807, j4 = -50.763, 
     j5 = 23.412, j6 = -10.574, e1 = 87.205, e2=0, e3=0)
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)


    # pre 放杯子动作（临近放杯子位置上方）
    pick_6 = x5.Joint(j1 =-39.853,j2 = -52.416, j3 = -62.818, j4 = -64.683, 
     j5 = 43.744, j6 = -1.954, e1 = 87.205, e2=0, e3=0)
    x5.movj(handle_L, pick_6, add_data)
    x5.wait_move_done(handle_L)

    # 放杯子（将杯子放在目标位置）
    pick_7 = x5.Joint(j1 =-47.123,j2 = -51.262, j3 = -52.629, j4 = -51.509, 
     j5 = 45.781, j6 = -14.476, e1 = 87.210, e2=0, e3=0)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(2)

    # pre 离开放杯子位置动作1（临近放杯子位置上方）
    pick_8 = x5.Joint(j1 =-39.853,j2 = -52.416, j3 = -62.818, j4 = -64.683, 
     j5 = 43.744, j6 = -1.954, e1 = 87.205, e2=0, e3=0)
    x5.movj(handle_L, pick_8, add_data)
    x5.wait_move_done(handle_L)

    # pre 离开放杯子位置动作2（上扬）
    pick_9 = x5.Joint(j1 =-43.836,j2 = -40.784, j3 = -69.693, j4 = -75.668, 
     j5 = 43.75, j6 = -15.309, e1 = 87.207, e2=0, e3=0)
    x5.movj(handle_L, pick_9, add_data)
    x5.wait_move_done(handle_L)

    # 回到初始化
    pick_10 = INIT_JOINT_L 
    x5.movj(handle_L, pick_10, add_data)
    x5.wait_move_done(handle_L)
    hand_l.setpos(1000, 1000, 1000, 1000, 1000, 0)

 

def put_coffee_cup_with_coffee(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    """
    pass



if __name__ == "__main__":
    hand_l = InspireHandR(port="COM12", baudrate=115200, hand_id=1)
    hand_r = InspireHandR(port="COM14", baudrate=115200, hand_id=2)
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    hand_l.set_default_speed(500,500,500,500,500,500)
    # hand_r.set_default_speed(200,200,200,200,200,200)

    add_data_1 = x5.MovPointAdd(vel=100, acc=100)
    add_data_2 = x5.MovPointAdd(vel=100, cnt=100, acc=100, dec=100, offset =-1,
    offset_data=(10,0,0,0,0,0,0,0,0))
    # 连接机器人
    handle_l = x5.connect("192.168.1.9")
    handle_r = x5.connect("192.168.1.10")
    # put_coffee_cup(handle_l,handle_r,hand_l,hand_r,add_data_1)

    press_button(handle_l,handle_r,hand_l,hand_r,add_data_1)

    # get_coffee_cup_with_coffee(handle_l,handle_r,hand_l,hand_r,add_data_1)

    # # hand_r.setpos(1000,1000,1000,1000,1000,0)
    # # time.sleep(1)
    # # hand_l.setpos(1000,1000,1000,1000,1000,0)
    # # time.sleep(1)
    # # put_coffee_cup(handle_l,handle_r,hand_l,hand_r,add_data_1)
    # press_cup_extractor_button(handle_l,handle_r,hand_l,hand_r,add_data_1)
    # get_coffee_cup(handle_l,handle_r,hand_l,hand_r,add_data_1)