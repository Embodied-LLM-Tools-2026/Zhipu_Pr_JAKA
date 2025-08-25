import xapi.api as x5
import copy
import time
import os
import sys
import numpy as np

# X_LRRobot = np.array([0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]) 
"""
P_LRRobot = x:0.3, y:0.2, z:0, X_LRRobot.inv = np.array([0,1,0],[0,0,1],[1,0,0]) 
"""

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
INIT_POS_R = x5.Pose(x=-145.160249452737, y=-199.35900829533145, z=84.71502397446304, a=160.6069585824988, b=-85.4579200755279, c=14.351717712227035, e1=96.11870531058125, e2=-0.0006866455078125, e3=-0.010833740234375)
INIT_POINT_R = x5.Point(pose=INIT_POS_R, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_R = x5.Joint(j1 = -6.362, j2 = -81.738, j3 = 104.467, j4 = -95.218, j5 = 4.181, j6 = -85.024, e1 = -77.554, e2 = -0.001, e3 = -0.011)

# 初始化左臂
INIT_POS_L = x5.Pose(x=-184.90416189951029, y=203.02375649010347, z=107.66267344899968, a=66.04016912016178, b=-86.68490315803206, c=121.8029965956341, e1=-99.97461614507827, e2=-0.0006866455078125, e3=0.0061798095703125)
INIT_POINT_L = x5.Point(pose=INIT_POS_L, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_L = x5.Joint(j1 = 5.694, j2 = -82.975, j3 = -106.95, j4 = -84.678, j5 = -7.358, j6 = -90.755, e1 = 69.521, e2 = -0.001, e3 = 0.006)

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
    with AGVClient(ip='192.168.1.50') as agv:
        # agv.go_to_point_in_world(0.420,0.038,0, 0)
        pose_result = agv.get_pose()
        x, y, angle = pose_result
        print("pose_result = ",pose_result)

    delta_x = (x + 0.153)*1000
    delta_y = (y + 1.027)*1000
    # delta_angle = (angle + )
    print(delta_x)
    print(delta_y)

    # 回到初始化
    pick_1 = INIT_JOINT_L 
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)
    hand_l.setpos(1000, 1000, 1000, 1000, 1000, 0)

    INIT_JOINT_R = x5.Joint(j1=-7.277, j2=-80.381, j3=97.428, j4=-91.135, j5=-24.051, j6=-74.093, e1=12.445-90, e2=0, e3=0)
    x5.movj(handle_R, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_R)

    # pre 拿杯子动作1（修改）
    pick_2 = x5.Joint(j1 = -33.254, j2 = -58.097, j3 = -64.264, j4 = -80.521, 
                      j5 = 37.83, j6 = -23.082, e1 = 86.889, e2 = -0.004, e3 = 0.001)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # pre 拿杯子动作2（修改）
    pick_3 = x5.Joint(j1 = -39.923, j2 = -51.457, j3 = -67.319, j4 = -67.409, 
                      j5 = 42.58, j6 = -1.991, e1 = 87.206, e2 = -0.005, e3 = -0.001)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 拿杯子
    pick_4 = x5.Joint(j1 =-47.123,j2 = -51.262, j3 = -52.629, j4 = -51.509, 
     j5 = 45.781, j6 = -14.476, e1 = 87.210, e2=0, e3=0)
    


    pick_4_point = x5.Pose(x=-192.26252193008182, y=538.3044279671338, z=158.26904829683812, a=-103.3156725565794, b=4.746906928726649, c=5.1939242251689555, e1=-42.67976105448165, e2=0.0006866455078125, e3=-0.001373291015625)
    pick_4_point = x5.Point(pose=pick_4_point, uf=0, tf=0, cfg=(0,0,0,7))

    # # 计算新的调整后的抓取点位  机器人y轴对应agv x轴，机器人右臂z轴对应agv y轴
    new_pick_4_point = copy.deepcopy(pick_4_point)
    new_pick_4_point.pose.z = 158.26904829683812 - delta_y
    new_pick_4_point.pose.y = 538.3044279671338 - delta_x
    print(new_pick_4_point)
    # 逆解，求笛卡尔点位p1的对应关节坐标
    jp4 = x5.cnvrt_j(handle_L, new_pick_4_point, 1, pick_4)

    x5.movj(handle_L, jp4, add_data)
    x5.wait_move_done(handle_L)

    # 合手
    hand_l.setpos(750, 700, 700, 650, 500, 0)
    time.sleep(0.5)

    # 举起杯子
    pick_5 = x5.Joint(j1 = -53.591, j2 = -53.49, j3 = -53.291, j4 = -51.393, 
                      j5 = 44.289, j6 = -17.586, e1 = 87.211, e2 = -0.003, e3 = 0.003)
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)


    # 放杯子（咖啡机接水处）
    pick_7 = x5.Joint(j1 = -56.851, j2 = -73.339, j3 = -41.695, j4 = -46.621, j5 = 50.705, j6 = 0.042, e1 = 87.174, e2 = -0.004, e3 = 0.019)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)


    # 张开手
    hand_l.setpos(1000, 1000, 1000, 1000, 1000, 0)
    time.sleep(0.5)


    # 收手动作2（稍微远离放杯处）
    pick_3 = x5.Joint(j1 = -51.748, j2 = -53.451, j3 = -51.876, j4 = -63.107, j5 = 52.991, j6 = -7.184, e1 = 87.174, e2 = -0.005, e3 = 0.019)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)


def press_button(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    点击做咖啡按钮
    """
    with AGVClient(ip='192.168.1.50') as agv:
        # agv.go_to_point_in_world(0.420,0.038,0, 0)
        pose_result = agv.get_pose()
        x, y, angle = pose_result
        print("pose_result = ",pose_result)

    delta_x = (x + 0.1553)*1000
    delta_y = (y + 1.0309)*1000
    # delta_angle = (angle + )
    print(delta_x)
    print(delta_y)
    # 靠近点击按钮1
    pick_3 = x5.Joint(j1 = -44.902, j2 = -70.286, j3 = -88.649, j4 = -61.257, 
                      j5 = 80.224, j6 = -19.512, e1 = 87.44, e2 = -0.021, e3 = 0.006)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    hand_l.setpos(0,0,0,1000,0,0)
    time.sleep(0.5)

    # 点击按钮
    pick_2 = x5.Joint(j1 = -64.502, j2 = -67.85, j3 = -60.973, j4 = -67.409, j5 = 110.475, j6 = -20.965, e1 = 88.546, e2 = -0.001, e3 = 0.021)
    
    pick_2_point = x5.Pose(x=22.379362617902128, y=563.2459797839849, z=-15.102345356584667, a=-134.83396548189097, b=49.43601815125561, c=-49.73207522866588, e1=-54.65169179081774, e2=-0.001373291015625, e3=0.020599365234375)
    pick_2_point = x5.Point(pose=pick_2_point, uf=0, tf=0, cfg=(0,0,0,7))
    target_agv_point = [-0.1575,-1.0214,-0.0074]
    from correct_hand import correct_left_arm
    P_hand_LeftArmCur = correct_left_arm(target_agv_point,pick_2_point)
    new_pick_2_point = x5.Point(pose=P_hand_LeftArmCur, uf=0, tf=0, cfg=(0,0,0,7))
    # 计算新的调整后的抓取点位  机器人y轴对应agv x轴，机器人右臂z轴对应agv y轴
    new_pick_2_point = copy.deepcopy(pick_2_point)
    new_pick_2_point.pose.z = -15.102345356584667 - delta_y
    new_pick_2_point.pose.y = 563.2459797839849 - delta_x
    print(new_pick_2_point)
    # 逆解，求笛卡尔点位p1的对应关节坐标
    jp1 = x5.cnvrt_j(handle_L, new_pick_2_point, 1, pick_2)

    x5.movj(handle_L, jp1, add_data)
    x5.wait_move_done(handle_L)





    # # 靠近点击按钮1
    # pick_3 = x5.Joint(j1 = -35.863, j2 = -72.277, j3 = -102.053, j4 = -77.859, 
    #                   j5 = 77.805, j6 = -28.431, e1 = 87.441, e2 = -0.006, e3 = 0.001)
    # x5.movj(handle_L, pick_3, add_data)
    # x5.wait_move_done(handle_L)

    # # 靠近动作1（远离放杯处） #TODO: 修改动作序号
    # pick_3 = x5.Joint(j1 = -41.434, j2 = -58.217, j3 = -62.943, j4 = -62.141, 
    #                   j5 = 44.192, j6 = -9.458, e1 = 87.205, e2 = -0.001, e3 = -0.001)
    # x5.movj(handle_L, pick_3, add_data)
    # x5.wait_move_done(handle_L)

    # hand_l.setpos(820, 820, 820, 820, 1000, 0)
    # time.sleep(0.5)



def get_coffee_cup_with_coffee(handle_L,handle_R,hand_l,hand_r,add_data):   
    """
    将做好的咖啡从咖啡机接水处拿走
    """
    hand_l.setpos(800, 800, 800, 820, 1000, 0)
    time.sleep(2)

    # 靠近动作（靠近放杯处）
    pick_2 = x5.Joint(j1 =-50.472,j2 = -67.304, j3 = -44.792, j4 = -56.411, 
     j5 = 32.597, j6 = 3.897, e1 = 87.189, e2=0, e3=0)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 抓杯子（咖啡机接水处）
    pick_3 = x5.Joint(j1 = -55.006, j2 = -74.22, j3 = -45.164, j4 = -47.319, 
                      j5 = 52.916, j6 = 1.635, e1 = 87.204, e2 = -0.005, e3 = 0.001)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    hand_l.setpos(750, 700, 700, 650, 500, 0)

    # 升起杯子（咖啡机接水处）
    pick_4 = x5.Joint(j1 =-62.378,j2 = -58.391, j3 = -26.807, j4 = -50.763, 
     j5 = 23.412, j6 = -10.574, e1 = 87.205, e2=0, e3=0)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)


    # 将杯子放在右手可以够住的地方的上方动作1
    pick_5 = x5.Joint(j1 = -57.206, j2 = -47.858, j3 = -31.67, j4 = -71.925, 
                      j5 = 25.901, j6 = -22.743, e1 = 87.203, e2 = -0.003, e3 = 0.002)
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)

    # 将杯子放在右手可以够住的地方的上方动作2
    pick_7 = x5.Joint(j1 = -51.803, j2 = -47.705, j3 = -39.376, j4 = -70.198, 
                      j5 = 38.053, j6 = -24.907, e1 = 87.35, e2 = -0.003, e3 = 0.001)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    # 将杯子放在右手可以够住的地方
    pick_9 = x5.Joint(j1 = -47.119, j2 = -49.864, j3 = -39.845, j4 = -67.525, 
                      j5 = 41.673, j6 = -24.622, e1 = 87.344, e2 = -0.005, e3 = 0.002)
    x5.movj(handle_L, pick_9, add_data)
    x5.wait_move_done(handle_L)

    # 松手
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(0.5)

    # 抬起手
    pick_9 = x5.Joint(j1 = -42.418, j2 = -43.11, j3 = -57.251, j4 = -75.078, j5 = 39.458, j6 = -31.697, e1 = 87.345, e2 = -0.005, e3 = 0.002)
    x5.movj(handle_L, pick_9, add_data)
    x5.wait_move_done(handle_L)
 

def put_coffee_cup_with_coffee(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    """
    hand_r.setpos(1000, 1000, 1000,1000,1000,0)

    ## 保证右手在初始化状态
    pick_1 = INIT_JOINT_R
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 靠近杯子动作
    pick_2 = x5.Joint(j1 = 48.476, j2 = -71.505, j3 = 52.333, j4 = -73.266, 
                      j5 = -43.73, j6 = -31.605, e1 = -93.41, e2 = 0, e3 = -0.022)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)
    
    # 抓握杯子动作
    pick_3 = x5.Joint(j1 = 47.083, j2 = -88.892, j3 = 55.147, j4 = -43.16, 
                      j5 = -53.034, j6 = -31.099, e1 = -96.923, e2 = 0.001, e3 = -0.032)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # 合手
    hand_r.setpos(750, 700, 700, 650, 500, 0)
    
    # 抬起水杯
    pick_2 = x5.Joint(j1 = 41.457, j2 = -54.411, j3 = 85.154, j4 = -65.744, 
                      j5 = -68.96, j6 = -53.4, e1 = -72.72, e2 = 0.002, e3 = -0.042)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # 放在置物台上方
    pick_2 = x5.Joint(j1 = 46.172, j2 = -71.418, j3 = 91.062, j4 = -52.335, j5 = -83.655, j6 = -34.416, e1 = -72.89, e2 = 0.005, e3 = -0.052)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)


    # 放在置物台上
    pick_2 = x5.Joint(j1 = 43.55, j2 = -73.231, j3 = 91.292, j4 = -51.396, j5 = -84.224, j6 = -31.195, e1 = -74.011, e2 = 0.006, e3 = -0.063)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # 松手
    hand_r.setpos(1000, 1000, 1000,1000,1000,0)
    time.sleep(0.5)

    # 抬起手臂
    pick_2 = x5.Joint(j1 = 44.826, j2 = -62.205, j3 = 102.228, j4 = -65.67, j5 = -81.574, j6 = -50.841, e1 = -74.014, e2 = 0.009, e3 = -0.073)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)
    
    # 回到初始化
    pick_3 = INIT_JOINT_R
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)


def move_to_coffee_machine_and_make_coffee(handle_l,handle_r,hand_l,hand_r,add_data_1):
    """
    移动到咖啡机并开始制作咖啡（按下按钮）。该函数返回后会播报语音"咖啡正在制作中，请您稍等片刻"
    """
    time.sleep(3)
    put_coffee_cup(handle_l,handle_r,hand_l,hand_r,add_data_1)

    press_button(handle_l,handle_r,hand_l,hand_r,add_data_1)

def get_coffee_and_serve(handle_l,handle_r,hand_l,hand_r,add_data_1):
    """
    将咖啡从咖啡机接水处拿走并放到传送带上。该函数返回后会播报语音"请享用您的咖啡"
    """
    time.sleep(3)

    get_coffee_cup_with_coffee(handle_l,handle_r,hand_l,hand_r,add_data_1)
    put_coffee_cup_with_coffee(handle_l,handle_r,hand_l,hand_r,add_data_1)


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
    # # time.sleep(32)

    # get_coffee_cup_with_coffee(handle_l,handle_r,hand_l,hand_r,add_data_1)
    # put_coffee_cup_with_coffee(handle_l,handle_r,hand_l,hand_r,add_data_1)
