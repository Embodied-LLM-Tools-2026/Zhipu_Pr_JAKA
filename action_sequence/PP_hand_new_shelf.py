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
    from controller.agv_client import AGVClient
except ImportError as e:
    print(f"导入InspireHandR或AGVClient失败，请检查controller/hand_controller.py和controller/AGV_controller.py路径。错误信息: {e}")
    # 如果导入失败，创建占位类以避免NameError
    class InspireHandR:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InspireHandR类导入失败，请检查controller/hand_controller.py文件是否存在且无语法错误。")
    class AGVClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("AGVClient类导入失败，请检查controller/AGV_controller.py文件是否存在且无语法错误。")

# from correct_hand import correct_left_arm, correct_right_arm

# 初始化右臂
INIT_POS_R = x5.Pose(x=-166.48863412000844, y=-249.65707063473792, z=-18.802048871171785, a=169.26500545380372, b=-55.70720124022451, c=7.111138510153568, e1=48.61143291529611, e2=0.003999999999999999, e3=250.0)
INIT_POINT_R = x5.Point(pose=INIT_POS_R, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_R = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 250)

# 初始化左臂
INIT_POS_L = x5.Pose(x=-149.3199993845242, y=240.45491129421245, z=-15.187910316510676, a=-172.05249999999998, b=-50.210699999999996, c=-6.759200000000021, e1=-48.61143290082501, e2=-0.009, e3=0.016)
INIT_POINT_L = x5.Point(pose=INIT_POS_L, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_L = x5.Joint(j1 = -6.809, j2 = -55.111, j3 = -63.25, j4 = -94.793, j5 = 0.773, j6 = -75.875, e1 = 116.933, e2 = -0.009, e3 = 0.016)

# 对左臂和右臂的点进行纠偏
# def correct_pos_left(pick_2_point,handle_L,pick_2):
#     pick_2_point = x5.Point(pose=pick_2_point, uf=0, tf=0, cfg=(0,0,0,7))
#     target_agv_point = [-0.047,-0.033,-0.0087266]
#     P_hand_LeftArmCur = correct_left_arm(target_agv_point,pick_2_point)
#     new_pick_2_point = x5.Point(pose=P_hand_LeftArmCur, uf=0, tf=0, cfg=(0,0,0,7))
#     # 逆解，求笛卡尔点位p1的对应关节坐标
#     jp1 = x5.cnvrt_j(handle_L, new_pick_2_point, 1, pick_2)
#     return jp1

# def correct_pos_right(pick_2_point,handle_R,pick_2):
#     pick_2_point = x5.Point(pose=pick_2_point, uf=0, tf=0, cfg=(0,0,0,7))
#     target_agv_point = [-0.047,-0.033,-0.0087266]
#     P_hand_RightArmCur = correct_right_arm(target_agv_point,pick_2_point)
#     new_pick_2_point = x5.Point(pose=P_hand_RightArmCur, uf=0, tf=0, cfg=(0,0,0,7))
#     # 逆解，求笛卡尔点位p1的对应关节坐标
#     jp1 = x5.cnvrt_j(handle_R, new_pick_2_point, 1, pick_2)
#     return jp1

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

def move_to_shelf():
    """
    移动到货架
    """
    with AGVClient(ip='192.168.1.50') as agv:
        pose_result = agv.get_pose()
        if abs(pose_result[2]) < 0.4:
            agv.go_to_point_in_world(-0.047,-0.033,0, 0)
        else:
            agv.rotation(3.14)
            agv.go_to_point_in_world(-0.047,-0.033,0, 0)

def back_bar_station():
    """
    移动到吧台
    """
    with AGVClient(ip='192.168.1.50') as agv:
        pose_result = agv.get_pose()
        bar_pose = [-0.770,-0.028,3.14,1]
        err = sum(abs(x - y) for x, y in zip(pose_result[:3], bar_pose[:3]))
        print(f"当前与吧台位置之间的误差：{err}")
        if err > 0.15:
            if abs(pose_result[2]) > 2.5:
                agv.go_to_point_in_world(*bar_pose)
            else:
                bar_pose[2] = 0
                agv.go_to_point_in_world(*bar_pose)
                agv.rotation(3.14)
        else:
            print("已到达吧台位置，不再执行back_bar_station")

def pick_5_1(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = -95.921, j2 = -24.397, j3 = -14.665, j4 = -74.85, 
                        j5 = -3.943, j6 = -80.743, e1 = 116.925, e2 = -0.021, e3 = 0.035)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = -89.23, j2 = -4.283, j3 = -14.426, j4 = -40.944, 
                        j5 = -1.206, j6 = -62.485, e1 = 93.972, e2 = -0.025, e3 = 0.041)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 抓取点位
    pick_3 = x5.Joint(j1 = -90.369, j2 = -36.611, j3 = -7.863, j4 = -34.779, 
                        j5 = 4.734, j6 = -34.461, e1 = 93.969, e2 = -0.029, e3 = 0.041)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 抓取
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)


    # 收回点位
    pick_4 = x5.Joint(j1 = -89.887, j2 = -0.858, j3 = -26.124, j4 = -66.105, 
                        j5 = 3.801, j6 = -74.576, e1 = 93.975, e2 = -0.027, e3 = 0.041)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

   # 初始化
    pick_5  = INIT_JOINT_L
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)

def pick_5_2(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = -95.921, j2 = -24.397, j3 = -14.665, j4 = -74.85, 
                        j5 = -3.943, j6 = -80.743, e1 = 116.925, e2 = -0.021, e3 = 0.035)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = -90.409, j2 = -20.402, j3 = -13.251, j4 = -46.682, 
                        j5 = 0.523, j6 = -52.491, e1 = 93.969, e2 = -0.027, e3 = 0.042)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 抓取点位
    pick_3 = x5.Joint(j1 = -95.933, j2 = -46.826, j3 = 0.851, j4 = -34.697, 
                        j5 = 2.882, j6 = -45.236, e1 = 88.178, e2 = -0.034, e3 = 0.045)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 抓取
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)


    # 收回点位
    pick_4 = x5.Joint(j1 = -97.052, j2 = -25.694, j3 = -7.001, j4 = -54.101, 
                        j5 = -8.915, j6 = -76.81, e1 = 88.139, e2 = -0.035, e3 = 0.044)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

   # 初始化
    pick_5  = INIT_JOINT_L
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)

def pick_5_3(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = -95.921, j2 = -24.397, j3 = -14.665, j4 = -74.85, 
                        j5 = -3.943, j6 = -80.743, e1 = 116.925, e2 = -0.021, e3 = 0.035)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = -87.525, j2 = -24.519, j3 = -11.803, j4 = -59.57, 
                        j5 = -3.626, j6 = -51.441, e1 = 93.965, e2 = -0.028, e3 = 0.04)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 抓取点位
    pick_3 = x5.Joint(j1 = -95.107, j2 = -61.767, j3 = -2.217, j4 = -28.786, 
                        j5 = 4.785, j6 = -49.844, e1 = 93.961, e2 = -0.031, e3 = 0.045)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 抓取
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_4 = x5.Joint(j1 = -82.66, j2 = -29.614, j3 = -22.496, j4 = -62.463, 
                        j5 = -8.651, j6 = -79.895, e1 = 93.997, e2 = -0.034, e3 = 0.043)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

   # 初始化
    pick_5  = INIT_JOINT_L
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)



def pick_5_6(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 初始化点位
    pick_1 = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 450)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R) 

    # 预抓取点位1
    pick_2 = x5.Joint(j1 = 53.074, j2 = -24.471, j3 = 46.66, j4 = -81.41, 
                        j5 = -2.023, j6 = -74.106, e1 = -120.008, e2 = 0.009, e3 = 449.967)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # 预抓取点位2
    pick_3 = x5.Joint(j1 = 75.809, j2 = -13.03, j3 = 32.935, j4 = -39.577, 
                        j5 = 0.212, j6 = -70.477, e1 = -83.596, e2 = 0.008, e3 = 449.944)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # 抓取点位
    pick_4 = x5.Joint(j1 = 83.773, j2 = -42.168, j3 = 40.875, j4 = -43.659, 
                        j5 = -58.028, j6 = -47.506, e1 = -56.599, e2 = 0.014, e3 = 449.924)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)

    # 抓取
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 回收点位
    pick_5 = x5.Joint(j1 = 61.677, j2 = -22.165, j3 = 59.546, j4 = -87.168, 
                        j5 = -10.245, j6 = -69.699, e1 = -56.599, e2 = 0.024, e3 = 449.905)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

    # 初始化
    pick_6  = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 450)

    x5.movj(handle_R, pick_6, add_data)
    x5.wait_move_done(handle_R)

def pick_5_5(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 初始化点位
    pick_1 = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 450)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R) 

    # 预抓取点位1
    pick_2 = x5.Joint(j1 = 53.074, j2 = -24.471, j3 = 46.66, j4 = -81.41, 
                        j5 = -2.023, j6 = -74.106, e1 = -120.008, e2 = 0.009, e3 = 449.967)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # 预抓取点位2
    pick_3 = x5.Joint(j1 = 80.04, j2 = -30.324, j3 = 31.59, j4 = -59.438, 
                        j5 = -19.581, j6 = -44.341, e1 = -83.598, e2 = 0.01, e3 = 449.933)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # 抓取点位
    pick_4 = x5.Joint(j1 = 91.192, j2 = -49.366, j3 = 14.363, j4 = -50.168, 
                        j5 = -33.927, j6 = -36.335, e1 = -57.965, e2 = 0.016, e3 = 449.902)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)

    # 抓取
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_5 = x5.Joint(j1 = 85.965, j2 = -27.793, j3 = 31.019, j4 = -79.459, 
                        j5 = -7.721, j6 = -68.284, e1 = -57.963, e2 = 0.019, e3 = 449.882)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

    # 初始化
    pick_6  = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 450)

    x5.movj(handle_R, pick_6, add_data)
    x5.wait_move_done(handle_R)

def pick_5_4(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 初始化点位
    pick_1 = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 450)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R) 

    # 预抓取点位1
    pick_2 = x5.Joint(j1 = 53.074, j2 = -24.471, j3 = 46.66, j4 = -81.41, 
                        j5 = -2.023, j6 = -74.106, e1 = -120.008, e2 = 0.009, e3 = 449.967)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # 预抓取点位2
    pick_3 = x5.Joint(j1 = 92.123, j2 = -40.019, j3 = 31.772, j4 = -57.328, 
                        j5 = -35.315, j6 = -51.875, e1 = -68.105, e2 = -0.005, e3 = 449.893)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # 抓取点位
    pick_4 = x5.Joint(j1 = 92.789, j2 = -73.004, j3 = 19.098, j4 = -32.706, 
                        j5 = -39.264, j6 = -41.004, e1 = -58.101, e2 = 0.013, e3 = 449.912)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)

    # 抓取
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_5 = x5.Joint(j1 = 75.341, j2 = -42.96, j3 = 49.918, j4 = -78.964, 
                        j5 = -17.49, j6 = -69.401, e1 = -58.103, e2 = 0.012, e3 = 449.902)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

    # 初始化
    pick_6  = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 450)

    x5.movj(handle_R, pick_6, add_data)
    x5.wait_move_done(handle_R)

def pick_4_6(handle_L,handle_R,hand_l,hand_r,add_data):
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = 53.074, j2 = -24.471, j3 = 46.66, j4 = -81.41, 
                        j5 = -2.023, j6 = -74.106, e1 = -120.008, e2 = 0.009, e3 = 250)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = 52.115, j2 = -6.47, j3 = 54.512, j4 = -51.087, 
                        j5 = -7.022, j6 = -73.882, e1 = -87.924, e2 = 0.015, e3 = 249.979)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # 抓取点位
    pick_3 = x5.Joint(j1 = 53.665, j2 = -49.575, j3 = 69.89, j4 = -54.788, 
                        j5 = -58.19, j6 = -30.534, e1 = -77.566, e2 = 0.019, e3 = 249.958)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # 抓取
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_4 = x5.Joint(j1 = 35.983, j2 = -39.265, j3 = 80.375, j4 = -82.341, 
                        j5 = -27.824, j6 = -80.198, e1 = -77.565, e2 = 0.019, e3 = 249.947)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)

    # 初始化
    pick_5  = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 250)

    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

def pick_4_5(handle_L,handle_R,hand_l,hand_r,add_data):
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = 53.074, j2 = -24.471, j3 = 46.66, j4 = -81.41, 
                        j5 = -2.023, j6 = -74.106, e1 = -120.008, e2 = 0.009, e3 = 250)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = 50.317, j2 = -18.34, j3 = 53.955, j4 = -76.164, 
                        j5 = -16.469, j6 = -48.307, e1 = -87.928, e2 = 0.016, e3 = 249.957)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # 抓取点位
    pick_3 = x5.Joint(j1 = 73.536, j2 = -57.11, j3 = 37.016, j4 = -43.775, 
                        j5 = -41.755, j6 = -35.297, e1 = -80.264, e2 = 0.016, e3 = 249.947)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)
    
    # 抓取
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_4 = x5.Joint(j1 = 71.819, j2 = -42.461, j3 = 36.738, j4 = -66.883, 
                        j5 = -19.744, j6 = -76.158, e1 = -80.268, e2 = 0.017, e3 = 249.936)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)

    # 初始化
    pick_5  = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 250)

    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

def pick_4_4(handle_L,handle_R,hand_l,hand_r,add_data):
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = 53.074, j2 = -24.471, j3 = 46.66, j4 = -81.41, 
                        j5 = -2.023, j6 = -74.106, e1 = -120.008, e2 = 0.009, e3 = 250)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = 75.982, j2 = -44.072, j3 = 33.849, j4 = -60.029, 
                        j5 = -29.384, j6 = -45.459, e1 = -87.72, e2 = 0.016, e3 = 249.946)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # 抓取点位
    pick_3 = x5.Joint(j1 = 77.979, j2 = -70.579, j3 = 19.851, j4 = -44.221, 
                        j5 = -20.658, j6 = -22.304, e1 = -87.721, e2 = 0.016, e3 = 249.936)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)
    
    # 抓取
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_4 = x5.Joint(j1 = 78.961, j2 = -58.477, j3 = 21.404, j4 = -77.868, 
                        j5 = -15.59, j6 = -58.113, e1 = -87.816, e2 = 0.017, e3 = 249.925)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)

    # 初始化
    pick_5  = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 250)

    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

def pick_4_1(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = -88.453, j2 = -1.753, j3 = -5.414, j4 = -67.082, 
                        j5 = -6.036, j6 = -81.614, e1 = 116.92, e2 = -0.022, e3 = 0.04)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = -86.576, j2 = -9.962, j3 = -2.254, j4 = -41.425, 
                        j5 = -6.888, j6 = -58.527, e1 = 107.945, e2 = -0.023, e3 = 0.038)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 抓取点位
    pick_3 = x5.Joint(j1 = -81.048, j2 = -35.919, j3 = -0.779, j4 = -35.546, 
                        j5 = -19.9, j6 = -37.72, e1 = 107.945, e2 = -0.026, e3 = 0.041)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 抓取
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_4 = x5.Joint(j1 = -79.184, j2 = -16.552, j3 = -9.314, j4 = -66.805, 
                        j5 = -15.564, j6 = -78.17, e1 = 107.943, e2 = -0.027, e3 = 0.042)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

   # 初始化
    pick_5  = INIT_JOINT_L
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)


def pick_4_2(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = -88.453, j2 = -1.753, j3 = -5.414, j4 = -67.082, 
                        j5 = -6.036, j6 = -81.614, e1 = 116.92, e2 = -0.022, e3 = 0.04)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = -86.576, j2 = -9.962, j3 = -2.254, j4 = -41.425, 
                        j5 = -6.888, j6 = -58.527, e1 = 107.945, e2 = -0.023, e3 = 0.038)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 抓取点位
    pick_3 = x5.Joint(j1 = -84.192, j2 = -41.577, j3 = 2.046, j4 = -48.746, 
                        j5 = -16.34, j6 = -31.569, e1 = 107.944, e2 = -0.023, e3 = 0.041)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # 抓取
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_4 = x5.Joint(j1 = -81.426, j2 = -25.328, j3 = -2.137, j4 = -79.516, 
                        j5 = -15.706, j6 = -65.958, e1 = 107.943, e2 = -0.022, e3 = 0.043)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

   # 初始化
    pick_5  = INIT_JOINT_L
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)


def pick_4_3(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 预抓取点位1
    pick_1 = x5.Joint(j1 = -88.453, j2 = -1.753, j3 = -5.414, j4 = -67.082, 
                        j5 = -6.036, j6 = -81.614, e1 = 116.92, e2 = -0.022, e3 = 0.04)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    # 预抓取点位2
    pick_2 = x5.Joint(j1 = -86.259, j2 = -32.952, j3 = -0.434, j4 = -51.663, 
                        j5 = -13.497, j6 = -47.714, e1 = 107.943, e2 = -0.023, e3 = 0.042)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    # 抓取点位
    pick_3 = x5.Joint(j1 = -84.673, j2 = -55.646, j3 = -0.483, j4 = -51.603, 
                        j5 = -11.874, j6 = -24.118, e1 = 107.943, e2 = -0.024, e3 = 0.042)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)
    # 抓取
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 收回点位
    pick_4 = x5.Joint(j1 = -76.023, j2 = -28.698, j3 = -12.969, j4 = -76.325, 
                        j5 = -5.488, j6 = -72.912, e1 = 107.944, e2 = -0.026, e3 = 0.043)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

   # 初始化
    pick_5  = INIT_JOINT_L
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)

def move_to_pick_height_pitch_angle(handle_L,handle_R,add_data, height, pitch_angle):
    """
    到达抓取的高度和头的俯仰角
    """
    joint_r = copy.copy(INIT_JOINT_R)
    joint_r.e3 = height
    x5.movj(handle_R, joint_r, add_data)
    x5.wait_move_done(handle_R)
    joint_l = copy.copy(INIT_JOINT_L)
    joint_l.e3 = pitch_angle
    x5.movj(handle_L, joint_l, add_data)
    x5.wait_move_done(handle_L)

def main():
    hand_l = InspireHandR(port="COM12", baudrate=115200, hand_id=1)
    hand_r = InspireHandR(port="COM14", baudrate=115200, hand_id=2)
    hand_l.set_default_speed(200,200,200,200,200,200)
    hand_r.set_default_speed(200,200,200,200,200,200)
    hand_r.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)
    add_data_1 = x5.MovPointAdd(vel=100, acc=100)
    add_data_2 = x5.MovPointAdd(vel=100, cnt=100, acc=100, dec=100, offset =-1,
    offset_data=(10,0,0,0,0,0,0,0,0))
    # 连接机器人
    handle_l = x5.connect("192.168.1.9")
    handle_r = x5.connect("192.168.1.10")
    # move_to_pick_height_pitch_angle(handle_l,handle_r,add_data_1, -4, 28)
    init_robot(handle_l, handle_r, add_data_1, hand_l, hand_r)
    pick_4_3(handle_l, handle_r, hand_l, hand_r, add_data_1)

if __name__ == "__main__":
    main()

