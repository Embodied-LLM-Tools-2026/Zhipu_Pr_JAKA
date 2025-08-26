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
INIT_POS_R = x5.Pose(x=-166.48863412000844, y=-249.65707063473792, z=-18.802048871171785, a=169.26500545380372, b=-55.70720124022451, c=7.111138510153568, e1=48.61143291529611, e2=0.003999999999999999, e3=160.0)
INIT_POINT_R = x5.Point(pose=INIT_POS_R, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_R = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 160)

# 初始化左臂
INIT_POS_L = x5.Pose(x=-177.8952033844295, y=240.45491129918008, z=-30.583778310313168, a=-172.05245680637137, b=-50.2107365583525, c=-6.759244589737118, e1=-48.61143291529611, e2=-0.009, e3=0.016)
INIT_POINT_L = x5.Point(pose=INIT_POS_L, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_L = x5.Joint(j1 = -7.135, j2 = -58.596, j3 = -60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = 120, e2 = -0.009, e3 = 0.016)

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

def safe_robot(handle_l, handle_r, add_data):
    # 初始化左臂
    pick_2 = x5.Joint(j1 =101.441,j2 = -54.909, j3 = 8.036, j4 = -34.658, 
     j5 = -23.874, j6 = -9.847, e1 = 17.289, e2=0, e3=450)
    x5.movj(handle_l, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_l)
    # 初始化右臂
    x5.movj(handle_r, pick_2, add_data)
    x5.wait_move_done(handle_r)

def move_to_shelf():
    with AGVClient(ip='192.168.1.50') as agv:
        pose_result = agv.get_pose()
        if abs(pose_result[2]) < 0.4:
            agv.go_to_point_in_world(-0.047,-0.033,0, 0)
        else:
            agv.rotation(3.14)
            agv.go_to_point_in_world(-0.047,-0.033,0, 0)
def back_bar_station():
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


def pick_5_5(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 计算与目标参考点位的差值
    with AGVClient(ip='192.168.1.50') as agv:
        # agv.go_to_point_in_world(-0.047,-0.033,0, 0)
        pose_result = agv.get_pose()
        x, y, angle = pose_result
        print("pose_result = ",pose_result)

    delta_x = (x+0.047)*1000  #-0.0874+0.08=-0.0074
    delta_y = (y+0.033)*1000 #-0.0376+0.035=-0.0026

    # pick_1 = x5.Joint(j1 = 101.021, j2 = 6.288, j3 = 14.716, j4 = -84.667, 
    #  j5 = -23.874, j6 = -27.920, e1 = 17.289-90, e2=0, e3=450)
    # # 到达和货架同一高度
    # x5.movj(handle_R, pick_1, add_data)
    # x5.wait_move_done(handle_R)

    # # 到达抓取点位
    # pick_2 = x5.Joint(j1 =101.441,j2 = -54.909, j3 = 8.036, j4 = -34.658, 
    #  j5 = -23.874, j6 = -9.847, e1 = 17.289-90, e2=0, e3=450)
    # pick_2_point = x5.Pose(x=143.616, y=-598.703, z=152.984, a=98.703, b=0.220, c=13.085, e1=7.011, e2=0, e3=450)
    # pick_2_point = x5.Point(pose=pick_2_point, uf=0, tf=0, cfg=(0,0,0,7))

    # # 计算新的调整后的抓取点位  机器人y轴对应agv x轴，机器人右臂z轴对应agv y轴
    # new_pick_2_point = copy.deepcopy(pick_2_point)
    # new_pick_2_point.pose.z = 152.984 - delta_y
    # new_pick_2_point.pose.y = -598.703 + delta_x
    # print(new_pick_2_point)
    # # 逆解，求笛卡尔点位p1的对应关节坐标
    # jp1 = x5.cnvrt_j(handle_R, new_pick_2_point, 1, pick_2)


    # x5.movj(handle_R, pick_2, add_data)
    # x5.wait_move_done(handle_R)

    # # grasp
    # hand_r.setpos(350,350,350,250,250,0)
    # time.sleep(0.5)

    # # 升起
    # pick_3 = x5.Joint(j1 =106.506,j2 = -51.921, j3 = 8.036, j4 = -37.324, 
    #  j5 = -23.874, j6 = -9.847, e1 = 17.289-90, e2=0, e3=450)
    # x5.movj(handle_R, pick_3, add_data)
    # x5.wait_move_done(handle_R)

    # # # 返回和货架同一高度点
    # pick_4 = x5.Joint(j1 = 101.021, j2 = 6.288, j3 = 14.716, j4 = -84.667, 
    #  j5 = -23.874, j6 = -27.920, e1 = 17.289-90, e2=0, e3=450)
    # x5.movj(handle_R, pick_4, add_data)
    # x5.wait_move_done(handle_R)


    # # pre 待机位置
    # pick_5 = x5.Joint(j1 = 85.512, j2 = -14.709, j3 = 14.716, j4 = -102.856, 
    #  j5 = -23.874, j6 = -27.920, e1 = -72.711, e2=0, e3=450)
    # x5.movj(handle_R, pick_5, add_data)
    # x5.wait_move_done(handle_R)
    # # 回到待机位置
    # pick_6 = x5.Joint(j1 = 37.176, j2 = -5.808, j3 = 14.215, j4 = -48.033 ,
    #  j5 =25.527, j6 = -62.882, e1 = -84.449, e2=0, e3=160)
    # x5.movj(handle_R, pick_6, add_data)
    # x5.wait_move_done(handle_R)
    # print("抓取5号")

    # # 回到待机位置1
    # pick_7 = x5.Joint(j1 = 37.176, j2 = 15.535, j3 = 14.215, j4 = -68.999 ,
    #  j5 =25.527, j6 = -87.195, e1 = -84.449, e2=0, e3=160)
    # x5.movj(handle_R, pick_7, add_data)
    # x5.wait_move_done(handle_R)

    # # 回到待机位置2
    # pick_8 = x5.Joint(j1 =33.298, j2 = -2.419, j3 = 24.806, j4 = -77.036 ,
    # j5 = 21.211, j6 = -88, e1 = -83.869, e2=0, e3=130)
    # x5.movj(handle_R, pick_8, add_data)
    # x5.wait_move_done(handle_R)

    # # 开始移动到吧台
    # with AGVClient(ip='192.168.1.50') as agv:
    #     agv.go_to_point_in_world(-0.770,-0.028,0, 1)
    #     agv.rotation(3.14)

    # 预设点位，需要设定第二个
    # 放置点位
    pick_9 = x5.Joint(j1 =40.214, j2 = -61.393, j3 = 20.058, j4 = -13.470 ,
    j5 = 26.100, j6 = -77.139, e1 = -142.442, e2=0, e3=130)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    # time.sleep(1)
    # # 松手
    # hand_r.setpos(1000,1000,1000,1000,1000,1000)
    # time.sleep(1)

    # 手部收起
    pick_10 = x5.Joint(j1 =42.044, j2 = -60.326, j3 = 19.844, j4 = -9.654 ,
    j5 = 25.038, j6 = -80.082, e1 = -138.685, e2=0, e3=130)
    x5.movj(handle_R, pick_10, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    pick_11 = x5.Joint(j1 =41.743, j2 = -31.878, j3 = 29.285, j4 = -51.909 ,
    j5 = 20.548, j6 =-59.635, e1 = -131.058, e2=0, e3=130)
    x5.movj(handle_R, pick_11, add_data)
    x5.wait_move_done(handle_R)
    print("point11")

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

def pick_5_3(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 计算与目标参考点位的差值
    with AGVClient(ip='192.168.1.50') as agv:
        # agv.go_to_point_in_world(-0.047,-0.033,0, 0)
        pose_result = agv.get_pose()
        x, y, angle = pose_result
        print("pose_result = ",pose_result)

    delta_x = (x+0.047)*1000  #-0.0874+0.08=-0.0074
    delta_y = (y+0.033)*1000 #-0.0376+0.035=-0.0026

    # 到达和货架同一高度
    pick_1 = x5.Joint(j1 = 99.192, j2 = -28.025, j3 = 23.489, j4 = -82.349, 
                      j5 = -25.374, j6 = -19.298, e1 = -72.714, e2 = 0.001, e3 = 449.988)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    pick_3 = x5.Joint(j1 = 99.598, j2 = -83.667, j3 = 25.367, j4 = -28.36, j5 = -113.684, 
                      j6 = -13.142, e1 = 3.341, e2 = 0.025, e3 = 450)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # grasp
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # 返回和货架同一高度点
    pick_5 = x5.Joint(j1 = 99.192, j2 = -28.025, j3 = 23.489, j4 = -82.349, 
                      j5 = -25.374, j6 = -19.298, e1 = -72.714, e2 = 0.001, e3 = 449.988)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)


    # pre 待机位置
    pick_5 = x5.Joint(j1 = 62.82, j2 = -34.522, j3 = 6.827, j4 = -86.195, 
                    j5 = -8.202, j6 = -45.654, e1 = -72.657, e2 = 0.003, e3 = 160)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

    # 初始化右臂
    x5.movj(handle_R, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_R)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.770,-0.028,0, 1)
        agv.rotation(3.14)

    # # 放置点位
    pick_9 = x5.Joint(j1 = 48.573, j2 = -58.697, j3 = 18.297, j4 = -15.27, 
                      j5 = 18.217, j6 = -75.537, e1 = -128.447, e2 = 0.002, e3 = 129.978)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)

    # 放置点位
    pick_9 = x5.Joint(j1 =40.214, j2 = -61.393, j3 = 20.058, j4 = -13.470 ,
    j5 = 26.100, j6 = -77.139, e1 = -142.442, e2=0, e3=130)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    time.sleep(1)
    # 松手
    hand_r.setpos(1000,1000,1000,1000,1000,1000)
    time.sleep(1)

    # 手部收起
    pick_10 = x5.Joint(j1 =42.044, j2 = -60.326, j3 = 19.844, j4 = -9.654 ,
    j5 = 25.038, j6 = -80.082, e1 = -138.685, e2=0, e3=130)
    x5.movj(handle_R, pick_10, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    pick_11 = x5.Joint(j1 =41.743, j2 = -31.878, j3 = 29.285, j4 = -51.909 ,
    j5 = 20.548, j6 =-59.635, e1 = -131.058, e2=0, e3=130)
    x5.movj(handle_R, pick_11, add_data)
    x5.wait_move_done(handle_R)
    print("point11")

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

def pick_5_2(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 计算与目标参考点位的差值
    with AGVClient(ip='192.168.1.50') as agv:
        # agv.go_to_point_in_world(-0.047,-0.033,0, 0)
        pose_result = agv.get_pose()
        x, y, angle = pose_result
        print("pose_result = ",pose_result)

    delta_x = (x+0.047)*1000  #-0.0874+0.08=-0.0074
    delta_y = (y+0.033)*1000 #-0.0376+0.035=-0.0026

    pick_1 = x5.Joint(j1 = -29.064, j2 = -19.764, j3 = -117.556, j4 = -76.98, 
                      j5 = 40.379, j6 = -45.667, e1 = 85.808, e2 = -0.012, e3 = 0.023)
    pick_1_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 450)
    x5.movj(handle_L, pick_1, add_data)
    x5.movj(handle_R, pick_1_r, add_data)
    x5.wait_move_done(handle_L)

    pick_2 = x5.Joint(j1 = -96.365, j2 = -44.592, j3 = -11.472, j4 = -60.509, 
                      j5 = -16.184, j6 = -21.804, e1 = 108.051, e2 = -0.014, e3 = 0.024)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)

    pick_3 = x5.Joint(j1 = -105.427, j2 = -79.437, j3 = 3.266, j4 = -19.841, 
                      j5 = 12.029, j6 = -17.646, e1 = 70.15, e2 = -0.016, e3 = 0.026)
    
    # pick_3_point = x5.Pose(x=154.48724580637725, y=629.9681690697987, z=-46.0863549227175, a=-116.24592904467046, 
    #                         b=-2.685726643901727, c=-7.871574549223857, e1=3.2127072989953818, e2=-0.016, e3=0.026)
    # jp3 = correct_pos_left(pick_3_point,handle_L,pick_3)

    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # grasp
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)

    # # # 升起
    pick_4 = x5.Joint(j1 = -125.15, j2 = -53.591, j3 = 10.242, j4 = -54.451, 
                      j5 = 43.435, j6 = -15.461, e1 = 22.874, e2 = -0.016, e3 = 0.026)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

    # # # # 返回和货架同一高度点
    pick_5 = x5.Joint(j1 = -29.064, j2 = -19.764, j3 = -117.556, j4 = -76.98, 
                      j5 = 40.379, j6 = -45.667, e1 = 85.808, e2 = -0.012, e3 = 0.023)
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)


    # pre 待机位置
    pick_5 = x5.Joint(j1 = -14.825, j2 = -46.428, j3 = -73.754, j4 = -77.2, 
                      j5 = 39.374, j6 = -65.61, e1 = 85.805, e2 = -0.014, e3 = 0.026)
    pick_5_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 160)
    x5.movj(handle_R, pick_5_r, add_data)
    x5.movj(handle_L, pick_5, add_data)
    x5.wait_move_done(handle_L)

    x5.movj(handle_L, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_L)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.770,-0.028,0, 1)
        agv.rotation(3.14)

    pick_6_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 130)
    x5.movj(handle_R, pick_6_r, add_data)
    x5.wait_move_done(handle_R)

    pick_7 = x5.Joint(j1 = -14.814, j2 = -47.563, j3 = -81.056, j4 = -60.062, 
                        j5 = 26.778, j6 = -42.687, e1 = 120.015, e2 = -0.015, e3 = 0.03)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    # pre 待机位置
    pick_6 = x5.Joint(j1 = -16.894, j2 = -53.155, j3 = -75.999, j4 = -51.853, 
                      j5 = 21.785, j6 = -42.576, e1 = 119.997, e2 = -0.015, e3 = 0.028)
    x5.movj(handle_L, pick_6, add_data)
    x5.wait_move_done(handle_L)

    time.sleep(1)
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)

    pick_7 = x5.Joint(j1 = -11.511, j2 = -44.173, j3 = -79.625, j4 = -50.977, 
                      j5 = 22.385, j6 = -46.064, e1 = 119.994, e2 = -0.016, e3 = 0.03)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    pick_8 = x5.Joint(j1 = -13.285, j2 = -43.628, j3 = -92.183, j4 = -66.133, 
                      j5 = 12.743, j6 = -66.699, e1 = 119.989, e2 = -0.019, e3 = 0.03)
    x5.movj(handle_L, pick_8, add_data)
    x5.wait_move_done(handle_L)

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

def pick_4_4(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取2层4号
    """
    # with AGVClient(ip='192.168.1.51') as agv:
    #     agv.go_to_point_in_world(0.084,0.016,0.0043633, 0)

    kk = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 250)
    pick_1 = x5.Joint(j1 = 71.612, j2 = -21.748, j3 = 29.314, j4 = -75.875, 
                      j5 = -13.322, j6 = -31.03, e1 = -92.485, e2 = 0.009, e3 = 250)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 抓取
    pick_2 = x5.Joint(j1 = 83.777, j2 = -64.818, j3 = 19.032, j4 = -38.3, 
                      j5 = -17.455, j6 = -9.949, e1 = -92.486, e2 = 0.01, e3 = 249.967)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # grasp
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    pick_3 = x5.Joint(j1 = 69.632, j2 = -60.255, j3 = 31.299, j4 = -57.121, 
                      j5 = 8.709, j6 = -49.495, e1 = -92.485, e2 = 0.01, e3 = 249.956)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)



    pick_4 = x5.Joint(j1 = 45.998, j2 = -55.449, j3 = 17.531, j4 = -53.389, 
                      j5 = 8.767, j6 = -75.659, e1 = -92.485, e2 = 0.011, e3 = 160)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)

    # 初始化右臂
    x5.movj(handle_R, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_R)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.770,-0.028,0, 1)
        agv.rotation(3.14)

    # # 放置点位
    pick_9 = x5.Joint(j1 = 48.573, j2 = -58.697, j3 = 18.297, j4 = -15.27, 
                      j5 = 18.217, j6 = -75.537, e1 = -128.447, e2 = 0.002, e3 = 129.978)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)

    # 放置点位
    pick_9 = x5.Joint(j1 =40.214, j2 = -61.393, j3 = 20.058, j4 = -13.470 ,
    j5 = 26.100, j6 = -77.139, e1 = -142.442, e2=0, e3=130)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    time.sleep(1)
    # 松手
    hand_r.setpos(1000,1000,1000,1000,1000,1000)
    time.sleep(1)

    # 手部收起
    pick_10 = x5.Joint(j1 =42.044, j2 = -60.326, j3 = 19.844, j4 = -9.654 ,
    j5 = 25.038, j6 = -80.082, e1 = -138.685, e2=0, e3=130)
    x5.movj(handle_R, pick_10, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    pick_11 = x5.Joint(j1 = 46.852, j2 = -30.602, j3 = 37.145, j4 = -56.024, 
                       j5 = 21.365, j6 = -65.495, e1 = -131.059, e2 = -0.001, e3 = 129.99)
    x5.movj(handle_R, pick_11, add_data)
    x5.wait_move_done(handle_R)
    print("point11")

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

def pick_4_2(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取2层2号
    """
    # with AGVClient(ip='192.168.1.51') as agv:
    #     agv.go_to_point_in_world(0.084,0.016,0.0043633, 0)

    pick_1_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 250)
    pick_1 = x5.Joint(j1 = -64.878, j2 = -56.474, j3 = -54.681, j4 = -51.617, 
                      j5 = 8.454, j6 = -47.613, e1 = 120.008, e2 = -0.014, e3 = 0.027)
    x5.movj(handle_R, pick_1_r, add_data)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)


    pick_2 = x5.Joint(j1 = -59.478, j2 = -62.579, j3 = -60.609, j4 = -71.27, 
                      j5 = 60.579, j6 = -2.536, e1 = 81.595, e2 = -0.017, e3 = 0.03)
    x5.movj(handle_L, pick_2, add_data)
    x5.wait_move_done(handle_L)
    
    # 抓取
    pick_3 = x5.Joint(j1 = -72.925, j2 = -79.403, j3 = -42.929, j4 = -40.334, 
                      j5 = 60.579, j6 = 0.51, e1 = 81.595, e2 = -0.017, e3 = 0.03)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # grasp
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)

    pick_4 = x5.Joint(j1 = -67.496, j2 = -72.225, j3 = -55.285, j4 = -66.802, 
                    j5 = 49.156, j6 = -50.898, e1 = 81.592, e2 = -0.017, e3 = 0.029)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

    
    x5.movj(handle_L, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_L)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.770,-0.028,0, 1)
        agv.rotation(3.14)

    pick_6_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 130)
    x5.movj(handle_R, pick_6_r, add_data)
    x5.wait_move_done(handle_R)

    pick_7 = x5.Joint(j1 = -14.814, j2 = -47.563, j3 = -81.056, j4 = -60.062, 
                        j5 = 26.778, j6 = -42.687, e1 = 120.015, e2 = -0.015, e3 = 0.03)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    # pre 待机位置
    pick_6 = x5.Joint(j1 = -16.894, j2 = -53.155, j3 = -75.999, j4 = -51.853, 
                      j5 = 21.785, j6 = -42.576, e1 = 119.997, e2 = -0.015, e3 = 0.028)
    x5.movj(handle_L, pick_6, add_data)
    x5.wait_move_done(handle_L)

    time.sleep(1)
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)

    pick_7 = x5.Joint(j1 = -11.511, j2 = -44.173, j3 = -79.625, j4 = -50.977, 
                      j5 = 22.385, j6 = -46.064, e1 = 119.994, e2 = -0.016, e3 = 0.03)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    pick_8 = x5.Joint(j1 = -13.285, j2 = -43.628, j3 = -92.183, j4 = -66.133, 
                      j5 = 12.743, j6 = -66.699, e1 = 119.989, e2 = -0.019, e3 = 0.03)
    x5.movj(handle_L, pick_8, add_data)
    x5.wait_move_done(handle_L)

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

def pick_3_4(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取2层4号
    """
    # with AGVClient(ip='192.168.1.51') as agv:
    #     agv.go_to_point_in_world(0.084,0.016,0.0043633, 0)

    kk = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 100)
    pick_1 = x5.Joint(j1 = 32.272, j2 = -60.631, j3 = 75.835, j4 = -67.425, 
                      j5 = -24.454, j6 = -30.747, e1 = -120.004, e2 = 0.003, e3 = 99.99)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 抓取
    pick_2 = x5.Joint(j1 = 53.997, j2 = -84.264, j3 = 73.623, j4 = -24.059, 
                      j5 = -38.557, j6 = -15.206, e1 = -121.053, e2 = 0.005, e3 = 99.969)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # grasp
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    pick_3 = x5.Joint(j1 = 44.913, j2 = -78.004, j3 = 57.975, j4 = -43.954, 
                      j5 = -29.221, j6 = -74.287, e1 = -122.594, e2 = 0.001, e3 = 130)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # 初始化右臂
    x5.movj(handle_R, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_R)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.770,-0.028,0, 1)
        agv.rotation(3.14)

    # # 放置点位
    pick_9 = x5.Joint(j1 = 48.573, j2 = -58.697, j3 = 18.297, j4 = -15.27, 
                      j5 = 18.217, j6 = -75.537, e1 = -128.447, e2 = 0.002, e3 = 129.978)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)

    # 放置点位
    pick_9 = x5.Joint(j1 =40.214, j2 = -61.393, j3 = 20.058, j4 = -13.470 ,
    j5 = 26.100, j6 = -77.139, e1 = -142.442, e2=0, e3=130)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    time.sleep(1)
    # 松手
    hand_r.setpos(1000,1000,1000,1000,1000,1000)
    time.sleep(1)

    # 手部收起
    pick_10 = x5.Joint(j1 =42.044, j2 = -60.326, j3 = 19.844, j4 = -9.654 ,
    j5 = 25.038, j6 = -80.082, e1 = -138.685, e2=0, e3=130)
    x5.movj(handle_R, pick_10, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    pick_11 = x5.Joint(j1 =41.743, j2 = -31.878, j3 = 29.285, j4 = -51.909 ,
    j5 = 20.548, j6 =-59.635, e1 = -131.058, e2=0, e3=130)
    x5.movj(handle_R, pick_11, add_data)
    x5.wait_move_done(handle_R)
    print("point11")

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

def pick_3_2(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取2层2号
    """
    # with AGVClient(ip='192.168.1.51') as agv:
    #     agv.go_to_point_in_world(0.084,0.016,0.0043633, 0)

    pick_1_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 100)
    pick_1 = x5.Joint(j1 = -35.217, j2 = -63.211, j3 = -68.367, j4 = -63.357, 
                      j5 = 11.933, j6 = -25.536, e1 = 119.997, e2 = -0.014, e3 = 0.032)
    x5.movj(handle_R, pick_1_r, add_data)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)
    
    # # 抓取
    pick_3 = x5.Joint(j1 = -59.917, j2 = -87.048, j3 = -63.24, j4 = -20.89, 
                      j5 = 40.026, j6 = -11.208, e1 = 119.995, e2 = -0.013, e3 = 0.032)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # # grasp
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)

    pick_4 = x5.Joint(j1 = -40.892, j2 = -76.207, j3 = -60.597, j4 = -54.268,
                      j5 = 33.77, j6 = -67.144, e1 = 119.998, e2 = -0.014, e3 = 0.032)
    pick_4_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 130)
    x5.movj(handle_R, pick_4_r, add_data)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

    x5.movj(handle_L, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_L)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.770,-0.028,0, 1)
        agv.rotation(3.14)

    pick_6_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 130)
    x5.movj(handle_R, pick_6_r, add_data)
    x5.wait_move_done(handle_R)

    pick_7 = x5.Joint(j1 = -14.814, j2 = -47.563, j3 = -81.056, j4 = -60.062, 
                        j5 = 26.778, j6 = -42.687, e1 = 120.015, e2 = -0.015, e3 = 0.03)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    # pre 待机位置
    pick_6 = x5.Joint(j1 = -16.894, j2 = -53.155, j3 = -75.999, j4 = -51.853, 
                      j5 = 21.785, j6 = -42.576, e1 = 119.997, e2 = -0.015, e3 = 0.028)
    x5.movj(handle_L, pick_6, add_data)
    x5.wait_move_done(handle_L)

    time.sleep(1)
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)

    pick_7 = x5.Joint(j1 = -11.511, j2 = -44.173, j3 = -79.625, j4 = -50.977, 
                      j5 = 22.385, j6 = -46.064, e1 = 119.994, e2 = -0.016, e3 = 0.03)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    pick_8 = x5.Joint(j1 = -13.285, j2 = -43.628, j3 = -92.183, j4 = -66.133, 
                      j5 = 12.743, j6 = -66.699, e1 = 119.989, e2 = -0.019, e3 = 0.03)
    x5.movj(handle_L, pick_8, add_data)
    x5.wait_move_done(handle_L)

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

def pick_2_4(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取2层4号
    """
    kk = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 21.399, e3 = -4)
    pick_1 = x5.Joint(j1 = 16.584, j2 = -28.354, j3 = 58.291, j4 = -97.506, 
                      j5 = -10.38, j6 = -7.371, e1 = -107.629, e2 = 21.438, e3 = -3.963)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R) 

    # 抓取
    pick_2 = x5.Joint(j1 = 39.663, j2 = -64.769, j3 = 55.219, j4 = -61.124, 
                      j5 = -37.838, j6 = -9.236, e1 = -107.629, e2 = 21.44, e3 = -3.963)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)

    # grasp
    hand_r.setpos(200,200,200,200,200,0)
    time.sleep(1)

    pick_3 = x5.Joint(j1 = 31.144, j2 = -59.942, j3 = 39.116, j4 = -77.883, 
                      j5 = -33.918, j6 = -48.242, e1 = -107.65, e2 = 0.002, e3 = -3.947)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # 初始化右臂
    x5.movj(handle_R, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_R)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.770,-0.028,0, 1)
        agv.rotation(3.14)

    # # 放置点位
    pick_9 = x5.Joint(j1 = 48.573, j2 = -58.697, j3 = 18.297, j4 = -15.27, 
                      j5 = 18.217, j6 = -75.537, e1 = -128.447, e2 = 0.002, e3 = 129.978)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)

    # # 放置点位
    pick_9 = x5.Joint(j1 =40.214, j2 = -61.393, j3 = 20.058, j4 = -13.470 ,
    j5 = 26.100, j6 = -77.139, e1 = -142.442, e2=0, e3=130)
    x5.movj(handle_R, pick_9, add_data)
    x5.wait_move_done(handle_R)

    time.sleep(1)
    # 松手
    hand_r.setpos(1000,1000,1000,1000,1000,1000)
    time.sleep(1)

    # 手部收起
    pick_10 = x5.Joint(j1 =42.044, j2 = -60.326, j3 = 19.844, j4 = -9.654 ,
    j5 = 25.038, j6 = -80.082, e1 = -138.685, e2=0, e3=130)
    x5.movj(handle_R, pick_10, add_data)
    x5.wait_move_done(handle_R)
    print("point8")

    pick_11 = x5.Joint(j1 =41.743, j2 = -31.878, j3 = 29.285, j4 = -51.909 ,
    j5 = 20.548, j6 =-59.635, e1 = -131.058, e2=0, e3=130)
    x5.movj(handle_R, pick_11, add_data)
    x5.wait_move_done(handle_R)
    print("point11")

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

def pick_2_2(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取2层2号
    """

    pick_1_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 21.399, e3 = -4)
    pick_1 = x5.Joint(j1 = -17.233, j2 = -58.881, j3 = -81.396, j4 = -70.528, 
                      j5 = 34.565, j6 = -37.225, e1 = 119.999, e2 = -0.014, e3 = 0.012)
    x5.movj(handle_R, pick_1_r, add_data)
    x5.wait_move_done(handle_R)
    x5.movj(handle_L, pick_1, add_data)
    x5.wait_move_done(handle_L)

    # 抓取
    pick_3 = x5.Joint(j1 = -38.938, j2 = -80.171, j3 = -76.324, j4 = -52.365, 
                      j5 = 54.22, j6 = -21.904, e1 = 119.998, e2 = -0.015, e3 = 0.032)
    x5.movj(handle_L, pick_3, add_data)
    x5.wait_move_done(handle_L)

    # grasp
    hand_l.setpos(200,200,200,200,200,0)
    time.sleep(1)

    pick_4 = x5.Joint(j1 = -27.128, j2 = -73.933, j3 = -51.485, j4 = -67.795, 
                      j5 = 50.856, j6 = -64.144, e1 = 119.991, e2 = -0.016, e3 = 0.031)
    pick_4_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0, e3 = -4)
    x5.movj(handle_R, pick_4_r, add_data)
    x5.movj(handle_L, pick_4, add_data)
    x5.wait_move_done(handle_L)

    x5.movj(handle_L, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_L)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.770,-0.028,0, 1)
        agv.rotation(3.14)

    pick_6_r = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, 
                        j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 130)
    x5.movj(handle_R, pick_6_r, add_data)
    x5.wait_move_done(handle_R)

    pick_7 = x5.Joint(j1 = -14.814, j2 = -47.563, j3 = -81.056, j4 = -60.062, 
                        j5 = 26.778, j6 = -42.687, e1 = 120.015, e2 = -0.015, e3 = 0.03)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    # pre 待机位置
    pick_6 = x5.Joint(j1 = -16.894, j2 = -53.155, j3 = -75.999, j4 = -51.853, 
                      j5 = 21.785, j6 = -42.576, e1 = 119.997, e2 = -0.015, e3 = 0.028)
    x5.movj(handle_L, pick_6, add_data)
    x5.wait_move_done(handle_L)

    time.sleep(1)
    hand_l.setpos(1000,1000,1000,1000,1000,0)
    time.sleep(1)

    pick_7 = x5.Joint(j1 = -11.511, j2 = -44.173, j3 = -79.625, j4 = -50.977, 
                      j5 = 22.385, j6 = -46.064, e1 = 119.994, e2 = -0.016, e3 = 0.03)
    x5.movj(handle_L, pick_7, add_data)
    x5.wait_move_done(handle_L)

    pick_8 = x5.Joint(j1 = -13.285, j2 = -43.628, j3 = -92.183, j4 = -66.133, 
                      j5 = 12.743, j6 = -66.699, e1 = 119.989, e2 = -0.019, e3 = 0.03)
    x5.movj(handle_L, pick_8, add_data)
    x5.wait_move_done(handle_L)

    init_robot(handle_L, handle_R, add_data, hand_l, hand_r)

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

def move_to_LM():
    """
    到达LM点位
    """
    with AGVClient(ip='192.168.192.5') as agv:
        # agv.go_to_target_LM("LM1", "LM2")
        agv.go_to_target_LM("LM2", "LM4")
        time.sleep(20)
        agv.go_to_target_LM("LM4", "LM3")
        time.sleep(15)

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
    # move_to_shelf()

    # pick_5_5(handle_l, handle_r, hand_l, hand_r,add_data_1)

    # pick_5_3(handle_l, handle_r, hand_l, hand_r,add_data_1)
    # pick_5_2(handle_l, handle_r, hand_l, hand_r,add_data_1)

    # pick_4_4(handle_l, handle_r, hand_l, hand_r,add_data_1)
    # pick_4_2(handle_l, handle_r, hand_l, hand_r,add_data_1)

    # pick_3_4(handle_l, handle_r, hand_l, hand_r,add_data_1)
    # pick_3_2(handle_l, handle_r, hand_l, hand_r,add_data_1)
    
    # pick_2_4(handle_l, handle_r, hand_l, hand_r,add_data_1)
    # pick_2_2(handle_l, handle_r, hand_l, hand_r,add_data_1)



    # back_bar_station()

if __name__ == "__main__":
    main()

