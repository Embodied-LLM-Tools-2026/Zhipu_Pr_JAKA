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
INIT_JOINT_R = x5.Joint(j1=-7.277, j2=-80.381, j3=126.895, j4=-91.135, j5=-24.051, j6=-74.093, e1=12.445-90, e2=0, e3=160)

# 初始化左臂
INIT_POS_L = x5.Pose(x=-200, y=245, z=5, a=120, b=5, c=115, e1=-90, e2=0, e3=0)
INIT_POINT_L = x5.Point(pose=INIT_POS_L, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_L = x5.Joint(j1=4.927, j2=-80.496, j3=-126.895, j4=-87.812, j5=22.909, j6=-79.830, e1=-20.340+90, e2=0, e3=0)

def init_robot(handle_l, handle_r, add_data):
    # 初始化左臂
    x5.movj(handle_l, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_l)
    # 初始化右臂
    x5.movj(handle_r, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_r)


def safe_robot(handle_l, handle_r, add_data):
    # 初始化左臂
    pick_2 = x5.Joint(j1 =101.441,j2 = -54.909, j3 = 8.036, j4 = -34.658, 
     j5 = -23.874, j6 = -9.847, e1 = 17.289, e2=0, e3=450)
    x5.movj(handle_l, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_l)
    # 初始化右臂
    x5.movj(handle_r, pick_2, add_data)
    x5.wait_move_done(handle_r)

def wave(handle_l, handle_r, add_data):
    """
    挥手
    """
    ## 抬手
    bow_joint_pose1 = copy.copy(INIT_JOINT_R)
    bow_joint_pose1.j1 = 27.19087
    bow_joint_pose1.j2 = -21.89376
    bow_joint_pose1.j3 = 150
    bow_joint_pose1.j4 = -102.61023
    bow_joint_pose1.j5 = 6.44644
    bow_joint_pose1.j6 = -26.36375
    bow_joint_pose1.e1 = -97-60

    bow_joint_pose2 = copy.copy(INIT_JOINT_R)
    bow_joint_pose2.j1 = 27.448
    bow_joint_pose2.j2 = -2.349
    bow_joint_pose2.j3 = 150
    bow_joint_pose2.j4 = -64.47834
    bow_joint_pose2.j5 = 6.43283
    bow_joint_pose2.j6 = 6.43283
    bow_joint_pose2.e1 = -97-60
    
    ## 向左挥手
    x5.movj(handle_r, bow_joint_pose1, add_data)
    x5.wait_move_done(handle_r)
    ## 向右挥手
    x5.movj(handle_r, bow_joint_pose2, add_data)
    x5.wait_move_done(handle_r)

    ## 向左挥手
    x5.movj(handle_r, bow_joint_pose1, add_data)
    x5.wait_move_done(handle_r)
    ## 向右挥手
    x5.movj(handle_r, bow_joint_pose2, add_data)
    x5.wait_move_done(handle_r)

    ## 向左挥手
    x5.movj(handle_r, bow_joint_pose1, add_data)
    x5.wait_move_done(handle_r)
    ## 向右挥手
    x5.movj(handle_r, bow_joint_pose2, add_data)
    x5.wait_move_done(handle_r)

    ## 回到初始点位
    x5.movj(handle_r, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_r)

def move_to_shelf():
    with AGVClient(ip='192.168.1.51') as agv:
        agv.go_to_point_in_world(-0.255,-0.039,0, 1)

def pick_2_4(handle_L,handle_R,add_data,gripper_left,gripper_right):
    """
    抓取2层4号
    """
    pick_2 = x5.Joint(j1=91.998, j2 = -6.429, j3 = 21.269, j4 = -98.156 ,
    j5 = -13.637, j6 = -26.405, e1 = -59.672, e2=0, e3=130)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)
    print("抓取2号")

    # 抓取3号
    pick_3 = x5.Joint(j1=84.154, j2 = -29.201, j3 = 22.216, j4 = -97.854,
    j5 = -86.503, j6 = -12.653, e1 = -1.377, e2=0, e3=130)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # 抓取
    gripper_right.setpos(72)

    print("抓取3号")
    pick_4 = x5.Joint(j1=98.120, j2 = -26.607, j3 = 24.123, j4 = -102.5,
    j5 = -87.652, j6 = -24.386, e1 = 11.634, e2=0, e3=130)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)
    print("抓取4号")

    pick_2 = x5.Joint(j1=91.998, j2 = 3.349, j3 = 21.269, j4 = -98.156 ,
    j5 = -13.637, j6 = -35.073, e1 = -59.672, e2=0, e3=130)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)
    print("抓取2号")

    pick_5 = x5.Joint(j1 =33.298, j2 = -2.419, j3 = 24.806, j4 = -77.036 ,
    j5 = 21.211, j6 = -88, e1 = -83.869, e2=0, e3=130)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)
    print("抓取5号")

def pick_2_2(handle_L,handle_R,add_data,gripper_left,gripper_right):
    """
    抓取2层2号
    """

    pick_4 = x5.Joint(j1=63.393, j2 = -27.739, j3 = 51.683, j4 = -96.139,
    j5 = -16.647, j6 = -40.161, e1 = -64.811, e2=0, e3=130)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)
    print("抓取2号")

    pick_3 = x5.Joint(j1=85.507, j2 = -57.212, j3 = 30.204, j4 = -81.092,
    j5 = -73.298, j6 = -27.747, e1 = -24.445, e2=0, e3=130)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)
    print("抓取2号")


    pick_2 = x5.Joint(j1=86.106, j2 = -69.889, j3 = 26.685, j4 = -66.611,
    j5 = -68.702, j6 = -24.336, e1 = -34.496, e2=0, e3=130)
    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)
    print("抓取2号")

    gripper_right.setpos(72)

    pick_3 = x5.Joint(j1=85.507, j2 = -57.212, j3 = 30.204, j4 = -81.092,
    j5 = -73.298, j6 = -27.747, e1 = -24.445, e2=0, e3=130)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)
    print("抓取2号")

    pick_4 = x5.Joint(j1=63.393, j2 = -27.739, j3 = 51.683, j4 = -96.139,
    j5 = -16.647, j6 = -40.161, e1 = -64.811, e2=0, e3=130)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)
    print("抓取2号")

    pick_5 = x5.Joint(j1 =33.298, j2 = -2.419, j3 = 24.806, j4 = -77.036 ,
    j5 = 21.211, j6 = -88, e1 = -83.869, e2=0, e3=130)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)
    print("抓取5号")


def pick_1_5(handle_L,handle_R,add_data,gripper_left,gripper_right):
    """
    抓取
    """
    # 计算与目标参考点位的差值
    with AGVClient(ip='192.168.1.51') as agv:
        # agv.go_to_point_in_world(-0.255,-0.039,0, 1)
        pose_result = agv.get_pose()
        x, y, angle = pose_result
    delta_x = (x-0.016)*1000  #-0.0874+0.08=-0.0074
    delta_y = (y+0.037)*1000 #-0.0376+0.035=-0.0026


    # 到达和货架同一高度
    pick_0 = x5.Joint(j1 = 99.772, j2 = -11.213, j3 = 31.413, j4 = -85.976, 
     j5 = -24.194, j6 = -57.412, e1 = -73.696, e2=0, e3=450)
    x5.movj(handle_R, pick_0, add_data)
    x5.wait_move_done(handle_R)

    # 到达和货架同一高度
    pick_1 = x5.Joint(j1 = 99.772, j2 = 6.549, j3 = 15.746, j4 = -85.976, 
     j5 = -23.976, j6 = -46.358, e1 = -57.616, e2=0, e3=450)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 到达抓取点位（预设未调整） new
    pick_2 = x5.Joint(j1 =99.942,j2 = -25.440, j3 = 14.215, j4 = -56.875, 
     j5 = -23.397, j6 = -49.442, e1 = -64.349, e2=0, e3=450)
    pick_2_pose = x5.Pose(x=115.928, y=-448.185, z=211.988, a=128.667, b=-2.779, c=2.933, e1=8.023, e2=0, e3=450)
    pick_2_point = x5.Point(pose=pick_2_pose, uf=0, tf=0, cfg=(0,0,0,7))

    # 计算新的调整后的抓取点位
    new_pick_2_point = copy.deepcopy(pick_2_point)
    new_pick_2_point.pose.z = 211.988 - delta_x
    new_pick_2_point.pose.y = -448.185 + delta_y
    print(new_pick_2_point)
    # 逆解，求笛卡尔点位p1的对应关节坐标
    jp1 = x5.cnvrt_j(handle_R, new_pick_2_point, 1, pick_2)

    x5.movj(handle_R, pick_2, add_data)
    x5.wait_move_done(handle_R)
    time.sleep(1)
    ## 夹手
    gripper_right.setpos(72)
    time.sleep(1)

    # 升起 new
    pick_3 = x5.Joint(j1 =104.943,j2 = -6.804, j3 = 14.215, j4 = -39.943, 
     j5 = -14.913, j6 = -49.442, e1 = -84.449, e2=0, e3=450)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)



    pick_5 = x5.Joint(j1 = 37.176, j2 = -5.808, j3 = 14.215, j4 = -48.033 ,
     j5 =25.527, j6 = -62.882, e1 = -84.449, e2=0, e3=160)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

    # with AGVClient(ip='192.168.1.51') as agv:
    #     agv.go_to_point_in_world(-0.779,-0.037,-3.14,1)

    # # 放置
    # pick_6 = x5.Joint(j1 = 58.186, j2 = -70.701, j3 = -15.641, j4 = -6.684, 
    #  j5 = 46.422, j6 = -43.152, e1 = -132.991, e2=0, e3=160)
    # x5.movj(handle_R, pick_6, add_data)
    # x5.wait_move_done(handle_R)
    # time.sleep(1)
    # ## 松手
    # time.sleep(1)

    # pick_7 = x5.Joint(j1 = 3.828, j2 = -2.122, j3 = 25.323, j4 = -89.585 ,
    #  j5 = 107.625, j6 = -77.819, e1 = -157.687, e2=0, e3=160)
    # x5.movj(handle_R, pick_7, add_data)
    # x5.wait_move_done(handle_R)

    # # pre place
    # pick_7 = x5.Joint(j1 = -38.652, j2 = -2.122, j3 = 43.077, j4 = -40.520, 
    #  j5 = 117.594, j6 = -65.691, e1 = -140.857, e2=0, e3=160)
    # x5.movj(handle_R, pick_7, add_data)
    # x5.wait_move_done(handle_R)


    # # 到达安全点
    # pick_8 = x5.Joint(j1 =-12.429,j2 = 4.995, j3 = 5.556, j4 = -89.311, 
    #  j5 = 117.431, j6 = -88.0, e1 = -137.357, e2=0, e3=160)
    # x5.movj(handle_R, pick_8, add_data)
    # x5.wait_move_done(handle_R)





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
    with AGVClient(ip='192.168.1.51') as agv:
        agv.go_to_point_in_world(-0.779,-0.037,-3.14,1)

def main():
    gripper_left = GripperController(port="COM10", baudrate=115200)
    gripper_right = GripperController(port="COM11", baudrate=115200)
    gripper_left.set_temp_torque(15)
    gripper_right.set_temp_torque(15)
    gripper_left.setpos(0)
    gripper_right.setpos(0)

    add_data_1 = x5.MovPointAdd(vel=100, acc=100)
    add_data_2 = x5.MovPointAdd(vel=100, cnt=100, acc=100, dec=100, offset =-1,
    offset_data=(10,0,0,0,0,0,0,0,0))
    # # 连接机器人
    handle_l = x5.connect("192.168.1.9")
    handle_r = x5.connect("192.168.1.10")

    # # safe_robot(handle_l, handle_r, add_data_1)
    # with AGVClient(ip='192.168.192.5') as agv:
    #     # agv.go_to_target_LM("LM1", "LM2")
    #     agv.go_to_target_LM("LM2", "LM4")
    #     time.sleep(15)
    #     agv.go_to_target_LM("LM4", "LM3")
    #     time.sleep(15)
    #     # print(agv.get_pose())
    # time.sleep(35)
    # move_to_LM()
    init_robot(handle_l, handle_r, add_data_1)
    pick_2_4(handle_l, handle_r, add_data_1,gripper_left,gripper_right)
    # wave(handle_l, handle_r, add_data_1)
    # pick_1_5(handle_l, handle_r, add_data_1,gripper_left,gripper_right)
    # move_to_pick_height_pitch_angle(handle_l, handle_r, hand_l, hand_r, add_data_1, 200, 0)
    # init_robot(handle_l, handle_r, add_data_1)


if __name__ == "__main__":
    main()