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

def safe_robot(handle_l, handle_r, add_data):
    # 初始化左臂
    pick_2 = x5.Joint(j1 =101.441,j2 = -54.909, j3 = 8.036, j4 = -34.658, 
     j5 = -23.874, j6 = -9.847, e1 = 17.289, e2=0, e3=450)
    x5.movj(handle_l, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_l)
    # 初始化右臂
    x5.movj(handle_r, pick_2, add_data)
    x5.wait_move_done(handle_r)

def pick_1_5(handle_L,handle_R,hand_l, hand_r, add_data):
    """
    抓取
    """
    # 计算与目标参考点位的差值
    with AGVClient(ip='192.168.1.50') as agv:
        pose_result = agv.get_pose()
        x, y, angle = pose_result
    delta_x = (x-0.019)*1000  #-0.0874+0.08=-0.0074
    delta_y = (y+0.028)*1000 #-0.0376+0.035=-0.0026


    # 到达和货架同一高度
    pick_0 = x5.Joint(j1 = 99.772, j2 = -11.213, j3 = 31.413, j4 = -85.976, 
     j5 = -24.194, j6 = -57.412, e1 = -73.696, e2=0, e3=450)
    x5.movj(handle_R, pick_0, add_data)
    x5.wait_move_done(handle_R)

    # 到达和货架同一高度
    pick_1 = x5.Joint(j1 = 101.021, j2 = 6.288, j3 = 14.716, j4 = -84.667, 
     j5 = -23.874, j6 = -27.920, e1 = 17.289-90, e2=0, e3=450)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 到达抓取点位（预设未调整） new
    pick_2 = x5.Joint(j1 =119.698,j2 = -39.530, j3 = -23.615, j4 = -59.361, 
     j5 = 11.857, j6 = -6.561, e1 = -84.501, e2=0, e3=450)
    pick_2_pose = x5.Pose(x=153.800, y=-528.378, z=174.374, a=102.804, b=-2.376, c=8.520, e1=-16.605, e2=0, e3=450)
    pick_2_point = x5.Point(pose=pick_2_pose, uf=0, tf=0, cfg=(0,0,0,7))

    # 计算新的调整后的抓取点位
    new_pick_2_point = copy.deepcopy(pick_2_point)
    new_pick_2_point.pose.z = 174.374 - delta_x
    new_pick_2_point.pose.y = -528.378 + delta_y
    print(new_pick_2_point)
    # 逆解，求笛卡尔点位p1的对应关节坐标
    jp1 = x5.cnvrt_j(handle_R, new_pick_2_point, 1, pick_2)

#     # # print(jp3)
    x5.movj(handle_R, jp1, add_data)
    x5.wait_move_done(handle_R)
    # grasp
    hand_r.setpos(472,509,589,670,736,0)
    time.sleep(1)

    # 升起 new
    pick_3 = x5.Joint(j1 =140.679,j2 = -24.452, j3 = -30.001, j4 = -56.009, 
     j5 = -21.567, j6 = -29.905, e1 = -53.686, e2=0, e3=450)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # # 返回和货架同一高度点
    pick_4 = x5.Joint(j1 = 101.021, j2 = 6.288, j3 = 14.716, j4 = -84.667, 
     j5 = -23.874, j6 = -27.920, e1 = 17.289-90, e2=0, e3=450)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)

    # pre place
    pick_5 = x5.Joint(j1 = -38.652, j2 = -2.122, j3 = 43.077, j4 = -40.520, 
     j5 = 117.594, j6 = -65.691, e1 = -140.857, e2=0, e3=160)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)

    # 放置
    pick_6 = x5.Joint(j1 = 4.175, j2 = -24.606, j3 = -16.329, j4 = -48.184, 
     j5 = 162.139, j6 = -82.084, e1 = -163.851, e2=0, e3=160)
    x5.movj(handle_R, pick_6, add_data)
    x5.wait_move_done(handle_R)
    time.sleep(2)
    hand_r.setpos(1000,1000,1000,1000,1000,1000)
    time.sleep(2)

    # pre place
    pick_7 = x5.Joint(j1 = -38.652, j2 = -2.122, j3 = 43.077, j4 = -40.520, 
     j5 = 117.594, j6 = -65.691, e1 = -140.857, e2=0, e3=160)
    x5.movj(handle_R, pick_7, add_data)
    x5.wait_move_done(handle_R)


    # 到达安全点
    pick_8 = x5.Joint(j1 =-12.429,j2 = 4.995, j3 = 5.556, j4 = -89.311, 
     j5 = 117.431, j6 = -88.0, e1 = -137.357, e2=0, e3=160)
    x5.movj(handle_R, pick_8, add_data)
    x5.wait_move_done(handle_R)

def move_to_shelf():
    with AGVClient(ip='192.168.1.50') as agv:
        pose_result = agv.get_pose()
        if abs(pose_result[2]) < 0.4:
            agv.go_to_point_in_world(0.450,0.038,0, 0)
        else:
            agv.rotation(3.14)
            agv.go_to_point_in_world(0.450,0.038,0, 0)

def back_bar_station():
    with AGVClient(ip='192.168.1.50') as agv:
        pose_result = agv.get_pose()
        if abs(pose_result[2]) > 2.5:
            agv.go_to_point_in_world(-0.300,0.038,3.14, 1)
            # agv.rotation(1.57)
        else:
            agv.go_to_point_in_world(-0.300,0.038,0, 1)
            agv.rotation(3.14)

def pick_5_5(handle_L,handle_R,hand_l,hand_r,add_data):
    """
    抓取
    """
    # 计算与目标参考点位的差值
    with AGVClient(ip='192.168.1.50') as agv:
        # agv.go_to_point_in_world(0.420,0.038,0, 0)
        pose_result = agv.get_pose()
        x, y, angle = pose_result
    delta_x = (x-0.450)*1000  #-0.0874+0.08=-0.0074
    delta_y = (y-0.038)*1000 #-0.0376+0.035=-0.0026

    pick_1 = x5.Joint(j1 = 101.021, j2 = 6.288, j3 = 14.716, j4 = -84.667, 
     j5 = -23.874, j6 = -27.920, e1 = 17.289-90, e2=0, e3=450)
    # 到达和货架同一高度
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 到达抓取点位
    pick_2 = x5.Joint(j1 =101.441,j2 = -54.909, j3 = 8.036, j4 = -34.658, 
     j5 = -23.874, j6 = -9.847, e1 = 17.289-90, e2=0, e3=450)
    pick_2_point = x5.Pose(x=143.616, y=-598.703, z=152.984, a=98.703, b=0.220, c=13.085, e1=7.011, e2=0, e3=450)
    pick_2_point = x5.Point(pose=pick_2_point, uf=0, tf=0, cfg=(0,0,0,7))

    # 计算新的调整后的抓取点位  机器人y轴对应agv x轴，机器人右臂z轴对应agv y轴
    new_pick_2_point = copy.deepcopy(pick_2_point)
    new_pick_2_point.pose.z = 152.984 - delta_y
    new_pick_2_point.pose.y = -598.703 + delta_x
    print(new_pick_2_point)
    # 逆解，求笛卡尔点位p1的对应关节坐标
    jp1 = x5.cnvrt_j(handle_R, new_pick_2_point, 1, pick_2)


    x5.movj(handle_R, jp1, add_data)
    x5.wait_move_done(handle_R)

    # grasp
    hand_r.setpos(472,509,589,670,736,0)
    time.sleep(1)

    # 升起
    pick_3 = x5.Joint(j1 =106.506,j2 = -51.921, j3 = 8.036, j4 = -37.324, 
     j5 = -23.874, j6 = -9.847, e1 = 17.289-90, e2=0, e3=450)
    x5.movj(handle_R, pick_3, add_data)
    x5.wait_move_done(handle_R)

    # # 返回和货架同一高度点
    pick_4 = x5.Joint(j1 = 101.021, j2 = 6.288, j3 = 14.716, j4 = -84.667, 
     j5 = -23.874, j6 = -27.920, e1 = 17.289-90, e2=0, e3=450)
    x5.movj(handle_R, pick_4, add_data)
    x5.wait_move_done(handle_R)


    # pre 待机位置
    pick_5 = x5.Joint(j1 = 85.512, j2 = -14.709, j3 = 14.716, j4 = -102.856, 
     j5 = -23.874, j6 = -27.920, e1 = -72.711, e2=0, e3=450)
    x5.movj(handle_R, pick_5, add_data)
    x5.wait_move_done(handle_R)
    # 回到待机位置
    pick_6 = x5.Joint(j1 = 37.176, j2 = -5.808, j3 = 14.215, j4 = -48.033 ,
     j5 =25.527, j6 = -62.882, e1 = -84.449, e2=0, e3=160)
    x5.movj(handle_R, pick_6, add_data)
    x5.wait_move_done(handle_R)
    print("抓取5号")

    # 回到待机位置1
    pick_7 = x5.Joint(j1 = 37.176, j2 = 15.535, j3 = 14.215, j4 = -68.999 ,
     j5 =25.527, j6 = -87.195, e1 = -84.449, e2=0, e3=160)
    x5.movj(handle_R, pick_7, add_data)
    x5.wait_move_done(handle_R)

    # 回到待机位置2
    pick_8 = x5.Joint(j1 =33.298, j2 = -2.419, j3 = 24.806, j4 = -77.036 ,
    j5 = 21.211, j6 = -88, e1 = -83.869, e2=0, e3=130)
    x5.movj(handle_R, pick_8, add_data)
    x5.wait_move_done(handle_R)

    # 开始移动到吧台
    with AGVClient(ip='192.168.1.50') as agv:
        agv.go_to_point_in_world(-0.300,0.038,0, 1)
        agv.rotation(3.14)

    # 预设点位，需要设定第二个
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
    hand_l.set_default_speed(100,100,100,100,100,100)
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
    init_robot(handle_l, handle_r, add_data_1, hand_l, hand_r)

if __name__ == "__main__":
    main()

