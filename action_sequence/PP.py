import xapi.api as x5
import copy
import time
import os

try:
    from controller.hand_controller import InspireHandR
    from controller.AGV_controller import AGVClient
except ImportError:
    # 如果找不到模块，尝试使用相对路径导入
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from controller.hand_controller import InspireHandR
        from controller.AGV_controller import AGVClient

    except ImportError:
        print("无法导入InspireHandR，请检查controller/hand_controller.py路径。")



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
    with AGVClient(ip='192.168.192.5') as agv:
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





def move_to_pick_height_pitch_angle(handle_L,handle_R,hand_l, hand_r, add_data, height, pitch_angle):
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
    hand_l = InspireHandR(port="COM11", baudrate=115200, hand_id=1)
    hand_r = InspireHandR(port="COM12", baudrate=115200, hand_id=2)
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
    handle_l = x5.connect("192.168.1.7")
    handle_r = x5.connect("192.168.1.8")

    # # safe_robot(handle_l, handle_r, add_data_1)
    # with AGVClient(ip='192.168.192.5') as agv:
    #     # agv.go_to_target_LM("LM1", "LM2")
    #     agv.go_to_target_LM("LM2", "LM4")
    #     time.sleep(15)
    #     agv.go_to_target_LM("LM4", "LM3")
    #     time.sleep(15)
    #     # print(agv.get_pose())
    # time.sleep(35)
    move_to_LM()
    init_robot(handle_l, handle_r, add_data_1, hand_l, hand_r)

    pick_1_5(handle_l, handle_r, hand_l, hand_r, add_data_1)
    # move_to_pick_height_pitch_angle(handle_l, handle_r, hand_l, hand_r, add_data_1, 200, 0)
    init_robot(handle_l, handle_r, add_data_1, hand_l, hand_r)


if __name__ == "__main__":
    main()