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
    delta_x = (x+0.08)*1000  #-0.0874+0.08=-0.0074
    delta_y = (y+0.035)*1000 #-0.0376+0.035=-0.0026

    # 到达和货架同一高度
    pick_1 = x5.Joint(j1 = 101.021, j2 = 6.288, j3 = 14.716, j4 = -84.667, 
     j5 = -23.874, j6 = -27.920, e1 = 17.289-90, e2=0, e3=450)
    x5.movj(handle_R, pick_1, add_data)
    x5.wait_move_done(handle_R)

    # 到达抓取点位（预设未调整）
    pick_2 = x5.Joint(j1 =102.436,j2 = -66.410, j3 = 7.210, j4 = -9.507, 
     j5 = -8.365, j6 = -22.865, e1 = -88.701, e2=0, e3=450)
    pick_2_pose = x5.Pose(x=143.615, y=-622.309, z=152.984, a=98.703, b=0.221, c=13.085, e1=7.010, e2=0, e3=450)
    pick_2_point = x5.Point(pose=pick_2_pose, uf=0, tf=0, cfg=(0,0,0,7))

    # 计算新的调整后的抓取点位
    new_pick_2_point = copy.deepcopy(pick_2_point)
    new_pick_2_point.pose.z = 152.984 + delta_x
    new_pick_2_point.pose.y = -622.309 - delta_y
    # 逆解，求笛卡尔点位p1的对应关节坐标
    jp1 = x5.cnvrt_j(handle_R, new_pick_2_point, 0, pick_2)
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

    # # # safe_robot(handle_l, handle_r, add_data_1)
    # with AGVClient(ip='192.168.192.5') as agv:
    #     # agv.go_to_target_LM("LM1", "LM2")
    #     agv.go_to_target_LM("LM2", "LM1")
    # time.sleep(35)

    init_robot(handle_l, handle_r, add_data_1)

    pick_1_5(handle_l, handle_r, hand_l, hand_r, add_data_1)
    
    init_robot(handle_l, handle_r, add_data_1)

if __name__ == "__main__":
    main()