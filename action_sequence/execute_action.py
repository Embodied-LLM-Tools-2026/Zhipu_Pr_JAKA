from cgi import print_directory
import xapi.api as x5
import copy

# 初始化右臂
INIT_POS_R = x5.Pose(x=-166.48863412000844, y=-249.65707063473792, z=-18.802048871171785, a=169.26500545380372, b=-55.70720124022451, c=7.111138510153568, e1=48.61143291529611, e2=0.003999999999999999, e3=160.0)
INIT_POINT_R = x5.Point(pose=INIT_POS_R, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_R = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -79.671, e1 = -120, e2 = 0.004, e3 = 160)

# 初始化左臂
INIT_POS_L = x5.Pose(x=-149.3199993845242, y=240.45491129421245, z=-15.187910316510676, a=-172.05249999999998, b=-50.210699999999996, c=-6.759200000000021, e1=-48.61143290082501, e2=-0.009, e3=0.016)
INIT_POINT_L = x5.Point(pose=INIT_POS_L, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_L = x5.Joint(j1 = -6.809, j2 = -55.111, j3 = -63.25, j4 = -94.793, j5 = 0.773, j6 = -75.875, e1 = 116.933, e2 = -0.009, e3 = 0.016)

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

    ## 回到初始点位
    x5.movj(handle_r, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_r)

def bow(handle_l, handle_r,add_data):
    """
    鞠躬
    """
    ## 鞠躬
    bow_joint_pose = copy.copy(INIT_JOINT_R)
    bow_joint_pose.e2 = 20
    bow_joint_pose.j5 = -75
    bow_joint_pose.j6 = -88
    
    bow_joint_pose1 = x5.Joint(j1 = 7.892, j2 = -58.596, j3 = 60.106, j4 = -89, j5 = 3.16, j6 = -46.4, e1 = -120, e2 = 0.004, e3 = 160)
    x5.movj(handle_r, bow_joint_pose1, add_data)
    x5.wait_move_done(handle_r)

    # 向下鞠躬
    bow_joint_pose2 = x5.Joint(j1 = 7.887, j2 = -58.601, j3 = 60.103, j4 = -88.976, j5 = -77.456, j6 = -56.704, e1 = -120.003, e2 = 20.647, e3 = 159.99)
    x5.movj(handle_r, bow_joint_pose2, add_data)
    x5.wait_move_done(handle_r)
    ## 回到初始点位

    x5.movj(handle_r, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_r)

def Nod(handle_L,handle_R,add_data):
    """
    点头
    """

    # 点头的 point
    point_L = copy.copy(INIT_POINT_L)
    point_L.pose.e3 += 20
    # 第一次向下 
    # x5.servol(handle_L, 100)
    x5.movl(handle_L, point_L, add_data)
    x5.wait_move_done(handle_L)

    # # 第一次向上
    x5.movl(handle_L, INIT_POINT_L, add_data)
    x5.wait_move_done(handle_L)

    # 第二次向下
    x5.movl(handle_L, point_L, add_data)
    x5.wait_move_done(handle_L)

    # 第二次向上
    x5.movl(handle_L, INIT_POINT_L, add_data)
    x5.wait_move_done(handle_L)

def Shake_head(handle_L,handle_R,add_data):
    """
    摇头
    """

    point_L_1 = copy.copy(INIT_POINT_L)
    point_L_2 = copy.copy(INIT_POINT_L)
    point_L_1.pose.e2 += 20
    point_L_2.pose.e2 -= 20
    x5.movl(handle_L, point_L_1, add_data)
    x5.wait_move_done(handle_L)
    x5.movl(handle_L, point_L_2, add_data)
    x5.wait_move_done(handle_L)
    x5.movl(handle_L, point_L_1, add_data)
    x5.wait_move_done(handle_L)
    x5.movl(handle_L, INIT_POINT_L, add_data)
    x5.wait_move_done(handle_L)

 
def rotate_head_to_angle(handle_L,handle_R,add_data, angle, incremental=False, back_to_init=False):
    """
    转头到指定角度
    """
    point_L_1 = copy.copy(INIT_POINT_L)
    if incremental:
        point_L_1.pose.e2 += angle
    else:
        point_L_1.pose.e2 = angle
    x5.movl(handle_L, point_L_1, add_data)
    x5.wait_move_done(handle_L)
    if back_to_init:
        x5.movl(handle_L, INIT_POINT_L, add_data)
        x5.wait_move_done(handle_L)
        

def init_robot(handle_l, handle_r, add_data):
    # 初始化左臂
    x5.movj(handle_l, INIT_JOINT_L, add_data)
    x5.wait_move_done(handle_l)
    # 初始化右臂
    x5.movj(handle_r, INIT_JOINT_R, add_data)
    x5.wait_move_done(handle_r)

def main():



    add_data_1 = x5.MovPointAdd(vel=100, acc=100)
    add_data_2 = x5.MovPointAdd(vel=100, cnt=100, acc=100, dec=100, offset =-1,
    offset_data=(10,0,0,0,0,0,0,0,0))
    # 连接机器人
    handle_l = x5.connect("192.168.1.9")
    handle_r = x5.connect("192.168.1.10")
    print("连接成功")
    init_robot(handle_l, handle_r, add_data_1)
    print("初始化成功")
    # 点头
    Nod(handle_l, handle_r,add_data_2)
    # 摇头
    Shake_head(handle_l, handle_r,add_data_2)
    # 挥手
    wave(handle_l, handle_r,add_data_1)
    # 鞠躬
    bow(handle_l, handle_r,add_data_1)

if __name__ == "__main__":
    main()
