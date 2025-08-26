from cgi import print_directory
import xapi.api as x5
import copy

# 初始化右臂
INIT_POS_R = x5.Pose(x=-200, y=-245, z=5, a=170.075, b=-59.624, c=-16.416, e1=90, e2=0, e3=160)
INIT_POINT_R = x5.Point(pose=INIT_POS_R, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_R = x5.Joint(j1=-7.277, j2=-80.381, j3=97.428, j4=-91.135, j5=-24.051, j6=-74.093, e1=12.445-90, e2=0, e3=160)

# 初始化左臂
INIT_POS_L = x5.Pose(x=-200, y=245, z=5, a=172.652, b=-59.853, c=30.601, e1=-90, e2=0, e3=0)
INIT_POINT_L = x5.Point(pose=INIT_POS_L, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_L = x5.Joint(j1=5.096, j2=-80.406, j3=-97.344, j4=-90.463, j5=23.051, j6=-77.647, e1=-20.340+90, e2=0, e3=0)

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

def bow(handle_l, handle_r,add_data):
    """
    鞠躬
    """
    ## 鞠躬
    bow_joint_pose = copy.copy(INIT_JOINT_R)
    bow_joint_pose.e2 = 20
    bow_joint_pose.j5 = -75
    bow_joint_pose.j6 = -88
    
    # 向下鞠躬
    x5.movj(handle_r, bow_joint_pose, add_data)
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
    if angle == None:
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
    handle_l = x5.connect("192.168.1.7")
    handle_r = x5.connect("192.168.1.8")
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
