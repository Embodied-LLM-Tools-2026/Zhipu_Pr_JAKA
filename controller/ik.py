import xapi.api as x5
import time as t
import copy
 # 控制器IP地址
ip = "192.168.1.8"
 # 创建连接

pick_2 = x5.Joint(j1 = 102.436,j2 = -66.410, j3 = 7.210, j4 = -9.507, 
    j5 = -8.365, j6 = -22.865, e1 = -88.701, e2=0, e3=450)


pick_2_pose = x5.Pose(x=143.615, y=-622.309, z=152.984, a=98.703, b=0.221, c=13.085, e1=7.010, e2=0, e3=450)
pick_2_point = x5.Point(pose=pick_2_pose, uf=0, tf=0, cfg=(0,0,0,7))

new_pick_2_point = copy.deepcopy(pick_2_point)
new_pick_2_point.pose.z = 152.984 - 7.4
new_pick_2_point.pose.y = -622.309 + 2.6






handle = x5.connect(ip)
print(f"\n 控制器句柄号：\n{handle}\n")
# try:
# 逆解，求笛卡尔点位p1的对应关节坐标
p1 = new_pick_2_point
# 设定逆解类型及参考的关节点位
# type, joint = 0,INIT_JOINT_L
# 逆解求p1的对应关节点位
jp1 = x5.cnvrt_j(handle, new_pick_2_point, 0, pick_2)
print(jp1)
# print(INIT_JOINT_L)
# except x5.RobException as rex:
#     print(f"错误代码: {rex.error_code}")  
# print(f"错误内容: {rex.error_message}")  
# t.sleep(1)