import xapi.api as x5
import time as t
 # 控制器IP地址
ip = "192.168.1.8"
 # 创建连接

INIT_POS_L = x5.Pose(x=-200, y=245, z=5, a=160, b=-59.853, c=30.601, e1=-90, e2=0, e3=0)
INIT_POINT_L = x5.Point(pose=INIT_POS_L, uf=0, tf=0, cfg=(0,0,0,7))
INIT_JOINT_L = x5.Joint(j1=5.096, j2=-80.406, j3=-97.344, j4=-90.463, j5=23.051, j6=-77.647, e1=-20.340+90, e2=0, e3=0)


handle = x5.connect(ip)
print(f"\n 控制器句柄号：\n{handle}\n")
# try:
# 逆解，求笛卡尔点位p1的对应关节坐标
p1 = INIT_POINT_L
# 设定逆解类型及参考的关节点位
type, joint = 0,INIT_JOINT_L
# 逆解求p1的对应关节点位
jp1 = x5.cnvrt_j(handle, p1, type, joint)
print(jp1)
print(INIT_JOINT_L)
# except x5.RobException as rex:
#     print(f"错误代码: {rex.error_code}")  
# print(f"错误内容: {rex.error_message}")  
# t.sleep(1)