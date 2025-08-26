import xapi.api as x5


import xapi.api as x5
import time as t
# 控制器 IP 地址
ip = "192.168.1.6"
# 创建连接
handle = x5.connect(ip)
print(f"\n 控制器句柄号：\n{handle}\n")

# 逆解，求笛卡尔点位 p1 的对应关节坐标
p1 = x5.Point((200,200,0,0,0,0,0,0,0),0,0,(0,0,0,1))
# 设定逆解类型及参考的关节点位
type, joint = 0, x5.Joint(46,-1,39,180,0,0,0,0,0)
# 逆解求 p1 的对应关节点位
jp1 = x5.cnvrt_j(handle, p1, type, joint)
print(jp1)

# 逆解类型（inverse_type）参数说明：
# 0: onlyconf，只考虑给定的关节配置（cfg），不考虑转动数（turn），
#    只返回与参考关节配置cfg一致的解，适合已知机械臂当前姿态且只想要同一配置的解。
# 1: conf&turn，同时考虑关节配置（cfg）和转动数（turn），
#    返回与参考关节配置和转动数都一致的解，适合对机械臂姿态和关节转动数都有要求的场景。
# 2: shortest，返回与参考关节角度距离最近的解（欧氏距离最小），
#    不强制要求cfg和turn完全一致，适合只关心运动距离最短、效率最高的场景。

# 举例说明：
# - 如果你想让机械臂保持当前的关节配置，只需用0（onlyconf）。
# - 如果你还要求转动数（比如多圈关节），用1（conf&turn）。
# - 如果你只想让机械臂以最短路径到达目标点，不关心配置和转动数，用2（shortest）。


def calibration_action(handle, point, joint, type):
    """
    逆解，求笛卡尔点位 p1 的对应关节坐标
    Args:
        handle: 控制器句柄
        point: 笛卡尔点位
        joint: 参考关节点位
        type: 逆解类型 一般使用0
    Returns:
        关节点位
    """
    jp1 = x5.cnvrt_j(handle, point, type, joint)
    return jp1

if __name__ == "__main__":

    ip = "192.168.1.6"
    # 创建连接
    handle = x5.connect(ip)

    point = x5.Point((200,200,0,0,0,0,0,0,0),0,0,(0,0,0,1))
    joint_reference = x5.Joint(46,-1,39,180,0,0,0,0,0)
    joint = calibration_action(handle, point, joint_reference, 0)