import xapi.api as x5
import time
import math
import numpy as np

import numpy as np

def create_transform_matrix(a, b, c):
    """
    创建表示位姿的4x4变换矩阵
    
    参数:
        a: x坐标
        b: y坐标
        c: 绕z轴的旋转角度(弧度)
        
    返回:
        4x4变换矩阵
    """
    # 旋转部分 (绕z轴旋转)
    cos_c = np.cos(c)
    sin_c = np.sin(c)
    rotation_matrix = np.array([
        [cos_c, -sin_c, 0],
        [sin_c,  cos_c, 0],
        [0,      0,     1]
    ])
    
    # 平移部分
    translation = np.array([a, b, 0])
    
    # 创建4x4齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = rotation_matrix  # 旋转部分
    T[:3, 3] = translation       # 平移部分
    
    return T

# 示例使用
a = -0.126  # x坐标
b = -1.007  # y坐标
c = np.pi * -1.66 / 180 # 绕z轴旋转45度(π/4弧度)

P_AGVWorld = create_transform_matrix(a, b, c)
print(P_AGVWorld)

a2 = -0.1265
b2 = -1.0056
c2 = -0.0206
print("位姿的4x4变换矩阵:")
P_AGV2World = create_transform_matrix(a2, b2, c2)
print(P_AGV2World)

# AGV 到 LA 的变换矩阵
X_AGVLA = np.array([[0,0,1,0],[1,0,0,-0.3],[0,1,0,-0.2],[0,0,0,1]])

# AGV 坐标系下的点
P_LAAGV = np.array([0.3, 0.2, 0, 1])

Origin = np.array([0,0,0,1])
# AGV 坐标系下的点

if __name__ == "__main__":
    # 简单示例：读取当前关节为起点，S型速度曲线执行到目标关节
    pass