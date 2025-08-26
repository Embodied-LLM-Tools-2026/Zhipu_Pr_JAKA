import numpy as np
from scipy.spatial.transform import Rotation
from agv_client import AGVClient
import xapi.api as x5

'''
纠偏：小车在实际运动过程中难以到达设定的目标点位，因此需要进行调整，根据小车的实际位姿和机械臂末端的实际姿态，调整机械臂末端的实际位姿
以下公式以行主序来表示：
P_hand_ArmCur = P_hand_ArmTar @ X_ArmTar_AGVTar @ X_AGVTar_AGVCur @ X_AGVcur_ArmCur
= P_hand_ArmTar @ X_Arm_AGV @ X_AGVTar_AGVCur @ inv(X_AGVCur_ArmCur)
其中X_AGVTar_AGVcur = X_AGVTar_world @ inv(X_AGVCur_world)

变量解释：
P_hand_ArmTar: 机械臂末端在机械臂基坐标系下的目标位姿（录制的点位）(类型：x5.Pose) 
X_Arm_AGV: 机械臂基坐标系与AGV坐标系之间的坐标变换 （固定不变的，由机器人本体决定）(类型：列表或者元组) 
X_AGVTar_AGVcur: 目标AGV坐标系与当前AGV坐标系之间的坐标变换 （需要根据小车实际运动情况计算出来）

输出参数：
Pose_hand_ArmCur: 机械臂末端在当前机械臂基坐标系下的新位姿(类型：x5.Pose) 
'''
def correct_hand_pose(X_Arm_AGV,Pose_AGVTar_world,Pose_hand_ArmTar):
    # 获取当前AGV的位姿
    with AGVClient(ip='192.168.1.50') as agv:
        pose_result = agv.get_pose()
        x, y, theta = pose_result
    X_AGVCur_world = np.array([[np.cos(theta),np.sin(theta),0,0],
                              [-np.sin(theta),np.cos(theta),0,0],
                              [0,0,1,0],
                              [x,y,0,1]])
    # 解析P_AGVTar_world为齐次矩阵
    x, y, theta = Pose_AGVTar_world
    X_AGVTar_world = np.array([[np.cos(theta),np.sin(theta),0,0],
                              [-np.sin(theta),np.cos(theta),0,0],
                              [0,0,1,0],
                              [x,y,0,1]])
    # 计算目标AGV坐标系与当前AGV坐标系之间的坐标变换
    X_AGVTar_AGVcur = X_AGVTar_world @ np.linalg.inv(X_AGVCur_world)

    # 解析 Pose_hand_ArmTar 为齐次矩阵
    # 将6自由度位姿（xyzabc，abc为角度制）转换为齐次变换矩阵
    all_pose_list = np.array(Pose_hand_ArmTar.tolist())
    pose_list = all_pose_list[:6]
    x, y, z, a, b, c = pose_list
    R_matrix = Rotation.from_euler('zyx', [c, b, a], degrees=True).as_matrix()# 计算旋转矩阵（ZYX欧拉角，先绕Z，再Y，再X）
    Pose_hand_ArmTar_array = np.eye(4)
    Pose_hand_ArmTar_array[:3, :3] = R_matrix
    Pose_hand_ArmTar_array[:3, 3] = [x, y, z]
    P_hand_ArmTar = Pose_hand_ArmTar_array.T #转为行主序

    # 计算机械臂末端在当前机械臂基坐标系下的新位姿
    P_hand_ArmCur = P_hand_ArmTar @ X_Arm_AGV @ X_AGVTar_AGVcur @ np.linalg.inv(X_Arm_AGV)
    # 将P_hand_ArmCur齐次变换矩阵转为6自由度位姿
    T = P_hand_ArmCur.T
    t = T[:3, 3]
    c,b,a = Rotation.from_matrix(T[:3, :3]).as_euler('zyx', degrees=True) 
    Pose_hand_ArmCur = x5.Pose(x=t[0], y=t[1], z=t[2], a=a, b=b, c=c, e1=all_pose_list[6], e2=all_pose_list[7], e3=all_pose_list[8])
    return Pose_hand_ArmCur

def correct_left_arm(X_AGVTar_world,P_hand_ArmTar):
    X_LeftArm_AGV = np.array([[0,0,1,0],
                              [1,0,0,0],
                              [0,1,0,0],
                              [301,206.23,0,1]])
    P_hand_LeftArmCur = correct_hand_pose(X_LeftArm_AGV,X_AGVTar_world,P_hand_ArmTar)
    return P_hand_LeftArmCur

def correct_right_arm(X_AGVTar_world,P_hand_ArmTar):
    X_RightArm_AGV = np.array([[0,0,1,0],
                              [1,0,0,0],
                              [0,1,0,0],
                              [301,-206.23,0,1]])
    P_hand_RightArmCur = correct_hand_pose(X_RightArm_AGV,X_AGVTar_world,P_hand_ArmTar)
    return P_hand_RightArmCur
