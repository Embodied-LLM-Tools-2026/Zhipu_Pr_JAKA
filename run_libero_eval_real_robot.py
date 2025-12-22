'''
LIBERO评估脚本 - 真实JAKA机器人版本 (无ROS2依赖)

使用真实的JAKA机器人和相机替代LIBERO仿真环境。
直接使用 OpenCV 读取相机和底层 API 控制机器人。

数据接口:
1. 相机图像: 直接用 OpenCV 读取 (cv2.VideoCapture)
2. 末端位置: 通过 JAKA SDK 获取
3. 末端位姿: 通过 JAKA SDK 获取
4. 夹爪状态: 通过 Modbus/RS485 直接控制

动作执行:
1. 位置/位姿增量: 直接调用 JAKA SDK servo_p 接口
2. 夹爪: 通过 gripper_controller (Modbus) 直接控制
'''

import collections
import dataclasses
import logging
import math
import time
import pathlib
import os
import sys
import threading
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
import cv2
import tyro
import rclpy
from rclpy.node import Node
from jaka_msgs.srv import ServoMove, ServoMoveEnable

# 添加路径以导入 gripper_controller 和其他本地模块
current_dir = str(pathlib.Path(__file__).parent)
sys.path.insert(0, current_dir)

try:
    from jaka_teleop.gripper_controller import GripperController
except ImportError:
    try:
        # 备选：直接从当前目录导入
        sys.path.insert(0, os.path.join(current_dir, 'jaka_teleop'))
        from gripper_controller import GripperController
    except ImportError:
        logging.warning("⚠️ 无法导入 GripperController，夹爪功能将被禁用")
        GripperController = None


import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


import logging
import time
import websockets.sync.client

from typing import Dict, Optional, Tuple

class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers, 
                    # ping_interval=60, ping_timeout=120
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)


class JakaServoClientAsync(Node):
    """JAKA 机器人客户端（支持 IK、joint_move、servo_p）"""
    
    def __init__(self):
        super().__init__('jaka_robot_client')
        self.servo_move_enable_client = self.create_client(ServoMoveEnable, '/jaka_driver/servo_move_enable')
        self.servo_p_client = self.create_client(ServoMove, '/jaka_driver/servo_p')
        
        # 导入 IK 和 joint_move 的消息类型
        from jaka_msgs.srv import GetIK, Move
        self.get_ik_client = self.create_client(GetIK, '/jaka_driver/get_ik')
        self.joint_move_client = self.create_client(Move, '/jaka_driver/joint_move')

        while not self.servo_move_enable_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('servo_move_enable service not available, waiting again...')
        
        # 等待 IK 和 joint_move 服务
        while not self.get_ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_ik service not available, waiting again...')
        
        while not self.joint_move_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('joint_move service not available, waiting again...')
        
        self.servo_req = ServoMove.Request()

    def enable_servo_mode(self):
        """启用伺服模式"""
        enable_request = ServoMoveEnable.Request()
        enable_request.enable = True
        return self.servo_move_enable_client.call_async(enable_request)

    def send_servo_request(self, pose_delta):
        """发送 servo_p 请求（增量控制）"""
        self.servo_req.pose = pose_delta
        return self.servo_p_client.call_async(self.servo_req)
    
    def send_servo_request_blocking(self, pose_delta, timeout_sec=10.0):
        """同步阻塞调用 servo_p（等待完成后才返回）"""
        self.servo_req.pose = pose_delta
        future = self.servo_p_client.call_async(self.servo_req)
    
    def call_get_ik(self, target_pose, ref_joints):
        """调用 get_ik 服务获取关节解
        
        Args:
            target_pose: [x, y, z, rx, ry, rz] (mm/rad)
            ref_joints: 参考关节角 [6 个值]
        
        Returns:
            Future 对象
        """
        from jaka_msgs.srv import GetIK
        request = GetIK.Request()
        request.cartesian_pose = target_pose
        request.ref_joint = list(ref_joints)
        return self.get_ik_client.call_async(request)
    
    def call_joint_move(self, joint_target, speed=5.0, acc=5.0):
        """调用 joint_move 服务移动到目标关节角
        
        Args:
            joint_target: 目标关节角 [6 个值]
            speed: 速度 (rad/s)
            acc: 加速度 (rad/s^2)
        
        Returns:
            Future 对象
        """
        from jaka_msgs.srv import Move
        request = Move.Request()
        request.pose = list(joint_target)
        request.has_ref = False
        request.ref_joint = [0.0]
        request.mvvelo = speed
        request.mvacc = acc
        request.mvtime = 0.0
        request.mvradii = 0.0
        request.coord_mode = 0
        request.index = 0
        return self.joint_move_client.call_async(request)

@dataclasses.dataclass
class Args:
    # 模型服务器参数
    host: str = "localhost"
    port: int = 8001
    resize_size: int = 448
    replan_steps: int = 16  # 每次规划的动作步数

    # 相机参数
    camera_devices: list = dataclasses.field(default_factory=lambda: [0, 2])  # 相机设备索引
    camera_width: int = 640
    camera_height: int = 480
    
    # 夹爪参数
    gripper_port: str = "/dev/ttyUSB0"  # 夹爪串口
    gripper_baud: int = 115200
    enable_gripper: bool = True  # 是否启用夹爪控制

    # JAKA机器人相关参数
    robot_ip: str = "192.168.10.90"  # JAKA机器人IP地址
    num_trials: int = 5  # 尝试次数
    max_steps: int = 300  # 每个试验的最大步数
    
    # 位置控制参数
    position_scale: float = 1.0 # 位置缩放系数
    orientation_scale: float = 1.0  # 姿态缩放系数
    
    # 工具参数
    video_out_path: str = "data/libero/videos_real"
    seed: int = 7
    use_euler_delta: bool = True  # 是否计算欧拉角增量
    euler_angle_range: float = math.pi  # 欧拉角范围 [-π, π]


class RealRobotDataCollector:
    """从真实JAKA机器人和相机收集数据"""
    
    def __init__(self, args: Args):
        self.args = args
        self.logger = logging.getLogger("RealRobotDataCollector")
        
    # 初始化相机
        self.cameras = {}
        self.latest_frames = {}
        self._init_cameras()
        self.count=0

        # 初始化夹爪
        self.gripper = None
        if args.enable_gripper:
            self._init_gripper()
        
        # 初始化 JAKA servo 客户端
        self.servo_client = JakaServoClientAsync()
        future = self.servo_client.enable_servo_mode()
        rclpy.spin_until_future_complete(self.servo_client, future)
        response = future.result()
        self.logger.info(f'✅ Servo mode enabled: {response.message if hasattr(response, "message") else "OK"}')
        # 当前状态
        self.current_eef_pos = np.array([0.0, 0.0, 0.0])  # m
        self.current_eef_euler = np.array([0.0, 0.0, 0.0])  # rad
        self.current_tool_eef_pose = np.array([0.0, 0.0, 0.0])  # m
        self.current_tool_eef_euler = np.array([0.0, 0.0, 0.0])  # rad
        self.tool_first_msg_received = False
        self.ref_joints = np.array([0.0] * 6)  # 参考关节角
        self.current_gripper_state = True  # False=关闭, True=打开
        
        self.prev_target_pos=None
        self.prev_target_euler=None

        self.logger.info("✅ RealRobotDataCollector 已初始化")
    
    def _init_cameras(self):
        """初始化相机"""
        for device_id in self.args.camera_devices:
            try:
                cap = cv2.VideoCapture(device_id)
                if not cap.isOpened():
                    self.logger.warning(f"⚠️ 无法打开相机设备 {device_id}")
                    continue
                
                # 设置分辨率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
                
                self.cameras[device_id] = cap
                self.latest_frames[device_id] = None
                self.logger.info(f"📷 相机 {device_id} 已初始化")
            except Exception as e:
                self.logger.error(f"❌ 初始化相机 {device_id} 失败: {e}")
    
    def _init_gripper(self):
        """初始化夹爪"""
        if GripperController is None:
            self.logger.warning("⚠️ GripperController 不可用")
            return
        
        try:
            self.gripper = GripperController(
                self.args.gripper_port,
                baud=self.args.gripper_baud,
                timeout=0.3
            )
            self.gripper.open()
            self.logger.info(f"✅ 夹爪已初始化 ({self.args.gripper_port})")
        except Exception as e:
            self.logger.error(f"❌ 初始化夹爪失败: {e}")
            self.gripper = None

    def get_robot_matrix(self, msg: TwistStamped):
        """
        更新当前机器人末端的位置和姿态
        """
        self.current_tool_pose = msg
        if self.current_tool_pose is None:
            self.logger.warning("尚未接收到机器人末端位姿数据")
            return

        # 提取位置 (单位: mm → m)
        px = self.current_tool_pose.twist.linear.x 
        py = self.current_tool_pose.twist.linear.y 
        pz = self.current_tool_pose.twist.linear.z

        # 提取姿态角度 (单位: 度 → 弧度)
        rx_rad = math.radians(self.current_tool_pose.twist.angular.x)
        ry_rad = math.radians(self.current_tool_pose.twist.angular.y)
        rz_rad = math.radians(self.current_tool_pose.twist.angular.z)

        # 更新当前状态
        if not self.tool_first_msg_received:
            self.tool_first_msg_received = True
            self.current_eef_pos = np.array([px, py, pz])  # m
            self.current_eef_euler = np.array([rx_rad, ry_rad, rz_rad])  # rad
        
        self.current_tool_eef_pose = np.array([px, py, pz]) 
        self.current_tool_eef_euler = np.array([rx_rad, ry_rad, rz_rad])
        self.logger.info(f"当前末端位置: {self.current_tool_eef_pose}")
        self.logger.info(f"当前末端姿态 (欧拉角): {self.current_tool_eef_euler}")

    def get_observation(self) -> Optional[Dict]:
        """获取当前观察数据
        
        返回:
            dict: 包含以下键值对:
                - "images": {device_id: preprocessed_image}
                - "eef_pos": numpy array [x, y, z] (m)
                - "eef_euler": numpy array [roll, pitch, yaw] (rad)
                - "gripper_state": bool (True=打开, False=关闭)
        """
        try:
            # 读取相机图像
            images = {}
            for device_id in self.args.camera_devices:
                if device_id not in self.cameras:
                    self.logger.warning(f"⚠️ 相机 {device_id} 不可用")
                    return None
                
                cap = self.cameras[device_id]
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    self.logger.warning(f"⚠️ 无法从相机 {device_id} 读取图像")
                    return None
                
                # 预处理图像
                img = self._preprocess_image(frame)
                images[device_id] = img
            
            return {
                "images": images,
                "eef_pos": self.current_eef_pos.copy(),
                "eef_euler": np.rad2deg(self.current_eef_euler.copy()),
                "gripper_state": self.current_gripper_state
            }
        except Exception as e:
            self.logger.error(f"❌ 获取观察数据失败: {e}")
            return None
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """预处理图像：旋转 180° + 缩放 + 填充"""
        # 旋转 180 度
        # img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 缩放并填充到目标尺寸
        img = cv2.resize(img, (self.args.resize_size, self.args.resize_size))
        
        # 转换为 uint8
        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        return img
    
    @staticmethod
    def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """保持宽高比地缩放图像并填充"""
        h, w = img.shape[:2]
        scale = min(target_h / h, target_w / w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # 填充到目标尺寸
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        bottom = target_h - new_h - top
        right = target_w - new_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded
    
    @staticmethod
    def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
        """欧拉角 [roll, pitch, yaw] 转四元数 [x, y, z, w]"""
        rx, ry, rz = euler
        
        cy = np.cos(rz * 0.5)
        sy = np.sin(rz * 0.5)
        cp = np.cos(ry * 0.5)
        sp = np.sin(ry * 0.5)
        cr = np.cos(rx * 0.5)
        sr = np.sin(rx * 0.5)
        
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy
        
        return np.array([x, y, z, w])
    
    def _normalize_euler_angle(self, angle: float) -> float:
        """将角度归一化到 [-π, π] 范围内"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    def _euler_to_matrix(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角 [roll, pitch, yaw] -> 旋转矩阵, 约定和 JAKA 一致: R = Rz(yaw) * Ry(pitch) * Rx(roll)"""
        rx, ry, rz = euler

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0],
                    [0, cx, -sx],
                    [0, sx,  cx]])
        Ry = np.array([[ cy, 0, sy],
                    [ 0,  1, 0],
                    [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                    [sz,  cz, 0],
                    [0,   0,  1]])

        R = Rz @ Ry @ Rx
        return R


    def _rotation_matrix_to_rpy(self, R: np.ndarray) -> np.ndarray:
        """旋转矩阵 -> [roll, pitch, yaw]，和你之前 servo 节点同一个公式"""
        # roll (X 轴)
        roll = math.atan2(R[2, 1], R[2, 2])
        # pitch (Y 轴)
        pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        # yaw (Z 轴)
        yaw = math.atan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw], dtype=float)

    def _compute_euler_delta(self, current_euler: np.ndarray, target_euler: np.ndarray) -> np.ndarray:
        """
        通过旋转矩阵在 SO(3) 里计算: current -> target 的相对旋转，
        然后再转回 [roll, pitch, yaw] 作为 servo_p 的增量。
        """
        # 1. 欧拉角 -> 旋转矩阵
        R_cur = self._euler_to_matrix(current_euler)
        R_tgt = self._euler_to_matrix(target_euler)

        # 2. 相对旋转: 先当前，后目标
        R_rel = R_tgt @ R_cur.T 
        self.logger.info(f"R_rel: \n{R_rel}")
        # 3. 相对旋转 -> rpy 增量
        delta_rpy = self._rotation_matrix_to_rpy(R_rel)

        # 4. 可选：再做一次归一化到 [-pi, pi]
        for i in range(3):
            delta_rpy[i] = self._normalize_euler_angle(delta_rpy[i])

        return delta_rpy


    def _subdivide_action(self, delta_pos: np.ndarray, delta_euler: np.ndarray, depth: int = 0) -> list:
        """递归细分动作，确保每一步都在安全范围内
        
        安全范围：
        - 位置增量: ≤ 8mm
        - 角度增量: ≤ 5°
        
        如果超过限制，对半分割并递归处理
        
        Args:
            delta_pos: 位置增量 (mm)
            delta_euler: 欧拉角增量 (rad)
            depth: 递归深度（防止无限递归）
        
        Returns:
            list: 细分后的 [(delta_pos, delta_euler), ...] 列表
        """
        MAX_POS_DELTA = 8.0  # mm
        MAX_ANGLE_DELTA = math.radians(5)  # rad
        MAX_DEPTH = 10  # 最大递归深度
        
        # 计算位置增量的大小
        pos_magnitude = np.linalg.norm(delta_pos)
        
        # 计算角度增量中的最大值
        angle_magnitude = np.max(np.abs(delta_euler))
        
        # 检查是否需要细分
        needs_subdivision = (pos_magnitude > MAX_POS_DELTA or 
                            angle_magnitude > MAX_ANGLE_DELTA or
                            depth > MAX_DEPTH)
        
        if not needs_subdivision:
            # 符合安全范围，返回单步
            return [(delta_pos, delta_euler)]
        
        if depth >= MAX_DEPTH:
            # 达到最大递归深度，强制返回（防止无限递归）
            # self.logger.warning(f"⚠️ 动作细分达到最大深度 {MAX_DEPTH}，强制返回")
            return [(delta_pos, delta_euler)]
        
        # 对半分割
        half_pos = delta_pos * 0.5
        half_euler = delta_euler * 0.5
        
        # self.logger.debug(f"📍 细分动作 (深度 {depth+1}): "
                        #  f"pos {pos_magnitude:.2f}mm→{np.linalg.norm(half_pos):.2f}mm, "
                        #  f"angle {math.degrees(angle_magnitude):.2f}°→{math.degrees(np.max(np.abs(half_euler))):.2f}°")
        
        # 递归处理两个半步
        steps1 = self._subdivide_action(half_pos, half_euler, depth + 1)
        steps2 = self._subdivide_action(half_pos, half_euler, depth + 1)
        
        return steps1 + steps2
    
    def _execute_subdivided_actions(self, subdivisions: list) -> bool:
        """执行细分后的动作列表
        
        Args:
            subdivisions: [(delta_pos, delta_euler), ...] 列表
        
        Returns:
            bool: 是否全部执行成功
        """
        for idx, (delta_pos, delta_euler) in enumerate(subdivisions):
            # 调用 servo_p（阻塞方式）
            pose_delta = list(delta_pos) + list(delta_euler)
            # pose_delta = list(delta_pos) + list([0.0, 0.0, 0.0])  # 只控制位置，保持姿态不变
            servo_result = self.servo_client.send_servo_request(pose_delta)
            rclpy.spin_until_future_complete(self.servo_client, servo_result)
            if servo_result is None:
                self.logger.error(f"❌ servo_p 超时 (步 {idx + 1}/{len(subdivisions)})")
                return False
            
        
        # self.logger.info(f"✅ 所有细分步完成")
        return True


    def execute_action(self, action: np.ndarray, is_first_action: bool = False) -> bool:
        """执行动作向量
        
        策略：
        1. 第一个action: 使用ROS2话题更新的真实位姿作为current
        2. 后续action: 假设前一个动作已完成，用前一个目标值作为current
        3. 计算动作的位置和角度增量
        4. 细分增量，确保每一步都在安全范围内（位置≤8mm，角度≤5°）
        5. 逐步执行细分后的动作
        
        Args:
            action: [x, y, z, rx, ry, rz, gripper]
                - 位置: mm（绝对位置，机器人坐标系）
                - 位姿: 度（绝对欧拉角，机器人坐标系）
                - 夹爪: -1.0 (无操作) / 0.0 (关闭) / 1.0 (打开)
            is_first_action: 是否为第一个action
        
        Returns:
            bool: 是否执行成功
        """
        try:
            # 提取动作分量
            target_pos = action[:3]  # mm (绝对位置)
            target_euler_deg = action[3:6]  # 度 (绝对欧拉角)
            target_euler = np.deg2rad(target_euler_deg)  # 转为弧度
            gripper_cmd = action[6]
            
            # ============ 更新当前位姿 ============
            if is_first_action:
                # 第一个action: 使用话题更新的真实位姿（已由回调函数更新到 self.current_eef_pos 和 self.current_eef_euler）
                self.logger.info("🔵 [第一个action] 使用话题位姿作为当前位姿")
                self.logger.info(f"current_eef_euler 639={self.current_eef_euler}")
                self.logger.info(f"target_euler 640={target_euler}")
                self.current_eef_pos = self.current_tool_eef_pose.copy()
                self.current_eef_euler = self.current_tool_eef_euler.copy()
            else:
                # 后续action: 假设前一个动作已完成，用前一个目标值作为当前位姿
                if self.prev_target_pos is not None and self.prev_target_euler is not None:
                    self.current_eef_pos = self.prev_target_pos.copy()
                    self.current_eef_euler = self.prev_target_euler.copy()
                    # self.logger.info("🔷 [后续action] 使用前一个目标值作为当前位姿")
                else:
                    self.logger.warning("⚠️ 前一个目标值不存在，使用话题位姿")
            
            self.logger.info(f"📍 目标: pos={target_pos} mm, euler={target_euler_deg}°, gripper={gripper_cmd:.1f}")
            self.logger.info(f"📍 当前: pos={self.current_eef_pos} mm, euler={np.rad2deg(self.current_eef_euler)}°")
            
            # ============ 位置增量计算 ============
            delta_pos = target_pos - self.current_eef_pos
            delta_pos = delta_pos * self.args.position_scale

            # ============ 姿态增量计算 ============
            delta_euler = self._compute_euler_delta(self.current_eef_euler, target_euler)
            delta_euler = delta_euler * self.args.orientation_scale
            self.logger.info(f"📐 增量: pos={np.linalg.norm(delta_pos):.2f}mm, euler={np.rad2deg(delta_euler)}°")
            
            # ============ 细分动作（统一处理所有 action）============
            pos_magnitude = np.linalg.norm(delta_pos)
            angle_magnitude = np.max(np.abs(delta_euler))
            
            # 检查是否需要细分
            if pos_magnitude > 8.0 or angle_magnitude > math.radians(5):
                subdivisions = self._subdivide_action(delta_pos, delta_euler)
                # self.logger.info(f"✂️ 细分为 {len(subdivisions)} 个小步")
                
                # 执行细分后的动作
                success = self._execute_subdivided_actions(subdivisions)
                if not success:
                    return False
            else:
                # 直接执行单步（阻塞方式）
                pose_delta = list(delta_pos) + list(delta_euler)  # 只控制位置，保持姿态不变
                # pose_delta = list(delta_pos) + list([0.0, 0.0, 0.0])  # 只控制位置，保持姿态不变
                servo_result = self.servo_client.send_servo_request(pose_delta)
                rclpy.spin_until_future_complete(self.servo_client, servo_result)
                if servo_result is None:
                    self.logger.error("❌ servo_p 超时")
                    return False
                
                # self.logger.info("✅ servo_p 完成")
            
            # 更新目标状态记录
            self.prev_target_pos = target_pos.copy()
            self.prev_target_euler = target_euler.copy()
            
            # ============ 夹爪控制 ============
            if self.gripper is not None:
                if gripper_cmd > 0.5:  # 打开
                    self.gripper.open()
                    self.current_gripper_state = True
                    self.logger.info("🤖 已打开夹爪")
                elif gripper_cmd < 0.5:  # 关闭
                    self.gripper.close()
                    self.current_gripper_state = False
                    self.logger.info("🤖 已关闭夹爪")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 执行动作异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def reset_target_states(self):
        """重置目标状态记录"""
        self.prev_target_pos = None
        self.prev_target_euler = None

    def get_joint_state(self, msg: JointState):
        """更新关节状态（用于 IK 参考）"""
        if len(msg.position) >= 6:
            self.ref_joints = np.array(msg.position[:6])

    def create_joint_state_subscription(self, node):
        """创建关节状态订阅（需要传入ROS2 Node）"""
        from sensor_msgs.msg import JointState
        self.joint_state_subscription = node.create_subscription(
            JointState, "/jaka_driver/joint_position", self.get_joint_state, 1
        )
        self.logger.info("✅ 已订阅 /jaka_driver/joint_position 话题")

    def create_tool_position_subscription(self, node):
        """创建工具位置订阅（需要传入ROS2 Node）"""
        self.tool_pos_subscription = node.create_subscription(
            TwistStamped, "/jaka_driver/tool_position", self.get_robot_matrix, 5
        )
        self.logger.info("✅ 已订阅 /jaka_driver/tool_position 话题")

    def cleanup(self):
        """清理资源"""
        for cap in self.cameras.values():
            if cap is not None:
                cap.release()
        if self.servo_client is not None:
            self.servo_client.destroy_node()
        self.logger.info("✅ 资源已清理")


class ClientPolicy:
    """WebSocket 客户端策略"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = f"ws://{host}:{port}"
        self.logger = logging.getLogger("WebsocketClientPolicy")
    
    def infer(self, request_data: dict) -> dict:
        """向模型服务器发送推理请求"""
        import json
        packer = Packer()
        client = WebsocketClientPolicy(self.host, self.port)
        try:
            # import websocket
            # ws = websocket.create_connection(self.url, timeout=10)
            # print(self.url)
            # 处理图像数据
            payload = {
                # "observation": {
                #     "images": {
                #         k: v.tolist() if isinstance(v, np.ndarray) else v
                #         for k, v in request_data.items()
                #         if k.startswith("observation.images")
                #     },
                #     "state": request_data.get("observation.state", [])
                # },
                "observation.images.camera_1": request_data["observation.images.image"],
                "observation.images.camera_0": request_data["observation.images.wrist_image"],
                "observation.state": request_data.get("observation.state", []),
                "task": request_data.get("task", ""),
                "n_action_steps": request_data.get("n_action_steps", 10),
                "is_ep_start": request_data.get("is_ep_start", False)
            }
            self.logger.info(f"payload state : {payload['observation.state']}")
            # ws.send(packer.pack(payload))
            # result = unpackb(ws.recv())
            # ws.close()
            result = client.infer(payload)
            print(result)
            return result
        except Exception as e:
            self.logger.error(f"❌ WebSocket 请求失败: {e}")
            raise


def eval_libero_real_robot(args: Args) -> None:
    """在真实JAKA机器人上运行 LIBERO 评估"""
    
    # 初始化 ROS2
    rclpy.init()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("eval_libero_real_robot")
    
    # 创建临时 Node 用于订阅
    temp_node = Node('eval_node_temp')
    
    # 创建数据收集器
    logger.info("🚀 初始化真实机器人数据收集器...")

    collector = RealRobotDataCollector(args)
    node = collector.servo_client
    # 创建工具位置和关节状态订阅
    collector.create_tool_position_subscription(node)
    collector.create_joint_state_subscription(node)
    
    # 等待初始位姿数据和关节状态（自旋10次以接收ROS2消息）
    logger.info("⏳ 等待机器人初始位姿数据和关节状态...")
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(0.1)
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    # 连接到模型服务器
    try:
        logger.info(f"🔌 连接到模型服务器: ws://{args.host}:{args.port}")
        client = ClientPolicy(args.host, args.port)
    except Exception as e:
        logger.error(f"❌ 无法初始化 WebSocket 客户端: {e}")
        collector.cleanup()
        return
    
    logger.info(f"✅ 连接成功")
    
    # 评估循环
    total_episodes = 0
    total_successes = 0
    
    for trial_idx in range(args.num_trials):
        logger.info(f"\n{'='*60}")
        logger.info(f"试验 {trial_idx + 1}/{args.num_trials}")
        logger.info(f"{'='*60}")
        
        # 获取初始观察
        obs = collector.get_observation()
        if obs is None:
            logger.error("❌ 无法获取初始观察数据，跳过此试验")
            continue
        
        logger.info("✅ 初始数据已就绪，开始执行")
        
        # 试验循环
        action_plan = collections.deque()
        replay_images = []
        step = 0
        done = False
        collector.reset_target_states()
        while True:
            try:
                # 处理 ROS2 回调（非阻塞）
                rclpy.spin_once(temp_node, timeout_sec=0.01)
                
                # 获取观察
                
                # time.sleep(3)  # 等待相机和机器人状态更新
                obs = collector.get_observation()
                if obs is None:
                    logger.warning("⚠️ 无法获取观察数据")
                    break

                
                # 保存图像用于视频
                first_device_id = args.camera_devices[0]
                first_image = obs["images"][first_device_id]
                replay_images.append(first_image)
                
                if not action_plan:
                    # 需要新的动作计划
                    second_device_id = args.camera_devices[1] if len(args.camera_devices) > 1 else first_device_id
                    
                    request_data = {
                        "observation.images.image": obs["images"][first_device_id],
                        "observation.images.wrist_image": obs["images"][second_device_id],
                        "observation.state": np.concatenate([
                            obs["eef_pos"],
                            obs["eef_euler"],
                            [1.0 if obs["gripper_state"] else 0.0]
                        ]),
                        "task": "stack the bowls",
                        # "task": "wipe off the water stains on the table",
                        "n_action_steps": args.replan_steps,
                        "is_ep_start": step == 0
                    }
                    
                    logger.info(f"🔄 步骤 {step}: 查询模型...")
                    action_chunk = client.infer(request_data)["actions"]
                    
                    if len(action_chunk) < args.replan_steps:
                        logger.warning(f"⚠️ 模型返回的动作数不足: {len(action_chunk)}")
                    
                    action_plan.extend(action_chunk[:args.replan_steps])
                
                

                # 执行动作
                if action_plan:
                    action = action_plan.popleft()
                    # if step % 4 != 0:
                    #     step+=1
                    #     continue 
                    # success = collector.execute_action(np.array(action))
                    is_first_action = (step == 0)
                    
                    success = collector.execute_action(np.array(action), is_first_action=is_first_action)
                    if not success:
                        logger.error("❌ 动作执行失败")
                        # 继续下一步，不完全中止
                
                step += 1
                # time.sleep(0.1)  # 给机器人时间响应
                
            except Exception as e:
                logger.error(f"❌ 步骤 {step} 异常: {e}")
                break
        
        total_episodes += 1
        
        # 保存视频
        if replay_images:
            suffix = "success" if done else "completed"
            video_path = pathlib.Path(args.video_out_path) / f"rollout_trial_{trial_idx}_{suffix}.mp4"
            
            if len(replay_images) > 0:
                h, w = replay_images[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
                
                for frame in replay_images:
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    if len(frame.shape) == 2:  # 灰度图
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    out.write(frame)
                out.release()
                logger.info(f"📹 视频已保存: {video_path}")
        
        logger.info(f"📊 试验 {trial_idx + 1} 完成, 共 {step} 步")
    
    # 清理
    collector.cleanup()
    temp_node.destroy_node()
    rclpy.shutdown()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 总体统计")
    logger.info(f"{'='*60}")
    logger.info(f"总试验数: {total_episodes}")
    logger.info(f"成功数: {total_successes}")


if __name__ == "__main__":
    tyro.cli(eval_libero_real_robot)
