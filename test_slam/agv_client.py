from math import fabs
import socket
import json
import time
import struct
import signal
import sys
import numpy as np

PACK_FMT_STR = '!BBHLH6s'

class AGVClient:
    """AGV客户端控制类"""
    
    def __init__(self, ip='192.168.192.5', timeout=5):
        # 设置信号处理器
        self._setup_signal_handler()
        """
        初始化AGV客户端
        
        Args:
            ip (str): AGV服务器IP地址
            port (int): AGV服务器端口
            timeout (int): 连接超时时间(秒)
        """
        self.ip = ip
        self.timeout = timeout
        self.socket_stater = None
        self.socket_controller = None
        self.socket_navigator = None
        self.connected = False
        self._navigation_active = False  # 标记导航是否正在进行
    
    def _setup_signal_handler(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            if self._navigation_active:
                print("\n检测到Ctrl+C，正在取消导航...")
                try:
                    self.cancel_navigation()
                except Exception as e:
                    print(f"取消导航时出错: {e}")
                finally:
                    print("程序退出")
                    sys.exit(0)
            else:
                print("\n程序退出")
                sys.exit(0)
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)
    
    def connect(self):
        """连接到AGV服务器"""
        try:
            self.socket_stater = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_stater.connect((self.ip, 19204))
            self.socket_stater.settimeout(self.timeout)

            self.socket_controller = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_controller.connect((self.ip, 19205))
            self.socket_controller.settimeout(self.timeout)

            self.socket_navigator = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_navigator.connect((self.ip, 19206))
            self.socket_navigator.settimeout(self.timeout)

            self.connected = True
            print(f"成功连接到AGV服务器 {self.ip}")
            return True

        except Exception as e:
            print(f"连接AGV服务器失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开与AGV服务器的连接"""
        if self.socket_stater:
            self.socket_stater.close()
            self.socket_stater = None
        if self.socket_controller:
            self.socket_controller.close()
            self.socket_controller = None
        if self.socket_navigator:
            self.socket_navigator.close()
            self.socket_navigator = None
        if self.socket_stater is None and self.socket_controller is None and self.socket_navigator is None:
            self.connected = False
            print("已断开AGV服务器连接")
    
    def _pack_message(self, req_id, msg_type, msg={}):
        """
        打包消息
        
        Args:
            req_id (int): 请求ID
            msg_type (int): 消息类型
            msg (dict): 消息内容
            
        Returns:
            bytes: 打包后的消息
        """
        msg_len = 0
        json_str = json.dumps(msg)
        if msg != {}:
            msg_len = len(json_str)
        
        raw_msg = struct.pack(PACK_FMT_STR, 0x5A, 0x01, req_id, msg_len, msg_type, b'\x00\x00\x00\x00\x00\x00')
        
        if msg != {}:
            raw_msg += bytearray(json_str, 'ascii')
        
        return raw_msg
    
    def _recv_and_unpack_response(self, data, socket_type):
        """
        解析响应数据
        
        Args:
            data (bytes): 响应数据
            
        Returns:
            tuple: (header, json_data)
        """
        if len(data) < 16:
            raise ValueError("响应数据包头部错误")
        
        header = struct.unpack(PACK_FMT_STR, data[:16])
        json_data_len = header[3]
        
        # 读取JSON数据
        json_data = b''
        remaining_len = json_data_len
        read_size = 1024
        
        while remaining_len > 0:
            if socket_type == 0:
                recv_data = self.socket_stater.recv(min(read_size, remaining_len))
            elif socket_type == 1:
                recv_data = self.socket_controller.recv(min(read_size, remaining_len))
            elif socket_type == 2:
                recv_data = self.socket_navigator.recv(min(read_size, remaining_len))
            else:
                raise ValueError("无效的socket类型")
            json_data += recv_data
            remaining_len -= len(recv_data)
        
        return header, json_data
    
    
    def send_message(self, msg_type, msg_data={}, req_id=1, socket_type=0):
        """
        发送自定义消息
        
        Args:
            msg_type (int): 消息类型
            msg_data (dict): 消息数据
            req_id (int): 请求ID
            socket_type (int): 0 stater读取状态, 1 controller控制, 2 navigator导航, 默认controller
            
        Returns:
            dict: 响应数据
        """
        if not self.connected:
            print("未连接到AGV服务器")
            return None
        
        try:
            packed_msg = self._pack_message(req_id, msg_type, msg_data)
            if socket_type == 0:
                print("发送状态获取消息", packed_msg)
                self.socket_stater.send(packed_msg)
                response_header = self.socket_stater.recv(16)
            elif socket_type == 1:
                print("发送控制消息", packed_msg)
                self.socket_controller.send(packed_msg)
                response_header = self.socket_controller.recv(16)
            elif socket_type == 2:
                print("发送导航消息", packed_msg)
                self.socket_navigator.send(packed_msg)
                response_header = self.socket_navigator.recv(16)
            else:
                raise ValueError("无效的socket类型")
            
            header, json_data = self._recv_and_unpack_response(response_header, socket_type)
            response_json = json.loads(json_data.decode('ascii'))
            return response_json
            
        except Exception as e:
            print(f"发送消息失败: {e}")
            return None
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
    
    def get_pose(self):
        """
        获取AGV当前在世界坐标系下的位置
        Returns:
            tuple: (x, y, angle)
        """
        response = self.send_message(1004,socket_type=0)
        if response:
            return response['x'], response['y'], response['angle']
        else:
            return None
    
    def get_velocity(self):
        """
        获取AGV当前速度
        Returns:
            tuple: (vx, vy, vw)
        """
        response = self.send_message(1005,socket_type=0)
        if response:
            return response['vx'], response['vy'], response['w']
        else:
            return None
    
    def get_blocked(self):
        """
        获取AGV当前块信息
        Returns:
            tuple: (blocked, block_x, block_y)
        """
        response = self.send_message(1006,socket_type=0)
        if response:
            return response['blocked'], response['block_x'], response['block_y']
        else:
            return None
    
    def get_pointcloud(self, return_beams3D=False):
        """
        获取激光雷达的点云
        """
        if return_beams3D:
            msg_data = {'return_beams3D': True}
            response = self.send_message(1009, msg_data, socket_type=0)
        else:
            response = self.send_message(1009, socket_type=0)
        if response:
            return response['lasers']
        else:
            return None
    
    def get_navigation_status(self,simple=True):
        """
        获取导航状态
        """
        if simple:
            msg_data = {'simple': True}
            response = self.send_message(1020, msg_data, socket_type=0)
        else:
            response = self.send_message(1020, socket_type=0)
        if response:
            if simple:
                return response['status']
            else:
                return response['status'], response['task_type'], response['target_point']
        else:
            return None
    
    def get_relocalization_status(self):
        response = self.send_message(1021, socket_type=0)
        if response:
            return response['reloc_status'] 
        else:
            return None

    def get_map_status(self):
        response = self.send_message(1022, socket_type=0)
        if response:
            return response['loadmap_status'] 
        else:
            return None
    
    def navigation_locker(self):
        while True:
            status = self.get_navigation_status()
            if status == 0:
                print("导航状态：agv未进行导航")
                break
            elif status == 2:
                print("导航状态：agv正在导航,按ctrl+c终止")
                time.sleep(1)  # 添加延时避免过于频繁的查询
            elif status == 3:
                print("导航状态：agv暂停导航")
                break
            elif status == 4:
                print("导航状态：agv完成导航")
                break
            elif status == 5:
                print("导航状态：agv导航失败")
                break
            elif status == 6:
                print("导航状态：agv取消导航")
                break
            else:
                print("导航状态：未知")
                break
        print("退出导航")

    def go_to_point_in_robot(self, x, y, theta):
        msg_data = {
            "script_name": "syspy/goPath.py",
            "script_args": {
                "x": x,
                "y": y,
                "theta": theta,
                "reachAngle": 0.001,
                "reachDist": 0.001,
                "coordinate": "robot"
            },
            "operation": "Script",
            "id": "SELF_POSITION",
            "source_id": "SELF_POSITION",
            "task_id": "12344321"
        }
        response = self.send_message(3051, msg_data, socket_type=2)
        if response:
            print(f"导航指令({x,y})发送成功，响应内容：")
            print(response)
            # 等待导航完成
            self._navigation_active = True
            try:
                self.navigation_locker()
            finally:
                self._navigation_active = False
        else:
            print("任务发送失败")
        return 
    
    def go_to_point_in_world(self, x, y, theta):
        msg_data = {
        "script_name": "syspy/goPath.py",
        "script_args": {
            "x": x,
            "y": y,
            "theta": theta,
            "reachAngle": 0.001,
            "reachDist": 0.001,
            "coordinate": "world"
        },
        "operation": "Script",
        "id": "SELF_POSITION",
        "source_id": "SELF_POSITION",
        "task_id": "12344321"
        }
        response = self.send_message(3051, msg_data, socket_type=2)
        if response:
            print(f"导航指令({x,y})发送成功，响应内容：")
            print(response)
            # 等待导航完成
            self._navigation_active = True
            try:
                self.navigation_locker()
            finally:
                self._navigation_active = False
        else:
            print("任务发送失败")
        return 
    
    def cancel_navigation(self):
        response = self.send_message(3003, socket_type=2)
        if response:
            print("取消导航指令发送成功，响应内容：")
            print(response)
        else:
            print("任务发送失败")
        return
    
    # ========== 建图控制方法 ==========
    def start_slam(self, slam_type=1, real_time=False, screen_width=None, screen_height=None):
        """
        开始扫描地图
        
        Args:
            slam_type (int): 扫图类型：1=2D扫图, 2=2D实时扫图, 3=3D扫图, 4=3D实时扫图
            real_time (bool): 是否开启实时扫图数据传输
            screen_width (int): 屏幕宽(像素)
            screen_height (int): 屏幕高(像素)
        
        Returns:
            bool: 是否成功
        """
        msg_data = {}
        if slam_type != 1:
            msg_data['slam_type'] = slam_type
        if real_time:
            msg_data['real_time'] = real_time
        if screen_width is not None:
            msg_data['screen_width'] = screen_width
        if screen_height is not None:
            msg_data['screen_height'] = screen_height
            
        response = self.send_message(6100, msg_data, socket_type=1)
        if response and response.get('ret_code') == 0:
            print("开始扫描地图指令发送成功")
            return True
        else:
            print(f"扫描地图指令发送失败: {response}")
            return False
    
    def stop_slam(self):
        """
        停止扫描地图
        
        Returns:
            bool: 是否成功
        """
        response = self.send_message(6101, socket_type=1)
        if response and response.get('ret_code') == 0:
            print("停止扫描地图指令发送成功")
            return True
        else:
            print(f"停止扫描地图指令发送失败: {response}")
            return False
    
    def get_slam_status(self, return_resultmap=False):
        """
        获取扫图状态
        
        Args:
            return_resultmap (bool): 是否返回构建完的地图
            
        Returns:
            int or tuple: 扫图状态 (0=没有扫图, 1=正在扫图(离线), 2=正在实时扫图, 3=正在3D扫图, 4=正在实时3D扫图)
                         如果return_resultmap=True，返回(status, resultmap)
        """
        msg_data = {}
        if return_resultmap:
            msg_data['return_resultmap'] = True
            
        response = self.send_message(1025, msg_data, socket_type=0)
        if response:
            if return_resultmap:
                return response.get('slam_status'), response.get('resultmap')
            else:
                return response.get('slam_status')
        else:
            return None
    
    def get_maps_info(self):
        """
        查询机器人载入的地图以及储存的地图
        
        Returns:
            dict: 包含current_map, maps等信息，失败时返回None
        """
        response = self.send_message(1300, socket_type=0)
        if response and response.get('ret_code') == 0:
            return {
                'current_map': response.get('current_map'),
                'current_map_md5': response.get('current_map_md5'),
                'maps': response.get('maps', []),
                'map_files_info': response.get('map_files_info', [])
            }
        else:
            print(f"获取地图信息失败: {response}")
            return None
    
    def upload_and_load_map(self, map_data, map_name=None):
        """
        上传并切换载入地图
        
        Args:
            map_data (str): 地图数据（JSON格式）
            map_name (str): 地图名称（可选）
            
        Returns:
            bool: 是否成功
        """
        # 这里需要特殊处理，因为地图数据很大，需要直接发送地图数据而不是JSON
        import json
        
        try:
            # 验证地图数据是否为有效JSON
            if isinstance(map_data, str):
                json.loads(map_data)  # 验证JSON格式
                map_bytes = map_data.encode('utf-8')
            else:
                print("地图数据必须是JSON字符串格式")
                return False
            
            # 直接发送地图数据
            if not self.connected:
                print("未连接到AGV服务器")
                return False
            
            # 构建请求头
            req_id = 1
            msg_type = 2025  # 上传并切换载入地图
            msg_len = len(map_bytes)
            
            import struct
            header = struct.pack(PACK_FMT_STR, 0x5A, 0x01, req_id, msg_len, msg_type, b'\x00\x00\x00\x00\x00\x00')
            
            # 发送头部和地图数据
            self.socket_controller.send(header + map_bytes)
            
            # 接收响应
            response_header = self.socket_controller.recv(16)
            header, json_data = self._recv_and_unpack_response(response_header, 1)
            response_json = json.loads(json_data.decode('ascii'))
            
            if response_json.get('ret_code') == 0:
                print(f"地图上传成功{'：' + map_name if map_name else ''}")
                return True
            else:
                print(f"地图上传失败: {response_json}")
                return False
                
        except Exception as e:
            print(f"上传地图时出现异常: {e}")
            return False
    
    # ========== 定位控制方法 ==========
    def relocalize(self, x=None, y=None, angle=None, length=None, is_auto=False, home=False):
        """
        重定位
        
        Args:
            x (float): 世界坐标系中的x坐标，单位m
            y (float): 世界坐标系中的y坐标，单位m
            angle (float): 世界坐标系中的角度，单位rad
            length (float): 重定位区域半径，单位m
            is_auto (bool): 是否为自动重定位
            home (bool): 在RobotHome重定位
            
        Returns:
            bool: 是否成功
        """
        msg_data = {}
        
        if is_auto:
            msg_data['isAuto'] = True
        elif home:
            msg_data['home'] = True
        else:
            if x is not None:
                msg_data['x'] = x
            if y is not None:
                msg_data['y'] = y
            if angle is not None:
                msg_data['angle'] = angle
            if length is not None:
                msg_data['length'] = length
                
        response = self.send_message(2002, msg_data, socket_type=1)
        if response and response.get('ret_code') == 0:
            print("重定位指令发送成功")
            return True
        else:
            print(f"重定位指令发送失败: {response}")
            return False
    
    # ========== 基础运动控制方法 ==========
    def translate(self, dist, vx=None, vy=None, mode=1):
        """
        平动控制
        
        Args:
            dist (float): 直线运动距离，绝对值，单位m
            vx (float): 机器人坐标系下X方向运动速度，正为向前，负为向后，单位m/s
            vy (float): 机器人坐标系下Y方向运动速度，正为向左，负为向右，单位m/s
            mode (int): 0=里程模式, 1=定位模式
            
        Returns:
            bool: 是否成功
        """
        msg_data = {'dist': dist}
        if vx is not None:
            msg_data['vx'] = vx
        if vy is not None:
            msg_data['vy'] = vy
        if mode != 0:
            msg_data['mode'] = mode
            
        response = self.send_message(3055, msg_data, socket_type=1)
        if response and response.get('ret_code') == 0:
            print(f"平动指令发送成功，距离：{dist}m")
            return True
        else:
            print(f"平动指令发送失败: {response}")
            return False
    
    def rotate(self, angle, vw, mode=1):
        """
        转动控制
        
        Args:
            angle (float): 转动的角度，绝对值，单位rad
            vw (float): 转动的角速度，正为逆时针转，负为顺时针转，单位rad/s
            mode (int): 0=里程模式, 1=定位模式
            
        Returns:
            bool: 是否成功
        """
        msg_data = {
            'angle': angle,
            'vw': vw
        }
        if mode != 0:
            msg_data['mode'] = mode
            
        response = self.send_message(3056, msg_data, socket_type=1)
        if response and response.get('ret_code') == 0:
            print(f"转动指令发送成功，角度：{angle}rad，角速度：{vw}rad/s")
            return True
        else:
            print(f"转动指令发送失败: {response}")
            return False
    
    def rotate_in_place(self, turns=1.0, angular_velocity=1.0, mode=1):
        """
        原地转圈
        
        Args:
            turns (float): 转圈数，正数为逆时针，负数为顺时针
            angular_velocity (float): 角速度，单位rad/s
            mode (int): 0=里程模式, 1=定位模式
            
        Returns:
            bool: 是否成功
        """
        import math
        angle = abs(turns) * 2 * math.pi
        vw = angular_velocity if turns > 0 else -angular_velocity
        
        return self.rotate(angle, vw, mode)
    
    def spin_tray(self, increase_angle=None, robot_angle=None, global_angle=None, direction=0):
        """
        托盘旋转控制
        
        Args:
            increase_angle (float): 在当前托盘角度基础上增加的角度，单位rad
                                   正数=逆时针，负数=顺时针
            robot_angle (float): 将托盘转到机器人坐标系下的目标角度，单位rad
            global_angle (float): 将托盘转到世界坐标系下的目标角度，单位rad
            direction (int): 旋转方向，0=就近，1=逆时针，-1=顺时针
                            仅在使用robot_angle或global_angle时有效
        
        Returns:
            bool: 是否成功
            
        Note:
            - 三种角度参数只能指定一种
            - 托盘旋转不能与平动(3055)、转动(3056)、其他托盘操作(3058)同时进行
        """
        msg_data = {}
        
        # 检查参数互斥性
        angle_params = [increase_angle, robot_angle, global_angle]
        non_none_params = [p for p in angle_params if p is not None]
        
        if len(non_none_params) == 0:
            print("托盘旋转控制需要指定至少一个角度参数")
            return False
        elif len(non_none_params) > 1:
            print("托盘旋转控制只能指定一种角度参数")
            return False
        
        # 设置消息数据
        if increase_angle is not None:
            msg_data['increase_spin_angle'] = increase_angle
            print(f"托盘增量旋转: {increase_angle}rad ({'逆时针' if increase_angle > 0 else '顺时针'})")
        elif robot_angle is not None:
            msg_data['robot_spin_angle'] = robot_angle
            if direction != 0:
                msg_data['spin_direction'] = direction
            direction_str = {0: '就近', 1: '逆时针', -1: '顺时针'}.get(direction, '未知')
            print(f"托盘转到机器人坐标系角度: {robot_angle}rad ({direction_str})")
        elif global_angle is not None:
            msg_data['global_spin_angle'] = global_angle
            if direction != 0:
                msg_data['spin_direction'] = direction
            direction_str = {0: '就近', 1: '逆时针', -1: '顺时针'}.get(direction, '未知')
            print(f"托盘转到世界坐标系角度: {global_angle}rad ({direction_str})")
        
        response = self.send_message(3057, msg_data, socket_type=1)
        if response and response.get('ret_code') == 0:
            print("托盘旋转指令发送成功")
            return True
        else:
            print(f"托盘旋转指令发送失败: {response}")
            return False
    
    # ========== 高级导航方法 ==========
    def navigate_path(self, move_task_list):
        """
        指定路径导航
        
        Args:
            move_task_list (list): 移动任务列表，每个任务包含id, source_id, task_id等
            
        Returns:
            bool: 是否成功
        """
        msg_data = {'move_task_list': move_task_list}
        response = self.send_message(3066, msg_data, socket_type=2)
        if response and response.get('ret_code') == 0:
            print("指定路径导航指令发送成功")
            return True
        else:
            print(f"指定路径导航指令发送失败: {response}")
            return False
    
    def get_path_between_points(self, source_id, target_id):
        """
        查询任意两点之间的路径
        
        Args:
            source_id (str): 起点ID
            target_id (str): 终点ID
            
        Returns:
            list: 路径站点列表
        """
        msg_data = {
            'source_id': source_id,
            'target_id': target_id
        }
        response = self.send_message(1303, msg_data, socket_type=0)
        if response and response.get('ret_code') == 0:
            return response.get('list', [])
        else:
            print(f"查询路径失败: {response}")
            return None
    
    # ========== 多种导航方法实现 ==========
    def navigate_to_point_method1(self, x, y, theta):
        """
        导航方法1：使用现有的世界坐标导航
        """
        return self.go_to_point_in_world(x, y, theta)
    
    def navigate_to_point_method2(self, x, y, theta):
        """
        导航方法2：使用机器人坐标导航
        """
        current_pose = self.get_pose()
        if current_pose is None:
            print("无法获取当前位置")
            return False
            
        current_x, current_y, current_angle = current_pose
        # 转换为机器人坐标系
        import math
        dx = x - current_x
        dy = y - current_y
        robot_x = dx * math.cos(-current_angle) - dy * math.sin(-current_angle)
        robot_y = dx * math.sin(-current_angle) + dy * math.cos(-current_angle)
        robot_theta = theta - current_angle
        
        return self.go_to_point_in_robot(robot_x, robot_y, robot_theta)
    
    def navigate_to_point_method3(self, x, y, theta, intermediate_points=None):
        """
        导航方法3：分段导航，先到中间点再到目标点
        """
        if intermediate_points is None:
            # 默认在中点设置一个中间点
            current_pose = self.get_pose()
            if current_pose is None:
                return False
            current_x, current_y, _ = current_pose
            mid_x = (current_x + x) / 2
            mid_y = (current_y + y) / 2
            intermediate_points = [(mid_x, mid_y, theta)]
        
        # 逐个导航到中间点
        for i, (ix, iy, itheta) in enumerate(intermediate_points):
            print(f"导航到中间点{i+1}: ({ix}, {iy}, {itheta})")
            if not self.go_to_point_in_world(ix, iy, itheta):
                print(f"导航到中间点{i+1}失败")
                return False
        
        # 最后导航到目标点
        print(f"导航到目标点: ({x}, {y}, {theta})")
        return self.go_to_point_in_world(x, y, theta) 

    def get_station_info(self):
        """
        获取地图中所有站点信息
        Returns:
            list: 站点信息列表，每个站点包含 id, type, x, y, r, desc
        """
        response = self.send_message(1301, socket_type=0)
        if response and response.get('ret_code') == 0:
            return response.get('stations', [])
        else:
            print(f"获取站点信息失败: {response}")
            return []

    def find_station_coordinates(self, station_id):
        """
        查询指定站点的坐标
        Args:
            station_id (str): 站点ID
        Returns:
            tuple: (x, y, r) 或 None
        """
        stations = self.get_station_info()
        for station in stations:
            if station['id'] == station_id:
                return station['x'], station['y'], station['r']
        print(f"未找到站点 {station_id}")
        return None

    def navigate_between_stations_method1(self, source_id, target_id):
        """
        第一种方法：基于go_to_target_LM的监控导航
        直接调用go_to_target_LM，并时刻查询导航状态来打印当前target_point
        
        Args:
            source_id (str): 起始站点ID
            target_id (str): 目标站点ID
        Returns:
            bool: 导航是否成功
        """
        print(f"========== 开始站点导航（方法1）：{source_id} -> {target_id} ==========")
        
        # 调用原有的go_to_target_LM方法
        msg_data = {
            "source_id": source_id,
            "id": target_id,
            "method": "backward",
            "max_speed": 0.2,
            "max_wspeed": 0.2,
            "max_acc": 0.1,
            "max_wacc": 0.1,
            "duration": 100,
            # "orientation": 90,
            "spin": True
        }
        
        response = self.send_message(3051, msg_data, socket_type=2)
        if not response:
            print("导航指令发送失败")
            return False
        
        print(f"导航指令发送成功，开始监控导航状态...")
        print(f"响应内容: {response}")
        
        # 开始监控导航状态和目标点
        self._navigation_active = True
        try:
            while True:
                # 获取详细导航状态（包含target_point）
                status_result = self.get_navigation_status(simple=False)
                
                if status_result is None:
                    print("无法获取导航状态")
                    break
                
                status, task_type, target_point = status_result
                
                # 打印当前导航状态和目标点
                print(f"导航状态: {status}, 任务类型: {task_type}")
                if target_point:
                    print(f"当前目标点坐标: x={target_point[0]:.4f}, y={target_point[1]:.4f}, θ={target_point[2]:.4f}")
                
                # 判断导航状态
                if status == 0:
                    print("导航状态：AGV未进行导航")
                    break
                elif status == 2:
                    print("导航状态：AGV正在导航中...")
                elif status == 3:
                    print("导航状态：AGV暂停导航")
                    break
                elif status == 4:
                    print("导航状态：AGV完成导航")
                    print(f"✅ 成功到达目标站点 {target_id}")
                    return True
                elif status == 5:
                    print("导航状态：AGV导航失败")
                    return False
                elif status == 6:
                    print("导航状态：AGV取消导航")
                    return False
                else:
                    print(f"导航状态：未知状态 {status}")
                    break
                
                time.sleep(1)  # 每秒查询一次状态
                
        except KeyboardInterrupt:
            print("\n用户中断导航")
            self.cancel_navigation()
            return False
        finally:
            self._navigation_active = False
        
        print("导航监控结束")
        return False

    def navigate_between_stations_method2(self, source_id, target_id):
        """
        第二种方法：分步导航
        先查询目标站点坐标，然后先直线运动到位置，再旋转到角度
        
        Args:
            source_id (str): 起始站点ID  
            target_id (str): 目标站点ID
        Returns:
            bool: 导航是否成功
        """
        print(f"========== 开始站点导航（方法2）：{source_id} -> {target_id} ==========")
        
        # 1. 查询目标站点坐标
        print(f"正在查询目标站点 {target_id} 的坐标...")
        target_coords = self.find_station_coordinates(target_id)
        if target_coords is None:
            print(f"❌ 无法找到目标站点 {target_id} 的坐标")
            return False
        
        target_x, target_y, target_r = target_coords
        print(f"目标站点坐标: x={target_x:.4f}, y={target_y:.4f}, r={target_r:.4f}")
        
        # 2. 获取当前位置
        current_pose = self.get_pose()
        if current_pose is None:
            print("❌ 无法获取当前位置")
            return False
        
        current_x, current_y, current_angle = current_pose
        print(f"当前位置: x={current_x:.4f}, y={current_y:.4f}, angle={current_angle:.4f}")
        
        # 3. 计算移动距离和角度差异
        distance = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        angle_diff = abs(target_r - current_angle)
        
        print(f"需要移动距离: {distance:.4f}m, 角度调整: {angle_diff:.4f}rad")
        
        try:
            # 4. 第一步：先移动到目标位置（只到达x,y，暂时不考虑角度）
            print(f"步骤1: 移动到目标位置 ({target_x:.4f}, {target_y:.4f})")
            
            # 使用世界坐标导航，但保持当前角度
            self.go_to_point_in_world(target_x, target_y, current_angle)
            
            # 等待第一步移动完成
            print("等待位置移动完成...")
            move_success = self._wait_for_navigation_complete()
            
            if not move_success:
                print("❌ 移动到目标位置失败")
                return False
            
            print("✅ 成功移动到目标位置")
            
            # 5. 第二步：旋转到目标角度
            if angle_diff > 0.01:  # 只有角度差异大于0.01弧度才进行旋转
                print(f"步骤2: 旋转到目标角度 {target_r:.4f}rad")
                
                # 计算需要旋转的角度
                rotation_angle = target_r - current_angle
                # 规范化角度到[-π, π]
                while rotation_angle > np.pi:
                    rotation_angle -= 2 * np.pi
                while rotation_angle < -np.pi:
                    rotation_angle += 2 * np.pi
                
                print(f"需要旋转角度: {rotation_angle:.4f}rad")
                
                # 使用delta移动进行最终的精确旋转定位
                final_success = self.go_to_point_in_world(target_x, target_y, target_r)
                
                print("等待角度调整完成...")
                rotation_success = self._wait_for_navigation_complete()
                
                if not rotation_success:
                    print("❌ 角度调整失败")
                    return False
                
                print("✅ 成功调整到目标角度")
            else:
                print("角度差异很小，无需调整")
            
            # 6. 验证最终位置
            final_pose = self.get_pose()
            if final_pose:
                final_x, final_y, final_angle = final_pose
                position_error = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
                angle_error = abs(final_angle - target_r)
                
                print(f"最终位置: x={final_x:.4f}, y={final_y:.4f}, angle={final_angle:.4f}")
                print(f"位置误差: {position_error:.4f}m, 角度误差: {angle_error:.4f}rad")
                
                # 判断是否到达目标精度
                if position_error < 0.1 and angle_error < 0.1:
                    print(f"✅ 成功到达目标站点 {target_id}（分步导航）")
                    return True
                else:
                    print(f"⚠️  到达目标但精度不够，位置误差: {position_error:.4f}m, 角度误差: {angle_error:.4f}rad")
                    return True  # 仍然算作成功，但精度不够
            
            return True
            
        except KeyboardInterrupt:
            print("\n用户中断导航")
            self.cancel_navigation()
            return False
        except Exception as e:
            print(f"❌ 分步导航过程中出现异常: {e}")
            return False

    def _wait_for_navigation_complete(self, timeout=60):
        """
        等待导航完成的辅助方法
        Args:
            timeout (int): 超时时间（秒）
        Returns:
            bool: 导航是否成功完成
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_navigation_status(simple=True)
            
            if status == 0:  # 未进行导航
                return True
            elif status == 4:  # 完成导航
                return True
            elif status == 5:  # 导航失败
                print("导航失败")
                return False
            elif status == 6:  # 取消导航
                print("导航被取消")
                return False
            elif status == 2:  # 正在导航
                time.sleep(0.5)
                continue
            else:
                time.sleep(0.5)
                continue
        
        print("导航超时")
        return False 

def test_station_navigation():
    """测试两种站点间导航方法"""
    print("AGV站点导航方法测试")
    print("=" * 50)
    print("1. 基于go_to_target_LM的监控导航")
    print("2. 分步导航（查询坐标+分步移动）")
    print("3. 查看所有站点信息")
    print("4. 对比两种方法")
    print("0. 退出")
    
    try:
        choice = input("\n请选择测试方法 (0-4): ").strip()
        
        # 创建AGV客户端
        agv = AGVClient(ip='192.168.192.5')
        if not agv.connect():
            print("连接AGV失败")
            return
        
        try:
            if choice == '1':
                print("\n========== 测试方法1：基于go_to_target_LM的监控导航 ==========")
                source_id = input("请输入起始站点ID (如 LM1): ").strip() or "LM1"
                target_id = input("请输入目标站点ID (如 LM2): ").strip() or "LM2"
                
                success = agv.navigate_between_stations_method1(source_id, target_id)
                print(f"方法1导航结果: {'✅ 成功' if success else '❌ 失败'}")
                
            elif choice == '2':
                print("\n========== 测试方法2：分步导航 ==========")
                source_id = input("请输入起始站点ID (如 LM1): ").strip() or "LM1"
                target_id = input("请输入目标站点ID (如 LM2): ").strip() or "LM2"
                
                success = agv.navigate_between_stations_method2(source_id, target_id)
                print(f"方法2导航结果: {'✅ 成功' if success else '❌ 失败'}")
                
            elif choice == '3':
                print("\n========== 查看所有站点信息 ==========")
                stations = agv.get_station_info()
                if stations:
                    print(f"共找到 {len(stations)} 个站点:")
                    for station in stations:
                        print(f"  ID: {station['id']:<10} 类型: {station['type']:<15} "
                              f"坐标: ({station['x']:.3f}, {station['y']:.3f}, {station['r']:.3f})")
                else:
                    print("未找到任何站点信息")
                    
            elif choice == '4':
                print("\n========== 对比两种导航方法 ==========")
                source_id = input("请输入起始站点ID (如 LM1): ").strip() or "LM1"
                target_id = input("请输入目标站点ID (如 LM2): ").strip() or "LM2"
                
                print("\n--- 开始方法1测试 ---")
                start_time = time.time()
                success1 = agv.navigate_between_stations_method1(source_id, target_id)
                time1 = time.time() - start_time
                
                time.sleep(2)  # 短暂等待
                
                print("\n--- 开始方法2测试 ---")
                start_time = time.time()
                success2 = agv.navigate_between_stations_method2(source_id, target_id)
                time2 = time.time() - start_time
                
                # 结果对比
                print(f"\n========== 导航方法对比结果 ==========")
                print(f"方法1（监控导航）: {'✅ 成功' if success1 else '❌ 失败'}, 耗时: {time1:.1f}秒")
                print(f"方法2（分步导航）: {'✅ 成功' if success2 else '❌ 失败'}, 耗时: {time2:.1f}秒")
                
            elif choice == '0':
                print("退出测试")
                
            else:
                print("无效选择")
                
        finally:
            agv.disconnect()
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试异常: {e}")

def main():
    # 使用新的控制类
    with AGVClient(ip='192.168.192.5') as agv:
        # ==========获取当前机器人的建图与定位状态==========
        map_status = agv.get_map_status()
        if map_status == 0:
            print("建图状态：agv未载入地图")
        elif map_status == 1:
            print("建图状态：agv已载入地图")
        elif map_status == 2:
            print("建图状态：agv正在载入地图")
        
        localization_status = agv.get_relocalization_status()
        if localization_status == 0:
            print("定位状态：agv未进行定位")
        elif localization_status == 1:
            print("定位状态：agv已进行定位")
        elif localization_status == 2:
            print("定位状态：agv正在定位")

        # ==========读取位置信息==========
        pose_result = agv.get_pose()
        if pose_result:
            x, y, angle = pose_result
            print(f"agv在世界坐标系下的位置为:({x},{y}), 旋转角为:{angle}")

        # ==========导航==========
        # 回到地图0点，请确认0点位置安全后运行
        #agv.go_to_point_in_world(0,0,0)
        # 移动到固定流程的点位，请确认位置安全后运行
        #agv.go_to_point_in_world(-0.8328,-0.0176,3.1252)
        # 向前移动1m，请确认目标安全后运行
        #agv.go_to_point_in_robot(1,0,0)

if __name__ == '__main__':
    main()