from math import fabs
import re
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
                self.navigation_locker
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
        # if response:
        #     print(f"导航指令({x,y})发送成功，响应内容：")
        #     print(response)
        #     # 等待导航完成
        #     self._navigation_active = True
        #     try:
        #         self.navigation_locker()
        #     finally:
        #         self._navigation_active = False
        # else:
        #     print("任务发送失败")
        print(response)
        return 
    
    def go_to_target_LM(self, source_id, id):
        msg_data = {
            "source_id": source_id,
            "id": id,
            # "angle": 1.57,
            "method": "backward",
            "max_speed": 1,
            "max_wspeed": 1,
            "max_acc": 1,
            "max_wacc": 1,
            "duration": 100,
            "orientation": 90,
            "spin": True
        }
        response = self.send_message(3051, msg_data, socket_type=2)
        if response:
            print("导航指令发送成功，响应内容：")
            print(response)
        else:
            print("任务发送失败")

    def go_to_point_in_world_delta(self, delta_x, delta_y):
        msg_data = {"dist":np.sqrt(delta_x*delta_x+delta_y*delta_y),"vx":delta_x/np.sqrt(delta_x*delta_x+delta_y*delta_y),"vy":delta_y/np.sqrt(delta_x*delta_x+delta_y*delta_y)}
        response = self.send_message(3055, msg_data, socket_type=2)
        if response:
            print(f"导航指令({delta_x},{delta_y})发送成功，响应内容：")
            print(response)
            # 等待导航完成
            self._navigation_active = True
        #     try:
        #         self.navigation_locker()
        #     finally:
        #         self._navigation_active = False
        # else:
        #     print("任务发送失败")
        return 

    def cancel_navigation(self):
        response = self.send_message(3003, socket_type=2)
        if response:
            print("取消导航指令发送成功，响应内容：")
            print(response)
        else:
            print("任务发送失败")
        return 

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

        # for i in range(1):
        #     x, y, angle = agv.get_pose()
        #     print(f"agv在世界坐标系下的位置为:({x},{y}), 旋转角为:{angle}")

        #     delta_x = (x+0.08)
        #     delta_y = (y+0.035)
        #     agv.go_to_point_in_world_delta(delta_x, delta_y)
        #     time.sleep(1)


        # ==========导航==========
        # 回到地图0点，请确认0点位置安全后运行
        #agv.go_to_point_in_world(0,0,0)
        # 移动到固定流程的点位，请确认位置安全后运行
        #agv.go_to_point_in_world(-0.8328,-0.0176,3.1252)
        # 向前移动1m，请确认目标安全后运行
        #agv.go_to_point_in_robot(1,0,0)
        # agv.go_to_target_LM("LM1", "LM2")
        # agv.go_to_point_in_world(-0.080, -0.035, -0.0202458)
        # while 
        # agv.go_to_point_in_world_delta(0.0001, -0.0001)
if __name__ == '__main__':
    main()