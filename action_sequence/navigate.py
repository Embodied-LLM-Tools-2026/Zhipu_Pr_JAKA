import socket, json, ast
import requests
import time
import threading

class Navigate:
    def __init__(self):
        self.HOST, self.PORT = "192.168.10.10", 31001
        self.base_url = f"http://{self.HOST}:{self.PORT}"
        self.move_api = "/api/move?marker={target}"
        self.status_api = "/api/robot_status"
        self.ui_url =  "http://127.0.0.1:8000"
        self.pose_endpoint = f"{self.ui_url}/api/agv/pose/update"
        # ========== 全局状态变量（由监控线程定期更新）==========
        # 位置信息
        self.current_pose = {
            "theta": 0.0,
            "x": 0.0,
            "y": 0.0
        }
        
        # 导航状态
        self.navigation_state = {
            "move_status": None,      # "succeeded", "running", "failed" 等
            "is_navigating": False,   # 是否正在导航
            "last_target": None       # 最后的导航目标
        }
        
        # 完整的 status 响应（用于其他扩展字段）
        self.last_status_response = {}
        
        # 监控线程控制
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        self._socket = None           # 长连接 socket
        self.start_pose_monitoring(poll_interval=0.1)  # 启动状态监控线程

    def navigate_to_target(self, target):
        try:
            command = self.move_api.format(target=target)
            # Create a TCP/IP socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Connect to the server
                sock.connect((self.HOST, self.PORT))
                
                # Send the command
                sock.sendall(command.encode('utf-8'))
                
                # Receive the response
                response = sock.recv(4096).decode('utf-8')
                
                # Parse the JSON response
                response_json = json.loads(response)
                print(f"导航指令已发送: {command}")
                print("RESP:", response_json)
                time.sleep(4)
                return self.wait_until_navigation_complete(
                    poll_interval=0.1,
                    timeout=60
                )
        
        except Exception as e:
            print(f"导航指令发送失败: {e}")
            return False
    
    def move_to_position(self, theta, x, y, poll_interval=1.0, timeout=60):
        """
        移动到指定位置坐标并等待完成
        
        参数:
            theta: 方向角（弧度或度数，取决于AGV接口）
            x: X坐标（米）
            y: Y坐标（米）
            poll_interval: 状态检查间隔（秒）
            timeout: 超时时间（秒）
        
        返回:
            True: 成功到达目标位置
            False: 超时或失败
        """
        # 格式化位置参数：x,y,theta 顺序
        location_str = f"{x},{y},{theta}"
        
        try:
            # 发送移动指令
            command = f"/api/move?location={location_str}"
            print(f"🚀 发送移动指令: 位置({x}, {y}, {theta})")
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.HOST, self.PORT))
                sock.sendall(command.encode('utf-8'))
                
                response = sock.recv(4096).decode('utf-8')
                response_json = json.loads(response)
                
                print(f"✅ 移动指令已发送，AGV 响应: {response_json}")
                
        except Exception as e:
            print(f"❌ 移动指令发送失败: {e}")
            return False
        time.sleep(0.5)  # 确保指令被处理
        # 等待移动完成（使用全局 navigation_state 变量）
        return self.wait_until_navigation_complete(
            poll_interval=poll_interval,
            timeout=timeout
        )
        


    def wait_until_navigation_complete(self, poll_interval=1.0, timeout=60):
        """
        等待导航完成（通过轮询全局 navigation_state 变量）
        这比原来的方法更高效，因为使用了已有的后台监控数据
        """
        start_time = time.time()
        print(f"⏳ 等待导航完成（超时: {timeout}s）...")
        
        while time.time() - start_time < timeout:
            with self._lock:
                move_status = self.navigation_state.get("move_status")
                print(f"当前导航状态move_status: {move_status}")
            if move_status == "succeeded":
                print("✅ 导航成功完成！")
                return True
            elif move_status == "failed":
                print("❌ 导航失败！")
                return False
            elif move_status == "running":
                print("🔄 导航进行中...")
            elif move_status is None:
                print("⚠️ 导航状态未知，继续等待...")
            
            time.sleep(poll_interval)
        
        print("⏱️ 导航等待超时！")
        return False

    def start_pose_monitoring(self, poll_interval=0.1):
        """
        启动后台线程定期获取 AGV 状态信息（使用 socket 长连接）
        poll_interval: 轮询间隔（秒），默认0.1s
        """
        if self._monitoring:
            print("⚠️ AGV 状态监控已在运行")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._status_monitoring_loop,
            args=(poll_interval,),
            daemon=True
        )
        self._monitor_thread.start()
        print(f"✅ AGV 状态监控已启动 (轮询间隔: {poll_interval:.1f}s)")
    
    def stop_pose_monitoring(self):
        """停止状态监控并关闭 socket"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2)
            self._close_socket()
            print("✅ AGV 状态监控已停止")
    
    def _close_socket(self):
        """关闭 socket 连接"""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
    
    def _status_monitoring_loop(self, poll_interval):
        """
        后台循环：建立长连接并定期查询 AGV 状态
        一次连接，多次查询，解析并更新所有全局状态变量
        """
        max_retries = 3
        retry_count = 0
        
        while self._monitoring:
            try:
                # 如果连接不存在或已断开，建立新连接
                if self._socket is None:
                    print(f"🔌 尝试连接 AGV ({self.HOST}:{self.PORT})...")
                    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._socket.settimeout(3)
                    self._socket.connect((self.HOST, self.PORT))
                    print(f"✅ 已连接到 AGV")
                    retry_count = 0
                
                # 发送查询命令
                query_cmd = self.status_api + "\n"
                self._socket.sendall(query_cmd.encode('utf-8'))
                
                # 接收响应（使用 socket 接收）
                response = self._socket.recv(4096).decode('utf-8')
                
                if response:
                    try:
                        status_data = json.loads(response)
                        # print(f"📡 收到 AGV 状态响应:{status_data}")
                        self._update_global_state(status_data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON 解析失败: {e}")
                
            except socket.timeout:
                print(f"⚠️ Socket 超时")
                self._close_socket()
                retry_count += 1
                
            except ConnectionRefusedError:
                print(f"⚠️ AGV 连接被拒绝 ({self.HOST}:{self.PORT})")
                self._close_socket()
                retry_count += 1
                
            except Exception as e:
                print(f"⚠️ AGV 连接错误: {e}")
                self._close_socket()
                retry_count += 1
            
            # 重连退避
            if retry_count >= max_retries and self._monitoring:
                print(f"🔄 连接失败 {retry_count} 次，{poll_interval:.1f}s 后重试...")
                time.sleep(poll_interval * 2)
            elif self._monitoring:
                time.sleep(poll_interval)
    
    def _update_global_state(self, status_data):
        """
        解析 AGV status 响应，更新所有全局状态变量
        """
        with self._lock:
            # 保存完整响应
            self.last_status_response = status_data
            
            # 提取位置信息
            result = status_data.get('results', {})
            pose = status_data.get('results', {}).get('current_pose', None)
            
            if isinstance(pose, dict):
                data = pose

            # 2) 是序列 [theta, x, y]
            elif isinstance(pose, (list, tuple)) and len(pose) >= 3:
                data = {'theta': pose[0], 'x': pose[1], 'y': pose[2]}

            # 3) 是字符串，需要去掉前缀并解析
            elif isinstance(pose, str):
                s = pose.strip()
                if s.startswith('RESP:'):
                    s = s[len('RESP:'):].strip()
                # 先尝试 JSON，再回退到 Python 字面量（支持单引号）
                try:
                    data = json.loads(s)
                except json.JSONDecodeError:
                    # 单引号或非严格 JSON 的情况
                    data = ast.literal_eval(s)
            else:
                print(f"[update_pose] 不支持的 pose 类型: {type(pose).__name__}")
                return

            # 取值与类型校验
            try:
                theta = float(data['theta'])
                x = float(data['x'])
                y = float(data['y'])
            except (KeyError, TypeError, ValueError) as e:
                print(f"[update_pose] 解析失败：{e}; 原始 data={data!r}")
                return

            # 写入当前位姿（内部存弧度或度数都可以，这里保留原始单位以免混淆）
            self.current_pose["theta"] = theta
            self.current_pose["x"] = x
            self.current_pose["y"] = y
            try:
                resp = requests.post(
                    self.pose_endpoint,
                    json={
                        "theta": theta,
                        "x": x,
                        "y": y
                    },
                    timeout=2
                )
            except Exception as e:
                print(f"[pose推送失败] {e}")
            # 打印时按需要转成度数显示
            # print(f"📍 AGV 位置: θ={theta:.2f}°, x={x:.2f}m, y={y:.2f}m")
            
            # 提取导航状态
            nav_status = result.get('move_status', None)
            # print(nav_status)
            if nav_status is not None:
                self.navigation_state["move_status"] = nav_status
                self.navigation_state["is_navigating"] = (nav_status == "running")
                
                # 根据状态打印日志
                # status_symbol = "🔄" if nav_status == "running" else "✅" if nav_status == "succeeded" else "❌"
                # print(f"{status_symbol} 导航状态: {nav_status}")
    
    def get_current_pose(self):
        """
        获取当前存储的 AGV 位置信息
        返回: {"theta": float, "x": float, "y": float}
        """
        with self._lock:
            return self.current_pose.copy()
    
    def set_current_pose(self, theta, x, y):
        """
        手动设置 AGV 位置（用于初始化或外部更新）
        """
        with self._lock:
            self.current_pose["theta"] = float(theta)
            self.current_pose["x"] = float(x)
            self.current_pose["y"] = float(y)
    
    def is_navigating(self):
        """检查是否正在导航"""
        with self._lock:
            return self.navigation_state.get("is_navigating", False)
    
    def get_navigation_state(self):
        """获取导航状态信息"""
        with self._lock:
            return self.navigation_state.copy()
    
    def get_last_status_response(self):
        """获取最后一次 status 响应的完整内容（用于扩展字段）"""
        with self._lock:
            return self.last_status_response.copy()

    # 兼容原有接口
    def navigate_to_location(self, location):
        self.navigate_to_target(location)
        return self.wait_until_navigation_complete()


def main():
    navigate = Navigate()

if __name__ == '__main__':
    main()