"""
AGV 2D建图、定位、导航完整工作流程管理类
"""
import time
import math
from agv_client import AGVClient


class AGVWorkflow:
    """AGV完整工作流程管理类"""
    
    def __init__(self, ip='192.168.192.5', timeout=5):
        """
        初始化工作流程管理器
        
        Args:
            ip (str): AGV服务器IP地址
            timeout (int): 连接超时时间(秒)
        """
        self.agv = AGVClient(ip, timeout)
        self.slam_points = []  # 建图时要访问的点
        self.navigation_methods = {}  # 存储不同的导航方法
        self._setup_navigation_methods()
    
    def _setup_navigation_methods(self):
        """设置不同的导航方法供测试"""
        self.navigation_methods = {
            'method1_world_coordinate': self.agv.navigate_to_point_method1,
            'method2_robot_coordinate': self.agv.navigate_to_point_method2,
            'method3_segmented': self.agv.navigate_to_point_method3,
            'method4_path_navigation': self._navigate_using_path,
            'method5_translate_rotate': self._navigate_using_translate_rotate
        }
    
    def _navigate_using_path(self, x, y, theta):
        """方法4：使用路径导航（需要预定义站点）"""
        # 这需要预先设置的站点，这里作为示例
        print("路径导航方法需要预定义的站点ID")
        return False
    
    def _navigate_using_translate_rotate(self, x, y, theta):
        """方法5：使用基础的平动和转动组合导航"""
        current_pose = self.agv.get_pose()
        if current_pose is None:
            print("无法获取当前位置")
            return False
        
        current_x, current_y, current_angle = current_pose
        
        # 计算距离和角度
        dx = x - current_x
        dy = y - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # 先转向目标方向
        angle_diff = target_angle - current_angle
        # 规范化角度到[-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        if abs(angle_diff) > 0.1:  # 如果角度差大于0.1弧度
            print(f"转向目标方向，角度差：{angle_diff}")
            if not self.agv.rotate(abs(angle_diff), 1.0 if angle_diff > 0 else -1.0):
                return False
            time.sleep(2)  # 等待转动完成
        
        # 平移到目标位置
        if distance > 0.1:  # 如果距离大于0.1米
            print(f"平移到目标位置，距离：{distance}")
            if not self.agv.translate(distance, 0.5):
                return False
            time.sleep(distance/0.5 + 1)  # 等待平移完成
        
        # 最后转到目标角度
        final_angle_diff = theta - target_angle
        while final_angle_diff > math.pi:
            final_angle_diff -= 2 * math.pi
        while final_angle_diff < -math.pi:
            final_angle_diff += 2 * math.pi
            
        if abs(final_angle_diff) > 0.1:
            print(f"转到目标角度，角度差：{final_angle_diff}")
            if not self.agv.rotate(abs(final_angle_diff), 1.0 if final_angle_diff > 0 else -1.0):
                return False
        
        return True
    
    def connect(self):
        """连接到AGV"""
        return self.agv.connect()
    
    def disconnect(self):
        """断开连接"""
        self.agv.disconnect()
    
    def wait_for_status(self, get_status_func, target_status, timeout=30, check_interval=1):
        """
        等待状态达到目标值
        
        Args:
            get_status_func: 获取状态的函数
            target_status: 目标状态值或状态值列表
            timeout: 超时时间
            check_interval: 检查间隔
            
        Returns:
            bool: 是否成功达到目标状态
        """
        start_time = time.time()
        target_statuses = target_status if isinstance(target_status, list) else [target_status]
        
        while time.time() - start_time < timeout:
            current_status = get_status_func()
            if current_status in target_statuses:
                return True
            print(f"当前状态：{current_status}，等待目标状态：{target_statuses}")
            time.sleep(check_interval)
        
        print(f"等待状态超时，当前状态：{get_status_func()}")
        return False
    
    def stage1_mapping(self, mapping_points=None, turns_per_point=1.0, angular_velocity=1.0, auto_mapping=False, turn_direction='left', forward_distance=1.0, side_distance=0.5):
        """
        第一阶段：2D建图
        
        Args:
            mapping_points (list): 建图要访问的点列表 [(x1,y1,theta1), ...]，当auto_mapping=True时可为None
            turns_per_point (float): 每个点转圈数
            angular_velocity (float): 转动角速度
            auto_mapping (bool): 是否使用自动路径规划建图（类似扫地机器人）
            turn_direction (str): 自动建图时的转向方向 'left' 或 'right'
            forward_distance (float): 自动建图时的前进距离
            side_distance (float): 自动建图时的侧移距离
            
        Returns:
            bool: 建图是否成功
        """
        print("========== 开始第一阶段：2D建图 ==========")
        
        # 1. 开始建图
        if not self.agv.start_slam(slam_type=1):
            print("启动建图失败")
            return False
        
        # 2. 等待建图状态变为正在建图
        if not self.wait_for_status(self.agv.get_slam_status, [1, 2], timeout=10):
            print("建图状态未正常启动")
            return False
        
        # 3. 选择建图模式
        if auto_mapping:
            print("建图已启动，开始自动路径规划建图...")
            success = self._auto_mapping_sweep(turn_direction, forward_distance, side_distance)
        else:
            if mapping_points is None:
                print("手动建图模式需要提供建图点列表")
                return False
            print("建图已启动，开始按指定路径建图...")
            success = self._manual_mapping_points(mapping_points, turns_per_point, angular_velocity)
        
        if not success:
            print("建图过程失败")
            return False
        
        # 4. 停止建图
        print("\n建图路径完成，停止建图...")
        if not self.agv.stop_slam():
            print("停止建图失败")
            return False
        
        # 5. 轮询扫图状态直到返回非空地图数据
        print("等待建图数据生成...")
        new_map_data = self._wait_for_map_data(timeout=60)
        if new_map_data is None:
            print("获取建图数据失败")
            return False
        
        print(f"成功获取建图数据，大小：{len(new_map_data)} 字符")
        
        # 6. 检查当前载入的地图并上传新地图
        if not self._ensure_new_map_loaded(new_map_data):
            print("地图载入验证失败")
            return False
        
        print("========== 第一阶段：2D建图完成 ==========")
        return True
    
    def _wait_for_map_data(self, timeout=60, check_interval=2):
        """
        轮询扫图状态直到返回非空的地图数据
        
        Args:
            timeout (int): 超时时间
            check_interval (int): 检查间隔
            
        Returns:
            str: 地图数据，失败时返回None
        """
        print("开始轮询扫图状态，等待地图数据...")
        start_time = time.time()
        poll_count = 0
        
        while time.time() - start_time < timeout:
            poll_count += 1
            print(f"第 {poll_count} 次轮询扫图状态...")
            
            # 查询扫图状态并请求返回地图数据
            result = self.agv.get_slam_status(return_resultmap=True)
            if result is None:
                print("查询扫图状态失败")
                time.sleep(check_interval)
                continue
            
            slam_status, resultmap = result
            print(f"扫图状态：{slam_status}")
            
            # 检查是否有地图数据
            if resultmap and resultmap.strip():
                print("✅ 成功获取地图数据")
                return resultmap
            else:
                print("地图数据为空，继续轮询...")
                time.sleep(check_interval)
        
        print(f"轮询超时（{timeout}秒），未能获取地图数据")
        return None
    
    def _ensure_new_map_loaded(self, new_map_data):
        """
        确保新建图的地图已经载入
        
        Args:
            new_map_data (str): 新生成的地图数据
            
        Returns:
            bool: 是否成功
        """
        print("检查当前载入的地图...")
        
        # 1. 获取当前地图信息
        maps_info = self.agv.get_maps_info()
        if maps_info is None:
            print("无法获取当前地图信息")
            return False
        
        current_map = maps_info.get('current_map')
        current_map_md5 = maps_info.get('current_map_md5')
        
        print(f"当前载入地图：{current_map}")
        print(f"当前地图MD5：{current_map_md5}")
        
        # 2. 计算新地图的MD5
        import hashlib
        new_map_md5 = hashlib.md5(new_map_data.encode('utf-8')).hexdigest()
        print(f"新建图MD5：{new_map_md5}")
        
        # 3. 检查是否是同一个地图
        if current_map_md5 == new_map_md5:
            print("✅ 当前载入的地图就是刚才扫描的地图")
            return True
        
        # 4. 不是同一个地图，需要上传并载入新地图
        print("当前载入的地图不是刚才扫描的地图，开始上传新地图...")
        
        # 生成新地图名称
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_map_name = f"slam_map_{timestamp}"
        
        # 上传并载入新地图
        if not self.agv.upload_and_load_map(new_map_data, new_map_name):
            print("上传新地图失败")
            return False
        
        # 5. 等待地图载入完成
        print("等待新地图载入...")
        time.sleep(5)
        
        # 6. 验证地图是否载入成功
        maps_info_after = self.agv.get_maps_info()
        if maps_info_after is None:
            print("无法获取载入后的地图信息")
            return False
        
        new_current_map_md5 = maps_info_after.get('current_map_md5')
        if new_current_map_md5 == new_map_md5:
            print("✅ 新地图载入成功")
            return True
        else:
            print(f"⚠️ 地图载入验证失败，期望MD5：{new_map_md5}，实际MD5：{new_current_map_md5}")
            return False
    
    def _manual_mapping_points(self, mapping_points, turns_per_point, angular_velocity):
        """
        手动指定点的建图方法（原有方法）
        """
        # 按顺序移动到各个建图点
        for i, (x, y, theta) in enumerate(mapping_points):
            print(f"\n--- 移动到建图点 {i+1}/{len(mapping_points)}: ({x}, {y}, {theta}) ---")
            
            # 移动到目标点
            if not self.agv.go_to_point_in_world(x, y, theta):
                print(f"移动到建图点 {i+1} 失败")
                continue
            
            # 等待导航完成
            if not self.wait_for_status(self.agv.get_navigation_status, [0, 4], timeout=60):
                print(f"导航到建图点 {i+1} 超时")
                continue
            
            print(f"已到达建图点 {i+1}，开始原地转圈...")
            
            # 原地转圈进行建图
            if not self.agv.rotate_in_place(turns_per_point, angular_velocity):
                print(f"在建图点 {i+1} 转圈失败")
                continue
            
            # 等待转圈完成
            rotation_time = abs(turns_per_point) * 2 * math.pi / angular_velocity
            time.sleep(rotation_time + 2)
            
            print(f"建图点 {i+1} 完成")
        
        return True
    
    def _auto_mapping_sweep(self, turn_direction='left', forward_distance=1.0, side_distance=0.5):
        """
        自动路径规划建图方法（类似扫地机器人）
        
        Args:
            turn_direction (str): 转向方向 'left' 或 'right'
            forward_distance (float): 前进距离
            side_distance (float): 侧移距离
            
        Returns:
            bool: 建图是否成功
        """
        print(f"开始自动路径规划建图，转向方向：{turn_direction}")
        
        # 设置转向角度（左转90度或右转90度）
        turn_angle = math.pi / 2 if turn_direction == 'left' else -math.pi / 2
        turn_angular_velocity = 0.5  # 转向角速度
        forward_velocity = 0.3       # 前进速度
        
        # 建图循环计数器
        sweep_count = 0
        max_sweeps = 50  # 最大扫描次数，防止无限循环
        
        while sweep_count < max_sweeps:
            print(f"\n--- 第 {sweep_count + 1} 次扫描循环 ---")
            
            # 1. 直行阶段：向前移动直到遇到障碍
            print("阶段1：向前直行...")
            forward_success = self._move_forward_until_blocked(forward_distance, forward_velocity)
            
            if not forward_success:
                print("前进过程中出现错误")
                break
            
            # 2. 转向阶段：转90度
            print(f"阶段2：{'左转' if turn_direction == 'left' else '右转'}90度...")
            if not self.agv.rotate(abs(turn_angle), turn_angular_velocity if turn_angle > 0 else -turn_angular_velocity):
                print("转向失败")
                break
            
            # 等待转向完成
            turn_time = abs(turn_angle) / abs(turn_angular_velocity)
            time.sleep(turn_time + 1)
            
            # 3. 侧移阶段：移动一小段距离
            print("阶段3：侧向移动...")
            side_blocked = self._move_side_distance(side_distance, forward_velocity)
            
            if side_blocked:
                print("侧移时检测到阻挡，建图完成")
                break
            
            # 4. 再次转向：再转90度，恢复直行方向
            print(f"阶段4：再次{'左转' if turn_direction == 'left' else '右转'}90度，恢复直行...")
            if not self.agv.rotate(abs(turn_angle), turn_angular_velocity if turn_angle > 0 else -turn_angular_velocity):
                print("第二次转向失败")
                break
            
            # 等待转向完成
            time.sleep(turn_time + 1)
            
            sweep_count += 1
            print(f"第 {sweep_count} 次扫描循环完成")
        
        if sweep_count >= max_sweeps:
            print(f"达到最大扫描次数 {max_sweeps}，自动结束建图")
        
        print(f"自动路径规划建图完成，共执行 {sweep_count} 次扫描")
        return True
    
    def _move_forward_until_blocked(self, max_distance, velocity):
        """
        向前移动直到遇到障碍或达到最大距离
        
        Args:
            max_distance (float): 最大移动距离
            velocity (float): 移动速度
            
        Returns:
            bool: 是否成功完成移动
        """
        moved_distance = 0.0
        step_distance = 0.2  # 每次移动0.2米
        
        while moved_distance < max_distance:
            # 检查前方是否被阻挡
            blocked_info = self.agv.get_blocked()
            if blocked_info and blocked_info[0]:  # blocked为True
                print(f"检测到前方障碍，已前进 {moved_distance:.2f}m")
                return True
            
            # 继续向前移动一小步
            remaining_distance = max_distance - moved_distance
            step = min(step_distance, remaining_distance)
            
            if not self.agv.translate(step, vx=velocity):
                print("前进移动失败")
                return False
            
            # 等待移动完成
            move_time = step / velocity
            time.sleep(move_time + 0.5)
            
            moved_distance += step
            print(f"已前进 {moved_distance:.2f}m / {max_distance:.2f}m")
        
        print(f"达到最大前进距离 {max_distance}m，未检测到障碍")
        return True
    
    def _move_side_distance(self, distance, velocity):
        """
        侧向移动指定距离，检查是否遇到障碍
        
        Args:
            distance (float): 侧移距离
            velocity (float): 移动速度
            
        Returns:
            bool: 是否遇到障碍（True表示遇到障碍，应结束建图）
        """
        # 检查移动前是否已经被阻挡
        blocked_info = self.agv.get_blocked()
        if blocked_info and blocked_info[0]:
            print("侧移前检测到障碍")
            return True
        
        # 执行侧移
        if not self.agv.translate(distance, vx=velocity):
            print("侧移失败")
            return False
        
        # 等待侧移完成
        move_time = distance / velocity
        time.sleep(move_time + 0.5)
        
        # 检查侧移后是否遇到障碍
        blocked_info = self.agv.get_blocked()
        if blocked_info and blocked_info[0]:
            print("侧移后检测到障碍")
            return True
        
        print(f"侧移 {distance:.2f}m 完成，未检测到障碍")
        return False
    
    def stage2_localization(self, auto_relocalize=True, manual_position=None):
        """
        第二阶段：定位
        
        Args:
            auto_relocalize (bool): 是否自动重定位
            manual_position (tuple): 手动指定位置 (x, y, angle)
            
        Returns:
            bool: 定位是否成功
        """
        print("========== 开始第二阶段：定位 ==========")
        
        # 1. 检查地图状态
        map_status = self.agv.get_map_status()
        if map_status != 1:
            print(f"地图未加载，状态：{map_status}")
            return False
        
        print("地图已加载，开始重定位...")
        
        # 2. 进行重定位
        if auto_relocalize:
            if not self.agv.relocalize(is_auto=True):
                print("自动重定位失败")
                return False
        else:
            if manual_position is None:
                print("手动重定位需要指定位置")
                return False
            x, y, angle = manual_position
            if not self.agv.relocalize(x=x, y=y, angle=angle):
                print("手动重定位失败")
                return False
        
        # 3. 等待定位完成
        if not self.wait_for_status(self.agv.get_relocalization_status, 1, timeout=30):
            print("重定位超时失败")
            return False
        
        # 4. 验证定位结果
        pose = self.agv.get_pose()
        if pose is None:
            print("无法获取定位后的位置")
            return False
        
        x, y, angle = pose
        print(f"定位成功，当前位置：({x:.3f}, {y:.3f}, {angle:.3f})")
        
        print("========== 第二阶段：定位完成 ==========")
        return True
    
    def stage3_navigation(self, target_points, methods_to_test=None):
        """
        第三阶段：导航测试
        
        Args:
            target_points (list): 目标点列表 [(x1,y1,theta1), (x2,y2,theta2), ...]
            methods_to_test (list): 要测试的导航方法列表，None表示测试所有方法
            
        Returns:
            dict: 各方法的测试结果
        """
        print("========== 开始第三阶段：导航测试 ==========")
        
        if methods_to_test is None:
            methods_to_test = list(self.navigation_methods.keys())
        
        results = {}
        
        for method_name in methods_to_test:
            if method_name not in self.navigation_methods:
                print(f"未知的导航方法：{method_name}")
                continue
            
            print(f"\n--- 测试导航方法：{method_name} ---")
            method_func = self.navigation_methods[method_name]
            method_results = []
            
            for i, (x, y, theta) in enumerate(target_points):
                print(f"导航到目标点 {i+1}/{len(target_points)}: ({x}, {y}, {theta})")
                
                start_time = time.time()
                success = method_func(x, y, theta)
                end_time = time.time()
                
                if success:
                    # 等待导航完成
                    nav_success = self.wait_for_status(self.agv.get_navigation_status, [0, 4], timeout=60)
                    duration = end_time - start_time
                    
                    # 验证到达精度
                    final_pose = self.agv.get_pose()
                    if final_pose:
                        final_x, final_y, final_angle = final_pose
                        position_error = math.sqrt((final_x - x)**2 + (final_y - y)**2)
                        angle_error = abs(final_angle - theta)
                        
                        method_results.append({
                            'target': (x, y, theta),
                            'final': (final_x, final_y, final_angle),
                            'position_error': position_error,
                            'angle_error': angle_error,
                            'duration': duration,
                            'success': nav_success
                        })
                        
                        print(f"导航完成，位置误差：{position_error:.3f}m，角度误差：{angle_error:.3f}rad")
                    else:
                        method_results.append({
                            'target': (x, y, theta),
                            'success': False,
                            'error': '无法获取最终位置'
                        })
                else:
                    method_results.append({
                        'target': (x, y, theta),
                        'success': False,
                        'error': '导航指令发送失败'
                    })
                
                # 短暂休息
                time.sleep(2)
            
            results[method_name] = method_results
        
        print("========== 第三阶段：导航测试完成 ==========")
        return results
    
    def run_complete_workflow(self, mapping_points=None, target_points=None, navigation_methods=None, auto_mapping=False, turn_direction='left', forward_distance=1.0, side_distance=0.5):
        """
        运行完整的工作流程
        
        Args:
            mapping_points (list): 建图点列表，auto_mapping=True时可为None
            target_points (list): 导航目标点列表
            navigation_methods (list): 要测试的导航方法
            auto_mapping (bool): 是否使用自动路径规划建图
            turn_direction (str): 自动建图时的转向方向 'left' 或 'right'
            forward_distance (float): 自动建图时的前进距离
            side_distance (float): 自动建图时的侧移距离
            
        Returns:
            dict: 完整的执行结果
        """
        print("========== 开始AGV完整工作流程 ==========")
        
        # 连接AGV
        if not self.connect():
            return {'success': False, 'error': '连接AGV失败'}
        
        try:
            # 第一阶段：建图
            slam_success = self.stage1_mapping(
                mapping_points=mapping_points,
                auto_mapping=auto_mapping,
                turn_direction=turn_direction,
                forward_distance=forward_distance,
                side_distance=side_distance
            )
            if not slam_success:
                return {'success': False, 'error': '建图阶段失败'}
            
            # 第二阶段：定位
            localization_success = self.stage2_localization()
            if not localization_success:
                return {'success': False, 'error': '定位阶段失败'}
            
            # 第三阶段：导航测试
            navigation_results = self.stage3_navigation(target_points, navigation_methods)
            
            return {
                'success': True,
                'slam_success': slam_success,
                'localization_success': localization_success,
                'navigation_results': navigation_results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'工作流程异常：{str(e)}'}
        
        finally:
            self.disconnect()
    
    def generate_test_report(self, results):
        """
        生成测试报告
        
        Args:
            results (dict): 导航测试结果
        """
        print("\n" + "="*50)
        print("          AGV导航方法测试报告")
        print("="*50)
        
        for method_name, method_results in results.items():
            print(f"\n--- {method_name} ---")
            
            success_count = sum(1 for r in method_results if r.get('success', False))
            total_count = len(method_results)
            success_rate = success_count / total_count * 100 if total_count > 0 else 0
            
            print(f"成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
            
            if success_count > 0:
                # 计算平均误差
                valid_results = [r for r in method_results if r.get('success', False) and 'position_error' in r]
                if valid_results:
                    avg_pos_error = sum(r['position_error'] for r in valid_results) / len(valid_results)
                    avg_angle_error = sum(r['angle_error'] for r in valid_results) / len(valid_results)
                    avg_duration = sum(r['duration'] for r in valid_results) / len(valid_results)
                    
                    print(f"平均位置误差: {avg_pos_error:.3f}m")
                    print(f"平均角度误差: {avg_angle_error:.3f}rad")
                    print(f"平均导航时间: {avg_duration:.1f}s")
            
            # 显示失败的案例
            failed_results = [r for r in method_results if not r.get('success', False)]
            if failed_results:
                print("失败案例:")
                for r in failed_results:
                    print(f"  目标 {r['target']}: {r.get('error', '未知错误')}")


def main():
    """测试完整工作流程"""
    print("AGV 2D建图、定位、导航工作流程测试")
    print("=" * 50)
    print("1. 手动指定点建图模式")
    print("2. 自动路径规划建图模式（扫地机器人式）")
    
    try:
        choice = input("请选择建图模式 (1-2): ").strip()
        
        # 创建工作流程管理器
        workflow = AGVWorkflow(ip='192.168.192.5')
        
        # 定义导航测试目标点
        target_points = [
            (1.0, 1.0, 0.0),      # 中心点
            (2.0, 0.0, 1.57),     # 右下角
            (0.0, 2.0, -1.57),    # 左上角
            (0.0, 0.0, 0.0),      # 起始点
        ]
        
        # 选择要测试的导航方法
        methods_to_test = [
            'method1_world_coordinate',
            'method5_translate_rotate'
        ]
        
        if choice == '1':
            # 手动指定点建图模式
            print("使用手动指定点建图模式")
            mapping_points = [
                (0.0, 0.0, 0.0),      # 起始点
                (2.0, 0.0, 0.0),      # 前方2米
                (2.0, 2.0, 1.57),     # 右转到右上角
                (0.0, 2.0, 3.14),     # 左方到左上角
                (0.0, 0.0, 0.0),      # 回到起始点
            ]
            
            results = workflow.run_complete_workflow(
                mapping_points=mapping_points,
                target_points=target_points,
                navigation_methods=methods_to_test,
                auto_mapping=False
            )
            
        elif choice == '2':
            # 自动路径规划建图模式
            print("使用自动路径规划建图模式（扫地机器人式）")
            
            # 获取自动建图参数
            turn_direction = input("转向方向 (left/right) [默认: left]: ").strip() or 'left'
            forward_distance = float(input("前进距离 (米) [默认: 2.0]: ").strip() or '2.0')
            side_distance = float(input("侧移距离 (米) [默认: 0.8]: ").strip() or '0.8')
            
            results = workflow.run_complete_workflow(
                mapping_points=None,  # 自动建图不需要指定点
                target_points=target_points,
                navigation_methods=methods_to_test,
                auto_mapping=True,
                turn_direction=turn_direction,
                forward_distance=forward_distance,
                side_distance=side_distance
            )
        else:
            print("无效选择")
            return
        
        # 显示结果
        if results['success']:
            print("\n完整工作流程执行成功！")
            workflow.generate_test_report(results['navigation_results'])
        else:
            print(f"\n工作流程失败: {results['error']}")
            
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行异常: {e}")


if __name__ == '__main__':
    main()