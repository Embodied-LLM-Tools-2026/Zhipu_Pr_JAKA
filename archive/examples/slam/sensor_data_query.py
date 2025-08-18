import sys
import os
import json
import time
from datetime import datetime

# 添加父目录到路径，以便导入AGVClient
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from action_sequence.agv_client import AGVClient


class SensorDataQuery:
    """传感器数据查询类"""
    
    def __init__(self, ip='192.168.192.5', timeout=5):
        """
        初始化传感器数据查询客户端
        
        Args:
            ip (str): AGV服务器IP地址
            timeout (int): 连接超时时间(秒)
        """
        self.agv_client = AGVClient(ip, timeout)
        
    def connect(self):
        """连接到AGV服务器"""
        return self.agv_client.connect()
    
    def disconnect(self):
        """断开连接"""
        self.agv_client.disconnect()
    
    def get_imu_data(self):
        """
        获取机器人IMU数据
        
        Returns:
            dict: IMU数据，包含yaw, roll, pitch, 加速度计和陀螺仪数据等
        """
        print("正在查询IMU数据...")
        response = self.agv_client.send_message(1014, socket_type=0)
        
        if response and response.get('ret_code') == 0:
            print("IMU数据获取成功")
            return response
        else:
            print(f"IMU数据获取失败: {response}")
            return None
    
    def get_laser_data(self, return_beams3D=False):
        """
        获取机器人激光点云数据
        
        Args:
            return_beams3D (bool): 是否返回多线激光数据
            
        Returns:
            dict: 激光点云数据
        """
        print("正在查询激光点云数据...")
        
        if return_beams3D:
            msg_data = {'return_beams3D': True}
            response = self.agv_client.send_message(1009, msg_data, socket_type=0)
        else:
            response = self.agv_client.send_message(1009, socket_type=0)
        
        if response and response.get('ret_code') == 0:
            print("激光点云数据获取成功")
            return response
        else:
            print(f"激光点云数据获取失败: {response}")
            return None
    
    def display_imu_data(self, imu_data):
        """
        显示IMU数据
        
        Args:
            imu_data (dict): IMU数据
        """
        if not imu_data:
            print("无IMU数据可显示")
            return
        
        print("\n=== IMU数据 ===")
        print(f"偏航角(yaw): {imu_data.get('yaw', 'N/A'):.6f} rad")
        print(f"滚转角(roll): {imu_data.get('roll', 'N/A'):.6f} rad")
        print(f"俯仰角(pitch): {imu_data.get('pitch', 'N/A'):.6f} rad")
        
        print(f"\n加速度计数据:")
        print(f"  X轴: {imu_data.get('acc_x', 'N/A')}")
        print(f"  Y轴: {imu_data.get('acc_y', 'N/A')}")
        print(f"  Z轴: {imu_data.get('acc_z', 'N/A')}")
        
        print(f"\n陀螺仪数据:")
        print(f"  X轴: {imu_data.get('rot_x', 'N/A')}")
        print(f"  Y轴: {imu_data.get('rot_y', 'N/A')}")
        print(f"  Z轴: {imu_data.get('rot_z', 'N/A')}")
        
        print(f"\n四元数:")
        print(f"  qx: {imu_data.get('qx', 'N/A'):.6f}")
        print(f"  qy: {imu_data.get('qy', 'N/A'):.6f}")
        print(f"  qz: {imu_data.get('qz', 'N/A'):.6f}")
        print(f"  qw: {imu_data.get('qw', 'N/A'):.6f}")
        
        if 'imu_header' in imu_data:
            header = imu_data['imu_header']
            print(f"\n时间戳信息:")
            print(f"  数据时间戳: {header.get('data_nsec', 'N/A')} ns")
            print(f"  发布时间戳: {header.get('pub_nsec', 'N/A')} ns")
            print(f"  帧ID: {header.get('frame_id', 'N/A')}")
        
        print(f"\nAPI时间戳: {imu_data.get('create_on', 'N/A')}")
    
    def display_laser_data(self, laser_data, show_detailed_beams=False, max_beams_display=10):
        """
        显示激光点云数据
        
        Args:
            laser_data (dict): 激光点云数据
            show_detailed_beams (bool): 是否显示详细的光束数据
            max_beams_display (int): 最大显示的光束数量
        """
        if not laser_data:
            print("无激光点云数据可显示")
            return
        
        lasers = laser_data.get('lasers', [])
        print(f"\n=== 激光点云数据 ===")
        print(f"激光传感器数量: {len(lasers)}")
        
        for i, laser in enumerate(lasers):
            print(f"\n--- 激光传感器 {i+1} ---")
            
            # 设备信息
            device_info = laser.get('device_info', {})
            print(f"设备名称: {device_info.get('device_name', 'N/A')}")
            print(f"扫描频率: {device_info.get('scan_freq', 'N/A')} Hz")
            print(f"角度范围: {device_info.get('min_angle', 'N/A')}° ~ {device_info.get('max_angle', 'N/A')}°")
            print(f"距离范围: {device_info.get('min_range', 'N/A')}m ~ {device_info.get('max_range', 'N/A')}m")
            print(f"步长: {device_info.get('pub_step', 'N/A')}°")
            
            # 安装信息
            install_info = laser.get('install_info', {})
            print(f"安装位置: x={install_info.get('x', 'N/A')}m, y={install_info.get('y', 'N/A')}m, z={install_info.get('z', 'N/A')}m")
            print(f"安装角度: yaw={install_info.get('yaw', 'N/A')}°")
            print(f"倒装: {install_info.get('upside', 'N/A')}")
            
            # 光束数据
            beams = laser.get('beams', [])
            print(f"光束数量: {len(beams)}")
            
            if show_detailed_beams and beams:
                print(f"\n光束数据（显示前{min(max_beams_display, len(beams))}个）:")
                print("角度(°)     距离(m)    有效   障碍物  虚拟   信号强度")
                print("-" * 55)
                for j, beam in enumerate(beams[:max_beams_display]):
                    angle = beam.get('angle', 'N/A')
                    dist = beam.get('dist', 'N/A')
                    valid = beam.get('valid', False)
                    is_obstacle = beam.get('is_obstacle', False)
                    is_virtual = beam.get('is_virtual', False)
                    rssi = beam.get('rssi', 'N/A')
                    
                    print(f"{angle:>8.2f}  {dist:>8.3f}  {valid:>5}  {is_obstacle:>6}  {is_virtual:>5}  {rssi:>8}")
            
            # 有效光束统计
            valid_beams = [beam for beam in beams if beam.get('valid', False)]
            obstacle_beams = [beam for beam in beams if beam.get('is_obstacle', False)]
            print(f"有效光束: {len(valid_beams)}/{len(beams)}")
            print(f"障碍物光束: {len(obstacle_beams)}")
            
            if valid_beams:
                distances = [beam['dist'] for beam in valid_beams if isinstance(beam.get('dist'), (int, float))]
                if distances:
                    print(f"距离统计: 最小={min(distances):.3f}m, 最大={max(distances):.3f}m, 平均={sum(distances)/len(distances):.3f}m")
    
    def save_data_to_file(self, imu_data=None, laser_data=None, filename=None):
        """
        保存传感器数据到文件
        
        Args:
            imu_data (dict): IMU数据
            laser_data (dict): 激光点云数据
            filename (str): 文件名，如不指定则使用时间戳
        """
        if not imu_data and not laser_data:
            print("没有数据可保存")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensor_data_{timestamp}.json"
        
        data_to_save = {
            'timestamp': datetime.now().isoformat(),
            'imu_data': imu_data,
            'laser_data': laser_data
        }
        
        try:
            # 确保slam目录存在
            save_dir = os.path.dirname(__file__)
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            print(f"数据已保存到: {filepath}")
            
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def query_all_data(self, save_to_file=False, show_detailed_laser=False):
        """
        查询所有传感器数据
        
        Args:
            save_to_file (bool): 是否保存到文件
            show_detailed_laser (bool): 是否显示详细的激光数据
        """
        print("开始查询所有传感器数据...")
        
        # 获取IMU数据
        imu_data = self.get_imu_data()
        
        # 获取激光点云数据
        laser_data = self.get_laser_data()
        
        # 显示数据
        if imu_data:
            self.display_imu_data(imu_data)
        
        if laser_data:
            self.display_laser_data(laser_data, show_detailed_laser)
        
        # 保存数据
        if save_to_file:
            self.save_data_to_file(imu_data, laser_data)
        
        return imu_data, laser_data
    
    def continuous_query(self, interval=1.0, duration=None, save_to_file=False):
        """
        连续查询传感器数据
        
        Args:
            interval (float): 查询间隔（秒）
            duration (float): 查询持续时间（秒），None表示无限制
            save_to_file (bool): 是否保存到文件
        """
        print(f"开始连续查询传感器数据，间隔: {interval}秒")
        if duration:
            print(f"持续时间: {duration}秒")
        else:
            print("按Ctrl+C停止查询")
        
        start_time = time.time()
        count = 0
        
        try:
            while True:
                count += 1
                current_time = time.time()
                
                print(f"\n=== 第{count}次查询 ({datetime.now().strftime('%H:%M:%S')}) ===")
                
                # 查询数据
                imu_data, laser_data = self.query_all_data(save_to_file)
                
                # 检查是否超过持续时间
                if duration and (current_time - start_time) >= duration:
                    print(f"\n查询完成，总共查询了{count}次")
                    break
                
                # 等待下次查询
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n用户中断，总共查询了{count}次")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='查询机器人传感器数据')
    parser.add_argument('--ip', default='192.168.192.5', help='AGV服务器IP地址')
    parser.add_argument('--save', action='store_true', help='保存数据到文件')
    parser.add_argument('--detailed-laser', action='store_true', help='显示详细的激光数据')
    parser.add_argument('--continuous', action='store_true', help='连续查询模式')
    parser.add_argument('--interval', type=float, default=1.0, help='连续查询间隔（秒）')
    parser.add_argument('--duration', type=float, help='连续查询持续时间（秒）')
    parser.add_argument('--imu-only', action='store_true', help='仅查询IMU数据')
    parser.add_argument('--laser-only', action='store_true', help='仅查询激光数据')
    
    args = parser.parse_args()
    
    print("传感器数据查询工具")
    print(f"连接到AGV服务器: {args.ip}")
    
    try:
        with SensorDataQuery(ip=args.ip) as sensor_query:
            if args.continuous:
                # 连续查询模式
                sensor_query.continuous_query(
                    interval=args.interval,
                    duration=args.duration,
                    save_to_file=args.save
                )
            else:
                # 单次查询模式
                if args.imu_only:
                    # 仅查询IMU
                    imu_data = sensor_query.get_imu_data()
                    if imu_data:
                        sensor_query.display_imu_data(imu_data)
                        if args.save:
                            sensor_query.save_data_to_file(imu_data=imu_data)
                
                elif args.laser_only:
                    # 仅查询激光
                    laser_data = sensor_query.get_laser_data()
                    if laser_data:
                        sensor_query.display_laser_data(laser_data, args.detailed_laser)
                        if args.save:
                            sensor_query.save_data_to_file(laser_data=laser_data)
                
                else:
                    # 查询所有数据
                    sensor_query.query_all_data(
                        save_to_file=args.save,
                        show_detailed_laser=args.detailed_laser
                    )
    
    except Exception as e:
        print(f"程序运行出错: {e}")


if __name__ == '__main__':
    main()