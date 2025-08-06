import minimalmodbus
import serial.tools.list_ports
from enum import IntEnum
import time
import threading
import queue

class GripperCommand(IntEnum):
    """夹具命令枚举"""
    DISABLE = 0
    ENABLE = 1

class FingerIndex(IntEnum):
    """手指索引枚举"""
    FINGER_1 = 0
    FINGER_2 = 1

class CTAG2F120Gripper:
    """CTAG2F120 两指夹具控制类 - 基于真实协议的连续控制"""
    
    def __init__(self, port=None, slave_id=1):
        """
        初始化两指夹具控制器
        
        Args:
            port: 串口端口，如果为None则自动检测
            slave_id: 设备ID，默认为1
        """
        self.port = port or self._detect_port()
        self.slave_id = slave_id
        self.instrument = minimalmodbus.Instrument(self.port, slave_id)
        
        # 串口配置
        self.instrument.serial.baudrate = 115200
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout = 0.5
        self.instrument.close_port_after_each_call = True
        
        # 基础控制寄存器
        self.REG_CLOSE = 0x0028     # 闭合指令寄存器
        self.REG_OPEN = 0x0029      # 张开指令寄存器  
        self.REG_SYNC = 0x002A      # 同步寄存器
        self.REG_MOVING = 0x002B    # 移动状态寄存器
        self.REG_POSITION_REACHED = 0x002C  # 位置到达寄存器
        self.REG_GRIPPING = 0x0032  # 夹持状态寄存器
        
        # 配置寄存器
        self.REG_OPEN_POSITION = 0x0048   # 张开位置配置
        self.REG_CLOSE_POSITION = 0x0049  # 闭合位置配置
        
        # 两个手指控制寄存器
        self.finger_regs = {
            FingerIndex.FINGER_1: {
                'set_position': 0x000A,    # 设置位置-1
                'output_force': 0x000B,    # 输出力-1
                'speed': 0x000C,           # 速度-1
                'acceleration': 0x000D,    # 加速度-1
                'actual_position': 0x000E, # 实际位置-1
                'torque': 0x000F,          # 扭力-1
                'actual_speed': 0x0010,    # 实际速度-1
                'voltage': 0x0011,         # 电压-1
                'current': 0x0012,         # 电流-1
                'temperature': 0x0013,     # 温度-1
                'enable': 0x004A           # 指端1使能
            },
            FingerIndex.FINGER_2: {
                'set_position': 0x0014,    # 设置位置-2
                'output_force': 0x0015,    # 输出力-2
                'speed': 0x0016,           # 速度-2
                'acceleration': 0x0017,    # 加速度-2
                'actual_position': 0x0018, # 实际位置-2
                'torque': 0x0019,          # 扭力-2
                'actual_speed': 0x001A,    # 实际速度-2
                'voltage': 0x001B,         # 电压-2
                'current': 0x001C,         # 电流-2
                'temperature': 0x001D,     # 温度-2
                'enable': 0x004B           # 指端2使能
            }
        }
        
        # 力传感器寄存器（两个手指都有）
        self.sensor_regs = {
            FingerIndex.FINGER_1: {
                'finger_id': 0x0078,       # 传感器对应手指号码
                'force_x': 0x0079,         # X切向力
                'force_y': 0x007A,         # Y切向力  
                'force_z': 0x007B,         # Z法向力
                'zero_x': 0x007C,          # X切向力置零值
                'zero_y': 0x007D,          # Y切向力置零值
                'zero_z': 0x007E,          # Z法向力置零值
                'can_id': 0x007F,          # 传感器CAN_ID
                'zero_cmd': 0x0080         # 传感器清零
            },
            FingerIndex.FINGER_2: {
                'finger_id': 0x0082,       # 传感器对应手指号码
                'force_x': 0x0083,         # X切向力
                'force_y': 0x0084,         # Y切向力
                'force_z': 0x0085,         # Z法向力
                'zero_x': 0x0086,          # X切向力置零值
                'zero_y': 0x0087,          # Y切向力置零值
                'zero_z': 0x0088,          # Z法向力置零值
                'can_id': 0x0089,          # 传感器CAN_ID
                'zero_cmd': 0x008A         # 传感器清零
            }
        }
        
        # 当前状态 - 改为两指
        self.current_positions = [100, 100]  # 当前位置 (0-100%)
        self.target_positions = [100, 100]   # 目标位置 (0-100%)
        self.current_forces = [25, 25]       # 当前力度 (0-100%)
        self.current_speeds = [40, 40]       # 当前速度 (0-32766步/s)
        
        # 运动状态
        self.is_moving = False
        self.position_tolerance = 2  # 位置误差容限(%)
        
        # 实时监控
        self.monitor_thread = None
        self.monitor_running = False
        self.status_queue = queue.Queue()

    def _detect_port(self):
        """自动检测串口端口"""
        ports = [p.device for p in serial.tools.list_ports.comports()
                 if 'USB Serial' in p.description or 'RS485' in p.description or 'CH340' in p.description]
        if not ports:
            raise RuntimeError('未检测到RS485端口')
        return ports[0]

    def connect(self):
        """连接设备并测试通信"""
        try:
            # 读取硬件版本号来测试连接
            hw_version = self.instrument.read_register(0x0000)
            fw_version = self.instrument.read_register(0x0001)
            print(f'连接成功，设备ID: {self.slave_id}')
            print(f'硬件版本: {hw_version:#06x}, 固件版本: {fw_version:#06x}')
            
            # 读取当前状态
            self._update_current_status()
            
            # 启动实时监控
            self.start_monitoring()
            return True
        except Exception as e:
            raise ConnectionError(f'连接失败: {str(e)}')

    def disconnect(self):
        """断开连接"""
        self.stop_monitoring()
        print('两指夹具已断开连接')

    # ==================== 基础控制功能 ====================
    
    def set_sync_mode(self, sync_enabled=True):
        """
        设置指端同步模式
        
        Args:
            sync_enabled (bool): True为同步运行，False为不同步运行
        """
        try:
            value = GripperCommand.ENABLE if sync_enabled else GripperCommand.DISABLE
            self.instrument.write_register(self.REG_SYNC, value)
            mode_str = "同步" if sync_enabled else "不同步"
            print(f'两指设置为{mode_str}运行模式')
        except Exception as e:
            raise RuntimeError(f'设置同步模式失败: {str(e)}')

    def get_sync_mode(self):
        """获取当前指端同步模式"""
        try:
            value = self.instrument.read_register(self.REG_SYNC)
            return bool(value)
        except Exception as e:
            raise RuntimeError(f'读取同步模式失败: {str(e)}')

    def close_gripper(self):
        """执行闭合指令（全局）"""
        try:
            self.instrument.write_register(self.REG_CLOSE, GripperCommand.ENABLE)
            self.target_positions = [0, 0]  # 闭合为0%
            print('两指夹具全局闭合指令已发送')
        except Exception as e:
            raise RuntimeError(f'闭合指令执行失败: {str(e)}')

    def open_gripper(self):
        """执行张开指令（全局）"""
        try:
            self.instrument.write_register(self.REG_OPEN, GripperCommand.ENABLE)
            self.target_positions = [100, 100]  # 张开为100%
            print('两指夹具全局张开指令已发送')
        except Exception as e:
            raise RuntimeError(f'张开指令执行失败: {str(e)}')

    # ==================== 单指控制功能 ====================
    
    def set_finger_position(self, finger: FingerIndex, position: int, speed: int = None):
        """
        设置单个手指位置
        
        Args:
            finger: 手指索引 (0-1)
            position: 目标位置 (0-100, 0为闭合，100为张开)
            speed: 运动速度 (可选，0-32766步/s)
        """
        if not 0 <= position <= 100:
            raise ValueError("位置值必须在0-100之间")
        
        try:
            reg_addr = self.finger_regs[finger]['set_position']
            self.instrument.write_register(reg_addr, position)
            self.target_positions[finger.value] = position
            
            # 设置速度（如果提供）
            if speed is not None:
                self.set_finger_speed(finger, speed)
            
            print(f'手指{finger.value + 1}设置位置: {position}%')
        except Exception as e:
            raise RuntimeError(f'设置手指{finger.value + 1}位置失败: {str(e)}')

    def set_finger_force(self, finger: FingerIndex, force: int):
        """
        设置单个手指输出力
        
        Args:
            finger: 手指索引 (0-1)
            force: 输出力 (0-100%, 40%约为15N)
        """
        if not 0 <= force <= 100:
            raise ValueError("力度值必须在0-100之间")
        
        try:
            reg_addr = self.finger_regs[finger]['output_force']
            self.instrument.write_register(reg_addr, force)
            self.current_forces[finger.value] = force
            print(f'手指{finger.value + 1}设置力度: {force}% (约{force * 0.375:.1f}N)')
        except Exception as e:
            raise RuntimeError(f'设置手指{finger.value + 1}力度失败: {str(e)}')

    def set_finger_speed(self, finger: FingerIndex, speed: int):
        """
        设置单个手指速度
        
        Args:
            finger: 手指索引 (0-1)
            speed: 速度 (0-32766步/s, 50步/s = 0.732 RPM)
        """
        if not 0 <= speed <= 32766:
            raise ValueError("速度值必须在0-32766之间")
        
        try:
            reg_addr = self.finger_regs[finger]['speed']
            self.instrument.write_register(reg_addr, speed)
            self.current_speeds[finger.value] = speed
            rpm = speed * 0.732 / 50  # 转换为RPM
            print(f'手指{finger.value + 1}设置速度: {speed}步/s ({rpm:.2f} RPM)')
        except Exception as e:
            raise RuntimeError(f'设置手指{finger.value + 1}速度失败: {str(e)}')

    def set_finger_acceleration(self, finger: FingerIndex, acceleration: int):
        """
        设置单个手指加速度
        
        Args:
            finger: 手指索引 (0-1)
            acceleration: 加速度 (0-254, 单位100步/s²)
        """
        if not 0 <= acceleration <= 254:
            raise ValueError("加速度值必须在0-254之间")
        
        try:
            reg_addr = self.finger_regs[finger]['acceleration']
            self.instrument.write_register(reg_addr, acceleration)
            actual_acc = acceleration * 100  # 实际加速度
            print(f'手指{finger.value + 1}设置加速度: {actual_acc}步/s²')
        except Exception as e:
            raise RuntimeError(f'设置手指{finger.value + 1}加速度失败: {str(e)}')

    def enable_finger(self, finger: FingerIndex, enabled: bool = True):
        """
        使能/禁用单个手指
        
        Args:
            finger: 手指索引 (0-1)
            enabled: 是否使能 (True=保持力矩, False=不保持力矩)
        """
        try:
            reg_addr = self.finger_regs[finger]['enable']
            value = GripperCommand.ENABLE if enabled else GripperCommand.DISABLE
            self.instrument.write_register(reg_addr, value)
            status = "使能" if enabled else "禁用"
            print(f'手指{finger.value + 1}{status}')
        except Exception as e:
            raise RuntimeError(f'设置手指{finger.value + 1}使能状态失败: {str(e)}')

    # ==================== 批量控制功能 ====================
    
    def set_both_positions(self, positions: list, speeds: list = None):
        """
        设置两个手指位置
        
        Args:
            positions: 两个手指的位置列表 [finger1, finger2] (0-100)
            speeds: 两个手指的速度列表（可选）
        """
        if len(positions) != 2:
            raise ValueError("必须提供2个手指的位置")
        
        for i, position in enumerate(positions):
            finger = FingerIndex(i)
            speed = speeds[i] if speeds else None
            self.set_finger_position(finger, position, speed)

    def set_both_forces(self, forces: list):
        """
        设置两个手指力度
        
        Args:
            forces: 两个手指的力度列表 [finger1, finger2] (0-100)
        """
        if len(forces) != 2:
            raise ValueError("必须提供2个手指的力度")
        
        for i, force in enumerate(forces):
            finger = FingerIndex(i)
            self.set_finger_force(finger, force)

    def set_both_speeds(self, speeds: list):
        """
        设置两个手指速度
        
        Args:
            speeds: 两个手指的速度列表 [finger1, finger2] (0-32766)
        """
        if len(speeds) != 2:
            raise ValueError("必须提供2个手指的速度")
        
        for i, speed in enumerate(speeds):
            finger = FingerIndex(i)
            self.set_finger_speed(finger, speed)

    # ==================== 状态获取功能 ====================
    
    def get_finger_position(self, finger: FingerIndex):
        """获取单个手指实际位置"""
        try:
            reg_addr = self.finger_regs[finger]['actual_position']
            position = self.instrument.read_register(reg_addr)
            self.current_positions[finger.value] = position
            return position
        except Exception as e:
            print(f'读取手指{finger.value + 1}位置失败: {str(e)}')
            return self.target_positions[finger.value]

    def get_finger_force(self, finger: FingerIndex):
        """获取单个手指当前扭力"""
        try:
            reg_addr = self.finger_regs[finger]['torque']
            torque = self.instrument.read_register(reg_addr)
            return torque
        except Exception as e:
            print(f'读取手指{finger.value + 1}扭力失败: {str(e)}')
            return 0

    def get_finger_status(self, finger: FingerIndex):
        """获取单个手指完整状态"""
        try:
            regs = self.finger_regs[finger]
            return {
                'finger_id': finger.value + 1,
                'actual_position': self.instrument.read_register(regs['actual_position']),
                'target_position': self.target_positions[finger.value],
                'torque': self.instrument.read_register(regs['torque']),
                'actual_speed': self.instrument.read_register(regs['actual_speed']),
                'voltage': self.instrument.read_register(regs['voltage']) * 0.1,  # 转换为V
                'current': self.instrument.read_register(regs['current']) * 6.5,  # 转换为mA
                'temperature': self.instrument.read_register(regs['temperature']),
                'enabled': bool(self.instrument.read_register(regs['enable']))
            }
        except Exception as e:
            raise RuntimeError(f'获取手指{finger.value + 1}状态失败: {str(e)}')

    def get_both_positions(self):
        """获取两个手指位置"""
        positions = []
        for finger in FingerIndex:
            positions.append(self.get_finger_position(finger))
        return positions

    def get_system_status(self):
        """获取系统整体状态"""
        try:
            sync_mode = self.get_sync_mode()
            is_moving = bool(self.instrument.read_register(self.REG_MOVING))
            position_reached = bool(self.instrument.read_register(self.REG_POSITION_REACHED))
            is_gripping = bool(self.instrument.read_register(self.REG_GRIPPING))
            
            finger_statuses = []
            for finger in FingerIndex:
                finger_statuses.append(self.get_finger_status(finger))
            
            return {
                'sync_enabled': sync_mode,
                'is_moving': is_moving,
                'position_reached': position_reached,
                'is_gripping': is_gripping,
                'fingers': finger_statuses,
                'target_positions': self.target_positions.copy(),
                'current_positions': self.get_both_positions()
            }
        except Exception as e:
            raise RuntimeError(f'获取系统状态失败: {str(e)}')

    # ==================== 力传感器功能 ====================
    
    def get_finger_forces(self, finger: FingerIndex):
        """
        获取手指力传感器数据
        
        Args:
            finger: 手指索引 (0-1，两个手指都有力传感器)
        
        Returns:
            dict: 包含X/Y/Z方向力的字典 (单位: 0.01N)
        """
        try:
            regs = self.sensor_regs[finger]
            force_x = self.instrument.read_register(regs['force_x'])
            force_y = self.instrument.read_register(regs['force_y'])
            force_z = self.instrument.read_register(regs['force_z'])
            
            # 处理负值（补码形式）
            if force_x > 32767:
                force_x -= 65536
            if force_y > 32767:
                force_y -= 65536
            
            return {
                'force_x': force_x * 0.01,  # 转换为N
                'force_y': force_y * 0.01,  # 转换为N
                'force_z': force_z * 0.01,  # 转换为N
                'finger_id': finger.value + 1
            }
        except Exception as e:
            raise RuntimeError(f'获取手指{finger.value + 1}力传感器数据失败: {str(e)}')

    def zero_force_sensor(self, finger: FingerIndex):
        """
        力传感器置零
        
        Args:
            finger: 手指索引 (0-1，两个手指都有力传感器)
        """
        try:
            reg_addr = self.sensor_regs[finger]['zero_cmd']
            self.instrument.write_register(reg_addr, 1)  # 0变1触发置零
            print(f'手指{finger.value + 1}力传感器置零完成')
        except Exception as e:
            raise RuntimeError(f'手指{finger.value + 1}力传感器置零失败: {str(e)}')

    # ==================== 实时监控功能 ====================
    
    def _update_current_status(self):
        """更新当前状态"""
        self.current_positions = self.get_both_positions()

    def start_monitoring(self):
        """开始实时状态监控"""
        if not self.monitor_running:
            self.monitor_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print('实时监控已启动')

    def stop_monitoring(self):
        """停止实时状态监控"""
        self.monitor_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        print('实时监控已停止')

    def _monitor_loop(self):
        """监控循环"""
        while self.monitor_running:
            try:
                # 更新位置状态
                self._update_current_status()
                
                # 检查运动状态
                self.is_moving = bool(self.instrument.read_register(self.REG_MOVING))
                
                # 将状态信息放入队列
                status = self.get_system_status()
                if not self.status_queue.full():
                    self.status_queue.put(status)
                
                time.sleep(0.2)  # 200ms更新一次
            except Exception as e:
                print(f'监控线程错误: {str(e)}')
                time.sleep(0.5)

    def get_latest_status(self):
        """获取最新状态信息（非阻塞）"""
        try:
            return self.status_queue.get_nowait()
        except queue.Empty:
            return None

    # ==================== 高级控制功能 ====================
    
    def wait_for_completion(self, timeout=10):
        """等待运动完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not bool(self.instrument.read_register(self.REG_MOVING)):
                position_reached = bool(self.instrument.read_register(self.REG_POSITION_REACHED))
                if position_reached:
                    print('运动完成')
                    return True
            time.sleep(0.1)
        
        print('等待运动完成超时')
        return False

    def calibrate(self):
        """两指夹具校准 - 完整的张开闭合循环"""
        print('开始两指夹具校准...')
        
        # 设置同步模式
        self.set_sync_mode(True)
        
        # 全张开
        print('张开夹具...')
        self.open_gripper()
        self.wait_for_completion()
        time.sleep(1)
        
        # 全闭合
        print('闭合夹具...')
        self.close_gripper()
        self.wait_for_completion()
        time.sleep(1)
        
        # 回到中间位置
        print('回到中间位置...')
        self.set_both_positions([50, 50])
        self.wait_for_completion()
        
        print('两指夹具校准完成')

    def gentle_grip(self, force=30):
        """轻柔夹持"""
        self.set_both_forces([force, force])
        self.set_both_positions([20, 20])  # 轻微闭合
        print(f'执行轻柔夹持，力度: {force}%')

    def firm_grip(self, force=80):
        """牢固夹持"""
        self.set_both_forces([force, force])
        self.set_both_positions([10, 10])  # 较大程度闭合
        print(f'执行牢固夹持，力度: {force}%')

    def emergency_stop(self):
        """紧急停止"""
        try:
            # 禁用所有手指
            for finger in FingerIndex:
                self.enable_finger(finger, False)
            print('紧急停止已执行 - 两个手指已禁用')
        except Exception as e:
            print(f'紧急停止失败: {str(e)}')

    def save_configuration(self):
        """保存当前配置到FLASH"""
        try:
            self.instrument.write_register(0x0034, 1)  # 数值保存寄存器
            print('配置保存到FLASH')
        except Exception as e:
            print(f'保存配置失败: {str(e)}')

    def restore_defaults(self):
        """恢复默认参数配置"""
        try:
            self.instrument.write_register(0x0035, 1)  # 恢复默认参数寄存器
            print('已恢复默认参数配置')
        except Exception as e:
            print(f'恢复默认配置失败: {str(e)}')

    # ==================== 连续控制功能 ====================
    
    def smooth_move(self, finger: FingerIndex, target_position: int, duration: float, steps: int = 20):
        """
        平滑移动到目标位置
        
        Args:
            finger: 手指索引
            target_position: 目标位置 (0-100)
            duration: 移动时间（秒）
            steps: 分解步数
        """
        current_pos = self.get_finger_position(finger)
        step_size = (target_position - current_pos) / steps
        step_time = duration / steps
        
        print(f'手指{finger.value + 1}平滑移动: {current_pos}% -> {target_position}%')
        
        for i in range(steps + 1):
            pos = int(current_pos + step_size * i)
            self.set_finger_position(finger, pos)
            time.sleep(step_time)

    def smooth_move_both(self, target_positions: list, duration: float, steps: int = 20):
        """
        两指同时平滑移动
        
        Args:
            target_positions: 目标位置列表 [finger1, finger2]
            duration: 移动时间（秒）
            steps: 分解步数
        """
        if len(target_positions) != 2:
            raise ValueError("必须提供2个手指的位置")
        
        current_positions = self.get_both_positions()
        step_time = duration / steps
        
        print(f'两指同时平滑移动: {current_positions} -> {target_positions}')
        
        for i in range(steps + 1):
            positions = []
            for j in range(2):
                step_size = (target_positions[j] - current_positions[j]) / steps
                pos = int(current_positions[j] + step_size * i)
                positions.append(pos)
            
            self.set_both_positions(positions)
            time.sleep(step_time)

    def continuous_grip_control(self, target_force: float, max_position: int = 30):
        """
        连续力控夹持
        
        Args:
            target_force: 目标力度 (N)
            max_position: 最大闭合位置 (0-100)
        """
        print(f'开始连续力控夹持，目标力度: {target_force}N')
        
        # 设置力度
        force_percent = min(int(target_force / 0.375), 100)
        self.set_both_forces([force_percent, force_percent])
        
        # 逐步闭合直到达到目标力度
        for position in range(100, max_position, -5):
            self.set_both_positions([position, position])
            time.sleep(0.2)
            
            # 检查力传感器反馈
            try:
                forces1 = self.get_finger_forces(FingerIndex.FINGER_1)
                forces2 = self.get_finger_forces(FingerIndex.FINGER_2)
                
                # 计算总力
                total_force = abs(forces1['force_z']) + abs(forces2['force_z'])
                
                if total_force >= target_force:
                    print(f'达到目标力度: {total_force:.2f}N，位置: {position}%')
                    break
                    
            except Exception as e:
                print(f'力传感器读取失败: {e}')
                continue
        
        print('连续力控夹持完成')

if __name__ == '__main__':
    gripper = None
    try:
        # 初始化两指夹具
        gripper = CTAG2F120Gripper(port=None)
        print(f"检测到的端口: {gripper.port}")
        
        # 连接设备
        gripper.connect()
        
        # 校准夹具
        print("开始校准...")
        gripper.calibrate()
        
        # 测试单指控制
        print("测试单指控制...")
        gripper.set_finger_position(FingerIndex.FINGER_1, 30, speed=100)
        gripper.set_finger_position(FingerIndex.FINGER_2, 50, speed=150)
        
        time.sleep(3)
        
        # 测试批量控制
        print("测试批量控制...")
        gripper.set_both_positions([80, 60], speeds=[100, 100])
        gripper.wait_for_completion()
        
        # 测试平滑移动
        print("测试平滑移动...")
        gripper.smooth_move_both([20, 30], duration=2.0)
        
        # 测试力传感器
        try:
            print("测试力传感器...")
            forces1 = gripper.get_finger_forces(FingerIndex.FINGER_1)
            forces2 = gripper.get_finger_forces(FingerIndex.FINGER_2)
            print(f"手指1力传感器: {forces1}")
            print(f"手指2力传感器: {forces2}")
        except Exception as e:
            print(f"力传感器测试失败: {e}")
        
        # 显示最终状态
        status = gripper.get_system_status()
        print(f"系统状态: {status}")

    except Exception as e:
        print(f"错误: {str(e)}")
        if gripper:
            gripper.emergency_stop()
    finally:
        if gripper:
            gripper.disconnect()
        print("程序结束")