import minimalmodbus
import serial.tools.list_ports
import time

def test_modbus_connection():
    """测试Modbus连接"""
    print("=== Modbus连接测试 ===")
    
    # 检测端口
    ports = [p.device for p in serial.tools.list_ports.comports()]
    print(f"可用端口: {ports}")
    
    # 查找RS485端口
    rs485_ports = [p.device for p in serial.tools.list_ports.comports()
                   if 'USB Serial' in p.description or 'RS485' in p.description or 'CH340' in p.description]
    
    if not rs485_ports:
        print("未检测到RS485端口，尝试使用第一个可用端口")
        if ports:
            port = ports[0]
        else:
            print("没有可用端口")
            return False
    else:
        port = rs485_ports[0]
    
    print(f"使用端口: {port}")
    
    # 测试不同的设备ID
    for slave_id in [1, 2, 3]:
        print(f"\n--- 测试设备ID: {slave_id} ---")
        
        try:
            # 创建Modbus设备
            instrument = minimalmodbus.Instrument(port, slave_id)
            
            # 串口配置
            instrument.serial.baudrate = 115200
            instrument.serial.bytesize = 8
            instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
            instrument.serial.stopbits = 1
            instrument.serial.timeout = 2.0
            instrument.close_port_after_each_call = True
            
            print(f"  配置: 波特率=115200, 设备ID={slave_id}")
            
            # 测试读取硬件版本 (0x00)
            print(f"  尝试读取寄存器 0x00 (硬件版本)...")
            hw_version = instrument.read_register(0x00)
            print(f"  硬件版本: {hw_version:#06x}")
            
            # 测试读取固件版本 (0x01)
            print(f"  尝试读取寄存器 0x01 (固件版本)...")
            fw_version = instrument.read_register(0x01)
            print(f"  固件版本: {fw_version:#06x}")
            
            print(f"  ✓ 设备ID {slave_id} 连接成功!")
            
            # 测试写入同步寄存器
            print(f"  尝试写入同步寄存器 0x002A...")
            instrument.write_register(0x002A, 1)
            print(f"  ✓ 同步寄存器写入成功")
            
            # 测试读取同步寄存器
            print(f"  尝试读取同步寄存器 0x002A...")
            sync_value = instrument.read_register(0x002A)
            print(f"  同步寄存器值: {sync_value}")
            
            print(f"\n✓ 设备ID {slave_id} 完全测试通过!")
            return True, port, slave_id
            
        except Exception as e:
            print(f"  ✗ 设备ID {slave_id} 测试失败: {str(e)}")
            continue
    
    return False, None, None

def main():
    print("简单Modbus测试工具")
    print("=" * 40)
    
    success, port, slave_id = test_modbus_connection()
    
    if success:
        print(f"\n✓ 测试成功!")
        print(f"  推荐配置:")
        print(f"    端口: {port}")
        print(f"    设备ID: {slave_id}")
        print(f"    波特率: 115200")
        
        print(f"\n现在可以运行主程序:")
        print(f"  python controller/gripper_controller.py")
    else:
        print(f"\n✗ 测试失败")
        print(f"请检查:")
        print(f"1. 夹具是否已上电")
        print(f"2. USB转RS485转换器是否正确连接")
        print(f"3. 驱动程序是否正确安装")

if __name__ == "__main__":
    main() 