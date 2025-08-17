import minimalmodbus
import serial.tools.list_ports
import time

def test_basic_connection():
    """测试基本连接"""
    print("=== 测试基本连接 ===")
    
    # 检测端口
    ports = [p.device for p in serial.tools.list_ports.comports()
             if 'USB Serial' in p.description or 'RS485' in p.description or 'CH340' in p.description]
    
    if not ports:
        print("未检测到RS485端口")
        return False
    
    port = ports[0]
    print(f"检测到端口: {port}")
    
    # 尝试不同的设备ID
    for slave_id in [1, 2, 3]:
        print(f"\n尝试设备ID: {slave_id}")
        try:
            instrument = minimalmodbus.Instrument(port, slave_id)
            
            # 串口配置
            instrument.serial.baudrate = 115200
            instrument.serial.bytesize = 8
            instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
            instrument.serial.stopbits = 1
            instrument.serial.timeout = 1.0
            instrument.close_port_after_each_call = True
            
            # 测试读取硬件版本
            print(f"  尝试读取寄存器 0x00...")
            hw_version = instrument.read_register(0x00)
            print(f"  硬件版本: {hw_version:#06x}")
            
            # 测试读取固件版本
            print(f"  尝试读取寄存器 0x01...")
            fw_version = instrument.read_register(0x01)
            print(f"  固件版本: {fw_version:#06x}")
            
            print(f"  ✓ 设备ID {slave_id} 连接成功!")
            return True, port, slave_id
            
        except Exception as e:
            print(f"  ✗ 设备ID {slave_id} 连接失败: {str(e)}")
            continue
    
    return False, None, None

def test_register_access(port, slave_id):
    """测试寄存器访问"""
    print(f"\n=== 测试寄存器访问 (端口: {port}, 设备ID: {slave_id}) ===")
    
    try:
        instrument = minimalmodbus.Instrument(port, slave_id)
        
        # 串口配置
        instrument.serial.baudrate = 115200
        instrument.serial.bytesize = 8
        instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
        instrument.serial.stopbits = 1
        instrument.serial.timeout = 1.0
        instrument.close_port_after_each_call = True
        
        # 测试一些基础寄存器
        test_registers = [
            (0x00, "硬件版本"),
            (0x01, "固件版本"),
            (0x0A, "手指1设置位置"),
            (0x0B, "手指1输出力"),
            (0x0C, "手指1速度"),
            (0x0E, "手指1实际位置"),
            (0x14, "手指2设置位置"),
            (0x15, "手指2输出力"),
            (0x16, "手指2速度"),
            (0x18, "手指2实际位置"),
            (0x28, "闭合指令"),
            (0x29, "张开指令"),
            (0x2A, "同步寄存器"),
            (0x2B, "移动状态"),
            (0x2C, "位置到达"),
            (0x32, "夹持状态"),
        ]
        
        for reg_addr, reg_name in test_registers:
            try:
                value = instrument.read_register(reg_addr)
                print(f"  ✓ {reg_name} (0x{reg_addr:02X}): {value}")
            except Exception as e:
                print(f"  ✗ {reg_name} (0x{reg_addr:02X}): 读取失败 - {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"寄存器访问测试失败: {str(e)}")
        return False

def main():
    print("夹具连接诊断工具")
    print("=" * 50)
    
    # 测试基本连接
    success, port, slave_id = test_basic_connection()
    
    if success:
        print(f"\n✓ 基本连接成功! 端口: {port}, 设备ID: {slave_id}")
        
        # 测试寄存器访问
        test_register_access(port, slave_id)
        
        print(f"\n建议的配置:")
        print(f"  端口: {port}")
        print(f"  设备ID: {slave_id}")
        print(f"  波特率: 115200")
        
    else:
        print("\n✗ 连接失败")
        print("请检查:")
        print("1. 夹具是否已上电")
        print("2. USB转RS485转换器是否正确连接")
        print("3. 驱动程序是否正确安装")
        print("4. 串口是否被其他程序占用")

if __name__ == "__main__":
    main() 