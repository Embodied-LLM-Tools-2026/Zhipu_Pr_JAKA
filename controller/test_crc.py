#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
CRC校验测试文件
演示如何使用CRC16校验函数
"""

from gripper_controller import InspireHandR

def test_crc_functions():
    """测试CRC校验函数"""
    
    # 创建手部控制器实例（不连接串口）
    hand = InspireHandR()
    
    # 测试数据
    test_data = [0x01, 0x03, 0x06, 0x0C, 0x00, 0x01]
    
    print("=== CRC16校验测试 ===")
    print(f"测试数据: {[hex(x) for x in test_data]}")
    
    # 测试Modbus CRC16
    print("\n--- Modbus CRC16 ---")
    crc_modbus = hand.crc16_modbus(test_data)
    crc_low, crc_high = hand.calculate_crc16(test_data, "modbus")
    print(f"CRC16 Modbus: 0x{crc_modbus:04X}")
    print(f"CRC低字节: 0x{crc_low:02X}")
    print(f"CRC高字节: 0x{crc_high:02X}")
    
    # 测试CCITT CRC16
    print("\n--- CCITT CRC16 ---")
    crc_ccitt = hand.crc16_ccitt(test_data)
    crc_low_ccitt, crc_high_ccitt = hand.calculate_crc16(test_data, "ccitt")
    print(f"CRC16 CCITT: 0x{crc_ccitt:04X}")
    print(f"CRC低字节: 0x{crc_low_ccitt:02X}")
    print(f"CRC高字节: 0x{crc_high_ccitt:02X}")
    
    # 测试校验验证
    print("\n--- 校验验证测试 ---")
    is_valid_modbus = hand.verify_crc16(test_data, crc_low, crc_high, "modbus")
    is_valid_ccitt = hand.verify_crc16(test_data, crc_low_ccitt, crc_high_ccitt, "ccitt")
    
    print(f"Modbus校验验证: {'通过' if is_valid_modbus else '失败'}")
    print(f"CCITT校验验证: {'通过' if is_valid_ccitt else '失败'}")
    
    # 测试错误数据
    print("\n--- 错误数据测试 ---")
    wrong_crc_low, wrong_crc_high = 0x00, 0x00
    is_valid_wrong = hand.verify_crc16(test_data, wrong_crc_low, wrong_crc_high, "modbus")
    print(f"错误CRC校验验证: {'通过' if is_valid_wrong else '失败'} (期望结果: 失败)")

def test_with_real_data():
    """使用真实数据测试CRC校验"""
    
    hand = InspireHandR()
    
    # 模拟一个完整的Modbus RTU帧
    # 设备ID + 功能码 + 寄存器地址 + 寄存器数量
    modbus_frame = [0x01, 0x03, 0x00, 0x00, 0x00, 0x0A]
    
    print("\n=== 真实Modbus数据测试 ===")
    print(f"Modbus帧: {[hex(x) for x in modbus_frame]}")
    
    # 计算CRC
    crc_low, crc_high = hand.calculate_crc16(modbus_frame, "modbus")
    print(f"计算的CRC: 0x{crc_high:02X} 0x{crc_low:02X}")
    
    # 构建完整帧（包含CRC）
    complete_frame = modbus_frame + [crc_low, crc_high]
    print(f"完整帧: {[hex(x) for x in complete_frame]}")
    
    # 验证CRC
    is_valid = hand.verify_crc16(modbus_frame, crc_low, crc_high, "modbus")
    print(f"CRC验证: {'通过' if is_valid else '失败'}")

def test_crc_online_verification():
    """与在线CRC计算器验证结果"""
    
    hand = InspireHandR()
    
    # 测试数据：01 03 06 0C 00 01
    test_data = [0x01, 0x03, 0x06, 0x0C, 0x00, 0x01]
    
    print("\n=== 在线验证测试 ===")
    print("测试数据: 01 03 06 0C 00 01")
    print("可以在 http://www.ip33.com/crc.html 验证结果")
    
    # Modbus CRC16
    crc_modbus = hand.crc16_modbus(test_data)
    print(f"Modbus CRC16: 0x{crc_modbus:04X}")
    
    # CCITT CRC16
    crc_ccitt = hand.crc16_ccitt(test_data)
    print(f"CCITT CRC16: 0x{crc_ccitt:04X}")

if __name__ == "__main__":
    test_crc_functions()
    test_with_real_data()
    test_crc_online_verification() 