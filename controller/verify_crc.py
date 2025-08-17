#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
验证CRC计算是否正确
"""

def crc16_modbus(data):
    """
    CRC16 Modbus校验函数
    使用多项式 0x8005 (x^16 + x^15 + x^2 + 1)
    初始值: 0xFFFF
    异或值: 0x0000
    """
    crc = 0xFFFF  # 初始值
    
    for byte in data:
        crc ^= byte  # 异或当前字节
        
        for _ in range(8):  # 处理8位
            if crc & 0x0001:  # 如果最低位为1
                crc = (crc >> 1) ^ 0xA001  # 右移1位并异或多项式
            else:
                crc = crc >> 1  # 右移1位
                
    return crc

def calculate_crc16(data):
    """计算CRC16校验码，返回低字节和高字节"""
    crc = crc16_modbus(data)
    return (crc & 0xFF, (crc >> 8) & 0xFF)

# 测试数据：01 03 06 0C 00 01
test_data = [0x01, 0x03, 0x06, 0x0C, 0x00, 0x01]
expected_crc_low = 0x44
expected_crc_high = 0x81

print("测试数据:", [hex(x) for x in test_data])
print("期望的CRC: 0x{:02X} 0x{:02X}".format(expected_crc_high, expected_crc_low))

# 计算CRC
crc_low, crc_high = calculate_crc16(test_data)
print("计算的CRC: 0x{:02X} 0x{:02X}".format(crc_high, crc_low))

# 验证
if crc_low == expected_crc_low and crc_high == expected_crc_high:
    print("✓ CRC计算正确！")
else:
    print("✗ CRC计算错误！")
    print("期望: 0x{:02X} 0x{:02X}".format(expected_crc_high, expected_crc_low))
    print("实际: 0x{:02X} 0x{:02X}".format(crc_high, crc_low))

# 显示完整的Modbus帧
complete_frame = test_data + [crc_low, crc_high]
print("完整Modbus帧:", [hex(x) for x in complete_frame]) 