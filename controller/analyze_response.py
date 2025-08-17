#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
分析Modbus RTU响应数据
"""

def crc16_modbus(data):
    """CRC16 Modbus校验函数"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc = crc >> 1
    return crc

def analyze_modbus_response(response_hex):
    """分析Modbus响应"""
    print("=== Modbus RTU响应分析 ===")
    print(f"原始数据: {response_hex}")
    
    # 解析各个字段
    device_id = int(response_hex[0], 16)
    function_code = int(response_hex[1], 16)
    byte_count = int(response_hex[2], 16)
    
    print(f"\n字段解析:")
    print(f"设备ID: 0x{device_id:02X} ({device_id})")
    print(f"功能码: 0x{function_code:02X} ({function_code})")
    print(f"字节数: 0x{byte_count:02X} ({byte_count})")
    
    # 解析数据
    data_bytes = []
    for i in range(3, 3 + byte_count):
        data_bytes.append(int(response_hex[i], 16))
    
    print(f"数据内容: {[hex(x) for x in data_bytes]}")
    
    # 计算数据值
    if len(data_bytes) == 2:
        data_value = (data_bytes[0] << 8) + data_bytes[1]
        print(f"数据值: 0x{data_value:04X} ({data_value})")
    
    # 解析CRC
    crc_low = int(response_hex[-2], 16)
    crc_high = int(response_hex[-1], 16)
    print(f"CRC校验码: 0x{crc_high:02X} 0x{crc_low:02X}")
    
    # 验证CRC
    data_for_crc = []
    for i in range(len(response_hex) - 2):  # 不包括CRC
        data_for_crc.append(int(response_hex[i], 16))
    
    calculated_crc = crc16_modbus(data_for_crc)
    calculated_low = calculated_crc & 0xFF
    calculated_high = (calculated_crc >> 8) & 0xFF
    
    print(f"\nCRC验证:")
    print(f"期望CRC: 0x{crc_high:02X} 0x{crc_low:02X}")
    print(f"计算CRC: 0x{calculated_high:02X} 0x{calculated_low:02X}")
    
    if calculated_low == crc_low and calculated_high == crc_high:
        print("✓ CRC校验通过！")
    else:
        print("✗ CRC校验失败！")
    
    return {
        'device_id': device_id,
        'function_code': function_code,
        'byte_count': byte_count,
        'data': data_bytes,
        'crc_valid': (calculated_low == crc_low and calculated_high == crc_high)
    }

# 分析您提供的响应数据
response_data = ['01', '03', '02', '00', '0a', '38', '43']
result = analyze_modbus_response(response_data)

print(f"\n=== 总结 ===")
print(f"这是一个来自设备 {result['device_id']} 的读保持寄存器响应")
print(f"功能码 {result['function_code']} 表示读保持寄存器")
print(f"返回了 {result['byte_count']} 个字节的数据")
print(f"数据值为: {result['data'][0] * 256 + result['data'][1]}")
print(f"CRC校验: {'通过' if result['crc_valid'] else '失败'}") 