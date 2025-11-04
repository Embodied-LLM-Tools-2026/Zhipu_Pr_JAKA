#!/usr/bin/env python
# -*- coding:utf-8 -*-
#SHIJIANGAN

import time
import serial
import numpy as np

## 功能码
"""
0x03 读取寄存器
0x06 写入寄存器
0x10 写多个寄存器
"""

class GripperController:
    def __init__(self, port: str = "COM11", baudrate: int = 115200, gripper_id: int = 1):

        self.ser = serial.Serial(port, baudrate)
        self.ser.isOpen()
        self.gripper_id = gripper_id

    #把十六进制或十进制的数转成bytes
    def num2str(self,num):
        str = hex(num)
        str = str[2:4]
        if(len(str) == 1):
            str = '0'+ str
        # str = bytes.fromhex(str)
        str = bytes.fromhex(str)     
        #print(str)
        return str
    
    def data2bytes(self, data):
        # 将角度值转换为4位十六进制字符串，不足4位前面补0
        hex_str = f"{data:04X}"
        # 转换为bytes
        bytes_data = bytes.fromhex(hex_str)
        return bytes_data
   
    #求校验和
    def crc16_modbus(self, data, length=None):
        """
        CRC16 Modbus校验函数
        使用多项式 0x8005 (x^16 + x^15 + x^2 + 1)
        初始值: 0xFFFF
        异或值: 0x0000
        
        Args:
            data: 字节数组或列表
            length: 数据长度，如果为None则使用data的长度
            
        Returns:
            CRC16校验值 (16位整数)
        """
        if length is None:
            length = len(data)
            
        crc = 0xFFFF  # 初始值
        
        for i in range(length):
            crc ^= data[i]  # 异或当前字节
            
            for _ in range(8):  # 处理8位
                if crc & 0x0001:  # 如果最低位为1
                    crc = (crc >> 1) ^ 0xA001  # 右移1位并异或多项式
                else:
                    crc = crc >> 1  # 右移1位
                    
        return crc
    
    def crc16_ccitt(self, data, length=None):
        """
        CRC16 CCITT校验函数
        使用多项式 0x1021 (x^16 + x^12 + x^5 + 1)
        初始值: 0xFFFF
        异或值: 0x0000
        
        Args:
            data: 字节数组或列表
            length: 数据长度，如果为None则使用data的长度
            
        Returns:
            CRC16校验值 (16位整数)
        """
        if length is None:
            length = len(data)
            
        crc = 0xFFFF  # 初始值
        
        for i in range(length):
            crc ^= (data[i] << 8)  # 将当前字节左移8位后异或
            
            for _ in range(8):  # 处理8位
                if crc & 0x8000:  # 如果最高位为1
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF  # 左移1位并异或多项式
                else:
                    crc = (crc << 1) & 0xFFFF  # 左移1位
                    
        return crc
    
    def calculate_crc16(self, data, crc_type="modbus"):
        """
        计算CRC16校验码
        
        Args:
            data: 字节数组或列表
            crc_type: CRC类型，"modbus" 或 "ccitt"
            
        Returns:
            (crc_low, crc_high): CRC低字节和高字节
        """
        if crc_type.lower() == "modbus":
            crc = self.crc16_modbus(data)
        elif crc_type.lower() == "ccitt":
            crc = self.crc16_ccitt(data)
        else:
            raise ValueError("不支持的CRC类型，请使用 'modbus' 或 'ccitt'")
            
        # 返回低字节和高字节 (小端序)
        return (crc & 0xFF, (crc >> 8) & 0xFF)
    
    def verify_crc16(self, data, expected_crc_low, expected_crc_high, crc_type="modbus"):
        """
        验证CRC16校验码
        
        Args:
            data: 字节数组或列表
            expected_crc_low: 期望的CRC低字节
            expected_crc_high: 期望的CRC高字节
            crc_type: CRC类型，"modbus" 或 "ccitt"
            
        Returns:
            bool: 校验是否通过
        """
        calculated_low, calculated_high = self.calculate_crc16(data, crc_type)
        return (calculated_low == expected_crc_low) and (calculated_high == expected_crc_high)

    def homing(self):
        '''复位
        功能：回到最初位置
        '''
        self.setpos(20)
        self.run_gripper()
    
    def enable_gripper(self):
        '''功能：使能灵巧手
        指令帧长度：6Bytes
        指令号：0x50（CMD_MC_SET_DRVALL_SEEKPOS）
        数据内容：6 个驱动器的目标位置，每个位置为 2Bytes（小端模式低字节先发送），共12Bytes，目标位置的有效值为 0~2000，若为 0xFFFF （-1），则表示不需要设置该驱动器的目标
                位置，因此可单独设置某个驱动器的目标位置
        '''
        # INSERT_YOUR_CODE
        '''
        该命令为执行器使能相关命令，地址为 0x0100。该寄存器地址为可读写地址，寄存器
        数据为 0 时执行器处于失能状态，为 1 是执行器处于使能状态。
        发送数据：01 06 01 00 00 01 49 F6
        返回数据：01 06 01 00 00 01 49 F6
        '''
        # 构造指令帧
        b = [0]*8
        b[0] = self.gripper_id  # 设备ID，通常为1
        b[1] = 0x06             # 功能码
        b[2] = 0x01             # 寄存器高字节
        b[3] = 0x00             # 寄存器低字节
        b[4] = 0x00             # 数据高字节
        b[5] = 0x01             # 数据低字节（1为使能，0为失能）

        # 计算CRC16校验码
        crc_low, crc_high = self.calculate_crc16(b[0:6], "modbus")
        b[6] = crc_low
        b[7] = crc_high

        # 向串口发送数据
        putdata = b''
        for i in range(8):
            putdata = putdata + self.num2str(b[i])
        print("发送使能指令:", putdata.hex(" "))
        self.ser.write(putdata)
        time.sleep(0.2)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)
        print("使能响应:", getdata.hex(" "))

    def disable_gripper(self):
        '''功能：禁用灵巧手
        指令帧长度：6Bytes
        指令号：0x50（CMD_MC_SET_DRVALL_SEEKPOS）
        数据内容：6 个驱动器的目标位置，每个位置为 2Bytes（小端模式低字节先发送），共12Bytes，目标位置的有效值为 0~2000，若为 0xFFFF （-1），则表示不需要设置该驱动器的目标
                位置，因此可单独设置某个驱动器的目标位置
        '''
    # INSERT_YOUR_CODE
        '''
        该命令为执行器下使能相关命令，地址为 0x0100。该寄存器地址为可读写地址，寄存器
        数据为 0 时执行器处于失能状态，为 1 是执行器处于使能状态。
        发送数据：01 06 01 00 00 00 88 36
        返回数据：01 06 01 00 00 00 88 36
        '''
        # 构造指令帧
        b = [0]*8
        b[0] = self.gripper_id  # 设备ID，通常为1
        b[1] = 0x06             # 功能码
        b[2] = 0x01             # 寄存器高字节
        b[3] = 0x00             # 寄存器低字节
        b[4] = 0x00             # 数据高字节
        b[5] = 0x00             # 数据低字节（1为使能，0为失能）

        # 计算CRC16校验码
        crc_low, crc_high = self.calculate_crc16(b[0:6], "modbus")
        b[6] = crc_low
        b[7] = crc_high

        # 向串口发送数据
        putdata = b''
        for i in range(8):
            putdata = putdata + self.num2str(b[i])
        print("发送下使能指令:", putdata.hex(" "))
        self.ser.write(putdata)
        time.sleep(0.2)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)
        print("下使能响应:", getdata.hex(" "))

    def run_gripper(self):
        '''
        功能：触发临时区运动
        说明：向0x0108寄存器写1，触发执行器以0x0103~0x0106的参数运行到0x0102所设定的临时区运动位置。
        发送数据：01 06 01 08 00 01 C8 34
        返回数据：01 06 01 08 00 01 C8 34
        '''
        b = [0]*8
        b[0] = self.gripper_id  # 设备ID，通常为1
        b[1] = 0x06             # 功能码
        b[2] = 0x01             # 寄存器高字节
        b[3] = 0x08             # 寄存器低字节
        b[4] = 0x00             # 数据高字节
        b[5] = 0x01             # 数据低字节（1为触发，0为不触发）

        # 计算CRC16校验码
        crc_low, crc_high = self.calculate_crc16(b[0:6], "modbus")
        b[6] = crc_low
        b[7] = crc_high

        # 向串口发送数据
        putdata = b''
        for i in range(8):
            putdata = putdata + self.num2str(b[i])
        print("发送临时区运动触发指令:", putdata.hex(" "))
        self.ser.write(putdata)
        time.sleep(0.2)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)
        print("临时区运动触发响应:", getdata.hex(" "))

    def is_force_sensor_reached(self):
        '''
        判断力传感器是否到达
        '''
        b = [0]*8
        b[0] = self.gripper_id  # 设备ID，通常为1
        b[1] = 0x03             # 功能码
        b[2] = 0x06             # 寄存器高字节
        b[3] = 0x01             # 寄存器低字节
        b[4] = 0x00             # 数据高字节
        b[5] = 0x01             # 数据低字节（读取1个寄存器）

        # 计算CRC16校验码
        crc_low, crc_high = self.calculate_crc16(b[0:6], "modbus")
        b[6] = crc_low
        b[7] = crc_high

        # 向串口发送数据
        putdata = b''
        for i in range(8):
            putdata = putdata + self.num2str(b[i])
        print("发送力矩到达判断指令:", putdata.hex(" "))
        self.ser.write(putdata)
        time.sleep(0.2)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)
        print("力矩到达判断响应:", getdata.hex(" "))

        # 解析返回数据
        if len(getdata) < 7:
            print("返回数据长度不足，无法判断力矩是否到达")
            return False

        # 返回数据格式：01 03 02 00 00 B8 44
        # Data[3] = 0x00, Data[4] = 0x00，表示未到达；0x00, 0x01表示到达
        try:
            data_list = list(getdata)
            if data_list[0] != self.gripper_id or data_list[1] != 0x03:
                print("返回数据格式错误")
                return False
            # 数据长度为2，数据在data_list[3]和data_list[4]
            value = (data_list[3] << 8) | data_list[4]
            if value == 1:
                return True
            else:
                return False
        except Exception as e:
            print("解析力矩到达判断响应异常:", e)
            return False

    def is_position_reached(self):
        '''
        判断位置是否到达（0x0602）
        说明：读取0x0602寄存器，判断位置是否到达。寄存器值为0表示未到达，为1表示已到达。
        发送数据：01 03 06 02 00 01 25 42
        返回数据：01 03 02 00 01 79 84
        返回值：True（到达），False（未到达）
        '''
        b = [0]*8
        b[0] = self.gripper_id  # 设备ID，通常为1
        b[1] = 0x03             # 功能码
        b[2] = 0x06             # 寄存器高字节
        b[3] = 0x02             # 寄存器低字节
        b[4] = 0x00             # 数据高字节
        b[5] = 0x01             # 数据低字节（读取1个寄存器）

        # 计算CRC16校验码
        crc_low, crc_high = self.calculate_crc16(b[0:6], "modbus")
        b[6] = crc_low
        b[7] = crc_high

        # 向串口发送数据
        putdata = b''
        for i in range(8):
            putdata = putdata + self.num2str(b[i])
        print("发送位置到达判断指令:", putdata.hex(" "))
        self.ser.write(putdata)
        time.sleep(0.2)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)
        print("位置到达判断响应:", getdata.hex(" "))

        # 解析返回数据
        if len(getdata) < 7:
            print("返回数据长度不足，无法判断位置是否到达")
            return False

        # 返回数据格式：01 03 02 00 01 79 84
        # Data[3] = 0x00, Data[4] = 0x01，表示已到达
        try:
            data_list = list(getdata)
            if data_list[0] != self.gripper_id or data_list[1] != 0x03:
                print("返回数据格式错误")
                return False
            # 数据长度为2，数据在data_list[3]和data_list[4]
            value = (data_list[3] << 8) | data_list[4]
            if value == 1:
                return True
            else:
                return False
        except Exception as e:
            print("解析位置到达返回数据异常:", e)
            return False


    def setpos(self,width):
        '''功能：主控单元设置灵巧手中 6 个直线驱动器的目标位置，使灵巧手完成相应的手势
                动作。灵巧手中的 6 个直线伺服驱动器的 ID 号为 1-6，其中小拇指的 ID 为 1、无名指的 ID
                为 2、中指的 ID 为 3、食指的 ID 为 4、大拇指弯曲指关节 ID 为 5、大拇指旋转指关节 ID
                为 6。
        指令帧长度：18Bytes
        指令号：0x50（CMD_MC_SET_DRVALL_SEEKPOS）
        数据内容：6 个驱动器的目标位置，每个位置为 2Bytes（小端模式低字节先发送），共12Bytes，目标位置的有效值为 0~2000，若为 0xFFFF （-1），则表示不需要设置该驱动器的目标
                位置，因此可单独设置某个驱动器的目标位置
        '''
        datanum = 0x0C
        b = [0]*13
        # 设备ID
        b[0] = self.gripper_id
        b[1] = 0x10

        # 寄存器地址
        b[2] = 0x01
        b[3] = 0x02

        # 寄存器数量
        b[4] = 0x00
        b[5] = 0x02

        # 字节数
        b[6] = 0x04

        # 数据
        # 使用函数将20mm转换为4个字节并赋值到b[7]~b[10]
        def mm_to_bytes(val_mm):
            # 这里假设1mm对应100单位，20mm即2000
            pos = int(val_mm * 100)
            # 转换为2字节（小端），高位在前
            high = (pos >> 8) & 0xFF
            low = pos & 0xFF
            return [0x00, 0x00, high, low]

        b[7], b[8], b[9], b[10] = mm_to_bytes(width)

        # 校验码
        data_for_crc = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10]]
        crc_low, crc_high = self.calculate_crc16(data_for_crc, "modbus")
        b[11] = crc_low
        b[12] = crc_high

        #向串口发送数据
        putdata = b''
        
        for i in range(1,14):
            putdata = putdata + self.num2str(b[i-1])
        # print(putdata)
        self.ser.write(putdata)

        # print('读取目标位置发送的数据：%s'%putdata.hex(" "))
        time.sleep(0.5)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)
        # print('读取目标位置返回的数据：%s'%getdata.hex(" "))

        Data = getdata.hex(" ").split(" ")
        self.run_gripper()
        while self.is_force_sensor_reached() == False and self.is_position_reached() == False:
            time.sleep(0.2)
            print(self.is_force_sensor_reached())
            print(self.is_position_reached())
        return
    # INSERT_YOUR_CODE
    def set_temp_torque(self, percent):
        '''
        设置执行器临时运动力矩百分比
        该命令为执行器临时运动力矩，地址为0x0105。寄存器内数据为执行器最大力矩（0x0306）的百分比。
        发送数据：01 06 01 05 00 64 99 DC
        返回数据：01 06 01 05 00 64 99 DC
        参数:
            percent: 力矩百分比（0~100），如100表示100%
        '''
        if percent < 0 or percent > 100:
            print("力矩百分比应在0~100之间")
            return False

        b = [0]*8
        b[0] = self.gripper_id  # 设备ID，通常为1
        b[1] = 0x06             # 功能码
        b[2] = 0x01             # 寄存器高字节
        b[3] = 0x05             # 寄存器低字节
        b[4] = 0x00             # 数据高字节
        b[5] = percent & 0xFF   # 数据低字节（百分比）

        # 计算CRC16校验码
        crc_low, crc_high = self.calculate_crc16(b[0:6], "modbus")
        b[6] = crc_low
        b[7] = crc_high

        # 向串口发送数据
        putdata = b''
        for i in range(8):
            putdata = putdata + self.num2str(b[i])
        print("发送临时区运动力矩设定指令:", putdata.hex(" "))
        self.ser.write(putdata)
        time.sleep(0.2)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)
        print("临时区运动力矩设定响应:", getdata.hex(" "))

        # 简单校验返回
        if getdata == putdata:
            print("临时区运动力矩设定成功")
            return True
        else:
            print("临时区运动力矩设定失败")
            return False

    # def get_joint_position(self):
    #     '''
    #     功能：实时反馈位置信息，读取执行器的实时位置
    #     说明：读取寄存器地址0x0609和0x060A，返回执行器实时位置信息
    #     位置 = (high << 16) + low
    #     发送数据：01 03 06 09 00 02 14 81
    #     返回数据：01 03 04 00 00 00 00 FA 33
    #     '''
    #     b = [0] * 8
    #     # 设备ID
    #     b[0] = self.gripper_id if hasattr(self, 'gripper_id') else 0x01
    #     # 功能码
    #     b[1] = 0x03
    #     # 寄存器地址高字节、低字节
    #     b[2] = 0x06
    #     b[3] = 0x09
    #     # 数据：寄存器数量高字节、低字节（读取2个寄存器）
    #     b[4] = 0x00
    #     b[5] = 0x02
    #     # CRC16校验
    #     crc_low, crc_high = self.calculate_crc16(b[0:6], "modbus")
    #     b[6] = crc_low
    #     b[7] = crc_high

    #     # 发送数据
    #     putdata = b''
    #     for i in range(8):
    #         putdata = putdata + self.num2str(b[i])
    #     print("发送实时反馈位置信息指令:", putdata.hex(" "))
    #     self.ser.write(putdata)
    #     time.sleep(0.2)
    #     read_num = self.ser.inWaiting()
    #     getdata = self.ser.read(read_num)
    #     print("实时反馈位置信息响应:", getdata.hex(" "))

    #     # 解析返回数据
    #     if len(getdata) < 9:
    #         print("返回数据长度不足，无法解析实时位置信息")
    #         return None

    #     try:
    #         data_list = list(getdata)
    #         # 检查设备ID和功能码
    #         if data_list[0] != b[0] or data_list[1] != 0x03:
    #             print("返回数据格式错误")
    #             return None
    #         # 数据长度
    #         data_len = data_list[2]
    #         if data_len != 4:
    #             print("返回数据长度字段错误")
    #             return None
    #         # 位置 = (high << 16) + low
    #         # high: data_list[3] << 8 | data_list[4]
    #         # low:  data_list[5] << 8 | data_list[6]
    #         high = (data_list[3] << 8) | data_list[4]
    #         low = (data_list[5] << 8) | data_list[6]
    #         position = (high << 16) + low
    #         print(f"实时反馈位置信息：{position} (high: {high:#06x}, low: {low:#06x})")
    #         return position
    #     except Exception as e:
    #         print("解析实时反馈位置信息异常:", e)
    #         return None



if __name__ == "__main__":
    hand = GripperController(port="COM11", baudrate=115200)
    # hand.homing()
    hand.set_temp_torque(20)
    hand.setpos(0)
    # hand.setpos(120)
    # a = hand.is_force_sensor_reached()
    # print(a)
    # time.sleep(1)
    # a = hand.is_force_sensor_reached()
    # print(a)
    # time.sleep(1)
    # a = hand.is_force_sensor_reached()
    # print(a)
    # time.sleep(1)
    # a = hand.is_force_sensor_reached()
    # print(a)
    # time.sleep(1)
    # hand.setpos(20)
    # hand.get_joint_position()