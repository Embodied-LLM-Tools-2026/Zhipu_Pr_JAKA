#!/usr/bin/env python
# -*- coding:utf-8 -*-
#SHIJIANGAN

import time
import serial
import numpy as np
import serial.tools.list_ports
import minimalmodbus

class InspireHandR:
    def __init__(self, port: str = "COM11", baudrate: int = 115200, hand_id: int = 1):

        self.ser = serial.Serial(port, baudrate)
        self.ser.isOpen()
        self.hand_id = hand_id
        self.reset() # 手部复原
        #self.instrument = minimalmodbus.Instrument(self.port, 1)
    
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
   
    #求校验和,是除应答帧头外其余数据的累加和的低字节。
    def checknum(self,data,leng):
        result = 0
        for i in range(2,leng):
            result += data[i]
            
        result = result&0xff
        #print(result)
        return result

    def reset(self):
        '''复位
        功能：回到最初位置
        '''
        pos1 = 1000 #小拇指伸直0，弯曲2000
        pos2 = 1000 #无名指伸直0，弯曲2000
        pos3 = 1000 #中指伸直0，弯曲2000
        pos4 = 1000 #食指伸直0，弯曲2000
        pos5 = 1000 #大拇指伸直0，弯曲2000
        pos6 = 1000 #大拇指转向掌心 2000
        self.setpos(pos1,pos2,pos3,pos4,pos5,pos6) 
        return
    
       
    def get_target_position(self):
        '''读取目标位置
        功能：主控单元读取灵巧手当前 6 个驱动器的目标位置（0--2000）。
        指令帧长度：6Bytes
        指令号：0xD0（CMD_MC_READ_DRVALL_SEEKPOS）
        数据内容：无
        '''
        datanum = 0x01
        b = [0]*6
        #包头
        b[0] = 0xEB
        b[1] = 0x90

        #手的id号
        b[2] = self.hand_id

        #数据体长度
        b[3] = datanum

        #指令号
        b[4] = 0xD0

        #校验和
        b[5] = self.checknum(b, 5+datanum)
        
        #向串口发送数据
        putdata = b''
        
        for i in range(1,datanum+6):
            putdata = putdata + self.num2str(b[i-1])
        self.ser.write(putdata)

        # print('读取目标位置发送的数据：%s'%putdata.hex(" "))
        time.sleep(0.5)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)
        # print('读取目标位置返回的数据：%s'%getdata.hex(" "))

        Data = getdata.hex(" ").split(" ")
        setpos = [0.0]*6
        for i in range(1,12):
            if i%2 == 0:
                continue
            else:
                s1 = Data[5+i-1]
                s2 = Data[5+i]
                s = s2 + s1
                setpos[int((i-1)/2)] = int(s, 16)
        print('目标位置为：(%.2f, %.2f, %.2f, %.2f, %.2f, %.2f)'%(setpos[0], setpos[1], setpos[2], setpos[3], setpos[4], setpos[5]))

        
    def get_force_sensor(self):        
        '''功能：主控单元读取灵巧手指尖抓力传感器的原始值（0--4095）。
        指令帧长度：6Bytes
        指令号：0xD4（CMD_MC_READ_DRVALL_YBP_RAWDATA）
        数据内容：无。
        '''
        datanum = 0x01
        b = [0]*6
        #包头
        b[0] = 0xEB
        b[1] = 0x90

        #手的id号
        b[2] = self.hand_id

        #数据体长度
        b[3] = datanum

        #指令号
        b[4] = 0xD4

        #校验和
        b[5] = self.checknum(b, 5+datanum)
        
        #向串口发送数据
        putdata = b''
        
        for i in range(1,datanum+6):
            putdata = putdata + self.num2str(b[i-1])
        self.ser.write(putdata)

        time.sleep(0.5)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)

        Data = getdata.hex(" ").split(" ")
        setpower = np.zeros(5)
        for i in range(1,10):
            if i%2 == 0:
                continue
            else:
                s1 = Data[5+i-1]
                s2 = Data[5+i]
                s = s2 + s1
                setpower[int((i-1)/2)] = int(s, 16)
        # print('力传感器信息（依次为小指至大拇指的传感器的原始值）：(%.2f, %.2f, %.2f, %.2f, %.2f)'%(setpower[0], setpower[1], setpower[2], setpower[3], setpower[4]))
        return setpower

    def set_default_speed(self, speed1, speed2, speed3, speed4, speed5, speed6):
        '''功能：设置灵巧手六个自由度的上电初始速度（0-1000）。
        寄存器地址：从1032开始。
        '''
        if speed1 < 0 or speed1 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed2 < 0 or speed2 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed3 < 0 or speed3 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed4 < 0 or speed4 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed5 < 0 or speed5 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed6 < 0 or speed6 > 1000:
            print('数据超出正确范围：0-1000')
            return
        
        datanum = 0x0C
        b = [0]*(datanum + 8)
        #包头
        b[0] = 0xEB
        b[1] = 0x90

        #hand_id号
        b[2] = self.hand_id

        #数据体长度
        b[3] = 0x0F
        
        #写操作
        b[4] = 0x12

        #寄存器地址
        b[5] = 0xF2 # 低八位 (1032 = 0x0408)
        b[6] = 0x05 # 高八位
        
        #数据
        b[7] = self.data2bytes(speed1)[1]
        b[8] = self.data2bytes(speed1)[0]
        
        b[9] = self.data2bytes(speed2)[1]
        b[10] = self.data2bytes(speed2)[0]
        
        b[11] = self.data2bytes(speed3)[1]
        b[12] = self.data2bytes(speed3)[0]
        
        b[13] = self.data2bytes(speed4)[1]
        b[14] = self.data2bytes(speed4)[0]
        
        b[15] = self.data2bytes(speed5)[1]
        b[16] = self.data2bytes(speed5)[0]
        
        b[17] = self.data2bytes(speed6)[1]
        b[18] = self.data2bytes(speed6)[0]
        
        #校验和
        b[19] = self.checknum(b,datanum+7)

        
        #向串口发送数据
        putdata = b''
        
        for i in range(1 ,datanum+9):
            putdata = putdata + self.num2str(b[i-1])
        print("发送写入指令:", putdata.hex(" "))
        self.ser.write(putdata)
        time.sleep(0.5)
        read_num = self.ser.inWaiting()
        getdata= self.ser.read(read_num)
        print("写入响应:", getdata.hex(" "))

        return
    
    def setpos(self,pos1,pos2,pos3,pos4,pos5,pos6):
        '''功能：主控单元设置灵巧手中 6 个直线驱动器的目标位置，使灵巧手完成相应的手势
                动作。灵巧手中的 6 个直线伺服驱动器的 ID 号为 1-6，其中小拇指的 ID 为 1、无名指的 ID
                为 2、中指的 ID 为 3、食指的 ID 为 4、大拇指弯曲指关节 ID 为 5、大拇指旋转指关节 ID
                为 6。
        指令帧长度：18Bytes
        指令号：0x50（CMD_MC_SET_DRVALL_SEEKPOS）
        数据内容：6 个驱动器的目标位置，每个位置为 2Bytes（小端模式低字节先发送），共12Bytes，目标位置的有效值为 0~2000，若为 0xFFFF （-1），则表示不需要设置该驱动器的目标
                位置，因此可单独设置某个驱动器的目标位置
        '''
        global hand_id
        if pos1 <-1 or pos1 >2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos2 <-1 or pos2 >2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos3 <-1 or pos3 >2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos4 <-1 or pos4 >2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos5 <-1 or pos5 >2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos6 <-1 or pos6 >2000:
            print('数据超出正确范围：-1-2000')
            return
        
        datanum = 0x0C
        b = [0]*(datanum + 8)
        #包头
        b[0] = 0xEB
        b[1] = 0x90

        #hand_id号
        b[2] = self.hand_id

        #数据体长度
        b[3] = datanum + 3
         
        #写操作
        b[4] = 0x12

        #寄存器地址
        b[5] = 0xCE # 低八位
        b[6] = 0x05 # 高八位
        
        #数据
        b[7] = self.data2bytes(pos1)[1]
        b[8] = self.data2bytes(pos1)[0]
        
        b[9] = self.data2bytes(pos2)[1]
        b[10] = self.data2bytes(pos2)[0]
        
        b[11] = self.data2bytes(pos3)[1]
        b[12] = self.data2bytes(pos3)[0]
        
        b[13] = self.data2bytes(pos4)[1]
        b[14] = self.data2bytes(pos4)[0]
        
        b[15] = self.data2bytes(pos5)[1]
        b[16] = self.data2bytes(pos5)[0]
        
        b[17] = self.data2bytes(pos6)[1]
        b[18] = self.data2bytes(pos6)[0]
        
        #校验和
        b[19] = self.checknum(b,datanum+7)

        
        #向串口发送数据
        putdata = b''
        
        for i in range(1 ,datanum+9):
            putdata = putdata + self.num2str(b[i-1])
        print("发送写入指令:", putdata.hex(" "))
        self.ser.write(putdata)
        time.sleep(0.5)
        read_num = self.ser.inWaiting()
        getdata= self.ser.read(read_num)
        print("写入响应:", getdata.hex(" "))
 
        
        return

    def get_joint_position(self):
        '''功能：主控单元读取灵巧手 6 个直线驱动器的实际位置（0--2000）。
        指令帧长度：6Bytes
        指令号：0xD1（CMD_MC_READ_DRVALL_ACTPOS）
        数据内容：无。
        '''
        datanum = 0x04
        b = [0]*9
        #包头
        b[0] = 0xEB
        b[1] = 0x90

        #手的id号
        b[2] = self.hand_id

        #数据体长度
        b[3] = datanum

        #读寄存器
        b[4] = 0x11 

        # 寄存器起始位置低8位
        b[5] = 0x0A
        # 寄存器起始位置高8位
        b[6] = 0x06

        #指令号
        b[7] = 0x0C

        #校验和
        b[8] = self.checknum(b, 8)
        print(b)
        #向串口发送数据
        putdata = b''
        
        for i in range(1,10):
            putdata = putdata + self.num2str(b[i-1])
        self.ser.write(putdata)
        print("发送读取指令 get_joint_position:", putdata.hex(" "))
        time.sleep(0.5)
        read_num = self.ser.inWaiting()
        getdata = self.ser.read(read_num)

        Data = getdata.hex(" ").split(" ")
        print(Data)
        # Joint_position = np.zeros(5)
        # for i in range(1,10):
        #     if i%2 == 0:
        #         continue
        #     else:
        #         s1 = Data[5+i-1]
        #         s2 = Data[5+i]
        #         s = s2 + s1
        #         setpower[int((i-1)/2)] = int(s, 16)
        # # print('力传感器信息（依次为小指至大拇指的传感器的原始值）：(%.2f, %.2f, %.2f, %.2f, %.2f)'%(setpower[0], setpower[1], setpower[2], setpower[3], setpower[4]))
        # return setpower

    def grasp(self):
        '''功能：使用灵巧手指尖抓力传感器实现物体抓取。
        数据内容：无。
        '''
        i = [2000,2000,2000,2000,2000,2000] # 开始时手是张开的
        force_former = self.get_force_sensor() # 读取初始压力
        T = 19 # 迭代次数
        while T:
            T -= 1
            for index in range(len(i)):
                if i[index] > 0:
                    i[index] -= 100
            self.setpos(i[0],i[1],i[2],i[3],i[4],i[5])
            force = self.get_force_sensor()
            poor = force - force_former # 压力差，判断是否抓紧了
            below_threshold_id = np.argwhere(poor>1500) # 得到压力差大于1500的手指
            print(below_threshold_id)
            if below_threshold_id.size != 0: 
                for k in range(below_threshold_id.size): # 压力差大于1500的手指赋值-1
                    if int(below_threshold_id[k]) == 4:
                        i[4] = -1
                        i[5] = -1
                    else:
                        i[int(below_threshold_id[k])] = -1

            elif below_threshold_id.size == 5:
                break
    
    
    def close(self):
        self.ser.close()

if __name__ == "__main__":
    hand = InspireHandR(port="COM12", baudrate=115200, hand_id=2)
    hand.set_default_speed(100,100,100,100,100,100)
    time.sleep(3)
    hand.setpos(500,500,500,500,500,500)
    time.sleep(3)
    # hand.get_joint_position()
    # hand.setpos(1000,0,0,1000,500,500)
    # hand.close()