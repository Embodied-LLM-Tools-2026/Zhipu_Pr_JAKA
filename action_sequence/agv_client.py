import socket
import json
import time
import struct

PACK_FMT_STR = '!BBHLH6s'

class AGVClient:
    """AGV客户端控制类"""
    
    def __init__(self, ip='192.168.192.5', timeout=5):
        """
        初始化AGV客户端
        
        Args:
            ip (str): AGV服务器IP地址
            port (int): AGV服务器端口
            timeout (int): 连接超时时间(秒)
        """
        self.ip = ip
        self.timeout = timeout
        self.socket_stater = None
        self.socket_controller = None
        self.socket_navigator = None
        self.connected = False
    
    def connect(self):
        """连接到AGV服务器"""
        try:
            self.socket_stater = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_stater.connect((self.ip, 19204))
            self.socket_stater.settimeout(self.timeout)

            self.socket_controller = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_controller.connect((self.ip, 19205))
            self.socket_controller.settimeout(self.timeout)

            self.socket_navigator = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_navigator.connect((self.ip, 19206))
            self.socket_navigator.settimeout(self.timeout)

            self.connected = True
            print(f"成功连接到AGV服务器 {self.ip}")
            return True

        except Exception as e:
            print(f"连接AGV服务器失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开与AGV服务器的连接"""
        if self.socket_stater:
            self.socket_stater.close()
            self.socket_stater = None
        if self.socket_controller:
            self.socket_controller.close()
            self.socket_controller = None
        if self.socket_navigator:
            self.socket_navigator.close()
            self.socket_navigator = None
        if self.socket_stater is None and self.socket_controller is None and self.socket_navigator is None:
            self.connected = False
            print("已断开AGV服务器连接")
    
    def _pack_message(self, req_id, msg_type, msg={}):
        """
        打包消息
        
        Args:
            req_id (int): 请求ID
            msg_type (int): 消息类型
            msg (dict): 消息内容
            
        Returns:
            bytes: 打包后的消息
        """
        msg_len = 0
        json_str = json.dumps(msg)
        if msg != {}:
            msg_len = len(json_str)
        
        raw_msg = struct.pack(PACK_FMT_STR, 0x5A, 0x01, req_id, msg_len, msg_type, b'\x00\x00\x00\x00\x00\x00')
        
        if msg != {}:
            raw_msg += bytearray(json_str, 'ascii')
        
        return raw_msg
    
    def _recv_and_unpack_response(self, data, socket_type):
        """
        解析响应数据
        
        Args:
            data (bytes): 响应数据
            
        Returns:
            tuple: (header, json_data)
        """
        if len(data) < 16:
            raise ValueError("响应数据包头部错误")
        
        header = struct.unpack(PACK_FMT_STR, data[:16])
        json_data_len = header[3]
        
        # 读取JSON数据
        json_data = b''
        remaining_len = json_data_len
        read_size = 1024
        
        while remaining_len > 0:
            if socket_type == 0:
                recv_data = self.socket_stater.recv(min(read_size, remaining_len))
            elif socket_type == 1:
                recv_data = self.socket_controller.recv(min(read_size, remaining_len))
            elif socket_type == 2:
                recv_data = self.socket_navigator.recv(min(read_size, remaining_len))
            else:
                raise ValueError("无效的socket类型")
            json_data += recv_data
            remaining_len -= len(recv_data)
        
        return header, json_data
    
    
    def send_message(self, msg_type, msg_data=None, req_id=1, socket_type=0):
        """
        发送自定义消息
        
        Args:
            msg_type (int): 消息类型
            msg_data (dict): 消息数据
            req_id (int): 请求ID
            socket_type (int): 0 stater读取状态, 1 controller控制, 2 navigator导航, 默认controller
            
        Returns:
            dict: 响应数据
        """
        if not self.connected:
            print("未连接到AGV服务器")
            return None
        
        try:
            packed_msg = self._pack_message(req_id, msg_type, msg_data)
            if socket_type == 0:
                self.socket_stater.send(packed_msg)
                response_header = self.socket_stater.recv(16)
            elif socket_type == 1:
                self.socket_controller.send(packed_msg)
                response_header = self.socket_controller.recv(16)
            elif socket_type == 2:
                self.socket_navigator.send(packed_msg)
                response_header = self.socket_navigator.recv(16)
            else:
                raise ValueError("无效的socket类型")
            
            header, json_data = self._recv_and_unpack_response(response_header, socket_type)
            response_json = json.loads(json_data.decode('ascii'))
            return response_json
            
        except Exception as e:
            print(f"发送自定义消息失败: {e}")
            return None
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


# def packMasg(reqId, msgType, msg={}):
#     msgLen = 0
#     jsonStr = json.dumps(msg)
#     if (msg != {}):
#         msgLen = len(jsonStr)
#     rawMsg = struct.pack(PACK_FMT_STR, 0x5A, 0x01, reqId, msgLen,msgType, b'\x00\x00\x00\x00\x00\x00')
#     print("{:02X} {:02X} {:04X} {:08X} {:04X}"
#     .format(0x5A, 0x01, reqId, msgLen, msgType))

#     if (msg != {}):
#         rawMsg += bytearray(jsonStr,'ascii')
#         print(msg)

#     return rawMsg


def main():
    # 使用新的控制类
    print("=== 使用AGVClient控制类 ===")
    with AGVClient(ip='192.168.192.5') as agv:
        response = agv.send_message(1004)
        if response:
            print("任务发送成功，响应内容：")
            print(response)
        else:
            print("任务发送失败")
    
    # print("\n=== 原始代码测试 ===")
    # so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # so.connect((IP, Port))
    # so.settimeout(5)
    # test_msg = packMasg(1,1110,{"task_ids":["SEER78914"]})
    # print("\n\nreq:")
    # print(' '.join('{:02X}'.format(x) for x in test_msg))
    # so.send(test_msg)

    # dataall = b''
    # # while True:
    # print('\n\n\n')
    # try:
    #     data = so.recv(16)
    # except socket.timeout:
    #     print('timeout')
    #     so.close
    # jsonDataLen = 0
    # backReqNum = 0
    # if(len(data) < 16):
    #     print('pack head error')
    #     print(data)
    #     so.close()
    # else:
    #     header = struct.unpack(PACK_FMT_STR, data)
    #     print("{:02X} {:02X} {:04X} {:08X} {:04X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}       length: {}"
    #     .format(header[0], header[1], header[2], header[3], header[4],
    #     header[5][0], header[5][1], header[5][2], header[5][3], header[5][4], header[5][5],
    #     header[3]))
    #     jsonDataLen = header[3]
    #     backReqNum = header[4]
    # dataall += data
    # data = b''
    # readSize = 1024
    # try:
    #     while (jsonDataLen > 0):
    #         recv = so.recv(readSize)
    #         data += recv
    #         jsonDataLen -= len(recv)
    #         if jsonDataLen < readSize:
    #             readSize = jsonDataLen
    #     print(json.dumps(json.loads(data), indent=1))
    #     dataall += data
    #     print(' '.join('{:02X}'.format(x) for x in dataall))
    # except socket.timeout:
    #     print('timeout')

    # so.close()


if __name__ == '__main__':
    main()