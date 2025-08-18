from curses import raw
import struct
import json
PACK_FMT_STR = '!BBHLH6s'

msg = None
req_id = 1
msg_type = 1004

msg_len = 0
json_str = json.dumps(msg)
if msg != {}:
    msg_len = len(json_str)
raw_msg = struct.pack(PACK_FMT_STR, 0x5A, 0x01, req_id, msg_len, msg_type, b'\x00\x00\x00\x00\x00\x00')

if msg != {}:
    raw_msg += bytearray(json_str, 'ascii')

print(raw_msg)


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
