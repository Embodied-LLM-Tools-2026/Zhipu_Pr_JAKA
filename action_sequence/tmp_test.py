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