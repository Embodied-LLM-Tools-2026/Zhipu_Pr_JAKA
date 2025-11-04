from pymodbus.client.sync import ModbusSerialClient

ZX_SLAVE_DEFAULT = 1
ZX_REG_OPEN   = 0x0029     # 01 06 00 29 00 01
ZX_REG_CLOSE  = 0x0028     # 01 06 00 28 00 01
ZX_REG_SYS    = 0x002A     # 01 06 00 2A 00 01
ZX_REG_POS    = 0x000A     # write 1..100
ZX_REG_FORCE  = 0x000B     # write 20..320
ZX_REG_FORCE_RD = 0x0101   # read 1 word 

ZX_POS_MIN, ZX_POS_MAX = 1, 100
ZX_FORCE_MIN, ZX_FORCE_MAX = 20, 320


def build_modbus_client(
    port: str, baud: int, parity: str, stopbits: int, timeout: float
) -> ModbusSerialClient:
    client = ModbusSerialClient(
        method="rtu",
        port=port,
        baudrate=baud,
        parity=parity,         # 'N'/'E'/'O'
        stopbits=stopbits,     # 1 or 2
        bytesize=8,
        timeout=timeout,
    )
    if not client.connect():
        raise RuntimeError("串口打开失败或无法建立 Modbus 连接")
    return client

class GripperController:
    def __init__(self, device, baud=115200, parity='N', stopbits=1, timeout=0.3, slave=1):
        self.device = device
        self.baud = baud
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.slave = slave

    def open(self):
        client = build_modbus_client(self.device, self.baud, self.parity, self.stopbits, self.timeout)
        try:
            rr = client.write_register(ZX_REG_OPEN, 1, unit=self.slave)
            if rr.isError():
                raise RuntimeError(rr)
            print("Gripper OPEN: OK")
        finally:
            client.close()

    def close(self):
        client = build_modbus_client(self.device, self.baud, self.parity, self.stopbits, self.timeout)
        try:
            rr = client.write_register(ZX_REG_CLOSE, 1, unit=self.slave)
            if rr.isError():
                raise RuntimeError(rr)
            print("Gripper CLOSE: OK")
        finally:
            client.close()

    def set_position(self, pos):
        pos = int(pos)
        if not (ZX_POS_MIN <= pos <= ZX_POS_MAX):
            raise ValueError(f"position must be {ZX_POS_MIN}..{ZX_POS_MAX}")
        client = build_modbus_client(self.device, self.baud, self.parity, self.stopbits, self.timeout)
        try:
            rr = client.write_register(ZX_REG_POS, pos, unit=self.slave)
            if rr.isError():
                raise RuntimeError(rr)
            print(f"Gripper SET-POS({pos}): OK")
        finally:
            client.close()

    def set_force(self, force):
        f = int(force)
        if not (ZX_FORCE_MIN <= f <= ZX_FORCE_MAX):
            raise ValueError(f"force must be {ZX_FORCE_MIN}..{ZX_FORCE_MAX}")
        client = build_modbus_client(self.device, self.baud, self.parity, self.stopbits, self.timeout)
        try:
            rr = client.write_register(ZX_REG_FORCE, f, unit=self.slave)
            if rr.isError():
                raise RuntimeError(rr)
            print(f"Gripper SET-FORCE({f}): OK")
        finally:
            client.close()

    def deliver(self, obj_name=None):
        print(f"递交物品: {obj_name if obj_name else ''}")
        self.open()
        # 可根据实际递交动作扩展

# 用法示例：
# gripper = GripperController('/dev/ttyUSB0')
# gripper.open()
# gripper.close()
# gripper.set_position(50)
# gripper.set_force(100)
# gripper.deliver('水')
