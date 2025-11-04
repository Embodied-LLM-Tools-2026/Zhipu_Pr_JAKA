#!/usr/bin/env python3
# test_gripper_device.py
# 自动检测 ZX 夹爪串口设备并测试通信

import glob
import time
from gripper_controller import GripperController

# 枚举所有可能的串口设备
candidates = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
if not candidates:
    print("未发现任何串口设备 (/dev/ttyUSB* 或 /dev/ttyACM*)")
    exit(1)

print(f"发现串口设备: {candidates}")

for device in candidates:
    print(f"\n尝试连接设备: {device}")
    try:
        gripper = GripperController(device)
        print("尝试打开夹爪...")
        gripper.open()
        time.sleep(1)
        print("尝试关闭夹爪...")
        gripper.close()
        print(f"✅ 设备 {device} 通信正常！")
    except Exception as e:
        print(f"❌ 设备 {device} 通信失败: {e}")

print("测试完成。请观察夹爪动作和输出结果。")
