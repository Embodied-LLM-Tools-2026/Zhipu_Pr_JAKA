#!/usr/bin/env python3
# test_execute_from_csv.py
import rclpy
from rclpy.node import Node
from action_sequence.taskexecutor import TaskExecutor
from action_sequence.gripper_controller import GripperController

class DummyNavigator:
    def navigate_to_location(self, loc):
        print(f"[DummyNavigator] 导航到: {loc}")
    def wait_until_navigation_complete(self):
        print("[DummyNavigator] 导航完成")

def main():
    rclpy.init()
    node = rclpy.create_node('test_executor_node')
    navigator = DummyNavigator()
    gripper = GripperController('/dev/ttyUSB0')  # 根据实际串口修改
    executor = TaskExecutor(navigator, gripper, node)
    # executor.execute_from_csv('/home/sht/DIJA/Pr/action_sequence/action_csv/get.csv')  # 根据实际CSV路径修改
    executor.execute_from_csv('/home/sht/DIJA/Pr/action_sequence/action_csv/servo_record_bhc_20251029_140852.csv')  # 根据实际CSV路径修改
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
