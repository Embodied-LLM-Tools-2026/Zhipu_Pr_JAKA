#!/usr/bin/env python3
"""
测试新架构的 Navigate 类
演示：长连接 socket + 一个后台循环更新多个全局变量
"""

import sys
import time
from pathlib import Path

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action_sequence.navigate import Navigate

def test_navigate_socket():
    """测试 socket 长连接和全局状态更新"""
    
    print("=" * 60)
    print("🧪 测试 Navigate 类的 Socket 长连接架构")
    print("=" * 60)
    
    # 创建实例
    navigator = Navigate()
    
    # 启动后台监控（使用 socket 长连接）
    print("\n1️⃣ 启动后台监控线程...")
    navigator.start_pose_monitoring(poll_interval=0.5)  # 0.5s 查询一次
    
    # 等待几次查询
    print("\n2️⃣ 等待数据更新（10 秒）...\n")
    for i in range(10):
        time.sleep(1)
        
        # 查询全局变量（无需重新连接）
        pose = navigator.get_current_pose()
        nav_state = navigator.get_navigation_state()
        
        print(f"[{i+1}] 位置: θ={pose['theta']:.2f}°, x={pose['x']:.2f}m, y={pose['y']:.2f}m")
        print(f"    导航状态: {nav_state['move_status']} | 正在导航: {nav_state['is_navigating']}")
        
        # 演示完整响应
        if i == 0:
            resp = navigator.get_last_status_response()
            if resp:
                print(f"    完整响应: {resp}")
    
    # 停止监控
    print("\n3️⃣ 停止监控...\n")
    navigator.stop_pose_monitoring()
    
    print("=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    print("\n📌 架构优势:")
    print("  ✓ 一条 socket 长连接，多次查询（高效）")
    print("  ✓ 后台循环自动更新所有全局变量")
    print("  ✓ 其他模块直接读取全局变量（无需重复查询）")
    print("  ✓ 自动重连机制（连接断开自动恢复）")

if __name__ == "__main__":
    test_navigate_socket()
