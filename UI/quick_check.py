#!/usr/bin/env python3
"""快速检查：后端是否正确存储和返回位置数据"""
import requests
import sys

UI_URL = "http://127.0.0.1:8000"

print("\n" + "="*60)
print("🔍 快速诊断：AGV 位置更新问题")
print("="*60)

# 测试 1: 连接性
print("\n1️⃣  检查连接...")
try:
    resp = requests.get(UI_URL, timeout=2)
    print("   ✅ UI 服务运行正常")
except Exception as e:
    print(f"   ❌ 无法连接到 UI 服务: {e}")
    print(f"   💡 请先启动: cd /home/sht/DIJA/Pr/UI && python robot_ui_demo.py")
    sys.exit(1)

# 测试 2: 推送位置
print("\n2️⃣  推送位置数据...")
test_data = {"theta": 1.5708, "x": 100.5, "y": 200.3}
try:
    resp = requests.post(f"{UI_URL}/api/agv/pose/update", json=test_data, timeout=2)
    if resp.status_code == 200:
        print(f"   ✅ 推送成功")
        print(f"   📤 数据: {test_data}")
    else:
        print(f"   ❌ 推送失败 (HTTP {resp.status_code}): {resp.text}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ 推送异常: {e}")
    sys.exit(1)

# 测试 3: 读取位置
print("\n3️⃣  读取位置数据...")
try:
    resp = requests.get(f"{UI_URL}/api/agv/pose", timeout=2)
    data = resp.json()
    pose = data.get('pose', {})
    print(f"   ✅ 读取成功")
    print(f"   📖 数据: theta={pose.get('theta')}, x={pose.get('x')}, y={pose.get('y')}")
    
    # 验证
    if abs(pose.get('theta', 0) - test_data['theta']) < 0.01:
        print(f"\n   ✅ 数据正确存储并返回！")
    else:
        print(f"\n   ⚠️  数据不匹配!")
        print(f"      期望: {test_data}")
        print(f"      实际: {pose}")
except Exception as e:
    print(f"   ❌ 读取异常: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ 后端工作正常！问题在前端。")
print("\n📋 请按以下步骤诊断前端:")
print("   1. F12 打开浏览器控制台")
print("   2. 查看 Console 标签中的日志")
print("   3. 运行: python debug_pose.py")
print("   4. 再次查看浏览器控制台输出")
print("="*60 + "\n")
