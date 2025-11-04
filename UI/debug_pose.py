#!/usr/bin/env python3
"""
调试脚本：测试 AGV 位置推送和前端显示
"""
import requests
import time

UI_URL = "http://127.0.0.1:8000"

def test_pose_flow():
    """完整测试流程"""
    print("=" * 70)
    print("AGV 位置更新完整测试")
    print("=" * 70)
    
    # 1. 读取初始位置
    print("\n📖 [步骤1] 读取初始 AGV 位置...")
    try:
        resp = requests.get(f"{UI_URL}/api/agv/pose", timeout=3)
        initial_pose = resp.json()
        print(f"   ✅ 初始位置: {initial_pose['pose']}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return
    
    # 2. 推送新位置
    print("\n📝 [步骤2] 推送新位置到服务器...")
    new_pose = {
        "theta": 1.57,
        "x": 15.5,
        "y": 25.8
    }
    try:
        resp = requests.post(f"{UI_URL}/api/agv/pose/update", json=new_pose, timeout=3)
        print(f"   ✅ 推送成功: {resp.json()}")
        print(f"   📤 推送的数据: theta={new_pose['theta']}, x={new_pose['x']}, y={new_pose['y']}")
    except Exception as e:
        print(f"   ❌ 推送失败: {e}")
        return
    
    # 3. 等待一下，让前端轮询一次
    print("\n⏳ [步骤3] 等待前端轮询（100ms）...")
    time.sleep(0.2)
    
    # 4. 读取更新后的位置
    print("\n📖 [步骤4] 读取更新后的位置...")
    try:
        resp = requests.get(f"{UI_URL}/api/agv/pose", timeout=3)
        updated_pose = resp.json()
        print(f"   ✅ 更新后位置: {updated_pose['pose']}")
        
        # 验证数据
        pose = updated_pose['pose']
        if abs(pose['theta'] - new_pose['theta']) < 0.01 and \
           abs(pose['x'] - new_pose['x']) < 0.01 and \
           abs(pose['y'] - new_pose['y']) < 0.01:
            print(f"\n   ✅ ✅ ✅ 数据正确更新！")
        else:
            print(f"\n   ❌ 数据未正确更新")
            print(f"      期望: theta={new_pose['theta']}, x={new_pose['x']}, y={new_pose['y']}")
            print(f"      实际: theta={pose['theta']}, x={pose['x']}, y={pose['y']}")
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！现在请:")
    print("   1. 打开浏览器 http://127.0.0.1:8000")
    print("   2. 打开开发者工具 (F12 → Console)")
    print("   3. 查看浏览器控制台中的日志输出")
    print("   4. 检查 'AGV Position (Real-time)' 是否显示:")
    print(f"      θ=1.57rad, x=15.50m, y=25.80m")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_pose_flow()
    except KeyboardInterrupt:
        print("\n\n已取消")
