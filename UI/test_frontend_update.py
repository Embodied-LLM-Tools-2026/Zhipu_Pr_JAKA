#!/usr/bin/env python3
"""
直接测试：通过 Python 模拟浏览器请求，检查前端数据流
"""
import requests
import time

BASE_URL = "http://127.0.0.1:8000"

print("\n" + "="*70)
print("🔍 前端更新问题诊断 - Python 模拟请求")
print("="*70)

# 第 1 步：推送新数据到后端
print("\n1️⃣  【后端】推送新数据...")
test_values = {
    "theta": 3.14159,  # π rad
    "x": 123.456,
    "y": 789.012
}

resp = requests.post(
    f"{BASE_URL}/api/agv/pose/update",
    json=test_values,
    timeout=2
)
print(f"   推送响应: {resp.json()}")

# 第 2 步：查看后端是否保存了
print("\n2️⃣  【后端】读取保存的数据...")
resp = requests.get(f"{BASE_URL}/api/agv/pose", timeout=2)
backend_data = resp.json()
print(f"   后端数据: {backend_data['pose']}")

# 验证
if backend_data['pose']['theta'] == test_values['theta']:
    print("   ✅ 后端数据正确保存")
else:
    print("   ❌ 后端数据未保存")

# 第 3 步：现在让我们模拟浏览器会看到什么
print("\n3️⃣  【前端】HTML 页面测试...")
print("   打开浏览器访问: http://127.0.0.1:8000")
print("   按 F12 打开开发者工具，Console 标签")
print("   应该看到:")
print(f"   [AGV位置] 收到响应: {backend_data}")
print(f"   [AGV位置] 更新DOM - theta={backend_data['pose']['theta']:.2f}, x={backend_data['pose']['x']:.2f}, y={backend_data['pose']['y']:.2f}")

# 第 4 步：手动测试 JavaScript 的 DOM 更新
print("\n4️⃣  【测试】在浏览器控制台运行这段代码:")
print("""
// 检查 DOM 元素是否存在
console.log('theta 元素:', document.getElementById('theta'));
console.log('agv-x 元素:', document.getElementById('agv-x'));
console.log('agv-y 元素:', document.getElementById('agv-y'));

// 手动设置值
document.getElementById('theta').textContent = '3.14';
document.getElementById('agv-x').textContent = '123.46';
document.getElementById('agv-y').textContent = '789.01';

// 检查是否更新
console.log('更新后 theta:', document.getElementById('theta').textContent);
""")

print("\n" + "="*70)
print("📋 诊断清单:")
print("   ✓ 后端数据是否正确保存? " + ("✅ 是" if backend_data['pose']['theta'] == test_values['theta'] else "❌ 否"))
print("   ✓ 浏览器控制台是否有日志? → 需要你检查")
print("   ✓ 页面上的数字是否已更新? → 需要你检查")
print("   ✓ 手动测试 DOM 更新是否有效? → 需要你检查")
print("="*70 + "\n")

print("💡 可能的原因:")
print("   1. 前端的 tick() 函数没有被执行")
print("   2. tick() 函数出现异常，代码中止")
print("   3. DOM 元素的选择器不对")
print("   4. textContent 赋值有问题（可能被 CSS 隐藏）")
print("\n告诉我浏览器控制台看到了什么，我来帮你进一步诊断")
