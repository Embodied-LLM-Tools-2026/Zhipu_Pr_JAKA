#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# AGV 底盘控制测试脚本

echo "╔════════════════════════════════════════════════════════════╗"
echo "║    AGV 底盘控制鲁棒性测试 - 快速启动脚本                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查 Python 环境
echo "🔍 检查 Python 环境..."
python3 --version

# 检查必要的模块
echo ""
echo "🔍 检查依赖模块..."
python3 -c "from PIL import Image; print('  ✅ PIL')" 2>/dev/null || echo "  ❌ PIL (需要安装)"
python3 -c "import numpy; print('  ✅ numpy')" 2>/dev/null || echo "  ❌ numpy (需要安装)"
python3 -c "import socket; print('  ✅ socket')" 2>/dev/null || echo "  ✅ socket"
python3 -c "import json; print('  ✅ json')" 2>/dev/null || echo "  ✅ json"

echo ""
echo "⚠️  确保事项："
echo "  1. AGV 底盘服务已启动 (http://192.168.10.10:31001)"
echo "  2. 网络连接正常"
echo "  3. 底盘周围环境安全"
echo "  4. 按需在测试中修改参数"
echo ""

# 运行测试
echo "🚀 启动测试程序..."
echo ""
cd "$(dirname "$0")"
python3 test_real_chassis_control.py

echo ""
echo "✅ 测试完成！"
echo "📊 查看结果: test_chassis_report.json"
