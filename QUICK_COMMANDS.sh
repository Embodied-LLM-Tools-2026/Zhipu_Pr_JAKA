#!/bin/bash
# AGV 底盘控制测试 - 快速命令参考
# 保存此文件为 QUICK_COMMANDS.sh 并运行: bash QUICK_COMMANDS.sh

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        🤖 AGV 底盘控制测试 - 快速命令参考              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 定义命令
echo "📋 快速命令参考"
echo "────────────────────────────────────────────────────────────"
echo ""

echo "方法 1️⃣ : 直接运行测试"
echo "  命令: python3 test_real_chassis_control.py"
echo "  说明: 最直接的方式，启动完整测试"
echo ""

echo "方法 2️⃣ : 使用 Shell 脚本启动"
echo "  命令: bash run_chassis_test.sh"
echo "  说明: 会先检查环境，再启动测试"
echo ""

echo "方法 3️⃣ : 后台运行（保存日志）"
echo "  命令: nohup python3 test_real_chassis_control.py > test_output.log 2>&1 &"
echo "  说明: 后台运行，日志保存到 test_output.log"
echo ""

echo "方法 4️⃣ : 查看后台进程"
echo "  命令: ps aux | grep test_real_chassis"
echo "  说明: 查看测试程序是否运行中"
echo ""

echo "方法 5️⃣ : 杀死测试程序"
echo "  命令: pkill -f test_real_chassis_control"
echo "  说明: 紧急停止（如卡住了）"
echo ""

echo "────────────────────────────────────────────────────────────"
echo ""

echo "📊 查看结果"
echo "────────────────────────────────────────────────────────────"
echo ""

echo "1️⃣  查看 JSON 报告"
echo "  命令: cat test_chassis_report.json | python3 -m json.tool"
echo "  说明: 格式化显示测试报告"
echo ""

echo "2️⃣  简单统计"
echo "  命令: python3 -c \"import json; data=json.load(open('test_chassis_report.json')); print(f\\\"成功率: {data['summary']['success_rate']:.1f}%\\\")\""
echo "  说明: 快速看成功率"
echo ""

echo "3️⃣  查看日志（如果后台运行）"
echo "  命令: tail -f test_output.log"
echo "  说明: 实时查看日志"
echo ""

echo "────────────────────────────────────────────────────────────"
echo ""

echo "🔧 参数调整"
echo "────────────────────────────────────────────────────────────"
echo ""

echo "修改等待时间"
echo "  编辑: test_real_chassis_control.py"
echo "  找到: self.SHORT_WAIT = 3"
echo "  改为: self.SHORT_WAIT = 2  (更短) 或 self.SHORT_WAIT = 5 (更长)"
echo ""

echo "修改前进距离"
echo "  编辑: test_real_chassis_control.py"
echo "  找到: distance=0.5"
echo "  改为: distance=0.3  (更短) 或 distance=1.0 (更长)"
echo ""

echo "修改转向角度"
echo "  编辑: test_real_chassis_control.py"
echo "  找到: angle_deg=90"
echo "  改为: angle_deg=45  (更小) 或 angle_deg=180 (更大)"
echo ""

echo "────────────────────────────────────────────────────────────"
echo ""

echo "🐛 调试"
echo "────────────────────────────────────────────────────────────"
echo ""

echo "检查 AGV 连接"
echo "  命令: ping 192.168.10.10"
echo "  说明: 测试网络连接"
echo ""

echo "检查服务状态"
echo "  命令: curl -s http://192.168.10.10:31001/api/robot_status | python3 -m json.tool"
echo "  说明: 查询 AGV 当前状态"
echo ""

echo "运行 Python 语法检查"
echo "  命令: python3 -m py_compile test_real_chassis_control.py"
echo "  说明: 检查代码是否有语法错误"
echo ""

echo "────────────────────────────────────────────────────────────"
echo ""

echo "📚 查看文档"
echo "────────────────────────────────────────────────────────────"
echo ""

echo "快速开始"
echo "  命令: cat TEST_START_HERE.txt"
echo ""

echo "详细指南"
echo "  命令: cat CHASSIS_TEST_GUIDE.md"
echo ""

echo "快速参考"
echo "  命令: cat CHASSIS_TEST_QUICK_REFERENCE.txt"
echo ""

echo "流程图"
echo "  命令: cat TEST_FLOW_DIAGRAM.txt"
echo ""

echo "────────────────────────────────────────────────────────────"
echo ""

echo "✨ 常用组合"
echo "────────────────────────────────────────────────────────────"
echo ""

echo "运行测试并保存日志"
echo "  python3 test_real_chassis_control.py | tee test_$(date +%Y%m%d_%H%M%S).log"
echo ""

echo "运行测试并在完成后自动查看报告"
echo "  python3 test_real_chassis_control.py && python3 -m json.tool < test_chassis_report.json | less"
echo ""

echo "连续运行多次测试"
echo "  for i in {1..3}; do echo \"测试 \$i\"; python3 test_real_chassis_control.py; sleep 30; done"
echo ""

echo "────────────────────────────────────────────────────────────"
echo ""

echo "✅ 测试完成后的步骤"
echo "────────────────────────────────────────────────────────────"
echo ""

echo "1. 查看成功率"
echo "   确保 ≥ 75%"
echo ""

echo "2. 检查失败的测试"
echo "   看是否有特定的操作失败"
echo ""

echo "3. 根据结果决定"
echo "   - 成功率好 → 可用于实际应用"
echo "   - 部分失败 → 调整参数重新测试"
echo "   - 大量失败 → 检查硬件问题"
echo ""

echo "4. 归档报告"
echo "   mv test_chassis_report.json test_chassis_report_$(date +%Y%m%d_%H%M%S).json"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "👉 现在运行测试? (输入下面的命令)"
echo "   python3 test_real_chassis_control.py"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
