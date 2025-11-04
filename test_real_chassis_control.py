#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实 AGV 底盘控制鲁棒性测试

测试场景：
  1. 单独测试 control_chassis_to_center() - 目标对齐
  2. 单独测试 control_chassis_forward() - 直线前进
  3. 单独测试 control_turn_around() - 原地转向
  4. 连续测试 - 综合动作序列
  5. 压力测试 - 重复调用和边界情况
  6. 错误恢复测试 - 异常处理

作者: Robot Team
日期: 2025-10-23
"""

import os
import sys
import time
import math
import json
import traceback
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'voice')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'action_sequence')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))

from VLM import TaskProcessor
from navigate import Navigate
from task_logger import log_info, log_success, log_warning, log_error


class ChassisControlTester:
    """AGV 底盘控制测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.navigator = Navigate()
        self.processor = TaskProcessor()
        self.test_results = []
        self.test_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        # 等待时间（秒）
        self.SHORT_WAIT = 0.5
        self.MID_WAIT = 0.5
        self.LONG_WAIT = 0.5
        
        # 测试状态
        self.initial_pose = None
        self.start_time = None
        
    def print_header(self, title: str):
        """打印测试标题"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
    
    def print_step(self, step: str):
        """打印测试步骤"""
        print(f"\n➤ {step}")
    
    def print_result(self, success: bool, message: str, details: str = ""):
        """打印测试结果"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {message}")
        if details:
            print(f"         {details}")
    
    def record_result(self, test_name: str, success: bool, details: str = ""):
        """记录测试结果"""
        self.test_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_current_pose(self) -> Dict:
        """获取当前位置"""
        try:
            pose = self.navigator.get_current_pose()
            return pose
        except Exception as e:
            log_error(f"获取位置失败: {e}")
            return None
    
    def wait_for_stability(self, wait_time: float = 2):
        """等待底盘稳定"""
        print(f"  ⏳ 等待底盘稳定... ({wait_time}s)")
        time.sleep(wait_time)
    
    # =====================================================================
    # 测试 1: control_chassis_to_center() - 目标中心对齐
    # =====================================================================
    
    def test_chassis_to_center(self):
        """测试目标中心对齐"""
        self.print_header("测试 1: control_chassis_to_center() - 目标中心对齐")
        
        # 场景1: 目标在右侧 (+100px)
        self.print_step("场景1: 目标在右侧 (+100px)")
        try:
            bbox = [600, 400, 800, 600]  # 目标在右侧
            image_size = (1000, 1000)
            success = self.processor.control_chassis_to_center(
                bbox=bbox,
                image_size=image_size,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            pose = self.get_current_pose()
            self.print_result(success, f"目标右侧对齐", f"当前位置: {pose}")
            self.record_result("to_center_right_100px", success, f"pose: {pose}")
        except Exception as e:
            self.print_result(False, f"目标右侧对齐失败", str(e))
            self.record_result("to_center_right_100px", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.MID_WAIT)
        
        # 场景2: 目标在左侧 (-150px)
        self.print_step("场景2: 目标在左侧 (-150px)")
        try:
            bbox = [200, 400, 400, 600]  # 目标在左侧
            image_size = (1000, 1000)
            success = self.processor.control_chassis_to_center(
                bbox=bbox,
                image_size=image_size,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            pose = self.get_current_pose()
            self.print_result(success, f"目标左侧对齐", f"当前位置: {pose}")
            self.record_result("to_center_left_150px", success, f"pose: {pose}")
        except Exception as e:
            self.print_result(False, f"目标左侧对齐失败", str(e))
            self.record_result("to_center_left_150px", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.MID_WAIT)
        
        # 场景3: 目标已在中心（对齐）
        self.print_step("场景3: 目标已在中心（无需对齐）")
        try:
            bbox = [450, 400, 550, 600]  # 目标在中心
            image_size = (1000, 1000)
            success = self.processor.control_chassis_to_center(
                bbox=bbox,
                image_size=image_size,
                navigator=self.navigator
            )
            pose = self.get_current_pose()
            self.print_result(success, f"目标已对齐", f"当前位置: {pose}")
            self.record_result("to_center_already_aligned", success, f"pose: {pose}")
        except Exception as e:
            self.print_result(False, f"目标对齐检查失败", str(e))
            self.record_result("to_center_already_aligned", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.MID_WAIT)
        
        # 场景4: 大偏差 (+250px)
        self.print_step("场景4: 大偏差目标对齐 (+250px)")
        try:
            bbox = [750, 400, 950, 600]  # 大偏差
            image_size = (1000, 1000)
            success = self.processor.control_chassis_to_center(
                bbox=bbox,
                image_size=image_size,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            pose = self.get_current_pose()
            self.print_result(success, f"大偏差目标对齐", f"当前位置: {pose}")
            self.record_result("to_center_large_offset_250px", success, f"pose: {pose}")
        except Exception as e:
            self.print_result(False, f"大偏差对齐失败", str(e))
            self.record_result("to_center_large_offset_250px", False, str(e))
            traceback.print_exc()
    
    # =====================================================================
    # 测试 2: control_chassis_forward() - 直线前进
    # =====================================================================
    
    def test_chassis_forward(self):
        """测试直线前进"""
        self.print_header("测试 2: control_chassis_forward() - 直线前进")
        
        # 场景1: 前进 0.5m
        self.print_step("场景1: 前进 0.3m")
        try:
            pose_before = self.get_current_pose()
            print(f"  前进前位置: {pose_before}")
            
            success = self.processor.control_chassis_forward(
                distance=0.5,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            
            pose_after = self.get_current_pose()
            print(f"  前进后位置: {pose_after}")
            
            # 计算实际移动距离
            if pose_before and pose_after:
                actual_distance = math.sqrt(
                    (pose_after['x'] - pose_before['x'])**2 + 
                    (pose_after['y'] - pose_before['y'])**2
                )
                details = f"目标: 0.3m, 实际: {actual_distance:.3f}m, 误差: {abs(actual_distance - 0.3):.3f}m"
            else:
                details = "位置获取失败"

            self.print_result(success, f"前进 0.3m", details)
            self.record_result("forward_0.3m", success, details)
        except Exception as e:
            self.print_result(False, f"前进 0.5m 失败", str(e))
            self.record_result("forward_0.5m", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.MID_WAIT)
        
        # 场景2: 前进 1.0m
        self.print_step("场景2: 前进 0.3m")
        try:
            pose_before = self.get_current_pose()
            print(f"  前进前位置: {pose_before}")
            
            success = self.processor.control_chassis_forward(
                distance=0.5,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            
            pose_after = self.get_current_pose()
            print(f"  前进后位置: {pose_after}")
            
            if pose_before and pose_after:
                actual_distance = math.sqrt(
                    (pose_after['x'] - pose_before['x'])**2 + 
                    (pose_after['y'] - pose_before['y'])**2
                )
                details = f"目标: 0.3m, 实际: {actual_distance:.3f}m, 误差: {abs(actual_distance - 0.3):.3f}m"
            else:
                details = "位置获取失败"

            self.print_result(success, f"前进 0.3m", details)
            self.record_result("forward_0.3m", success, details)
        except Exception as e:
            self.print_result(False, f"前进 0.3m 失败", str(e))
            self.record_result("forward_0.3m", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.MID_WAIT)
        
        # 场景3: 小距离前进 (0.2m)
        self.print_step("场景3: 小距离前进 (0.2m)")
        try:
            pose_before = self.get_current_pose()
            print(f"  前进前位置: {pose_before}")
            
            success = self.processor.control_chassis_forward(
                distance=0.5,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            
            pose_after = self.get_current_pose()
            print(f"  前进后位置: {pose_after}")
            
            if pose_before and pose_after:
                actual_distance = math.sqrt(
                    (pose_after['x'] - pose_before['x'])**2 + 
                    (pose_after['y'] - pose_before['y'])**2
                )
                details = f"目标: 0.2m, 实际: {actual_distance:.3f}m, 误差: {abs(actual_distance - 0.2):.3f}m"
            else:
                details = "位置获取失败"
            
            self.print_result(success, f"小距离前进", details)
            self.record_result("forward_0.2m", success, details)
        except Exception as e:
            self.print_result(False, f"小距离前进失败", str(e))
            self.record_result("forward_0.2m", False, str(e))
            traceback.print_exc()
    
    # =====================================================================
    # 测试 3: control_turn_around() - 原地转向
    # =====================================================================
    
    def test_turn_around(self):
        """测试原地转向"""
        self.print_header("测试 3: control_turn_around() - 原地转向")
        
        # 场景1: 转向 90°
        self.print_step("场景1: 原地转向 90°")
        try:
            pose_before = self.get_current_pose()
            print(f"  转向前方向: {pose_before['theta']:.2f}°")
            
            success = self.processor.control_turn_around(
                turn_angle=1.57,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            
            pose_after = self.get_current_pose()
            print(f"  转向后方向: {pose_after['theta']:.2f}°")
            
            # 验证位置不变，方向改变
            if pose_before and pose_after:
                pos_change = math.sqrt(
                    (pose_after['x'] - pose_before['x'])**2 + 
                    (pose_after['y'] - pose_before['y'])**2
                )
                angle_change = abs(pose_after['theta'] - pose_before['theta'])
                details = f"位置移动: {pos_change:.3f}m (应≈0), 转向: {angle_change:.2f}° (应≈90°)"
            else:
                details = "位置获取失败"
            
            self.print_result(success, f"转向 90°", details)
            self.record_result("turn_90deg", success, details)
        except Exception as e:
            self.print_result(False, f"转向 90° 失败", str(e))
            self.record_result("turn_90deg", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.MID_WAIT)
        
        # 场景2: 转向 -45°（逆时针）
        self.print_step("场景2: 原地转向 -45°（逆时针）")
        try:
            pose_before = self.get_current_pose()
            print(f"  转向前方向: {pose_before['theta']:.2f}°")
            
            success = self.processor.control_turn_around(
                turn_angle=-0.785,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            
            pose_after = self.get_current_pose()
            print(f"  转向后方向: {pose_after['theta']:.2f}°")
            
            if pose_before and pose_after:
                pos_change = math.sqrt(
                    (pose_after['x'] - pose_before['x'])**2 + 
                    (pose_after['y'] - pose_before['y'])**2
                )
                angle_change = abs(pose_after['theta'] - pose_before['theta'])
                details = f"位置移动: {pos_change:.3f}m (应≈0), 转向: {angle_change:.2f}° (应≈45°)"
            else:
                details = "位置获取失败"
            
            self.print_result(success, f"转向 -45°", details)
            self.record_result("turn_minus45deg", success, details)
        except Exception as e:
            self.print_result(False, f"转向 -45° 失败", str(e))
            self.record_result("turn_minus45deg", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.MID_WAIT)
        
        # 场景3: 转向 180°（掉头）
        self.print_step("场景3: 原地转向 180°（掉头）")
        try:
            pose_before = self.get_current_pose()
            print(f"  转向前方向: {pose_before['theta']:.2f}°")
            
            success = self.processor.control_turn_around(
                turn_angle=3.14,
                navigator=self.navigator
            )
            self.wait_for_stability(self.SHORT_WAIT)
            
            pose_after = self.get_current_pose()
            print(f"  转向后方向: {pose_after['theta']:.2f}°")
            
            if pose_before and pose_after:
                pos_change = math.sqrt(
                    (pose_after['x'] - pose_before['x'])**2 + 
                    (pose_after['y'] - pose_before['y'])**2
                )
                angle_change = abs(pose_after['theta'] - pose_before['theta'])
                details = f"位置移动: {pos_change:.3f}m (应≈0), 转向: {angle_change:.2f}° (应≈180°)"
            else:
                details = "位置获取失败"
            
            self.print_result(success, f"转向 180°", details)
            self.record_result("turn_180deg", success, details)
        except Exception as e:
            self.print_result(False, f"转向 180° 失败", str(e))
            self.record_result("turn_180deg", False, str(e))
            traceback.print_exc()
    
    # =====================================================================
    # 测试 4: 连续操作序列测试
    # =====================================================================
    
    def test_continuous_sequence(self):
        """测试连续操作序列"""
        self.print_header("测试 4: 连续操作序列")
        
        self.print_step("序列1: 对齐 → 前进 → 转向")
        try:
            # 步骤1: 对齐
            print("  [1/3] 对齐目标...")
            bbox = [600, 400, 800, 600]
            success1 = self.processor.control_chassis_to_center(
                bbox=bbox,
                image_size=(1000, 1000),
                navigator=self.navigator
            )
            self.wait_for_stability(self.MID_WAIT)
            
            # 步骤2: 前进
            print("  [2/3] 前进 0.5m...")
            success2 = self.processor.control_chassis_forward(
                distance=0.5,
                navigator=self.navigator
            )
            self.wait_for_stability(self.MID_WAIT)
            
            # 步骤3: 转向
            print("  [3/3] 转向 45°...")
            success3 = self.processor.control_turn_around(
                turn_angle=0.785,
                navigator=self.navigator
            )
            self.wait_for_stability(self.MID_WAIT)
            
            pose_final = self.get_current_pose()
            overall_success = success1 and success2 and success3
            details = f"对齐:{success1}, 前进:{success2}, 转向:{success3}, 最终位置:{pose_final}"
            
            self.print_result(overall_success, f"序列完成", details)
            self.record_result("sequence_align_forward_turn", overall_success, details)
        except Exception as e:
            self.print_result(False, f"序列执行失败", str(e))
            self.record_result("sequence_align_forward_turn", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.LONG_WAIT)
        
        self.print_step("序列2: 前进 → 对齐 → 前进 → 转向（复杂序列）")
        try:
            # 步骤1: 前进
            print("  [1/4] 前进 0.3m...")
            success1 = self.processor.control_chassis_forward(
                distance=0.5,
                navigator=self.navigator
            )
            self.wait_for_stability(self.MID_WAIT)
            
            # 步骤2: 对齐
            print("  [2/4] 对齐目标（左侧）...")
            bbox = [300, 400, 500, 600]
            success2 = self.processor.control_chassis_to_center(
                bbox=bbox,
                image_size=(1000, 1000),
                navigator=self.navigator
            )
            self.wait_for_stability(self.MID_WAIT)
            
            # 步骤3: 再次前进
            print("  [3/4] 前进 0.8m...")
            success3 = self.processor.control_chassis_forward(
                distance=0.3,
                navigator=self.navigator
            )
            self.wait_for_stability(self.MID_WAIT)
            
            # 步骤4: 转向
            print("  [4/4] 转向 -90°...")
            success4 = self.processor.control_turn_around(
                turn_angle=-1.57,
                navigator=self.navigator
            )
            self.wait_for_stability(self.MID_WAIT)
            
            pose_final = self.get_current_pose()
            overall_success = success1 and success2 and success3 and success4
            details = f"前进1:{success1}, 对齐:{success2}, 前进2:{success3}, 转向:{success4}, 最终位置:{pose_final}"
            
            self.print_result(overall_success, f"复杂序列完成", details)
            self.record_result("sequence_forward_align_forward_turn", overall_success, details)
        except Exception as e:
            self.print_result(False, f"复杂序列执行失败", str(e))
            self.record_result("sequence_forward_align_forward_turn", False, str(e))
            traceback.print_exc()
    
    # =====================================================================
    # 测试 5: 压力测试 - 重复调用
    # =====================================================================
    
    def test_stress(self):
        """压力测试 - 重复调用"""
        self.print_header("测试 5: 压力测试 - 重复调用")
        
        self.print_step("压力测试1: 连续转向 10 次 (每次 10°)")
        try:
            success_count = 0
            for i in range(10):
                try:
                    success = self.processor.control_turn_around(
                        turn_angle=0.175,
                        navigator=self.navigator
                    )
                    if success:
                        success_count += 1
                    self.wait_for_stability(0.5)
                    print(f"    [{i+1}/10] 转向完成")
                except Exception as e:
                    print(f"    [{i+1}/10] 转向失败: {e}")
            
            pose_final = self.get_current_pose()
            success_rate = success_count / 10 * 100
            details = f"成功率: {success_rate:.1f}% ({success_count}/10), 最终位置: {pose_final}"
            
            self.print_result(success_rate >= 80, f"连续转向测试", details)
            self.record_result("stress_repeated_turns", success_rate >= 80, details)
        except Exception as e:
            self.print_result(False, f"压力测试失败", str(e))
            self.record_result("stress_repeated_turns", False, str(e))
            traceback.print_exc()
        
        self.wait_for_stability(self.LONG_WAIT)
        
        self.print_step("压力测试2: 连续对齐 5 次（不同目标位置）")
        try:
            success_count = 0
            bbox_list = [
                [200, 400, 400, 600],   # 左侧
                [600, 400, 800, 600],   # 右侧
                [450, 400, 550, 600],   # 中心
                [100, 400, 300, 600],   # 极左
                [700, 400, 900, 600],   # 极右
            ]
            
            for i, bbox in enumerate(bbox_list):
                try:
                    success = self.processor.control_chassis_to_center(
                        bbox=bbox,
                        image_size=(1000, 1000),
                        navigator=self.navigator
                    )
                    if success:
                        success_count += 1
                    self.wait_for_stability(self.MID_WAIT)
                    print(f"    [{i+1}/5] 对齐完成, bbox: {bbox}")
                except Exception as e:
                    print(f"    [{i+1}/5] 对齐失败: {e}")
            
            pose_final = self.get_current_pose()
            success_rate = success_count / 5 * 100
            details = f"成功率: {success_rate:.1f}% ({success_count}/5), 最终位置: {pose_final}"
            
            self.print_result(success_rate >= 80, f"连续对齐测试", details)
            self.record_result("stress_repeated_alignment", success_rate >= 80, details)
        except Exception as e:
            self.print_result(False, f"对齐压力测试失败", str(e))
            self.record_result("stress_repeated_alignment", False, str(e))
            traceback.print_exc()
    
    # =====================================================================
    # 测试 6: 错误恢复测试
    # =====================================================================
    
    def test_error_recovery(self):
        """错误恢复测试"""
        self.print_header("测试 6: 错误恢复和边界情况")
        
        self.print_step("边界测试1: 无效 bbox (超出范围)")
        try:
            # bbox 超出图像范围
            bbox = [1500, 1500, 2000, 2000]  # 完全超出 1000×1000
            success = self.processor.control_chassis_to_center(
                bbox=bbox,
                image_size=(1000, 1000),
                navigator=self.navigator
            )
            details = f"返回值: {success} (应该失败或返回 False)"
            # 这应该失败或被处理
            self.print_result(not success, f"无效 bbox 处理", details)
            self.record_result("edge_invalid_bbox", not success, details)
        except Exception as e:
            # 异常也表示正确处理
            details = f"捕获异常: {type(e).__name__}"
            self.print_result(True, f"无效 bbox 异常处理", details)
            self.record_result("edge_invalid_bbox", True, details)
        
        self.print_step("边界测试2: 0 距离前进")
        try:
            success = self.processor.control_chassis_forward(
                distance=0,
                navigator=self.navigator
            )
            details = f"返回值: {success} (应该成功或快速返回)"
            self.print_result(success is not None, f"0 距离处理", details)
            self.record_result("edge_zero_distance", success is not None, details)
        except Exception as e:
            details = f"捕获异常: {type(e).__name__}"
            self.print_result(True, f"0 距离异常处理", details)
            self.record_result("edge_zero_distance", True, details)
        
        self.print_step("边界测试3: 0 角度转向")
        try:
            success = self.processor.control_turn_around(
                turn_angle=0,
                navigator=self.navigator
            )
            details = f"返回值: {success} (应该成功或快速返回)"
            self.print_result(success is not None, f"0 角度处理", details)
            self.record_result("edge_zero_angle", success is not None, details)
        except Exception as e:
            details = f"捕获异常: {type(e).__name__}"
            self.print_result(True, f"0 角度异常处理", details)
            self.record_result("edge_zero_angle", True, details)
    
    # =====================================================================
    # 生成测试报告
    # =====================================================================
    
    def generate_report(self):
        """生成测试报告"""
        self.print_header("📊 测试总结报告")
        
        print(f"\n总测试数: {self.test_count}")
        print(f"✅ 成功: {self.success_count}")
        print(f"❌ 失败: {self.failure_count}")
        print(f"成功率: {(self.success_count / self.test_count * 100):.1f}%\n")
        
        # 按类别统计
        print("按类别统计:")
        print("-" * 60)
        
        categories = {
            "to_center": "目标对齐 (to_center)",
            "forward": "直线前进 (forward)",
            "turn": "转向 (turn)",
            "sequence": "连续序列 (sequence)",
            "stress": "压力测试 (stress)",
            "edge": "边界测试 (edge)"
        }
        
        for prefix, category_name in categories.items():
            results = [r for r in self.test_results if prefix in r['test_name']]
            if results:
                cat_success = sum(1 for r in results if r['success'])
                cat_total = len(results)
                rate = (cat_success / cat_total * 100) if cat_total > 0 else 0
                status = "✅" if rate >= 80 else "⚠️ " if rate >= 50 else "❌"
                print(f"  {status} {category_name}: {cat_success}/{cat_total} ({rate:.1f}%)")
        
        print("\n" + "=" * 60)
        print("详细结果:")
        print("=" * 60)
        for i, result in enumerate(self.test_results, 1):
            status = "✅" if result['success'] else "❌"
            print(f"{i:2d}. {status} {result['test_name']:40s} - {result['details'][:50]}")
        
        # 保存报告到文件
        report_file = "test_chassis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': self.test_count,
                    'passed': self.success_count,
                    'failed': self.failure_count,
                    'success_rate': (self.success_count / self.test_count * 100) if self.test_count > 0 else 0
                },
                'results': self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 报告已保存到: {report_file}")
    
    # =====================================================================
    # 主测试流程
    # =====================================================================
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " " * 20 + "🤖 AGV 底盘控制鲁棒性测试" + " " * 33 + "║")
        print("╚" + "="*78 + "╝")
        
        self.start_time = time.time()
        
        # 获取初始位置
        print("\n🔍 获取初始位置...")
        self.initial_pose = self.get_current_pose()
        if self.initial_pose:
            print(f"✅ 初始位置: {self.initial_pose}")
        else:
            print("⚠️  无法获取初始位置，继续测试...")
        
        print("\n准备开始测试，请确保底盘周围安全！")
        print("按 Enter 继续...")
        input()
        
        try:
            # 测试 1
            # self.test_chassis_to_center()
            
            # # 测试 2
            # self.test_chassis_forward()
            
            # # 测试 3
            # self.test_turn_around()
            
            # # 测试 4
            # self.test_continuous_sequence()
            
            # # 测试 5
            self.test_stress()
            
            # # 测试 6
            # self.test_error_recovery()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  测试被用户中断")
        except Exception as e:
            print(f"\n\n❌ 测试发生错误: {e}")
            traceback.print_exc()
        finally:
            # 生成报告
            self.generate_report()
            
            # 获取最终位置
            print("\n🔍 获取最终位置...")
            final_pose = self.get_current_pose()
            if final_pose:
                print(f"✅ 最终位置: {final_pose}")
            
            elapsed = time.time() - self.start_time
            print(f"\n总耗时: {elapsed:.1f} 秒")


def main():
    """主函数"""
    print("\n")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│        AGV 底盘控制鲁棒性测试程序                      │")
    print("│                                                         │")
    print("│  ⚠️  警告: 此程序将控制 AGV 实际移动                   │")
    print("│  请确保:                                                │")
    print("│    1. 底盘周围环境安全                                  │")
    print("│    2. AGV 在可控范围内                                  │")
    print("│    3. 已启动底盘控制服务                                │")
    print("│                                                         │")
    print("└─────────────────────────────────────────────────────────┘\n")
    
    response = input("确认开始测试? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("已取消测试")
        return
    
    # 创建测试器并运行
    tester = ChassisControlTester()
    time.sleep(2)  # 确保所有系统准备就绪
    tester.run_all_tests()


if __name__ == "__main__":
    main()
