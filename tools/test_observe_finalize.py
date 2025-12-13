#!/usr/bin/env python3
"""
测试脚本: 从不同起始位姿运行 observe -> finalize_target_pose 流程

用法:
    # 手动放置模式 (不移动机器人，每次测试前暂停等待用户手动移动):
    python3 tools/test_observe_finalize.py --target cola
    
    # 自动移动模式 (机器人自动移动到起始位姿):
    python3 tools/test_observe_finalize.py --target cola --move --poses "0,0,0; -1.34,7.75,-1.14"
    
    # 指定单个位姿:
    python3 tools/test_observe_finalize.py --target cola --move --poses "-1.34,7.75,-1.14"
    
参数:
    --target: 目标物体名称 (如 cola, bottle 等)
    --poses: 起始位姿列表，格式: "theta,x,y; theta,x,y; ..."  (theta为弧度，x/y为米)
    --move: 启用自动移动到起始位姿
    --distance: 目标距离 (米)，默认 0.75
    --timeout: 导航超时 (秒)，默认 60
"""

import argparse
import math
import sys
import time
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from action_sequence.navigate import Navigate
from voice.perception.observer import VLMObserver, ObservationContext
from voice.perception.localize_target import (
    fetch_snapshot,
    localize_from_service,
    localize_object,
)
from voice.control.task_structures import ObservationPhase


def parse_poses(poses_str: str) -> list:
    """解析位姿字符串为列表 [(theta, x, y), ...]"""
    poses = []
    if not poses_str:
        return poses
    for part in poses_str.split(";"):
        part = part.strip()
        if not part:
            continue
        values = [float(v.strip()) for v in part.split(",")]
        if len(values) != 3:
            raise ValueError(f"位姿格式错误: {part}, 应为 'theta,x,y'")
        poses.append(tuple(values))
    return poses


def print_separator(title: str = ""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def run_observe_and_localize(
    observer: VLMObserver,
    navigator: Navigate,
    target_name: str,
    pose_index: int,
) -> dict:
    """
    执行一次 observe + depth localization，返回结果字典
    """
    result = {
        "pose_index": pose_index,
        "robot_pose": None,
        "observation": None,
        "depth_info": None,
        "world_coords": None,
        "error": None,
    }
    
    try:
        # 获取当前位姿
        current_pose = navigator.get_current_pose()
        result["robot_pose"] = current_pose
        print(f"\n📍 当前机器人位姿:")
        print(f"    theta = {current_pose['theta']:.4f} rad ({math.degrees(current_pose['theta']):.2f}°)")
        print(f"    x = {current_pose['x']:.4f} m")
        print(f"    y = {current_pose['y']:.4f} m")
        
        # 创建观测上下文
        context = ObservationContext(step=1, max_steps=1)
        
        # 执行观测 (强制使用 VLM)
        print(f"\n🔍 调用 VLM 观测目标: '{target_name}' ...")
        observation, payload = observer.observe(
            target_name=target_name,
            phase=ObservationPhase.INITIAL,
            context=context,
            navigator=navigator,
            force_vlm=True,
        )
        result["observation"] = observation
        
        if not observation.found:
            result["error"] = "VLM 未找到目标"
            print(f"❌ VLM 未找到目标 '{target_name}'")
            return result
        
        print(f"✅ VLM 找到目标!")
        print(f"    bbox: {observation.bbox}")
        print(f"    confidence: {observation.confidence}")
        print(f"    range_estimate: {observation.range_estimate}")
        if observation.original_image_path:
            print(f"    image_path: {observation.original_image_path}")
        
        # 执行深度定位
        print(f"\n📐 执行深度定位 ...")
        if observation.bbox and observation.depth_snapshot:
            depth_info = localize_from_service(
                snapshot=observation.depth_snapshot,
                bbox=observation.bbox,
                range_estimate=observation.range_estimate,
            )
            result["depth_info"] = depth_info
            
            if depth_info:
                obj_center_3d = depth_info.get("obj_center_3d")
                tune_angle = depth_info.get("tune_angle", 0.0)
                mask_available = depth_info.get("surface_mask_available", False)
                
                print(f"✅ 深度定位成功!")
                print(f"    obj_center_3d (相机坐标系): {obj_center_3d}")
                print(f"    tune_angle: {tune_angle:.4f} rad ({math.degrees(tune_angle):.2f}°)")
                print(f"    surface_mask_available: {mask_available}")
                
                # 计算世界坐标 (复制 finalize_target_pose 的逻辑)
                if obj_center_3d:
                    cam_x, cam_y, cam_z = obj_center_3d
                    vec = np.array([cam_x, cam_y, cam_z, 1.0], dtype=float).reshape(4, 1)
                    T_mat = np.array([
                        [0, 1, 0, 180],
                        [-1, 0, 0, 50],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ], dtype=float)
                    jaka_vec = T_mat @ vec
                    
                    X_AB, Y_AB, Z_AB = jaka_vec[:3].ravel()
                    print(f"\n🔧 相机->机器人坐标变换后 (mm):")
                    print(f"    X_AB = {X_AB:.2f} mm")
                    print(f"    Y_AB = {Y_AB:.2f} mm")
                    print(f"    Z_AB = {Z_AB:.2f} mm")
                    
                    # 计算世界坐标
                    X_OA = current_pose["x"] * 1000
                    Y_OA = current_pose["y"] * 1000
                    theta_OA = current_pose["theta"]
                    
                    X_OB = X_OA + (X_AB * math.cos(theta_OA) - Y_AB * math.sin(theta_OA))
                    Y_OB = Y_OA + (X_AB * math.sin(theta_OA) + Y_AB * math.cos(theta_OA))
                    
                    obj_world_x = X_OB / 1000.0
                    obj_world_y = Y_OB / 1000.0
                    
                    result["world_coords"] = {
                        "x": obj_world_x,
                        "y": obj_world_y,
                        "robot_frame_mm": [X_AB, Y_AB, Z_AB],
                    }
                    
                    print(f"\n🌍 目标物体世界坐标:")
                    print(f"    x = {obj_world_x:.4f} m")
                    print(f"    y = {obj_world_y:.4f} m")
            else:
                result["error"] = "深度定位失败"
                print(f"❌ 深度定位失败")
        else:
            result["error"] = "缺少 bbox 或 depth_snapshot"
            print(f"❌ 缺少 bbox 或 depth_snapshot")
            
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def run_finalize_target_pose(
    navigator: Navigate,
    depth_info: dict,
    target_distance: float = 0.75,
) -> dict:
    """
    执行 finalize_target_pose 的移动逻辑
    """
    result = {
        "success": False,
        "target_pose": None,
        "error": None,
    }
    
    try:
        obj_center_3d = depth_info.get("obj_center_3d")
        tune_angle = float(depth_info.get("tune_angle", 0.0))
        mask_available = bool(depth_info.get("surface_mask_available"))
        
        if not obj_center_3d:
            result["error"] = "缺少 obj_center_3d"
            return result
        
        # 相机->机器人坐标变换
        cam_x, cam_y, cam_z = obj_center_3d
        vec = np.array([cam_x, cam_y, cam_z, 1.0], dtype=float).reshape(4, 1)
        T_mat = np.array([
            [0, 1, 0, 180],
            [-1, 0, 0, 50],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        jaka_vec = T_mat @ vec
        X_AB, Y_AB, Z_AB = jaka_vec[:3].ravel()
        
        # 获取当前位姿
        pose = navigator.get_current_pose()
        X_OA = pose["x"] * 1000
        Y_OA = pose["y"] * 1000
        theta_OA = pose["theta"]
        
        # 计算世界坐标
        X_OB = X_OA + (X_AB * math.cos(theta_OA) - Y_AB * math.sin(theta_OA))
        Y_OB = Y_OA + (X_AB * math.sin(theta_OA) + Y_AB * math.cos(theta_OA))
        obj_world_x = X_OB / 1000.0
        obj_world_y = Y_OB / 1000.0
        
        print(f"\n📍 目标物体世界坐标: ({obj_world_x:.3f}, {obj_world_y:.3f}) m")
        
        # 计算目标朝向
        if mask_available:
            target_theta = theta_OA + tune_angle
            while target_theta > math.pi:
                target_theta -= 2 * math.pi
            while target_theta < -math.pi:
                target_theta += 2 * math.pi
            orientation_source = "sam_edge"
        else:
            dx = obj_world_x - pose["x"]
            dy = obj_world_y - pose["y"]
            if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                target_theta = theta_OA
            else:
                target_theta = math.atan2(dy, dx)
            orientation_source = "target_vector"
        
        # 计算目标位置 (物体前方 target_distance 处)
        target_x = obj_world_x - target_distance * math.cos(target_theta)
        target_y = obj_world_y - target_distance * math.sin(target_theta)
        
        # 横向偏移 (让右臂中心对准目标)
        ARM_LATERAL_OFFSET = 0.15
        target_x += ARM_LATERAL_OFFSET * math.cos(target_theta + math.pi / 2)
        target_y += ARM_LATERAL_OFFSET * math.sin(target_theta + math.pi / 2)
        
        result["target_pose"] = {
            "theta": target_theta,
            "x": target_x,
            "y": target_y,
            "orientation_source": orientation_source,
        }
        
        print(f"\n🎯 计算出的目标底盘位姿:")
        print(f"    theta = {target_theta:.4f} rad ({math.degrees(target_theta):.2f}°)")
        print(f"    x = {target_x:.4f} m")
        print(f"    y = {target_y:.4f} m")
        print(f"    orientation_source = {orientation_source}")
        
        # 验证距离
        actual_distance = math.sqrt((obj_world_x - target_x)**2 + (obj_world_y - target_y)**2)
        print(f"    实际距离到目标: {actual_distance:.3f} m (期望: {target_distance:.3f} m)")
        
        # 执行移动
        print(f"\n🚀 开始移动到目标位置 ...")
        success = navigator.move_to_position(target_theta, target_x, target_y)
        result["success"] = success
        
        if success:
            print(f"✅ 移动成功!")
            final_pose = navigator.get_current_pose()
            print(f"    最终位姿: theta={final_pose['theta']:.4f}, x={final_pose['x']:.4f}, y={final_pose['y']:.4f}")
        else:
            result["error"] = "navigator.move_to_position 失败"
            print(f"❌ 移动失败!")
            
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="测试 observe + finalize_target_pose 流程")
    parser.add_argument("--target", type=str, default="cola", help="目标物体名称")
    parser.add_argument("--poses", type=str, default="", help="起始位姿列表: 'theta,x,y; theta,x,y; ...'")
    parser.add_argument("--move", action="store_true", help="启用自动移动到起始位姿")
    parser.add_argument("--distance", type=float, default=0.75, help="目标距离 (米)")
    parser.add_argument("--timeout", type=float, default=60, help="导航超时 (秒)")
    parser.add_argument("--skip-finalize", action="store_true", help="跳过 finalize (不移动)")
    args = parser.parse_args()
    
    print_separator("测试配置")
    print(f"目标物体: {args.target}")
    print(f"目标距离: {args.distance} m")
    print(f"自动移动到起始位姿: {'是' if args.move else '否 (手动放置)'}")
    print(f"执行 finalize 移动: {'否' if args.skip_finalize else '是'}")
    
    # 解析位姿列表
    poses = parse_poses(args.poses)
    if not poses:
        poses = [(0.0, 0.0, 0.0)]  # 默认当前位姿
        print(f"未指定位姿，使用当前位姿")
    else:
        print(f"位姿列表 ({len(poses)} 个):")
        for i, (theta, x, y) in enumerate(poses):
            print(f"  [{i}] theta={theta:.4f} ({math.degrees(theta):.2f}°), x={x:.4f}m, y={y:.4f}m")
    
    # 初始化组件
    print_separator("初始化")
    print("初始化 Navigator ...")
    navigator = Navigate()
    time.sleep(0.5)  # 等待状态监控线程启动
    
    print("初始化 VLMObserver ...")
    observer = VLMObserver()
    
    # 汇总结果
    all_results = []
    
    # 遍历每个起始位姿
    for pose_idx, (start_theta, start_x, start_y) in enumerate(poses):
        print_separator(f"测试 #{pose_idx + 1}/{len(poses)}")
        
        if args.move:
            # 自动移动到起始位姿
            print(f"🚀 移动到起始位姿: theta={start_theta:.4f}, x={start_x:.4f}, y={start_y:.4f}")
            success = navigator.move_to_position(start_theta, start_x, start_y, timeout=args.timeout)
            if not success:
                print(f"❌ 移动到起始位姿失败，跳过此测试")
                all_results.append({
                    "pose_index": pose_idx,
                    "start_pose": (start_theta, start_x, start_y),
                    "error": "移动到起始位姿失败",
                })
                continue
            time.sleep(1.0)  # 等待稳定
        else:
            # 手动放置模式
            print(f"📌 请手动将机器人移动到位姿 [{pose_idx}]:")
            print(f"    theta = {start_theta:.4f} rad ({math.degrees(start_theta):.2f}°)")
            print(f"    x = {start_x:.4f} m")
            print(f"    y = {start_y:.4f} m")
            input("按 Enter 继续 (或 Ctrl+C 退出) ...")
        
        # 执行 observe + localize
        obs_result = run_observe_and_localize(
            observer=observer,
            navigator=navigator,
            target_name=args.target,
            pose_index=pose_idx,
        )
        
        # 如果有深度信息且未跳过 finalize，执行 finalize
        finalize_result = None
        if obs_result["depth_info"] and not args.skip_finalize:
            print_separator(f"执行 finalize_target_pose #{pose_idx + 1}")
            finalize_result = run_finalize_target_pose(
                navigator=navigator,
                depth_info=obs_result["depth_info"],
                target_distance=args.distance,
            )
        
        all_results.append({
            "pose_index": pose_idx,
            "start_pose": (start_theta, start_x, start_y),
            "observe_result": obs_result,
            "finalize_result": finalize_result,
        })
    
    # 打印汇总
    print_separator("测试汇总")
    for res in all_results:
        idx = res["pose_index"]
        start = res["start_pose"]
        print(f"\n--- 测试 #{idx + 1} ---")
        print(f"起始位姿: theta={start[0]:.4f}, x={start[1]:.4f}, y={start[2]:.4f}")
        
        if "error" in res and res["error"]:
            print(f"❌ 错误: {res['error']}")
            continue
        
        obs = res.get("observe_result", {})
        if obs.get("error"):
            print(f"❌ 观测错误: {obs['error']}")
        elif obs.get("world_coords"):
            wc = obs["world_coords"]
            print(f"✅ 目标世界坐标: x={wc['x']:.4f}m, y={wc['y']:.4f}m")
        
        fin = res.get("finalize_result")
        if fin:
            if fin.get("success"):
                tp = fin["target_pose"]
                print(f"✅ finalize 成功: 目标位姿 theta={tp['theta']:.4f}, x={tp['x']:.4f}, y={tp['y']:.4f}")
            elif fin.get("error"):
                print(f"❌ finalize 错误: {fin['error']}")
    
    print_separator("测试完成")


if __name__ == "__main__":
    main()
