#!/usr/bin/env python3
"""
多摄像头实时监控示例
演示如何同时打开 RGB 和深度摄像头、读取参数、交互式操作
"""

from tools.detect_camera import CameraManager
import time

def example_basic():
    """基础示例：同时显示两个摄像头"""
    print("\n" + "="*60)
    print("示例 1: 基础使用 - 同时显示 RGB 和深度摄像头")
    print("="*60)
    
    manager = CameraManager()
    
    # 添加摄像头
    manager.add_camera('rgb', 3, camera_type='rgb')
    manager.add_camera('depth', 12, camera_type='depth')
    
    print("\n💡 快捷键提示:")
    print("   's' - 显示统计信息")
    print("   'q' / ESC - 退出")
    
    # 同时显示
    manager.display_all()
    
    # 显示最终统计
    manager.print_stats()

def example_depth_analysis():
    """深度分析示例：实时读取深度数据"""
    print("\n" + "="*60)
    print("示例 2: 深度分析 - 实时读取深度值")
    print("="*60)
    
    manager = CameraManager()
    
    # 只添加深度摄像头
    manager.add_camera('depth', 12, camera_type='depth')
    
    print("读取 10 秒内的深度数据...")
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < 10:
            frame = manager.read_frame('depth')
            if frame is not None:
                # 获取深度统计
                stats = manager.get_depth_stats('depth')
                
                # 获取特定像素深度值
                h, w = frame.shape[:2]
                center_depth = manager.get_pixel_depth('depth', w//2, h//2)
                
                print(f"时间: {(time.time()-start_time):.1f}s | "
                      f"深度范围: {stats['depth_min']:.0f}-{stats['depth_max']:.0f} | "
                      f"中心像素: {center_depth:.0f} | "
                      f"帧数: {manager.camera_info['depth']['frames']}")
                
                time.sleep(0.5)
    finally:
        manager.release_all()

def example_multi_rgb():
    """多 RGB 摄像头示例"""
    print("\n" + "="*60)
    print("示例 3: 多 RGB 摄像头 - 同时显示 3 个 RGB 源")
    print("="*60)
    
    manager = CameraManager()
    
    # 添加多个 RGB 摄像头
    manager.add_camera('front', 3, camera_type='rgb')
    manager.add_camera('left', 2, camera_type='rgb')
    manager.add_camera('right', 5, camera_type='rgb')
    
    # 显示 5 秒
    print("显示 5 秒后自动退出...")
    manager.display_all(duration=5)
    
    # 显示统计
    manager.print_stats()

def example_custom_processing():
    """自定义处理示例"""
    print("\n" + "="*60)
    print("示例 4: 自定义处理 - 深度图热力图可视化")
    print("="*60)
    
    manager = CameraManager()
    manager.add_camera('depth', 12, camera_type='depth')
    
    print("记录 5 秒的深度数据和处理...")
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < 5:
            frame = manager.read_frame('depth')
            if frame is not None:
                # 获取深度统计
                stats = manager.get_depth_stats('depth')
                
                # 自定义处理：显示深度分布
                if manager.camera_info['depth']['frames'] % 30 == 0:
                    print(f"\n帧 {manager.camera_info['depth']['frames']}:")
                    print(f"  图像尺寸: {stats['size']}")
                    print(f"  深度范围: {stats['depth_min']:.0f} ~ {stats['depth_max']:.0f}")
                    print(f"  平均深度: {stats['depth_mean']:.0f}")
                    print(f"  中心像素: {stats['center_pixel']:.0f}")
                
                time.sleep(0.033)  # ~30 FPS
    finally:
        manager.release_all()
        print("\n✅ 处理完成")

def main():
    print("\n" + "🎥 "*30)
    print("多摄像头实时监控系统 - 示例程序")
    print("🎥 "*30)
    
    print("\n请选择要运行的示例:")
    print("  1 - 基础使用（同时显示 RGB + 深度）")
    print("  2 - 深度分析（实时读取深度数据）")
    print("  3 - 多 RGB 摄像头（同时显示 3 个摄像头）")
    print("  4 - 自定义处理（深度数据分析）")
    print("  0 - 退出")
    
    choice = input("\n请输入选择 (0-4): ").strip()
    
    if choice == '1':
        example_basic()
    elif choice == '2':
        example_depth_analysis()
    elif choice == '3':
        example_multi_rgb()
    elif choice == '4':
        example_custom_processing()
    elif choice == '0':
        print("退出")
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
