"""
AGV 2D建图、定位、导航完整流程使用示例
"""
import time
from agv_client import AGVClient
from agv_workflow import AGVWorkflow


def example1_basic_agv_control():
    """示例1：基础AGV控制功能测试"""
    print("========== 示例1：基础AGV控制功能测试 ==========")
    
    with AGVClient(ip='192.168.192.5') as agv:
        # 获取当前状态
        print("--- 获取AGV状态 ---")
        pose = agv.get_pose()
        if pose:
            x, y, angle = pose
            print(f"当前位置: ({x:.3f}, {y:.3f}, {angle:.3f})")
        
        velocity = agv.get_velocity()
        if velocity:
            vx, vy, vw = velocity
            print(f"当前速度: vx={vx:.3f}, vy={vy:.3f}, vw={vw:.3f}")
        
        # 测试基础运动控制
        print("\n--- 测试基础运动控制 ---")
        print("向前平移0.5米...")
        if agv.translate(0.5, vx=0.2):
            time.sleep(3)  # 等待运动完成
        
        print("原地转90度...")
        if agv.rotate_in_place(turns=0.25, angular_velocity=0.5):  # 0.25圈 = 90度
            time.sleep(4)
        
        # 测试建图控制
        print("\n--- 测试建图控制 ---")
        slam_status = agv.get_slam_status()
        print(f"当前建图状态: {slam_status}")
        
        print("启动2D建图...")
        if agv.start_slam(slam_type=1):
            time.sleep(2)
            status = agv.get_slam_status()
            print(f"建图启动后状态: {status}")
            
            time.sleep(3)  # 建图3秒
            
            print("停止建图...")
            agv.stop_slam()


def example2_simple_navigation():
    """示例2：简单导航测试"""
    print("========== 示例2：简单导航测试 ==========")
    
    with AGVClient(ip='192.168.192.5') as agv:
        # 获取当前位置
        current_pose = agv.get_pose()
        if not current_pose:
            print("无法获取当前位置")
            return
        
        current_x, current_y, current_angle = current_pose
        print(f"起始位置: ({current_x:.3f}, {current_y:.3f}, {current_angle:.3f})")
        
        # 测试不同导航方法
        target_x, target_y, target_theta = current_x + 1.0, current_y + 1.0, 0.0
        
        print(f"\n--- 测试导航到目标点: ({target_x}, {target_y}, {target_theta}) ---")
        
        # 方法1：世界坐标导航
        print("方法1：世界坐标导航")
        if agv.navigate_to_point_method1(target_x, target_y, target_theta):
            # 等待导航完成
            while True:
                status = agv.get_navigation_status()
                if status in [0, 4]:  # 无导航或完成
                    break
                elif status == 5:  # 失败
                    print("导航失败")
                    break
                time.sleep(1)
        
        # 方法2：机器人坐标导航
        print("方法2：机器人坐标导航")
        if agv.navigate_to_point_method2(current_x, current_y, current_angle):
            # 等待导航完成
            while True:
                status = agv.get_navigation_status()
                if status in [0, 4]:
                    break
                elif status == 5:
                    print("导航失败")
                    break
                time.sleep(1)


def example3_complete_workflow():
    """示例3：完整工作流程测试"""
    print("========== 示例3：完整工作流程测试 ==========")
    
    # 创建工作流程管理器
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    # 定义建图点（小范围测试）
    mapping_points = [
        (0.0, 0.0, 0.0),      # 起始点
        (1.0, 0.0, 0.0),      # 前方1米
        (1.0, 1.0, 1.57),     # 右上角
        (0.0, 1.0, 3.14),     # 左上角
        (0.0, 0.0, 0.0),      # 回到起始点
    ]
    
    # 定义导航测试目标点
    target_points = [
        (0.5, 0.5, 0.0),      # 中心点
        (1.0, 0.0, 1.57),     # 右下角
        (0.0, 1.0, -1.57),    # 左上角
    ]
    
    # 选择要测试的导航方法（选择较可靠的方法）
    methods_to_test = [
        'method1_world_coordinate',
        'method5_translate_rotate'
    ]
    
    # 运行完整工作流程
    results = workflow.run_complete_workflow(
        mapping_points=mapping_points,
        target_points=target_points,
        navigation_methods=methods_to_test
    )
    
    if results['success']:
        print("完整工作流程执行成功！")
        workflow.generate_test_report(results['navigation_results'])
    else:
        print(f"工作流程失败: {results['error']}")


def example4_step_by_step_workflow():
    """示例4：分步骤执行工作流程"""
    print("========== 示例4：分步骤执行工作流程 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    if not workflow.connect():
        print("连接AGV失败")
        return
    
    try:
        # 第一阶段：建图
        print("\n开始第一阶段：建图")
        mapping_points = [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 1.57),
            (0.0, 0.5, 3.14),
            (0.0, 0.0, 0.0),
        ]
        
        slam_success = workflow.stage1_mapping(mapping_points, turns_per_point=0.5)
        if not slam_success:
            print("建图失败")
            return
        
        print("建图完成，等待5秒...")
        time.sleep(5)
        
        # 第二阶段：定位
        print("\n开始第二阶段：定位")
        localization_success = workflow.stage2_localization(auto_relocalize=True)
        if not localization_success:
            print("定位失败")
            return
        
        print("定位完成，等待3秒...")
        time.sleep(3)
        
        # 第三阶段：导航测试
        print("\n开始第三阶段：导航测试")
        target_points = [(0.3, 0.3, 0.0), (0.0, 0.0, 0.0)]
        methods_to_test = ['method1_world_coordinate']
        
        navigation_results = workflow.stage3_navigation(target_points, methods_to_test)
        workflow.generate_test_report(navigation_results)
        
    finally:
        workflow.disconnect()


def example6_auto_mapping_workflow():
    """示例6：自动路径规划建图工作流程（扫地机器人式）"""
    print("========== 示例6：自动路径规划建图工作流程 ==========")
    
    # 创建工作流程管理器
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    # 定义导航测试目标点
    target_points = [
        (0.5, 0.5, 0.0),      # 中心点
        (1.0, 0.0, 1.57),     # 右侧点
        (0.0, 1.0, -1.57),    # 上方点
    ]
    
    # 选择要测试的导航方法
    methods_to_test = [
        'method1_world_coordinate',
        'method5_translate_rotate'
    ]
    
    print("自动路径规划建图参数：")
    print("- 转向方向: 左转")  
    print("- 前进距离: 1.5米")
    print("- 侧移距离: 0.6米")
    
    # 运行完整工作流程（使用自动建图）
    results = workflow.run_complete_workflow(
        mapping_points=None,  # 自动建图不需要指定点
        target_points=target_points,
        navigation_methods=methods_to_test,
        auto_mapping=True,           # 启用自动建图
        turn_direction='left',       # 左转建图
        forward_distance=3,        # 前进1.5米
        side_distance=0.3            # 侧移0.6米
    )
    
    if results['success']:
        print("自动路径规划建图工作流程执行成功！")
        workflow.generate_test_report(results['navigation_results'])
    else:
        print(f"工作流程失败: {results['error']}")


def example7_compare_mapping_methods():
    """示例7：对比两种建图方法"""
    print("========== 示例7：对比两种建图方法 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    if not workflow.connect():
        print("连接AGV失败")
        return
    
    try:
        print("\n========== 方法1：手动指定点建图 ==========")
        
        # 手动指定点建图
        mapping_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.57),
            (0.0, 1.0, 3.14),
            (0.0, 0.0, 0.0),
        ]
        
        slam_success1 = workflow.stage1_mapping(
            mapping_points=mapping_points,
            turns_per_point=0.5,
            auto_mapping=False
        )
        
        if slam_success1:
            print("手动指定点建图完成")
        else:
            print("手动指定点建图失败")
        
        # 等待一段时间
        time.sleep(5)
        
        print("\n========== 方法2：自动路径规划建图 ==========")
        
        # 自动路径规划建图
        slam_success2 = workflow.stage1_mapping(
            mapping_points=None,
            auto_mapping=True,
            turn_direction='right',    # 右转建图
            forward_distance=1.0,      # 前进1米
            side_distance=0.5          # 侧移0.5米
        )
        
        if slam_success2:
            print("自动路径规划建图完成")
        else:
            print("自动路径规划建图失败")
        
        print("\n========== 建图方法对比总结 ==========")
        print(f"手动指定点建图: {'成功' if slam_success1 else '失败'}")
        print(f"自动路径规划建图: {'成功' if slam_success2 else '失败'}")
        
    finally:
        workflow.disconnect()


def example5_individual_method_testing():
    """示例5：单独测试各个控制方法"""
    print("========== 示例5：单独测试各个控制方法 ==========")
    
    with AGVClient(ip='192.168.192.5') as agv:
        print("--- 测试建图相关方法 ---")
        # 测试获取建图状态
        status = agv.get_slam_status()
        print(f"当前建图状态: {status}")
        
        print("\n--- 测试定位相关方法 ---")
        # 测试获取定位状态
        reloc_status = agv.get_relocalization_status()
        print(f"当前定位状态: {reloc_status}")
        
        # 测试自动重定位
        print("执行自动重定位...")
        if agv.relocalize(is_auto=True):
            print("重定位指令发送成功")
            time.sleep(3)
            final_status = agv.get_relocalization_status()
            print(f"重定位后状态: {final_status}")
        
        print("\n--- 测试运动控制方法 ---")
        # 测试原地转圈
        print("原地转半圈...")
        if agv.rotate_in_place(0.5, angular_velocity=0.8):
            time.sleep(4)
        
        # 测试平移
        print("向前平移30cm...")
        if agv.translate(0.3, vx=0.2):
            time.sleep(2)
        
        print("\n--- 测试导航状态监控 ---")
        # 获取导航状态
        nav_status = agv.get_navigation_status(simple=True)
        print(f"当前导航状态: {nav_status}")
        
        # 获取详细导航状态
        detailed_status = agv.get_navigation_status(simple=False)
        if detailed_status:
            status, task_type, target_point = detailed_status
            print(f"详细导航状态: status={status}, type={task_type}, target={target_point}")


def main():
    """主函数 - 选择要运行的示例"""
    print("AGV 2D建图、定位、导航系统 - 使用示例")
    print("=" * 60)
    print("1. 基础AGV控制功能测试")
    print("2. 简单导航测试") 
    print("3. 完整工作流程测试（手动指定点建图）")
    print("4. 分步骤执行工作流程")
    print("5. 单独测试各个控制方法")
    print("6. 自动路径规划建图工作流程（扫地机器人式）")
    print("7. 对比两种建图方法")
    print("0. 运行基础示例")
    
    try:
        choice = input("\n请选择要运行的示例 (0-7): ").strip()
        
        if choice == '1':
            example1_basic_agv_control()
        elif choice == '2':
            example2_simple_navigation()
        elif choice == '3':
            example3_complete_workflow()
        elif choice == '4':
            example4_step_by_step_workflow()
        elif choice == '5':
            example5_individual_method_testing()
        elif choice == '6':
            example6_auto_mapping_workflow()
        elif choice == '7':
            example7_compare_mapping_methods()
        elif choice == '0':
            print("运行基础示例...")
            example1_basic_agv_control()
            time.sleep(2)
            example2_simple_navigation()
            time.sleep(2)
            example5_individual_method_testing()
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行异常: {e}")


if __name__ == '__main__':
    main()