#!/usr/bin/env python3
"""
测试自动路径规划建图功能
"""
import time
from agv_workflow import AGVWorkflow


def test_auto_mapping_basic():
    """测试基本自动建图功能"""
    print("========== 测试基本自动建图功能 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    if not workflow.connect():
        print("连接AGV失败")
        return False
    
    try:
        # 测试自动建图
        success = workflow.stage1_mapping(
            auto_mapping=True,
            turn_direction='left',
            forward_distance=1.0,
            side_distance=0.5
        )
        
        if success:
            print("✅ 自动建图测试成功")
            return True
        else:
            print("❌ 自动建图测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 自动建图测试异常: {e}")
        return False
        
    finally:
        workflow.disconnect()


def test_auto_mapping_parameters():
    """测试不同参数的自动建图"""
    print("========== 测试不同参数的自动建图 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    if not workflow.connect():
        print("连接AGV失败")
        return False
    
    # 测试参数组合
    test_params = [
        {'turn_direction': 'left', 'forward_distance': 0.8, 'side_distance': 0.3},
        {'turn_direction': 'right', 'forward_distance': 1.2, 'side_distance': 0.6},
    ]
    
    results = []
    
    try:
        for i, params in enumerate(test_params):
            print(f"\n--- 测试参数组合 {i+1}: {params} ---")
            
            success = workflow.stage1_mapping(
                auto_mapping=True,
                **params
            )
            
            results.append({
                'params': params,
                'success': success
            })
            
            if success:
                print(f"✅ 参数组合 {i+1} 测试成功")
            else:
                print(f"❌ 参数组合 {i+1} 测试失败")
            
            # 短暂等待
            time.sleep(2)
        
        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        print(f"\n========== 参数测试总结 ==========")
        print(f"成功: {success_count}/{len(results)}")
        
        for i, result in enumerate(results):
            status = "✅ 成功" if result['success'] else "❌ 失败"
            print(f"参数组合 {i+1}: {result['params']} - {status}")
        
        return success_count == len(results)
        
    except Exception as e:
        print(f"❌ 参数测试异常: {e}")
        return False
        
    finally:
        workflow.disconnect()


def test_complete_workflow_with_auto_mapping():
    """测试包含自动建图的完整工作流程"""
    print("========== 测试包含自动建图的完整工作流程 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    # 定义测试目标点
    target_points = [
        (0.5, 0.5, 0.0),
        (1.0, 0.0, 1.57),
    ]
    
    # 选择可靠的导航方法
    navigation_methods = ['method1_world_coordinate']
    
    try:
        results = workflow.run_complete_workflow(
            mapping_points=None,           # 自动建图不需要指定点
            target_points=target_points,
            navigation_methods=navigation_methods,
            auto_mapping=True,             # 启用自动建图
            turn_direction='left',         # 左转建图
            forward_distance=1.0,          # 前进1米
            side_distance=0.5              # 侧移0.5米
        )
        
        if results['success']:
            print("✅ 完整工作流程测试成功")
            workflow.generate_test_report(results['navigation_results'])
            return True
        else:
            print(f"❌ 完整工作流程测试失败: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ 完整工作流程测试异常: {e}")
        return False


def test_mapping_methods_comparison():
    """测试两种建图方法对比"""
    print("========== 测试两种建图方法对比 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    if not workflow.connect():
        print("连接AGV失败")
        return False
    
    try:
        # 测试手动建图
        print("\n--- 测试手动指定点建图 ---")
        mapping_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.57),
            (0.0, 1.0, 3.14),
            (0.0, 0.0, 0.0),
        ]
        
        manual_success = workflow.stage1_mapping(
            mapping_points=mapping_points,
            turns_per_point=0.5,
            auto_mapping=False
        )
        
        time.sleep(3)
        
        # 测试自动建图
        print("\n--- 测试自动路径规划建图 ---")
        auto_success = workflow.stage1_mapping(
            auto_mapping=True,
            turn_direction='left',
            forward_distance=1.0,
            side_distance=0.5
        )
        
        # 结果对比
        print(f"\n========== 建图方法对比结果 ==========")
        print(f"手动指定点建图: {'✅ 成功' if manual_success else '❌ 失败'}")
        print(f"自动路径规划建图: {'✅ 成功' if auto_success else '❌ 失败'}")
        
        return manual_success and auto_success
        
    except Exception as e:
        print(f"❌ 建图方法对比测试异常: {e}")
        return False
        
    finally:
        workflow.disconnect()


def main():
    """运行所有测试"""
    print("AGV自动路径规划建图功能测试")
    print("=" * 50)
    
    tests = [
        ("基本自动建图功能", test_auto_mapping_basic),
        ("不同参数自动建图", test_auto_mapping_parameters),
        ("完整工作流程", test_complete_workflow_with_auto_mapping),
        ("建图方法对比", test_mapping_methods_comparison),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} - 通过")
            else:
                print(f"❌ {test_name} - 失败")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  {test_name} - 用户中断")
            break
            
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
            results.append((test_name, False))
        
        # 测试间隔
        time.sleep(2)
    
    # 测试总结
    print(f"\n{'='*50}")
    print("测试结果总结")
    print("=" * 50)
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:<20} - {status}")
    
    print(f"\n总体结果: {success_count}/{total_count} 通过")
    
    if success_count == total_count:
        print("🎉 所有测试通过！自动路径规划建图功能正常工作。")
    else:
        print("⚠️  部分测试失败，请检查相关功能。")


if __name__ == '__main__':
    main()