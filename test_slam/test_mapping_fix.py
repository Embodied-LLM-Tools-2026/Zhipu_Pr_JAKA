#!/usr/bin/env python3
"""
测试修正后的建图流程
验证停止建图后正确轮询地图数据并上传载入新地图的功能
"""
import time
from agv_workflow import AGVWorkflow


def test_mapping_data_retrieval():
    """测试建图数据获取和地图载入流程"""
    print("========== 测试修正后的建图流程 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    if not workflow.connect():
        print("连接AGV失败")
        return False
    
    try:
        print("\n--- 测试1：轮询地图数据功能 ---")
        
        # 启动建图
        if not workflow.agv.start_slam(slam_type=1):
            print("启动建图失败")
            return False
        
        print("建图已启动，等待5秒后停止建图...")
        time.sleep(5)
        
        # 停止建图
        if not workflow.agv.stop_slam():
            print("停止建图失败")
            return False
        
        # 测试轮询地图数据
        print("开始测试轮询地图数据功能...")
        map_data = workflow._wait_for_map_data(timeout=30, check_interval=2)
        
        if map_data:
            print(f"✅ 成功获取地图数据，大小：{len(map_data)} 字符")
            
            # 显示地图数据的前100个字符作为预览
            preview = map_data[:100] + "..." if len(map_data) > 100 else map_data
            print(f"地图数据预览：{preview}")
            
            print("\n--- 测试2：地图载入验证功能 ---")
            # 测试地图载入验证
            if workflow._ensure_new_map_loaded(map_data):
                print("✅ 地图载入验证成功")
                return True
            else:
                print("❌ 地图载入验证失败")
                return False
        else:
            print("❌ 未能获取地图数据")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {e}")
        return False
        
    finally:
        workflow.disconnect()


def test_complete_mapping_workflow():
    """测试完整的建图工作流程"""
    print("========== 测试完整建图工作流程 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    # 定义小范围建图点进行测试
    mapping_points = [
        (0.0, 0.0, 0.0),      # 起始点
        (0.5, 0.0, 0.0),      # 前方0.5米
        (0.5, 0.5, 1.57),     # 右上角
        (0.0, 0.5, 3.14),     # 左上角
        (0.0, 0.0, 0.0),      # 回到起始点
    ]
    
    try:
        success = workflow.stage1_mapping(
            mapping_points=mapping_points,
            turns_per_point=0.5,  # 每个点转圈0.5圈
            angular_velocity=1.0,
            auto_mapping=False
        )
        
        if success:
            print("✅ 完整建图工作流程测试成功")
            return True
        else:
            print("❌ 完整建图工作流程测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 完整建图工作流程测试异常: {e}")
        return False


def test_map_info_queries():
    """测试地图信息查询功能"""
    print("========== 测试地图信息查询功能 ==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    if not workflow.connect():
        print("连接AGV失败")
        return False
    
    try:
        print("\n--- 测试地图信息查询 ---")
        maps_info = workflow.agv.get_maps_info()
        
        if maps_info:
            print("✅ 成功获取地图信息")
            print(f"当前载入地图：{maps_info.get('current_map')}")
            print(f"当前地图MD5：{maps_info.get('current_map_md5')}")
            print(f"可用地图列表：{maps_info.get('maps')}")
            print(f"地图文件数量：{len(maps_info.get('map_files_info', []))}")
            
            return True
        else:
            print("❌ 获取地图信息失败")
            return False
            
    except Exception as e:
        print(f"❌ 地图信息查询测试异常: {e}")
        return False
        
    finally:
        workflow.disconnect()


def test_slam_status_with_resultmap():
    """测试带resultmap的扫图状态查询"""
    print("========== 测试扫图状态查询（带resultmap）==========")
    
    workflow = AGVWorkflow(ip='192.168.192.5')
    
    if not workflow.connect():
        print("连接AGV失败")
        return False
    
    try:
        print("\n--- 测试扫图状态查询 ---")
        
        # 测试普通状态查询
        status = workflow.agv.get_slam_status(return_resultmap=False)
        print(f"当前扫图状态：{status}")
        
        # 测试带resultmap的状态查询
        result = workflow.agv.get_slam_status(return_resultmap=True)
        if result:
            slam_status, resultmap = result
            print(f"扫图状态：{slam_status}")
            if resultmap and resultmap.strip():
                print(f"✅ 获取到地图数据，大小：{len(resultmap)} 字符")
            else:
                print("当前无地图数据")
            
            return True
        else:
            print("❌ 查询扫图状态失败")
            return False
            
    except Exception as e:
        print(f"❌ 扫图状态查询测试异常: {e}")
        return False
        
    finally:
        workflow.disconnect()


def main():
    """运行所有测试"""
    print("AGV建图流程修正测试")
    print("=" * 50)
    print("1. 测试建图数据获取和地图载入")
    print("2. 测试完整建图工作流程")
    print("3. 测试地图信息查询")
    print("4. 测试扫图状态查询")
    print("5. 运行所有测试")
    print("0. 退出")
    
    try:
        choice = input("\n请选择测试项目 (0-5): ").strip()
        
        if choice == '1':
            success = test_mapping_data_retrieval()
        elif choice == '2':
            success = test_complete_mapping_workflow()
        elif choice == '3':
            success = test_map_info_queries()
        elif choice == '4':
            success = test_slam_status_with_resultmap()
        elif choice == '5':
            print("运行所有测试...")
            tests = [
                ("地图信息查询", test_map_info_queries),
                ("扫图状态查询", test_slam_status_with_resultmap),
                ("建图数据获取", test_mapping_data_retrieval),
                ("完整建图流程", test_complete_mapping_workflow),
            ]
            
            results = []
            for test_name, test_func in tests:
                print(f"\n{'='*20} {test_name} {'='*20}")
                try:
                    success = test_func()
                    results.append((test_name, success))
                    print(f"{'✅ 通过' if success else '❌ 失败'}: {test_name}")
                except Exception as e:
                    results.append((test_name, False))
                    print(f"❌ 异常: {test_name} - {e}")
                
                time.sleep(2)  # 测试间隔
            
            # 总结
            print(f"\n{'='*50}")
            print("测试结果总结")
            print("=" * 50)
            success_count = sum(1 for _, success in results if success)
            total_count = len(results)
            
            for test_name, success in results:
                status = "✅ 通过" if success else "❌ 失败"
                print(f"{test_name:<20} - {status}")
            
            print(f"\n总体结果: {success_count}/{total_count} 通过")
            success = success_count == total_count
            
        elif choice == '0':
            print("退出测试")
            return
        else:
            print("无效选择")
            return
        
        print(f"\n{'='*50}")
        if success:
            print("🎉 测试通过！建图流程修正功能正常工作。")
        else:
            print("⚠️ 测试失败，请检查相关功能。")
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试异常: {e}")


if __name__ == '__main__':
    main()