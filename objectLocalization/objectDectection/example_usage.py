"""
智能货架饮料定位系统使用示例

使用流程：
1. 准备参考图片：将各种饮料的参考图片放在 reference_images/ 目录下
   命名格式：{饮料类型}_ref.jpg （如：可乐_ref.jpg）

2. 位置标定：对每种饮料进行一次位置标定
   - 在货架的所有可能位置都放上该种饮料
   - 调用 calibrate_positions() 方法

3. 实际使用：调用 find_drinks() 方法查找饮料位置
"""
import sys
import os
import time

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from objectLocalization.objectDectection.DrinkShelfLocator import DrinkShelfLocator


def main():
    """主使用示例"""
    
    # 1. 初始化系统
    print("=== 初始化智能货架饮料定位系统 ===")
    locator = DrinkShelfLocator(
        model_path="objectLocalization/objectDectection/weights/yoloe-11s-seg.pt",  # 模型路径
        reference_dir="objectLocalization/objectDectection/reference_images",       # 参考图片目录
        template_dir="objectLocalization/objectDectection/position_templates",      # 位置模板目录
        camera_id=0                             # 相机ID，0为默认相机
    )
    
    # 2. 显示已标定的饮料类型
    available_types = locator.get_available_drink_types()
    print(f"已标定的饮料类型: {available_types}")
    
    # 3. 位置标定示例（仅在首次使用时需要）
    if not available_types:
        print("\n=== 位置标定示例 ===")
        print("首次使用需要进行位置标定")
        
        # 标定可乐位置
        print("\n--- 标定位置 ---")
        print("请在货架的所有可能位置都放上雪碧，然后按回车继续...")
        input("准备好后按回车: ")
        
        calibration_result = locator.calibrate_positions("雪碧")
        print(f"雪碧标定结果: {calibration_result}")
        
        if calibration_result["success"]:
            print(f"成功标定了 {calibration_result['positions_found']} 个位置")
        else:
            print("标定失败，请检查：")
            print("1. 参考图片是否存在：reference_images/雪碧_ref.jpg")
            print("2. 货架上是否正确放置了雪碧")
            print("3. 相机是否正常工作")
            return
    
    # 4. 饮料查找示例
    print("\n=== 饮料查找示例 ===")
    
    # 查找3瓶雪碧
    print("\n--- 查找雪碧 ---")
    print("请调整货架上的雪碧分布（可以移除一些），然后按回车继续...")
    input("准备好后按回车: ")
    
    search_result = locator.find_drinks("雪碧", 3)
    print(f"查找结果: {search_result}")
    
    if search_result["success"]:
        positions = search_result["positions"]
        print(f"找到 {search_result['found_count']} 瓶雪碧")
        print(f"推荐位置编号: {positions}")
        print(f"货架上总共有 {search_result['total_available']} 瓶雪碧")
    else:
        print("查找失败：", search_result["message"])
    
    # 5. 显示标定信息
    print("\n=== 系统信息 ===")
    for drink_type in locator.get_available_drink_types():
        info = locator.get_calibration_info(drink_type)
        if info["available"]:
            print(f"{drink_type}: {info['total_positions']}个位置, 标定时间: {info['calibration_date']}")
        else:
            print(f"{drink_type}: {info['message']}")


def batch_calibration_example():
    """批量标定示例 - 适用于多种饮料的标定"""
    
    locator = DrinkShelfLocator(
        model_path="objectLocalization/objectDectection/weights/yoloe-11s-seg.pt",  # 模型路径
        reference_dir="objectLocalization/objectDectection/reference_images",       # 参考图片目录
        template_dir="objectLocalization/objectDectection/position_templates",      # 位置模板目录
        camera_id=0
    )
    
    # 定义要标定的饮料类型
    drink_types = ["可乐", "雪碧", "柠檬茶", "奶茶"]
    
    print("=== 批量位置标定 ===")
    for drink_type in drink_types:
        print(f"\n--- 标定 {drink_type} ---")
        print(f"请在货架所有位置放上 {drink_type}，然后按回车继续...")
        input(f"准备标定 {drink_type} 后按回车: ")
        
        result = locator.calibrate_positions(drink_type)
        if result["success"]:
            print(f"✓ {drink_type} 标定成功：{result['positions_found']} 个位置")
        else:
            print(f"✗ {drink_type} 标定失败：{result['message']}")
        
        # 短暂休息
        time.sleep(1)
    
    print("\n标定完成！")


def interactive_mode():
    """交互式使用模式"""
    
    locator = DrinkShelfLocator(
        model_path="objectLocalization/objectDectection/weights/yoloe-11s-seg.pt",
        reference_dir="objectLocalization/objectDectection/reference_images",
        template_dir="objectLocalization/objectDectection/position_templates",
        camera_id=0
    )
    
    while True:
        print("\n=== 智能货架饮料定位系统 ===")
        print("1. 查看已标定的饮料类型")
        print("2. 标定新饮料位置")
        print("3. 查找饮料")
        print("4. 查看标定信息")
        print("0. 退出")
        
        choice = input("请选择操作 (0-4): ").strip()
        
        if choice == "0":
            print("退出系统")
            break
        
        elif choice == "1":
            types = locator.get_available_drink_types()
            if types:
                print(f"已标定的饮料类型: {', '.join(types)}")
            else:
                print("还没有标定任何饮料类型")
        
        elif choice == "2":
            drink_type = input("请输入要标定的饮料类型: ").strip()
            if drink_type:
                print(f"请在货架所有位置放上 {drink_type}，然后按回车...")
                input("准备好后按回车: ")
                result = locator.calibrate_positions(drink_type)
                print(f"标定结果: {result['message']}")
        
        elif choice == "3":
            available_types = locator.get_available_drink_types()
            if not available_types:
                print("没有已标定的饮料类型，请先进行标定")
                continue
            
            print(f"可选饮料类型: {', '.join(available_types)}")
            drink_type = input("请输入要查找的饮料类型: ").strip()
            
            if drink_type not in available_types:
                print("该饮料类型未标定")
                continue
            
            try:
                quantity = int(input("请输入需要的数量: ").strip())
                if quantity <= 0:
                    print("数量必须大于0")
                    continue
            except ValueError:
                print("请输入有效的数量")
                continue
            
            print("请调整货架，然后按回车开始查找...")
            input("准备好后按回车: ")
            
            result = locator.find_drinks(drink_type, quantity)
            print(f"查找结果: {result['message']}")
            if result["success"]:
                print(f"推荐位置: {result['positions']}")
        
        elif choice == "4":
            types = locator.get_available_drink_types()
            if types:
                for drink_type in types:
                    info = locator.get_calibration_info(drink_type)
                    print(f"{drink_type}: {info}")
            else:
                print("没有标定信息")
        
        else:
            print("无效选择")


if __name__ == "__main__":
    # 选择运行模式
    print("请选择运行模式：")
    print("1. 完整示例")
    print("2. 批量标定")
    print("3. 交互模式")
    
    mode = input("请输入模式编号 (1-3): ").strip()
    
    try:
        if mode == "1":
            main()
        elif mode == "2":
            batch_calibration_example()
        elif mode == "3":
            interactive_mode()
        else:
            print("无效选择，运行默认示例")
            main()
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
    except Exception as e:
        print(f"程序出错: {str(e)}")
        print("请检查系统配置和依赖")
