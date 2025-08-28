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
import cv2
import numpy as np
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from objectLocalization.objectDectection.DrinkShelfLocator import DrinkShelfLocator
from objectLocalization.objectDectection.LocateExecuter_hand import ActionExecuter
from objectLocalization.objectDectection.SAM_Annotator import SAMAnnotator
from objectLocalization.objLocalization import layer_mapping, update_layer_mapping
from voice.config import Config

def preview_and_capture_image(drink_type: str, action_executer: ActionExecuter) -> tuple:
    """
    预览拍摄图像并进行实时调整
    Args:
        drink_type: 饮料类型
        action_executer: 机器人执行器
    Returns:
        (image_path, pitch_angle) 保存的图像路径和最终的俯仰角
    """
    print(f"\n=== 预览拍摄 {drink_type} ===")
    
    # 获取饮料的层数映射
    if drink_type not in layer_mapping:
        print(f"错误：未找到 {drink_type} 的层数映射")
        return None, None
    
    layer, pitch_angle, body_distance = layer_mapping[drink_type]
    print(f"当前设置：层{layer}, 俯仰角{pitch_angle}, 高度{body_distance}")
    
    # 移动机器人到指定位置
    print("正在移动机器人到拍摄位置...")
    action_executer.move_to_pick_height_pitch_angle(
        action_executer.handle_l, 
        action_executer.handle_r, 
        action_executer.add_data_1, 
        body_distance, 
        pitch_angle
    )
    
    # 初始化相机
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开相机")
        return None, None
    
    print("\n预览控制说明：")
    print("- 上方向键 或 W键：增加俯仰角")
    print("- 下方向键 或 S键：减少俯仰角")
    print("- 回车键：保存图像并继续")
    print("- ESC键：跳过保存直接继续")
    print("- 如果方向键不响应，请尝试使用W/S键")
    print(f"- 当前俯仰角: {pitch_angle}°")
    
    image_path = None
    
    while True:
        # 拍摄图像
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取图像")
            break
        
        # 添加白色掩码（与DrinkShelfLocator.capture_image一致）
        frame[0:200, :, :] = 255
        frame[350:, :, :] = 255
        
        # 显示图像 - 使用固定窗口名称，避免创建多个窗口
        window_name = f'预览 {drink_type}'
        cv2.imshow(window_name, frame)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        # 调试：打印按键码（仅在按下非ESC/回车键时）
        # if key != 0 and key != 27 and key != 13:
        #     print(f"调试：按下键码: {key}")
        
        if key == 27:  # ESC键
            print("跳过保存，直接进入下一步")
            break
        elif key == 13:  # 回车键
            # 保存图像
            reference_dir = "objectLocalization/objectDectection/reference_images"
            os.makedirs(reference_dir, exist_ok=True)
            image_path = os.path.join(reference_dir, f"{drink_type}_ref.jpg")
            cv2.imwrite(image_path, frame)
            print(f"图像已保存到: {image_path}")
            
            # 更新层数映射
            update_layer_mapping(drink_type, layer, pitch_angle, body_distance)
            break
        elif key in [82, 2490368, 119, 87]:  # 上方向键 (82, 2490368) 或 W键 (119, 87)
            pitch_angle -= 2
            print(f"俯仰角调整为: {pitch_angle}°")
            try:
                action_executer.move_to_pick_height_pitch_angle(
                    action_executer.handle_l, 
                    action_executer.handle_r, 
                    action_executer.add_data_1, 
                    body_distance, 
                    pitch_angle
                )
            except Exception as e:
                print(f"机器人移动失败: {e}")
        elif key in [84, 2621440, 115, 83]:  # 下方向键 (84, 2621440) 或 S键 (115, 83)
            pitch_angle += 2
            print(f"俯仰角调整为: {pitch_angle}°")
            try:
                action_executer.move_to_pick_height_pitch_angle(
                    action_executer.handle_l, 
                    action_executer.handle_r, 
                    action_executer.add_data_1, 
                    body_distance, 
                    pitch_angle
                )
            except Exception as e:
                print(f"机器人移动失败: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return image_path, pitch_angle

def annotate_reference_image(drink_type: str, image_path: str, locator: DrinkShelfLocator) -> bool:
    """
    对参考图像进行SAM标注
    Args:
        drink_type: 饮料类型
        image_path: 图像路径
        locator: 定位器实例
    Returns:
        是否成功获取边界框
    """
    print(f"\n=== SAM标注 {drink_type} ===")
    print("请在图像上点击饮料位置，系统将记录边界框信息")
    print("标注完成后按 'q' 退出")
    
    # 创建SAM标注器
    annotator = SAMAnnotator(model_type="sam")
    
    # 记录最后一次点击的边界框
    last_box = None
    
    # 重写on_click方法来记录边界框
    original_on_click = annotator.on_click
    def custom_on_click(event):
        nonlocal last_box
        original_on_click(event)
        if hasattr(annotator, 'box') and annotator.box is not None:
            last_box = annotator.box.copy()
            print(f"记录边界框: {last_box}")
    
    annotator.on_click = custom_on_click
    
    try:
        # 运行标注
        annotator.run_annotation(image_path)
        
        # 检查是否获取到边界框
        if last_box is not None:
            # 更新bbox_map
            if isinstance(last_box, np.ndarray):
                bbox = [int(x) for x in last_box.flatten()]  # 转换为整数列表
            else:
                bbox = [int(x) for x in last_box]  # 转换为整数
            locator.update_bbox_map(drink_type, bbox)
            print(f"成功获取 {drink_type} 的边界框: {bbox}")
            return True
        else:
            print(f"未获取到 {drink_type} 的边界框")
            return False
            
    except Exception as e:
        print(f"标注过程出错: {str(e)}")
        return False

def main():
    """主使用示例"""
    
    # 1. 初始化系统
    print("=== 初始化智能货架饮料定位系统 ===")
    locator = DrinkShelfLocator(
        model_path="objectLocalization/objectDectection/weights/yoloe-11l-seg.pt",  # 模型路径
        reference_dir="objectLocalization/objectDectection/reference_images",       # 参考图片目录
        template_dir="objectLocalization/objectDectection/position_templates",      # 位置模板目录
        camera_id=0,                            # 相机ID，0为默认相机
        config_dir="objectLocalization/objectDectection/config"                    # 配置文件目录
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
        model_path="objectLocalization/objectDectection/weights/yoloe-11l-seg.pt",  # 模型路径
        reference_dir="objectLocalization/objectDectection/reference_images",       # 参考图片目录
        template_dir="objectLocalization/objectDectection/position_templates",      # 位置模板目录
        camera_id=0,                            # 相机ID，0为默认相机
        config_dir="objectLocalization/objectDectection/config"                    # 配置文件目录
    )
    
    # 定义要标定的饮料类型
    drink_types = Config.drink_list
    
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
        model_path="objectLocalization/objectDectection/weights/yoloe-11l-seg.pt",
        reference_dir="objectLocalization/objectDectection/reference_images",
        template_dir="objectLocalization/objectDectection/position_templates",
        camera_id=0,                            # 相机ID，0为默认相机
        config_dir="objectLocalization/objectDectection/config"                    # 配置文件目录
    )

    # 初始化机器人执行器
    try:
        action_executer = ActionExecuter(
            robot_ip_left=Config.ROBOT_IP_LEFT, 
            robot_ip_right=Config.ROBOT_IP_RIGHT, 
            robot_available=True
        )
    except Exception as e:
        print(f"机器人初始化失败: {str(e)}")
        print("\n无法标定饮料或移动机器人")
    
    while True:
        print("\n=== 智能货架饮料定位系统 ===")
        print("1. 查看已标定的饮料类型")
        print("2. 标定新饮料位置")
        print("3. 查找饮料")
        print("4. 移动机器人到拍摄处")
        print("0. 退出")
        
        choice = input("请选择操作 (0-4): ").strip()
        
        if choice == "0":
            print("退出系统")
            break
        
        elif choice == "1":
            types = locator.get_available_drink_types()
            if types:
                print(f"已标定的饮料类型: {', '.join(types)}")
                for drink_type in types:
                    info = locator.get_calibration_info(drink_type)
                    print(f"{drink_type}: {info}")                
            else:
                print("还没有标定任何饮料类型")
        
        elif choice == "2":
            drink_type = input("请输入要标定的饮料类型: ").strip()
            if drink_type:
                print(f"\n=== 标定 {drink_type} 位置 ===")
                print("标定过程分为三步：")
                print("1. 预览拍摄参考图像")
                print("2. SAM标注获取边界框")
                print("3. 位置标定")
                
                # 第一步：预览拍摄
                print("\n--- 第一步：预览拍摄 ---")
                image_path, pitch_angle = preview_and_capture_image(drink_type, action_executer)
                
                if image_path is None:
                    print("预览拍摄失败，跳过标定")
                    continue
                
                # 第二步：SAM标注
                print("\n--- 第二步：SAM标注 ---")
                annotation_success = annotate_reference_image(drink_type, image_path, locator)
                
                if not annotation_success:
                    print("SAM标注失败，跳过标定")
                    continue
                
                # 第三步：位置标定
                print("\n--- 第三步：位置标定 ---")
                print(f"请在货架所有位置放上 {drink_type}，然后按回车...")
                input("准备好后按回车: ")
                result = locator.calibrate_positions(drink_type)
                print(f"标定结果: {result['message']}")
                
                if result["success"]:
                    print(f"✓ {drink_type} 标定完成！")
                    print(f"  - 参考图像: {image_path}")
                    print(f"  - 俯仰角: {pitch_angle}")
                    print(f"  - 位置数量: {result['positions_found']}")
                else:
                    print(f"✗ {drink_type} 标定失败: {result['message']}")
        
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
            
            result = locator.find_drinks(drink_type, quantity, "auto")
            print(f"查找结果: {result['message']}")
            if result["success"]:
                print(f"推荐位置: {result['positions']}")
        
        elif choice == "4":
            print("机器人将会移动到拍摄处")
            input("准备好后请按回车确认: ")
            action_executer.move_to_photoshop()
            print("机器人已移动到拍摄处")

        
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
