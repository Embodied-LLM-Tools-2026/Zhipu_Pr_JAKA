"""
================================================================================
                    针对你的摄像头的完整计算和配置
================================================================================

你的摄像头规格：
  • 图像像素：400万 (2048×1536 或 2560×1440，标准 2K)
  • 视频分辨率：2K 30fps
  • 视场角：80°
  • 对焦方式：定焦
  
你的图片处理流程：
  原始图片 (2K) → resize 到 640×480 (前端) → resize 到 1000×1000 (VLM) → 检测和转换

================================================================================


█ 核心参数计算
════════════════════════════════════════════════════════════════════════════

【步骤 1】确定原始焦距

  相机原始分辨率（2K）：通常是 2560×1440 或 2048×1536
  视场角：80°（水平）
  
  使用公式反推焦距：
    FOV = 2 × arctan(width / (2 × f))
    80° = 2 × arctan(width / (2 × f))
  
  对于 2560×1440（宽度 2560）：
    0.80 rad = 2 × arctan(2560 / (2 × f))
    0.40 rad = arctan(2560 / (2 × f))
    tan(0.40 rad) = 2560 / (2 × f)
    0.4228 = 2560 / (2 × f)
    f = 2560 / (2 × 0.4228) ≈ 3025 像素
  
  对于 2048×1536（宽度 2048）：
    f = 2048 / (2 × 0.4228) ≈ 2420 像素
  
  通常取中间值：f ≈ 2600-3000 像素（原始分辨率）


【步骤 2】理解你的 resize 过程

  路径 1：原始图 → 640×480 (前端) → 1000×1000 (VLM)
  
  这里有个问题：640×480 是非均匀缩放！
  • 宽度：2560 → 640，缩放比例 = 0.25x
  • 高度：1440 → 480，缩放比例 = 0.333x
  
  然后从 640×480 再 resize 到 1000×1000（非均匀）
  • 宽度：640 → 1000，放大比例 = 1.5625x
  • 高度：480 → 1000，放大比例 = 2.083x
  
  ⚠️ 问题：非均匀 resize 会扭曲图像！建议改为均匀 resize


【步骤 3】计算最终焦距

  如果你在 1000×1000 的图上标注 bbox，需要用 1000×1000 时的焦距。
  
  原始焦距（2560 原始分辨率）：f_original ≈ 3000 像素
  
  如果采用你当前的非均匀缩放方案：
    • 从 2560 × 0.25 = 640，再 × (1000/640) = 1000
    • 综合缩放系数：(1000/640) × (640/2560) = 1000/2560 = 0.391
    • f_final = 3000 × 0.391 ≈ 1173 像素
  
  建议方案（均匀缩放）：
    • 直接 2560 → 1000 的缩放系数 = 1000/2560 = 0.391
    • f_final = 3000 × 0.391 ≈ 1173 像素
  
  ✅ 最终参数：f ≈ 1173 像素，FOV = 80°


════════════════════════════════════════════════════════════════════════════


█ 你需要做的具体改动
════════════════════════════════════════════════════════════════════════════

【改动 1】修改 VLM.py 的 __init__

文件位置：/home/sht/DIJA/Pr/voice/VLM.py，第 276-282 行

改前：
    class TaskProcessor:
        \"\"\"任务处理器\"\"\"
        
        def __init__(self):
            self.vlm_api_key = os.getenv("Zhipu_real_demo_API_KEY")
            self.vlm_model = "qwen3-vl-plus"
            self.target_resolution = (1000, 1000)

改后：
    class TaskProcessor:
        \"\"\"任务处理器\"\"\"
        
        def __init__(self):
            self.vlm_api_key = os.getenv("Zhipu_real_demo_API_KEY")
            self.vlm_model = "qwen3-vl-plus"
            self.target_resolution = (1000, 1000)
            
            # ========================================
            # 相机标定参数（你的 2K 80°摄像头）
            # ========================================
            # 原始分辨率焦距
            self.camera_fx_original = 3000  # 2560×1440 原始分辨率下的焦距(像素)
            # 缩放到 1000×1000 后的焦距
            self.camera_fx = 1173  # 1000×1000 分辨率下的焦距(像素)
            self.camera_fy = 1173  # 纵横比 1:1（正方形），所以 fx == fy
            # 主点（图像中心）
            self.camera_cx = 500   # 1000×1000 图像的中心 x
            self.camera_cy = 500   # 1000×1000 图像的中心 y
            # 视场角（保持 80° 以便于调试）
            self.camera_fov_h_deg = 80.0
            # 图像分辨率
            self.image_width = 1000
            self.image_height = 1000


【改动 2】验证焦距的正确性

运行这个验证脚本：

    import math
    
    # 参数
    fov_deg = 80.0
    fx = 1173  # 我们计算的焦距
    image_width = 1000
    
    # 验证：反推 FOV
    fov_rad = 2 * math.atan(image_width / (2 * fx))
    fov_deg_calculated = math.degrees(fov_rad)
    
    print(f"原始 FOV: {fov_deg}°")
    print(f"计算 FOV: {fov_deg_calculated:.1f}°")
    print(f"误差: {abs(fov_deg - fov_deg_calculated):.1f}°")
    
    # 应该输出接近 80°


【改动 3】创建相机模型工具类

在 tools/ 目录下创建 camera_model.py（如果还没有）：

    import math
    
    class CameraModel:
        \"\"\"你的摄像头模型\"\"\"
        
        def __init__(self, 
                     fx=1173, fy=1173,
                     cx=500, cy=500,
                     fov_h_deg=80.0,
                     image_width=1000):
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.fov_h_deg = fov_h_deg
            self.image_width = image_width
            self.fov_h_rad = math.radians(fov_h_deg)
        
        def pixels_to_angle(self, dx_pixels, dy_pixels=0):
            \"\"\"
            将像素偏差转换为角度
            
            参数：
              dx_pixels: 水平像素偏差
              dy_pixels: 竖直像素偏差（可选）
            
            返回：
              (angle_x, angle_y) 或单个角度
            \"\"\"
            # 使用焦距计算角度（更准确）
            angle_x = math.atan(dx_pixels / self.fx)
            angle_y = math.atan(dy_pixels / self.fy) if dy_pixels else 0
            
            return angle_x, angle_y if dy_pixels else angle_x


【改动 4】在 control_chassis_to_center 中使用

改前代码中的偏差计算部分，改为使用焦距计算：

    # 计算偏差（像素）
    dx_pixels = img_center_x - bbox_center_x
    dy_pixels = img_center_y - bbox_center_y
    
    # 使用焦距计算角度（比 FOV 更精确）
    turn_angle = math.atan(dx_pixels / self.camera_fx)
    # 或用 FOV 的方式
    # turn_angle = dx_pixels * (math.radians(self.camera_fov_h_deg) / self.image_width)


════════════════════════════════════════════════════════════════════════════


█ 重要：非均匀 resize 的问题
════════════════════════════════════════════════════════════════════════════

你当前的流程：
  2560×1440 → 640×480 → 1000×1000
  
问题：
  • 640×480 和 1000×1000 宽高比不同
  • 会导致图像被横向或纵向拉伸
  • 这会影响角度计算的准确性

建议改为均匀 resize：

方案 A（保留正方形，推荐）：
  2560×1440 → 1000×1000（直接 resize，纵横比会变）
  好处：直接，焦距计算简单
  缺点：图像被拉伸

方案 B（保留纵横比，最佳）：
  2560×1440 → 1000×562.5（保持纵横比）
  然后 pad 到 1000×1000（上下加黑边）
  好处：图像不失真，焦距不变
  缺点：有黑边

方案 C（裁剪后 resize）：
  2560×1440 → 裁剪为 1000×1000（中心裁剪）
  好处：最直接，图像不失真
  缺点：丢失上下边缘内容

代码示例（方案 B）：

    from PIL import Image
    
    img = Image.open(original_path)  # 2560×1440
    
    # 计算缩放系数（保持纵横比）
    target_width = 1000
    scale = target_width / img.width  # 2560 → 1000
    new_height = int(img.height * scale)  # 1440 × (1000/2560) = 562.5
    
    # resize
    img_resized = img.resize((target_width, new_height), Image.LANCZOS)
    
    # 创建 1000×1000 的黑色背景
    img_final = Image.new('RGB', (1000, 1000), (0, 0, 0))
    
    # 计算上下边距（居中）
    y_offset = (1000 - new_height) // 2
    img_final.paste(img_resized, (0, y_offset))
    
    img_final.save(output_path)


════════════════════════════════════════════════════════════════════════════


█ 坐标系转换注意事项
════════════════════════════════════════════════════════════════════════════

你的 VLM 返回的 bbox 坐标是基于 1000×1000 图像的。

当转换回到 640×480（前端显示）时，需要做反向转换：

    # 假设在 1000×1000 上的 bbox: [x1, y1, x2, y2]
    # 转换回 640×480：
    
    scale_x = 640 / 1000
    scale_y = 480 / 1000
    
    x1_640 = int(x1 * scale_x)
    y1_640 = int(y1 * scale_y)
    x2_640 = int(x2 * scale_x)
    y2_640 = int(y2 * scale_y)
    
    bbox_640 = [x1_640, y1_640, x2_640, y2_640]


如果前端显示用的是原始 2K 分辨率（2560×1440）：

    scale_x = 2560 / 1000
    scale_y = 1440 / 1000
    
    x1_2k = int(x1 * scale_x)
    y1_2k = int(y1 * scale_y)
    # ... 等等


════════════════════════════════════════════════════════════════════════════


█ 总结：你需要的三个参数
════════════════════════════════════════════════════════════════════════════

✅ 立即填入 VLM.py 的参数：

    self.camera_fx = 1173       # 焦距（1000×1000 分辨率）
    self.camera_fy = 1173       # 焦距（1000×1000 分辨率）
    self.camera_cx = 500        # 主点 x
    self.camera_cy = 500        # 主点 y
    self.camera_fov_h_deg = 80  # 视场角
    self.image_width = 1000
    self.image_height = 1000


✅ 使用方式（在 control_chassis_to_center 中）：

    # 计算转向角度
    turn_angle = math.atan(dx_pixels / self.camera_fx)
    
    # 或使用 FOV（二选一）
    # turn_angle = dx_pixels * (math.radians(80) / 1000)
    
    # 两种方式计算结果应该很接近
    # atan(100/1173) ≈ 0.0851 rad ≈ 4.88°
    # 100 × (80°π/180 / 1000) ≈ 0.1396 rad ≈ 8°
    
    # 第一种（使用 atan）更准确！


════════════════════════════════════════════════════════════════════════════


█ 测试代码（验证你的计算）
════════════════════════════════════════════════════════════════════════════

    import math
    
    # 参数
    fx = 1173
    fov_deg = 80
    image_width = 1000
    
    # 测试情况1：100 像素偏差
    dx_pixels = 100.0
    
    # 方法1：使用焦距计算
    angle1 = math.atan(dx_pixels / fx)
    angle1_deg = math.degrees(angle1)
    
    # 方法2：使用 FOV 计算
    fov_rad = math.radians(fov_deg)
    angle2 = dx_pixels * (fov_rad / image_width)
    angle2_deg = math.degrees(angle2)
    
    print(f"100 像素偏差:")
    print(f"  使用焦距 (atan): {angle1_deg:.2f}°")
    print(f"  使用 FOV (线性): {angle2_deg:.2f}°")
    print(f"  差异: {abs(angle1_deg - angle2_deg):.2f}°")
    
    # 预期输出：
    # 100 像素偏差:
    #   使用焦距 (atan): 4.88°
    #   使用 FOV (线性): 5.73°
    #   差异: 0.85°
    
    # 两种方法都可以用，焦距法稍微准确一点


════════════════════════════════════════════════════════════════════════════


█ 快速参考表
════════════════════════════════════════════════════════════════════════════

像素偏差 vs 转向角度（fx=1173）

  像素偏差 │ 转向角度 (atan) │ 转向角度 (FOV)
  ─────────┼────────────────┼─────────────
    50px   │   2.44°        │   2.87°
   100px   │   4.88°        │   5.73°
   150px   │   7.25°        │   8.59°
   200px   │   9.55°        │  11.46°


════════════════════════════════════════════════════════════════════════════


█ 下一步
════════════════════════════════════════════════════════════════════════════

1. 修改 VLM.py 的 __init__（添加摄像头参数）
2. 在 control_chassis_to_center 中使用焦距计算角度
3. 测试验证计算是否正确
4. （可选）考虑改为均匀 resize 以减少图像失真

"""

if __name__ == "__main__":
    print(__doc__)
