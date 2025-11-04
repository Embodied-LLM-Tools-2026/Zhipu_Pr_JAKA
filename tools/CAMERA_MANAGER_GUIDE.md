# 多摄像头管理工具使用指南

## 功能概述

升级后的 `detect_camera.py` 现在支持：
- ✅ **同时打开多个摄像头**（RGB、深度等）
- ✅ **实时显示所有摄像头画面**
- ✅ **读取深度图参数**（size、像素深度值等）
- ✅ **显示摄像头统计信息**（帧数、分辨率、FPS、深度范围等）
- ✅ **交互式键盘快捷键**

## 核心类：`CameraManager`

### 基本用法

```python
from tools.detect_camera import CameraManager

# 创建管理器
manager = CameraManager()

# 添加 RGB 摄像头
manager.add_camera('rgb', 3, camera_type='rgb')

# 添加深度摄像头
manager.add_camera('depth', 12, camera_type='depth')

# 同时显示所有摄像头
manager.display_all(duration=None)  # None = 无限制，直到按 'q' 或 ESC

# 显示统计信息
manager.print_stats()
```

### API 详解

#### 1. `add_camera(name, index, camera_type='rgb')`
添加一个摄像头到管理器。

**参数：**
- `name` (str): 摄像头名称，用于标识（如 'rgb', 'depth', 'left', 'right'）
- `index` (int): 摄像头索引（0, 1, 2, ...）
- `camera_type` (str): 摄像头类型 ('rgb' 或 'depth')

**返回：** True（成功）或 False（失败）

**示例：**
```python
manager.add_camera('front_rgb', 3, camera_type='rgb')
manager.add_camera('depth_sensor', 12, camera_type='depth')
```

#### 2. `read_frame(name)`
读取指定摄像头的一帧。

**返回：** numpy array（图像数据）或 None

#### 3. `get_depth_stats(name)`
获取深度摄像头的统计信息。

**返回字典包含：**
- `size`: (width, height) - 图像尺寸
- `depth_min`: 最小深度值
- `depth_max`: 最大深度值
- `depth_mean`: 平均深度值
- `center_pixel`: 中心像素的深度值

**示例：**
```python
stats = manager.get_depth_stats('depth')
print(f"深度范围: {stats['depth_min']} - {stats['depth_max']}")
print(f"图像大小: {stats['size']}")
print(f"中心像素深度: {stats['center_pixel']}")
```

#### 4. `get_pixel_depth(name, x, y)`
获取特定像素 (x, y) 的深度值。

**返回：** 该像素的深度值（0-255）

**示例：**
```python
depth_at_center = manager.get_pixel_depth('depth', 320, 240)
print(f"(320, 240) 的深度值: {depth_at_center}")
```

#### 5. `display_all(duration=None)`
同时显示所有已添加的摄像头。

**参数：**
- `duration` (float/None): 显示时长（秒），None 表示无限制

**键盘快捷键：**
- `'q'` 或 `ESC`: 退出显示
- `'s'`: 打印统计信息到控制台

#### 6. `print_stats()`
打印所有摄像头的详细统计信息。

#### 7. `release_all()`
释放所有摄像头资源（自动在显示退出时调用）。

---

## 使用示例

### 示例 1: 同时显示两个 RGB 摄像头

```python
from tools.detect_camera import CameraManager

manager = CameraManager()
manager.add_camera('left', 2, camera_type='rgb')
manager.add_camera('right', 5, camera_type='rgb')
manager.display_all()
manager.print_stats()
```

### 示例 2: RGB + 深度摄像头（获取像素深度值）

```python
from tools.detect_camera import CameraManager
import time

manager = CameraManager()
manager.add_camera('rgb', 3, camera_type='rgb')
manager.add_camera('depth', 12, camera_type='depth')

# 启动实时显示（后台线程处理）
# 或者在另一个线程中调用

# 实时查询深度信息
while True:
    # 获取中心像素深度值
    center_depth = manager.get_pixel_depth('depth', 320, 240)
    if center_depth is not None:
        print(f"中心像素深度: {center_depth}")
    
    # 获取整体深度统计
    stats = manager.get_depth_stats('depth')
    if stats:
        print(f"深度范围: {stats['depth_min']} - {stats['depth_max']}")
    
    time.sleep(0.5)
```

### 示例 3: 多摄像头拼接显示

```python
from tools.detect_camera import CameraManager
import cv2

manager = CameraManager()
manager.add_camera('front', 3)
manager.add_camera('left', 2)
manager.add_camera('right', 5)
manager.add_camera('depth', 12, camera_type='depth')

# 自定义显示逻辑
while True:
    frames = {}
    for name in manager.cameras:
        frame = manager.read_frame(name)
        if frame is not None:
            frames[name] = frame
    
    if not frames:
        break
    
    # 自定义拼接或处理逻辑
    # ... 你的代码 ...
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

manager.release_all()
```

---

## 运行方式

### 直接运行（使用默认配置）

```bash
cd /home/sht/DIJA/Pr
python3 tools/detect_camera.py
```

这会启动：
- RGB 摄像头（索引 3）
- 深度摄像头（索引 12）

### 自定义脚本

创建 `my_camera_app.py`：

```python
from tools.detect_camera import CameraManager

manager = CameraManager()

# 根据你的硬件配置修改索引
manager.add_camera('rgb_main', 3, camera_type='rgb')
manager.add_camera('depth_main', 12, camera_type='depth')

# 添加更多摄像头（根据需要）
# manager.add_camera('rgb_secondary', 2, camera_type='rgb')

# 显示
manager.display_all()
```

运行：
```bash
python3 my_camera_app.py
```

---

## 常见问题

### Q: 如何找到可用的摄像头索引？

**A:** 使用以下脚本扫描可用摄像头：

```python
import cv2

for i in range(20):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera {i}: {w}x{h}")
        cap.release()
```

### Q: 深度值的范围是什么？

**A:** 对于灰度深度图像，范围是 0-255。对于真实深度传感器（如 RealSense），可能是毫米级别的实际距离值。当前脚本将彩色图转换为灰度来读取"深度"。

### Q: 可以同时添加无限个摄像头吗？

**A:** 理论上可以，但受硬件和 USB 带宽限制。建议同时使用不超过 4-6 个摄像头以保证流畅性。

### Q: 如何只显示特定摄像头？

**A:** 只为需要的摄像头调用 `add_camera()`，其他摄像头不添加即可。

---

## 扩展建议

- 集成真实深度传感器库（pyrealsense2、kinect-azure 等）
- 添加录制视频功能
- 支持图像处理管道（检测、追踪等）
- 添加线程池以提高多摄像头性能

