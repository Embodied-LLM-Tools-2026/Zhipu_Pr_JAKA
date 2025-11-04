# 多摄像头管理工具 - 快速参考

## 快速开始

### 方式 1: 运行默认示例（同时显示 RGB + 深度）

```bash
cd /home/sht/DIJA/Pr
python3 tools/detect_camera.py
```

**快捷键：**
- `q` / `ESC` - 退出
- `s` - 显示统计信息

---

### 方式 2: 运行交互式示例

```bash
python3 tools/camera_examples.py
```

选择要运行的示例（1-4）

---

## 最小代码示例

```python
from tools.detect_camera import CameraManager

# 创建管理器
manager = CameraManager()

# 添加摄像头
manager.add_camera('rgb', 3, camera_type='rgb')
manager.add_camera('depth', 12, camera_type='depth')

# 显示所有摄像头
manager.display_all()

# 获取统计信息
manager.print_stats()
```

---

## 常用 API

| 方法 | 说明 | 示例 |
|------|------|------|
| `add_camera(name, idx)` | 添加摄像头 | `manager.add_camera('rgb', 3)` |
| `read_frame(name)` | 读取单帧 | `frame = manager.read_frame('rgb')` |
| `get_depth_stats(name)` | 获取深度统计 | `stats = manager.get_depth_stats('depth')` |
| `get_pixel_depth(name, x, y)` | 获取像素深度值 | `depth = manager.get_pixel_depth('depth', 320, 240)` |
| `display_all(duration)` | 显示所有摄像头 | `manager.display_all()` |
| `print_stats()` | 打印统计信息 | `manager.print_stats()` |
| `release_all()` | 释放资源 | `manager.release_all()` |

---

## 深度数据读取

### 获取图像尺寸
```python
stats = manager.get_depth_stats('depth')
width, height = stats['size']  # (w, h)
```

### 获取深度值范围
```python
depth_min = stats['depth_min']
depth_max = stats['depth_max']
depth_mean = stats['depth_mean']
```

### 获取特定像素深度值
```python
# 获取 (x, y) 像素的深度
depth_value = manager.get_pixel_depth('depth', x, y)

# 获取中心像素
stats = manager.get_depth_stats('depth')
center_pixel_depth = stats['center_pixel']
```

---

## 文件结构

```
Pr/
├── tools/
│   ├── detect_camera.py              # 主要工具类
│   ├── camera_examples.py             # 交互式示例
│   └── CAMERA_MANAGER_GUIDE.md        # 完整文档
└── ...
```

---

## 摄像头索引参考

当前已知的摄像头索引：
- **索引 2**: RGB 摄像头（备选）
- **索引 3**: RGB 摄像头（主要）
- **索引 5**: RGB 摄像头（右侧，备选）
- **索引 6**: RGB 摄像头（右侧）
- **索引 12**: 深度摄像头

要查找更多摄像头，运行：
```bash
python3 -c "from tools.detect_camera import *; import cv2; [print(f'Camera {i}') for i in range(20) if cv2.VideoCapture(i).isOpened()]"
```

---

## 常见问题

**Q: 如何同时显示 4 个摄像头？**
```python
manager.add_camera('cam1', 2)
manager.add_camera('cam2', 3)
manager.add_camera('cam3', 5)
manager.add_camera('cam4', 12)
manager.display_all()
```

**Q: 如何只获取深度数据而不显示？**
```python
manager.add_camera('depth', 12, camera_type='depth')
for _ in range(100):
    manager.read_frame('depth')
    stats = manager.get_depth_stats('depth')
    print(stats)
```

**Q: 如何保存深度值？**
```python
depth_values = []
for _ in range(100):
    frame = manager.read_frame('depth')
    if frame is not None:
        depth_values.append(frame.copy())
# 后续处理 depth_values
```

---

## 扩展建议

- 支持实时深度滤波（中位数、高斯）
- 添加深度图颜色映射（热力图）
- 集成 OpenCV 目标检测
- 支持多线程实时处理
- 添加视频录制功能

