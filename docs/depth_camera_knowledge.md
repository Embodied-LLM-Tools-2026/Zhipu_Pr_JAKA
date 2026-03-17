# 深度相机知识详解

## 一、什么是深度相机？

### 1.1 定义

**深度相机**（Depth Camera）是一种能够获取场景中每个像素点到相机距离信息的相机。

```
普通相机：RGB图像 → 每个像素有颜色信息 (R, G, B)
深度相机：深度图 → 每个像素有距离信息 (Z)

RGBD相机：同时获取 RGB + Depth
```

### 1.2 为什么需要深度相机？

```
┌─────────────────────────────────────────────────────────────┐
│                    传统相机的局限                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  问题 1：无法知道物体距离                                     │
│  ┌─────────────┐                                             │
│  │   🍎 苹果   │  ← 多大？多远？                              │
│  └─────────────┘                                             │
│                                                              │
│  问题 2：无法进行3D定位                                       │
│  - 机器人抓取需要知道物体的3D位置                            │
│  - 导航避障需要知道障碍物的距离                              │
│                                                              │
│  问题 3：无法重建3D场景                                       │
│  - SLAM需要深度信息                                          │
│  - 场景理解需要3D信息                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 深度相机的优势

| 特性 | 说明 |
|------|------|
| **3D感知** | 直接获取场景的三维信息 |
| **距离测量** | 知道每个像素的距离 |
| **物体定位** | 可以计算物体的3D坐标 |
| **避障导航** | 检测障碍物距离 |
| **场景重建** | 构建3D点云地图 |

---

## 二、深度相机的工作原理

### 2.1 主流技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                    深度相机技术路线                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 结构光（Structured Light）                               │
│     ┌───────────────────────────────────────────────┐       │
│     │  投射器 ──→ 投射已知图案 ──→ 场景              │       │
│     │                                              │       │
│     │  相机 ──→ 拍摄变形图案 ──→ 计算深度           │       │
│     └───────────────────────────────────────────────┘       │
│     代表：Intel RealSense D400系列（早期）、iPhone Face ID   │
│     优点：精度高、室内效果好                                 │
│     缺点：受环境光影响、户外效果差                           │
│                                                              │
│  2. 双目立体视觉（Stereo Vision）                            │
│     ┌───────────────────────────────────────────────┐       │
│     │     左相机 ──→ 图像L ──┐                       │       │
│     │                       ├──→ 视差 ──→ 深度      │       │
│     │     右相机 ──→ 图像R ──┘                       │       │
│     └───────────────────────────────────────────────┘       │
│     代表：Intel RealSense D435、ZED相机                      │
│     优点：成本低、户外可用                                   │
│     缺点：计算量大、依赖物体纹理                             │
│                                                              │
│  3. 飞行时间（ToF, Time of Flight）                          │
│     ┌───────────────────────────────────────────────┐       │
│     │  发射器 ──→ 发射红外光 ──→ 场景               │       │
│     │                                              │       │
│     │  接收器 ──→ 接收反射光 ──→ 计算往返时间       │       │
│     │                                              │       │
│     │  距离 = 光速 × 往返时间 / 2                   │       │
│     └───────────────────────────────────────────────┘       │
│     代表：Orbbec Femto、Azure Kinect、iPhone LiDAR           │
│     优点：测量范围大、抗干扰                                 │
│     缺点：分辨率较低、成本较高                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 本项目使用的相机

**Orbbec（奥比中光）深度相机**

```python
# 使用的SDK
from pyorbbecsdk import (
    Pipeline,          # 相机管道
    Config,            # 配置
    AlignFilter,       # 对齐滤波器
    OBSensorType,      # 传感器类型
    OBFormat,          # 数据格式
    transformation2dto3d,  # 2D到3D转换
)

# 相机型号示例
# - Orbbec Femto Bolt: ToF相机，适合机器人导航
# - Orbbec Astra: 结构光相机，适合室内近距离
```

---

## 三、深度相机的核心参数

### 3.1 内参（Intrinsics）

**内参**描述相机的光学特性，用于像素坐标到相机坐标的转换。

```
┌─────────────────────────────────────────────────────────────┐
│                      相机内参                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│              图像平面                                        │
│           ┌─────────────┐                                    │
│           │      cy     │                                    │
│           │      ↓      │                                    │
│           │      •──────┼────── cx ──→                       │
│           │     (主点)  │                                    │
│           │             │                                    │
│           └─────────────┘                                    │
│                                                              │
│  fx, fy: 焦距（像素单位）                                    │
│  cx, cy: 主点坐标（图像中心）                                │
│  width, height: 图像分辨率                                   │
│                                                              │
│  内参矩阵 K:                                                 │
│       │ fx   0   cx │                                        │
│   K = │ 0   fy   cy │                                        │
│       │ 0    0    1 │                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```python
# 代码中的内参
intrinsics = {
    "fx": 607.5,    # x方向焦距（像素）
    "fy": 607.5,    # y方向焦距（像素）
    "cx": 319.5,    # 主点x坐标
    "cy": 239.5,    # 主点y坐标
    "width": 640,   # 图像宽度
    "height": 480,  # 图像高度
}
```

**物理意义**：
- `fx, fy`：焦距，决定了视场角大小。焦距越大，视场角越小，看到的范围越窄
- `cx, cy`：光轴与图像平面的交点，通常接近图像中心

### 3.2 外参（Extrinsics）

**外参**描述相机在世界坐标系中的位置和姿态。

```
┌─────────────────────────────────────────────────────────────┐
│                      相机外参                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  外参矩阵 T (4x4):                                           │
│                                                              │
│       │ r11  r12  r13  tx │                                  │
│   T = │ r21  r22  r23  ty │                                  │
│       │ r31  r32  r33  tz │                                  │
│       │  0    0    0    1 │                                  │
│                                                              │
│  R (旋转矩阵): 3x3，描述相机姿态                              │
│  t (平移向量): 3x1，描述相机位置                              │
│                                                              │
│  用途：                                                       │
│  - 将相机坐标转换到世界坐标                                   │
│  - 多相机标定时描述相机之间的相对位置                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```python
# 代码中的外参
extrinsic = {
    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 旋转矩阵
    "translation": [0, 0, 0],  # 平移向量
}
```

### 3.3 深度值（Depth Value）

```
深度图格式：
- 数据类型：uint16（0-65535）
- 单位：毫米（mm）
- 分辨率：640x480 或 1280x720

示例：
depth[100, 200] = 1500  # 表示该像素距离相机1500mm（1.5米）

特殊值：
- 0: 无效深度（太近、太远或反射）
- 65535: 超出测量范围
```

---

## 四、坐标系转换

### 4.1 坐标系定义

```
┌─────────────────────────────────────────────────────────────┐
│                      坐标系定义                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 像素坐标系 (u, v)                                        │
│     ┌─────────────┐                                          │
│     │(0,0)        │                                          │
│     │    u ──→    │  原点：图像左上角                        │
│     │    │        │  单位：像素                              │
│     │    v        │                                          │
│     └─────────────┘                                          │
│                                                              │
│  2. 图像坐标系 (x, y)                                        │
│     ┌─────────────┐                                          │
│     │      │      │  原点：图像中心                          │
│     │  ────•────  │  单位：物理单位（mm）                    │
│     │      │      │  x = (u - cx) * pixel_size              │
│     └─────────────┘  y = (v - cy) * pixel_size              │
│                                                              │
│  3. 相机坐标系 (Xc, Yc, Zc)                                  │
│           Zc                                                 │
│           ↑                                                  │
│           │                                                  │
│           │                                                  │
│           └────→ Xc                                          │
│          /                                                   │
│         /                                                    │
│        Yc                                                    │
│     原点：相机光心                                            │
│     X轴：向右                                                │
│     Y轴：向下                                                │
│     Z轴：向前（光轴方向）                                    │
│                                                              │
│  4. 机器人坐标系 (Xr, Yr, Zr)                                │
│     根据机器人定义，通常：                                    │
│     原点：机器人基座                                          │
│     X轴：前进方向                                            │
│     Y轴：左侧方向                                            │
│     Z轴：向上方向                                            │
│                                                              │
│  5. 世界坐标系 (Xw, Yw, Zw)                                  │
│     原点：场景中固定点                                        │
│     用于描述环境中物体的绝对位置                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 像素坐标到相机坐标

**核心公式**：将2D像素点转换为3D空间点

```
┌─────────────────────────────────────────────────────────────┐
│                像素坐标 → 相机坐标                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  输入：                                                       │
│    (u, v): 像素坐标                                          │
│    depth: 深度值（mm）                                       │
│    intrinsics: 相机内参                                      │
│                                                              │
│  公式：                                                       │
│    Zc = depth                                                │
│    Xc = (u - cx) * Zc / fx                                   │
│    Yc = (v - cy) * Zc / fy                                   │
│                                                              │
│  推导：                                                       │
│    根据相似三角形原理：                                       │
│    Xc / Zc = (u - cx) / fx                                   │
│    Yc / Zc = (v - cy) / fy                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```python
def pixel_to_camera(u: float, v: float, depth: float, intrinsics: dict):
    """
    像素坐标转相机坐标
    
    Args:
        u, v: 像素坐标
        depth: 深度值（mm）
        intrinsics: 相机内参
    
    Returns:
        Xc, Yc, Zc: 相机坐标系下的3D坐标（mm）
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    
    Zc = depth
    Xc = (u - cx) * Zc / fx
    Yc = (v - cy) * Zc / fy
    
    return Xc, Yc, Zc
```

### 4.3 相机坐标到机器人坐标

```python
def camera_to_robot(Xc, Yc, Zc, extrinsic):
    """
    相机坐标转机器人坐标
    
    Args:
        Xc, Yc, Zc: 相机坐标
        extrinsic: 相机外参（相机到机器人的变换矩阵）
    
    Returns:
        Xr, Yr, Zr: 机器人坐标系下的3D坐标
    """
    # 构建4x1齐次坐标
    point_camera = np.array([Xc, Yc, Zc, 1])
    
    # 获取变换矩阵
    T = extrinsic["matrix"]  # 4x4变换矩阵
    
    # 变换到机器人坐标
    point_robot = T @ point_camera
    
    return point_robot[0], point_robot[1], point_robot[2]
```

### 4.4 完整转换流程

```
┌─────────────────────────────────────────────────────────────┐
│                    坐标转换流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  像素坐标 (u, v)                                             │
│       │                                                      │
│       │ + 深度值 depth                                       │
│       │ + 内参 intrinsics                                   │
│       ↓                                                      │
│  相机坐标 (Xc, Yc, Zc)                                       │
│       │                                                      │
│       │ + 外参 extrinsic                                    │
│       ↓                                                      │
│  机器人坐标 (Xr, Yr, Zr)                                     │
│       │                                                      │
│       │ + 世界变换                                           │
│       ↓                                                      │
│  世界坐标 (Xw, Yw, Zw)                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、项目中的深度相机应用

### 5.1 数据获取

```python
# 代码位置：voice/perception/localize_target.py

from pyorbbecsdk import Pipeline, Config, AlignFilter

def capture_aligned_rgbd():
    """
    同步获取对齐的 RGB 和 Depth 帧
    
    Returns:
        rgb_frame: BGR格式的RGB图像
        snapshot: 深度快照（包含深度图、内参、外参）
    """
    # 1. 创建管道
    pipeline = Pipeline()
    config = Config()
    
    # 2. 配置流
    config.enable_stream(OBSensorType.COLOR_SENSOR, 640, 480, OBFormat.RGB)
    config.enable_stream(OBSensorType.DEPTH_SENSOR, 640, 480, OBFormat.Y16)
    
    # 3. 启动管道
    pipeline.start(config)
    
    # 4. 创建对齐滤波器（将深度对齐到彩色）
    align_filter = AlignFilter(align_to_color=True)
    
    # 5. 获取帧
    frames = pipeline.wait_for_frames(1000)
    
    # 6. 对齐
    aligned_frames = align_filter.process(frames)
    
    # 7. 提取数据
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    # 8. 转换为numpy
    rgb = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    
    # 9. 获取内参
    intrinsics = depth_frame.get_intrinsic()
    
    return rgb, depth, intrinsics
```

### 5.2 深度数据结构

```python
@dataclass
class DepthSnapshot:
    """深度快照数据结构"""
    depth: np.ndarray      # 深度图 (H, W)，单位mm
    intrinsics: Any        # 相机内参
    extrinsic: Any         # 相机外参
    dtype: str = "uint16"  # 数据类型
```

### 5.3 目标定位流程

```python
# 代码位置：voice/perception/localize_target.py

class TargetLocalizer:
    """目标定位器"""
    
    def localize_target(
        self,
        bbox: List[float],      # 目标边界框 [x_min, y_min, x_max, y_max]
        snapshot: DepthSnapshot,
        surface_points: List[List[float]] = None,  # 背景平面点
    ) -> Dict[str, Any]:
        """
        定位目标物体
        
        步骤：
        1. 计算边界框中心
        2. 获取中心点深度
        3. 像素坐标转相机坐标
        4. 相机坐标转机器人坐标
        5. 计算物体高度（如果有背景点）
        """
        # 1. 计算边界框中心
        center_u = (bbox[0] + bbox[2]) / 2
        center_v = (bbox[1] + bbox[3]) / 2
        
        # 2. 获取中心点深度
        depth_value = snapshot.depth[int(center_v), int(center_u)]
        
        # 3. 像素坐标转相机坐标
        Xc, Yc, Zc = self.pixel_to_camera(
            center_u, center_v, depth_value, snapshot.intrinsics
        )
        
        # 4. 相机坐标转机器人坐标
        Xr, Yr, Zr = self.camera_to_robot(Xc, Yc, Zc, snapshot.extrinsic)
        
        return {
            "camera_center": (Xc, Yc, Zc),
            "robot_center": (Xr, Yr, Zr),
            "depth": depth_value,
        }
```

---

## 六、深度相机的标定

### 6.1 标定的目的

```
┌─────────────────────────────────────────────────────────────┐
│                      标定目的                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 内参标定                                                 │
│     - 获取准确的焦距 fx, fy                                  │
│     - 获取准确的主点 cx, cy                                  │
│     - 消除镜头畸变                                           │
│                                                              │
│  2. 外参标定                                                 │
│     - 确定相机在机器人上的位置                               │
│     - 建立相机与机器人坐标系的转换关系                       │
│                                                              │
│  3. 深度标定                                                 │
│     - 校正深度测量误差                                       │
│     - 消除系统性偏差                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 手眼标定（Hand-Eye Calibration）

```
场景：相机安装在机器人末端

目的：求取相机与机器人末端的变换矩阵

步骤：
1. 移动机器人到多个不同位置
2. 在每个位置拍摄标定板
3. 记录机器人位姿和标定板位姿
4. 求解 AX = XB 方程

常用工具：
- OpenCV: cv2.calibrateHandEye()
- ROS: hand_eye_calibration 包
- EasyHandEye: 图形化标定工具
```

---

## 七、常见问题与解决方案

### 7.1 深度数据缺失

```
问题：某些区域深度值为0或无效

原因：
1. 物体表面反光（镜面、金属）
2. 物体吸收红外光（黑色物体）
3. 超出测量范围（太近或太远）

解决方案：
1. 使用多帧平均
2. 使用形态学滤波填充
3. 结合RGB信息推断
```

```python
def fill_depth_holes(depth: np.ndarray) -> np.ndarray:
    """填充深度图空洞"""
    # 使用形态学闭运算
    kernel = np.ones((5, 5), np.uint8)
    depth_filled = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    
    # 使用双边滤波平滑
    depth_smooth = cv2.bilateralFilter(
        depth_filled.astype(np.float32), 
        d=5, sigmaColor=30, sigmaSpace=30
    )
    
    return depth_smooth.astype(np.uint16)
```

### 7.2 深度噪声

```
问题：深度值存在噪声，不够稳定

解决方案：
1. 时域滤波：多帧平均
2. 空域滤波：双边滤波
3. 置信度过滤：只使用高置信度区域
```

### 7.3 RGB与深度不对齐

```
问题：RGB图像和深度图像素不对应

原因：
- RGB相机和深度相机物理位置不同
- 两个相机分辨率可能不同

解决方案：
- 使用对齐滤波器（AlignFilter）
- 将深度图对齐到RGB图
```

```python
# 使用对齐滤波器
align_filter = AlignFilter(align_to_color=True)
aligned_frames = align_filter.process(frames)
```

---

## 八、项目代码结构

```
voice/perception/
├── localize_target.py     # 深度相机核心模块
│   ├── DepthSnapshot      # 深度快照数据结构
│   ├── fetch_snapshot()   # 从API获取深度数据
│   ├── fetch_aligned_rgbd()  # 获取RGBD数据
│   ├── TargetLocalizer    # 目标定位器
│   ├── pixel_to_camera()  # 像素→相机坐标
│   └── camera_to_robot()  # 相机→机器人坐标
│
├── observer.py            # VLM观测模块
│   └── VLMObserver        # 结合VLM和深度相机
│
├── sam_worker.py          # SAM分割模块
│   └── sam_mask_worker()  # 生成物体mask
│
└── catalog_worker.py      # 场景物体识别
    └── SceneCatalogWorker # 列出场景中所有物体
```

---

## 九、关键代码示例

### 9.1 完整的目标定位流程

```python
def detect_and_localize(target_name: str):
    """检测目标并定位"""
    
    # 1. 获取RGBD数据
    rgb_frame, snapshot = fetch_aligned_rgbd()
    
    # 2. VLM检测目标
    vlm_result = vlm_detect(rgb_frame, target_name)
    bbox = vlm_result["bbox"]  # [x_min, y_min, x_max, y_max]
    
    # 3. 深度定位
    localizer = TargetLocalizer()
    result = localizer.localize_target(
        bbox=bbox,
        snapshot=snapshot,
        surface_points=vlm_result.get("surface_points"),
    )
    
    # 4. 输出结果
    print(f"目标: {target_name}")
    print(f"相机坐标: {result['camera_center']} mm")
    print(f"机器人坐标: {result['robot_center']} mm")
    print(f"距离: {result['depth']} mm")
    
    return result
```

### 9.2 点云生成

```python
def generate_point_cloud(snapshot: DepthSnapshot) -> np.ndarray:
    """生成点云"""
    height, width = snapshot.depth.shape
    points = []
    
    for v in range(height):
        for u in range(width):
            depth = snapshot.depth[v, u]
            if depth == 0:
                continue
            
            # 像素坐标转相机坐标
            Xc, Yc, Zc = pixel_to_camera(
                u, v, depth, snapshot.intrinsics
            )
            
            points.append([Xc, Yc, Zc])
    
    return np.array(points)
```

---

## 十、总结

### 核心知识点

| 概念 | 说明 |
|------|------|
| **深度相机** | 能获取场景距离信息的相机 |
| **内参** | 相机光学特性（焦距、主点） |
| **外参** | 相机在世界中的位置姿态 |
| **深度图** | 每个像素存储距离值（mm） |
| **坐标转换** | 像素→相机→机器人→世界 |

### 项目中的应用

1. **目标定位**：将VLM检测的2D边界框转换为3D坐标
2. **抓取规划**：计算抓取点和姿态
3. **导航避障**：感知周围环境
4. **场景重建**：构建3D点云地图

### 常用库

| 库名 | 用途 |
|------|------|
| **pyorbbecsdk** | Orbbec相机官方SDK |
| **OpenCV** | 图像处理、标定 |
| **NumPy** | 矩阵运算 |
| **Open3D** | 点云处理 |

---

**文档版本**：v1.0  
**创建日期**：2026-03-16  
**适用人群**：项目开发者、机器人学习者
