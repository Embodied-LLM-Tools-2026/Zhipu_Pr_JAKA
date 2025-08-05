# AGV 2D建图、定位、导航系统使用指南

## 概述

本系统在原有 `agv_client.py` 基础上扩展实现了完整的2D建图、定位、导航全流程控制，包含：

- **扩展的 AGVClient 类**：添加了建图、定位、基础运动控制等新方法
- **AGVWorkflow 工作流程管理类**：管理完整的三阶段流程
- **多种导航方法**：提供5种不同的导航实现供性能对比

## 系统架构

```
AGVClient (基础控制)
    ├── 建图控制: start_slam(), stop_slam(), get_slam_status()
    ├── 定位控制: relocalize()
    ├── 运动控制: translate(), rotate(), rotate_in_place()
    ├── 导航控制: navigate_path(), get_path_between_points()
    └── 多种导航方法: method1~5

AGVWorkflow (工作流程管理)
    ├── 第一阶段: stage1_mapping() - 按指定点建图
    ├── 第二阶段: stage2_localization() - 自动定位
    └── 第三阶段: stage3_navigation() - 多方法导航测试
```

## 快速开始

### 1. 基础使用

```python
from agv_client import AGVClient

# 使用上下文管理器自动连接和断开
with AGVClient(ip='192.168.192.5') as agv:
    # 获取当前位置
    x, y, angle = agv.get_pose()
    print(f"当前位置: ({x}, {y}, {angle})")
    
    # 基础运动控制
    agv.translate(1.0, vx=0.5)  # 向前1米
    agv.rotate_in_place(1)      # 原地转一圈
    
    # 导航到目标点
    agv.go_to_point_in_world(2.0, 2.0, 0.0)
```

### 2. 完整工作流程使用

```python
from agv_workflow import AGVWorkflow

# 创建工作流程管理器
workflow = AGVWorkflow(ip='192.168.192.5')

# 定义建图路径点
mapping_points = [
    (0.0, 0.0, 0.0),    # 起始点
    (2.0, 0.0, 0.0),    # 前方2米
    (2.0, 2.0, 1.57),   # 右上角
    (0.0, 2.0, 3.14),   # 左上角
    (0.0, 0.0, 0.0),    # 回到起始点
]

# 定义导航测试目标点
target_points = [
    (1.0, 1.0, 0.0),    # 中心点
    (2.0, 0.0, 1.57),   # 右下角
]

# 运行完整流程
results = workflow.run_complete_workflow(
    mapping_points=mapping_points,
    target_points=target_points
)

# 生成测试报告
if results['success']:
    workflow.generate_test_report(results['navigation_results'])
```

## 核心功能详解

### 建图控制方法

```python
# 开始2D建图
agv.start_slam(slam_type=1, real_time=False)

# 获取建图状态
status = agv.get_slam_status()
# 0=没有扫图, 1=正在扫图, 2=实时扫图, 3=3D扫图, 4=实时3D扫图

# 停止建图
agv.stop_slam()
```

### 定位控制方法

```python
# 自动重定位
agv.relocalize(is_auto=True)

# 手动指定位置重定位
agv.relocalize(x=1.0, y=1.0, angle=0.0)

# 在RobotHome重定位
agv.relocalize(home=True)

# 获取定位状态
status = agv.get_relocalization_status()
# 0=初始化中, 1=定位成功, 2=正在定位
```

### 基础运动控制

```python
# 平移控制
agv.translate(dist=1.0, vx=0.5)  # 向前1米，速度0.5m/s

# 转动控制
agv.rotate(angle=1.57, vw=1.0)   # 转90度，角速度1.0rad/s

# 原地转圈
agv.rotate_in_place(turns=1, angular_velocity=1.0)  # 转1圈
```

### 多种导航方法

系统提供5种导航方法供性能对比：

```python
# 方法1：世界坐标导航（最常用）
agv.navigate_to_point_method1(x, y, theta)

# 方法2：机器人坐标导航
agv.navigate_to_point_method2(x, y, theta)

# 方法3：分段导航
agv.navigate_to_point_method3(x, y, theta, intermediate_points)

# 方法4：路径导航（需要预定义站点）
agv.navigate_path(move_task_list)

# 方法5：基础运动组合导航
# 内部使用translate()和rotate()组合实现
agv.navigate_to_point_method5(x, y, theta)
```

## 三阶段工作流程

### 第一阶段：建图

系统提供两种建图模式：

#### 模式1：手动指定点建图

```python
# 定义建图点序列
mapping_points = [(x1,y1,θ1), (x2,y2,θ2), ...]

# 执行建图流程
workflow.stage1_mapping(
    mapping_points=mapping_points,
    turns_per_point=1,      # 每个点转圈数
    angular_velocity=1.0,   # 转动角速度
    auto_mapping=False      # 手动建图模式
)
```

**建图流程**：
1. 启动2D建图
2. 按顺序导航到各个建图点
3. 在每个点原地转圈进行扫描
4. 停止建图
5. **轮询扫图状态**直到返回非空地图数据
6. **检查当前载入地图**是否为新建图
7. **自动上传并载入**新地图（如需要）

#### 模式2：自动路径规划建图（扫地机器人式）

```python
# 执行自动建图流程
workflow.stage1_mapping(
    mapping_points=None,         # 自动建图不需要指定点
    auto_mapping=True,           # 启用自动建图
    turn_direction='left',       # 转向方向: 'left' 或 'right'
    forward_distance=2.0,        # 前进距离（米）
    side_distance=0.8            # 侧移距离（米）
)
```

**自动建图流程**：
1. 启动2D建图
2. 机器人向前直行直到遇到障碍
3. 转向90度（左转或右转）
4. 侧移一小段距离
5. 再转向90度恢复直行方向
6. 重复步骤2-5，直到侧移时遇到障碍
7. 停止建图
8. **轮询扫图状态**直到返回非空地图数据
9. **检查当前载入地图**是否为新建图
10. **自动上传并载入**新地图（如需要）

**自动建图参数说明**：
- `turn_direction`: 转向方向，'left'=左转建图，'right'=右转建图
- `forward_distance`: 每次最大前进距离，遇到障碍会提前停止
- `side_distance`: 每次侧移距离，用于换行

**优势对比**：
- **手动模式**：路径可控，适合已知环境，建图精度高
- **自动模式**：无需预先规划，适合未知环境，覆盖更全面

### 第二阶段：定位

```python
# 自动定位
workflow.stage2_localization(auto_relocalize=True)

# 手动指定位置定位  
workflow.stage2_localization(
    auto_relocalize=False,
    manual_position=(x, y, angle)
)
```

### 第三阶段：导航测试

```python
# 定义测试目标点
target_points = [(x1,y1,θ1), (x2,y2,θ2), ...]

# 选择要测试的导航方法
methods_to_test = [
    'method1_world_coordinate',
    'method2_robot_coordinate',
    'method5_translate_rotate'
]

# 执行导航测试
results = workflow.stage3_navigation(target_points, methods_to_test)

# 生成测试报告
workflow.generate_test_report(results)
```

## 使用示例

### 运行预定义示例

```bash
cd action_sequence
python example_usage.py
```

选择对应的示例：
- **示例1**：基础AGV控制功能测试
- **示例2**：简单导航测试
- **示例3**：完整工作流程测试（手动指定点建图）
- **示例4**：分步骤执行工作流程
- **示例5**：单独测试各个控制方法
- **示例6**：自动路径规划建图工作流程（扫地机器人式）
- **示例7**：对比两种建图方法

### 自定义测试

#### 手动指定点建图模式

```python
from agv_workflow import AGVWorkflow

workflow = AGVWorkflow(ip='你的AGV_IP')

# 根据实际环境调整坐标点
my_mapping_points = [
    (0.0, 0.0, 0.0),     # 根据实际环境调整
    (1.0, 0.0, 0.0),
    # ... 更多点
]

my_target_points = [
    (0.5, 0.5, 0.0),     # 根据实际需求调整
    # ... 更多目标点
]

# 运行手动建图流程
results = workflow.run_complete_workflow(
    mapping_points=my_mapping_points,
    target_points=my_target_points,
    navigation_methods=['method1_world_coordinate'],  # 选择可靠的方法
    auto_mapping=False  # 手动建图模式
)
```

#### 自动路径规划建图模式

```python
from agv_workflow import AGVWorkflow

workflow = AGVWorkflow(ip='你的AGV_IP')

my_target_points = [
    (0.5, 0.5, 0.0),     # 根据实际需求调整
    # ... 更多目标点
]

# 运行自动建图流程
results = workflow.run_complete_workflow(
    mapping_points=None,  # 自动建图不需要指定点
    target_points=my_target_points,
    navigation_methods=['method1_world_coordinate'],
    auto_mapping=True,           # 启用自动建图
    turn_direction='left',       # 左转建图模式
    forward_distance=2.0,        # 前进距离2米
    side_distance=0.8            # 侧移距离0.8米
)
```

## 注意事项

### 安全须知
1. **确保环境安全**：运行前确认AGV周围环境安全，无障碍物
2. **坐标系检查**：验证坐标系方向与实际环境一致
3. **急停准备**：随时准备使用 `Ctrl+C` 中断程序或AGV的物理急停按钮

### 参数调整
1. **IP地址**：修改为实际AGV的IP地址
2. **坐标点**：根据实际环境调整建图点和目标点坐标
3. **运动参数**：根据AGV性能调整速度和加速度参数
4. **超时时间**：根据环境大小调整各种超时参数

### 错误处理
- 程序包含完整的错误处理和状态监控
- 支持 `Ctrl+C` 优雅中断
- 每个阶段失败时会输出详细错误信息

### 性能优化
- 建图点不宜过多，建议5-10个关键点
- 导航测试点数量适中，避免过长测试时间
- 可根据需要调整转圈数和运动速度

## 扩展开发

### 添加新的导航方法

```python
def navigate_to_point_method_custom(self, x, y, theta):
    """自定义导航方法"""
    # 实现你的导航逻辑
    pass

# 在AGVWorkflow中注册
workflow.navigation_methods['method_custom'] = agv.navigate_to_point_method_custom
```

### 自定义工作流程

```python
class CustomWorkflow(AGVWorkflow):
    def custom_stage(self):
        """自定义阶段"""
        # 实现自定义功能
        pass
```

## 故障排除

### 常见问题

1. **连接失败**
   - 检查AGV IP地址和网络连接
   - 确认AGV服务器正在运行

2. **建图失败**
   - 检查AGV是否处于合适的初始位置
   - 确认环境光照和特征点充足

3. **定位失败**
   - 确认地图已正确加载
   - 检查AGV当前位置是否在地图范围内

4. **导航失败**
   - 验证目标点坐标是否合理
   - 检查路径是否有障碍物

### 调试技巧

1. **单步调试**：使用分步骤执行模式
2. **状态监控**：实时查看AGV各种状态
3. **日志输出**：程序包含详细的执行日志
4. **参数调整**：从小范围、低速度开始测试

## API参考

详细的API文档请参考：
- `agv_client.py` - 基础控制方法
- `agv_workflow.py` - 工作流程管理方法
- `action_sequence/AGV指令文档/` - 底层通信协议文档