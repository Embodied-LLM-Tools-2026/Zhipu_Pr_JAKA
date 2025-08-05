# AGV自动路径规划建图功能说明

## 功能概述

新增了类似扫地机器人的自动路径规划建图方法，机器人能够自主探索环境并完成建图，无需预先指定建图点位置。

## 工作原理

### 基本流程

```
开始建图 → 向前直行 → 遇到障碍? → 转向90° → 侧移一段距离 → 再转向90° → 继续直行
                ↑                                                            ↓
                └──────────────── 重复循环 ←─────────────────────────────────┘
                                    ↓
                               侧移时遇到障碍? → 结束建图
```

### 详细步骤

1. **启动建图**：开始2D SLAM建图模式
2. **直行阶段**：向前移动直到遇到障碍物或达到最大距离
3. **转向阶段**：左转或右转90度（根据参数设定）
4. **侧移阶段**：移动一小段距离进行"换行"
5. **恢复阶段**：再次转向90度，恢复直行方向
6. **循环执行**：重复步骤2-5
7. **结束条件**：当侧移时检测到障碍物，自动结束建图

## 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `turn_direction` | str | 'left' | 转向方向：'left'=左转建图，'right'=右转建图 |
| `forward_distance` | float | 1.0 | 每次最大前进距离（米），遇障碍提前停止 |
| `side_distance` | float | 0.5 | 每次侧移距离（米），用于"换行" |

## 使用方法

### 基本使用

```python
from agv_workflow import AGVWorkflow

workflow = AGVWorkflow(ip='192.168.192.5')

# 自动建图
success = workflow.stage1_mapping(
    auto_mapping=True,           # 启用自动建图
    turn_direction='left',       # 左转建图
    forward_distance=2.0,        # 前进2米
    side_distance=0.8            # 侧移0.8米
)
```

### 完整工作流程

```python
# 运行包含自动建图的完整流程
results = workflow.run_complete_workflow(
    mapping_points=None,         # 自动建图不需要指定点
    target_points=[(1,1,0), (2,0,0)],  # 后续导航测试点
    auto_mapping=True,           # 启用自动建图
    turn_direction='right',      # 右转建图
    forward_distance=1.5,        # 前进1.5米
    side_distance=0.6            # 侧移0.6米
)
```

## 建图路径示例

### 左转建图模式 (turn_direction='left')

```
起始位置 → → → → → 障碍物
                    ↓ (左转90°)
← ← ← ← ← ← ← ← ← ←
↓ (左转90°)
→ → → → → → → → → 障碍物
                    ↓ (左转90°)
← ← ← ← ← ← ← ← ← ←
↓ (左转90° + 侧移)
...
```

### 右转建图模式 (turn_direction='right')

```
起始位置 → → → → → 障碍物
                    ↓ (右转90°)
                    → → → → → → → → → →
                                      ↓ (右转90°)
                    ← ← ← ← ← ← ← ← ← 障碍物
                    ↓ (右转90° + 侧移)
                    ...
```

## 安全机制

1. **分步移动**：每次只移动0.2米，实时检测障碍物
2. **最大循环限制**：防止无限循环，最多执行50次扫描
3. **状态监控**：实时监控建图状态和运动状态
4. **优雅中断**：支持Ctrl+C中断，自动停止建图

## 障碍物检测

使用AGV的内置传感器检测障碍物：
- 调用 `get_blocked()` 方法获取阻挡状态
- 返回 `(blocked, block_x, block_y)` 元组
- `blocked=True` 时表示检测到障碍物

## 参数调优建议

### 环境适配

| 环境类型 | forward_distance | side_distance | turn_direction |
|----------|------------------|---------------|----------------|
| 小房间（<20㎡） | 1.0-1.5m | 0.4-0.6m | left |
| 中房间（20-50㎡） | 1.5-2.5m | 0.6-0.8m | left |
| 大房间（>50㎡） | 2.0-3.0m | 0.8-1.2m | left |
| 狭长走廊 | 3.0-5.0m | 0.3-0.5m | right |

### 建图质量优化

1. **提高覆盖率**：减小 `side_distance`，增加扫描行数
2. **提高效率**：增大 `forward_distance` 和 `side_distance`
3. **提高精度**：降低移动速度，在代码中调整 `forward_velocity`

## 示例程序

### 示例6：自动路径规划建图

```bash
cd action_sequence
python example_usage.py
# 选择 6
```

### 示例7：对比两种建图方法

```bash
cd action_sequence  
python example_usage.py
# 选择 7
```

## 与手动建图对比

| 特性 | 手动指定点建图 | 自动路径规划建图 |
|------|----------------|------------------|
| **预先规划** | 需要 | 不需要 |
| **环境适应** | 固定路径 | 自适应 |
| **覆盖范围** | 取决于预设点 | 自动最大化 |
| **实施难度** | 需要环境知识 | 即插即用 |
| **建图精度** | 高（可控点位） | 中等（自动探索） |
| **适用场景** | 已知环境 | 未知环境 |

## 技术实现细节

### 核心算法

```python
def _auto_mapping_sweep(self, turn_direction, forward_distance, side_distance):
    while sweep_count < max_sweeps:
        # 1. 直行直到遇到障碍
        self._move_forward_until_blocked(forward_distance, velocity)
        
        # 2. 转向90度
        self.agv.rotate(turn_angle, angular_velocity)
        
        # 3. 侧移一段距离
        side_blocked = self._move_side_distance(side_distance, velocity)
        if side_blocked:
            break  # 侧移遇障碍，结束建图
            
        # 4. 再转向90度恢复直行
        self.agv.rotate(turn_angle, angular_velocity)
```

### 关键方法

- `_move_forward_until_blocked()`: 前进直到遇障碍
- `_move_side_distance()`: 侧移并检测障碍
- `get_blocked()`: 获取AGV障碍物检测状态

## 故障处理

### 常见问题

1. **建图不完整**
   - 增加扫描循环次数
   - 减小侧移距离
   - 检查传感器状态

2. **建图时间过长**
   - 增大移动距离参数
   - 提高移动速度
   - 设置合理的最大循环次数

3. **撞墙或卡住**
   - 检查障碍物检测功能
   - 调整前进步长
   - 确认环境安全

### 调试技巧

1. **日志监控**：关注控制台输出的详细步骤
2. **参数调试**：从小范围、低速度开始测试
3. **分段测试**：先测试基础运动，再测试完整流程

## 扩展开发

### 自定义建图策略

```python
class CustomWorkflow(AGVWorkflow):
    def custom_auto_mapping(self):
        """自定义自动建图策略"""
        # 实现你的建图算法
        pass
```

### 高级功能扩展

- 动态调整参数
- 多传感器融合
- 路径优化算法
- 建图质量评估

这个自动路径规划建图功能为AGV系统提供了强大的自主建图能力，特别适合未知环境的快速建图需求。