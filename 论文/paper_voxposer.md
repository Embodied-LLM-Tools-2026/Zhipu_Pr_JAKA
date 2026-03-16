# 论文精读：VoxPoser

## 论文基本信息

| 项目 | 内容 |
|------|------|
| **标题** | VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models |
| **作者** | Wenlong Huang, Chen Wang, et al. (Stanford, Li Fei-Fei 团队) |
| **发表时间** | 2023年7月 |
| **会议** | CoRL 2023 |
| **arXiv ID** | 2307.05973 |
| **项目主页** | https://voxposer.github.io/ |

---

## 核心思想速览

### 🎯 核心问题：零样本机器人轨迹生成

这篇论文的核心在于通过**LLM + VLM 生成 3D 值图**，实现零样本机器人轨迹规划。

**核心痛点**：
- 传统方法需要针对每个任务训练专门的模型
- 难以处理开放集指令和物体
- 缺乏对复杂约束的理解（如"不要碰到杯子"）

### ⚙️ 核心机制：LLM 代码生成 → VLM 感知 → 3D 值图 → 轨迹规划

将轨迹生成转化为**代码生成 + 值图组合**问题：

```
用户指令 → LLM 生成代码 → 调用 VLM → 生成 3D 值图 → 轨迹优化
```

**关键创新**：
- **Affordance + Constraint**：值图包含"可行区域"和"约束区域"
- **代码驱动**：LLM 生成 Python 代码来组合值图
- **零样本泛化**：无需训练即可处理新任务

**示例**：
```python
# 用户指令："把苹果放进抽屉，不要碰到杯子"

# LLM 生成的代码
def compose_value_map(scene):
    # Affordance: 苹果的可抓取区域
    apple_grasp_map = vlm.get_grasp_map("apple")
    
    # Constraint: 避开杯子
    cup_avoid_map = vlm.get_avoid_map("cup")
    
    # Affordance: 抽屉的放置区域
    drawer_place_map = vlm.get_place_map("drawer")
    
    # 组合值图
    value_map = apple_grasp_map * (1 - cup_avoid_map) * drawer_place_map
    
    return value_map
```

### 💡 核心意义

- ✅ **零样本泛化**：无需训练即可处理新任务
- ✅ **可解释性**：生成的代码清晰展示推理过程
- ✅ **灵活性**：支持复杂的空间约束
- ✅ **鲁棒性**：通过值图规划，抗干扰能力强

### 📊 一句话总结

> **"用 LLM 写代码组合 VLM 的感知结果，生成 3D 值图指导机器人运动"**

---

## 一、研究背景与动机

### 1.1 传统方法的局限

**传统轨迹规划**：
- 需要精确的目标位置
- 难以处理语义约束
- 泛化能力差

**问题示例**：
```
用户指令："把苹果放进抽屉，不要碰到杯子"

传统方法：
❌ 需要手动标注苹果、抽屉、杯子的位置
❌ 需要手动设计避障约束
❌ 难以泛化到新场景
```

### 1.2 核心洞察

**关键发现**：
- LLM 擅长理解语义约束和空间关系
- VLM 可以将语义映射到 3D 空间
- 值图可以自然地组合多个约束

**解决思路**：
> 让 LLM 生成代码，调用 VLM 生成值图，组合成完整的约束

---

## 二、方法：VoxPoser 框架

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户指令                                │
│        "把苹果放进抽屉，不要碰到杯子"                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   LLM（代码生成）                            │
│                                                              │
│  输入：用户指令 + 可用 API                                   │
│  输出：Python 代码                                           │
│                                                              │
│  生成的代码：                                                │
│  1. 调用 VLM 获取物体信息                                    │
│  2. 生成 Affordance 值图                                     │
│  3. 生成 Constraint 值图                                     │
│  4. 组合值图                                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   VLM（视觉-语言模型）                       │
│                                                              │
│  输入：RGB-D 图像 + 文本查询                                 │
│  输出：3D 值图                                               │
│                                                              │
│  功能：                                                      │
│  - get_grasp_map(object) → 可抓取区域                       │
│  - get_place_map(location) → 可放置区域                     │
│  - get_avoid_map(object) → 避障区域                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   值图组合                                   │
│                                                              │
│  Affordance 值图：高值 = 可行区域                            │
│  Constraint 值图：低值 = 禁止区域                            │
│                                                              │
│  组合方式：                                                  │
│  final_map = affordance_map * (1 - constraint_map)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   轨迹规划                                   │
│                                                              │
│  使用模型预测控制（MPC）在值图上规划轨迹                     │
│  优化目标：最大化路径上的累积值                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 3D 值图生成

**值图定义**：
```python
class ValueMap3D:
    """3D 值图"""
    
    def __init__(self, resolution: float = 0.01):
        self.resolution = resolution
        self.map = np.zeros((100, 100, 100))  # 1m x 1m x 1m
    
    def get_value(self, position: np.ndarray) -> float:
        """获取某个位置的值"""
        index = (position / self.resolution).astype(int)
        return self.map[index[0], index[1], index[2]]
    
    def set_value(self, position: np.ndarray, value: float):
        """设置某个位置的值"""
        index = (position / self.resolution).astype(int)
        self.map[index[0], index[1], index[2]] = value
```

**VLM 生成值图**：
```python
class VLMValueMapper:
    """VLM 值图生成器"""
    
    def get_grasp_map(self, object_name: str, scene) -> ValueMap3D:
        """生成可抓取区域值图"""
        # 1. 使用 VLM 检测物体
        detections = self.vlm.detect(scene.image, object_name)
        
        # 2. 获取 3D 位置
        positions = []
        for det in detections:
            pos_3d = scene.get_3d_position(det.bbox)
            positions.append(pos_3d)
        
        # 3. 生成值图（高斯分布）
        value_map = ValueMap3D()
        for pos in positions:
            # 在物体周围生成高值区域
            value_map.add_gaussian(pos, sigma=0.05, amplitude=1.0)
        
        return value_map
    
    def get_avoid_map(self, object_name: str, scene) -> ValueMap3D:
        """生成避障区域值图"""
        detections = self.vlm.detect(scene.image, object_name)
        
        value_map = ValueMap3D()
        for det in detections:
            pos_3d = scene.get_3d_position(det.bbox)
            # 在物体周围生成低值区域
            value_map.add_gaussian(pos_3d, sigma=0.1, amplitude=-1.0)
        
        return value_map
```

### 2.3 值图组合

**组合策略**：
```python
def compose_value_maps(affordance_maps: List[ValueMap3D], 
                       constraint_maps: List[ValueMap3D]) -> ValueMap3D:
    """组合多个值图"""
    final_map = ValueMap3D()
    
    # 初始化为 1
    final_map.map = np.ones_like(final_map.map)
    
    # 乘以 Affordance 值图
    for affordance_map in affordance_maps:
        final_map.map *= affordance_map.map
    
    # 乘以 (1 - Constraint 值图)
    for constraint_map in constraint_maps:
        final_map.map *= (1 - constraint_map.map)
    
    return final_map
```

### 2.4 轨迹规划

**MPC 规划器**：
```python
class MPCTrajectoryPlanner:
    """模型预测控制轨迹规划器"""
    
    def plan(self, value_map: ValueMap3D, start_pos: np.ndarray, 
             goal_pos: np.ndarray) -> List[np.ndarray]:
        """规划轨迹"""
        trajectory = [start_pos]
        current_pos = start_pos
        
        for _ in range(self.max_steps):
            # 采样多个候选动作
            candidate_actions = self._sample_actions(current_pos)
            
            # 评估每个动作的累积值
            best_action = None
            best_value = -np.inf
            
            for action in candidate_actions:
                next_pos = current_pos + action
                value = self._evaluate_trajectory(next_pos, goal_pos, value_map)
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            # 执行最佳动作
            current_pos = current_pos + best_action
            trajectory.append(current_pos)
            
            # 检查是否到达目标
            if np.linalg.norm(current_pos - goal_pos) < 0.05:
                break
        
        return trajectory
```

---

## 三、实验与结果

### 3.1 实验设置

**机器人平台**：
- 真实机器人：Franka Emika Panda
- 仿真环境： tabletop 场景

**任务类型**：
1. **基础操作**：抓取、放置
2. **约束操作**：避障、精细操作
3. **复杂任务**：多步骤任务

### 3.2 实验结果

| 任务类型 | 成功率 |
|---------|--------|
| 基础操作 | 82% |
| 约束操作 | 75% |
| 复杂任务 | 68% |
| 新物体泛化 | 78% |

**关键发现**：
- ✅ 零样本泛化能力强
- ✅ 能处理复杂的空间约束
- ✅ 对动态干扰有鲁棒性

---

## 四、创新点总结

| 创新点 | 描述 |
|--------|------|
| **1. 3D 值图** | 首次用 3D 值图表示机器人操作约束 |
| **2. 代码驱动组合** | LLM 生成代码组合 VLM 的感知结果 |
| **3. 零样本泛化** | 无需训练即可处理新任务 |
| **4. 可解释性** | 生成的代码清晰展示推理过程 |

---

## 五、可借鉴之处（针对我们的项目）

### 5.1 直接可借鉴

#### 借鉴 1：值图表示

**应用到我们的项目**：

```python
# voice/perception/value_map.py

class ValueMap3D:
    """3D 值图表示"""
    
    def __init__(self, world_model):
        self.world = world_model
        self.resolution = 0.01  # 1cm
        self.map = np.zeros((100, 100, 100))
    
    def update_from_world(self):
        """从世界模型更新值图"""
        # 清空值图
        self.map = np.zeros_like(self.map)
        
        # 为每个物体添加值
        for obj_name, obj in self.world.objects.items():
            if obj.visible:
                self._add_object_value(obj_name, obj.position)
    
    def _add_object_value(self, obj_name: str, position: np.ndarray):
        """为物体添加值"""
        # 根据物体类型设置不同的值
        if "grasp" in obj_name:
            self._add_gaussian(position, sigma=0.05, amplitude=1.0)
        elif "avoid" in obj_name:
            self._add_gaussian(position, sigma=0.1, amplitude=-1.0)
```

#### 借鉴 2：代码生成值图

**改进 planner.py**：

```python
# voice/agents/planner.py

class ValueMapPlanner:
    """基于值图的规划器"""
    
    def generate_value_map_code(self, instruction: str) -> str:
        """生成值图代码"""
        prompt = f"""
        指令：{instruction}
        
        生成 Python 代码来组合值图。
        可用 API：
        - vlm.get_grasp_map(object_name)
        - vlm.get_place_map(location_name)
        - vlm.get_avoid_map(object_name)
        
        示例：
        # "把苹果放进抽屉"
        grasp_map = vlm.get_grasp_map("apple")
        place_map = vlm.get_place_map("drawer")
        final_map = grasp_map * place_map
        """
        
        code = self.llm.generate(prompt)
        return code
```

#### 借鉴 3：轨迹规划

**实现 MPC 规划器**：

```python
# voice/control/trajectory_planner.py

class MPCTrajectoryPlanner:
    """MPC 轨迹规划器"""
    
    def plan_trajectory(self, value_map: ValueMap3D, 
                       start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """规划轨迹"""
        # 使用值图指导轨迹规划
        trajectory = []
        current_pos = start
        
        while np.linalg.norm(current_pos - goal) > 0.05:
            # 采样候选动作
            candidates = self._sample_actions(current_pos)
            
            # 选择最佳动作
            best_action = self._select_best_action(
                candidates, current_pos, goal, value_map
            )
            
            current_pos = current_pos + best_action
            trajectory.append(current_pos)
        
        return trajectory
```

### 5.2 需要改进的地方

#### 改进 1：动态值图更新

**实时更新值图**：

```python
class DynamicValueMap:
    """动态值图"""
    
    def __init__(self):
        self.value_map = ValueMap3D()
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _update_loop(self):
        """持续更新值图"""
        while True:
            # 从传感器获取最新数据
            scene = self.get_current_scene()
            
            # 更新值图
            self.value_map.update_from_scene(scene)
            
            time.sleep(0.1)  # 10 Hz
```

#### 改进 2：学习值图

**从数据中学习值图**：

```python
class LearnedValueMapper:
    """学习版值图生成器"""
    
    def __init__(self):
        self.model = self._build_model()
    
    def train(self, demonstrations):
        """从演示中学习"""
        for demo in demonstrations:
            # 提取特征
            features = self._extract_features(demo.scene)
            
            # 计算目标值图
            target_map = self._compute_target_map(demo.trajectory)
            
            # 训练模型
            self.model.train(features, target_map)
    
    def predict(self, scene, instruction) -> ValueMap3D:
        """预测值图"""
        features = self._extract_features(scene, instruction)
        return self.model.predict(features)
```

### 5.3 与我们项目的结合点

| VoxPoser 组件 | 我们的对应组件 | 改进方向 |
|---------------|---------------|---------|
| 3D 值图 | 新增 value_map.py | 实现值图表示 |
| VLM 值图生成 | VLM.py | 增加值图生成功能 |
| 代码生成 | planner.py | 生成值图组合代码 |
| 轨迹规划 | executor.py | 实现 MPC 规划 |

---

## 六、总结与启发

### 6.1 核心思想

> **"用代码组合感知结果，生成可解释的 3D 约束"**

VoxPoser 的核心贡献在于：
1. **表示创新**：用 3D 值图表示操作约束
2. **组合策略**：通过代码组合多个约束
3. **零样本能力**：无需训练即可处理新任务

### 6.2 对我们项目的启发

1. **架构层面**：
   - ✅ 可以实现 3D 值图表示
   - ✅ 利用 VLM 生成值图
   - ✅ 通过代码组合约束

2. **技术层面**：
   - ✅ 实现 MPC 轨迹规划
   - ✅ 增强空间推理能力
   - ✅ 提高轨迹鲁棒性

3. **创新机会**：
   - 🚀 **动态值图**：实时更新值图
   - 🚀 **学习值图**：从数据中学习
   - 🚀 **多模态融合**：融合更多传感器信息

### 6.3 实施建议

**短期（1-2 周）**：
1. 实现 3D 值图表示
2. 测试 VLM 的值图生成能力
3. 实现简单的轨迹规划

**中期（1-2 个月）**：
1. 实现完整的 VoxPoser 框架
2. 在真实机器人上测试
3. 优化性能

**长期（3-6 个月）**：
1. 实现学习版值图
2. 扩展到更多任务
3. 发表论文或开源项目

---

## 七、参考文献

1. **VoxPoser 论文**：Huang, W., et al. "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models." CoRL 2023.
2. **Code as Policies**：Liang, J., et al. "Code as Policies: Language Model Programs for Embodied Control." ICRA 2023.
3. **MPC 规划**：Camacho, E. F., & Bordons, C. "Model Predictive Control." Springer 2007.

---

**文档创建时间**：2026-03-16  
**论文 arXiv ID**：2307.05973  
**PDF 文件名**：VoxPoser_Composable_3D_Value_Maps.pdf
