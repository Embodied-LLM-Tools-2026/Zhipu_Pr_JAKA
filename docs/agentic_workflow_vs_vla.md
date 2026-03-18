# Agentic Workflow 与 VLA 模型详解

## 文档概述

本文档详细介绍具身智能领域的两种主流技术路线：**Agentic Workflow（模块化工作流）** 和 **VLA模型（Vision-Language-Action端到端模型）**，分析它们的原理、优劣势、适用场景，以及在当前项目中的应用情况。

---

## 一、背景：具身智能的核心挑战

### 1.1 问题定义

具身智能（Embodied AI）的核心挑战是让机器人能够：

```
感知环境 → 理解指令 → 规划动作 → 执行控制 → 反馈调整
```

**核心难点**：

| 挑战 | 说明 |
|------|------|
| **感知不确定性** | 环境复杂多变，物体识别和定位存在误差 |
| **指令理解** | 自然语言指令模糊、抽象，需要语义理解 |
| **动作空间** | 机器人动作是连续的高维空间，难以离散化 |
| **实时性要求** | 机器人控制需要高频响应（10-50Hz） |
| **泛化能力** | 需要适应新场景、新任务、新物体 |

### 1.2 两种技术路线

针对上述挑战，学术界发展出两种主流技术路线：

```
┌─────────────────────────────────────────────────────────────┐
│                    技术路线对比                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  路线1：Agentic Workflow（模块化工作流）                      │
│  ├── 思路：分而治之，各模块各司其职                           │
│  ├── 代表：SayCan, Code as Policies, VoxPoser等              │
│  └── 特点：无需训练，灵活可扩展                               │
│                                                              │
│  路线2：VLA模型（端到端视觉-语言-动作模型）                    │
│  ├── 思路：统一建模，端到端学习                               │
│  ├── 代表：RT-1, RT-2, Octo等                                │
│  └── 特点：低延迟，联合优化                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、Agentic Workflow（模块化工作流）

### 2.1 核心思想

**Agentic Workflow** 将复杂的机器人任务分解为多个模块，每个模块专注于特定功能：

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Workflow 架构                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  感知   │ →  │  规划   │ →  │  代码   │ →  │  执行   │  │
│  │  VLM    │    │  LLM    │    │ 生成    │    │ Executor│  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       ↓              ↓              ↓              ↓        │
│   图像理解       行为树规划      Python代码      机器人控制   │
│   物体检测       任务分解        API调用         状态反馈     │
│   场景描述       状态检查        动作序列                      │
│                                                              │
│  特点：多阶段、模块化、可解释、灵活可扩展                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 各模块详解

#### 2.2.1 感知模块（VLM）

**功能**：将视觉信息转化为语义信息

```python
class PerceptionModule:
    """感知模块：VLM（视觉-语言模型）"""
    
    def __init__(self):
        self.vlm = load_model("qwen3-vl-plus")  # 或 GPT-4V, GLM-4V
    
    def process(self, image) -> dict:
        # 1. 物体检测
        objects = self.vlm.detect_objects(image)
        # 输出：[{"name": "apple", "bbox": [x,y,w,h], "confidence": 0.95}]
        
        # 2. 场景描述
        scene_desc = self.vlm.describe(image, "描述场景中的物体和状态")
        # 输出："桌上有红色苹果、蓝色杯子，苹果在杯子左边"
        
        # 3. 空间关系
        relations = self.vlm.analyze_spatial(image)
        # 输出：{"apple": {"left_of": "cup", "on": "table"}}
        
        return {
            "objects": objects,
            "scene_description": scene_desc,
            "spatial_relations": relations
        }
```

**关键技术**：
- 目标检测（Object Detection）
- 语义分割（Semantic Segmentation）
- 场景理解（Scene Understanding）
- 空间推理（Spatial Reasoning）

#### 2.2.2 规划模块（LLM）

**功能**：将自然语言指令分解为可执行的子任务

```python
class PlanningModule:
    """规划模块：LLM（大语言模型）"""
    
    def __init__(self):
        self.llm = load_model("deepseek-chat")  # 或 GPT-4, GLM-4
    
    def plan(self, instruction: str, perception_result: dict) -> list:
        # 构建提示
        prompt = f"""
        # 当前环境
        物体：{perception_result['objects']}
        场景：{perception_result['scene_description']}
        
        # 可用动作
        - navigate_to(location): 导航到指定位置
        - pick(object): 抓取物体
        - place(object, location): 放置物体
        - open/close(container): 打开/关闭容器
        
        # 任务
        {instruction}
        
        # 生成执行计划（Python函数调用序列）
        """
        
        # LLM生成计划
        plan = self.llm.generate(prompt)
        # 输出：
        # navigate_to("kitchen")
        # pick("apple")
        # place("apple", "table")
        
        return self._parse_plan(plan)
```

**规划方式**：

| 方式 | 论文 | 特点 |
|------|------|------|
| **行为树规划** | - | 结构化、可组合、支持条件检查 |
| **Affordance评估** | SayCan | 评估动作可行性，避免不可执行的动作 |
| **程序化提示** | ProgPrompt | 注入情境信息，生成可执行程序 |
| **值图规划** | VoxPoser | 生成3D值图，支持复杂空间约束 |

#### 2.2.3 代码生成模块

**功能**：将抽象动作转化为具体的控制代码

```python
class CodeGenerationModule:
    """代码生成模块"""
    
    def __init__(self):
        self.llm = load_model("deepseek-chat")
    
    def generate(self, action: str, context: dict) -> str:
        prompt = f"""
        # 可用API
        - api.navigation.navigate_to(target)
        - api.manipulation.pick(object_name)
        - api.manipulation.place(position)
        - api.perception.detect(object_name)
        
        # 当前状态
        {context}
        
        # 任务
        {action}
        
        # 生成Python代码
        """
        
        code = self.llm.generate(prompt)
        # 输出：
        # def execute(api):
        #     obj = api.perception.detect("apple")
        #     api.manipulation.pick(obj)
        #     api.manipulation.place([0.5, 0.3, 0.1])
        
        return code
```

**代码生成方式**：

| 方式 | 论文 | 特点 |
|------|------|------|
| **直接代码生成** | Code as Policies | LLM直接生成Python代码 |
| **层次化生成** | Code as Policies | 主函数 + 辅助函数递归生成 |
| **模板填充** | - | 使用预定义模板，填充参数 |

#### 2.2.4 执行模块

**功能**：执行代码，控制硬件，收集反馈

```python
class ExecutionModule:
    """执行模块"""
    
    def __init__(self, robot_api):
        self.api = robot_api
        self.feedback_collector = FeedbackCollector()
    
    def execute(self, code: str) -> ExecutionResult:
        # 1. 安全检查
        if not self._validate_safety(code):
            return ExecutionResult(success=False, error="安全检查失败")
        
        # 2. 执行代码
        try:
            exec(code, {"api": self.api})
            success = True
        except Exception as e:
            success = False
            error = str(e)
        
        # 3. 收集反馈
        feedback = self.feedback_collector.collect()
        
        return ExecutionResult(
            success=success,
            error=error if not success else None,
            feedback=feedback
        )
```

### 2.3 代表性论文

| 论文 | 核心创新 | 发表时间 |
|------|---------|---------|
| **SayCan** | Affordance函数 + LLM决策 | 2022 |
| **Code as Policies** | 代码即策略 | 2022 |
| **ProgPrompt** | 程序化提示 | 2022 |
| **Inner Monologue** | 闭环反馈 | 2022 |
| **VoxPoser** | 3D值图规划 | 2023 |
| **Socratic Models** | 零样本模型组合 | 2022 |

### 2.4 优势与劣势

#### 优势

| 优势 | 详细说明 |
|------|---------|
| **可解释性** | 每个阶段的输出都可查看、可调试，便于理解系统行为 |
| **灵活性** | 可以替换任意模块（如换不同的LLM、VLM），适应不同需求 |
| **可扩展性** | 容易添加新功能（如添加反馈机制、新的感知能力） |
| **低成本** | 无需训练，使用现成的预训练模型，部署成本低 |
| **知识迁移** | 利用LLM的互联网知识，理解抽象概念（如"最小"、"左边"） |
| **错误恢复** | 模块化设计便于定位错误，支持重试和恢复 |

#### 劣势

| 劣势 | 详细说明 |
|------|---------|
| **延迟高** | 多阶段串行处理，总延迟累加（VLM + LLM + 代码生成 + 执行） |
| **模块断层** | 各模块独立优化，无法联合学习跨模态关联 |
| **错误传播** | 前一阶段错误会传播到后续阶段（如感知错误导致规划错误） |
| **实时性差** | 难以满足高频控制要求（通常需要10-50Hz） |
| **接口设计** | 需要精心设计模块间的接口和数据格式 |

### 2.5 典型延迟分析

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Workflow 延迟                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  VLM感知：     ~100-500ms                                    │
│       ↓                                                      │
│  LLM规划：     ~500-2000ms                                   │
│       ↓                                                      │
│  代码生成：    ~200-500ms                                    │
│       ↓                                                      │
│  代码执行：    ~50-100ms                                     │
│       ↓                                                      │
│  反馈收集：    ~50-100ms                                     │
│  ─────────────────────────                                   │
│  总延迟：      ~900-3200ms                                   │
│                                                              │
│  问题：无法满足实时控制要求（需要<100ms）                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、VLA模型（Vision-Language-Action）

### 3.1 核心思想

**VLA模型** 将视觉、语言、动作统一到一个端到端的神经网络中：

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA 模型架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    VLA Model                         │    │
│  │                                                      │    │
│  │   输入：                                              │    │
│  │   ├── 图像 → Vision Encoder → 图像Token              │    │
│  │   └── 文本 → Text Tokenizer → 文本Token              │    │
│  │                    ↓                                 │    │
│  │              统一Transformer                         │    │
│  │                    ↓                                 │    │
│  │   输出：动作Token → Action Decoder → 连续动作         │    │
│  │                                                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  特点：单阶段、端到端、低延迟、需要大量训练数据               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心技术：动作Token化

**关键创新**：将连续的机器人动作离散化为Token，与语言Token统一建模

```python
class ActionTokenizer:
    """动作Token化器"""
    
    def __init__(self, action_dim=7, bins=256):
        """
        Args:
            action_dim: 动作维度（如[x,y,z,rx,ry,rz,gripper] = 7）
            bins: 离散化bin数量
        """
        self.action_dim = action_dim
        self.bins = bins
    
    def encode(self, action: np.ndarray) -> List[int]:
        """
        连续动作 → Token
        
        Args:
            action: [x, y, z, rx, ry, rz, gripper]
                   范围：[-1, 1]
        
        Returns:
            tokens: [token_x, token_y, token_z, ...]
        """
        tokens = []
        for value in action:
            # 归一化到 [0, 1]
            normalized = (value + 1) / 2
            # 离散化到 [0, bins-1]
            token = int(normalized * (self.bins - 1))
            tokens.append(token)
        return tokens
    
    def decode(self, tokens: List[int]) -> np.ndarray:
        """
        Token → 连续动作
        """
        action = []
        for token in tokens:
            # 反归一化
            normalized = token / (self.bins - 1)
            # 恢复到 [-1, 1]
            value = normalized * 2 - 1
            action.append(value)
        return np.array(action)

# 示例
tokenizer = ActionTokenizer()
action = np.array([0.3, -0.5, 0.2, 0.0, 0.0, 0.0, 1.0])  # 抓取动作
tokens = tokenizer.encode(action)
# 输出：[166, 64, 153, 128, 128, 128, 255]

# 与语言Token统一
text_tokens = text_tokenizer.encode("pick the apple")
# 输出：[523, 1024, 890]

# 统一序列输入模型
input_sequence = image_tokens + text_tokens
output_sequence = model(input_sequence)
action_tokens = output_sequence[-7:]  # 最后7个token是动作
action = tokenizer.decode(action_tokens)
```

### 3.3 代表性模型

#### 3.3.1 RT-1（Robotics Transformer 1）

**Google, 2022**

```
RT-1 特点：
├── 架构：Transformer + Token Learner
├── 输入：图像 + 自然语言指令
├── 输出：离散动作（11维，每个维度离散化为256个bin）
├── 训练数据：13个机器人，17个月，130k episodes
└── 推理速度：3Hz（~330ms）
```

**RT-1的局限**：
- 只能执行训练过的任务
- 无法理解抽象概念（如"最小的物体"）
- 泛化能力有限

#### 3.3.2 RT-2（Vision-Language-Action Model）

**Google DeepMind, 2023**

```
RT-2 特点：
├── 架构：基于PaLM-E / PaLI-X的VLA模型
├── 核心创新：利用互联网规模数据预训练
├── 涌现能力：
│   ├── 理解抽象概念（"最小的物体"）
│   ├── 理解模糊描述（"更快"、"轻轻"）
│   ├── 空间推理（"左边"、"中间"）
│   └── 跨物体泛化（没见过的物体）
├── 训练数据：机器人数据 + 互联网数据
└── 推理速度：1-5Hz
```

**RT-2的关键创新**：

```python
# RT-2训练数据
training_data = [
    # 机器人数据（动作监督）
    (image, "pick the apple", action_tokens),
    
    # 互联网数据（仅语言监督）
    (None, "the smallest object is the one with minimal volume", None),
    (None, "to pick up means to lift from the ground", None),
    (None, "left means negative x direction in robot coordinates", None),
]

# 训练后，RT-2理解：
# - "最小" = minimal volume/size
# - "拿起" = lift from ground
# - "左边" = negative x direction
```

#### 3.3.3 Octo

**UC Berkeley, 2024**

```
Octo 特点：
├── 开源通用机器人策略
├── 支持80万轨迹预训练
├── 跨形态迁移：仅需100条数据微调
├── 支持多种机器人：Franka, WidowX, Google Robot等
└── 模块化设计：可替换视觉编码器、动作解码器
```

#### 3.3.4 PaLM-E

**Google, 2023**

```
PaLM-E 特点：
├── 多模态Token化：图像、状态、文本统一为Token
├── 具身推理：将机器人状态作为输入
├── 支持多种任务：导航、操作、问答
└── 参数规模：540B
```

### 3.4 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA 模型训练流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  阶段1：预训练（互联网规模数据）                              │
│  ├── 数据：网页文本、图像-文本对、代码                        │
│  ├── 目标：学习语言理解、视觉理解、世界知识                   │
│  └── 模型：PaLM, PaLI-X等                                    │
│                                                              │
│  阶段2：机器人数据微调                                       │
│  ├── 数据：机器人演示轨迹（图像、指令、动作）                 │
│  ├── 目标：学习动作生成                                       │
│  └── 方法：动作Token化 + 自回归生成                          │
│                                                              │
│  阶段3：混合训练（RT-2创新）                                  │
│  ├── 数据：机器人数据 + 互联网数据                           │
│  ├── 目标：保持语言能力 + 学习动作生成                        │
│  └── 效果：涌现能力（理解抽象概念）                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 优势与劣势

#### 优势

| 优势 | 详细说明 |
|------|---------|
| **低延迟** | 单次前向传播，适合实时控制（100-300ms） |
| **联合优化** | 视觉、语言、动作联合学习，跨模态关联更强 |
| **知识迁移** | 利用互联网数据预训练，理解抽象概念 |
| **涌现能力** | 展现出未显式训练的能力（如理解"最小"） |
| **端到端** | 无需模块间接口设计，减少信息损失 |

#### 劣势

| 劣势 | 详细说明 |
|------|---------|
| **训练成本高** | 需要大量机器人数据（数万到数百万轨迹）和计算资源 |
| **黑盒问题** | 难以解释为什么做出某个动作，可解释性差 |
| **难以调试** | 输出错误时难以定位问题，无法查看中间状态 |
| **灵活性差** | 修改功能需要重新训练，无法快速迭代 |
| **数据依赖** | 需要高质量的机器人演示数据，收集成本高 |

### 3.6 典型延迟分析

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA 模型延迟                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  图像编码：    ~30-50ms                                      │
│       ↓                                                      │
│  Transformer： ~50-150ms                                     │
│       ↓                                                      │
│  动作解码：    ~10-20ms                                      │
│  ─────────────────────────                                   │
│  总延迟：      ~90-220ms                                     │
│                                                              │
│  优势：可以满足实时控制要求（<100ms级别）                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、两种方式的深度对比

### 4.1 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                    架构对比                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Agentic Workflow：                                          │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                      │
│  │ VLM │ → │ LLM │ → │Code │ → │Exec │                      │
│  └─────┘   └─────┘   └─────┘   └─────┘                      │
│   感知      规划      生成      执行                          │
│   独立      独立      独立      独立                          │
│                                                              │
│  VLA模型：                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │              统一模型                    │                │
│  │  感知 + 规划 + 执行（联合优化）          │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 能力对比

| 能力维度 | Agentic Workflow | VLA模型 |
|----------|------------------|---------|
| **延迟** | ❌ 高（900-3200ms） | ✅ 低（90-220ms） |
| **实时控制** | ❌ 不适合 | ✅ 适合 |
| **复杂规划** | ✅ 适合（多步推理） | ❌ 不擅长 |
| **可解释性** | ✅ 高（可查看每步） | ❌ 低（黑盒） |
| **灵活性** | ✅ 高（模块可替换） | ❌ 低（需重新训练） |
| **训练成本** | ✅ 无需训练 | ❌ 需要大量数据和算力 |
| **部署成本** | ✅ 低（使用现成模型） | ❌ 高（需要GPU推理） |
| **错误恢复** | ✅ 容易（模块化设计） | ❌ 困难（端到端） |
| **知识迁移** | ✅ 利用LLM知识 | ✅ 利用预训练知识 |
| **新任务适应** | ✅ 快速（修改提示） | ❌ 慢（需要微调） |

### 4.3 适用场景

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| **实时控制**（如导航避障） | VLA | 需要低延迟响应 |
| **复杂任务规划**（如"准备早餐"） | Agentic | 需要多步推理和分解 |
| **研究原型** | Agentic | 快速迭代，无需训练 |
| **生产部署**（高频控制） | VLA | 稳定、低延迟 |
| **可解释性要求高** | Agentic | 可以查看每步决策 |
| **资源受限** | Agentic | 无需GPU训练 |

### 4.4 代表论文对比

| 方面 | Agentic Workflow | VLA模型 |
|------|------------------|---------|
| **代表论文** | SayCan, Code as Policies, VoxPoser, Inner Monologue | RT-1, RT-2, Octo, PaLM-E |
| **发表时间** | 2022-2023 | 2022-2024 |
| **研究机构** | Google, Stanford, USC等 | Google DeepMind, Berkeley等 |
| **核心贡献** | 模块化设计、代码生成、反馈机制 | 动作Token化、知识迁移、涌现能力 |

---

## 五、混合架构：最佳实践

### 5.1 为什么需要混合架构？

两种方式各有优劣，混合架构可以取长补短：

```
┌─────────────────────────────────────────────────────────────┐
│                    混合架构                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  高层规划：Agentic Workflow                                  │
│  ├── 优势：复杂推理、可解释、灵活                            │
│  └── 任务：任务分解、状态检查、错误恢复                      │
│                                                              │
│  底层控制：VLA模型                                           │
│  ├── 优势：低延迟、实时控制                                  │
│  └── 任务：动作生成、轨迹规划                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 混合架构设计

```python
class HybridController:
    """混合控制器：Agentic Workflow + VLA"""
    
    def __init__(self):
        # Agentic Workflow 组件
        self.vlm = VLMObserver()           # 感知
        self.planner = BehaviorPlanner()    # 规划
        self.feedback = FeedbackCollector() # 反馈
        
        # VLA 组件
        self.vla = load_vla_model("rt2")    # 底层控制
    
    def execute(self, instruction: str, image):
        # 阶段1：Agentic Workflow - 高层规划
        # 1.1 感知
        scene = self.vlm.observe(image)
        
        # 1.2 规划
        plan = self.planner.make_plan(instruction, scene)
        # 输出：["navigate_to(kitchen)", "pick(apple)", "place(table)"]
        
        # 阶段2：VLA - 底层控制
        for action in plan:
            # 2.1 VLA生成动作
            action_vector = self.vla(image, action)
            # 输出：[x, y, z, rx, ry, rz, gripper]
            
            # 2.2 执行
            result = self.robot.execute(action_vector)
            
            # 2.3 反馈（Agentic）
            if not result.success:
                # Agentic Workflow处理错误
                plan = self.planner.replan(result.error)
        
        return result
```

### 5.3 混合架构的优势

| 优势 | 说明 |
|------|------|
| **两全其美** | 高层推理用Agentic，底层控制用VLA |
| **可解释** | 高层决策可查看，底层动作低延迟 |
| **灵活** | 可以单独替换高层或底层模块 |
| **鲁棒** | VLA失败时可以用Agentic恢复 |

---

## 六、当前项目的实现分析

### 6.1 项目架构

当前项目采用 **Agentic Workflow** 架构：

```
┌─────────────────────────────────────────────────────────────┐
│                  当前项目架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    VLM.py                             │   │
│  │  - RobotCommandProcessor                              │   │
│  │  - VLMObserver                                        │   │
│  │  功能：图像理解、物体检测、场景描述                     │   │
│  │  模型：qwen3-vl-plus                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   planner.py                          │   │
│  │  - BehaviorPlanner                                    │   │
│  │  功能：LLM生成行为树、任务分解、状态检查                 │   │
│  │  模型：deepseek-chat                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   engineer.py                         │   │
│  │  - EngineerAgent                                      │   │
│  │  功能：LLM生成Python代码、动态动作注册                   │   │
│  │  模型：deepseek-chat                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   executor.py                         │   │
│  │  - SkillExecutor                                      │   │
│  │  功能：执行动作、调用硬件API、状态反馈                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    apis.py                            │   │
│  │  - RobotAPI                                           │   │
│  │  功能：统一硬件接口（导航、感知、操作、夹爪）             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 实现特点

| 特点 | 说明 |
|------|------|
| **架构类型** | ✅ Agentic Workflow（模块化工作流） |
| **VLA模型** | ❌ 未采用（没有端到端训练的VLA模型） |
| **感知** | VLM（qwen3-vl-plus） |
| **规划** | LLM（deepseek-chat, GLM-4.5-Flash） |
| **代码生成** | LLM（deepseek-chat） |
| **执行** | 预定义的API + 动态生成的代码 |

### 6.3 可改进方向

| 改进方向 | 说明 | 优先级 |
|---------|------|--------|
| **引入VLA** | 用于底层实时控制 | 中 |
| **闭环反馈** | 实现Inner Monologue机制 | 高 |
| **Affordance评估** | 实现SayCan的可行性评估 | 高 |
| **值图规划** | 实现VoxPoser的3D值图 | 中 |

---

## 七、总结与建议

### 7.1 核心结论

| 结论 | 说明 |
|------|------|
| **两种方式各有优劣** | Agentic适合复杂规划，VLA适合实时控制 |
| **可以共存** | 混合架构可以取长补短 |
| **当前项目采用Agentic** | 无需训练，灵活可扩展 |
| **建议引入VLA** | 用于底层实时控制，提升响应速度 |

### 7.2 选择建议

```
┌─────────────────────────────────────────────────────────────┐
│                    选择决策树                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  是否需要实时控制？                                          │
│  ├── 是 → 是否有训练资源？                                   │
│  │         ├── 是 → VLA模型                                 │
│  │         └── 否 → Agentic + 优化延迟                      │
│  │                                                           │
│  └── 否 → 是否需要复杂规划？                                 │
│            ├── 是 → Agentic Workflow                        │
│            └── 否 → 两者皆可                                 │
│                                                              │
│  最佳实践：混合架构                                          │
│  - 高层规划：Agentic Workflow                               │
│  - 底层控制：VLA模型                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 参考文献

**Agentic Workflow**：
1. SayCan: Do As I Can, Not As I Say (2022)
2. Code as Policies: Language Model Programs for Embodied Control (2022)
3. ProgPrompt: Generating Situated Robot Task Plans (2022)
4. Inner Monologue: Embodied Reasoning through Planning (2022)
5. VoxPoser: Composable 3D Value Maps (2023)
6. Socratic Models: Composing Zero-Shot Multimodal Reasoning (2022)

**VLA模型**：
1. RT-1: Robotics Transformer for Real-World Control (2022)
2. RT-2: Vision-Language-Action Models (2023)
3. Octo: An Open-Source Generalist Robot Policy (2024)
4. PaLM-E: An Embodied Multimodal Language Model (2023)

---

**文档创建时间**：2026-03-18  
**关键词**：Agentic Workflow, VLA, 具身智能, 机器人控制
