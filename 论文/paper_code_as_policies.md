# 论文精读：Code as Policies

## 论文基本信息

| 项目 | 内容 |
|------|------|
| **标题** | Code as Policies: Language Model Programs for Embodied Control |
| **作者** | Jacky Liang, Wenlong Huang, Fei Xia, et al. (Google Robotics) |
| **发表时间** | 2022年9月 |
| **会议** | ICRA 2023 |
| **arXiv ID** | 2209.07753 |
| **项目主页** | https://code-as-policies.github.io/ |

---

## 核心思想速览

### 🎯 核心问题：代码即策略

这篇论文的核心在于将**机器人策略表示为可执行的程序代码**，而非预定义的动作序列。

**核心痛点**：
- 传统机器人需要预定义大量的运动原语（motion primitives）
- 难以处理空间几何推理、模糊描述（如"更快"）等复杂情况
- 缺乏对新指令的泛化能力

### ⚙️ 核心机制：LLM 生成代码 → 直接执行

将机器人控制问题转化为**代码生成问题**：

```
用户指令 → LLM 生成 Python 代码 → 直接执行
```

**关键能力**：
- **空间几何推理**：使用 NumPy、Shapely 等库进行几何计算
- **泛化能力**：通过 few-shot 提示，适应新指令
- **上下文理解**：根据场景调整模糊描述的具体值（如"更快"的速度）

**代码示例**：
```python
# 用户指令："把蓝色方块放到绿色方块左边"
# LLM 生成的代码：

def execute_policy(api):
    # 感知：获取物体位置
    blue_cube = api.detect_object("blue cube")
    green_cube = api.detect_object("green cube")
    
    # 几何推理：计算目标位置
    target_pos = green_cube.position + np.array([-0.1, 0, 0])
    
    # 执行：抓取并放置
    api.pick(blue_cube)
    api.place(target_pos)
```

### 💡 核心意义

- ✅ **无需预定义原语**：LLM 自动组合 API 调用
- ✅ **强泛化能力**：通过 few-shot 学习新任务
- ✅ **可解释性**：生成的代码可读、可调试
- ✅ **空间推理**：利用代码库进行复杂几何计算

### 📊 一句话总结

> **"将机器人策略从预定义动作序列，转变为 LLM 生成的可执行代码"**

---

## 一、研究背景与动机

### 1.1 传统方法的局限

**传统机器人编程**：
```python
# 需要预定义大量原语
primitives = {
    "pick": pick_object,
    "place": place_object,
    "push": push_object,
    # ... 每个新任务都需要新原语
}
```

**问题**：
- ❌ 难以处理空间推理（如"左边"、"中间"）
- ❌ 难以理解模糊描述（如"更快"、"轻轻"）
- ❌ 泛化能力差，新任务需要重新编程

### 1.2 核心洞察

**关键发现**：
- LLM 在代码补全任务上表现优异
- 代码可以表达复杂的逻辑和几何推理
- Python 代码可以直接调用感知和控制 API

**解决思路**：
> 让 LLM 生成 Python 代码，直接作为机器人策略执行

---

## 二、方法：Code as Policies 框架

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户指令                                │
│          "把蓝色方块放到两个绿色方块中间"                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   LLM（代码生成）                            │
│                                                              │
│  输入：                                                      │
│  - 用户指令                                                  │
│  - Few-shot 示例（注释 + 代码）                              │
│  - 可用 API 列表                                             │
│                                                              │
│  输出：Python 代码                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   代码解析与执行                             │
│                                                              │
│  1. 解析代码                                                 │
│  2. 检查安全性（禁止危险操作）                               │
│  3. 执行代码                                                 │
│  4. 返回结果                                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   机器人执行                                 │
│                                                              │
│  - 调用感知 API（目标检测、位姿估计）                        │
│  - 调用控制 API（抓取、移动、放置）                          │
│  - 反馈执行结果                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Few-shot 提示结构

**提示模板**：
```python
# 示例 1
# 指令：把红色方块放进盒子里

def execute_policy(api):
    red_cube = api.detect_object("red cube")
    box = api.detect_object("box")
    api.pick(red_cube)
    api.place(box.position)
    return "success"

# 示例 2
# 指令：把所有方块排成一行

def execute_policy(api):
    cubes = api.detect_objects("cube")
    positions = np.linspace(-0.5, 0.5, len(cubes))
    for cube, x in zip(cubes, positions):
        target_pos = np.array([x, 0, 0])
        api.pick(cube)
        api.place(target_pos)
    return "success"

# 新指令：{user_instruction}
```

### 2.3 层次化代码生成

**核心创新**：递归定义未实现的函数

```python
# LLM 生成的代码可能包含未定义函数
def execute_policy(api):
    objects = api.detect_objects("cube")
    sorted_objects = sort_by_size(objects)  # 未定义
    # ...

# LLM 自动生成 sort_by_size 的实现
def sort_by_size(objects):
    return sorted(objects, key=lambda obj: obj.size)
```

**优势**：
- ✅ 可以生成更复杂的代码
- ✅ 提高代码的可读性和模块化
- ✅ 在 HumanEval 基准上达到 39.8% 的准确率

---

## 三、实验与结果

### 3.1 实验设置

**机器人平台**：
- 真实机器人：UR5 机械臂 + Robotiq 夹爪
- 仿真环境： tabletop 场景

**任务类型**：
1. **基础操作**：抓取、放置、推动
2. **空间推理**："放在中间"、"排成一行"
3. **模糊描述**："更快"、"轻轻"

### 3.2 实验结果

| 任务类型 | 成功率 |
|---------|--------|
| 基础操作 | 85% |
| 空间推理 | 72% |
| 模糊描述 | 68% |
| 新物体泛化 | 75% |

**关键发现**：
- ✅ 空间推理任务表现优异
- ✅ 对新物体有良好的泛化能力
- ✅ 能根据上下文调整模糊描述的具体值

---

## 四、创新点总结

| 创新点 | 描述 |
|--------|------|
| **1. 代码即策略** | 首次将机器人策略表示为 LLM 生成的可执行代码 |
| **2. 层次化代码生成** | 递归定义未实现函数，提高代码复杂度 |
| **3. 空间几何推理** | 利用代码库（NumPy、Shapely）进行复杂几何计算 |
| **4. Few-shot 泛化** | 通过少量示例学习新任务 |

---

## 五、可借鉴之处（针对我们的项目）

### 5.1 直接可借鉴

#### 借鉴 1：代码生成框架

**应用到我们的 `engineer.py`**：

```python
# voice/agents/engineer.py

class CodeEngineer:
    """代码生成工程师"""
    
    def generate_action_code(self, ticket: ActionTicket) -> str:
        prompt = self._build_prompt(ticket)
        code = self.llm.generate(prompt)
        return self._extract_code(code)
    
    def _build_prompt(self, ticket: ActionTicket) -> str:
        return f"""
# 示例 1
# 任务：抓取目标物体
def run(api, runtime, **kwargs):
    target = kwargs.get("target")
    obj = api.perception.observe(target)
    api.manipulation.grasp(obj)
    return {{"status": "success"}}

# 示例 2  
# 任务：导航到指定位置
def run(api, runtime, **kwargs):
    location = kwargs.get("location")
    api.navigation.goto_marker(location)
    return {{"status": "success"}}

# 新任务：{ticket.description}
def run(api, runtime, **kwargs):
    # 你的代码
"""
```

#### 借鉴 2：层次化代码生成

**改进代码生成质量**：

```python
def generate_hierarchical_code(self, task_description: str) -> str:
    # 第一轮：生成主函数
    main_code = self.llm.generate(f"# {task_description}\ndef run(api, runtime, **kwargs):\n    # ...")
    
    # 第二轮：识别未定义的函数
    undefined_functions = self._find_undefined_functions(main_code)
    
    # 第三轮：生成未定义函数的实现
    for func_name in undefined_functions:
        func_code = self.llm.generate(f"def {func_name}(...):\n    # ...")
        main_code += f"\n\n{func_code}"
    
    return main_code
```

#### 借鉴 3：安全性检查

**防止危险操作**：

```python
class CodeValidator:
    """代码安全验证器"""
    
    FORBIDDEN_OPERATIONS = [
        "os.system",
        "subprocess",
        "eval",
        "exec",
        "__import__",
    ]
    
    def validate(self, code: str) -> bool:
        # 检查禁止的操作
        for op in self.FORBIDDEN_OPERATIONS:
            if op in code:
                return False
        
        # 检查语法错误
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        
        return True
```

### 5.2 需要改进的地方

#### 改进 1：代码执行监控

**增加执行反馈**：

```python
class CodeExecutor:
    """代码执行器（带监控）"""
    
    def execute(self, code: str, api: RobotAPI) -> ExecutionResult:
        try:
            # 创建执行环境
            local_vars = {"api": api, "np": np}
            
            # 执行代码
            exec(code, {"__builtins__": {}}, local_vars)
            
            # 获取结果
            result = local_vars.get("run")(api, None)
            
            return ExecutionResult(success=True, result=result)
        
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
```

#### 改进 2：代码缓存与复用

**避免重复生成**：

```python
class CodeCache:
    """代码缓存系统"""
    
    def __init__(self):
        self.cache = {}  # task_description -> code
    
    def get_or_generate(self, task_description: str, generator) -> str:
        # 计算任务描述的哈希
        task_hash = hash(task_description)
        
        # 检查缓存
        if task_hash in self.cache:
            return self.cache[task_hash]
        
        # 生成新代码
        code = generator(task_description)
        self.cache[task_hash] = code
        
        return code
```

### 5.3 与我们项目的结合点

| Code as Policies 组件 | 我们的对应组件 | 改进方向 |
|----------------------|---------------|---------|
| LLM 代码生成 | `engineer.py` | 已有，可增强层次化生成 |
| 代码执行 | `executor.py` | 增加安全检查和监控 |
| API 封装 | `apis.py` | 已有，可直接使用 |
| Few-shot 示例 | 配置文件 | 创建示例库 |

---

## 六、总结与启发

### 6.1 核心思想

> **"代码是表达机器人策略的最佳语言"**

Code as Policies 的核心贡献在于：
1. **范式转变**：从预定义原语到代码生成
2. **能力扩展**：利用代码库进行复杂推理
3. **泛化提升**：通过 few-shot 学习新任务

### 6.2 对我们项目的启发

1. **架构层面**：
   - ✅ 我们的 `engineer.py` 已经实现了代码生成
   - ✅ 可以借鉴层次化代码生成
   - ✅ 可以增加代码安全检查

2. **技术层面**：
   - ✅ 创建 Few-shot 示例库
   - ✅ 实现代码缓存机制
   - ✅ 增强代码执行监控

3. **创新机会**：
   - 🚀 **代码优化**：自动优化生成的代码
   - 🚀 **多语言支持**：支持生成不同语言的代码
   - 🚀 **在线学习**：从执行结果中学习改进代码

### 6.3 实施建议

**短期（1-2 周）**：
1. 创建 Few-shot 示例库
2. 实现代码安全检查
3. 测试现有代码生成功能

**中期（1-2 个月）**：
1. 实现层次化代码生成
2. 增加代码缓存机制
3. 优化代码执行监控

**长期（3-6 个月）**：
1. 研究代码优化技术
2. 实现在线学习机制
3. 发表论文或开源项目

---

## 七、参考文献

1. **Code as Policies 论文**：Liang, J., et al. "Code as Policies: Language Model Programs for Embodied Control." ICRA 2023.
2. **HumanEval 基准**：Chen, M., et al. "Evaluating Large Language Models Trained on Code." arXiv 2021.
3. **Codex 模型**：Brown, T., et al. "Language Models are Few-Shot Learners." NeurIPS 2020.

---

**文档创建时间**：2026-03-16  
**论文 arXiv ID**：2209.07753  
**PDF 文件名**：Code_as_Policies_Language_Model_Programs.pdf
