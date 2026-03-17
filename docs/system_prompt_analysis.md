# System Prompt 分析与优化指南

## 一、什么是 System Prompt？

### 1.1 定义

**System Prompt**（系统提示词）是发送给大语言模型（LLM）的特殊指令，用于设定模型的角色、行为方式和输出格式。

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM 消息结构                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  messages = [                                                │
│      {                                                       │
│          "role": "system",    ← 系统提示词                   │
│          "content": "你是服务机器人任务规划器..."            │
│      },                                                      │
│      {                                                       │
│          "role": "user",      ← 用户输入                     │
│          "content": "帮我拿一瓶可乐"                         │
│      },                                                      │
│      {                                                       │
│          "role": "assistant", ← 模型回复                     │
│          "content": "好的，我来帮你拿可乐..."                │
│      }                                                       │
│  ]                                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 System Prompt 的作用

| 作用 | 说明 |
|------|------|
| **角色设定** | 告诉模型"你是谁" |
| **行为约束** | 告诉模型"该怎么做" |
| **输出格式** | 告诉模型"输出什么样" |
| **知识边界** | 告诉模型"知道什么/不知道什么" |

---

## 二、项目中的 System Prompt 使用位置

### 2.1 总览

```
┌─────────────────────────────────────────────────────────────┐
│                System Prompt 使用位置                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. VLM.py - 意图识别                                        │
│     └── 判断用户语音是聊天还是指令                           │
│                                                              │
│  2. VLM.py - 聊天回复                                        │
│     └── 生成对用户的自然语言回复                             │
│                                                              │
│  3. planner.py - 行为树规划                                  │
│     └── 将任务分解为行为树节点                               │
│                                                              │
│  4. planner.py - 失败反思                                    │
│     └── 分析失败原因并提出改进建议                           │
│                                                              │
│  5. processor.py - Function Call                             │
│     └── 控制机器人执行抓取任务                               │
│                                                              │
│  6. observer.py - 目标检测                                   │
│     └── VLM 检测目标物体位置                                 │
│                                                              │
│  7. engineer.py - 代码生成                                   │
│     └── 根据需求生成 Python 动作模块                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、各位置详细分析

### 3.1 意图识别（VLM.py）

**位置**：`voice/agents/VLM.py` - `process_command()`

**当前实现**：

```python
prompt = f"""
请你扮演一台语音交互机器人上的决策模块，以下中文语音文本是用户对你说的话，请你判断用户的意图，并将判断结果以JSON格式返回。
特别注意：若用户文本明显不是在与你沟通，如文本不完整、无意义、内容杂乱，判断结果的confidence必须设为0。
在与用户交互时，你可以根据传入的图片辅助理解，可以在回复时加上礼貌得体的问候等,如果你不需要图片信息，可以忽略图片部分。

语音文本："{text}"

可能的意图包括：
1. 聊天 - 普通对话内容（关键词：你好、天气、新闻、笑话等）
2. 指令 - 控制机器人执行具体的动作（支持的动作类型：打招呼/摆手、摇头、点头、鞠躬、其他）
...

输出的标准格式如下：
{{
    "intent": "command"或"chat",
    "action": "动作类型...",
    "obj_name": "饮料类型...",
    "num": "数量...",
    "confidence": 0或1,
    "description": "意图或动作描述"
    "response" : "当intent为command时，给出对用户的简短回应"
}}
"""
```

**问题分析**：

| 问题 | 影响 |
|------|------|
| Prompt 过长 | Token 消耗大，响应慢 |
| 动作类型硬编码 | 新增动作需要修改代码 |
| 饮料列表硬编码 | 泛化能力差 |
| 中英文混杂 | 可能影响理解 |

**优化建议**：

```python
INTENT_SYSTEM_PROMPT = """
你是机器人的意图识别模块。分析用户输入，返回JSON格式的意图判断结果。

## 输出格式
{
    "intent": "chat|command",
    "action": "动作类型",
    "params": {"obj_name": "对象", "num": 数量},
    "confidence": 0.0-1.0,
    "response": "简短回应"
}

## 动作类型
{action_list}

## 可识别对象
{object_list}

## 规则
1. confidence > 0.8 表示明确意图
2. 无法识别时 intent 设为 "chat"
3. 只返回 JSON，无其他内容
"""
```

---

### 3.2 聊天回复（VLM.py）

**位置**：`voice/agents/VLM.py` - `generate_chat_response()`

**当前实现**：

```python
chat_prompt = f"""
用户的输入是："{text}"
你的名字是"家卡"，你是中国人民大学机器人创新实践基地研发的一台人型机器人，你目前会执行的动作包括：打招呼、摇头、点头、鞠躬。
如果用户用类似的称呼比如节卡,可能也是在叫你。请根据用户的输入，生成一个自然的回答，简单问题回答可以简短一点，复杂问题回答可以长一点，不过必须在200字以内，不然会超出token限制。注意，回答会被TTS语音系统朗读：
【默认行为】
- 大多数情况下，用户只是和你聊天或随口提问，并不是在下达任务命令。请直接进行自然的对话，不用说"我会根据你的要求回答"之类的话。也不要重复用户的输入。
例如：
用户说：1+1等于几？
你应该回答：1+1等于2。

【语气风格要求】
- 不要使用"好的"、"当然可以"等作为开头。
- 不要使用"嘿"、"哎呀"这类拟声词，也不要加表情符号。
- 不要在回答前加"你说得是"、"家卡同学说"等固定前缀。
- 不要在回答中使用表情或者任何用声音念出来会让人难以理解的词语,因为你的输出会交给TTS语音系统朗读。

【身份规则】
- 你是一个带语音交互功能的有单手且装有轮子的人形机器人。
- 用户叫你"家卡"时，是在叫你，不是说公司。
- 你底层的大模型用的是智谱的GLM-4.5-Flash
- 你现在所在的地方是北京市海淀区中国人民大学机器人创新实践基地。
"""
```

**问题分析**：

| 问题 | 影响 |
|------|------|
| 身份信息硬编码 | 更换场景需要改代码 |
| 动作列表硬编码 | 新增动作需要改 prompt |
| 规则分散 | 维护困难 |

**优化建议**：

```python
CHAT_SYSTEM_PROMPT = """
你是{name}，{description}。

## 身份信息
- 类型：{robot_type}
- 位置：{location}
- 能力：{capabilities}
- 底层模型：{llm_model}

## 回复规则
1. 字数限制：{max_words}字以内
2. 输出会被 TTS 朗读，避免：
   - 表情符号、拟声词
   - 难以朗读的内容
3. 直接回答，不要重复用户输入
4. 语气自然友好

## 当前上下文
{context}
"""
```

---

### 3.3 行为树规划（planner.py）

**位置**：`voice/agents/planner.py` - `plan()`

**当前实现**：

```python
payload = {
    "model": self.llm_model,
    "messages": [
        {
            "role": "system",
            "content": "你是服务机器人任务规划器，返回符合要求的JSON行为树。",
        },
        {
            "role": "user",
            "content": json.dumps(prompt, ensure_ascii=False),
        },
    ],
    "temperature": 0.1,
    "top_p": 0.85,
    "response_format": {"type": "json_object"},
}
```

**问题分析**：

| 问题 | 影响 |
|------|------|
| System Prompt 过于简单 | 缺乏约束，输出可能不稳定 |
| 没有说明可用动作 | 模型可能生成不存在的动作 |
| 没有错误处理指导 | 失败时不知道如何处理 |

**优化建议**：

```python
PLANNER_SYSTEM_PROMPT = """
你是机器人任务规划器，负责将用户任务分解为行为树。

## 可用动作
{available_actions}

## 行为树节点类型
- sequence: 顺序执行
- selector: 选择执行
- check: 条件检查
- action: 执行动作
- repeat_until: 循环执行

## 输出格式
{
    "type": "sequence",
    "children": [...]
}

## 规则
1. 只使用可用动作列表中的动作
2. 每个动作必须有明确的参数
3. 复杂任务需要分解为多个步骤
4. 输出必须是合法的 JSON
"""
```

---

### 3.4 失败反思（planner.py）

**位置**：`voice/agents/planner.py` - `reflect_on_failure()`

**当前实现**：

```python
payload = {
    "model": self.llm_model,
    "messages": [
        {
            "role": "system",
            "content": "你是机器人任务专家，负责分析失败原因并提出改进建议，务必输出JSON。",
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ],
    "temperature": 0.2,
    "top_p": 0.8,
    "response_format": {"type": "json_object"},
}
```

**优化建议**：

```python
REFLECTION_SYSTEM_PROMPT = """
你是机器人任务执行分析专家，负责分析失败原因并提出改进建议。

## 分析维度
1. 环境因素：光照、遮挡、物体位置
2. 硬件因素：传感器、执行器状态
3. 算法因素：检测精度、规划合理性
4. 任务因素：目标是否可达

## 输出格式
{
    "failure_type": "perception|planning|execution|environment",
    "root_cause": "根本原因",
    "suggestions": ["建议1", "建议2"],
    "retry_strategy": "重试策略"
}

## 规则
1. 分析必须基于实际错误信息
2. 建议必须可执行
3. 输出必须是合法的 JSON
"""
```

---

### 3.5 Function Call（processor.py）

**位置**：`voice/agents/function_call/processor.py`

**当前实现**：

```python
self.system_prompt = os.getenv(
    "FUNCTION_CALL_SYSTEM_PROMPT",
    (
        "You are a robotics planner controlling a real robot. "
        "CRITICAL RULES:\n"
        "1. You MUST use tool_calls (function calls) to execute any action. NEVER describe actions in text.\n"
        "2. Do NOT say 'I will call predict_grasp_point' - instead, actually call the function.\n"
        "3. Do NOT include function names in your text response. Use the tools directly.\n"
        "4. Only respond with a final JSON summary AFTER the grasp is complete or truly impossible.\n"
        "5. Never fabricate execution results - always call the actual function and wait for results.\n"
        "If you need to predict a grasp point, call predict_grasp_point(). If you need to execute grasp, call execute_grasp()."
    ),
)
```

**优点**：
- ✅ 规则清晰明确
- ✅ 支持环境变量配置
- ✅ 强调必须调用函数

**可优化点**：

```python
FUNCTION_CALL_SYSTEM_PROMPT = """
你是机器人控制系统，通过函数调用控制真实机器人执行抓取任务。

## 核心规则
1. **必须使用函数调用**：不要在文本中描述动作，直接调用函数
2. **禁止编造结果**：必须等待函数返回真实结果
3. **禁止重复调用**：同一函数不要重复调用

## 可用函数
{function_descriptions}

## 执行流程
1. 观察场景 → observe_scene()
2. 检测目标 → detect_target(target_name)
3. 预测抓取点 → predict_grasp_point()
4. 执行抓取 → execute_grasp()
5. 确认结果 → check_grasp_result()

## 错误处理
- 目标未找到：扩大搜索范围或报告失败
- 抓取失败：调整姿态后重试
- 连续失败3次：报告任务失败

## 输出格式
任务完成后返回 JSON 摘要：
{
    "success": true/false,
    "message": "执行结果描述",
    "steps_taken": ["步骤1", "步骤2", ...]
}
"""
```

---

### 3.6 目标检测（observer.py）

**位置**：`voice/perception/observer.py` - `_build_prompt()`

**当前实现**：

```python
def _build_prompt(self, target_name: str, ...) -> str:
    base = [
        f"你是部署在服务机器人上的视觉助手。目标物体是"{target_name}"。",
        "请分析图像并返回结构化JSON，仅包含视觉信息，不要给出动作建议。",
        "如果未发现目标，请保持found=false并给出简要中文分析。",
        "字段说明：",
        "{",
        '  "found": true/false,',
        '  "bbox": [x_min, y_max, x_max, y_min],',
        '  "confidence": number (0-1),',
        '  "surface_points": [[x, y], ...] // 1-2个背景平面点',
        "}",
    ]
    return "\n".join(base)
```

**优点**：
- ✅ 输出格式明确
- ✅ 强调只返回视觉信息

**优化建议**：

```python
DETECTION_SYSTEM_PROMPT = """
你是机器人的视觉感知模块，负责检测和定位目标物体。

## 任务
检测图像中的目标物体：{target_name}

## 输出格式
{
    "found": true/false,
    "bbox": [x_min, y_min, x_max, y_max],
    "confidence": 0.0-1.0,
    "surface_points": [[x1, y1], [x2, y2]],
    "analysis": "未找到时的分析说明"
}

## 坐标说明
- bbox: 目标边界框，像素坐标
- surface_points: 承载物体的平面点，用于计算高度
- 坐标原点：图像左上角

## 规则
1. 只返回视觉信息，不给出动作建议
2. 未找到目标时 found=false，并说明原因
3. 确保坐标在图像范围内
4. 只输出 JSON，无其他内容
"""
```

---

### 3.7 代码生成（engineer.py）

**位置**：`voice/agents/engineer.py`

**当前实现**：

```python
payload = {
    "model": os.getenv("DEEPSEEK_ENGINEER_MODEL", "deepseek-chat"),
    "messages": [
        {
            "role": "system",
            "content": "你是机器人应用的高级工程师，负责根据需求编写可靠的Python动作模块。",
        },
        {
            "role": "user",
            "content": json.dumps(prompt, ensure_ascii=False),
        },
    ],
    "temperature": 0.2,
    "top_p": 0.8,
}
```

**问题分析**：

| 问题 | 影响 |
|------|------|
| System Prompt 过于简单 | 生成的代码可能不符合规范 |
| 没有代码模板 | 代码风格不统一 |
| 没有错误处理要求 | 代码可能缺乏健壮性 |

**优化建议**：

```python
ENGINEER_SYSTEM_PROMPT = """
你是机器人应用开发工程师，负责根据需求编写 Python 动作模块。

## 代码规范
1. 函数签名：def action_name(runtime: SkillRuntime) -> ExecutionResult
2. 必须包含异常处理
3. 必须包含日志输出
4. 使用类型注解

## 可用 API
- runtime.navigator: 导航控制
- runtime.world_model: 世界模型
- runtime.observation: 观测数据
- runtime.extra: 额外参数

## 代码模板
```python
def {action_name}(runtime: SkillRuntime) -> ExecutionResult:
    \"\"\"动作描述\"\"\"
    try:
        log_info(f"开始执行: {action_name}")
        # 实现逻辑
        return ExecutionResult(success=True, message="执行成功")
    except Exception as e:
        log_error(f"执行失败: {e}")
        return ExecutionResult(success=False, message=str(e))
```

## 规则
1. 只生成可执行的 Python 代码
2. 使用项目已有的 API 和工具
3. 代码必须有完整的错误处理
"""
```

---

## 四、优化策略总结

### 4.1 模板化设计

```
┌─────────────────────────────────────────────────────────────┐
│                    System Prompt 模板                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  class SystemPromptTemplate:                                │
│      def __init__(self, template: str, variables: dict):    │
│          self.template = template                           │
│          self.variables = variables                         │
│                                                              │
│      def render(self, **kwargs) -> str:                     │
│          return self.template.format(                       │
│              **self.variables,                              │
│              **kwargs                                       │
│          )                                                  │
│                                                              │
│  # 使用示例                                                  │
│  INTENT_PROMPT = SystemPromptTemplate(                      │
│      template=INTENT_SYSTEM_PROMPT,                         │
│      variables={                                            │
│          "action_list": get_available_actions(),            │
│          "object_list": get_available_objects(),            │
│      }                                                      │
│  )                                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 配置化设计

```python
# config/prompts.yaml

intent_recognition:
  system_prompt: |
    你是机器人的意图识别模块...
  variables:
    action_list: ${actions.available}
    object_list: ${objects.available}

chat_response:
  system_prompt: |
    你是{name}，{description}...
  variables:
    name: ${robot.name}
    description: ${robot.description}
    max_words: 200
```

### 4.3 版本管理

```
prompts/
├── v1/
│   ├── intent.txt
│   ├── planner.txt
│   └── engineer.txt
├── v2/
│   ├── intent.txt
│   ├── planner.txt
│   └── engineer.txt
└── current -> v2/  # 软链接指向当前版本
```

---

## 五、优化收益预估

| 优化项 | 收益 |
|--------|------|
| **模板化** | 新增动作/对象只需修改配置，无需改代码 |
| **配置化** | 不同场景使用不同配置，提高泛化能力 |
| **版本管理** | 方便 A/B 测试，持续优化 |
| **结构化输出** | 减少 JSON 解析错误，提高稳定性 |

---

## 六、实施建议

### 6.1 短期优化（1-2周）

1. **提取硬编码内容**
   - 将动作列表、对象列表提取到配置文件
   - 将机器人身份信息提取到配置文件

2. **统一输出格式**
   - 所有 System Prompt 使用相同的 JSON 格式规范
   - 添加格式验证

### 6.2 中期优化（1个月）

1. **实现模板系统**
   - 创建 SystemPromptTemplate 类
   - 支持变量替换和条件渲染

2. **添加版本管理**
   - 建立 prompts 目录结构
   - 支持版本切换

### 6.3 长期优化（持续）

1. **A/B 测试**
   - 对比不同 System Prompt 的效果
   - 持续优化

2. **自动化评估**
   - 建立评估数据集
   - 自动评估 System Prompt 质量

---

**文档版本**：v1.0  
**创建日期**：2026-03-16  
**适用人群**：项目开发者、Prompt 工程师
