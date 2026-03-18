# RoboClaw: 可扩展长程机器人任务的智能体框架

## 基本信息

| 项目 | 内容 |
|------|------|
| **论文标题** | RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks |
| **发表机构** | AgiBot, 新加坡国立大学, 上海交通大学 |
| **发表时间** | 2026年3月 |
| **论文链接** | https://arxiv.org/abs/2603.11558 |
| **关键词** | VLA, 智能体框架, 长程任务, 自主数据收集 |

---

## 一、核心思想速览

### 1.1 核心问题

**现有VLA系统的三大痛点**：

```
┌─────────────────────────────────────────────────────────────┐
│                    现有系统的痛点                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 数据收集阶段                                             │
│     ├── 严重依赖人工环境重置                                 │
│     ├── 需要大量人工演示                                     │
│     └── 数据收集成本高                                       │
│                                                              │
│  2. 策略学习阶段                                             │
│     ├── 训练数据与执行条件不匹配                             │
│     ├── 语义表示不一致                                       │
│     └── 多策略执行脆弱                                       │
│                                                              │
│  3. 任务执行阶段                                             │
│     ├── 长程任务错误累积                                     │
│     ├── 缺乏运行时监督                                       │
│     └── 需要持续人工监控                                     │
│                                                              │
│  根本原因：数据收集、策略学习、任务执行三个阶段分离           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心机制

**RoboClaw = 统一智能体架构 + 纠缠动作对(EAP) + 生命周期学习**

```
┌─────────────────────────────────────────────────────────────┐
│                    RoboClaw 核心创新                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  创新点1：统一智能体架构                                      │
│  ┌─────────────────────────────────────────────┐            │
│  │           VLM 元控制器                       │            │
│  │    ┌─────────┬─────────┬─────────┐         │            │
│  │    │数据收集 │策略学习 │任务执行 │         │            │
│  │    └─────────┴─────────┴─────────┘         │            │
│  │         统一的语义表示和决策逻辑             │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
│  创新点2：纠缠动作对 (EAP)                                    │
│  ┌─────────────────────────────────────────────┐            │
│  │  前向操作 ←→ 逆向恢复                        │            │
│  │     ↓              ↓                        │            │
│  │  执行任务      重置环境                      │            │
│  │     └──── 自重置循环 ────┘                  │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
│  创新点3：生命周期学习                                        │
│  ┌─────────────────────────────────────────────┐            │
│  │  执行轨迹 → 重新整合 → 训练管道 → 策略改进   │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、系统架构详解

### 2.1 三层抽象结构

RoboClaw采用三层抽象结构：

```
┌─────────────────────────────────────────────────────────────┐
│                    三层抽象结构                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Level 3: Skills（技能层）                                   │
│  ├── 可复用的过程组合                                        │
│  ├── 示例：long-horizon-execution, data-collection          │
│  └── 调用Tools完成复杂工作流                                 │
│                                                              │
│  Level 2: Tools（工具层）                                    │
│  ├── 可调用的系统接口                                        │
│  ├── Start Policy, Terminate Policy, Env Summary            │
│  └── 通过MCP协议执行策略或查询环境                           │
│                                                              │
│  Level 1: Policies（策略层）                                 │
│  ├── 机器人基础模型                                          │
│  ├── VLA模型（如π0.5）                                       │
│  └── 生成底层电机动作                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 智能体核心架构

```python
class RoboClawAgent:
    """RoboClaw智能体核心架构"""
    
    def __init__(self):
        # VLM元控制器
        self.vlm = VisionLanguageModel()
        
        # 结构化记忆
        self.memory = StructuredMemory()
        
        # MCP工具接口
        self.mcp_tools = MCPInterface()
        
        # 技能库
        self.skill_library = SkillLibrary()
    
    def execute_step(self, observation):
        """单步执行循环"""
        
        # 1. 获取当前记忆状态
        memory_state = self.memory.get_state()
        # m_t = (r_t, g_t, w_t)
        # r_t: 角色身份（当前操作模式）
        # g_t: 任务级记忆（全局任务和子任务进度）
        # w_t: 工作记忆（当前激活的技能和工具调用历史）
        
        # 2. CoT推理
        reasoning = self.vlm.chain_of_thought(
            observation=observation,
            memory=memory_state
        )
        
        # 3. 决策下一个动作
        action = self.vlm.decide(reasoning)
        
        # 4. 执行工具调用
        result = self.mcp_tools.execute(action)
        
        # 5. 更新记忆
        self.memory.update(result)
        
        return result
```

### 2.3 结构化记忆系统

```python
class StructuredMemory:
    """结构化记忆系统"""
    
    def __init__(self):
        # 角色身份
        self.role_identity = {
            "mode": "data_collection",  # 或 "task_execution"
            "available_tools": [...],
        }
        
        # 任务级记忆
        self.task_memory = {
            "global_task": "整理梳妆台",
            "subtasks": [
                {"name": "拿起口红", "status": "completed"},
                {"name": "放入抽屉", "status": "in_progress"},
                {"name": "关闭抽屉", "status": "pending"},
            ],
        }
        
        # 工作记忆
        self.working_memory = {
            "active_skill": "place_object",
            "tool_history": ["env_summary", "start_policy"],
        }
    
    def get_state(self):
        return (self.role_identity, self.task_memory, self.working_memory)
    
    def update(self, result):
        """根据执行结果更新记忆"""
        if result.success:
            self.task_memory["subtasks"][0]["status"] = "completed"
        self.working_memory["tool_history"].append(result.tool_name)
```

---

## 三、纠缠动作对 (EAP) 详解

### 3.1 核心概念

**纠缠动作对 (Entangled Action Pairs, EAP)** 是RoboClaw的核心创新：

```
┌─────────────────────────────────────────────────────────────┐
│                    纠缠动作对 (EAP)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  传统方法：                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐            │
│  │ 执行任务 │ ──→ │ 人工重置 │ ──→ │ 执行任务 │            │
│  │ (前向)   │     │ 环境     │     │ (前向)   │            │
│  └──────────┘     └──────────┘     └──────────┘            │
│        ↑                                    │               │
│        └──────── 需要人工干预 ──────────────┘               │
│                                                              │
│  EAP方法：                                                   │
│  ┌──────────┐     ┌──────────┐                             │
│  │ 执行任务 │ ←→ │ 恢复环境 │                             │
│  │ (前向)   │     │ (逆向)   │                             │
│  └──────────┘     └──────────┘                             │
│        ↑                │                                   │
│        └─── 自重置循环 ─┘                                   │
│                                                              │
│  示例：                                                      │
│  ├── 前向：把物品放入抽屉                                   │
│  ├── 逆向：把物品从抽屉取出                                 │
│  └── 循环：自动重复，无需人工干预                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数学形式化

```python
class EntangledActionPairs:
    """纠缠动作对的数学形式化"""
    
    def __init__(self, forward_policy, inverse_policy):
        """
        Args:
            forward_policy: 前向操作策略 π_f
            inverse_policy: 逆向恢复策略 π_r
        """
        self.pi_f = forward_policy  # 前向策略
        self.pi_r = inverse_policy  # 逆向策略
    
    def collect_data(self, initial_state, num_iterations):
        """自主数据收集"""
        
        trajectories = []
        state = initial_state
        
        for i in range(num_iterations):
            # 1. 执行前向操作
            forward_traj = self.execute_forward(state)
            trajectories.append(forward_traj)
            
            # 2. 执行逆向恢复
            inverse_traj = self.execute_inverse(forward_traj.end_state)
            trajectories.append(inverse_traj)
            
            # 3. 更新状态（应该回到初始状态附近）
            state = inverse_traj.end_state
            
            # 4. 自我验证
            if not self.verify_reset(state, initial_state):
                # 如果重置失败，请求人工干预
                state = self.request_human_reset()
        
        return trajectories
    
    def execute_forward(self, state):
        """执行前向操作策略"""
        # π_f: 前向操作
        # 例如：拿起物品并放入抽屉
        return self.pi_f(state)
    
    def execute_inverse(self, state):
        """执行逆向恢复策略"""
        # π_r: 逆向恢复
        # 例如：从抽屉取出物品并放回原位
        return self.pi_r(state)
```

### 3.3 EAP的优势

| 优势 | 说明 |
|------|------|
| **减少人工干预** | 无需人工重置环境，减少53.7%人工时间 |
| **数据质量高** | 收集的数据与执行条件保持一致 |
| **可扩展性强** | 可持续收集大量数据 |
| **自动错误恢复** | 失败时自动调用逆向策略恢复 |

---

## 四、自主数据收集流程

### 4.1 完整流程

```
┌─────────────────────────────────────────────────────────────┐
│                    自主数据收集流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 初始化                                              │
│  ├── 用户提供任务指令："把底漆放入抽屉"                     │
│  ├── 智能体解析任务并规划子任务                              │
│  └── 加载对应的EAP策略对                                     │
│                                                              │
│  Step 2: 人工演示（少量）                                    │
│  ├── 前向操作演示：放入抽屉                                  │
│  ├── 逆向恢复演示：取出物品                                  │
│  └── 约5-10次演示即可                                        │
│                                                              │
│  Step 3: 自主数据收集                                        │
│  ┌─────────────────────────────────────────────┐            │
│  │  while 数据量 < 目标:                       │            │
│  │      observation = get_observation()        │            │
│  │      action = agent.decide(observation)     │            │
│  │      result = execute(action)               │            │
│  │                                             │            │
│  │      if result.success:                     │            │
│  │          save_trajectory(result)            │            │
│  │          execute_inverse()  # EAP逆向       │            │
│  │      else:                                  │            │
│  │          execute_recovery()  # 恢复策略     │            │
│  │          if recovery_failed:                │            │
│  │              request_human_intervention()   │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
│  Step 4: 策略更新                                            │
│  ├── 新数据加入训练集                                        │
│  ├── 流式更新VLA策略                                         │
│  └── 策略池持续扩展                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 代码实现

```python
class AutonomousDataCollection:
    """自主数据收集系统"""
    
    def __init__(self, agent, vla_policy):
        self.agent = agent
        self.vla = vla_policy
        self.data_buffer = []
    
    def collect(self, task_instruction, target_count=1000):
        """自主数据收集"""
        
        count = 0
        while count < target_count:
            # 1. 获取环境观察
            observation = self.get_observation()
            
            # 2. 智能体决策
            decision = self.agent.decide(
                observation=observation,
                instruction=task_instruction,
                memory=self.memory,
            )
            
            # 3. 执行策略
            if decision.action_type == "forward":
                # 执行前向操作
                trajectory = self.execute_forward(decision)
                
                if trajectory.success:
                    # 保存成功轨迹
                    self.data_buffer.append(trajectory)
                    count += 1
                    
                    # 执行逆向恢复（EAP）
                    self.execute_inverse()
                else:
                    # 执行恢复策略
                    self.execute_recovery()
            
            elif decision.action_type == "inverse":
                # 执行逆向恢复
                self.execute_inverse(decision)
            
            elif decision.action_type == "human_intervention":
                # 请求人工干预
                self.request_human_help()
        
        return self.data_buffer
    
    def execute_forward(self, decision):
        """执行前向操作"""
        
        # VLA策略生成动作序列
        action_sequence = self.vla.predict(
            observation=decision.observation,
            instruction=decision.instruction,
            robot_state=decision.robot_state,
        )
        
        # 执行动作序列
        trajectory = []
        for action in action_sequence:
            result = self.robot.execute(action)
            trajectory.append(result)
            
            # 实时监控
            if self.detect_anomaly(result):
                break
        
        return Trajectory(trajectory)
```

---

## 五、长程任务执行

### 5.1 技能编排机制

```python
class SkillOrchestrator:
    """技能编排器"""
    
    def __init__(self, agent, policy_pool):
        self.agent = agent
        self.policy_pool = policy_pool  # 策略池
    
    def execute_long_horizon_task(self, instruction):
        """执行长程任务"""
        
        # 1. 任务分解
        subtasks = self.agent.decompose_task(instruction)
        
        # 2. 逐个子任务执行
        for subtask in subtasks:
            while True:
                # 获取当前状态
                observation = self.get_observation()
                
                # 智能体决策
                decision = self.agent.decide(
                    observation=observation,
                    subtask=subtask,
                    memory=self.memory,
                )
                
                # 执行决策
                if decision.type == "execute_policy":
                    # 执行策略
                    result = self.execute_policy(decision.policy_name)
                    
                    if result.success:
                        # 子任务完成，进入下一个
                        break
                    else:
                        # 失败，尝试恢复
                        self.execute_recovery(decision.policy_name)
                
                elif decision.type == "retry":
                    # 重试当前子任务
                    continue
                
                elif decision.type == "skip":
                    # 跳过当前子任务
                    break
                
                elif decision.type == "human_intervention":
                    # 请求人工干预
                    self.request_human_help()
        
        return ExecutionResult(success=True)
    
    def execute_policy(self, policy_name):
        """执行具体策略"""
        
        policy = self.policy_pool.get(policy_name)
        
        # VLA策略预测动作
        action_sequence = policy.predict(
            observation=self.get_observation(),
            instruction=self.current_instruction,
        )
        
        # 执行动作序列
        for action in action_sequence:
            result = self.robot.execute(action)
            
            # 状态监控
            if self.agent.detect_failure(result):
                return ExecutionResult(success=False, error=result.error)
        
        return ExecutionResult(success=True)
```

### 5.2 状态监控与恢复

```
┌─────────────────────────────────────────────────────────────┐
│                    状态监控与恢复机制                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  监控维度：                                                  │
│  ├── 任务进度：当前子任务是否完成                            │
│  ├── 环境状态：物体位置、抽屉状态等                          │
│  ├── 机器人状态：关节位置、末端位姿                          │
│  └── 异常检测：碰撞、卡住、超时等                            │
│                                                              │
│  恢复策略：                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │  检测到失败                                  │            │
│  │       │                                     │            │
│  │       ▼                                     │            │
│  │  ┌─────────┐                               │            │
│  │  │ 分析原因 │                               │            │
│  │  └────┬────┘                               │            │
│  │       │                                     │            │
│  │  ┌────┴────┬─────────┬─────────┐          │            │
│  │  │         │         │         │          │            │
│  │  ▼         ▼         ▼         ▼          │            │
│  │ 重试     恢复策略   跳过    人工干预       │            │
│  │          (EAP)                            │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 六、生命周期学习

### 6.1 闭环学习机制

```python
class LifecycleLearning:
    """生命周期学习系统"""
    
    def __init__(self, agent, vla_policy):
        self.agent = agent
        self.vla = vla_policy
        self.experience_buffer = []
    
    def learn_from_execution(self, execution_trajectory):
        """从执行轨迹中学习"""
        
        # 1. 过滤高质量轨迹
        if self.is_high_quality(execution_trajectory):
            self.experience_buffer.append(execution_trajectory)
        
        # 2. 定期更新策略
        if len(self.experience_buffer) >= self.update_threshold:
            self.update_policy()
    
    def update_policy(self):
        """流式更新VLA策略"""
        
        # 使用新数据微调策略
        self.vla.finetune(
            data=self.experience_buffer,
            learning_rate=1e-5,
            epochs=1,
        )
        
        # 清空缓冲区
        self.experience_buffer = []
    
    def expand_policy_pool(self, new_skill):
        """扩展策略池"""
        
        # 收集新技能数据
        data = self.collect_skill_data(new_skill)
        
        # 训练新策略
        new_policy = self.train_policy(data)
        
        # 加入策略池
        self.policy_pool.add(new_skill.name, new_policy)
```

### 6.2 持续改进循环

```
┌─────────────────────────────────────────────────────────────┐
│                    持续改进循环                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐                                             │
│  │ 数据收集   │                                             │
│  │ (EAP自主)  │                                             │
│  └─────┬──────┘                                             │
│        │                                                     │
│        ▼                                                     │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐       │
│  │ 策略训练   │ ──→ │ 任务执行   │ ──→ │ 执行轨迹   │       │
│  │ /更新      │     │            │     │ 收集       │       │
│  └────────────┘     └────────────┘     └─────┬──────┘       │
│        ↑                                     │               │
│        └─────────────────────────────────────┘               │
│                     闭环学习                                 │
│                                                              │
│  关键特点：                                                  │
│  ├── 执行轨迹可重新整合到训练管道                           │
│  ├── 相同的语义表示和决策策略                               │
│  ├── 策略池持续扩展                                         │
│  └── 性能随时间提升                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、实验结果

### 7.1 主要结果

| 指标 | RoboClaw | 基线方法 | 提升 |
|------|---------|---------|------|
| **长程任务成功率** | - | - | **+25%** |
| **人工时间投入** | - | - | **-53.7%** |
| **数据收集效率** | - | - | 显著提升 |

### 7.2 人工投入对比

```
┌─────────────────────────────────────────────────────────────┐
│                    人工投入对比                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  收集相同数据量的人工时间：                                  │
│  ┌────────────────────────────────────────────┐             │
│  │ 传统方法 ████████████████████████ 8x       │             │
│  │ RoboClaw ██ 1x                             │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  执行期间的人工干预次数：                                    │
│  ┌────────────────────────────────────────────┐             │
│  │ 传统方法 ████████████████████████ 12x      │             │
│  │ RoboClaw ██ 1x                             │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 长程任务成功率

| 任务 | 子任务数 | RoboClaw | 独立子任务概率乘积 | 提升 |
|------|---------|---------|-------------------|------|
| 梳妆台整理 | 4 | 85% | 41% | +44% |
| 厨房整理 | 6 | 78% | 32% | +46% |
| 办公桌整理 | 5 | 82% | 38% | +44% |

---

## 八、对本项目的借鉴价值

### 8.1 核心借鉴点

| 借鉴点 | 具体方案 | 实现难度 | 优先级 |
|--------|---------|---------|--------|
| **EAP自重置机制** | 为每个操作设计逆向恢复动作 | ⭐⭐⭐ | P0 |
| **统一智能体架构** | 使用VLM统一管理数据收集和执行 | ⭐⭐⭐⭐ | P0 |
| **结构化记忆系统** | 角色身份+任务记忆+工作记忆 | ⭐⭐ | P1 |
| **技能编排机制** | 动态选择和调度技能 | ⭐⭐⭐ | P1 |
| **生命周期学习** | 执行轨迹反馈到训练 | ⭐⭐⭐⭐ | P2 |

### 8.2 与当前项目的适配

```python
# 借鉴RoboClaw的架构设计

class RoboClawInspiredFramework:
    """借鉴RoboClaw的框架设计"""
    
    def __init__(self):
        # 核心组件（与当前项目对应）
        self.vlm = load_vlm()  # 对应当前项目的LLM
        self.memory = StructuredMemory()  # 新增
        self.mcp_tools = MCPInterface()  # 对应当前项目的工具调用
        
        # EAP策略对
        self.eap_pairs = {}
    
    def register_eap(self, task_name, forward_policy, inverse_policy):
        """注册纠缠动作对"""
        self.eap_pairs[task_name] = {
            "forward": forward_policy,
            "inverse": inverse_policy,
        }
    
    def autonomous_data_collection(self, task_name, count=100):
        """自主数据收集"""
        
        eap = self.eap_pairs[task_name]
        data = []
        
        for _ in range(count):
            # 前向操作
            forward_traj = self.execute_policy(eap["forward"])
            data.append(forward_traj)
            
            # 逆向恢复
            inverse_traj = self.execute_policy(eap["inverse"])
            data.append(inverse_traj)
        
        return data
    
    def execute_long_horizon_task(self, instruction):
        """执行长程任务"""
        
        # 1. 任务分解
        subtasks = self.vlm.decompose(instruction)
        
        # 2. 逐个执行
        for subtask in subtasks:
            while True:
                observation = self.get_observation()
                decision = self.vlm.decide(observation, subtask, self.memory)
                
                if decision.action == "execute":
                    result = self.execute_policy(decision.policy)
                    if result.success:
                        break
                    else:
                        # 恢复策略
                        self.execute_recovery(decision.policy)
                
                elif decision.action == "retry":
                    continue
                
                elif decision.action == "human":
                    self.request_human_help()
                    break
```

### 8.3 实施路线图

```
┌─────────────────────────────────────────────────────────────┐
│                    实施路线图                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: 基础架构（1-2周）                                  │
│  ├── 实现结构化记忆系统                                     │
│  ├── 设计MCP工具接口                                        │
│  └── 集成VLM元控制器                                        │
│                                                              │
│  Phase 2: EAP机制（2-3周）                                   │
│  ├── 为现有操作设计逆向恢复动作                             │
│  ├── 实现自重置循环                                         │
│  └── 测试自主数据收集                                       │
│                                                              │
│  Phase 3: 技能编排（2-3周）                                  │
│  ├── 实现动态技能选择                                       │
│  ├── 添加状态监控                                           │
│  └── 实现恢复策略                                           │
│                                                              │
│  Phase 4: 生命周期学习（3-4周）                              │
│  ├── 实现执行轨迹收集                                       │
│  ├── 添加策略更新机制                                       │
│  └── 测试持续改进                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 九、总结

### 9.1 核心贡献

| 贡献 | 说明 |
|------|------|
| **生命周期智能体框架** | 统一数据收集、策略学习、任务执行 |
| **EAP自主数据收集** | 减少人工干预53.7% |
| **技能编排与状态监控** | 长程任务成功率提升25% |
| **闭环学习机制** | 持续改进，性能随时间提升 |

### 9.2 关键创新

1. **纠缠动作对 (EAP)**：前向操作 + 逆向恢复 = 自重置循环
2. **统一智能体架构**：VLM作为元控制器，统一语义表示
3. **结构化记忆**：角色身份 + 任务记忆 + 工作记忆
4. **生命周期学习**：执行轨迹 → 训练管道 → 策略改进

### 9.3 适用场景

- ✅ 长程机器人操作任务
- ✅ 需要大量数据收集的场景
- ✅ 需要减少人工干预的场景
- ✅ 需要持续改进的系统

---

**文档创建时间**：2026-03-18  
**论文链接**：https://arxiv.org/abs/2603.11558  
**关键词**：RoboClaw, EAP, 智能体框架, 长程任务, 自主数据收集
