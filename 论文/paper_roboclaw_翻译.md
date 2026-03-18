# RoboClaw: 一种可扩展长程机器人任务的智能体框架

**RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks**

---

**作者**: Ruiying Li¹'²*, Yunlang Zhou¹'³*, Yuyao Zhu¹'³, Kylin Chen¹, Jingyuan Wang¹, Sukai Wang¹, Kongtao Hu¹, Minhui Yu¹, Bowen Jiang¹, Zhan Su¹'³, Jiayao Ma¹, Xin He¹, Yongjian Shen¹, Yang Yang¹, Guanghui Ren¹, Maoqing Yao¹, Wenhao Wang¹†, and Yao Mu³'⁴†

**单位**:
1. AgiBot, China (智元机器人，中国)
2. National University of Singapore (新加坡国立大学)
3. Shanghai Jiao Tong University, Shanghai 200240, China (上海交通大学)
4. MoE Key Lab of Artificial Intelligence, AI Institute, SJTU (人工智能教育部重点实验室，上海交通大学人工智能研究院)

*Equal contribution (同等贡献)
† Corresponding authors (通讯作者): wangwenhao@agibot.com, muyao@sjtu.edu.cn

---

## 摘要 (Abstract)

Vision-Language-Action (VLA) 系统在语言驱动的机器人操作方面展现出了强大的潜力。然而，将其扩展到长程任务 (long-horizon tasks) 仍然具有挑战性。现有的流水线通常将数据收集、策略学习和部署分离开来，导致严重依赖人工环境重置 (manual environment resets) 以及脆弱的多策略执行 (brittle multi-policy execution)。我们提出了 RoboClaw，一种智能体机器人框架 (agentic robotics framework)，它在单一的 VLM 驱动控制器下统一了数据收集、策略学习和任务执行。在策略层面，RoboClaw 引入了纠缠动作对 (Entangled Action Pairs, EAP)，它将前向操作行为 (forward manipulation behaviors) 与逆向恢复动作 (inverse recovery actions) 耦合，形成用于自主数据收集的自重置循环 (self-resetting loops)。这种机制使得在最小人工干预的情况下实现持续的在线策略数据获取 (continuous on-policy data acquisition) 和迭代策略优化 (iterative policy refinement)。在部署阶段，同一个智能体执行高层推理 (high-level reasoning) 并动态编排已学习的策略原语 (learned policy primitives) 来完成长程任务。通过在收集和执行阶段保持一致的上下文语义 (consistent contextual semantics)，RoboClaw 减少了两个阶段之间的不匹配，并提高了多策略的鲁棒性。在真实世界操作任务中的实验表明，与传统的开环流水线相比，该方法具有更好的稳定性和可扩展性，同时在整个机器人生命周期中显著减少了人工投入，在长程任务上实现了比基线方法高 25% 的成功率提升，并将人工时间投入减少了 53.7%。

---

## 1 引言 (Introduction)

视觉-语言-动作 (Vision-Language-Action, VLA) 系统的最新进展已经展示了语言驱动机器人操作的巨大潜力，使多模态模型能够将语言指令和视觉观察直接映射到机器人动作 [3,4,5,8,14]。然而，将这一范式扩展到复杂的真实世界操作任务仍然是一个关键挑战。真实世界的机器人任务本质上是长程的和组合性的 (long-horizon and compositional)，需要按顺序执行多个相互依赖的子任务。

为此，VLA 系统通常依赖大规模机器人数据来学习多样化的任务策略。然而，在真实机器人环境中构建此类数据集通常需要大量的人工参与。操作人员必须收集演示、反复重置环境、监控失败、过滤轨迹、评估模型性能，并在下游长程任务执行期间监督机器人行为。随着任务复杂度的增加，这种以人为中心的数据收集和部署过程变得越来越昂贵且难以扩展。此外，这些阶段通常由不同的人员处理，导致系统流水线中出现信息鸿沟。因此，任务状态的解释、子任务边界或成功标准可能在不同阶段有所不同，使得在整个系统中难以保持一致的任务语义。

此外，当数据收集、模型学习和任务执行由独立的过程驱动时，训练数据覆盖的状态分布往往无法反映部署时遇到的条件，导致训练和执行之间的不匹配。语义和分布上的这种不一致使得长程任务特别脆弱，其中小错误可能会传播并在执行过程中级联放大。因此，一个关键挑战是如何在可扩展的语言驱动机器人系统中，在数据收集、策略学习和执行之间建立统一的语义表示和决策机制。

为了解决这个问题，我们引入了 RoboClaw，一种用于长程机器人操作的统一智能体架构 (unified agent architecture)，如图 1 所示。RoboClaw 遵循一个简单的交互范式：用户发送任务指令，机器人自主推理并执行任务。在这个框架中，视觉-语言模型 (Vision-Language-Model, VLM) 充当元控制器 (meta-controller)，通过上下文学习 (in-context learning, ICL) [7] 执行高层决策，对环境观察和结构化记忆进行推理。与依赖人工监督或预定义规划器的传统系统不同，RoboClaw 在单一的智能体循环中统一了数据收集、策略学习和任务执行，在整个系统生命周期中实现一致的任务语义和决策逻辑，并将机器人操作从人工门控操作 (human-gated operation) 转向智能体操作 (agentic operation)。

在数据获取阶段，RoboClaw 引入了纠缠动作对 (Entangled Action Pairs, EAP)，这是一种显著减少人工环境重置需求的机制。对于每个操作策略，我们将前向执行行为与互补的逆向恢复行为配对，形成一个自重置循环 (self-resetting loop)，允许机器人反复返回到可重用的前置条件区域 (reusable precondition region)。在智能体控制下，这些配对动作交替执行，实现持续的在线数据收集，无需频繁的人工干预。与依赖人工重置或演示的传统流水线相比，这种机制在保持收集数据与执行条件对齐的同时，大幅减少了人工投入。

在任务执行期间，RoboClaw 也依赖智能体来编排技能调用。智能体不是遵循静态技能序列或需要持续的人工监控，而是根据当前上下文动态选择和调度模块化技能。通过持续监控子任务状态并验证执行条件，智能体执行运行时监督 (runtime supervision) 并在必要时触发恢复行为，与基线方法相比，在长程任务上实现了 25% 更高的成功率。

最后，RoboClaw 建立了闭环生命周期学习机制 (closed-loop lifecycle learning mechanism)。在下游长程任务执行期间生成的执行轨迹可以在相同的上下文语义和决策策略下重新整合到训练流水线中，实现现有策略的持续改进和策略池的扩展。通过在单一智能体框架内统一数据获取、模型学习和任务执行，系统可以积累经验并随时间提高性能。当检测到异常情况或安全约束时，系统也可以请求人工干预，确保操作安全的同时将人工负担减少 53.7%。

我们的贡献总结如下：

- **机器人的生命周期智能体框架 (A lifecycle agentic framework for robotics)**：我们引入了 RoboClaw，一个统一数据收集、策略学习和长程任务执行的智能体框架，实现一致的上下文语义并显著减少人工负担。

- **学习驱动的自主数据收集 (Learning-driven autonomous data collection)**：我们提出了纠缠动作对 (EAP)，一种数据引擎，将前向操作策略与逆向行为耦合形成自重置循环，实现持续的在线数据收集并保持收集数据与执行条件的对齐。

- **长程任务的技能编排和状态监控 (Skill orchestration and status monitoring for long-horizon tasks)**：我们设计了一个上下文驱动的决策架构，其中 VLM 通过对结构化记忆的上下文学习执行高层推理，实现长程机器人操作的技能编排和状态监控。

---

## 2 相关工作 (Related Works)

### 2.1 闭环数据收集 (Closed-loop Data Collection)

最近的方法探索了闭环和半自动化流水线来扩展机器人学习，超越了标准的遥操作系统，如 AnyTeleop [18]、GELLO [27] 和 Mobile ALOHA [9]。为了减少真实世界中的人工负担，像 RoboCopilot [26] 这样的系统利用人在环的残差修正 (human-in-the-loop residual corrections)。Genie Centurion [23] 引入了一种由任务哨兵 (Task Sentinel) 指导的"回退-优化" (rewind-and-refine) 机制，该哨兵自主检测失败以请求人工干预，VLAC [29] 也有类似的机制。此外，FieldGen [24] 通过解耦操作阶段半自动化真实世界收集，仅使用人工演示进行精细操作，同时通过吸引场 (attraction fields) 自动合成多样化的预操作轨迹。

为了完全自动化数据收集，像 MimicGen [21]、GenH2R-Sim [25] 和 RoboCasa [17] 这样的系统在仿真中合成大规模演示。最近的进展还利用大语言模型 (LLMs) 进行任务规划和自动执行。例如，RoboTwin 2.0 [6] 使用 MLLMs 和仿真在环反馈来迭代验证和优化任务执行代码。HumanoidGen [13] 利用 LLMs 为人形机器人操作生成空间约束，并采用基于 STCR 的树搜索机制来改进长程任务规划。此外，CyberDemo [22] 通过自动课程学习 (Auto Curriculum Learning) 引入了学习驱动的闭环，根据策略当前的成功率动态调整数据增强的复杂性。

虽然这些工作成功地将闭环反馈集成到数据合成或人工辅助修正中，但它们在真实世界部署期间往往缺乏自主适应性。为了解决这一差距，我们的工作提出了一个完全学习驱动的自动化数据收集框架。关键的是，与需要人工干预或预定义场的系统不同，我们在推理期间引入了自主过程监控和技能调度。这使得能够在无需人工协助的情况下，在动态环境中实现实时错误恢复和鲁棒执行。

### 2.2 具身任务的基础模型 (Foundation Models for Embodied Tasks)

近年来，视觉-语言-动作 (VLA) 模型，如 PaLM-E [?]、RT-2 [4]、OpenVLA [14] 和 π₀ [3]，通过统一感知、语言和动作，推动了语言条件化的机器人控制发展，但在长程任务中仍然容易受到错误累积的影响。大语言模型也被用于规划，包括作为零样本规划器的语言模型 (Language Models as Zero-Shot Planners) [10]、代码即策略 (Code as Policies) [16] 和 VoxPoser [11]，改进了任务分解但提供的执行时监督有限。分层方法如 SayCan [1]、HAMSTER [15]、HiRobot [20] 和 Agentic Robot [28] 引入了结构化的子任务抽象和规划-验证机制，而 π₀.₅ [2] 在统一的 VLA 框架内加强了多阶段推理。Inner Monologue [12] 和 LITEN [19] 通过重规划增强了鲁棒性，但执行期间的持续过程级监督 (sustained process-level supervision) 仍然很大程度上未被探索。

相比之下，我们提出了一种在推理时运行的上下文感知监督智能体 (context-aware supervisory agent)，持续监控子任务执行并动态选择重试、恢复或人工干预策略。与特定任务结构或技能库解耦，我们的设计实现了可扩展的真实世界长程具身智能体。

---

## 3 方法 (Method)

我们提出了 RoboClaw，一种用于长程机器人操作的智能体框架，它统一了自主数据收集和任务执行。RoboClaw 使用现成的视觉-语言模型 (VLM) 作为高层控制器，对视觉观察和系统上下文进行推理，决定调用哪个技能。

系统在闭环智能体交互周期中运行。给定观察和结构化记忆，VLM 执行思维链 (chain-of-thought, CoT) 推理来解释当前任务状态、评估进度并确定下一个动作。

RoboClaw 以 OpenClaw 风格将结构化记忆与模块化技能库集成，使智能体能够为复杂工作流（如数据收集和任务执行）组合可重用的能力。我们将系统构建为三个层次的抽象：技能 (Skills)、工具 (Tools) 和策略 (Policies)，其中更高层调用更低层来完成任务。我们按相反顺序介绍这三个组件：

- **策略 (Policies)**：指产生低级电机动作的机器人基础模型，在我们的系统中实现为视觉-语言-动作 (VLA) 模型。
- **工具 (Tools)**：是可调用的系统接口（如启动策略、终止策略、环境摘要），允许智能体通过模型上下文协议 (Model Context Protocol, MCP) 执行策略或查询环境。
- **技能 (Skills)**：表示编排工具的可重用过程，例如，"长程执行" (long-horizon-execution) 技能可能调用环境摘要，然后调用启动策略来执行操作策略。

### 3.1 通过 RoboClaw 智能体框架实现自主机器人任务执行和数据收集

基于上述分层设计，我们现在介绍使 RoboClaw 能够执行自主任务执行和数据收集的整体执行框架。

如图 2 所示，RoboClaw 将智能体的感知、推理和动作组织成一个闭环决策过程，迭代更新记忆并与环境交互。

在每个时间步 t，智能体维护一个结构化记忆状态 m_t，为推理和规划提供上下文信息。记忆由三个组件组成：

- **角色身份 r_t**：指定智能体当前的操作模式和可用工具集。
- **任务级记忆 g_t**：记录全局任务及其分解的子任务及其执行状态，使智能体能够跟踪长程任务进度。
- **工作记忆 w_t**：存储短期执行上下文，如当前激活的技能和工具调用历史。

在执行期间，智能体持续检索和更新这个结构化记忆。

给定当前观察和记忆状态，VLM 通过思维链 (CoT) 规划过程执行结构化推理。推理过程首先解释当前场景并识别环境中的相关元素。然后确定当前目标或子任务并评估成功完成的标准。基于此评估，智能体评估当前状态是否满足任务要求或是否需要纠正动作，最后决定要执行的下一个动作。

为了将高层推理与机器人控制桥接，框架通过模型上下文协议 (MCP) 接口提供一组外部工具。这些工具允许智能体启动、终止或切换控制策略，检索环境摘要，查询机器人状态，并在必要时请求人工干预。通过调用这些工具，智能体将 VLM 生成的高层计划转换为可执行动作。

总体而言，智能体在迭代循环中运行：它从结构化记忆和环境观察中检索相关信息，执行基于 CoT 的推理来确定下一个动作，并执行相应的工具调用。结果被写回记忆，形成持续的感知-推理-动作循环，直到任务完成。

### 3.2 通过纠缠动作对实现自重置数据收集

在数据收集期间，RoboClaw 作为数据收集器运行，在由结构化记忆、CoT 规划模块和工具接口组成的闭环中与环境交互（见图 2）。

在时间步 t，智能体接收视觉观察 o_t 并维护一个结构化记忆状态：

$$m_t = (r_t, g_t, w_t) \quad (1)$$

其中 r_t 表示 RoboClaw 的角色身份，g_t 是记录全局任务和子任务进度的任务级记忆，w_t 是存储当前激活的技能和工具调用历史的工作记忆。通过观察和记忆，智能体可以推理当前场景和任务执行状态。

智能体从候选子任务集 Z 中选择下一个子任务 z_t：

$$z_t = \text{RoboClaw}(m_t, o_t), \quad z_t \in \mathcal{Z} \quad (2)$$

智能体评估子任务是否已成功完成，并相应地更新任务记忆 g_t。

RoboClaw 中的低级操作策略使用视觉-语言-动作 (VLA) 模型 π₀.₅ [2] 实现。VLA 策略联合处理视觉观察、语言指令和机器人本体感受状态 (robot proprioceptive states)，生成可执行的机器人动作。

在我们的系统中，语言指令不是由人工操作员直接提供的。相反，它是在 MCP 工具调用期间由 RoboClaw 智能体动态生成的。当 RoboClaw 决定执行技能时，它产生一个描述当前子任务的结构化指令，用于条件化策略。

形式上，策略预测一个短期动作序列：

$$A_t = \pi_{0.5}(o_t, l_t, q_t) \quad (3)$$

其中 o_t 表示视觉观察，l_t 表示 RoboClaw 智能体生成的指令，q_t 表示机器人关节状态。预测的动作块（长度为 H）定义为：

$$A_t = [a_t, \ldots, a_{t+H-1}] \quad (4)$$

策略被训练为使用条件流匹配目标 (conditional flow matching objective) 建模分布 p(A_t | o_t, l_t, q_t)。它学习一个速度场 v_θ，将标准高斯噪声分布传输到真实动作分布。

**纠缠动作对 (Entangled Action Pairs, EAP)**

我们引入纠缠动作对 (EAP) 来实现自主数据收集。对于每个前向操作策略 π^→_θ（例如"将物品放入抽屉"），我们定义一个互补的逆向恢复策略 π^←_θ（例如"从抽屉中取出物品"）。这对策略形成一个纠缠动作对：

$$(\pi^→_θ, π^←_θ) \in \Pi_{EAP} \quad (5)$$

EAP 的关键属性是逆向策略能够将环境恢复到前向策略的可重用前置条件区域。这形成了一个自重置循环：

1. **前向执行**：执行 π^→_θ 完成操作任务
2. **逆向恢复**：执行 π^←_θ 将环境恢复到初始状态
3. **重复**：循环继续，无需人工干预

**数据收集过程**

算法 1 描述了完整的自主数据收集过程：

```
算法 1：通过 EAP 进行自主数据收集
输入：前向策略 π^→_θ，逆向策略 π^←_θ，迭代次数 N
输出：数据集 D

1: 初始化数据集 D = ∅
2: for i = 1 to N do
3:     // 前向执行
4:     获取观察 o_t
5:     生成指令 l_t
6:     执行 A_t = π^→_θ(o_t, l_t, q_t)
7:     记录轨迹 τ^→ = {(o_t, l_t, a_t, r_t)}
8:     D ← D ∪ τ^→
9:     
10:    // 逆向恢复
11:    获取观察 o'_t
12:    生成恢复指令 l'_t
13:    执行 A'_t = π^←_θ(o'_t, l'_t, q'_t)
14:    记录轨迹 τ^← = {(o'_t, l'_t, a'_t, r'_t)}
15:    D ← D ∪ τ^←
16:    
17:    // 验证重置
18:    if 环境未正确重置 then
19:        请求人工干预
20:    end if
21: end for
22: return D
```

### 3.3 长程任务执行与技能编排

在部署阶段，RoboClaw 执行长程任务，需要顺序执行多个子任务。智能体使用与第 3.2 节介绍的相同的闭环决策结构，其中智能体对当前观察 o_t 和结构化记忆 m_t 进行推理，选择下一个子任务 z_t。

给定选定的子任务，RoboClaw 通过 MCP 工具接口从正向策略集 {π^→_{θ_k}}^K_{k=1} 中调用相应的前向策略。

在执行期间，智能体定期查询环境摘要和机器人状态（例如通过 Fetch Robot Stats 和 Env Summary）来监控任务进度。这些反馈信号被写入工作记忆 w_t，用于评估当前子任务是否已完成。

如果子任务的成功条件得到满足，智能体更新任务级记忆 g_t 并继续执行任务计划中的下一个子任务。否则，智能体可能会重试相同的策略或通过 Change Policy 工具切换到另一个前向策略。

如果系统检测到重复失败或意外的环境状态，RoboClaw 尝试通过重新规划并从前向技能集中选择替代技能来进行恢复。当自主恢复不成功或触发安全条件时，智能体通过 MCP 接口的 Call Human 工具升级到人工干预。这种设计允许系统在大多数情况下自主运行，同时为安全关键情况保留人工监督。

重要的是，部署期间生成的轨迹被记录并纳入数据集 D。这些轨迹捕获了真实任务执行期间遇到的额外状态分布，可用于进一步优化技能策略 {π^→_{θ_k}}。通过这种方式，部署不仅执行任务，还作为改进技能库的额外经验来源。

通过在数据收集和部署之间共享相同的决策循环和技能接口，RoboClaw 形成了一个统一的生命周期学习框架，其中执行持续改进底层技能。

---

## 4 实验 (Experiments)

RoboClaw 旨在解决真实世界环境中机器人操作系统面临的四个关键挑战：(1) 提高数据收集效率，(2) 提高子任务策略的成功率，(3) 改善复杂长程任务的性能，(4) 从失败中学习。

为了评估 RoboClaw 的能力，我们设计了一组真实世界操作任务作为我们的实验场景。所有实验都在 Agibot G01 平台上进行，这是一个安装在移动底座上的双臂移动操作机器人。该平台提供 20 个自由度（不包括末端执行器），每只手臂配备一个 AGIBOT OmniPicker 夹爪，这是一种具有单个主动自由度的自适应夹爪。

基于这个实验平台，我们的评估关注以下四个关键问题：

### 4.1 RoboClaw 能否提高数据收集效率？

为了回答问题 1，我们比较了 RoboClaw 与传统数据收集流水线所需的人工投入。传统流水线需要人工操作员在每次演示后重置环境，而 RoboClaw 的 EAP 机制允许自主重置。

**实验设置**：
- 任务：将物品放入抽屉
- 基线：人工重置流水线
- 我们的方法：EAP 自重置

**结果**（图 4）：
- 人工时间投入：RoboClaw 比基线减少 53.7%
- 人工干预次数：RoboClaw 比基线减少 8 倍

### 4.2 逆向重置策略是否可靠？

为了回答问题 2，我们评估了四个操作任务的逆向重置策略的成功率。

**表 2：四个操作任务的逆向重置策略成功率**

| 任务 | 身体乳液 (Body Lotion) | 底漆 (Primer) | 口红 (Lipstick) | 纸巾 (Tissue Wipe) |
|------|------------------------|---------------|-----------------|-------------------|
| 成功率 | 36/50 | 38/50 | 43/50 | 39/50 |

结果表明，逆向策略在所有任务上都达到了合理的成功率（72%-86%），验证了 EAP 机制的有效性。可靠的逆向策略有助于维持稳定的自重置循环，允许机器人在最小人工干预的情况下持续收集轨迹。

### 4.3 RoboClaw 能否更好地处理长程任务？

最后，我们研究了 RoboClaw 在复杂长程任务中的作用。长程任务通常由多个顺序步骤组成，步骤之间存在依赖关系，对规划和执行稳定性提出更高要求。

为了回答问题 3，我们在梳妆台整理任务 (vanity table organization task) 上评估了 RoboClaw 与两个基线的性能。

- **基线 1**：使用在相同数据集上训练的 π₀.₅ 模型，但没有 RoboClaw 框架。
- **基线 2**：将预期成功率估计为四个子任务策略成功率的乘积。

这种比较允许我们分离和分析 RoboClaw 对长程任务性能的贡献。

**RoboClaw 训练流水线包含三个数据来源**：
a) 人工收集的演示数据
b) 自主收集的 RoboClaw 轨迹
c) 自主 rollout 失败后的人工干预

梳妆台整理任务的实验结果如图 4(c) 所示。结果表明，RoboClaw 在长程任务上显著优于两个基线。这种改进来自于 RoboClaw 监控任务进度并在必要时自动调用恢复策略的能力。

图 5 展示了一个代表性的执行序列，包括 RoboClaw 的规划轨迹和梳妆台整理任务期间工具调用的序列。

### 4.4 RoboClaw 能否从失败中学习？

在长程执行期间，RoboClaw 从执行上下文和交互历史中总结常见的失败模式。基于这些观察，我们识别了两类失败：

**非退化失败 (Non-degrading failures)**：指环境状态基本保持不变，失败可以通过重试相同策略来解决的情况。例如，在乳液瓶抓取策略期间，夹爪可能会错过物体或稍微偏离目标关闭，导致空抓。由于瓶子保持直立且其姿态基本不变，智能体可以简单地重试相同的策略，无需额外的恢复动作。

**退化失败 (Degrading failures)**：当失败以阻止立即重试的方式改变环境状态时发生。例如，失败的抓取可能导致乳液瓶翻倒或从初始位置滑开，使其处于抓取策略正常前置条件区域之外。在这种情况下，需要额外的恢复动作来恢复可行状态，然后才能继续任务。

在早期 rollout 阶段，这种退化失败通常需要人工干预来恢复场景。然而，随着 RoboClaw 积累执行经验，这些恢复行为逐渐作为专门的恢复策略纳入策略库。在后续执行中，智能体可以自主调用这些恢复策略来恢复环境并恢复任务，无需人工干预。

这一观察表明，迭代 rollout 不仅提高了现有策略的鲁棒性，还使系统能够通过学习恢复策略来扩展其行为库。随着策略库增长到包括名义策略和恢复行为，RoboClaw 逐步增加了它可以可靠处理的环境状态范围。

---

## 5 结论 (Conclusion)

我们提出了 RoboClaw，一种智能体机器人框架，它在单一的 VLM 驱动智能体循环中统一了数据获取、策略学习和长程任务执行。虽然该框架展示了在统一流水线中集成推理、感知和动作的潜力，但它也面临几个局限性，包括基于云的大模型引入的潜在延迟，以及构建可重用环境状态所需的实际逆向重置行为假设。尽管存在这些挑战，RoboClaw 为可扩展的具身 AI 系统提供了一个有前景的基础。随着 VLM 和 VLA 模型的持续进步，我们相信这种统一的智能体范式将在实现更自主和自适应的机器人系统方面发挥越来越重要的作用。

---

## 附录 A：训练超参数

**表 1：π₀.₅ 微调的训练超参数**

| 通用设置 | | LoRA 设置 | |
|---------|---------|----------|---------|
| 精度 (Precision) | bfloat16 | 秩 (Rank, r) | 16 |
| 批量大小 (Batch size) | 16 | Alpha (α) | 16 |
| 训练步数 (Training steps) | 10k | Dropout | 0.1 |
| 预热步数 (Warmup steps) | 100 | 目标模块 (Target modules) | all-linear |
| 学习率 (Learning rate) | 2.5 × 10⁻⁵ | | |
| 推理步数 (Inference steps) | 3 | | |
| 梯度检查点 (Gradient checkpointing) | ✓ | | |

---

## 附录 B：前向操作策略成功率

**表 3：前向操作策略在 rollout 迭代中的成功率**

| 迭代 | 身体乳液 (Body Lotion) | 底漆 (Primer) | 口红 (Lipstick) | 纸巾 (Tissue Wipe) |
|------|------------------------|---------------|-----------------|-------------------|
| 1 | 21/50 | 23/50 | 2/50 | 11/50 |
| 2 | 25/50 | 31/50 | 8/50 | 15/50 |
| 3 | 30/50 | 35/50 | 15/50 | 20/50 |
| 4 | 35/50 | 40/50 | 22/50 | 26/50 |
| 5 | 38/50 | 43/50 | 28/50 | 30/50 |

结果表明，随着迭代进行，策略成功率持续提高，验证了生命周期学习机制的有效性。

---

## 参考文献 (References)

1. Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., et al.: Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691 (2022)

2. Black, K., Brown, N., Driess, D., Equi, M., Foster, T., Fusai, N., Groom, L., Karamcheti, S., Lattimore, S., Li-Bell, A., et al.: π₀.₅: A vision-language-action flow model. arXiv preprint arXiv:2410.24164 (2024)

3. Black, K., Nakamoto, M., Atreya, P., Walke, H., Schwinghammer, C., Kim, D., Brown, N., Karamcheti, S., Nair, S., Mitchell, M., et al.: π₀: A vision-language-action flow model. arXiv preprint arXiv:2410.24164 (2024)

4. Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., et al.: Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818 (2023)

5. Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al.: Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817 (2023)

6. Chen, Y., Hu, Y., Xu, J., Peng, Y., Zeng, A., et al.: Robotwin 2.0: Scalable robot manipulation data generation with simulation-in-the-loop feedback. arXiv preprint arXiv.2501.02502 (2025)

7. Dong, Q., Li, L., Dai, D., Zheng, C., Wu, Z., Chang, X., Sun, X., Xu, J., Li, L.: A survey on in-context learning. arXiv preprint arXiv:2301.00234 (2023)

8. Driess, D., Xia, F., Sajjadi, M.S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al.: Palm-e: An embodied multimodal language model. arXiv preprint arXiv.2303.03378 (2023)

9. Fu, Z., Zhao, T.Z., Finn, C.: Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. arXiv preprint arXiv:2401.02117 (2024)

10. Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., et al.: Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In: International Conference on Machine Learning. pp. 9118–9127. PMLR (2022)

11. Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., et al.: Voxposer: Composable 3d value maps for robotic manipulation with language models

12. Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., Sermanet, P., Brown, N., Jackson, T., Luu, L., Levine, S., Hausman, K., Ichter, B.: Inner monologue: Embodied reasoning through planning with language models

13. Jing, Z., Yang, S., Ao, J., Xiao, T., Jiang, Y.G., Bai, C.: Humanoidgen: Data generation for bimanual dexterous manipulation via llm reasoning (Nov 2025). https://doi.org/10.48550/arXiv.2507.00833

14. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Sanketi, P., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., Finn, C.: Openvla: An open-source vision-language-action model

15. Li, Y., Deng, Y., Zhang, J., Jang, J., Memmel, M., Yu, R., Garrett, C.R., Ramos, F., Fox, D., Li, A., Gupta, A., Goyal, A.: Hamster: Hierarchical action models for open-world robot manipulation (May 2025). https://doi.org/10.48550/arXiv.2502.05485

16. Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., Zeng, A.: Code as policies: Language model programs for embodied control (May 2023). https://doi.org/10.48550/arXiv.2209.07753

17. Nasiriany, S., Maddukuri, A., Zhang, L., Parikh, A., Lo, A., Joshi, A., Mandlekar, A., Zhu, Y.: Robocasa: Large-scale simulation of everyday tasks for generalist robots (Jun 2024). https://doi.org/10.48550/arXiv.2406.02523

18. Qin, Y., Yang, W., Huang, B., Wyk, K., Su, H., Wang, X., Chao, Y.W., Fox, D.: Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system. In: Robotics: Science and Systems XIX. Robotics: Science and Systems Foundation (Jul 2023). https://doi.org/10.15607/RSS.2023.XIX.015

19. Shah, A., Chen, W., Godbole, A., Mora, F., Seshia, S.A., Levine, S.: Learning affordances at inference-time for vision-language-action models (Oct 2025). https://doi.org/10.48550/arXiv.2510.19752

20. Shi, L.X., Ichter, B., Equi, M., Ke, L., Pertsch, K., Vuong, Q., Tanner, J., Walling, A., Wang, H., Fusai, N., Li-Bell, A., Driess, D., Groom, L., Levine, S., Finn, C.: Hi robot: Open-ended instruction following with hierarchical vision-language-action models (Jul 2025). https://doi.org/10.48550/arXiv.2502.19417

21. Wang, C., Fan, L., Sun, J., Zhang, R., Fei-Fei, L., Xu, D., Zhu, Y., Anandkumar, A.: Mimicplay: Long-horizon imitation learning by watching human play (Oct 2023). https://doi.org/10.48550/arXiv.2302.12422

22. Wang, J., Qin, Y., Kuang, K., Korkmaz, Y., Gurumoorthy, A., Su, H., Wang, X.: Cyberdemo: Augmenting simulated human demonstration for real-world dexterous manipulation (Mar 2024). https://doi.org/10.48550/arXiv.2402.14795

23. Wang, W., Song, J., Liu, C., Ma, J., Feng, S., Wang, J., Jiang, Y., Chen, K., Zhan, S., Wang, Y., et al.: Genie centurion: Accelerating scalable real-world robot training with human rewind-and-refine guidance. arXiv preprint arXiv:2505.18793 (2025)

24. Wang, W., Ye, K., Zhou, X., Chen, T., Min, C., Zhu, Q., Yang, X., Shen, Y., Yang, Y., Yao, M., et al.: Fieldgen: From teleoperated pre-manipulation trajectories to field-guided data generation. arXiv preprint arXiv:2510.20774 (2025)

25. Wang, Z., Chen, J., Chen, Z., Xie, P., Chen, R., Yi, L.: Genh2r: Learning generalizable human-to-robot handover via scalable simulation, demonstration, and imitation. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16362–16372. IEEE, Seattle, WA, USA (Jun 2024). https://doi.org/10.1109/CVPR52733.2024.01548

26. Wu, P., Shentu, Y., Liao, Q., Jin, D., Guo, M., Sreenath, K., Lin, X., Abbeel, P.: Robocopilot: Human-in-the-loop interactive imitation learning for robot manipulation (2025), https://arxiv.org/abs/2503.07771

27. Wu, P., Shentu, Y., Yi, Z., Lin, X., Abbeel, P.: Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators (Jul 2024). https://doi.org/10.48550/arXiv.2309.13037

28. Yang, Z., Chen, Y., Zhou, X., Yan, J., Song, D., Liu, Y., Li, Y., Zhang, Y., Zhou, P., Chen, H., Sun, L.: Agentic robot: A brain-inspired framework for vision-language-action models in embodied agents (Jun 2025). https://doi.org/10.48550/arXiv.2505.23450

29. Zhai, S., Zhang, Q., Zhang, T., Huang, F., Zhang, H., Zhou, M., Zhang, S., Liu, L., Lin, S., Pang, J.: A vision-language-action-critic model for robotic real-world reinforcement learning. arXiv preprint arXiv:2509.15937 (2025)

---

## 附录 C：对本项目的借鉴价值

### 核心借鉴点

| 借鉴点 | 重要性 | 实现难度 | 说明 |
|--------|--------|----------|------|
| **EAP 自重置机制** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 核心创新，显著减少人工干预 |
| **统一智能体架构** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | VLM 作为元控制器统一管理全生命周期 |
| **结构化记忆系统** | ⭐⭐⭐⭐ | ⭐⭐ | 角色身份、任务级记忆、工作记忆三层设计 |
| **技能编排机制** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 动态选择和调度技能，运行时监督 |
| **生命周期学习** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 执行轨迹反馈到训练管道 |

### 实施路线图

**阶段 1：基础架构搭建**
1. 实现 MCP 工具接口
2. 设计结构化记忆系统
3. 集成 VLM 作为元控制器

**阶段 2：EAP 机制实现**
1. 定义前向/逆向策略接口
2. 实现自重置循环逻辑
3. 开发异常检测和人工干预机制

**阶段 3：技能编排系统**
1. 构建技能库
2. 实现动态技能选择
3. 开发状态监控和恢复机制

**阶段 4：生命周期学习**
1. 设计轨迹记录系统
2. 实现数据回流管道
3. 建立持续训练机制

---

*本文档为 RoboClaw 论文的完整中文翻译，专业术语保留英文原文或添加括号注释。*
