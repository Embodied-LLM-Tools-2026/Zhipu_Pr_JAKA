# JAKA 人形机器人智能控制系统 (Zhipu_Pr_JAKA) 深度解析

## 1. 项目概览

本项目是一个集成了**语音交互、视觉感知、大模型规划与机械臂控制**的智能机器人系统。其核心目标是构建一个能够理解自然语言指令、感知环境、自主规划并执行复杂任务（如饮料抓取与递交）的具身智能体。

系统采用**模块化分层架构**，通过 LLM（Large Language Model）和 VLM（Vision Language Model）实现高层认知，结合传统的机器人控制算法实现底层动作，形成了完整的 **Think-Do-Think** 闭环。

---

## 2. 系统架构与数据流

### 2.1 核心架构图

```mermaid
graph TD
    User[用户] <--> Audio[语音模块 (ASR/TTS/VAD)]
    Audio --> Main[主控循环 (VoiceRobotController)]
    
    subgraph "Agent Layer (认知层)"
        VLM[VLM Agent (Qwen3-VL)]
        Planner[Planner Agent (DeepSeek)]
        Engineer[Engineer Agent (Code Gen)]
    end
    
    Main --> Processor[Task Processor]
    Processor <--> VLM
    Processor <--> Planner
    Processor <--> Engineer
    
    subgraph "Control Layer (决策与执行层)"
        Processor --> API[RobotAPI]
        API --> WorldModel[World Model]
        API --> Executor[SkillExecutor]
    end
    
    subgraph "Perception Layer (感知层)"
        Camera[Orbbec Camera] --> Observer[VLM Observer]
        Observer --> Localizer[Target Localizer]
        Localizer --> ZeroGrasp[ZeroGrasp Service]
    end
    
    subgraph "Hardware Layer (硬件层)"
        Executor --> Nav[底盘导航 (TCP/HTTP)]
        Executor --> Arm[JAKA 机械臂 (ROS2)]
        Executor --> Gripper[夹爪 (Modbus)]
    end
    
    Observer --> WorldModel
```

### 2.2 典型任务数据流： "帮我拿一瓶可乐"

1.  **语音输入**: 用户语音 -> `ASR.SenseVoiceRecognizer.recognize` -> 文本 "帮我拿一瓶可乐"。
2.  **意图识别**: 文本 -> `VLM.RobotCommandProcessor.process_command` -> JSON `{intent: "command", action: "get_drink", obj_name: "可乐"}`。
3.  **任务规划**: 目标 "get_drink(可乐)" -> `Planner.BehaviorPlanner.make_plan` -> 生成行为树 (Sequence: Observe -> Navigate -> Grasp -> Handover)。
4.  **执行循环**: `TaskProcessor` 逐个 tick 行为树节点。
    *   **观测**: `PerceptionAPI.observe("可乐")` -> `VLMObserver` 拍照 -> VLM 识别 BBox -> `TargetLocalizer` 映射为 3D 坐标 -> 更新 `WorldModel`。
    *   **导航**: `NavigationAPI.goto_pose` -> `SkillExecutor._skill_navigate_area` -> 底盘移动到最佳抓取位置。
    *   **精确定位**: `ManipulationAPI.execute_skill("finalize_target_pose")` -> 结合深度数据微调底盘角度和位置。
    *   **抓取预测**: `ManipulationAPI.execute_skill("predict_grasp_point")` -> `TargetLocalizer` 调用 ZeroGrasp -> 返回 TCP 6D 位姿。
    *   **抓取执行**: `ManipulationAPI.execute_skill("execute_grasp")` -> `SkillExecutor` 调用 ROS2 IK 服务 -> 机械臂运动 -> 夹爪闭合。
    *   **递交**: `ManipulationAPI.execute_skill("handover_item")` -> 机械臂回到递交姿态 -> 夹爪张开。
5.  **语音反馈**: 每个阶段的状态变化 -> `TTS.TextToSpeechEngine` -> 语音播报 "找到可乐了"、"正在抓取"、"给您可乐"。

---

## 3. 模块详解与核心函数

### 3.1 Agent Layer (认知层)

#### **`voice/agents/VLM.py`**
负责视觉语言模型的交互和任务处理器的核心逻辑。

*   **`RobotCommandProcessor` 类**:
    *   `process_command(text: str) -> Dict`: 核心入口。调用 VLM/LLM 分析用户文本意图，返回结构化 JSON（包含 intent, action, obj_name 等）。
    *   `capture_image(cam_name: str) -> str`: 调用本地 HTTP 接口获取图像并上传到云端，返回 URL。
    *   `web_search(query: str) -> str`: (可选) 调用智谱联网搜索 API 获取实时信息。

*   **`TaskProcessor` 类**:
    *   `__init__`: 初始化各子模块 (Navigator, WorldModel, Observer, Executor, Planner)。
    *   `process_command(text: str)`: 处理自然语言指令，生成并执行计划。
    *   `_execute_plan(plan: CompiledPlan)`: 执行编译后的行为树，维护 `BehaviorTreeRunner` 的 tick 循环。
    *   `_handle_execution_result(result: ExecutionResult)`: 处理节点执行结果，更新历史记录，触发反思。
    *   `_trigger_reflection(...)`: 当任务失败时，调用 `ReflectionAdvisor` 生成诊断建议。

#### **`voice/agents/planner.py`**
负责将高层目标转化为可执行的行为树。

*   **`BehaviorPlanner` 类**:
    *   `make_plan(goal: str, world_model, plan_context) -> CompiledPlan`: 主入口。构建 Prompt，包含可用动作列表、世界模型快照和历史上下文。
    *   `_request_plan(...) -> Dict`: 调用 DeepSeek API 生成 JSON 格式的行为树结构。
    *   `_validate_plan(plan_dict)`: 校验生成的行为树是否符合 schema（节点类型、动作名称是否合法）。
    *   `_compile_plan(root, world_model) -> List[PlanNode]`: 将树结构扁平化为执行步骤序列（用于 UI 显示）。

*   **`ReflectionAdvisor` 类**:
    *   `reflect(goal, plan_entry, execution_history) -> Dict`: 分析失败的执行轨迹，生成原因诊断 (`diagnosis`) 和改进建议 (`adjustment_hint`)。

#### **`voice/agents/engineer.py`**
负责动态生成缺失的动作代码。

*   **`EngineerAgent` 类**:
    *   `process_ticket(ticket: ActionTicket) -> ActionEntry`: 处理动作需求单。
    *   `_generate_action_code(ticket, name) -> str`: 构建 Prompt 让 LLM 生成 Python 代码，要求包含 `run(api, runtime, **kwargs)` 函数。
    *   `_write_action_file(...)`: 将生成的代码写入 `actions/` 目录。
    *   `_register_placeholder(...)`: 如果未配置 LLM，则生成一个占位文件。

---

### 3.2 Control Layer (决策与执行层)

#### **`voice/control/apis.py`**
统一的 RobotAPI 接口层，屏蔽底层实现。

*   **`NavigationAPI`**:
    *   `goto_marker(marker)`: 导航到预定义点。
    *   `goto_pose(x, y, theta)`: 导航到具体坐标。
    *   `wait_until_idle()`: 阻塞直到导航结束。
*   **`PerceptionAPI`**:
    *   `observe(target, ...)`: 触发观测流程，返回观测结果。
*   **`ManipulationAPI`**:
    *   `execute_skill(name, args)`: 调用 `SkillExecutor` 的具体技能。
    *   `move_tcp(pose)`: 机械臂运动 (IK + PTP)。
*   **`GripperAPI`**:
    *   `open()`, `close()`: 夹爪控制。

#### **`voice/control/executor.py`**
核心执行器，包含所有原子技能的实现。

*   **`SkillExecutor` 类**:
    *   `execute(node, runtime) -> ExecutionResult`: 统一执行入口，根据 node.name 分发到具体 `_skill_` 方法。
    *   `_skill_rotate_scan(args, runtime)`: 控制底盘旋转指定角度。
    *   `_skill_search_area(args, runtime)`: 多次旋转扫描。
    *   `_skill_navigate_area(args, runtime)`: 导航到区域或坐标。
    *   `_skill_approach_far(args, runtime)`: 当距离目标较远 (>2m) 时，粗略靠近。
    *   `_skill_finalize_target_pose(args, runtime)`: **关键函数**。结合深度数据，计算机器人相对于物体的最佳抓取站位（距离、角度），并控制底盘移动到位。
    *   `_skill_predict_grasp_point(args, runtime)`: **关键函数**。获取对齐的 RGB-D 数据，调用 `TargetLocalizer` 和 ZeroGrasp，计算最佳抓取位姿 (TCP Pose)。
    *   `_skill_execute_grasp(args, runtime)`: **关键函数**。解析预测的 TCP Pose，调用 `_ArmIKClient` 进行逆运动学解算，控制机械臂运动并闭合夹爪。
    *   `_skill_handover_item(args, runtime)`: 执行递交动作序列。

*   **`_ArmIKClient` 类**:
    *   `solve_ik(pose, ref_joints)`: 调用 ROS2 `/jaka_driver/get_ik` 服务计算关节角。
    *   `execute_joint_move(joints)`: 调用 ROS2 `/jaka_driver/joint_move` 服务执行运动。

#### **`voice/control/world_model.py`**
环境状态管理。

*   **`WorldModel` 类**:
    *   `update_from_observation(target_id, observation)`: 根据观测结果更新物体状态（可见性、位置、置信度）。
    *   `snapshot() -> Dict`: 生成当前状态的快照，供 Planner 使用。

---

### 3.3 Perception Layer (感知层)

#### **`voice/perception/observer.py`**
视觉观测流程控制器。

*   **`VLMObserver` 类**:
    *   `observe(target_name, ...)`: 主流程。
        1.  `_capture_image`: 获取图像。
        2.  `_build_tracker_observation`: (可选) 尝试使用 CSRT 跟踪器复用上一帧结果。
        3.  `_call_vlm`: 如果跟踪失败或强制刷新，上传图片调用 VLM，Prompt 包含 "Find {target_name}"。
        4.  `_parse_response`: 解析 VLM 返回的 BBox 和描述。
        5.  `_push_detection_to_frontend`: 推送结果到 UI。

#### **`voice/perception/localize_target.py`**
3D 定位与坐标转换核心。

*   **`TargetLocalizer` 类**:
    *   `localize_object(bbox, snapshot, ...)`: 结合 2D BBox 和深度图，计算物体在相机坐标系下的 3D 中心点。
    *   `run_zero_grasp_inference(...)`: 调用 ZeroGrasp 服务预测抓取点。
*   **辅助函数**:
    *   `fetch_aligned_rgbd()`: 从 API 获取同步对齐的 RGB 和 Depth 帧。
    *   `_deproject_pixel_to_point(pixel, depth, intrinsics)`: 像素坐标转相机坐标 (针孔相机模型)。
    *   `transform_camera_to_robot(point)`: 相机坐标系 -> 机器人基座坐标系转换（硬编码的外参矩阵）。

---

### 3.4 Audio & Main Layer (交互与主控)

#### **`main_hand.py`**
程序入口与主控循环。

*   **`VoiceRobotController` 类**:
    *   `run()`: 主循环。监听语音输入 -> VAD 检测 -> ASR 识别 -> `_process_voice_command`。
    *   `_process_voice_command(text)`: 调用 `VLM.process_command` 获取意图，如果是指令则交给 `TaskProcessor`，如果是聊天则直接 TTS 回复。
    *   `_speak(text)`: 调用 TTS 引擎播放语音。

#### **`voice/audio/`**
*   **`ASR.py`**: `SenseVoiceRecognizer` 封装 FunASR SenseVoiceSmall 模型。
*   **`TTS.py`**: `TextToSpeechEngine` 封装 Edge-TTS 和 Kokoro，支持流式生成。
*   **`VAD.py`**: `SileroVAD` 封装 Silero VAD 模型，处理音频分帧与人声检测。

---

## 4. 关键技术细节

### 4.1 坐标系转换链
系统涉及多个坐标系的转换，这是抓取成功的关键：
1.  **Pixel (u, v)**: 图像像素坐标。
2.  **Camera (Xc, Yc, Zc)**: 通过内参矩阵反投影得到。
    *   $Z_c = depth[v, u]$
    *   $X_c = (u - c_x) * Z_c / f_x$
    *   $Y_c = (v - c_y) * Z_c / f_y$
3.  **Robot Base (Xr, Yr, Zr)**: 通过手眼标定外参矩阵转换。
    *   代码中硬编码了变换逻辑 (`transform_camera_to_robot`)：
    *   $X_r = Y_c + 50.0$
    *   $Y_r = -X_c + 180.0$
    *   $Z_r = Z_c$
4.  **World (Xw, Yw, Zw)**: 结合底盘里程计 (Odometry) 转换。
    *   $X_w = X_{odom} + (Z_r \cos\theta - Y_r \sin\theta)$
    *   $Y_w = Y_{odom} + (Z_r \sin\theta + Y_r \cos\theta)$

### 4.2 抓取策略
*   **粗定位**: `approach_far` 技能只利用 2D 视觉或粗略深度，将机器人移动到距离物体约 1-2 米处。
*   **精定位**: `finalize_target_pose` 利用深度数据计算物体精确位置，调整底盘使得物体位于机械臂最佳工作空间（通常是右前方）。
*   **抓取检测**: ZeroGrasp 返回的是 6D 位姿（位置 + 旋转矩阵）。
*   **执行**: `execute_grasp` 会先移动到 `Pre-Grasp` 点（抓取点沿 Approach 轴后退 15cm），然后直线插补进给，闭合夹爪，再抬起。

### 4.3 提示词工程 (Prompt Engineering)
*   **VLM Prompt**: 包含图像和文本，要求输出 JSON，明确区分 "chat" 和 "command" 意图，并提取参数。
*   **Planner Prompt**: 定义了行为树的 Schema（允许的节点类型），提供了可用动作列表 (`allowed_actions`) 及其详细文档，以及当前世界模型的快照。要求 LLM 输出符合语法的 JSON 行为树。
*   **Engineer Prompt**: 提供了 RobotAPI 的接口文档，要求生成符合特定签名的 Python 代码，并包含错误处理。

---

## 5. 总结

Zhipu_Pr_JAKA 通过精细的模块划分，将大模型的泛化能力与传统机器人的精确控制能力结合。
*   **VLM** 解决了“看到什么”和“用户想要什么”的问题。
*   **Planner** 解决了“怎么做”的逻辑编排问题。
*   **Executor & API** 解决了“如何动”的物理实现问题。
*   **Engineer** 解决了“能力不足时怎么办”的进化问题。

这种架构具有极强的扩展性，未来可以通过增加新的 Skill 和 API，轻松支持更多种类的任务。
