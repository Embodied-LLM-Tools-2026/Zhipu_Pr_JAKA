# Engineer Collaboration Log（可作为重启对话时的上下文）

> 本文件用于向未来的对话描述当前工程状态、关键信息和下一步计划。任何新的工作、环境约束或想法都应追加到这里，保证重新进入会话后能够无缝衔接。

---

## 0. 当前工作区/环境

- 仓库路径：`/home/sht/DIJA/Pr`
- 主要目录：
  - `voice/`：核心运行时代码，已拆分为 `voice/agents`（Planner/Engineer/VLM）、`voice/control`（SkillExecutor/RobotAPI/WorldModel）、`voice/perception`（Observer/深度定位）、`voice/audio`（ASR/TTS/VAD/音频工具）、`voice/utils`（配置/依赖）。
  - `action_sequence/`：底盘、夹爪等低层控制
  - `actions/`（新增）：动态加载的高阶 Action 模块
  - `docs/`：文档（包括本文件、planner_engineer.md、engineer_reference.md）
  - `tools/`：工具集合，已拆分为 `tools/camera`、`tools/vision`、`tools/logging`、`tools/ui`
  - `examples/`：原 `test/` 中的示例/调试脚本统一迁移于此
  - `tests/`：可执行的 pytest/hardware 用例（当前包含 `tests/hardware`）
- 关键依赖：DashScope（Qwen3-VL）、DeepSeek Chat、ZeroGrasp、Orbbec 深度相机、JAKA ROS2 驱动、Modbus 夹爪
- 推荐环境变量：
  - `ENABLE_DYNAMIC_ACTIONS=1`
  - `ENABLE_CODE_ENGINEER=1`
  - `DEEPSEEK_ENGINEER_API_KEY=...`
  - `DEEPSEEK_API_BASE=https://api.deepseek.com`
  - `ACTION_LIBRARY_PATH` / `ACTION_TEST_DIR`（可选）
  - `TASK_PROCESSOR_MODE`（`behavior_tree` 或 `function_call`，用于切换旧行为树与新 Function Call 链路）
  - 其它参见 `docs/planner_engineer.md`

---

## 1. 已完成的主要工作

### 1.1 基础框架

- **TaskProcessor**：整合 VLM 观测、行为树规划（DeepSeek）、SkillExecutor 技能、世界模型记忆，以及 UI/日志输出（现位于 `voice/agents/VLM.py`）。
- **FunctionCallTaskProcessor**：全新的工具调用链路（`voice/agents/function_call/processor.py`），通过 LLM Function Call 直接调度 `RobotAPI` 原语，可与旧行为树模式并存，通过 `TASK_PROCESSOR_MODE` 选择。
- **RobotAPI**（`voice/control/apis.py`）：统一封装导航、观测、操作、夹爪、规划五类 API，便于 Planner、Engineer、LLM 直接调用。
- **SkillExecutor**：实现现有技能（导航、观测、靠近、抓取、递交、recover 等），当前实现是“任务级动作”而非真正的机械臂/传感器原语（代码迁入 `voice/control/executor.py`）。
- **WorldModel**：管理目标/机器人状态，供 Planner 和行为树使用。
- **VLMObserver、SceneCatalog**：负责图像捕获、VLM 调用、深度融合，并持续更新世界模型。
- Function-call 链路已支持自定义观测请求：`observe_scene` 可通过 `query` 提示 VLM 返回特定关系/状态描述（输出到 `analysis` 字段），功能与行为树共用同一个 Observer。

### 1.2 Planner ↔ Engineer 工作流

- **ActionRegistry**（`voice/action_registry.py`）：
  - 管理 `ActionTicket`（需求单）、`ActionEntry`（高阶动作）、`PrimitiveEntry`（原语描述）
  - 提供 `create_ticket/list/register/unregister` 等接口
- **EngineerAgent**（`voice/engineer.py`）：
  - 支持两种模式：占位注册或 DeepSeek 生成真实 action
  - 自动生成 `actions/<name>.py` + `tests/actions/test_<name>.py` 模板
  - 调用 DeepSeek Chat，内含 API surface 和模板指令
- **DynamicActionRunner**（`voice/dynamic_actions.py`）：
  - 根据 registry 或 `actions/<name>.py` 动态 import `run(api, runtime, **kwargs)`
  - 行为树执行时若 skill 不存在会尝试动态 action，失败再触发 ticket
- **TaskProcessor** 增强：
  - 自动创建需求单 `_request_action_ticket`（当 SkillExecutor 返回 `unsupported_skill`）
  - 暴露 `request_action/list_action_tickets/list_registered_actions/list_primitives`
  - 可选启用 `ENABLE_DYNAMIC_ACTIONS` 与 `ENABLE_CODE_ENGINEER`

### 1.3 文档/知识注入

- **docs/planner_engineer.md**：描述 Planner ↔ Engineer 流程、开关、模板示例、测试要求等。
- **docs/engineer_reference.md**：Engineer 包，含系统概览、RobotAPI 细节、动作模板、接口说明、安全注意事项；可作为喂给 DeepSeek 的上下文。

---

## 2. 当前技能（非真正原语） & 覆盖范围

> 现阶段 SkillExecutor 暴露的是“任务级技能”，虽然对 Planner 来说像原语，但内部仍然把底层运动/传感逻辑封装在黑盒里，尚未拆解为 TCP 位姿/力控等真正的最低层接口。

| 类别 | 技能举例 | 覆盖任务 |
| --- | --- | --- |
| 底盘/导航 | `rotate_scan`, `search_area`, `navigate_area`, `return_home`, `approach_far`, `recover` | 原地旋转、扫描、导航到 marker/pose、后退恢复 |
| 观测/感知 | `perception.observe`, `WorldModel.update_from_observation` | VLM + 深度观测、世界模型更新 |
| 抓取 | `finalize_target_pose`, `predict_grasp_point`, `execute_grasp` | ZeroGrasp 抓取流程 |
| 夹爪 | `open_gripper`, `close_gripper`, `handover_item` | 开合/递交 |
| 规划/反思 | `BehaviorPlanner.make_plan`, `ReflectionAdvisor.reflect` | DeepSeek 行为树规划 + 失败诊断 |

> 目前能完成“定位杯子→抓取→递交”的标准流程，但缺少通用的机械臂 TCP/Joints 控制、路径跟随、力控、夹爪姿态等原语，无法直接执行“倾倒水”“擦桌子”“开抽屉”等复杂操作，需要在 SkillExecutor/RobotAPI 中补充更细粒度的原子能力。

---

## 3. 近期讨论与结论

1. **Think-Do-Think 实现**：已经用 `plan_context + execution_history + reflection_log` 构建了“计划→执行→反思”闭环。但 LLM 尚未在动作失败后进行“专项诊断”，下一步可添加 Reflection LLM（已经预留）并将诊断推送到对话/UI。

2. **动态 Action/Engineer 流程**：
   - 当 SkillExecutor 缺少某个技能时，会创建 ticket 并（可选）交给 Engineer。
   - 启用 DeepSeek 后，EngineerAgent 会生成 `actions/<name>.py`（含模板注释）和简易测试；测试仍需人工或 CI 执行，当前代码未自动跑 pytest。
   - DynamicActionRunner 会在行为树执行时加载这些动作。

3. **原语 vs 任务特化函数**：
   - 决定优先构建“有限且通用”的原语（如 `move_tcp`, `set_gripper`, `observe` 等），所有高阶动作都从这些积木组合，不针对单一任务写专用接口。
   - Engineer 包已记录现有原语；后续在补充新的运动/感知能力时，也应进入原语集合。

4. **做更复杂任务（例如倒水、开抽屉）**：
   - 需要新增机械臂原语（TCP/Joints 控制、路径跟随、Apply force）、感知原语（抽屉开度、触觉/力反馈），以及稳健的安全守卫。
   - 考虑将这些拆分为底层 API，通过 RobotAPI 暴露，LLM 再组合出行动。

---

## 4. TODO / 下一步建议

1. **完善原语集合**  
   - 数据：增加 `capture_depth()`, `get_contact_state()`, `read_sensor(sensor_id)` 等；  
   - 行为：实现 `move_tcp`, `move_joint`, `shift_tcp`, `follow_path`, `apply_force`, `set_gripper(position/force)` 等通用接口；  
   - 安全：加入速度/范围限制、碰撞/力控回调，封装为 RobotAPI 原语。

2. **示例 Action & 测试**  
   - 手写至少一个 `actions/demo_*` 示例，验证 DynamicActionRunner 全链路；  
   - 在 CI 或脚本中运行 `pytest tests/actions`，确保生成的 action 被验证。

3. **Engineer Prompt 加强**  
   - 在 `EngineerAgent._generate_action_code` 中注入 `docs/engineer_reference.md` 内容（API 说明、模板示例、硬件注意事项）；  
   - 提供更多真实代码片段/日志，使 DeepSeek 熟悉系统风格。

4. **UI/CLI 支持**  
   - 开发命令或网页查看 `ActionRegistry`、pending tickets、执行历史；  
   - 提供 `cli request-action --name foo --desc ...` 之类的工具，方便人工操作。

5. **Planner 反思的对话反馈**  
   - 将 `reflection` 字段传回对话系统，让机器人能向用户解释失败原因；  
   - 在 Planner prompt 中利用 `adjustment_hint`，减少重复错误。

6. **知识库更新**  
   - 持续完善 `docs/engineer_reference.md`，包括更详细的硬件协议、坐标系、动作示例、常见坑；  
   - 一旦有新的原语或传感器，要在文档中记录，确保 LLM 始终看到最新信息。

---

## 5. 如何重启工作

1. 阅读本 `summary.md` 和 `docs/planner_engineer.md`、`docs/engineer_reference.md`，了解当前状态与接口。
2. 检查 `git status`，确认有哪些未提交文件（尤其是 `actions/` 内容）。
3. 根据 TODO 列表选择任务，例如：
   - “编写 move_tcp 原语并暴露到 RobotAPI”；
   - “实现第一个真实 action（如 demo_handover），并运行测试”；
   - “改善 Engineer prompt，注入参考文档”。
4. 完成后更新此 `summary.md`，以便下次继续协作。

> 记得在任何新动作/原语实现后，更新 ActionRegistry 和相关文档，并确保可选开关（`ENABLE_DYNAMIC_ACTIONS`/`ENABLE_CODE_ENGINEER`）配置正确。

---

## 6. 原语拆解方案（建议路线）

1. **抽离基础机械臂接口**  
   - `move_joint(joint_positions, speed, acc)`：直接发送关节角命令，可用于校准、抬升、避障。  
   - `move_tcp_linear(pose, speed, acc)`：直线插补到绝对 TCP 姿态。  
   - `shift_tcp(delta_xyz, delta_rpy, reference="tool"|"world")`：做小范围相对平移/姿态微调。  
   - `follow_tcp_path(waypoints, mode="linear"|"blend")`：执行擦拭、划圈等多段路径。  
   - `set_compliance(frame, stiffness, damping)` / `apply_force(axis, force, duration)`：支持推/拉/按压等力控动作。

2. **拆分抓取流程**  
   - `estimate_grasp_pose(observation)`：负责感知推理，只返回姿态/分值，不做运动。  
   - `move_to_pregrasp(grasp_pose, offset_mm)`：根据生成的姿态移动到抓取前安全点。  
   - `descend_to_grasp(grasp_pose, approach_mm)`：沿抓取方向平移接近。  
   - `execute_retreat(retreat_vector, speed)`：抓取后撤离，用于搬运或过渡。

3. **夹爪/末端原语**  
   - `set_gripper(width_mm, force)`：精确控制开口和力度，支持夹持不同物体。  
   - `rotate_wrist(angle_deg)` 或 `set_wrist_pose(rpy)`：支持旋钮、倒水等需要末端旋转的动作。  
   - `read_gripper_state()`：返回当前开度、电流，供力反馈/碰撞判断。

4. **感知/环境反馈**  
   - `capture_depth_patch(bbox)`、`get_surface_normal(point)`：为擦桌子等任务提供表面参数。  
   - `detect_contact(axis)`：通过力矩或夹爪电流判断接触，用于擦拭/关阀门。  
   - `update_world_object(id, pose/attributes)`：低层操作后同步世界模型。

5. **集成步骤**  
   - 在 `SkillExecutor` 中新增上述 `_primitive_*` 方法，并通过 `RobotAPI.manipulation` 暴露。  
   - 让现有 `_skill_execute_grasp`、`handover_item` 等高阶动作改为调用这些 primitive 的组合；Planner/动态 action 也只能使用公开的 primitive API。  
   - ActionRegistry 的 `PrimitiveEntry` 用于记录这些接口（signature/安全说明），Engineer prompt 注入最新列表，确保 LLM 了解可用积木。  
   - 增加最小化示例（如 `actions/demo_wipe_table`）演示如何用 `move_tcp_linear + shift_tcp + set_gripper + apply_force` 拼出新任务，同时编写 pytest 覆盖 API 拼装顺序。

---

（本文件可直接作为下一次会话的系统 prompt，让模型快速了解当前进度和上下文。）
