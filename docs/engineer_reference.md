# Engineer 参考包

面向 `EngineerAgent` 或其它 LLM 工程师，帮助其在不了解仓库的情况下编写可靠的 Action/Primitive。内容覆盖硬件、原语接口、服务端点、示例模板与安全注意事项。

---

## 1. 系统/硬件概览

| 模块 | 说明 |
| --- | --- |
| 底盘导航 | 通过 `action_sequence/navigate.py` 提供的 TCP `/api/move` 接口控制。可根据 marker（`/api/move?marker={target}`）或明确坐标（`/api/move?location=x,y,theta`）移动，`wait_until_navigation_complete` 会轮询 `move_status` 直到 `succeeded/failed`。 |
| 机械臂 (JAKA) | 使用 ROS2 服务 `/jaka_driver/get_ik`、`/jaka_driver/joint_move` 完成 IK 计算与运动执行，`SkillExecutor._skill_execute_grasp` 封装了流程。线性轴通过 `ros2 service call /jaka_driver/linear_move jaka_msgs/srv/Move { ... }` 调整前后位置。 |
| 夹爪 | `action_sequence/gripper_controller.py` 通过 Modbus 指令（寄存器 `0x0029` 开、`0x0028` 闭、`0x000A` 位置、`0x000B` 力度）控制。封装在 `GripperAPI` / `_skill_open_gripper` / `_skill_close_gripper`。 |
| 摄像头/VLM | RGB 画面由 UI REST 接口 `http://127.0.0.1:8000/api/capture?cam=front` 提供，返回 `{"url": "/path/to/image.jpg", "w":960,"h":540}`。`VLMObserver.observe` 会上传图像到 DashScope Qwen-3-VL 模型并解析观测。 |
| 深度/抓取识别 | `voice/localize_target.py` 使用奥比深度摄像头和 ZeroGrasp 服务估计 3D 坐标 (`camera_center` mm)，再转换到机器人/世界坐标。ZeroGrasp WebSocket 地址由 `Config.ZERO_GRASP_WS_URL` 指定。 |
| 世界模型 | `voice/world_model.py` 维护 `areas`、`objects`, `objects.{id}` 包含 `visible`, `attrs.range_estimate`, `camera_center`, `robot_center(mm)`, `world_center(m)`。 |

坐标转换：camera(mm) → robot(mm) (`SkillExecutor.transform_camera_to_robot`) → world(m) (`transform_robot_to_world` using底盘 `pose[x,y,theta]`)。`WorldModel.update_from_observation` 会自动记录 range/pose。

---

## 2. RobotAPI 原语

位于 `voice/apis.py`，由 TaskProcessor 注入 `RobotAPI` 实例 `api` 提供。

### NavigationAPI
| 方法 | 描述 |
| --- | --- |
| `goto_marker(marker: str) -> bool` | 通过 marker 移动到底盘预设位置。 |
| `goto_pose(theta,x,y,timeout=60) -> bool` | 直接移动到指定 pose (单位: θ 弧度, x/y 米)。 |
| `wait_until_idle(timeout=60)` | 阻塞直到当前导航完成。 |
| `current_pose() -> {"theta","x","y"}` | 返回最新 pose。 |
| `navigation_state()` | 返回 `{"move_status","is_navigating","last_target"}`。 |

### PerceptionAPI
- `observe(target, phase=ObservationPhase.SEARCH, force_vlm=False, max_steps=1)`：触发 VLM 观测并融合深度，返回 `(ObservationResult, payload)`。`ObservationResult` 字段包括 `found`, `bbox`, `confidence`, `range_estimate`, `camera_center(mm)`, `robot_center(mm)`, `world_center(m)`。
- `get_object_state(object_id)` 读取世界模型。

### ManipulationAPI
- `execute_skill(name, args=None, observation=None, extra=None)`：调用 `SkillExecutor._skill_*`（或动态 action）。常用技能：
  - `rotate_scan(angle_deg)`：原地旋转搜索。
  - `search_area(turns, angle_deg)`：多次旋转扫描。
  - `navigate_area(area/marker/pose)`：使用导航器移动到指定区域。
  - `approach_far(target)`：沿目标方向大步前进（>2m）。
  - `finalize_target_pose(target)`：结合深度数据精精准定位底盘姿态。
  - `predict_grasp_point(target)`：调用 ZeroGrasp 预测抓取姿态。
  - `execute_grasp`：执行机械臂抓取。
  - `open_gripper / close_gripper / handover_item`：操作夹爪。
  - `recover(distance)`：后退/重置状态。

### GripperAPI
- `open()` / `close()` / `deliver(item=None)`：通过串口控制夹爪。

### PlanningAPI
- `plan(goal, history)` 调用 DeepSeek 行为树规划器。
- `reflect(plan_entry, execution_turns)` 触发 `ReflectionAdvisor` 生成失败诊断/提示。

---

## 3. 动作模板示例

在 `actions/demo_handover.py`（示例）中推荐如下结构：

```python
from voice.task_structures import ExecutionResult
from task_logger import log_info, log_warning

def run(api, runtime, target="可乐", area="desk", **kwargs):
    # 导航到指定区域
    if not api.navigation.goto_marker(area):
        return ExecutionResult(status="failure", node="demo_handover", reason="navigate_failed")

    # 观测
    observation, _ = api.perception.observe(target, force_vlm=True)
    if not observation.found:
        log_warning(f"未找到目标 {target}")
        return ExecutionResult(status="failure", node="demo_handover", reason="observe_fail")

    # 调用现有技能完成抓取
    result = api.manipulation.execute_skill("predict_grasp_point", observation=observation)
    if not result.success:
        return result
    result = api.manipulation.execute_skill("execute_grasp", observation=observation)
    if not result.success:
        return result

    # 递交
    api.manipulation.execute_skill("handover_item", observation=observation, extra={"requested_item": target})
    api.manipulation.execute_skill("open_gripper", observation=observation)
    return ExecutionResult(status="success", node="demo_handover")
```

对应的测试样例（`tests/actions/test_demo_handover.py`）：

```python
import importlib

def test_module_import():
    module = importlib.import_module("actions.demo_handover")
    assert hasattr(module, "run")
```

Engineer 生成的代码需遵循此风格，并可在测试中引入更多仿真/Mock（例如假 Navigator/Perception）。

---

## 4. 服务/接口细节

### 摄像头/VLM
- 捕获接口：`GET http://127.0.0.1:8000/api/capture?cam=front&w=960&h=540` → `{url,w,h}`，`observer._capture_image` 已封装。
- VLM 模型：DashScope `qwen3-vl-plus`，prompt 见 `voice/observer.py._build_prompt`；支持 `force_vlm` 参数强制重新分析。
- Depth Snapshot：`localize_target.fetch_snapshot()` 读取 Orbbec SDK 缓存的深度帧，并附带内参/外参给 ZeroGrasp。

### ZeroGrasp 调用
- `TargetLocalizer._zero_grasp_predict` 发送请求到 `Config.ZERO_GRASP_WS_URL`，payload 包括 `camera_params`, `intrinsics`, `rgb image url`, `mask` 等。

### 导航/底盘
- `Navigate.navigate_to_target`：通过 TCP 发送 `/api/move?marker={target}`，然后调用 `wait_until_navigation_complete`；`Navigate.move_to_position` 类似，只是 `location=x,y,theta`。
- 状态通过 socket `/api/robot_status` 持续读取，`navigation_state.move_status` 可能为 `running/succeeded/failed`。

### 机械臂
- IK：`/jaka_driver/get_ik` 接受 TCP 位姿 `[x,y,z,rx,ry,rz]`，返回关节角。
- 执行：`/jaka_driver/joint_move` 包含 `joint_angle`, `speed`, `acc`, `mvtime`, `coord_mode` 等字段。
- 线性轴范围：`robot_x_mm` 在 `(-310, 200)` 内时自动对齐，否则跳过。

### 夹爪
- Modbus 连接参数：默认 `baud=115200`, `parity='N'`, `stopbits=1`，`open` 写 `0x0029=1`，`close` 写 `0x0028=1`，`set_force` 写 `0x000B`。

### 世界模型/任务记忆
- `WorldModel.snapshot()` 返回 `{"goal": "...", "objects": {...}, "robot": {"pose":[x,y,theta]}}`，Planner 会把最近 plan/execution 摘要存入 `plan_context`（`voice/VLM.py._append_plan_context`）。

---

## 5. Action Registry / 动态加载

- `ActionRegistry` (`voice/action_registry.py`) 管理三类实体：
  - `ActionTicket`: Planner 需求单（id/goal/description/inputs/outputs/constraints/examples）。
  - `ActionEntry`: 已注册动作（name/description/inputs/outputs/code_path/tests/version/author）。
  - `PrimitiveEntry`: 底层原语描述（name/description/api_signature/module/safety_notes）。
- `DynamicActionRunner` (`voice/dynamic_actions.py`) 会根据 ActionEntry `code_path` 或默认 `actions.{name}` 动态 import 并执行 `run(api, runtime, **kwargs)`。
- 动作返回值需是 `ExecutionResult` 或 `{"status","reason","evidence"}` 字典。
- `TaskProcessor.request_action()`/`list_action_tickets()`/`list_registered_actions()`/`list_primitives()` 提供外部访问接口。

---

## 6. 示例业务流程

**拿饮料**（Simplified）
1. `observe_scene(force_vlm=True)` → `search_area` (if not found) → `navigate_area` 到货架区域。
2. `observe_scene(force_vlm=True)` → `finalize_target_pose` → `predict_grasp_point` → `execute_grasp`。
3. `navigate_area` 到用户位置 → `handover_item(requested_item)` → `open_gripper`。
4. `return_home`（可选）。

Action 需要遵循该流程，利用现有 skill/原语组合；若发现缺口，可创建 ticket 让 Engineer 生成新的 skill 或 primitive。

---

## 7. 安全注意事项

1. **控制指令**：仅通过 RobotAPI/封装的 skill 调用底层硬件，禁止直接写串口或发送未授权命令。
2. **坐标/单位**：导航采用米 (m)、θ 为弧度；机械臂/摄像头转换使用毫米 (mm)。务必确认单位一致。
3. **夹爪限制**：位置 [1..100]、力度 [20..320]，调用前确保在范围内。
4. **异常处理**：所有动作出错必须返回 `ExecutionResult(status="failure", reason=...)`，不要忽略异常。
5. **网络依赖**：VLM/ZeroGrasp 调用可能超时，需提供重试/兜底逻辑；无法访问时请返回失败而非阻塞。
6. **调试输出**：使用 `task_logger.log_info/log_warning/log_error`，避免直接 `print`。
7. **测试与审核**：任何自动生成的 action/primitive 在实际部署前需要人工 review + 测试（可扩展 `tests/actions/` 目录）。

---

## 8. 提示：如何扩展

1. **添加动作**：创建 `actions/new_action.py`，仿照模板实现 `run`，编写 `tests/actions/test_new_action.py`，运行测试后 `ActionRegistry.register_action`。
2. **添加原语**：在 `voice/executor.py`/`voice/apis.py` 增加方法，并在 `ActionRegistry.register_primitive` 记录描述；公开到 RobotAPI 供 LLM 使用。
3. **Planner↔Engineer 流程**：Planner 发现缺失 → `_request_action_ticket` → Engineer 生成代码+注册 → `DynamicActionRunner` 执行 → 成功后注册可复用。

---
