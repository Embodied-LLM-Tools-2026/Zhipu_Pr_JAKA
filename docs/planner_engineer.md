# Planner ↔ Engineer 工作流说明

## 角色

- **Planner**：行为树规划器或 TaskProcessor 的计划模块。当发现某个动作缺失（例如 `SkillExecutor` 返回 `unsupported_skill`）时，会调用 `TaskProcessor.request_action(...)` 或内部 `_request_action_ticket`，在注册表中生成需求单。
- **Engineer**：代码代理（`voice/engineer.py:EngineerAgent`），接收需求单并登记占位的 ActionEntry。后续可接入真实 LLM 或人工流程来补完实现与测试。
- **Action Registry**：`voice/action_registry.py` 提供的内存注册表，管理 `tickets`、`actions`、`primitives` 三类信息。

## 启用方式

- `ENABLE_DYNAMIC_ACTIONS=1`：开启动态 action 加载器，TaskProcessor 会尝试在 `actions/` 目录和 registry 中加载高阶动作。
- `ENABLE_CODE_ENGINEER=1`：启用 `EngineerAgent` 占位流程，在需求单创建后自动登记 `ActionEntry`。
- 默认情况下两个开关都关闭，系统只维持 think→do→think 流程，不会加载或生成自定义动作。
- 若希望 Engineer 自动写代码，请确保再配置 `DEEPSEEK_ENGINEER_API_KEY`（或复用 `DEEPSEEK_API_KEY`）。可选 `DEEPSEEK_ENGINEER_MODEL` 指定模型，`ACTION_LIBRARY_PATH`/`ACTION_TEST_DIR` 自定义输出目录。

## 创建需求单

1. Planner 检测到动作缺失：
   - 运行行为树时 `SkillExecutor` 返回 `unsupported_skill` → 自动调用 `_request_action_ticket`；
   - 或者外部模块手动调用 `TaskProcessor.request_action(...)` 传入名称/描述/IO。
2. `ActionRegistry.create_ticket` 生成 `ActionTicket(ticket_id, goal, description, inputs, outputs, constraints)`，并写入 `TaskProcessor.list_action_tickets()` 可见的待处理列表。
3. 若设置 `ENABLE_CODE_ENGINEER=1`，`EngineerAgent.process_ticket` 会收到该 ticket 并注册一个占位 `ActionEntry`，便于后续继续跟进。

## Engineer 占位逻辑

目前 `EngineerAgent` 只做：
1. 根据 ticket 生成 action 名称（默认 `constraints.suggested_name` 或 `custom_<ticket_id>`）。
2. 在 `ActionRegistry` 中登记 `ActionEntry`，记录 code_path、IO 描述等元数据。
3. 通过日志提醒“需要补充真实实现”。

后续可以扩展为：
- 调用代码 LLM 生成 `.py` 文件；
- 运行单测/仿真验证；
- 自动将实现加入 `SkillExecutor` 或独立模块中。

## 查询接口

`TaskProcessor` 暴露了以下方法（可被 UI、调试脚本调用）：
- `request_action(name, description, inputs, outputs, constraints)`：手动创建需求单。
- `list_action_tickets()`：查看待处理需求单。
- `list_registered_actions()`：查看已登记的动作条目。
- `list_primitives()`：查看底层原语（待后续补充）。

`RobotAPI.build(..., registry=ActionRegistry)` 会把注册表引用暴露给外层，便于 Planner/Engineer 等组件共享。

## 下一步建议

- 把 `ActionRegistry` 持久化（例如写入 JSON/数据库）并加入版本控制。
- 给 `EngineerAgent` 接入真实的代码生成流程，自动编辑 `actions/xxx.py` 并触发测试。
- 在 UI 或 CLI 中提供查看/处理需求单的指令，让人工也能参与审批/合并。
- **动作模板**：每个 `actions/foo.py` 建议写成

```python
from voice.task_structures import ExecutionResult

def run(api, runtime, **kwargs):
    # 使用 api.navigation/api.perception 等原语
    api.navigation.goto_marker("desk")
    observation, _ = api.perception.observe("可乐")
    if not observation.found:
        return {"status": "failure", "reason": "target_not_found"}
    result = api.manipulation.execute_skill("pick", observation=observation)
    return result  # 可以返回 ExecutionResult 或 dict
```

- **测试与审核**：把验证脚本路径写进 `ActionEntry.tests`，在 CI/本地执行 `pytest tests/actions/test_foo.py`，通过后再 `register_action` 并关闭 ticket。
- **实现动作代码**：在 `actions/<name>.py` 中编写 `run(api, runtime, **kwargs)`，返回 `ExecutionResult` 或字典，随后运行测试并 `registry.register_action(...)`。
- **动态加载**：开启 `ENABLE_DYNAMIC_ACTIONS` 后，行为树遇到不在 `SkillExecutor` 里的 action 时，会先检查 registry 是否存在实现，如果有则动态 import 并执行。
- **LLM 自动生成**：启用 `ENABLE_CODE_ENGINEER` 且配置 DeepSeek Key 时，EngineerAgent 会把 ticket + RobotAPI 接口说明发给 DeepSeek 生成 `actions/<name>.py` 和对应 `tests/actions/test_<name>.py`，注册后即可被动态执行。
