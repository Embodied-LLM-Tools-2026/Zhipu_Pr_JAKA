Phase 0：打地基（1–2 天内可完成）
Step 0.1 统一 Trace 日志（先做这个，后面一切靠它）

做什么

为每次 skill 调用记录一条结构化日志：输入参数、precheck、执行结果、verify 结果、failure_code、recovery 尝试、耗时等。

输出到 JSONL/CSV（推荐 JSONL + 后处理脚本）。

DoD

任何 episode 结束后能自动生成：

success rate

mean tool calls

failure_code 分布

planner calls（LLM 调用次数）

Step 0.2 引入 FailureCode 枚举 + reason 映射表（P0 必须）

做什么

新建 FailureCode Enum（PERCEPTION/NAV/MANIP/INFRA...）

写 map_reason_to_failure_code(reason)->FailureCode

ExecutionResult 增加 failure_code 字段（兼容原 reason）

DoD

你现在日志里出现的所有高频 reason（no_observation, zerograsp_failed...）都能映射到枚举

统计脚本能按 failure_code 出饼图/柱图

Phase 1：显式 Verification Loop（2–4 天，RSS 分水岭）
Step 1.1 改造 SkillExecutor：强制 Action→Wait→Verify→Result

做什么

在 _call_skill 或 SkillExecutor 的统一入口，执行完 skill 一定跑 verifier

verifier 失败则把该 skill 判为失败（即使执行没报错）

DoD

任何 skill 执行后日志里都有 verified=true/false

失败时 failure_code 不再是 “执行异常”，而是 “验证失败 / 具体现象”

Step 1.2 先实现 3 个最关键 verifier（最小集合）

做什么

verify_target_visible(target)

verify_pose_ready()

verify_grasp_success()（夹爪宽度/电流 + 简单视觉确认）

DoD

golden path 全流程每一步都有 verifier

verifier 输出包含可用于 debug 的证据（confidence、error、current、width 等）

Phase 2：Structured Recovery（3–5 天，提升成功率 + 论文亮点）
Step 2.1 写 recovery_policy：failure_code → recovery plan

做什么

建一个表驱动的 recovery_policy(failure_code)，返回一串确定性工具调用（L1/L2）

默认策略：先 L1（反射），失败再 L2（局部重规划），最后才 L3（LLM 全局重规划）

DoD

每次失败都会记录：

是否触发 recovery

recovery 尝试次数

recovery 是否成功（恢复后任务最终成功）

能计算论文最关键指标：Recovery Rate

Step 2.2 实现 Level 1 Reflex（完全不经过 LLM）

做什么（最小 3 个反射动作）

IK_FAIL → nudge_base → retry

NAV_BLOCKED → backoff → retry

GRASP_SLIP/GRASP_FAIL → open→reset→regrasp

DoD

L1 的触发与结果都被记录

L1 成功时不需要再 call planner

Phase 3：先跑“能写论文”的实验（1 周内出结果）
Step 3.1 建立对比配置开关（为了消融/基线）

做什么

enable_verifier

enable_recovery

enable_failure_taxonomy

mode = tool_use / bt_only

DoD

同一套任务，一键切换出：

B0：tool_use only（无 verify/recovery）

B1：bt_only

Ours：verify+recovery+failure_code

Step 3.2 跑 Golden Path（T1）并出第一张表

做什么

每方法 30–50 episodes

输出 SR、time、tool calls、planner calls、failure dist

DoD

你能在文档里写出一句非常硬的结果：

“Verifier+Recovery 将 SR 从 X 提升到 Y，同时 Planner Calls 从 A 降到 B”

Step 3.3 跑 1–2 个 Stress Test（S1/S2 优先）

做什么

S1：感知链路压力（RGB/Depth 异常）

S2：ZeroGrasp 失败压力（无候选、pose unavailable）

DoD

Failure distribution 图明显变化

Recovery Rate 在压力场景下仍有提升

到 Phase 3 结束：你已经能投稿 RSS（哪怕没有 VLA）。

Phase 4：VLA-as-Function 接入（加分，不影响主线）
Step 4.1 先做 1 个 VLA skill：vla_grasp_finish（1–3 秒 horizon）

做什么

封装 call_vla_skill(instruction, image, state)->action_chunk

返回也必须走同一套 contract：precheck/verify/failure_code

DoD

这个 tool 可以被插进 execute_grasp 前后

有可统计的 failure_code（例如 VLA_NO_EFFECT, VLA_POLICY_OOB）

Step 4.2 Router 规则版（最容易写、最容易出论文对比）

做什么

规则：ZEROGRASP_FAILED 或 GRASP_FAIL → fallback to VLA finish

或：target 是软体/复杂接触 → 直接选 VLA

DoD

能跑出 “classic-only vs router-with-vla” 对比

指标重点：SR、Recovery Rate、Catastrophic Failures 是否下降

Phase 5：论文冲顶（工程+写作一体）
Step 5.1 生成“工具契约表”自动化（表格就是论文资产）

做什么

用代码从每个 skill 的 metadata 导出契约表（markdown/latex）

自动导出 failure taxonomy & recovery mapping

DoD

每次改技能，表格自动更新

论文 Method 表不用手改

Step 5.2 视频与 Failure Analysis（RSS 非常吃这个）

做什么

每个 failure_code 录 1–2 个代表视频

展示 recovery 的层级过程（L1→L2→L3）

DoD

你的视频能“讲清楚系统为何可靠”，不是只展示成功

最推荐的执行顺序（精简版清单）

按这个做，不会后悔：

Trace logging（JSONL/CSV）

FailureCode enum + reason 映射

强制 post-verify loop

3 个 verifier（visible / pose / grasp）

recovery_policy（L1/L2/L3）

L1 reflex recovery（不经 LLM）

baseline/ablation 开关

跑 T1（Golden）出第一张结果表

跑 S1/S2（压力）出 failure 分布 & recovery rate

再接 VLA skill + router（加分项）