1. 顶层目录结构设计
建议项目名就叫 aureka：
aureka/
├── aureka/                    # 主 Python 包
│   ├── __init__.py
│   ├── config/                # 配置与实验参数
│   │   ├── __init__.py
│   │   ├── base_config.py
│   │   └── tabletop_mvp.yaml
│   │
│   ├── core/                  # 核心 DSL 与通用抽象
│   │   ├── __init__.py
│   │   ├── skill.py           # SkillPrimitive, SkillLibrary
│   │   ├── context.py         # RobotContext 抽象与基类
│   │   ├── logging.py         # 统一日志、轨迹结构
│   │   └── metrics.py         # 通用指标（utility, redundancy 等）
│   │
│   ├── envs/                  # 各种环境适配（模拟 / 真机）
│   │   ├── __init__.py
│   │   ├── tabletop_mock.py   # 当前桌面MVP模拟环境
│   │   ├── isaac_tabletop.py  # 未来: Isaac Gym 适配
│   │   └── ros_robot.py       # 未来: ROS/真机适配
│   │
│   ├── executor/              # 执行器（技能运行 & 轨迹采集）
│   │   ├── __init__.py
│   │   ├── runner.py          # SkillExecutor, EvaluationRunner
│   │   └── planners.py        # 简单planner / 随机探索 / RL接口
│   │
│   ├── reflector/             # 反思器（日志→统计→反馈文本）
│   │   ├── __init__.py
│   │   ├── analyzer.py        # per-skill统计 & 库级统计
│   │   └── reflector.py       # ActionReflector (生成给LLM的反馈)
│   │
│   ├── generator/             # LLM 接口 & Prompt 生成
│   │   ├── __init__.py
│   │   ├── llm_client.py      # OpenAI/其他模型封装
│   │   ├── prompts.py         # System prompt & 模板
│   │   └── parser.py          # 解析 LLM 输出成 SkillPrimitive
│   │
│   ├── loop/                  # AUREKA 进化主循环
│   │   ├── __init__.py
│   │   ├── evolution.py       # AurekaEvolutionLoop
│   │   └── phases.py          # Phase1/2/3 (Repair/Discovery/Ortho)
│   │
│   ├── viz/                   # 可视化（可选）
│   │   ├── __init__.py
│   │   └── skill_viz.py       # PCA/轨迹分布/冗余heatmap
│   │
│   └── utils/                 # 杂项工具
│       ├── __init__.py
│       ├── safety.py          # 安全exec封装
│       └── sampling.py        # 参数采样工具
│
├── scripts/                   # 命令行脚本 / 实验入口
│   ├── run_tabletop_mvp.py
│   ├── run_phase1_repair.py
│   └── run_phase2_discovery.py
│
├── tests/                     # 单元测试
│   ├── test_skill_execution.py
│   ├── test_reflector.py
│   ├── test_parser.py
│   └── test_evolution_loop.py
│
├── examples/                  # 示例配置 & demo notebook
│   └── tabletop_demo.ipynb
│
├── pyproject.toml / setup.py
└── README.md
可以先把 tabletop_mock + runner + skill + reflector + generator + loop/evolution 跑通，就是 MVP。
2. 各模块职责与关键接口设计
2.1 core/：DSL + 抽象
core/context.py
定义统一的上下文接口，所有环境实现都继承这个基类：
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple

class RobotContext(ABC):
    """
    抽象的机器人上下文接口，封装可被Skill调用的低层API。
    """

    @abstractmethod
    def reset(self):
        """重置到一个标准初始状态（位置、夹爪等）"""

    @abstractmethod
    def get_ee_pose(self) -> np.ndarray:
        """返回当前末端位姿 (至少 [x, y, z])"""

    @abstractmethod
    def get_object_pose(self, name: str) -> np.ndarray:
        """返回指定物体的位姿"""

    @abstractmethod
    def move_ee_delta(self, dx: float, dy: float, dz: float) -> None:
        """末端增量运动，内部要做好安全软限位"""

    @abstractmethod
    def set_gripper(self, state: float) -> None:
        """设置夹爪状态 0.0~1.0"""

    # 可选扩展：base 移动、旋转、关节空间API等

    def log(self, event: str, data: Dict[str, Any]):
        """统一日志接口，方便后面记录数据"""
        pass
core/skill.py
动作 DSL + Skill Library：
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np

@dataclass
class SkillPrimitive:
    name: str
    description: str
    code: str
    param_spec: Dict[str, str] = field(default_factory=dict)

    tags: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)

    # 统计信息（由 executor 填）
    success_history: List[bool] = field(default_factory=list)
    state_changes: List[float] = field(default_factory=list)
    effect_vectors: List[np.ndarray] = field(default_factory=list)

    def to_prompt_str(self) -> str:
        return (
            f"Skill: {self.name}\n"
            f"Description: {self.description}\n"
            f"Tags: {self.tags}\n"
            f"Preconditions: {self.preconditions}\n"
            f"Postconditions: {self.postconditions}\n"
            f"Params: {self.param_spec}\n"
            f"Code:\n{self.code}\n"
        )

@dataclass
class SkillLibrary:
    """
    上一层用这个容器管理一组技能。
    """
    skills: List[SkillPrimitive] = field(default_factory=list)

    def get_by_name(self, name: str) -> Optional[SkillPrimitive]:
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def to_prompt_str(self) -> str:
        return "\n\n".join(s.to_prompt_str() for s in self.skills)
core/metrics.py
封装通用指标计算逻辑，Reflector 直接调用：
from typing import List, Dict, Tuple
import numpy as np
from .skill import SkillPrimitive

def compute_success_rate(skill: SkillPrimitive) -> float:
    if not skill.success_history:
        return 0.0
    return sum(skill.success_history) / len(skill.success_history)

def compute_utility(skill: SkillPrimitive, threshold: float = 1e-3) -> float:
    if not skill.state_changes:
        return 0.0
    non_zero = sum(1 for d in skill.state_changes if d > threshold)
    return non_zero / len(skill.state_changes)

def compute_effect_signature(skill: SkillPrimitive) -> np.ndarray | None:
    if not skill.effect_vectors:
        return None
    return np.mean(np.stack(skill.effect_vectors, axis=0), axis=0)

def compute_redundant_pairs(skills: List[SkillPrimitive],
                            cos_threshold: float = 0.9) -> List[Tuple[str, str, float]]:
    sigs: Dict[str, np.ndarray] = {}
    for s in skills:
        sig = compute_effect_signature(s)
        if sig is not None:
            sigs[s.name] = sig

    names = list(sigs.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v1, v2 = sigs[names[i]], sigs[names[j]]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
            if cos > cos_threshold:
                pairs.append((names[i], names[j], cos))
    return pairs
2.2 envs/：具体环境实现
envs/tabletop_mock.py
把你现在那份 RobotContext 挪到这里并继承 core.context.RobotContext：
import numpy as np
from typing import Dict, Any
from aureka.core.context import RobotContext

class TabletopMockContext(RobotContext):
    def __init__(self):
        self._init_ee_pose = np.array([0.3, 0.0, 0.5])
        self.ee_pose = self._init_ee_pose.copy()
        self.gripper_state = 0.0
        self.history = []

    def reset(self):
        self.ee_pose = self._init_ee_pose.copy()
        self.gripper_state = 0.0
        self.history.clear()

    def get_ee_pose(self) -> np.ndarray:
        return self.ee_pose.copy()

    def get_object_pose(self, obj_name: str) -> np.ndarray:
        # MVP: 固定一个方块
        return np.array([0.4, 0.1, 0.05])

    def move_ee_delta(self, dx: float, dy: float, dz: float) -> None:
        limit = 0.1
        dx, dy, dz = [max(min(v, limit), -limit) for v in [dx, dy, dz]]
        self.ee_pose += np.array([dx, dy, dz])
        self.log("move_ee_delta", {"dx": dx, "dy": dy, "dz": dz, "ee_pose": self.ee_pose.copy()})

    def set_gripper(self, state: float) -> None:
        self.gripper_state = max(0.0, min(1.0, state))
        self.log("set_gripper", {"state": self.gripper_state})

    def log(self, event: str, data: Dict[str, Any]):
        self.history.append((event, data))
        print(f"[Sim] {event}: {data}")
未来接 Isaac / ROS 只要实现同样接口即可。
2.3 executor/：执行技能并收集轨迹
executor/runner.py
import numpy as np
from typing import List, Dict, Any
from aureka.core.skill import SkillPrimitive, SkillLibrary
from aureka.core.context import RobotContext
from aureka.utils.safety import safe_exec
from aureka.utils.sampling import sample_params

class SkillExecutor:
    """
    负责：编译 skill.code → 可调用函数；执行多次；统计位移向量等。
    """

    def compile_skill(self, skill: SkillPrimitive):
        func = safe_exec(skill.code, skill.name)
        return func

    def run_skill(self, skill: SkillPrimitive, ctx: RobotContext,
                  num_trials: int = 3):
        func = self.compile_skill(skill)
        if func is None:
            skill.success_history.extend([False] * num_trials)
            skill.state_changes.extend([0.0] * num_trials)
            return

        for _ in range(num_trials):
            ctx.reset()
            start = ctx.get_ee_pose()
            kwargs = sample_params(skill.param_spec)

            try:
                success = func(ctx, **kwargs)
                end = ctx.get_ee_pose()
                disp_vec = end - start
                disp = float(np.linalg.norm(disp_vec))

                skill.success_history.append(bool(success))
                skill.state_changes.append(disp)
                skill.effect_vectors.append(disp_vec)
            except Exception:
                skill.success_history.append(False)
                skill.state_changes.append(0.0)

class EvaluationRunner:
    """
    对整个技能库进行批量评估。
    """
    def __init__(self, ctx: RobotContext, executor: SkillExecutor):
        self.ctx = ctx
        self.executor = executor

    def evaluate_library(self, lib: SkillLibrary, num_trials_per_skill: int = 3):
        for skill in lib.skills:
            self.executor.run_skill(skill, self.ctx, num_trials=num_trials_per_skill)
2.4 reflector/：统计 + 文字反馈
reflector/analyzer.py
封装 per-skill 和库级分析（用 core.metrics）：
from typing import Dict, Any, List
from aureka.core.skill import SkillPrimitive, SkillLibrary
from aureka.core import metrics

def analyze_skill(skill: SkillPrimitive) -> Dict[str, Any]:
    sr = metrics.compute_success_rate(skill)
    ut = metrics.compute_utility(skill)
    avg_effect = sum(skill.state_changes) / len(skill.state_changes) if skill.state_changes else 0.0
    return {
        "name": skill.name,
        "success_rate": sr,
        "utility": ut,
        "avg_effect": avg_effect,
    }

def analyze_library(lib: SkillLibrary) -> Dict[str, Any]:
    per_skill = [analyze_skill(s) for s in lib.skills]
    redundant_pairs = metrics.compute_redundant_pairs(lib.skills)
    return {
        "per_skill": per_skill,
        "redundant_pairs": redundant_pairs,
    }
reflector/reflector.py
from typing import List
from aureka.core.skill import SkillLibrary
from .analyzer import analyze_library

class ActionReflector:
    """
    把结构化统计信息转成供 LLM 使用的自然语言反馈。
    """

    def generate_feedback(self, lib: SkillLibrary) -> str:
        analysis = analyze_library(lib)
        lines: List[str] = []
        lines.append("Here is the performance analysis of the current Action Library:")

        for s in analysis["per_skill"]:
            name = s["name"]
            sr = s["success_rate"]
            ut = s["utility"]
            avg = s["avg_effect"]

            if sr == 0 and ut == 0 and avg == 0:
                lines.append(f"- Action `{name}`: UNTESTED or always failing.")
                continue

            if sr < 0.3:
                lines.append(
                    f"- Action `{name}`: FAILED OFTEN (Success={sr:.2f}, Utility={ut:.2f}). "
                    f"Likely bugs, safety violations, or unreachable targets."
                )
            elif avg < 0.01:
                lines.append(
                    f"- Action `{name}`: LOW EFFECT (Success={sr:.2f}, Utility={ut:.2f}, AvgEffect={avg:.4f}). "
                    f"It executes but causes almost no motion. Consider increasing step sizes or revising logic."
                )
            else:
                lines.append(
                    f"- Action `{name}`: GOOD (Success={sr:.2f}, Utility={ut:.2f}, AvgEffect={avg:.4f})."
                )

        pairs = analysis["redundant_pairs"]
        if pairs:
            lines.append("\nRedundancy Analysis:")
            for a, b, cs in pairs:
                lines.append(
                    f"- `{a}` and `{b}` have very similar effects (cosine similarity={cs:.2f}). "
                    f"Consider merging them into a single parametric skill."
                )

        lines.append("\nGeneral Suggestions:")
        lines.append("Check whether the library contains primitives for 'approaching', 'grasping', and 'lifting'. "
                     "If some are missing, CREATE new skills to fill the gap.")
        return "\n".join(lines)
2.5 generator/：LLM 接口 & Prompt
generator/prompts.py
AUREKA_SYSTEM_PROMPT = """
You are an expert Robot Control Architect. You design a library of high-level skill primitives for a tabletop manipulator.

CONTEXT:
1. Available low-level RobotContext API:
   - move_ee_delta(dx, dy, dz)
   - get_ee_pose()
   - get_object_pose(name)
   - set_gripper(state)

2. Current Action Library (each skill as Python code):
{skills_text}

3. Performance Feedback:
{feedback}

GOAL:
- Improve the Action Library according to the feedback.
- Keep useful skills, fix or remove failing or useless ones.
- Merge redundant skills, and introduce parameterized variants when two skills have similar effects.
- Create new skills if key capabilities (approach, grasp, lift) are missing.

HARD CONSTRAINTS:
- Do NOT import any external libraries. `np` is available for basic math.
- Do NOT call functions outside RobotContext and standard Python builtins.
- Each skill's `"code"` must define exactly ONE Python function.
- The function name must be identical to the `"name"` field.
- Each function must accept `ctx: RobotContext` as the first argument, followed by optional keyword arguments.

OUTPUT FORMAT:
Return a valid JSON list of skills. Example:
[
  {{
    "name": "approach_object",
    "description": "Move the end-effector above the object.",
    "tags": ["approach"],
    "preconditions": ["object 'box' is visible"],
    "postconditions": ["end-effector is above the object"],
    "params": {{"height": "float [0.05, 0.20]"}},
    "code": "def approach_object(ctx, height=0.1):\\n    obj = ctx.get_object_pose('box')\\n    ee = ctx.get_ee_pose()\\n    dx = obj[0] - ee[0]\\n    dy = obj[1] - ee[1]\\n    ctx.move_ee_delta(dx, dy, 0)\\n    return True"
  }}
]
"""
generator/llm_client.py
from typing import Any, Dict
import json

class LLMClient:
    def __init__(self, model_name: str = "gpt-4.1"):
        self.model_name = model_name
        # 在这里配置真实的openai/其他SDK

    def generate_skill_library(self, system_prompt: str) -> Any:
        # 调用真实LLM，这里先留接口
        # 返回原始文本（JSON字符串）
        raise NotImplementedError

generator/parser.py
from typing import List, Dict, Any
import json
from aureka.core.skill import SkillPrimitive, SkillLibrary

def parse_skill_library(json_text: str) -> SkillLibrary:
    data = json.loads(json_text)
    skills: List[SkillPrimitive] = []
    for d in data:
        skills.append(
            SkillPrimitive(
                name=d["name"],
                description=d.get("description", ""),
                code=d["code"],
                param_spec=d.get("params", {}),
                tags=d.get("tags", []),
                preconditions=d.get("preconditions", []),
                postconditions=d.get("postconditions", []),
            )
        )
    return SkillLibrary(skills=skills)
2.6 loop/：AUREKA 主循环
loop/evolution.py
from dataclasses import dataclass
from typing import Optional
from aureka.core.skill import SkillLibrary
from aureka.core.context import RobotContext
from aureka.executor.runner import SkillExecutor, EvaluationRunner
from aureka.reflector.reflector import ActionReflector
from aureka.generator.llm_client import LLMClient
from aureka.generator.prompts import AUREKA_SYSTEM_PROMPT
from aureka.generator.parser import parse_skill_library

@dataclass
class EvolutionConfig:
    num_iterations: int = 5
    num_trials_per_skill: int = 3

class AurekaEvolutionLoop:
    def __init__(self,
                 ctx: RobotContext,
                 llm_client: LLMClient,
                 initial_library: SkillLibrary,
                 config: EvolutionConfig):
        self.ctx = ctx
        self.llm = llm_client
        self.lib = initial_library
        self.config = config

        self.executor = SkillExecutor()
        self.runner = EvaluationRunner(ctx, self.executor)
        self.reflector = ActionReflector()

    def step_once(self):
        # 1. 执行当前技能库
        self.runner.evaluate_library(self.lib, num_trials_per_skill=self.config.num_trials_per_skill)

        # 2. 生成反馈文本
        feedback = self.reflector.generate_feedback(self.lib)
        skills_text = self.lib.to_prompt_str()
        prompt = AUREKA_SYSTEM_PROMPT.format(feedback=feedback, skills_text=skills_text)

        # 3. 调用LLM得到新技能库
        raw_json = self.llm.generate_skill_library(prompt)
        new_lib = parse_skill_library(raw_json)
        self.lib = new_lib  # 更新当前库

        # 清理上下文状态，为下一轮准备
        self.ctx.reset()

    def run(self):
        for it in range(self.config.num_iterations):
            print(f"\n=== AUREKA Iteration {it} ===")
            self.step_once()
        return self.lib
2.7 utils/：安全执行与参数采样
utils/safety.py
import types
from typing import Optional
import numpy as np

SAFE_BUILTINS = {
    "range": range,
    "min": min,
    "max": max,
    "abs": abs,
    "float": float,
    "int": int,
    "len": len,
}

def safe_exec(code: str, func_name: str):
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
    }
    local_scope = {}
    try:
        exec(code, safe_globals, local_scope)
        func = local_scope.get(func_name)
        if not isinstance(func, types.FunctionType):
            return None
        return func
    except Exception:
        return None
utils/sampling.py
import random
from typing import Dict, Any

def sample_params(param_spec: Dict[str, str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in param_spec.items():
        s = spec.lower()
        if "float" in s:
            if "[" in s and "]" in s:
                rng = s.split("[")[1].split("]")[0]
                low, high = map(float, rng.split(","))
            else:
                low, high = -0.1, 0.1
            params[name] = random.uniform(low, high)
        elif "int" in s:
            # 同理解析整数区间
            params[name] = 0
        else:
            # 先默认不采样字符串/枚举
            pass
    return params
2.8 scripts/：运行入口
scripts/run_tabletop_mvp.py
from aureka.envs.tabletop_mock import TabletopMockContext
from aureka.core.skill import SkillLibrary, SkillPrimitive
from aureka.generator.llm_client import LLMClient
from aureka.loop.evolution import AurekaEvolutionLoop, EvolutionConfig

class MockLLMClient(LLMClient):
    def generate_skill_library(self, system_prompt: str) -> str:
        # 这里先用你的 mock 返回，后面再接真实模型
        from .mock_llm_tabletop import mock_llm_response
        return mock_llm_response(system_prompt)

def main():
    ctx = TabletopMockContext()
    # 初始给一个很烂的技能
    initial_lib = SkillLibrary(skills=[
        SkillPrimitive(
            name="dummy_move",
            description="Does nothing.",
            code="def dummy_move(ctx):\n    return False",
            param_spec={}
        )
    ])

    llm = MockLLMClient()
    config = EvolutionConfig(num_iterations=3, num_trials_per_skill=3)

    loop = AurekaEvolutionLoop(ctx, llm, initial_lib, config)
    final_lib = loop.run()
    print("\nFinal Skill Library:")
    print(final_lib.to_prompt_str())

if __name__ == "__main__":
    main()
3. 一次完整 AUREKA 迭代数据流梳理
用一句话串起来你整个项目的结构：
初始化：
选一个 RobotContext（例如 TabletopMockContext）；
准备一个 SkillLibrary（可以先只有一个 dummy skill）；
创建 AurekaEvolutionLoop。
执行阶段（Executor）：
EvaluationRunner.evaluate_library 调用 SkillExecutor.run_skill；
通过 safe_exec 编译 SkillPrimitive.code 为函数，采样参数，运行多次；
对每次运行记录：success、state change、effect_vector。
统计与反思（Reflector）：
analyze_library 计算每个 skill 的 success_rate、utility、avg_effect 和库级 redundant_pairs；
ActionReflector.generate_feedback 把这些统计翻译成自然语言反馈文本。
代码生成（Generator）：
将 SkillLibrary.to_prompt_str() + feedback 代入 AUREKA_SYSTEM_PROMPT；
调用 LLMClient.generate_skill_library 得到 JSON 格式的新技能列表；
用 parse_skill_library 解析成新的 SkillLibrary。
更新与下一轮：
EvolutionLoop 用新 SkillLibrary 替换旧库；
重置 RobotContext；
进入下一轮迭代。