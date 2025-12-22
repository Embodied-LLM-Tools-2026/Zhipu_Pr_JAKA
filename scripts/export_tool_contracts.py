#!/usr/bin/env python3
"""
Export Tool Contracts table (Markdown + LaTeX) from the current skill set.

Usage:
  python scripts/export_tool_contracts.py

Outputs:
  artifacts/tool_contracts.md
  artifacts/tool_contracts.tex
"""

from __future__ import annotations

import os
import sys
import re
from typing import Dict, List, Any, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


ARTIFACT_DIR = "artifacts"
MD_PATH = os.path.join(ARTIFACT_DIR, "tool_contracts.md")
TEX_PATH = os.path.join(ARTIFACT_DIR, "tool_contracts.tex")

EXECUTOR_PATH = os.path.join(ROOT, "voice", "control", "executor.py")
RECOVERY_PATH = os.path.join(ROOT, "voice", "control", "recovery.py")
TASK_STRUCT_PATH = os.path.join(ROOT, "voice", "control", "task_structures.py")

# Minimal FailureCode names extracted via regex to avoid heavy imports
FAILURE_PATTERN = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*\"([^\"]+)\"", re.MULTILINE)


def load_failure_codes() -> Dict[str, str]:
    text = open(TASK_STRUCT_PATH, "r", encoding="utf-8").read()
    codes = {}
    for name, value in FAILURE_PATTERN.findall(text):
        codes[name] = value
    return codes

def list_skills() -> List[str]:
    text = open(EXECUTOR_PATH, "r", encoding="utf-8").read()
    skills = re.findall(r"def _skill_([a-zA-Z0-9_]+)\s*\(", text)
    return sorted(set(skills))


def verifier_mapping() -> Dict[str, str]:
    return {
        "search_area": "verify_target_visible",
        "approach_far": "verify_target_visible",
        "finalize_target_pose": "verify_pose_ready",
        "execute_grasp": "verify_grasp_success",
        "vla_grasp_finish": "verify_grasp_success",
    }


def failure_code_hints() -> Dict[str, List[str]]:
    codes = load_failure_codes()
    C = lambda name: codes.get(name, name.lower())
    return {
        "execute_grasp": [
            C("IK_FAIL"),
            C("GRASP_FAIL"),
            C("VERIFICATION_FAILED"),
        ],
        "predict_grasp_point": [
            C("ZEROGRASP_FAILED"),
            C("IK_FAIL"),
        ],
        "finalize_target_pose": [
            C("DEPTH_LOCALIZATION_FAILED"),
            C("NAV_BLOCKED"),
        ],
        "navigate_area": [
            C("NAV_BLOCKED"),
            C("MISSING_TARGET"),
        ],
        "search_area": [
            C("NO_OBSERVATION"),
            C("ROTATE_FAILED"),
        ],
        "rotate_scan": [C("ROTATE_FAILED")],
        "approach_far": [
            C("NAVIGATOR_UNAVAILABLE"),
            C("NO_OBSERVATION"),
        ],
        "open_gripper": [C("GRIPPER_UNAVAILABLE")],
        "close_gripper": [C("GRIPPER_UNAVAILABLE")],
        "vla_grasp_finish": [
            C("VLA_NO_EFFECT"),
            C("VLA_POLICY_OOB"),
            C("VERIFICATION_FAILED"),
        ],
        "recover": [C("UNKNOWN")],
    }

def recovery_levels_from_source() -> Dict[str, str]:
    text = open(RECOVERY_PATH, "r", encoding="utf-8").read()
    mapping: Dict[str, str] = {}
    for m in re.finditer(r"FailureCode\.([A-Z0-9_]+)\s*:\s*RecoveryPlan\(\s*level=\"(L[123])\"", text):
        mapping[m.group(1)] = m.group(2)
    return mapping


def build_rows() -> List[Dict[str, Any]]:
    skills = list_skills()
    verifier_map = verifier_mapping()
    failure_map = failure_code_hints()
    recovery_level_map = recovery_levels_from_source()
    rows: List[Dict[str, Any]] = []
    for skill in skills:
        failures = failure_map.get(skill, ["TODO"])
        recovery_levels = sorted(
            {recovery_level_map.get(fc.split(".")[-1].upper(), "—") for fc in failures if fc != "TODO"}
        )
        rows.append(
            {
                "skill": skill,
                "pre": "TODO",
                "constraints": "TODO",
                "verifier": verifier_map.get(skill, "—"),
                "failures": ", ".join(failures),
                "recovery": ", ".join([r for r in recovery_levels if r] or ["—"]),
            }
        )
    return rows


def render_md(rows: List[Dict[str, Any]]) -> str:
    header = "| Skill | Preconditions | Constraints | Post-Verifier | Failure Codes | Recovery |\n"
    header += "| --- | --- | --- | --- | --- | --- |\n"
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['skill']} | {row['pre']} | {row['constraints']} | {row['verifier']} | {row['failures']} | {row['recovery']} |"
        )
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
    }
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def render_tex(rows: List[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llllll}",
        r"\textbf{Skill} & \textbf{Preconditions} & \textbf{Constraints} & \textbf{Post-Verifier} & \textbf{Failure Codes} & \textbf{Recovery} \\ \hline",
    ]
    for row in rows:
        fields = [
            _latex_escape(row["skill"]),
            _latex_escape(row["pre"]),
            _latex_escape(row["constraints"]),
            _latex_escape(row["verifier"]),
            _latex_escape(row["failures"]),
            _latex_escape(row["recovery"]),
        ]
        lines.append(" & ".join(fields) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    md = render_md(rows)
    tex = render_tex(rows)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"exported {len(rows)} skills to {MD_PATH} and {TEX_PATH}")


if __name__ == "__main__":
    main()
