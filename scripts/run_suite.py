#!/usr/bin/env python3
"""
Lightweight suite runner to generate deterministic traces for baselines/ablations.

Usage:
  python scripts/run_suite.py --mode ours_full --episodes 30 --seed 42 --output_dir runs/golden_full
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List


def _rand_id(prefix: str = "") -> str:
    return prefix + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))


def simulate_episode(
    *,
    mode: str,
    episode_id: str,
    trace_path: str,
    rng: random.Random,
    scenario_id: str,
    episode_timeout_s: float,
    skill_timeout_s: float,
) -> None:
    skills = ["navigate_area", "observe_scene", "finalize_target_pose", "predict_grasp_point", "execute_grasp"]
    steps = rng.randint(4, 6)
    verifier_on = mode == "ours_full"
    recovery_on = mode == "ours_full"
    start_ts = time.time()

    with open(trace_path, "a", encoding="utf-8") as fp:
        for idx in range(1, steps + 1):
            if time.time() - start_ts > episode_timeout_s:
                fp.write(
                    json.dumps(
                        {
                            "episode_id": episode_id,
                            "step_id": idx,
                            "skill_name": "__episode_timeout",
                            "exec_status": "error",
                            "verified": False,
                            "failure_code": "infra.episode_timeout",
                            "evidence": {
                                "timeout_s": episode_timeout_s,
                                "elapsed_s": round(time.time() - start_ts, 3),
                                "scenario_id": scenario_id,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                return

            skill = skills[idx % len(skills)]
            elapsed_ms = rng.uniform(50, 400)
            failure = False
            failure_code = None
            reason = None

            if elapsed_ms / 1000.0 > skill_timeout_s:
                failure = True
                failure_code = "infra.skill_timeout"
                reason = "skill_timeout"
            else:
                base_success = 0.75 if mode == "ours_full" else 0.65
                failure = rng.random() > base_success

            verified = True if (verifier_on and not failure) else (False if verifier_on and failure else None)
            evidence: Dict[str, Any] = {}

            if failure and not failure_code:
                failure_code = rng.choice(
                    [
                        "nav.nav_blocked",
                        "manip.ik_fail",
                        "manip.grasp_fail",
                        "contract.verification_failed",
                    ]
                )
                reason = failure_code.split(".")[-1]

            event = {
                "episode_id": episode_id,
                "step_id": idx,
                "skill_name": skill,
                "exec_status": "error" if failure else "ok",
                "elapsed_ms": round(elapsed_ms, 3),
                "verified": verified,
                "failure_code": failure_code,
                "evidence": evidence,
                "scenario_id": scenario_id,
            }
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")

            if failure and recovery_on:
                rec_success = rng.random() > 0.3
                rec_event = {
                    "episode_id": episode_id,
                    "skill_name": "recover",
                    "exec_status": "ok",
                    "elapsed_ms": round(rng.uniform(20, 120), 3),
                    "verified": None,
                    "failure_code": None,
                    "evidence": {},
                    "recovery_level": "L1",
                    "recovery_attempt_idx": 1,
                    "recovery_action_name": "recover",
                    "recovery_success": rec_success,
                    "scenario_id": scenario_id,
                }
                fp.write(json.dumps(rec_event, ensure_ascii=False) + "\n")
                if rec_success:
                    retry = dict(event)
                    retry["failure_code"] = None
                    retry["exec_status"] = "ok"
                    retry["verified"] = True if verifier_on else None
                    retry["step_id"] = idx + 0.5
                    fp.write(json.dumps(retry, ensure_ascii=False) + "\n")


def run_suite(
    *,
    mode: str,
    episodes: int,
    seed: int,
    output_dir: str,
    episode_timeout_s: float,
    skill_timeout_s: float,
) -> None:
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, "trace.jsonl")
    if os.path.exists(trace_path):
        os.remove(trace_path)

    for ep_idx in range(episodes):
        ep_id = _rand_id("ep_")
        scenario_id = f"scenario_{rng.randint(1,5)}"
        meta = {
            "episode_id": ep_id,
            "step_id": 0,
            "skill_name": "__episode_meta",
            "exec_status": "ok",
            "verified": None,
            "failure_code": None,
            "evidence": {
                "seed": seed,
                "scenario_id": scenario_id,
                "mode": mode,
                "enable_verifier": mode == "ours_full",
                "enable_recovery": mode == "ours_full",
                "episode_index": ep_idx,
            },
        }
        with open(trace_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
        simulate_episode(
            mode=mode,
            episode_id=ep_id,
            trace_path=trace_path,
            rng=rng,
            scenario_id=scenario_id,
            episode_timeout_s=episode_timeout_s,
            skill_timeout_s=skill_timeout_s,
        )

    # compute metrics
    try:
        from scripts.compute_metrics import compute_metrics, load_traces, write_outputs
    except Exception:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compute_metrics", os.path.join(os.path.dirname(__file__), "compute_metrics.py")
        )
        compute_module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(compute_module)  # type: ignore
        compute_metrics = compute_module.compute_metrics  # type: ignore
        load_traces = compute_module.load_traces  # type: ignore
        write_outputs = compute_module.write_outputs  # type: ignore

    records = load_traces(output_dir)
    summary, ep_metrics = compute_metrics(records)
    write_outputs(summary, ep_metrics, output_dir)

    print("=== Run Summary ===")
    print(f"mode: {mode}")
    print(f"episodes: {episodes}")
    for k in ["success_rate", "mean_tool_calls", "mean_time_sec", "recovery_episode_rate", "recovery_success_rate"]:
        if k in summary:
            print(f"{k}: {summary[k]}")
    print(f"failure_code_top: {summary.get('failure_code_top')}")
    print(f"metrics saved to: {os.path.join(output_dir, 'metrics.json')} / metrics.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run suite to generate traces and metrics.")
    parser.add_argument("--mode", choices=["tool_use_only", "ours_full"], default="ours_full")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode_timeout_s", type=float, default=120.0)
    parser.add_argument("--skill_timeout_s", type=float, default=8.0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--exp_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = args.exp_name or f"{args.mode}"
        output_dir = os.path.join("runs", f"{name}_{stamp}")
    run_suite(
        mode=args.mode,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=output_dir,
        episode_timeout_s=args.episode_timeout_s,
        skill_timeout_s=args.skill_timeout_s,
    )


if __name__ == "__main__":
    main()
