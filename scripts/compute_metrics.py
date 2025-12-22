#!/usr/bin/env python3
"""
Compute basic offline metrics from trace.jsonl files under runs/.

Outputs:
  - Console summary
  - metrics.json
  - metrics.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import csv
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple


def load_traces(runs_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.isdir(runs_dir):
        print(f"[WARN] runs dir not found: {runs_dir}", file=sys.stderr)
        return records
    for root, _, files in os.walk(runs_dir):
        for name in files:
            if name != "trace.jsonl":
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            rec["_trace_path"] = path
                            records.append(rec)
                        except Exception:
                            continue
            except Exception as exc:
                print(f"[WARN] failed to read {path}: {exc}", file=sys.stderr)
    return records


def group_by_episode(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    episodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        ep = str(rec.get("episode_id") or "unknown")
        episodes[ep].append(rec)
    return episodes


def episode_success(trace: List[Dict[str, Any]]) -> bool:
    # Success if no failure skill at end; fallback: any verified failure marks fail.
    if not trace:
        return False
    status_values = [rec.get("exec_status") for rec in trace if rec.get("skill_name")]
    failure_codes = [rec.get("failure_code") for rec in trace if rec.get("failure_code")]
    if any(fc for fc in failure_codes):
        return False
    if any(status == "error" for status in status_values):
        return False
    return True


def compute_metrics(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    episodes = group_by_episode(records)
    ep_metrics: List[Dict[str, Any]] = []
    recovery_episode_count = 0
    recovery_success_count = 0
    planner_calls_total = 0  # Placeholder, no field yet

    for ep_id, trace in episodes.items():
        tools = [r for r in trace if r.get("skill_name")]
        failures = [r for r in tools if r.get("failure_code")]
        recoveries = [r for r in trace if r.get("recovery_level")]
        recovery_episode = bool(recoveries)
        if recovery_episode:
            recovery_episode_count += 1
            if any(r.get("recovery_success") for r in recoveries):
                recovery_success_count += 1
        total_elapsed = 0.0
        for r in tools:
            try:
                total_elapsed += float(r.get("elapsed_ms", 0.0)) / 1000.0
            except Exception:
                continue
        ep_metric = {
            "episode_id": ep_id,
            "tool_calls": len(tools),
            "success": episode_success(trace),
            "elapsed_sec": total_elapsed if total_elapsed > 0 else None,
            "failures": len(failures),
            "recovery_count": len(recoveries),
        }
        ep_metrics.append(ep_metric)

    total_eps = max(1, len(ep_metrics))
    success_rate = sum(1 for m in ep_metrics if m.get("success")) / total_eps
    mean_tool_calls = sum(m.get("tool_calls", 0) for m in ep_metrics) / total_eps
    mean_time = sum(m.get("elapsed_sec") or 0.0 for m in ep_metrics) / total_eps

    failure_counter: Counter = Counter()
    for rec in records:
        fc = rec.get("failure_code")
        if fc:
            failure_counter[fc] += 1

    summary = {
        "episodes": len(ep_metrics),
        "success_rate": success_rate,
        "mean_tool_calls": mean_tool_calls,
        "mean_time_sec": mean_time,
        "failure_code_top": failure_counter.most_common(5),
        "recovery_episode_rate": recovery_episode_count / total_eps,
        "recovery_success_rate": (recovery_success_count / recovery_episode_count) if recovery_episode_count else 0.0,
        "planner_calls": planner_calls_total,  # TODO: populate when field available
    }
    return summary, ep_metrics


def write_outputs(summary: Dict[str, Any], ep_metrics: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metrics.json")
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    fieldnames = ["episode_id", "success", "tool_calls", "elapsed_sec", "failures", "recovery_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ep_metrics:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics from trace.jsonl under runs/")
    parser.add_argument("--runs_dir", default="runs", help="Root directory containing run subfolders with trace.jsonl")
    parser.add_argument("--out_dir", default="runs", help="Directory to write metrics.json/csv (default: runs)")
    args = parser.parse_args()

    records = load_traces(args.runs_dir)
    if not records:
        print("No trace records found.", file=sys.stderr)
        sys.exit(1)

    summary, ep_metrics = compute_metrics(records)

    print("=== Metrics Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    write_outputs(summary, ep_metrics, args.out_dir)


if __name__ == "__main__":
    main()
