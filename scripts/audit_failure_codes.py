#!/usr/bin/env python3
"""
Audit failure_code / reason consistency from trace.jsonl.

Outputs to stdout:
- Top-20 failure_code frequencies
- Reasons with UNKNOWN/None failure_code
- Conflicts: same reason mapped to multiple failure_codes
- Evidence completeness warnings for top-10 failure codes
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def load_trace(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception:
                continue
    return records


def audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    fc_counter = Counter()
    unknown_reasons = Counter()
    reason_to_fc: Dict[str, set] = defaultdict(set)
    evidence_warn: Counter = Counter()

    for rec in records:
        fc = rec.get("failure_code")
        reason = rec.get("reason")
        if fc:
            fc_counter[fc] += 1
        if reason:
            reason_to_fc[reason].add(fc or "NONE")
        if (not fc) or fc == "unknown":
            if reason:
                unknown_reasons[reason] += 1
        # evidence completeness check
        if fc:
            ev = rec.get("evidence") or {}
            if not isinstance(ev, dict) or len(ev) == 0:
                evidence_warn[fc] += 1

    top_fc = fc_counter.most_common(20)
    conflicts = {r: fcs for r, fcs in reason_to_fc.items() if len([x for x in fcs if x]) > 1}

    return {
        "top_failure_codes": top_fc,
        "unknown_reasons": unknown_reasons.most_common(),
        "conflicts": {k: list(v) for k, v in conflicts.items()},
        "evidence_warn": evidence_warn.most_common(),
    }


def print_report(report: Dict[str, Any]) -> None:
    print("== Top-20 failure_code ==")
    for fc, cnt in report["top_failure_codes"]:
        print(f"{fc:35s} {cnt}")
    print("\n== Reasons with UNKNOWN/None failure_code ==")
    for reason, cnt in report["unknown_reasons"]:
        print(f"{cnt:5d}  {reason}")
    print("\n== Conflicts (same reason -> multiple failure_codes) ==")
    if not report["conflicts"]:
        print("(none)")
    else:
        for reason, fcs in report["conflicts"].items():
            print(f"{reason}: {fcs}")
    print("\n== Evidence completeness warnings (empty evidence) ==")
    for fc, cnt in report["evidence_warn"]:
        print(f"{fc:35s} {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit failure codes and reasons in trace.jsonl")
    parser.add_argument("trace", help="Path to trace.jsonl (e.g., runs/exp/trace.jsonl)")
    args = parser.parse_args()
    records = load_trace(args.trace)
    report = audit(records)
    print_report(report)


if __name__ == "__main__":
    main()
