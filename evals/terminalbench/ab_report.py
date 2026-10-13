#!/usr/bin/env python3
"""Per-arm and paired A/B report for Harbor + Pi jobs (TB3 or SWE-bench).

  python3 -m evals.terminalbench.ab_report --a JOB_DIR[,JOB_DIR] --b JOB_DIR[,JOB_DIR]

Reuses summarize.summarize_job and Harbor's result.json. Adds per task: agent wall
time, verifier time, turns (assistant messages), peak context (max input+cacheRead
over turns), verifier tests passed/total (CTRF), and an end reason:
completed | agent_timeout | endpoint_error | other_error.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from evals.terminalbench.summarize import (
    compact_seconds,
    elapsed,
    summarize_job,
    trajectory_records,
)


def turn_stats(trial_dir: Path) -> tuple[int, int]:
    """Return (turns, peak_context_tokens) from Pi message_end usage."""
    records, _ = trajectory_records(trial_dir)
    turns = 0
    peak = 0
    for rec in records:
        msg = rec.get("message") or rec
        if msg.get("role") != "assistant":
            continue
        turns += 1
        usage = msg.get("usage") or {}
        peak = max(peak, int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0))
    return turns, peak


def ctrf_counts(trial_dir: Path) -> tuple[int | None, int | None]:
    path = trial_dir / "verifier" / "ctrf.json"
    try:
        s = json.loads(path.read_text())["results"]["summary"]
        return int(s["passed"]), int(s["tests"])
    except (OSError, KeyError, ValueError):
        return None, None


def end_reason(row: dict[str, Any]) -> str:
    traj = row["trajectory"]
    if traj.get("timeout"):
        return "agent_timeout"
    if traj.get("final_stop_reason") == "error" or traj.get("final_error_message"):
        return "endpoint_error"
    if row["error"]:
        return "other_error"
    return "completed" if traj.get("normal_completion") else "other_error"


def load_arm(job_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for job in job_dirs:
        for row in summarize_job(job.resolve())["task_results"]:
            trial = Path(row["result_path"]).parent
            result = json.loads(Path(row["result_path"]).read_text())
            turns, peak = turn_stats(trial)
            passed, total = ctrf_counts(trial)
            row.update(
                agent_s=elapsed(result.get("agent_execution")),
                verifier_s=elapsed(result.get("verifier")),
                turns=turns,
                peak_ctx=peak,
                tests_passed=passed,
                tests_total=total,
                reason=end_reason(row),
                resolved=bool(row["reward"] and row["reward"] > 0),
            )
            rows[row["task"]] = row
    return rows


def stats(values: list[float]) -> str:
    if not values:
        return "median - mean -"
    return (
        f"median {compact_seconds(statistics.median(values))} "
        f"mean {compact_seconds(statistics.mean(values))}"
    )


def print_arm(label: str, rows: dict[str, dict[str, Any]]) -> None:
    allr = list(rows.values())
    res = [r for r in allr if r["resolved"]]
    all_t = [r["agent_s"] for r in allr if r["agent_s"] is not None]
    res_t = [r["agent_s"] for r in res if r["agent_s"] is not None]
    reasons: dict[str, int] = {}
    for r in allr:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    tok_in = sum(r["input_tokens"] for r in allr)
    tok_out = sum(r["output_tokens"] for r in allr)
    print(f"== arm {label}: tasks {len(allr)} resolved {len(res)} "
          f"({len(res) / max(len(allr), 1):.3f})  ends {reasons}")
    print(f"   agent wall, all tasks: {stats(all_t)}; resolved only: {stats(res_t)}")
    print(f"   tokens: prompt {tok_in} (cached {sum(r['cache_tokens'] for r in allr)}) "
          f"completion {tok_out}; turns {sum(r['turns'] for r in allr)}; "
          f"peak ctx max {max([r['peak_ctx'] for r in allr] + [0])}")


def cell(r: dict[str, Any] | None) -> str:
    if r is None:
        return "-"
    tests = "" if r["tests_total"] is None else f" {r['tests_passed']}/{r['tests_total']}t"
    return (f"{'PASS' if r['resolved'] else 'fail'}{tests} {compact_seconds(r['agent_s'])} "
            f"{r['reason'][:8]} T{r['turns']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="comma list of job dirs for arm A")
    p.add_argument("--b", help="comma list of job dirs for arm B")
    args = p.parse_args()
    arms = {"A": load_arm([Path(x) for x in args.a.split(",")])}
    if args.b:
        arms["B"] = load_arm([Path(x) for x in args.b.split(",")])
    for label, rows in arms.items():
        print_arm(label, rows)
    names = sorted(set().union(*[set(r) for r in arms.values()]))
    print(f"\n{'TASK':34} {'A: result tests time end turns':44} B: ...")
    wins = {"A": 0, "B": 0, "tie": 0}
    for n in names:
        a, b = arms["A"].get(n), arms.get("B", {}).get(n)
        print(f"{n[:34]:34} {cell(a):44} {cell(b)}")
        if a and b:
            wins["A" if a["resolved"] > b["resolved"] else "B" if b["resolved"] > a["resolved"] else "tie"] += 1
    if "B" in arms:
        print(f"paired: A-only {wins['A']}  B-only {wins['B']}  same {wins['tie']}")


if __name__ == "__main__":
    main()
