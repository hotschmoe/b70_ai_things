#!/usr/bin/env python3
"""Summarize Harbor jobs as verifier-correct completion over wall time."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds_between(timing: dict[str, Any] | None) -> float:
    if not timing:
        return 0.0
    start = parse_time(timing.get("started_at"))
    finish = parse_time(timing.get("finished_at"))
    return max(0.0, (finish - start).total_seconds()) if start and finish else 0.0


def trial_reward(trial: dict[str, Any]) -> float | None:
    result = trial.get("verifier_result") or {}
    rewards = result.get("rewards") or {}
    if not rewards:
        return None
    for key in ("reward", "taskgen_grader", "all"):
        if key in rewards and isinstance(rewards[key], (int, float)):
            return float(rewards[key])
    numeric = [float(value) for value in rewards.values() if isinstance(value, (int, float))]
    return numeric[0] if len(numeric) == 1 else (min(numeric) if numeric else None)


def agent_contexts(trial: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(trial.get("agent_result"), dict):
        return [trial["agent_result"]]
    return [
        step["agent_result"]
        for step in (trial.get("step_results") or [])
        if isinstance(step.get("agent_result"), dict)
    ]


def load_trials(job_dir: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
    embedded = job.get("trial_results")
    if isinstance(embedded, list) and embedded:
        return embedded
    trials: list[dict[str, Any]] = []
    for path in sorted(job_dir.glob("*/result.json")):
        data = json.loads(path.read_text())
        if "task_name" in data:
            trials.append(data)
    return trials


def summarize(job_dir: Path, curve_dir: Path | None = None) -> dict[str, Any]:
    job = json.loads((job_dir / "result.json").read_text())
    trials = load_trials(job_dir, job)
    start = parse_time(job.get("started_at"))
    if start is None:
        starts = [parse_time(t.get("started_at")) for t in trials]
        start = min(t for t in starts if t is not None)
    finish = parse_time(job.get("finished_at"))
    if finish is None:
        finishes = [parse_time(t.get("finished_at")) for t in trials]
        finish = max((t for t in finishes if t is not None), default=start)

    total = int(job.get("n_total_trials") or len(trials))
    completed: list[tuple[float, bool, dict[str, Any], float | None]] = []
    errors = 0
    rewards: list[float] = []
    agent_seconds = verifier_seconds = 0.0
    tokens = {"input": 0, "cache": 0, "output": 0}
    token_seen = {key: False for key in tokens}

    for trial in trials:
        end = parse_time(trial.get("finished_at"))
        if end is None:
            continue
        reward = trial_reward(trial)
        if reward is not None:
            rewards.append(reward)
        correct = reward is not None and reward >= 1.0 - 1e-9
        completed.append((max(0.0, (end - start).total_seconds()), correct, trial, reward))
        errors += int(trial.get("exception_info") is not None)
        agent_seconds += seconds_between(trial.get("agent_execution"))
        verifier_seconds += seconds_between(trial.get("verifier"))
        for context in agent_contexts(trial):
            for output_key, input_key in (
                ("input", "n_input_tokens"),
                ("cache", "n_cache_tokens"),
                ("output", "n_output_tokens"),
            ):
                value = context.get(input_key)
                if isinstance(value, int):
                    tokens[output_key] += value
                    token_seen[output_key] = True

    completed.sort(key=lambda item: item[0])
    correct_count = 0
    curve = [{"wall_seconds": 0.0, "correct": 0, "correct_pct": 0.0, "finished": 0}]
    for index, (elapsed, correct, _, _) in enumerate(completed, 1):
        correct_count += int(correct)
        curve.append(
            {
                "wall_seconds": elapsed,
                "correct": correct_count,
                "correct_pct": 100.0 * correct_count / total if total else 0.0,
                "finished": index,
            }
        )

    wall_seconds = max(0.0, (finish - start).total_seconds()) if finish else 0.0
    if curve and wall_seconds > curve[-1]["wall_seconds"]:
        curve.append({**curve[-1], "wall_seconds": wall_seconds})
    area = 0.0
    for left, right in zip(curve, curve[1:]):
        area += left["correct_pct"] * (right["wall_seconds"] - left["wall_seconds"])
    normalized_auc_pct = area / wall_seconds if wall_seconds else 0.0

    thresholds: dict[str, float | None] = {}
    for pct in (10, 25, 50):
        needed = math.ceil(total * pct / 100.0)
        point = next((row for row in curve if row["correct"] >= needed), None)
        thresholds[f"time_to_{pct}pct_correct_seconds"] = point["wall_seconds"] if point else None

    label = job_dir.name
    if curve_dir is not None:
        curve_dir.mkdir(parents=True, exist_ok=True)
        with (curve_dir / f"{label}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)

    eligible = max(0, total - errors)
    return {
        "job": label,
        "path": str(job_dir.resolve()),
        "n_total": total,
        "n_finished": len(completed),
        "n_correct": correct_count,
        "n_errors": errors,
        "correct_pct_raw": 100.0 * correct_count / total if total else 0.0,
        "correct_pct_model_only": 100.0 * correct_count / eligible if eligible else 0.0,
        "mean_reward_finished": sum(rewards) / len(rewards) if rewards else None,
        "wall_seconds": wall_seconds,
        "agent_seconds": agent_seconds,
        "verifier_seconds": verifier_seconds,
        "correct_per_wall_hour": 3600.0 * correct_count / wall_seconds if wall_seconds else 0.0,
        "normalized_correct_auc_pct": normalized_auc_pct,
        "tokens": {key: (value if token_seen[key] else None) for key, value in tokens.items()},
        **thresholds,
        "curve": curve,
    }


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value / 3600.0:.2f}h"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--curve-dir", type=Path)
    args = parser.parse_args()
    rows = [summarize(path, args.curve_dir) for path in args.jobs]
    print("| job | correct | wall | correct/hour | AUC | errors | agent | verifier |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['job']} | {row['n_correct']}/{row['n_total']} "
            f"({row['correct_pct_raw']:.1f}%) | {fmt_seconds(row['wall_seconds'])} | "
            f"{row['correct_per_wall_hour']:.2f} | {row['normalized_correct_auc_pct']:.1f}% | "
            f"{row['n_errors']} | {fmt_seconds(row['agent_seconds'])} | "
            f"{fmt_seconds(row['verifier_seconds'])} |"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
