#!/usr/bin/env python3
"""Summarize Harbor jobs for score, failures, tokens, and wall time."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def elapsed(section: dict[str, Any] | None) -> float | None:
    if not section:
        return None
    start = parse_time(section.get("started_at"))
    finish = parse_time(section.get("finished_at"))
    if start is None or finish is None:
        return None
    return (finish - start).total_seconds()


def trial_results(job_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(job_dir.glob("*/result.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        result["_result_path"] = str(path)
        results.append(result)
    return results


def summarize_job(job_dir: Path) -> dict[str, Any]:
    trials = trial_results(job_dir)
    if not trials:
        raise ValueError(f"no trial result.json files found under {job_dir}")

    starts = [parse_time(item.get("started_at")) for item in trials]
    finishes = [parse_time(item.get("finished_at")) for item in trials]
    valid_starts = [item for item in starts if item is not None]
    valid_finishes = [item for item in finishes if item is not None]
    rewards: list[float] = []
    input_tokens = 0
    cache_tokens = 0
    output_tokens = 0
    models: set[str] = set()
    errors = 0
    completed = 0
    phase_values: dict[str, list[float]] = {name: [] for name in PHASES}
    task_rows: list[dict[str, Any]] = []

    for result in trials:
        model = (
            result.get("agent_info", {})
            .get("model_info", {})
            .get("name")
        )
        if model:
            models.add(model)
        agent_result = result.get("agent_result") or {}
        input_tokens += int(agent_result.get("n_input_tokens") or 0)
        cache_tokens += int(agent_result.get("n_cache_tokens") or 0)
        output_tokens += int(agent_result.get("n_output_tokens") or 0)
        verifier_result = result.get("verifier_result") or {}
        reward = (verifier_result.get("rewards") or {}).get("reward")
        if reward is not None:
            rewards.append(float(reward))
        if result.get("finished_at"):
            completed += 1
        if result.get("exception_info"):
            errors += 1
        trial_phases: dict[str, float | None] = {}
        for phase in PHASES:
            seconds = elapsed(result.get(phase))
            trial_phases[phase] = seconds
            if seconds is not None:
                phase_values[phase].append(seconds)
        task_rows.append(
            {
                "task": result.get("task_name"),
                "reward": reward,
                "error": bool(result.get("exception_info")),
                "input_tokens": int(agent_result.get("n_input_tokens") or 0),
                "cache_tokens": int(agent_result.get("n_cache_tokens") or 0),
                "output_tokens": int(agent_result.get("n_output_tokens") or 0),
                "wall_seconds": elapsed(result),
                "phase_seconds": trial_phases,
                "result_path": result["_result_path"],
            }
        )

    phase_summary = {
        name: {
            "sum_seconds": sum(values),
            "median_seconds": statistics.median(values) if values else None,
        }
        for name, values in phase_values.items()
    }
    job_wall = None
    if valid_starts and valid_finishes:
        job_wall = (max(valid_finishes) - min(valid_starts)).total_seconds()
    lifecycle_path = job_dir / "b70_lifecycle.json"
    lifecycle = None
    if lifecycle_path.is_file():
        with lifecycle_path.open(encoding="utf-8") as handle:
            lifecycle = json.load(handle)

    return {
        "job": job_dir.name,
        "job_dir": str(job_dir),
        "models": sorted(models),
        "tasks": len(trials),
        "completed": completed,
        "errors": errors,
        "scored": len(rewards),
        "passes": sum(value > 0 for value in rewards),
        "reward_sum": sum(rewards),
        "reward_mean": statistics.mean(rewards) if rewards else None,
        "job_wall_seconds": job_wall,
        "lifecycle": lifecycle,
        "input_tokens": input_tokens,
        "cache_tokens": cache_tokens,
        "output_tokens": output_tokens,
        "phase_seconds": phase_summary,
        "task_results": task_rows,
    }


def compact_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    seconds = int(round(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_table(summaries: list[dict[str, Any]]) -> None:
    header = (
        f"{'JOB':36} {'TASKS':>5} {'SCORE':>9} {'PASS':>5} "
        f"{'ERR':>3} {'JOB_WALL':>10} {'TOTAL':>10} {'AGENT_SUM':>10} {'TOKENS':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        score = item["reward_mean"]
        score_text = "-" if score is None else f"{score:.4f}"
        agent_sum = item["phase_seconds"]["agent_execution"]["sum_seconds"]
        lifecycle = item.get("lifecycle") or {}
        total = lifecycle.get("end_to_end_seconds")
        tokens = item["input_tokens"] + item["output_tokens"]
        print(
            f"{item['job'][:36]:36} {item['tasks']:5d} {score_text:>9} "
            f"{item['passes']:5d} {item['errors']:3d} "
            f"{compact_seconds(item['job_wall_seconds']):>10} "
            f"{compact_seconds(total):>10} "
            f"{compact_seconds(agent_sum):>10} {tokens:10d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dirs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    summaries = [summarize_job(path.resolve()) for path in args.job_dirs]
    if args.as_json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        print_table(summaries)


if __name__ == "__main__":
    main()
