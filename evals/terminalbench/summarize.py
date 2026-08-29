#!/usr/bin/env python3
"""Summarize Harbor jobs for score, failures, tokens, and wall time."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")
DIRECT_EDIT_TOOLS = {"apply_patch", "edit", "patch", "write"}
SHELL_TOOLS = {"bash", "sh", "shell"}
TEST_COMMAND = re.compile(
    r"(?:"
    r"\b(?:pytest|tox|ctest)\b|"
    r"\bpython(?:3)?\s+-m\s+(?:pytest|unittest)\b|"
    r"\b(?:cargo|go)\s+test\b|"
    r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?"
    r"(?:test|check|lint|build|release)\b|"
    r"\bbun\s+(?:/app/)?dist/[^\s;|&]+"
    r")"
)
SHELL_MUTATOR = re.compile(
    r"(?:\bsed\s+-[^\s]*i\b|\bperl\s+-[^\s]*i\b|"
    r"\btee\b|\b(?:cp|mv|install)\b)"
)


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


def read_json_lines(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read complete JSONL records and fail closed on any malformed line."""
    records: list[dict[str, Any]] = []
    complete = True
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                complete = False
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                complete = False
    return records, complete


def trajectory_records(trial_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load Pi session JSONL, falling back to complete pi.txt events."""
    session_paths = sorted((trial_dir / "agent" / "pi" / "sessions").glob("*.jsonl"))
    records: list[dict[str, Any]] = []
    complete = True
    if session_paths:
        for path in session_paths:
            values, path_complete = read_json_lines(path)
            records.extend(values)
            complete = complete and path_complete
        return records, {
            "source": "session_jsonl",
            "source_paths": [str(path) for path in session_paths],
            "parse_complete": complete,
        }

    pi_path = trial_dir / "agent" / "pi.txt"
    if not pi_path.is_file():
        return [], {
            "source": "missing",
            "source_paths": [],
            "parse_complete": False,
        }

    values, complete = read_json_lines(pi_path)
    records = [value for value in values if value.get("type") == "message_end"]
    return records, {
        "source": "pi_txt",
        "source_paths": [str(pi_path)],
        "parse_complete": complete,
    }


def record_message(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("type") not in {"message", "message_end"}:
        return None
    message = record.get("message")
    return message if isinstance(message, dict) else None


def tool_target(call: dict[str, Any], task_root: str) -> str | None:
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return None
    raw = arguments.get("path") or arguments.get("file_path")
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("/"):
        return posixpath.normpath(raw)
    return posixpath.normpath(posixpath.join(task_root, raw))


def is_task_source(target: str | None, task_root: str) -> bool:
    if target is None:
        return False
    root = posixpath.normpath(task_root).rstrip("/")
    dist = root + "/dist"
    return (
        target.startswith(root + "/")
        and target != dist
        and not target.startswith(dist + "/")
    )


def shell_edit_is_ambiguous(call: dict[str, Any], task_root: str) -> bool:
    """Flag possible shell edits without claiming that an edit occurred."""
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    command = arguments.get("command")
    if not isinstance(command, str):
        return False
    mentions_root = task_root in command or re.search(
        rf"\bcd\s+{re.escape(task_root)}(?:\s|;|&|$)", command
    )
    if not mentions_root:
        return False
    if SHELL_MUTATOR.search(command):
        if not re.search(r"(?:/tmp/|\$T\b|\$TMP\b|\$TMPDIR\b)", command):
            return True
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return ">" in command
    for index, token in enumerate(tokens[:-1]):
        if token not in {">", ">>"}:
            continue
        raw_target = tokens[index + 1]
        if raw_target.startswith(("/dev/", "/tmp/", "$")):
            continue
        target = (
            posixpath.normpath(raw_target)
            if raw_target.startswith("/")
            else posixpath.normpath(posixpath.join(task_root, raw_target))
        )
        if is_task_source(target, task_root):
            return True
    return False


def is_test_call(call: dict[str, Any]) -> bool:
    name = str(call.get("name") or "").lower()
    if name in {"test", "tests"}:
        return True
    if name not in SHELL_TOOLS:
        return False
    arguments = call.get("arguments")
    command = arguments.get("command") if isinstance(arguments, dict) else None
    return isinstance(command, str) and TEST_COMMAND.search(command) is not None


def summarize_trajectory(
    trial_dir: Path,
    exception_info: dict[str, Any] | None,
    task_root: str = "/app",
) -> dict[str, Any]:
    records, source = trajectory_records(trial_dir)
    messages: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        message = record_message(record)
        if message is not None:
            messages.append((index, message))

    assistant_messages = [
        message for _, message in messages if message.get("role") == "assistant"
    ]
    final = assistant_messages[-1] if assistant_messages else {}
    final_reason = final.get("stopReason")
    raw_reason = final.get("rawStopReason")

    exception = exception_info if isinstance(exception_info, dict) else {}
    exception_type = exception.get("exception_type")
    timeout = isinstance(exception_type, str) and exception_type.endswith("TimeoutError")

    calls: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for record_index, message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if not isinstance(item, dict) or item.get("type") != "toolCall":
                continue
            call_id = item.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"record-{record_index}-content-{content_index}"
            if call_id in seen_call_ids:
                continue
            seen_call_ids.add(call_id)
            calls.append(
                {
                    "id": call_id,
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "record_index": record_index,
                }
            )

    results: dict[str, dict[str, Any]] = {}
    for _, message in messages:
        if message.get("role") != "toolResult":
            continue
        call_id = message.get("toolCallId")
        if isinstance(call_id, str):
            results[call_id] = message

    edit_attempts: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    ambiguous_shell_calls: list[str] = []
    for call in calls:
        name = str(call.get("name") or "").lower()
        if name in DIRECT_EDIT_TOOLS:
            target = tool_target(call, task_root)
            evidence = {
                "tool_call_id": call["id"],
                "tool": call.get("name"),
                "target": target,
            }
            edit_attempts.append(evidence)
            result = results.get(call["id"])
            if (
                is_task_source(target, task_root)
                and result is not None
                and result.get("isError") is False
            ):
                edits.append({**evidence, "record_index": call["record_index"]})
        elif name in SHELL_TOOLS and shell_edit_is_ambiguous(call, task_root):
            ambiguous_shell_calls.append(call["id"])

    test_evidence: list[dict[str, Any]] = []
    if edits:
        last_edit_index = max(edit["record_index"] for edit in edits)
        for call in calls:
            if call["record_index"] <= last_edit_index or not is_test_call(call):
                continue
            result = results.get(call["id"])
            if result is None:
                status = "unknown"
            elif result.get("isError") is False:
                status = "passed"
            elif result.get("isError") is True:
                status = "failed"
            else:
                status = "unknown"
            arguments = call.get("arguments")
            command = arguments.get("command") if isinstance(arguments, dict) else None
            test_evidence.append(
                {
                    "tool_call_id": call["id"],
                    "command": command,
                    "status": status,
                }
            )

    if any(item["status"] == "passed" for item in test_evidence):
        post_edit_status = "passed"
    elif any(item["status"] == "failed" for item in test_evidence):
        post_edit_status = "failed"
    elif test_evidence:
        post_edit_status = "unknown"
    else:
        post_edit_status = "none"

    length_stop = final_reason == "length" or raw_reason == "length"
    return {
        **source,
        "final_stop_reason": final_reason,
        "raw_stop_reason": raw_reason,
        "final_error_message": final.get("errorMessage"),
        "normal_completion": final_reason == "stop" and not exception,
        "length_stop": length_stop,
        "timeout": timeout,
        "timeout_type": exception_type if timeout else None,
        "timeout_message": exception.get("exception_message") if timeout else None,
        "tool_count": len(calls),
        "edit_attempt_count": len(edit_attempts),
        "edit_attempts": edit_attempts,
        "confirmed_edit_count": len(edits),
        "confirmed_edits": [
            {key: value for key, value in edit.items() if key != "record_index"}
            for edit in edits
        ],
        "edit_occurred": bool(edits),
        "edit_classification_complete": not ambiguous_shell_calls,
        "ambiguous_shell_tool_call_ids": ambiguous_shell_calls,
        "post_edit_test": {
            "attempted": bool(test_evidence),
            "passed": (
                True if post_edit_status == "passed"
                else False if post_edit_status == "failed"
                else None
            ),
            "status": post_edit_status,
            "attempt_count": len(test_evidence),
            "evidence": test_evidence,
        },
    }


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
    normal_completions = 0
    timeouts = 0
    length_stops = 0
    tool_calls = 0
    edit_trials = 0
    post_edit_test_trials = 0
    post_edit_test_passes = 0
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
        result_path = Path(result["_result_path"])
        trajectory = summarize_trajectory(
            result_path.parent,
            result.get("exception_info"),
        )
        normal_completions += int(trajectory["normal_completion"])
        timeouts += int(trajectory["timeout"])
        length_stops += int(trajectory["length_stop"])
        tool_calls += int(trajectory["tool_count"])
        edit_trials += int(trajectory["edit_occurred"])
        post_edit_test = trajectory["post_edit_test"]
        post_edit_test_trials += int(post_edit_test["attempted"])
        post_edit_test_passes += int(post_edit_test["passed"] is True)
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
                "trajectory": trajectory,
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
        "normal_completions": normal_completions,
        "timeouts": timeouts,
        "length_stops": length_stops,
        "tool_calls": tool_calls,
        "edit_trials": edit_trials,
        "post_edit_test_trials": post_edit_test_trials,
        "post_edit_test_passes": post_edit_test_passes,
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
        f"{'ERR':>3} {'NORM':>4} {'TIME':>4} {'LEN':>3} {'TOOLS':>5} "
        f"{'EDIT':>4} {'PTEST':>7} {'JOB_WALL':>10} {'TOTAL':>10} "
        f"{'AGENT_SUM':>10} {'TOKENS':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        score = item["reward_mean"]
        score_text = "-" if score is None else f"{score:.4f}"
        agent_sum = item["phase_seconds"]["agent_execution"]["sum_seconds"]
        lifecycle = item.get("lifecycle") or {}
        total = lifecycle.get("full_machine_seconds")
        if total is None:
            total = lifecycle.get("end_to_end_seconds")
        tokens = item["input_tokens"] + item["output_tokens"]
        post_tests = (
            f"{item['post_edit_test_passes']}/{item['post_edit_test_trials']}"
        )
        print(
            f"{item['job'][:36]:36} {item['tasks']:5d} {score_text:>9} "
            f"{item['passes']:5d} {item['errors']:3d} "
            f"{item['normal_completions']:4d} {item['timeouts']:4d} "
            f"{item['length_stops']:3d} {item['tool_calls']:5d} "
            f"{item['edit_trials']:4d} {post_tests:>7} "
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
