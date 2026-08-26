#!/usr/bin/env python3
"""Summarize June XPU timing steps and optional CUDAGraph replay traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


STEP_MARKER = "[vllm-xpu-timing-step] "


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_steps(path: Path) -> tuple[list[dict], list[str]]:
    steps: list[dict] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        marker_index = line.find(STEP_MARKER)
        if marker_index < 0:
            continue
        payload = line[marker_index + len(STEP_MARKER) :]
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            errors.append(f"server-log:{line_number}:{exc}")
            continue
        record["source_line"] = line_number
        steps.append(record)
    return steps, errors


def _summarize_labels(steps: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for step in steps:
        for row in step.get("summary_by_total_ms", []):
            grouped[(str(row.get("rank", "")), str(row.get("label", "")))].append(
                row
            )

    result = []
    for (rank, label), rows in grouped.items():
        totals = [float(row["total_ms"]) for row in rows]
        calls = [int(row["count"]) for row in rows]
        result.append(
            {
                "rank": rank,
                "label": label,
                "step_count": len(rows),
                "mean_calls_per_step": _mean([float(value) for value in calls]),
                "mean_total_ms_per_step": _mean(totals),
                "median_total_ms_per_step": _median(totals),
                "p90_total_ms_per_step": _percentile(totals, 0.90),
                "max_total_ms_per_step": max(totals),
            }
        )
    result.sort(key=lambda row: row["mean_total_ms_per_step"] or 0.0, reverse=True)
    return result


def _pure_decode_steps(steps: list[dict]) -> list[dict]:
    return [
        step
        for step in steps
        if bool((step.get("metadata") or {}).get("is_pure_decode"))
    ]


def _summarize_buckets(steps: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for step in steps:
        metadata = step.get("metadata") or {}
        key = (
            str(step.get("rank", "")),
            str(metadata.get("cudagraph_mode", "")),
            bool(metadata.get("is_pure_decode")),
            metadata.get("decode_bucket"),
            int(metadata.get("num_reqs", 0)),
            bool(metadata.get("skip_compiled")),
        )
        grouped[key].append(step)

    result = []
    for key, bucket_steps in grouped.items():
        forward_values = []
        visible_values = []
        for step in bucket_steps:
            visible = 0.0
            for row in step.get("summary_by_total_ms", []):
                total = float(row["total_ms"])
                visible += total
                if row.get("label") == "gpu_model_runner.model_forward":
                    forward_values.append(total)
            visible_values.append(visible)
        rank, mode, pure_decode, decode_bucket, num_reqs, skip_compiled = key
        result.append(
            {
                "rank": rank,
                "cudagraph_mode": mode,
                "is_pure_decode": pure_decode,
                "decode_bucket": decode_bucket,
                "num_reqs": num_reqs,
                "skip_compiled": skip_compiled,
                "step_count": len(bucket_steps),
                "mean_model_forward_ms": _mean(forward_values),
                "median_model_forward_ms": _median(forward_values),
                "p90_model_forward_ms": _percentile(forward_values, 0.90),
                "mean_nonexclusive_visible_ms": _mean(visible_values),
            }
        )
    result.sort(key=lambda row: row["step_count"], reverse=True)
    return result


def _load_replay_traces(paths: list[Path]) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}:{exc}")
                continue
            record["source_file"] = str(path)
            record["source_line"] = line_number
            records.append(record)
    return records, errors


def _summarize_replays(records: list[dict]) -> dict:
    by_event: dict[str, int] = defaultdict(int)
    by_piece: dict[tuple, int] = defaultdict(int)
    starts_by_request: dict[tuple, int] = defaultdict(int)
    for record in records:
        event = str(record.get("event", ""))
        by_event[event] += 1
        if event != "replay_start":
            continue
        piece_key = (
            str(record.get("tp_rank", "")),
            record.get("piecewise_index"),
            record.get("total_piecewise_compiles"),
            str(record.get("submod_name", "")),
            str(record.get("batch_descriptor", "")),
        )
        by_piece[piece_key] += 1
        request_ids = tuple(str(value) for value in record.get("req_ids", []))
        request_key = (
            str(record.get("tp_rank", "")),
            request_ids,
            str(record.get("batch_descriptor", "")),
        )
        starts_by_request[request_key] += 1

    pieces = [
        {
            "rank": key[0],
            "piecewise_index": key[1],
            "total_piecewise_compiles": key[2],
            "submod_name": key[3],
            "batch_descriptor": key[4],
            "replay_start_count": count,
        }
        for key, count in by_piece.items()
    ]
    pieces.sort(
        key=lambda row: (
            row["rank"],
            str(row["batch_descriptor"]),
            -1 if row["piecewise_index"] is None else int(row["piecewise_index"]),
        )
    )
    starts = list(starts_by_request.values())
    total_piecewise_values = sorted(
        {
            int(row[2])
            for row in by_piece
            if row[2] is not None
        }
    )
    piecewise_indices = sorted(
        {
            int(row[1])
            for row in by_piece
            if row[1] is not None
        }
    )
    replay_counts = sorted(by_piece.values())
    return {
        "record_count": len(records),
        "events": dict(sorted(by_event.items())),
        "replay_start_by_piece": pieces,
        "piecewise_topology": {
            "reported_total_piecewise_compiles": total_piecewise_values,
            "observed_piecewise_indices": piecewise_indices,
            "observed_piece_count": len(piecewise_indices),
            "min_replay_starts_per_piece": (
                replay_counts[0] if replay_counts else None
            ),
            "max_replay_starts_per_piece": (
                replay_counts[-1] if replay_counts else None
            ),
        },
        "request_batch_groups": len(starts),
        "replay_starts_per_request_batch": {
            "mean": _mean([float(value) for value in starts]),
            "median": _median([float(value) for value in starts]),
            "min": min(starts) if starts else None,
            "max": max(starts) if starts else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--timing-sync", choices=("0", "1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("replay_traces", type=Path, nargs="*")
    args = parser.parse_args()

    steps, step_errors = _load_steps(args.server_log)
    pure_decode_steps = _pure_decode_steps(steps)
    replay_records, replay_errors = _load_replay_traces(args.replay_traces)
    document = {
        "protocol": "qwen36-xpu-decode-timing-summary-v1",
        "timing_semantics": (
            "synchronized device-completion diagnostic; endpoint throughput is perturbed"
            if args.timing_sync == "1"
            else "unsynchronized host enqueue timing; not kernel duration"
        ),
        "timing_sync": args.timing_sync == "1",
        "server_log": str(args.server_log),
        "step_count": len(steps),
        "pure_decode_step_count": len(pure_decode_steps),
        "parse_errors": step_errors + replay_errors,
        "pure_decode_labels_by_mean_total_ms": _summarize_labels(
            pure_decode_steps
        ),
        "all_step_labels_by_mean_total_ms": _summarize_labels(steps),
        "step_buckets": _summarize_buckets(steps),
        "cudagraph_replay": _summarize_replays(replay_records),
        "nested_labels_are_nonexclusive": True,
        "label_scope_note": (
            "Use pure_decode_labels_by_mean_total_ms for decode conclusions. "
            "all_step_labels_by_mean_total_ms also includes sampled prefill "
            "steps and is retained only for forensic completeness."
        ),
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, sort_keys=True))
    return 0 if steps and not document["parse_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
