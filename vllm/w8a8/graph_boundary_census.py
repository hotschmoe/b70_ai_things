#!/usr/bin/env python3
"""Build a fail-closed per-token graph and collective boundary census."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
import re
import statistics
from typing import Any


STRUCTURAL_EVENTS = {
    "fences": "zeFenceReset",
    "host_waits": "zeEventHostSynchronize",
    "submissions": "zeCommandQueueExecuteCommandLists",
}
COLLECTIVE_NAMES = ("all_reduce", "all_gather", "reduce_scatter")
GRAPH_CALL = re.compile(
    r':\s*"(?P<type>[^"]+)"\s*=\s*torch[.]ops[.]'
    r"(?P<op>[A-Za-z0-9_.]+)"
)
TENSOR_TYPE = re.compile(r"^(?P<dtype>[A-Za-z0-9_]+)\[(?P<shape>[^]]+)\]$")


def load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    events = payload.get("traceEvents", payload)
    if not isinstance(events, list):
        raise ValueError(f"trace has no event list: {path}")
    return events


def parse_rank_path(value: str) -> tuple[int, Path]:
    rank_text, separator, path_text = value.partition("=")
    if not separator or not rank_text.isdigit() or not path_text:
        raise argparse.ArgumentTypeError("expected RANK=PATH")
    return int(rank_text), Path(path_text)


def contains(interval: dict[str, Any], event: dict[str, Any]) -> bool:
    if interval.get("pid") != event.get("pid"):
        return False
    timestamp = float(event.get("ts", -1.0))
    start = float(interval["ts"])
    return start <= timestamp <= start + float(interval.get("dur", 0.0))


def trace_census(path: Path, skip_iterations: int) -> dict[str, Any]:
    events = [event for event in load_trace(path) if event.get("ph") == "X"]
    iterations = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and str(event.get("name", "")).startswith("execute_context_")
        ),
        key=lambda event: float(event["ts"]),
    )
    if len(iterations) <= skip_iterations:
        raise ValueError(
            f"{path}: {len(iterations)} iterations cannot satisfy "
            f"skip_iterations={skip_iterations}"
        )

    rows = []
    for index, iteration in enumerate(iterations):
        owned_driver = [
            event
            for event in events
            if event.get("cat") == "xpu_driver"
            and event.get("tid") == iteration.get("tid")
            and contains(iteration, event)
        ]
        counts = {
            field: sum(event.get("name") == event_name for event in owned_driver)
            for field, event_name in STRUCTURAL_EVENTS.items()
        }
        rows.append(
            {
                "iteration_index": index,
                "annotation": str(iteration.get("name")),
                "cpu_range_us": float(iteration.get("dur", 0.0)),
                "graph_pieces": counts["fences"],
                **counts,
            }
        )

    steady = rows[skip_iterations:]
    signatures = [
        tuple(row[field] for field in ("graph_pieces", *STRUCTURAL_EVENTS))
        for row in steady
    ]
    signature_counts = collections.Counter(signatures)
    signature, signature_occurrences = signature_counts.most_common(1)[0]
    stable = len(signature_counts) == 1
    fields = ("graph_pieces", *STRUCTURAL_EVENTS)
    return {
        "trace": str(path),
        "iterations_total": len(rows),
        "iterations_skipped": skip_iterations,
        "target_token_iterations": len(steady),
        "stable_signature": stable,
        "signature_occurrences": signature_occurrences,
        "per_target_token": dict(zip(fields, signature)),
        "cpu_range_us": {
            "minimum": min(row["cpu_range_us"] for row in steady),
            "median": statistics.median(row["cpu_range_us"] for row in steady),
            "maximum": max(row["cpu_range_us"] for row in steady),
        },
        "observed_signatures": [
            {**dict(zip(fields, observed)), "iterations": count}
            for observed, count in sorted(signature_counts.items())
        ],
        "per_iteration": rows,
        "interpretation_guard": (
            "graph_pieces uses zeFenceReset as the replay-piece signature; "
            "XPUGraph kernels and collectives may remain opaque to Kineto"
        ),
    }


def normalize_dimension(value: str, decode_rows: int) -> int | str:
    value = value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return decode_rows


def compiled_collective_census(path: Path, decode_rows: int) -> dict[str, Any]:
    calls: collections.Counter[tuple[str, str, tuple[int | str, ...]]] = (
        collections.Counter()
    )
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = GRAPH_CALL.search(line)
        if match is None:
            continue
        operator = match.group("op")
        collective = next(
            (name for name in COLLECTIVE_NAMES if name in operator), None
        )
        if collective is None or "wait_tensor" in operator:
            continue
        tensor = TENSOR_TYPE.match(match.group("type"))
        if tensor is None:
            raise ValueError(
                f"{path}:{line_number}: collective output lacks one tensor type"
            )
        shape = tuple(
            normalize_dimension(dimension, decode_rows)
            for dimension in tensor.group("shape").split(",")
        )
        calls[(collective, tensor.group("dtype"), shape)] += 1
    if not calls:
        raise ValueError(f"no compiled collective calls found in {path}")
    return {
        "compiled_graph": str(path),
        "decode_rows": decode_rows,
        "collective_calls_per_target_token": sum(calls.values()),
        "collectives": [
            {
                "collective": collective,
                "dtype": dtype,
                "shape": list(shape),
                "count": count,
            }
            for (collective, dtype, shape), count in sorted(calls.items())
        ],
    }


def compact_collectives(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report["collectives"]


def build_report(
    traces: dict[int, Path],
    graphs: dict[int, Path],
    *,
    skip_iterations: int,
    decode_rows: int,
    control_value: float | None,
    profiled_value: float | None,
    minimum_profiled_ratio: float,
) -> dict[str, Any]:
    if sorted(traces) != [0, 1] or sorted(graphs) != [0, 1]:
        raise ValueError("exactly ranks 0 and 1 are required for traces and graphs")
    trace_reports = {
        rank: trace_census(path, skip_iterations) for rank, path in traces.items()
    }
    graph_reports = {
        rank: compiled_collective_census(path, decode_rows)
        for rank, path in graphs.items()
    }
    structural_agreement = (
        trace_reports[0]["stable_signature"]
        and trace_reports[1]["stable_signature"]
        and trace_reports[0]["per_target_token"]
        == trace_reports[1]["per_target_token"]
    )
    collective_agreement = compact_collectives(
        graph_reports[0]
    ) == compact_collectives(graph_reports[1])

    overhead = None
    overhead_passed = True
    if (control_value is None) != (profiled_value is None):
        raise ValueError("control and profiled values must be supplied together")
    if control_value is not None and profiled_value is not None:
        if control_value <= 0 or profiled_value <= 0:
            raise ValueError("control and profiled values must be positive")
        ratio = profiled_value / control_value
        overhead_passed = ratio >= minimum_profiled_ratio
        overhead = {
            "metric_direction": "higher_is_better",
            "control": control_value,
            "profiled": profiled_value,
            "profiled_over_control": ratio,
            "maximum_fractional_loss": 1.0 - minimum_profiled_ratio,
            "passed": overhead_passed,
        }

    passed = structural_agreement and collective_agreement and overhead_passed
    return {
        "protocol": "b70-graph-boundary-census-v1",
        "passed": passed,
        "rank_structural_agreement": structural_agreement,
        "rank_collective_agreement": collective_agreement,
        "traces": {str(rank): report for rank, report in trace_reports.items()},
        "compiled_graphs": {
            str(rank): report for rank, report in graph_reports.items()
        },
        "instrumentation_overhead": overhead,
        "interpretation_guard": (
            "This census describes the profiled target-only C1 decode token "
            "and compiled graph source. It is not an endpoint speed result."
        ),
    }


def unique_rank_paths(values: list[tuple[int, Path]], label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for rank, path in values:
        if rank in result:
            raise ValueError(f"duplicate {label} rank {rank}")
        if not path.is_file():
            raise ValueError(f"missing {label} for rank {rank}: {path}")
        result[rank] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", type=parse_rank_path, required=True)
    parser.add_argument(
        "--compiled-graph", action="append", type=parse_rank_path, required=True
    )
    parser.add_argument("--skip-iterations", type=int, default=2)
    parser.add_argument("--decode-rows", type=int, default=1)
    parser.add_argument("--control-value", type=float)
    parser.add_argument("--profiled-value", type=float)
    parser.add_argument("--minimum-profiled-ratio", type=float, default=0.75)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.skip_iterations < 0:
        parser.error("--skip-iterations must be nonnegative")
    if args.decode_rows <= 0:
        parser.error("--decode-rows must be positive")
    if not 0 < args.minimum_profiled_ratio <= 1:
        parser.error("--minimum-profiled-ratio must be in (0, 1]")

    try:
        report = build_report(
            unique_rank_paths(args.trace, "trace"),
            unique_rank_paths(args.compiled_graph, "compiled graph"),
            skip_iterations=args.skip_iterations,
            decode_rows=args.decode_rows,
            control_value=args.control_value,
            profiled_value=args.profiled_value,
            minimum_profiled_ratio=args.minimum_profiled_ratio,
        )
    except ValueError as error:
        parser.error(str(error))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    structural = report["traces"]["0"]["per_target_token"]
    collectives = report["compiled_graphs"]["0"]
    print(
        "CENSUS -> "
        f"pieces={structural['graph_pieces']} fences={structural['fences']} "
        f"waits={structural['host_waits']} submissions={structural['submissions']} "
        f"collectives={collectives['collective_calls_per_target_token']}"
    )
    print(
        "RANKS -> "
        f"structural={report['rank_structural_agreement']} "
        f"collectives={report['rank_collective_agreement']}"
    )
    if report["instrumentation_overhead"] is not None:
        print(
            "OVERHEAD -> profiled_over_control="
            f"{report['instrumentation_overhead']['profiled_over_control']:.6f} "
            f"passed={report['instrumentation_overhead']['passed']}"
        )
    print(f"VERDICT -> {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
