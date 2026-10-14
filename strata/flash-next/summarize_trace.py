#!/usr/bin/env python3
"""Summarize unitrace Chrome events without adding overlapping GPU times.

Keep the raw trace and unitrace submission report. This summary is diagnostic;
benchmark the same binary separately with tracing disabled.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def union_us(events):
    end = None
    total = 0.0
    for start, stop in sorted((e["ts"], e["ts"] + e["dur"]) for e in events):
        if end is None or stop > end:
            total += stop - max(start, end if end is not None else start)
        end = max(end if end is not None else stop, stop)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-us", type=float)
    parser.add_argument("--end-us", type=float)
    parser.add_argument("--timestamp-shift-us", type=float, default=0,
                        help="Explicit trace-to-client clock correction; retain its calibration evidence")
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text())
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    process_names = {e["pid"]: e["args"]["name"] for e in events
                     if e.get("ph") == "M" and e.get("name") == "process_name"}
    thread_names = {(e["pid"], e["tid"]): e["args"]["name"] for e in events
                    if e.get("ph") == "M" and e.get("name") == "thread_name"}
    groups = defaultdict(list)
    gpu = defaultdict(list)
    selected = []
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        event = dict(event, ts=event["ts"] + args.timestamp_shift_us)
        if args.start_us is not None and event["ts"] < args.start_us:
            continue
        if args.end_us is not None and event["ts"] + event["dur"] > args.end_us:
            continue
        selected.append(event)
        kind = "gpu" if event.get("cat") == "gpu_op" else "host"
        process = process_names.get(event["pid"], str(event["pid"]))
        thread = thread_names.get((event["pid"], event["tid"]), str(event["tid"]))
        groups[(kind, process, thread, event["name"])].append(event["dur"])
        if kind == "gpu":
            gpu[process].append(event)
    if not gpu:
        raise RuntimeError("No GPU events in selected interval: verify collection, flush and clock alignment")
    rows = []
    for (kind, process, thread, name), values in groups.items():
        rows.append(dict(kind=kind, process=process, engine_or_thread=thread,
                         operation=name, count=len(values), total_ms=sum(values)/1000,
                         mean_us=statistics.mean(values), median_us=statistics.median(values),
                         p95_us=percentile(values, .95), min_us=min(values), max_us=max(values)))
    rows.sort(key=lambda row: (row["kind"], -row["total_ms"]))
    args.output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output / "operations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(trace=str(args.trace), start_us=args.start_us, end_us=args.end_us,
                   timestamp_shift_us=args.timestamp_shift_us,
                   complete_events=len(selected), operations=rows,
                   gpu_busy_union_ms={name: union_us(items)/1000 for name, items in gpu.items()},
                   note="Diagnostic trace; per-device unions include overlapping engines once. "
                        "Do not sum device times or nested host ranges into request wall time. "
                        "Transfer bytes/direction and submission delays remain in the raw unitrace reports.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "operations"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
