#!/usr/bin/env python3
"""Parse and gate a replicated-MTP-embedding A-B-B-A campaign."""

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


ARMS = ("01_A1", "02_B1", "03_B2", "04_A2")


def read(path):
    return path.read_text(errors="replace")


def value(pattern, payload, label):
    match = re.search(pattern, payload, re.MULTILINE)
    if not match:
        raise ValueError(f"missing {label}")
    return float(match.group(1))


def gm(a, b):
    return math.sqrt(a * b)


def spread(a, b):
    mean = (a + b) / 2
    return 100 * abs(a - b) / mean if mean else 0


def throughput_delta(rows, key):
    return 100 * (
        gm(rows["02_B1"][key], rows["03_B2"][key])
        / gm(rows["01_A1"][key], rows["04_A2"][key])
        - 1
    )


def latency_delta(rows, key):
    return 100 * (
        gm(rows["01_A1"][key], rows["04_A2"][key])
        / gm(rows["02_B1"][key], rows["03_B2"][key])
        - 1
    )


def parse_arm(root, arm):
    directory = root / arm
    perf = read(directory / "perf_regime.log")
    soak = read(directory / "soak6400.log")
    mixed = read(directory / "mixed.log")
    pre1 = read(directory / "prefill_c1.log")
    pre4 = read(directory / "prefill_c4.log")
    code1 = read(directory / "code_c1.log")
    code4 = read(directory / "code_c4.log")
    server = read(directory / "server.log")
    info = json.loads(read(directory / "server_info.json"))
    deterministic = (directory / "deterministic.json").read_bytes()
    return {
        "perf_c1_decode": value(r"WARM\[c1\] decode=([0-9.]+)", perf, "perf c1"),
        "perf_c1_ttft_ms": value(r"WARM\[c1\].*TTFT=([0-9.]+)ms", perf, "c1 TTFT"),
        "perf_c4_stream": value(r"WARM\[c4\] decode=([0-9.]+)", perf, "c4 stream"),
        "perf_c4_agg": value(r"WARM\[c4\].*agg_out=([0-9.]+)", perf, "c4 agg"),
        "perf_c4_ttft_ms": value(r"WARM\[c4\].*TTFT=([0-9.]+)ms", perf, "c4 TTFT"),
        "soak_decode": value(r"OVERALL decode ([0-9.]+) t/s", soak, "soak"),
        "soak_ratio": value(r"first/last window ratio ([0-9.]+)x", soak, "soak ratio"),
        "prefill_c1_ttft_ms": value(r"TTFT avg=([0-9.]+)ms", pre1, "prefill c1"),
        "prefill_c4_ttft_ms": value(r"TTFT avg=([0-9.]+)ms", pre4, "prefill c4"),
        "code_c1": value(r"decode TG/stream avg=([0-9.]+)", code1, "code c1"),
        "code_c4_stream": value(r"decode TG/stream avg=([0-9.]+)", code4, "code c4 stream"),
        "code_c4_agg": value(r"\| agg=([0-9.]+)", code4, "code c4 agg"),
        "capacity": int(info["max_total_num_tokens"]),
        "deterministic_sha256": hashlib.sha256(deterministic).hexdigest(),
        "mixed_pass": "GATE PASS: all streams coherent" in mixed and "=== 24 streams:" in mixed,
        "soak_coherent": "coherence OK" in soak,
        "fatal": bool(
            re.search(
                r"device_lost|out_of_resources|ur_result_error|enginedead|!!!!|"
                r"(^|[^a-z])nan([^a-z]|$)",
                server,
                re.IGNORECASE | re.MULTILINE,
            )
        ),
    }


def main():
    root = Path(sys.argv[1]).resolve()
    rows = {arm: parse_arm(root, arm) for arm in ARMS}
    throughput_keys = (
        "perf_c1_decode",
        "soak_decode",
        "perf_c4_stream",
        "perf_c4_agg",
        "code_c1",
        "code_c4_stream",
        "code_c4_agg",
    )
    latency_keys = (
        "perf_c1_ttft_ms",
        "perf_c4_ttft_ms",
        "prefill_c1_ttft_ms",
        "prefill_c4_ttft_ms",
    )
    deltas = {f"{key}_pct": throughput_delta(rows, key) for key in throughput_keys}
    deltas.update({f"{key}_pct": latency_delta(rows, key) for key in latency_keys})
    baseline_capacity = statistics.fmean([rows["01_A1"]["capacity"], rows["04_A2"]["capacity"]])
    candidate_capacity = statistics.fmean([rows["02_B1"]["capacity"], rows["03_B2"]["capacity"]])
    hashes = {row["deterministic_sha256"] for row in rows.values()}

    checks = {
        "deterministic_byte_identical": len(hashes) == 1,
        "all_mixed_coherent": all(row["mixed_pass"] for row in rows.values()),
        "all_soaks_coherent": all(row["soak_coherent"] for row in rows.values()),
        "all_soaks_stable": all(0.95 <= row["soak_ratio"] <= 1.10 for row in rows.values()),
        "no_fatal_markers": not any(row["fatal"] for row in rows.values()),
        "capacity_ge_139264": candidate_capacity >= 139264,
        "capacity_retains_75pct": candidate_capacity >= 0.75 * baseline_capacity,
        "c1_both_pairs_win": (
            rows["02_B1"]["perf_c1_decode"] > rows["01_A1"]["perf_c1_decode"]
            and rows["03_B2"]["perf_c1_decode"] > rows["04_A2"]["perf_c1_decode"]
        ),
        "soak_both_pairs_win": (
            rows["02_B1"]["soak_decode"] > rows["01_A1"]["soak_decode"]
            and rows["03_B2"]["soak_decode"] > rows["04_A2"]["soak_decode"]
        ),
        "c1_gain_ge_2pct": deltas["perf_c1_decode_pct"] >= 2,
        "soak_gain_ge_2pct": deltas["soak_decode_pct"] >= 2,
        "code_c1_gain_ge_2pct": deltas["code_c1_pct"] >= 2,
        "c4_agg_no_regress_1pct": deltas["perf_c4_agg_pct"] >= -1,
        "code_c4_agg_no_regress_1pct": deltas["code_c4_agg_pct"] >= -1,
        "ttft_no_regress_2pct": min(deltas[f"{key}_pct"] for key in latency_keys) >= -2,
        "restart_c1_spread_le_5pct": max(
            spread(rows["01_A1"]["perf_c1_decode"], rows["04_A2"]["perf_c1_decode"]),
            spread(rows["02_B1"]["perf_c1_decode"], rows["03_B2"]["perf_c1_decode"]),
        ) <= 5,
        "restart_soak_spread_le_5pct": max(
            spread(rows["01_A1"]["soak_decode"], rows["04_A2"]["soak_decode"]),
            spread(rows["02_B1"]["soak_decode"], rows["03_B2"]["soak_decode"]),
        ) <= 5,
        "restart_c4_spread_le_8pct": max(
            spread(rows["01_A1"]["perf_c4_agg"], rows["04_A2"]["perf_c4_agg"]),
            spread(rows["02_B1"]["perf_c4_agg"], rows["03_B2"]["perf_c4_agg"]),
        ) <= 8,
    }
    passed = all(checks.values())
    summary = {
        "arms": rows,
        "balanced_deltas": deltas,
        "baseline_capacity": baseline_capacity,
        "candidate_capacity": candidate_capacity,
        "checks": checks,
        "pass": passed,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [f"VERDICT -> {'PASS' if passed else 'FAIL'}"]
    lines += [f"DELTA {key}={value:+.3f}%" for key, value in sorted(deltas.items())]
    lines += [f"CHECK {key}={'PASS' if value else 'FAIL'}" for key, value in checks.items()]
    verdict = "\n".join(lines) + "\n"
    (root / "verdict.txt").write_text(verdict)
    print(verdict, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
