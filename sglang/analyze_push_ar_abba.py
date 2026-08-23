#!/usr/bin/env python3
"""Parse and gate one campaign_push_ar_abba.sh result directory."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path


ARMS = ("01_A1", "02_B1", "03_B2", "04_A2")


def text(path: Path) -> str:
    return path.read_text(errors="replace")


def match_float(pattern: str, value: str, label: str) -> float:
    found = re.search(pattern, value, re.MULTILINE)
    if not found:
        raise ValueError(f"missing {label}")
    return float(found.group(1))


def cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return 100.0 * statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0


def spread(a: float, b: float) -> float:
    mean = (a + b) / 2.0
    return 100.0 * abs(a - b) / mean if mean else 0.0


def gm(a: float, b: float) -> float:
    return math.sqrt(a * b)


def throughput_delta(rows: dict[str, dict], key: str) -> float:
    return 100.0 * (gm(rows["02_B1"][key], rows["03_B2"][key]) /
                    gm(rows["01_A1"][key], rows["04_A2"][key]) - 1.0)


def latency_delta(rows: dict[str, dict], key: str) -> float:
    return 100.0 * (gm(rows["01_A1"][key], rows["04_A2"][key]) /
                    gm(rows["02_B1"][key], rows["03_B2"][key]) - 1.0)


def parse_arm(root: Path, arm: str) -> dict:
    d = root / arm
    phase = json.loads(text(d / "phase_p2048_g128.json"))
    runs = [r for r in phase["runs"] if "error" not in r]
    perf = text(d / "perf_regime.log")
    soak = text(d / "soak6400.log")
    mixed = text(d / "mixed.log")
    pre1 = text(d / "prefill_c1.log")
    pre4 = text(d / "prefill_c4.log")
    code1 = text(d / "code_c1.log")
    code4 = text(d / "code_c4.log")
    server = text(d / "server.log")
    row = {
        "phase_decode": float(phase["median_post_first_tok_s"]),
        "phase_ttft_ms": 1000.0 * float(phase["median_ttft_s"]),
        "phase_cv_pct": cv([float(r["post_first_tok_s"]) for r in runs]),
        "perf_c1_decode": match_float(r"WARM\[c1\] decode=([0-9.]+)", perf, "perf c1"),
        "perf_c1_ttft_ms": match_float(r"WARM\[c1\].*TTFT=([0-9.]+)ms", perf, "perf c1 TTFT"),
        "perf_c4_stream": match_float(r"WARM\[c4\] decode=([0-9.]+)", perf, "perf c4 stream"),
        "perf_c4_agg": match_float(r"WARM\[c4\].*agg_out=([0-9.]+)", perf, "perf c4 agg"),
        "perf_c4_ttft_ms": match_float(r"WARM\[c4\].*TTFT=([0-9.]+)ms", perf, "perf c4 TTFT"),
        "soak_decode": match_float(r"OVERALL decode ([0-9.]+) t/s", soak, "extended soak"),
        "soak_ratio": match_float(r"first/last window ratio ([0-9.]+)x", soak, "soak ratio"),
        "prefill_c1_ttft_ms": match_float(r"TTFT avg=([0-9.]+)ms", pre1, "prefill c1 TTFT"),
        "prefill_c4_ttft_ms": match_float(r"TTFT avg=([0-9.]+)ms", pre4, "prefill c4 TTFT"),
        "code_c1": match_float(r"decode TG/stream avg=([0-9.]+)", code1, "code c1"),
        "code_c4_stream": match_float(r"decode TG/stream avg=([0-9.]+)", code4, "code c4 stream"),
        "code_c4_agg": match_float(r"\| agg=([0-9.]+)", code4, "code c4 agg"),
        "mixed_pass": "GATE PASS: all streams coherent" in mixed and "=== 24 streams:" in mixed,
        "soak_coherent": "coherence OK" in soak,
        "fatal": bool(re.search(
            r"device_lost|out_of_resources|ur_result_error|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)",
            server,
            re.IGNORECASE | re.MULTILINE,
        )),
    }
    return row


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    rows = {arm: parse_arm(root, arm) for arm in ARMS}
    deltas = {
        "phase_decode_pct": throughput_delta(rows, "phase_decode"),
        "perf_c1_decode_pct": throughput_delta(rows, "perf_c1_decode"),
        "soak_decode_pct": throughput_delta(rows, "soak_decode"),
        "perf_c4_stream_pct": throughput_delta(rows, "perf_c4_stream"),
        "perf_c4_agg_pct": throughput_delta(rows, "perf_c4_agg"),
        "code_c1_pct": throughput_delta(rows, "code_c1"),
        "code_c4_stream_pct": throughput_delta(rows, "code_c4_stream"),
        "code_c4_agg_pct": throughput_delta(rows, "code_c4_agg"),
        "phase_ttft_pct": latency_delta(rows, "phase_ttft_ms"),
        "perf_c1_ttft_pct": latency_delta(rows, "perf_c1_ttft_ms"),
        "perf_c4_ttft_pct": latency_delta(rows, "perf_c4_ttft_ms"),
        "prefill_c1_ttft_pct": latency_delta(rows, "prefill_c1_ttft_ms"),
        "prefill_c4_ttft_pct": latency_delta(rows, "prefill_c4_ttft_ms"),
    }

    checks = {
        "all_mixed_coherent": all(r["mixed_pass"] for r in rows.values()),
        "all_soaks_coherent": all(r["soak_coherent"] for r in rows.values()),
        "all_soaks_stable": all(0.95 <= r["soak_ratio"] <= 1.10 for r in rows.values()),
        "no_fatal_markers": not any(r["fatal"] for r in rows.values()),
        "phase_c1_both_pairs_win": (
            rows["02_B1"]["phase_decode"] > rows["01_A1"]["phase_decode"]
            and rows["03_B2"]["phase_decode"] > rows["04_A2"]["phase_decode"]
        ),
        "soak_both_pairs_win": (
            rows["02_B1"]["soak_decode"] > rows["01_A1"]["soak_decode"]
            and rows["03_B2"]["soak_decode"] > rows["04_A2"]["soak_decode"]
        ),
        "phase_c1_gain_ge_3pct": deltas["phase_decode_pct"] >= 3.0,
        "soak_gain_ge_3pct": deltas["soak_decode_pct"] >= 3.0,
        "c4_stream_no_regress_2pct": deltas["perf_c4_stream_pct"] >= -2.0,
        "c4_agg_no_regress_2pct": deltas["perf_c4_agg_pct"] >= -2.0,
        "code_c4_agg_no_regress_2pct": deltas["code_c4_agg_pct"] >= -2.0,
        "ttft_no_regress_3pct": min(
            deltas["phase_ttft_pct"], deltas["perf_c1_ttft_pct"],
            deltas["perf_c4_ttft_pct"], deltas["prefill_c1_ttft_pct"],
            deltas["prefill_c4_ttft_pct"],
        ) >= -3.0,
        "within_process_cv_le_5pct": all(r["phase_cv_pct"] <= 5.0 for r in rows.values()),
        "restart_phase_spread_le_5pct": max(
            spread(rows["01_A1"]["phase_decode"], rows["04_A2"]["phase_decode"]),
            spread(rows["02_B1"]["phase_decode"], rows["03_B2"]["phase_decode"]),
        ) <= 5.0,
        "restart_soak_spread_le_5pct": max(
            spread(rows["01_A1"]["soak_decode"], rows["04_A2"]["soak_decode"]),
            spread(rows["02_B1"]["soak_decode"], rows["03_B2"]["soak_decode"]),
        ) <= 5.0,
        "restart_c4_spread_le_8pct": max(
            spread(rows["01_A1"]["perf_c4_agg"], rows["04_A2"]["perf_c4_agg"]),
            spread(rows["02_B1"]["perf_c4_agg"], rows["03_B2"]["perf_c4_agg"]),
        ) <= 8.0,
    }
    passed = all(checks.values())
    summary = {"arms": rows, "balanced_deltas": deltas, "checks": checks, "pass": passed}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    keys = ("phase_decode", "phase_ttft_ms", "perf_c1_decode", "perf_c4_agg",
            "soak_decode", "soak_ratio", "phase_cv_pct")
    with (root / "summary.tsv").open("w") as f:
        f.write("arm\t" + "\t".join(keys) + "\n")
        for arm in ARMS:
            f.write(arm + "\t" + "\t".join(f"{rows[arm][k]:.4f}" for k in keys) + "\n")

    lines = [f"VERDICT -> {'PASS' if passed else 'FAIL'}"]
    lines.extend(f"DELTA {key}={value:+.3f}%" for key, value in sorted(deltas.items()))
    lines.extend(f"CHECK {key}={'PASS' if value else 'FAIL'}" for key, value in checks.items())
    verdict = "\n".join(lines) + "\n"
    (root / "verdict.txt").write_text(verdict)
    print(verdict, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
