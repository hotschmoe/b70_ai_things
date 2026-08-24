#!/usr/bin/env python3
"""Parse and conservatively gate a C3b fused-boundary A-B-B-A campaign."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


ARMS = ("01_A1", "02_B1", "03_B2", "04_A2")
BASELINE_ARMS = ("01_A1", "04_A2")
CANDIDATE_ARMS = ("02_B1", "03_B2")
FATAL_RE = re.compile(
    r"device_lost|out_of_resources|ur_result_error|enginedead|!!!!|"
    r"(^|[^a-z])nan([^a-z]|$)",
    re.IGNORECASE | re.MULTILINE,
)
ROUTE_RE = re.compile(
    r"\[c3b-delayed-mlp\] ROUTES rank=(\d+) eligible=(\d+) "
    r"consumed=(\d+) generic=(\d+)"
)
M11_RE = re.compile(r"\[c3b-fused\] calls=(\d+) rows=11 hidden=5120")


def read(path: Path) -> str:
    return path.read_text(errors="replace")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def value(pattern: str, payload: str, label: str) -> float:
    found = re.search(pattern, payload, re.MULTILINE)
    if not found:
        raise ValueError(f"missing {label}")
    return float(found.group(1))


def gm(a: float, b: float) -> float:
    return math.sqrt(a * b)


def spread(a: float, b: float) -> float:
    mean = (a + b) / 2.0
    return 100.0 * abs(a - b) / mean if mean else 0.0


def cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return 100.0 * statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0


def throughput_delta(rows: dict[str, dict], key: str) -> float:
    return 100.0 * (
        gm(rows["02_B1"][key], rows["03_B2"][key])
        / gm(rows["01_A1"][key], rows["04_A2"][key])
        - 1.0
    )


def latency_delta(rows: dict[str, dict], key: str) -> float:
    return 100.0 * (
        gm(rows["01_A1"][key], rows["04_A2"][key])
        / gm(rows["02_B1"][key], rows["03_B2"][key])
        - 1.0
    )


def parse_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in read(path).splitlines():
        if "=" in line:
            key, item = line.split("=", 1)
            result[key] = item
    return result


def inspect_config(path: Path, expected_fused: bool, expected_source: str,
                   expected_so: str) -> bool:
    data = load_json(path)
    if not isinstance(data, list) or len(data) != 1:
        return False
    item = data[0]
    env = set(item.get("Config", {}).get("Env", []))
    flag = "1" if expected_fused else "0"
    required = {
        "B70_XPU_PUSH_AR=1",
        "B70_XPU_REPLICATE_MTP_EMBED=1",
        f"B70_XPU_DELAY_MLP_AR={flag}",
        f"B70_XPU_FUSED_MLP_AR_NORM={flag}",
        f"PUSH_AR_SO={expected_so}",
        "PUSH_AR_MIN_NUMEL=1048576",
        "PUSH_AR_GRAPH=0",
        "CCL_TOPO_P2P_ACCESS=0",
    }
    mounts = item.get("Mounts", [])
    mounted_source = any(
        mount.get("Destination") == "/work/push_ar"
        and str(Path(str(mount.get("Source", ""))).resolve())
        == str(Path(expected_source).resolve())
        and mount.get("RW") is False
        for mount in mounts
    )
    return not (required - env) and mounted_source


def mechanism(path: Path) -> dict:
    payload = read(path)
    routes = [tuple(map(int, row)) for row in ROUTE_RE.findall(payload)]
    latest = {}
    for rank, eligible, consumed, generic in routes:
        latest[rank] = {
            "eligible": eligible,
            "consumed": consumed,
            "generic": generic,
        }
    m11_calls = [int(item) for item in M11_RE.findall(payload)]
    return {
        "latest_routes": latest,
        "m11_calls": m11_calls,
        "install_hits": payload.count("[c3b-fused] installed"),
        "delayed_install_hits": payload.count("[c3b-delayed-mlp] installed"),
        "has_any_c3b": "[c3b-" in payload,
        "m11_generic_zero": (
            set(latest) == {0, 1}
            and all(
                row["eligible"] == row["consumed"]
                and row["generic"] == 0
                and row["consumed"] >= 4096
                for row in latest.values()
            )
            and len(m11_calls) >= 2
            and max(m11_calls, default=0) >= 4096
        ),
    }


def parse_arm(root: Path, arm: str) -> dict:
    directory = root / arm
    phase = load_json(directory / "phase_p2048_g128.json")
    phase_runs = [row for row in phase["runs"] if "error" not in row]
    perf = read(directory / "perf_regime.log")
    soak = read(directory / "soak6400.log")
    mechanism_soak = read(directory / "mechanism_soak.log")
    mixed = read(directory / "mixed.log")
    pre1 = read(directory / "prefill_c1.log")
    pre4 = read(directory / "prefill_c4.log")
    code1 = read(directory / "code_c1.log")
    code4 = read(directory / "code_c4.log")
    server = read(directory / "server.log")
    info = load_json(directory / "server_info.json")
    deterministic_bytes = (directory / "deterministic.json").read_bytes()
    deterministic = json.loads(deterministic_bytes)
    model = load_json(directory / "models.json")["data"][0]
    return {
        "phase_decode": float(phase["median_post_first_tok_s"]),
        "phase_ttft_ms": 1000.0 * float(phase["median_ttft_s"]),
        "phase_cv_pct": cv([float(row["post_first_tok_s"]) for row in phase_runs]),
        "perf_c1_decode": value(r"WARM\[c1\] decode=([0-9.]+)", perf, "perf c1"),
        "perf_c1_ttft_ms": value(
            r"WARM\[c1\].*TTFT=([0-9.]+)ms", perf, "perf c1 TTFT"
        ),
        "perf_c4_stream": value(r"WARM\[c4\] decode=([0-9.]+)", perf, "perf c4"),
        "perf_c4_agg": value(r"WARM\[c4\].*agg_out=([0-9.]+)", perf, "perf c4 agg"),
        "perf_c4_ttft_ms": value(
            r"WARM\[c4\].*TTFT=([0-9.]+)ms", perf, "perf c4 TTFT"
        ),
        "soak_decode": value(r"OVERALL decode ([0-9.]+) t/s", soak, "soak"),
        "soak_ratio": value(r"first/last window ratio ([0-9.]+)x", soak, "soak ratio"),
        "prefill_c1_ttft_ms": value(r"TTFT avg=([0-9.]+)ms", pre1, "prefill c1"),
        "prefill_c4_ttft_ms": value(r"TTFT avg=([0-9.]+)ms", pre4, "prefill c4"),
        "code_c1": value(r"decode TG/stream avg=([0-9.]+)", code1, "code c1"),
        "code_c4_stream": value(
            r"decode TG/stream avg=([0-9.]+)", code4, "code c4 stream"
        ),
        "code_c4_agg": value(r"\| agg=([0-9.]+)", code4, "code c4 agg"),
        "capacity": int(info["max_total_num_tokens"]),
        "model_id": str(model["id"]),
        "deterministic_sha256": hashlib.sha256(deterministic_bytes).hexdigest(),
        "deterministic_nonempty": len(deterministic) == 8 and all(
            ((row.get("reasoning_content") or "") + (row.get("content") or "")).strip()
            and int(row.get("completion_tokens") or 0) > 0
            for row in deterministic
        ),
        "mixed_pass": "GATE PASS: all streams coherent" in mixed and "=== 24 streams:" in mixed,
        "soak_coherent": "coherence OK" in soak,
        "mechanism_soak_coherent": "coherence OK" in mechanism_soak,
        "fatal": FATAL_RE.search(server) is not None,
        "mechanism": mechanism(directory / "mechanism.log"),
    }


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    manifest = parse_manifest(root / "manifest.txt")
    rows = {arm: parse_arm(root, arm) for arm in ARMS}
    throughput_keys = (
        "phase_decode",
        "perf_c1_decode",
        "soak_decode",
        "perf_c4_stream",
        "perf_c4_agg",
        "code_c1",
        "code_c4_stream",
        "code_c4_agg",
    )
    latency_keys = (
        "phase_ttft_ms",
        "perf_c1_ttft_ms",
        "perf_c4_ttft_ms",
        "prefill_c1_ttft_ms",
        "prefill_c4_ttft_ms",
    )
    deltas = {f"{key}_pct": throughput_delta(rows, key) for key in throughput_keys}
    deltas.update({f"{key}_pct": latency_delta(rows, key) for key in latency_keys})

    artifact_before = read(root / "artifacts.sha256")
    artifact_after = read(root / "artifacts_after.sha256")
    hashes = {line.split()[1]: line.split()[0] for line in artifact_before.splitlines()}
    deterministic_hashes = {row["deterministic_sha256"] for row in rows.values()}
    baseline_capacity = statistics.fmean(rows[arm]["capacity"] for arm in BASELINE_ARMS)
    candidate_capacity = statistics.fmean(rows[arm]["capacity"] for arm in CANDIDATE_ARMS)

    health_paths = [root / "health_pre_campaign.log"]
    for arm in ARMS:
        health_paths.extend((root / arm / "health_pre.log", root / arm / "health_post.log"))
    all_health_green = all(
        path.is_file() and "xpu-health: HEALTHY (cards 0 1)" in read(path)
        for path in health_paths
    )

    config_ok = {}
    for arm in BASELINE_ARMS:
        config_ok[arm] = inspect_config(
            root / arm / "container_inspect.json",
            False,
            manifest["base_pushdir"],
            manifest["base_push_ar_so"],
        )
    for arm in CANDIDATE_ARMS:
        config_ok[arm] = inspect_config(
            root / arm / "container_inspect.json",
            True,
            manifest["candidate_pushdir"],
            manifest["candidate_push_ar_so"],
        )

    checks = {
        "artifact_hashes_stable": artifact_before == artifact_after and len(hashes) == 8,
        "fast_max_rows_contract_11": manifest.get("candidate_fast_max_rows") == "11",
        "all_arm_configs_exact": all(config_ok.values()),
        "all_model_ids_exact": all(row["model_id"] == manifest["served"] for row in rows.values()),
        "deterministic_nonempty_8": all(row["deterministic_nonempty"] for row in rows.values()),
        "deterministic_byte_identical": len(deterministic_hashes) == 1,
        "baseline_has_no_c3b": all(
            not rows[arm]["mechanism"]["has_any_c3b"] for arm in BASELINE_ARMS
        ),
        "candidate_m11_fused_generic_zero": all(
            rows[arm]["mechanism"]["m11_generic_zero"] for arm in CANDIDATE_ARMS
        ),
        "candidate_installed_both_processes": all(
            rows[arm]["mechanism"]["install_hits"] >= 2
            and rows[arm]["mechanism"]["delayed_install_hits"] >= 2
            for arm in CANDIDATE_ARMS
        ),
        "all_mixed_coherent": all(row["mixed_pass"] for row in rows.values()),
        "all_soaks_coherent": all(row["soak_coherent"] for row in rows.values()),
        "all_mechanism_soaks_coherent": all(
            row["mechanism_soak_coherent"] for row in rows.values()
        ),
        "all_soaks_stable": all(0.95 <= row["soak_ratio"] <= 1.10 for row in rows.values()),
        "no_fatal_markers": not any(row["fatal"] for row in rows.values()),
        "all_card_health_green": all_health_green,
        "endpoint_left_down": read(root / "endpoint_state_before_analysis.txt").strip() == "down",
        "capacity_ge_139264": candidate_capacity >= 139264,
        "capacity_retains_99pct": candidate_capacity >= 0.99 * baseline_capacity,
        "phase_c1_both_pairs_win": (
            rows["02_B1"]["phase_decode"] > rows["01_A1"]["phase_decode"]
            and rows["03_B2"]["phase_decode"] > rows["04_A2"]["phase_decode"]
        ),
        "perf_c1_both_pairs_win": (
            rows["02_B1"]["perf_c1_decode"] > rows["01_A1"]["perf_c1_decode"]
            and rows["03_B2"]["perf_c1_decode"] > rows["04_A2"]["perf_c1_decode"]
        ),
        "soak_both_pairs_win": (
            rows["02_B1"]["soak_decode"] > rows["01_A1"]["soak_decode"]
            and rows["03_B2"]["soak_decode"] > rows["04_A2"]["soak_decode"]
        ),
        "phase_c1_gain_ge_2pct": deltas["phase_decode_pct"] >= 2.0,
        "perf_c1_gain_ge_2pct": deltas["perf_c1_decode_pct"] >= 2.0,
        "soak_gain_ge_2pct": deltas["soak_decode_pct"] >= 2.0,
        "code_c1_gain_ge_1pct": deltas["code_c1_pct"] >= 1.0,
        "c4_stream_no_regress_1_5pct": deltas["perf_c4_stream_pct"] >= -1.5,
        "c4_agg_no_regress_1_5pct": deltas["perf_c4_agg_pct"] >= -1.5,
        "code_c4_agg_no_regress_1_5pct": deltas["code_c4_agg_pct"] >= -1.5,
        "prefill_no_regress_1_5pct": min(
            deltas["prefill_c1_ttft_ms_pct"], deltas["prefill_c4_ttft_ms_pct"]
        ) >= -1.5,
        "all_ttft_no_regress_2_5pct": min(
            deltas[f"{key}_pct"] for key in latency_keys
        ) >= -2.5,
        "within_process_cv_le_5pct": all(row["phase_cv_pct"] <= 5.0 for row in rows.values()),
        "restart_c1_spread_le_5pct": max(
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
    summary = {
        "claim": "fused_boundary_serving_qualification",
        "arms": rows,
        "artifact_sha256": hashes,
        "balanced_deltas": deltas,
        "baseline_capacity": baseline_capacity,
        "candidate_capacity": candidate_capacity,
        "config_ok": config_ok,
        "checks": checks,
        "pass": passed,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )

    keys = (
        "phase_decode", "phase_ttft_ms", "perf_c1_decode", "perf_c4_agg",
        "soak_decode", "soak_ratio", "code_c1", "code_c4_agg", "phase_cv_pct",
    )
    with (root / "summary.tsv").open("w", encoding="ascii") as handle:
        handle.write("arm\t" + "\t".join(keys) + "\n")
        for arm in ARMS:
            handle.write(
                arm + "\t" + "\t".join(f"{rows[arm][key]:.4f}" for key in keys) + "\n"
            )

    lines = [f"VERDICT -> {'PASS' if passed else 'FAIL'}"]
    lines.extend(f"DELTA {key}={item:+.3f}%" for key, item in sorted(deltas.items()))
    lines.extend(f"CHECK {key}={'PASS' if item else 'FAIL'}" for key, item in checks.items())
    verdict = "\n".join(lines) + "\n"
    (root / "verdict.txt").write_text(verdict, encoding="ascii")
    print(verdict, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
