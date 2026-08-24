#!/usr/bin/env python3
"""Parse and fail-closed gate experiment 07 M<=11 W8A16 A-B-B-A."""

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


ARMS = ("01_A1", "02_B1", "03_B2", "04_A2")
A_ARMS = ("01_A1", "04_A2")
B_ARMS = ("02_B1", "03_B2")
FATAL_RE = re.compile(
    r"device_lost|out_of_resources|ur_result_error|enginedead|!!!!|"
    r"segmentation fault|(^|[^a-z])nan([^a-z]|$)|missing key|"
    r"unexpected key|size mismatch",
    re.IGNORECASE | re.MULTILINE,
)
GARBAGE_RE = re.compile(r"(\S)\1{9,}")
ACCEPT_RE = re.compile(r"accept len: ([0-9.]+)")


def read(path):
    return Path(path).read_text(errors="replace")


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_manifest(path):
    return {
        key: value
        for line in read(path).splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }


def value(pattern, payload, label):
    match = re.search(pattern, payload, re.MULTILINE)
    if not match:
        raise ValueError(f"missing {label}")
    return float(match.group(1))


def gm(left, right):
    return math.sqrt(left * right)


def spread(left, right):
    mean = (left + right) / 2.0
    return 100.0 * abs(left - right) / mean if mean else 0.0


def cv(values):
    mean = statistics.fmean(values)
    if len(values) < 2 or mean == 0.0:
        return 0.0
    return 100.0 * statistics.stdev(values) / mean


def throughput_delta(rows, key):
    return 100.0 * (
        gm(rows["02_B1"][key], rows["03_B2"][key])
        / gm(rows["01_A1"][key], rows["04_A2"][key])
        - 1.0
    )


def latency_delta(rows, key):
    return 100.0 * (
        gm(rows["01_A1"][key], rows["04_A2"][key])
        / gm(rows["02_B1"][key], rows["03_B2"][key])
        - 1.0
    )


def coherent_text(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    return (
        len(text) >= 300
        and len(words) >= 60
        and len({word.lower() for word in words}) >= 25
        and GARBAGE_RE.search(text) is None
    )


def inspect_config(path, arm, manifest, repo):
    data = load_json(path)
    if not isinstance(data, list) or len(data) != 1:
        return False
    item = data[0]
    candidate = arm in B_ARMS
    threshold = "11" if candidate else "1"
    served = manifest["served_b" if candidate else "served_a"]
    env = set(item.get("Config", {}).get("Env", []))
    required = {
        "B70_XPU_W8A8=1",
        "B70_XPU_W8A8_FUSED=1",
        f"B70_W8A16_M_MAX={threshold}",
        "B70_W8A16_ROUTE_DEBUG=0",
        "B70_W8A8_QUANT_LMHEAD=0",
        "B70_XPU_REPLICATE_MTP_EMBED=1",
        "B70_XPU_DELAY_MLP_AR=0",
        "B70_XPU_FUSED_MLP_AR_NORM=0",
        "B70_XPU_PUSH_AR=1",
        "PUSH_AR_MIN_NUMEL=0",
        "PUSH_AR_GRAPH=0",
        "CCL_TOPO_P2P_ACCESS=0",
        "B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so",
        f"PUSH_AR_SO={manifest['push_ar_so']}",
    }
    mounts = item.get("Mounts", [])

    def mounted(destination, source):
        return any(
            mount.get("Destination") == destination
            and Path(str(mount.get("Source", ""))).resolve()
            == Path(source).resolve()
            and mount.get("RW") is False
            for mount in mounts
        )

    config_mounts = [
        mount
        for mount in mounts
        if str(mount.get("Destination", "")).startswith("/models/")
        and str(mount.get("Destination", "")).endswith("/config.json")
    ]
    command = " ".join(str(part) for part in item.get("Config", {}).get("Cmd", []))
    command_ok = (
        f"--model-path '{manifest['model']}'" in command
        and f"--served-model-name '{served}'" in command
        and "--tp 2" in command
        and f"--context-length {manifest['ctx']}" in command
        and f"--max-running-requests {manifest['maxreq']}" in command
        and "--disable-cuda-graph" in command
        and "--disable-radix-cache" in command
        and "--speculative-num-steps 10" in command
        and "--speculative-num-draft-tokens 11" in command
    )
    return (
        item.get("Image") == manifest["image_id"]
        and not (required - env)
        and not config_mounts
        and command_ok
        and mounted("/work/kernel", manifest["kdir"])
        and mounted("/work/push_ar", manifest["pushdir"])
        and mounted(
            "/opt/venv/lib/python3.12/site-packages/w8a8_shim.py",
            repo / "sglang/patches/w8a8_shim.py",
        )
        and mounted(
            "/opt/venv/lib/python3.12/site-packages/mtp_replicated_embedding.py",
            repo / "sglang/patches/mtp_replicated_embedding.py",
        )
        and mounted(
            "/opt/venv/lib/python3.12/site-packages/push_ar_xpu.py",
            repo / "sglang/patches/push_ar_xpu.py",
        )
    )


def server_info_exact(info, arm, manifest):
    candidate = arm in B_ARMS
    served = manifest["served_b" if candidate else "served_a"]
    return (
        info.get("status") == "ready"
        and info.get("model_path") == manifest["model"]
        and info.get("served_model_name") == served
        and int(info.get("context_length") or 0) == int(manifest["ctx"])
        and int(info.get("tp_size") or 0) == 2
        and int(info.get("pp_size") or 0) == 1
        and int(info.get("max_running_requests") or 0) == int(manifest["maxreq"])
        and info.get("disable_cuda_graph") is True
        and info.get("disable_radix_cache") is True
        and int(info.get("speculative_num_steps") or 0) == 10
        and int(info.get("speculative_num_draft_tokens") or 0) == 11
    )


def parse_arm(root, arm, manifest):
    directory = root / arm
    phase = load_json(directory / "phase_p2048_g128.json")
    phase_runs = [row for row in phase["runs"] if "error" not in row]
    if len(phase_runs) != 5:
        raise ValueError(f"{arm}: phase benchmark must contain five clean runs")
    perf = read(directory / "perf_regime.log")
    soak = read(directory / "soak6400.log")
    mixed = read(directory / "mixed.log")
    pre1 = read(directory / "prefill_c1.log")
    pre4 = read(directory / "prefill_c4.log")
    server = read(directory / "server.log")
    deterministic_bytes = (directory / "deterministic.json").read_bytes()
    deterministic = json.loads(deterministic_bytes)
    fixed_bytes = (directory / "fixed_output.json").read_bytes()
    fixed = json.loads(fixed_bytes)
    info = load_json(directory / "server_info.json")
    model = load_json(directory / "models.json")["data"][0]
    candidate = arm in B_ARMS
    expected_id = manifest["served_b" if candidate else "served_a"]
    accepted = [float(item) for item in ACCEPT_RE.findall(server)]
    threshold = 11 if candidate else 1
    return {
        "phase_decode": float(phase["median_post_first_tok_s"]),
        "phase_ttft_ms": 1000.0 * float(phase["median_ttft_s"]),
        "phase_cv_pct": cv([float(row["post_first_tok_s"]) for row in phase_runs]),
        "perf_c1_decode": value(r"WARM\[c1\] decode=([0-9.]+)", perf, "perf c1"),
        "perf_c1_ttft_ms": value(r"WARM\[c1\].*TTFT=([0-9.]+)ms", perf, "perf c1 TTFT"),
        "perf_c4_stream": value(r"WARM\[c4\] decode=([0-9.]+)", perf, "perf c4"),
        "perf_c4_agg": value(r"WARM\[c4\].*agg_out=([0-9.]+)", perf, "perf c4 agg"),
        "perf_c4_ttft_ms": value(r"WARM\[c4\].*TTFT=([0-9.]+)ms", perf, "perf c4 TTFT"),
        "soak_decode": value(r"OVERALL decode ([0-9.]+) t/s", soak, "soak"),
        "soak_ratio": value(r"first/last window ratio ([0-9.]+)x", soak, "soak ratio"),
        "prefill_c1_ttft_ms": value(r"TTFT avg=([0-9.]+)ms", pre1, "prefill c1"),
        "prefill_c4_ttft_ms": value(r"TTFT avg=([0-9.]+)ms", pre4, "prefill c4"),
        "capacity": int(info.get("max_total_num_tokens") or 0),
        "accept_count": len(accepted),
        "accept_avg": statistics.fmean(accepted) if accepted else float("nan"),
        "model_id": str(model.get("id")),
        "identity_exact": model.get("id") == expected_id
        and server_info_exact(info, arm, manifest),
        "deterministic_sha256": hashlib.sha256(deterministic_bytes).hexdigest(),
        "deterministic_nonempty": len(deterministic) == 8
        and all(
            ((row.get("reasoning_content") or "") + (row.get("content") or "")).strip()
            and int(row.get("completion_tokens") or 0) > 0
            for row in deterministic
        ),
        "fixed_sha256": hashlib.sha256(fixed_bytes).hexdigest(),
        "fixed_coherent": coherent_text(
            (fixed.get("reasoning_content") or "") + (fixed.get("content") or "")
        ),
        "mixed_pass": "GATE PASS: all streams coherent" in mixed
        and "=== 24 streams: OK=24 ===" in mixed,
        "soak_coherent": "coherence OK" in soak,
        "fatal": FATAL_RE.search(server) is not None,
        "w8a8_installed": server.count(
            f"M<={threshold}=int8_gemm_w8a16, M>{threshold}=int8_gemm_w8a8, source=env"
        )
        >= 2,
        "route_debug_silent": "[w8a8-route]" not in server,
        "replicate_markers": server.count("[mtp-replicated-embed] target ENABLED") == 2
        and server.count("[mtp-replicated-embed] draft SHARE OK") == 2,
        "forbidden_feature_marker": any(
            marker in server
            for marker in (
                "[lmhead-int8]",
                "[delayed-mlp-ar] ENABLED",
                "[fused-mlp-ar-norm]",
            )
        ),
    }


def make_performance_checks(rows, deltas):
    latency_keys = (
        "phase_ttft_ms",
        "perf_c1_ttft_ms",
        "perf_c4_ttft_ms",
        "prefill_c1_ttft_ms",
        "prefill_c4_ttft_ms",
    )
    return {
        "phase_both_pairs_win": (
            rows["02_B1"]["phase_decode"] > rows["01_A1"]["phase_decode"]
            and rows["03_B2"]["phase_decode"] > rows["04_A2"]["phase_decode"]
        ),
        "balanced_phase_gain_ge_3pct": deltas["phase_decode_pct"] >= 3.0,
        "soak_both_pairs_nonregress": (
            rows["02_B1"]["soak_decode"] >= rows["01_A1"]["soak_decode"]
            and rows["03_B2"]["soak_decode"] >= rows["04_A2"]["soak_decode"]
        ),
        "balanced_soak_gain_ge_2pct": deltas["soak_decode_pct"] >= 2.0,
        "ttft_and_prefill_within_5pct": all(
            abs(deltas[f"{key}_pct"]) <= 5.0 for key in latency_keys
        ),
        "b_within_process_cv_le_5pct": all(
            rows[arm]["phase_cv_pct"] <= 5.0 for arm in B_ARMS
        ),
        "b_restart_phase_spread_le_5pct": spread(
            rows["02_B1"]["phase_decode"], rows["03_B2"]["phase_decode"]
        )
        <= 5.0,
        "b_restart_soak_spread_le_5pct": spread(
            rows["02_B1"]["soak_decode"], rows["03_B2"]["soak_decode"]
        )
        <= 5.0,
    }


def main():
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parent.parent
    manifest = parse_manifest(root / "manifest.txt")
    rows = {arm: parse_arm(root, arm, manifest) for arm in ARMS}
    throughput_keys = (
        "phase_decode",
        "perf_c1_decode",
        "perf_c4_stream",
        "perf_c4_agg",
        "soak_decode",
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
    deltas["accept_avg_pct"] = throughput_delta(rows, "accept_avg")
    deltas["accept_avg_abs"] = gm(
        rows["02_B1"]["accept_avg"], rows["03_B2"]["accept_avg"]
    ) - gm(rows["01_A1"]["accept_avg"], rows["04_A2"]["accept_avg"])
    config_ok = {
        arm: inspect_config(root / arm / "container_inspect.json", arm, manifest, repo)
        for arm in ARMS
    }
    artifact_before = read(root / "artifacts.sha256")
    artifact_after = read(root / "artifacts_after.sha256")
    hashes = {
        parts[1]: parts[0]
        for line in artifact_before.splitlines()
        if len(parts := line.split()) == 2
    }
    health_paths = [root / "health_pre_campaign.log", root / "health_final.log"]
    for arm in ARMS:
        health_paths.extend((root / arm / "health_pre.log", root / arm / "health_post.log"))
    capacities = [rows[arm]["capacity"] for arm in ARMS]
    checks = {
        "external_dual_card_lease_proven": "LEASE_CHECK PASS cards=0,1"
        in read(root / "lease_check.txt"),
        "artifact_hashes_stable": artifact_before == artifact_after and len(hashes) >= 15,
        "all_container_configs_exact": all(config_ok.values()),
        "all_server_identities_exact": all(row["identity_exact"] for row in rows.values()),
        "all_capacities_unchanged": len(set(capacities)) == 1 and capacities[0] >= 143360,
        "all_w8a8_thresholds_installed": all(row["w8a8_installed"] for row in rows.values()),
        "route_debug_off_and_silent": all(row["route_debug_silent"] for row in rows.values()),
        "all_replicated_mtp_markers_exact": all(row["replicate_markers"] for row in rows.values()),
        "all_c3b_and_lmhead_features_off": not any(
            row["forbidden_feature_marker"] for row in rows.values()
        ),
        "all_deterministic_corpora_nonempty": all(
            row["deterministic_nonempty"] for row in rows.values()
        ),
        "all_fixed_outputs_coherent": all(row["fixed_coherent"] for row in rows.values()),
        "b_fixed_outputs_byte_identical": rows["02_B1"]["fixed_sha256"]
        == rows["03_B2"]["fixed_sha256"],
        "b_deterministic_corpora_byte_identical": rows["02_B1"]["deterministic_sha256"]
        == rows["03_B2"]["deterministic_sha256"],
        "all_96_mixed_coherent": all(row["mixed_pass"] for row in rows.values()),
        "all_soaks_coherent": all(row["soak_coherent"] for row in rows.values()),
        "all_soaks_stable": all(0.95 <= row["soak_ratio"] <= 1.10 for row in rows.values()),
        "acceptance_observed_and_finite": all(
            row["accept_count"] > 0 and math.isfinite(row["accept_avg"])
            for row in rows.values()
        ),
        "no_fatal_markers": not any(row["fatal"] for row in rows.values()),
        "all_card_health_green": all(
            path.is_file() and "xpu-health: HEALTHY (cards 0 1)" in read(path)
            for path in health_paths
        ),
        "endpoint_left_down": read(root / "endpoint_state.txt").strip() == "down",
    }
    checks.update(make_performance_checks(rows, deltas))
    passed = all(checks.values())
    summary = {
        "claim": "c4_m11_w8a16_abba_qualification",
        "arms": rows,
        "artifact_sha256": hashes,
        "balanced_deltas": deltas,
        "acceptance_diagnostic_only": True,
        "config_ok": config_ok,
        "checks": checks,
        "pass": passed,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    columns = (
        "phase_decode",
        "phase_ttft_ms",
        "phase_cv_pct",
        "perf_c1_decode",
        "perf_c4_agg",
        "soak_decode",
        "soak_ratio",
        "accept_avg",
    )
    with (root / "summary.tsv").open("w", encoding="ascii") as handle:
        handle.write("arm\t" + "\t".join(columns) + "\n")
        for arm in ARMS:
            handle.write(
                arm + "\t" + "\t".join(f"{rows[arm][column]:.4f}" for column in columns) + "\n"
            )
    lines = [f"VERDICT -> {'PASS' if passed else 'FAIL'}"]
    lines.extend(f"DELTA {key}={result:+.3f}%" if key.endswith("_pct") else f"DELTA {key}={result:+.3f}" for key, result in sorted(deltas.items()))
    lines.extend(f"CHECK {key}={'PASS' if result else 'FAIL'}" for key, result in checks.items())
    verdict = "\n".join(lines) + "\n"
    (root / "verdict.txt").write_text(verdict, encoding="ascii")
    print(verdict, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
