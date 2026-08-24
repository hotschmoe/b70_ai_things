#!/usr/bin/env python3
"""Fail-closed analyzer for the C4 GDN out_proj-only INT8 mechanism."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from parse_tp2_math_census import trace_census


SERVED = "qwen36-27b-W8A8-sqgptq-GDN-OUT-RTN-mtp-c4-mechanism"
MODEL = "/models/qwen3.6-27b/w8a8-sqgptq-gdn-out-proj-int8"
BASELINE_CAPACITY = 143360
FATAL_RE = re.compile(
    r"device_lost|out_of_resources|ur_result_error|enginedead|!!!!|"
    r"segmentation fault|(^|[^a-z])nan([^a-z]|$)|missing key|"
    r"unexpected key|size mismatch",
    re.IGNORECASE | re.MULTILINE,
)
ACCEPT_RE = re.compile(r"accept len: ([0-9.]+), accept rate: ([0-9.]+)")
GARBAGE_RE = re.compile(r"(\S)\1{9,}")
SOAK_RE = re.compile(
    r"OVERALL decode ([0-9.]+) t/s \| first/last window ratio "
    r"([0-9.]+)x .* \| coherence ([A-Z]+)"
)
ROUTE_OPS = re.compile(
    r"^(aten::mm|_xpu_C::int8_gemm_w8a8|"
    r"_xpu_C::dynamic_per_token_int8_quant)$"
)


def read(path: Path) -> str:
    return path.read_text(errors="replace")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in read(path).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def longest_zero_run(rates: list[float]) -> int:
    longest = 0
    current = 0
    for rate in rates:
        if rate <= 0.01:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def coherent_text(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    return (
        len(text) >= 500
        and len(words) >= 100
        and len({word.lower() for word in words}) >= 30
        and GARBAGE_RE.search(text) is None
    )


def tensor_dims_match(dims, first, second, second_index=1):
    return (
        len(dims) > second_index
        and dims[0] == tuple(first)
        and dims[second_index] == tuple(second)
    )


def calls(rows, name, first, second, second_index=1):
    return sum(
        values["calls"]
        for (op_name, dims), values in rows.items()
        if op_name == name
        and tensor_dims_match(dims, first, second, second_index)
    )


def one_tensor_calls(rows, name, shape):
    return sum(
        values["calls"]
        for (op_name, dims), values in rows.items()
        if op_name == name and dims and dims[0] == tuple(shape)
    )


def route_summary(trace_path: Path):
    rows, _total_device_us, _matched_device_us = trace_census(
        trace_path, ROUTE_OPS
    )
    summary = {
        "w8a8_m11_qkvz": calls(
            rows,
            "_xpu_C::int8_gemm_w8a8",
            (11, 5120),
            (5120, 8192),
            3,
        ),
        "w8a8_m11_out_all": calls(
            rows,
            "_xpu_C::int8_gemm_w8a8",
            (11, 3072),
            (3072, 5120),
            3,
        ),
        "bf16_m11_qkvz": calls(
            rows, "aten::mm", (11, 5120), (5120, 8192)
        ),
        "bf16_m11_out_mtp": calls(
            rows, "aten::mm", (11, 3072), (3072, 5120)
        ),
        "bf16_m11_ba": calls(
            rows, "aten::mm", (11, 5120), (5120, 48)
        ),
        "bf16_m11_mtp_qkv": calls(
            rows, "aten::mm", (11, 5120), (5120, 7168)
        ),
        "quant_m11_k5120": one_tensor_calls(
            rows, "_xpu_C::dynamic_per_token_int8_quant", (11, 5120)
        ),
        "quant_m11_k3072": one_tensor_calls(
            rows, "_xpu_C::dynamic_per_token_int8_quant", (11, 3072)
        ),
    }
    checks = {
        "w8a8_m11_qkvz_absent": summary["w8a8_m11_qkvz"] == 0,
        "w8a8_m11_out_64_per_step": summary["w8a8_m11_out_all"] == 320,
        "bf16_m11_qkvz_48_per_step": summary["bf16_m11_qkvz"] == 240,
        "only_mtp_bf16_m11_out_remains": summary["bf16_m11_out_mtp"] == 5,
        "bf16_m11_ba_preserved": summary["bf16_m11_ba"] == 240,
        "bf16_m11_mtp_qkv_preserved": summary["bf16_m11_mtp_qkv"] == 5,
        "quant_m11_k5120_80_per_step": summary["quant_m11_k5120"] == 400,
        "quant_m11_k3072_64_per_step": summary["quant_m11_k3072"] == 320,
    }
    return summary, checks


def inspect_exact(path: Path, root: Path, image_id: str) -> bool:
    data = load_json(path)
    if not isinstance(data, list) or len(data) != 1:
        return False
    item = data[0]
    env = set(item.get("Config", {}).get("Env", []))
    required = {
        "B70_XPU_W8A8=1",
        "B70_XPU_W8A8_FUSED=1",
        "B70_W8A8_QUANT_LMHEAD=0",
        "B70_XPU_REPLICATE_MTP_EMBED=1",
        "B70_XPU_DELAY_MLP_AR=0",
        "B70_XPU_FUSED_MLP_AR_NORM=0",
        "B70_XPU_PUSH_AR=1",
        "PUSH_AR_MIN_NUMEL=0",
        "PUSH_AR_GRAPH=0",
        "CCL_TOPO_P2P_ACCESS=0",
        "B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so",
        "PUSH_AR_SO=/work/push_ar/libxpu_push_ar_graph.so",
    }
    mounts = item.get("Mounts", [])

    def mounted(destination: str, source: Path, read_only=True) -> bool:
        return any(
            mount.get("Destination") == destination
            and Path(str(mount.get("Source", ""))).resolve() == source.resolve()
            and (not read_only or mount.get("RW") is False)
            for mount in mounts
        )

    config_source = root / "candidate_config.json"
    command = " ".join(item.get("Config", {}).get("Cmd", []) or [])
    return (
        not (required - env)
        and item.get("Image") == image_id
        and item.get("HostConfig", {}).get("RestartPolicy", {}).get("Name")
        in ("", "no")
        and f"--model-path '{MODEL}'" in command
        and f"--served-model-name '{SERVED}'" in command
        and mounted(f"{MODEL}/config.json", config_source)
        and mounted(
            "/opt/venv/lib/python3.12/site-packages/w8a8_shim.py",
            Path(__file__).resolve().parent / "patches/w8a8_shim.py",
        )
        and mounted("/work/kernel", Path("/mnt/vm_8tb/b70/w8a8_kernel"))
    )


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    manifest = parse_manifest(root / "manifest.txt")
    audit = load_json(root / "checkpoint_audit.json")
    overlay = load_json(root / "candidate_config.json")
    server_info = load_json(root / "server_info_after.json")
    model = load_json(root / "models.json")["data"][0]
    deterministic_1 = load_json(root / "deterministic_1.json")
    deterministic_2 = load_json(root / "deterministic_2.json")
    fixed = load_json(root / "fixed_generation.json")
    server = read(root / "server.log")
    fixed_delta = read(root / "fixed_server_delta.log")

    trace_paths = sorted(
        Path(manifest["profile_dir"]).glob("*DECODE.trace.json.gz")
    )
    route_results = []
    route_checks = []
    for path in trace_paths:
        summary, checks = route_summary(path)
        route_results.append({"trace": str(path), "counts": summary})
        route_checks.append(checks)

    acceptance = [
        (float(length), float(rate))
        for length, rate in ACCEPT_RE.findall(fixed_delta)
    ]
    rates = [rate for _length, rate in acceptance]
    lengths = [length for length, _rate in acceptance]
    internal_accept = [
        float(state.get("avg_spec_accept_length") or 0.0)
        for state in server_info.get("internal_states", [])
    ]
    fixed_message = fixed["choices"][0]["message"]
    fixed_text = (fixed_message.get("reasoning_content") or "") + (
        fixed_message.get("content") or ""
    )
    fixed_tokens = int(fixed.get("usage", {}).get("completion_tokens") or 0)
    soak_match = SOAK_RE.search(read(root / "soak.log"))
    soak_rate = float(soak_match.group(1)) if soak_match else 0.0
    soak_degradation = float(soak_match.group(2)) if soak_match else 999.0
    soak_coherence = soak_match.group(3) if soak_match else "MISSING"

    quant = overlay.get("quantization_config", {})
    expected_ignore = {
        "lm_head",
        r"re:.*linear_attn\.in_proj_qkv$",
        r"re:.*linear_attn\.in_proj_z$",
        r"re:.*linear_attn\.in_proj_b$",
        r"re:.*linear_attn\.in_proj_a$",
        r"re:.*visual.*",
        r"re:.*mtp.*",
    }
    exact_info = (
        server_info.get("status") == "ready"
        and server_info.get("model_path") == MODEL
        and server_info.get("served_model_name") == SERVED
        and int(server_info.get("context_length") or 0) == 131072
        and int(server_info.get("tp_size") or 0) == 2
        and int(server_info.get("pp_size") or 0) == 1
        and int(server_info.get("max_running_requests") or 0) == 4
        and server_info.get("disable_cuda_graph") is True
        and server_info.get("disable_radix_cache") is True
        and int(server_info.get("speculative_num_steps") or 0) == 10
        and int(server_info.get("speculative_num_draft_tokens") or 0) == 11
    )
    capacity = int(server_info.get("max_total_num_tokens") or 0)
    checks = {
        "external_dual_card_lease_proven": "LEASE_CHECK PASS cards=0,1"
        in read(root / "lease_check.txt"),
        "checkpoint_audit_pass": audit.get("status") == "PASS",
        "checkpoint_exact_weight_count": audit.get("target_gdn_int8_weights")
        == 48,
        "checkpoint_exact_scale_count": audit.get("target_gdn_int8_scales")
        == 48,
        "checkpoint_exact_tp2_saving": audit.get(
            "tp2_runtime_bytes_saved_per_rank"
        )
        == 754_483_200,
        "overlay_exact_ignore": set(quant.get("ignore") or [])
        == expected_ignore,
        "container_config_exact": inspect_exact(
            root / "container_inspect.json", root, manifest["image_id"]
        ),
        "server_config_exact": exact_info,
        "served_model_exact": model.get("id") == SERVED,
        "capacity_nonregression": capacity >= BASELINE_CAPACITY,
        "fused_w8a8_installed_both_ranks": server.count(
            "[w8a8-shim] installed: FUSED hybrid"
        )
        >= 2,
        "exactly_two_decode_traces": len(trace_paths) == 2,
        "all_route_checks_pass": len(route_checks) == 2
        and all(all(item.values()) for item in route_checks),
        "fixed_generation_ge_512_tokens": fixed_tokens >= 512,
        "fixed_generation_coherent": coherent_text(fixed_text),
        "acceptance_has_samples": len(acceptance) >= 2,
        "acceptance_rate_ge_020": max(rates, default=0.0) >= 0.20,
        "acceptance_len_ge_3": max(lengths, default=0.0) >= 3.0,
        "acceptance_no_three_zero_run": longest_zero_run(rates) <= 2,
        "server_acceptance_average_ge_3": min(internal_accept, default=0.0)
        >= 3.0,
        "deterministic_same_process_byte_exact": deterministic_1
        == deterministic_2,
        "deterministic_corpus_has_8": len(deterministic_1) == 8,
        "mixed_24_of_24_coherent": (
            "=== 24 streams:" in read(root / "mixed.log")
            and "GATE PASS: all streams coherent" in read(root / "mixed.log")
        ),
        "soak_has_positive_rate": soak_rate > 0.0,
        "soak_stable": soak_degradation <= 1.25,
        "soak_coherent": soak_coherence == "OK",
        "no_fatal_server_markers": FATAL_RE.search(server) is None,
        "artifacts_unchanged": read(root / "artifacts.sha256")
        == read(root / "artifacts_after.sha256"),
        "health_pre_green": "xpu-health: HEALTHY (cards 0 1)"
        in read(root / "health_pre.log"),
        "health_post_green": "xpu-health: HEALTHY (cards 0 1)"
        in read(root / "health_post.log"),
        "endpoint_left_down": read(root / "endpoint_state.txt").strip()
        == "down",
    }
    passed = all(checks.values())
    summary = {
        "claim": "c4_gdn_out_proj_int8_mechanism",
        "capacity": capacity,
        "baseline_capacity": BASELINE_CAPACITY,
        "acceptance": acceptance,
        "internal_avg_spec_accept_length": internal_accept,
        "soak_rate": soak_rate,
        "soak_degradation": soak_degradation,
        "routes": route_results,
        "checks": checks,
        "pass": passed,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    lines = [f"VERDICT -> {'PASS' if passed else 'FAIL'}"]
    lines.append(
        f"CAPACITY -> candidate={capacity} baseline={BASELINE_CAPACITY}"
    )
    lines.append(
        f"ACCEPTANCE -> samples={acceptance} internal_avg={internal_accept}"
    )
    for result in route_results:
        lines.append(f"ROUTES -> {result['trace']} {result['counts']}")
    lines.extend(
        f"CHECK {name}={'PASS' if value else 'FAIL'}"
        for name, value in checks.items()
    )
    verdict = "\n".join(lines) + "\n"
    (root / "verdict.txt").write_text(verdict, encoding="ascii")
    print(verdict, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
