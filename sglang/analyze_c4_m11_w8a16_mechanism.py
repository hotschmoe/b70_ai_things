#!/usr/bin/env python3
"""Fail-closed analyzer for experiment 06 M<=11 W8A16 routing."""

import json
import re
import sys
from pathlib import Path

from parse_tp2_math_census import trace_census


SERVED = "qwen36-27b-W8A8-sqgptq-mtp-c4-m11-w8a16-mechanism"
MODEL = "/models/qwen3.6-27b/w8a8-sqgptq"
OPS = re.compile(
    r"^(_xpu_C::int8_gemm_w8a16|_xpu_C::int8_gemm_w8a8|"
    r"_xpu_C::dynamic_per_token_int8_quant)$"
)
FATAL = re.compile(
    r"device_lost|out_of_resources|ur_result_error|enginedead|!!!!|"
    r"segmentation fault|(^|[^a-z])nan([^a-z]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def calls(rows, op, a, b, b_index):
    return sum(
        values["calls"]
        for (name, dims), values in rows.items()
        if name == op
        and len(dims) > b_index
        and dims[0] == tuple(a)
        and dims[b_index] == tuple(b)
    )


def one_tensor_calls(rows, op, shape):
    return sum(
        values["calls"]
        for (name, dims), values in rows.items()
        if name == op and dims and dims[0] == tuple(shape)
    )


def route_summary(path):
    rows, _total, _matched = trace_census(path, OPS)
    shapes = {
        "gate_up": ((11, 5120), (5120, 17408), 320),
        "down": ((11, 8704), (8704, 5120), 320),
        "attn_qkv": ((11, 5120), (5120, 7168), 80),
        "attn_out": ((11, 3072), (3072, 5120), 80),
    }
    summary = {}
    checks = {}
    for label, (a, b, expected) in shapes.items():
        summary[f"w8a16_{label}"] = calls(
            rows, "_xpu_C::int8_gemm_w8a16", a, b, 1
        )
        summary[f"w8a8_{label}"] = calls(
            rows, "_xpu_C::int8_gemm_w8a8", a, b, 3
        )
        checks[f"w8a16_{label}_exact"] = summary[f"w8a16_{label}"] == expected
        checks[f"w8a8_{label}_absent"] = summary[f"w8a8_{label}"] == 0
    summary["quant_k5120"] = one_tensor_calls(
        rows, "_xpu_C::dynamic_per_token_int8_quant", (11, 5120)
    )
    summary["quant_k8704"] = one_tensor_calls(
        rows, "_xpu_C::dynamic_per_token_int8_quant", (11, 8704)
    )
    summary["quant_k3072"] = one_tensor_calls(
        rows, "_xpu_C::dynamic_per_token_int8_quant", (11, 3072)
    )
    checks["all_m11_quant_absent"] = all(
        summary[key] == 0
        for key in ("quant_k5120", "quant_k8704", "quant_k3072")
    )
    checks["w8a16_total_160_per_step"] = sum(
        summary[f"w8a16_{label}"] for label in shapes
    ) == 800
    return summary, checks


def inspect_exact(path, root, image_id):
    data = load_json(path)
    if not isinstance(data, list) or len(data) != 1:
        return False
    item = data[0]
    env = set(item.get("Config", {}).get("Env", []))
    required = {
        "B70_XPU_W8A8=1",
        "B70_XPU_W8A8_FUSED=1",
        "B70_W8A16_M_MAX=11",
        "B70_W8A16_ROUTE_DEBUG=1",
        "B70_W8A8_QUANT_LMHEAD=0",
        "B70_XPU_REPLICATE_MTP_EMBED=1",
        "B70_XPU_DELAY_MLP_AR=0",
        "B70_XPU_FUSED_MLP_AR_NORM=0",
        "CCL_TOPO_P2P_ACCESS=0",
    }
    command = " ".join(item.get("Config", {}).get("Cmd", []) or [])
    mounts = item.get("Mounts", [])
    shim = Path(__file__).resolve().parent / "patches/w8a8_shim.py"
    mounted = any(
        mount.get("Destination")
        == "/opt/venv/lib/python3.12/site-packages/w8a8_shim.py"
        and Path(str(mount.get("Source", ""))).resolve() == shim.resolve()
        and mount.get("RW") is False
        for mount in mounts
    )
    return (
        not (required - env)
        and item.get("Image") == image_id
        and f"--model-path '{MODEL}'" in command
        and f"--served-model-name '{SERVED}'" in command
        and mounted
    )


def main():
    root = Path(sys.argv[1]).resolve()
    manifest = dict(
        line.split("=", 1)
        for line in (root / "manifest.txt").read_text().splitlines()
        if "=" in line
    )
    traces = sorted(Path(manifest["profile_dir"]).glob("*DECODE.trace.json.gz"))
    summaries = []
    route_checks = []
    for trace in traces:
        summary, checks = route_summary(trace)
        summaries.append({"trace": str(trace), "counts": summary})
        route_checks.append(checks)
    server = (root / "server.log").read_text(errors="replace")
    models = load_json(root / "models.json")
    server_info = load_json(root / "server_info.json")
    checks = {
        "two_rank_traces": len(traces) == 2,
        "all_exact_routes": len(route_checks) == 2
        and all(all(item.values()) for item in route_checks),
        "threshold_install_log": "M<=11=int8_gemm_w8a16, M>11=int8_gemm_w8a8, source=env" in server,
        "m11_route_log": re.search(
            r"\[w8a8-route\] route=w8a16 M=11 .*m_max=11 relation=at_max",
            server,
        )
        is not None,
        "above_m11_route_log": re.search(
            r"\[w8a8-route\] route=w8a8 M=([1-9][0-9]+) .*m_max=11 relation=above_max",
            server,
        )
        is not None,
        "served_identity": models.get("data", [{}])[0].get("id") == SERVED,
        "capacity_ge_base": int(server_info.get("max_total_num_tokens") or 0)
        >= 143360,
        "container_exact": inspect_exact(
            root / "container_inspect.json", root, manifest["image_id"]
        ),
        "deterministic_repeat": (root / "deterministic_1.json").read_bytes()
        == (root / "deterministic_2.json").read_bytes(),
        "mixed_coherent": "GATE PASS: all streams coherent"
        in (root / "mixed.log").read_text(),
        "health_pre": "HEALTHY" in (root / "health_pre.log").read_text(),
        "health_post": "HEALTHY" in (root / "health_post.log").read_text(),
        "endpoint_down": (root / "endpoint_state.txt").read_text().strip() == "down",
        "no_fatal_marker": FATAL.search(server) is None,
    }
    result = {"schema": "c4_m11_w8a16_mechanism_v1", "routes": summaries, "checks": checks}
    (root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    passed = all(checks.values())
    (root / "verdict.txt").write_text("PASS\n" if passed else "FAIL\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
