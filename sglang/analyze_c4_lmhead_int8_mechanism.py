#!/usr/bin/env python3
"""Fail-closed analyzer for the repaired C4 INT8-lm_head mechanism gate."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


BASELINE_CAPACITY = 143360
MIN_CAPACITY = max(131072, int(BASELINE_CAPACITY * 0.95))
EXPECTED_IDENTITIES = {
    ("target", 0),
    ("target", 1),
    ("draft", 0),
    ("draft", 1),
}
EXPECTED_READY = {
    ("target", 0, "replaced"),
    ("target", 1, "replaced"),
    ("draft", 0, "aliased"),
    ("draft", 1, "aliased"),
}
READY_RE = re.compile(
    r"\[lmhead-int8\] ready role=(target|draft) rank=([01]) "
    r"N=124160 K=5120 storage=(replaced|aliased) w8a16_only=1 "
    r".*bf16_released=1"
)
SHARED_RE = re.compile(
    r"\[lmhead-int8\] SHARED role=draft rank=([01]) "
    r"same_weight=1 same_scale=1 w8a16_only=1"
)
ROUTE_RE = re.compile(
    r"\[lmhead-int8\] ROUTES role=(target|draft) rank=([01]) "
    r"calls=(\d+) latest_rows=(\d+) w8a16_only=1"
)
ACCEPT_RE = re.compile(
    r"accept len: ([0-9.]+), accept rate: ([0-9.]+)"
)
FATAL_RE = re.compile(
    r"device_lost|out_of_resources|ur_result_error|enginedead|!!!!|"
    r"segmentation fault|(^|[^a-z])nan([^a-z]|$)",
    re.IGNORECASE | re.MULTILINE,
)
GARBAGE_RE = re.compile(r"(\S)\1{9,}")


def read(path: Path) -> str:
    return path.read_text(errors="replace")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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
        and len(set(word.lower() for word in words)) >= 30
        and GARBAGE_RE.search(text) is None
    )


def inspect_exact(path: Path, repo: Path) -> bool:
    data = load_json(path)
    if not isinstance(data, list) or len(data) != 1:
        return False
    item = data[0]
    env = set(item.get("Config", {}).get("Env", []))
    required = {
        "B70_XPU_W8A8=1",
        "B70_XPU_W8A8_FUSED=1",
        "B70_W8A8_QUANT_LMHEAD=1",
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

    def mounted(destination: str, source: Path) -> bool:
        return any(
            mount.get("Destination") == destination
            and Path(str(mount.get("Source", ""))).resolve() == source.resolve()
            and mount.get("RW") is False
            for mount in mounts
        )

    return (
        not (required - env)
        and mounted(
            "/opt/venv/lib/python3.12/site-packages/w8a8_shim.py",
            repo / "sglang/patches/w8a8_shim.py",
        )
        and mounted(
            "/opt/venv/lib/python3.12/site-packages/woq_shim.py",
            repo / "sglang/patches/woq_shim.py",
        )
        and mounted(
            "/opt/venv/lib/python3.12/site-packages/mtp_replicated_embedding.py",
            repo / "sglang/patches/mtp_replicated_embedding.py",
        )
        and mounted("/work/kernel", Path("/mnt/vm_8tb/b70/w8a8_kernel"))
        and mounted(
            "/work/push_ar",
            repo / "vllm/contrib/vllm_push_allreduce/prebuilt",
        )
    )


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parent.parent
    server = read(root / "server.log")
    fixed_delta = read(root / "fixed_server_delta.log")
    info = load_json(root / "server_info_after.json")
    inspect = load_json(root / "container_inspect.json")
    model = load_json(root / "models.json")["data"][0]
    deterministic = load_json(root / "deterministic.json")
    fixed = load_json(root / "fixed_generation.json")

    ready = [
        (role, int(rank), storage)
        for role, rank, storage in READY_RE.findall(server)
    ]
    shared = [int(rank) for rank in SHARED_RE.findall(server)]
    routes = [
        (role, int(rank), int(calls), int(rows))
        for role, rank, calls, rows in ROUTE_RE.findall(server)
    ]
    routes_at_one = {
        (role, rank) for role, rank, calls, _rows in routes if calls == 1
    }
    meaningful_routes = {
        (role, rank) for role, rank, calls, _rows in routes if calls >= 100
    }

    acceptance = [
        (float(length), float(rate))
        for length, rate in ACCEPT_RE.findall(fixed_delta)
    ]
    rates = [rate for _length, rate in acceptance]
    lengths = [length for length, _rate in acceptance]
    internal_accept = [
        float(state.get("avg_spec_accept_length") or 0.0)
        for state in info.get("internal_states", [])
    ]

    message = fixed["choices"][0]["message"]
    fixed_text = (message.get("reasoning_content") or "") + (
        message.get("content") or ""
    )
    fixed_tokens = int(fixed.get("usage", {}).get("completion_tokens") or 0)
    deterministic_ok = len(deterministic) == 8 and all(
        int(row.get("completion_tokens") or 0) > 0
        and (
            (row.get("reasoning_content") or "") + (row.get("content") or "")
        ).strip()
        for row in deterministic
    )

    exact_info = (
        info.get("status") == "ready"
        and int(info.get("context_length") or 0) == 131072
        and int(info.get("tp_size") or 0) == 2
        and int(info.get("pp_size") or 0) == 1
        and int(info.get("max_running_requests") or 0) == 4
        and info.get("disable_cuda_graph") is True
        and info.get("disable_radix_cache") is True
        and int(info.get("speculative_num_steps") or 0) == 10
        and int(info.get("speculative_num_draft_tokens") or 0) == 11
    )
    capacity = int(info.get("max_total_num_tokens") or 0)
    checks = {
        "artifact_hashes_stable": read(root / "artifacts.sha256")
        == read(root / "artifacts_after.sha256"),
        "container_config_exact": inspect_exact(
            root / "container_inspect.json", repo
        ),
        "server_config_exact": exact_info,
        "served_model_exact": model.get("id")
        == "qwen36-27b-w8a8-gptq-mtp-c4-lmhead-int8-mechanism",
        "capacity_covers_131072": capacity >= 131072,
        "capacity_retains_95pct_of_143360": capacity >= MIN_CAPACITY,
        "exact_ready_identities": len(ready) == 4 and set(ready) == EXPECTED_READY,
        "exact_shared_ranks": len(shared) == 2 and set(shared) == {0, 1},
        "all_routes_seen_at_call_1": routes_at_one == EXPECTED_IDENTITIES,
        "all_routes_meaningful_ge_100_calls": meaningful_routes
        == EXPECTED_IDENTITIES,
        "fixed_generation_ge_512_tokens": fixed_tokens >= 512,
        "fixed_generation_coherent": coherent_text(fixed_text),
        "fixed_acceptance_has_samples": len(acceptance) >= 2,
        "fixed_acceptance_rate_ge_020": max(rates, default=0.0) >= 0.20,
        "fixed_acceptance_len_ge_3": max(lengths, default=0.0) >= 3.0,
        "fixed_acceptance_no_three_zero_run": longest_zero_run(rates) <= 2,
        "server_acceptance_average_ge_3": min(internal_accept, default=0.0)
        >= 3.0,
        "deterministic_corpus_nonempty": deterministic_ok,
        "concurrent_coherence_pass": (
            "=== 4 streams:" in read(root / "coherence.log")
            and "GATE PASS: all streams coherent" in read(root / "coherence.log")
        ),
        "no_fatal_server_markers": FATAL_RE.search(server) is None,
        "health_pre_green": "xpu-health: HEALTHY (cards 0 1)"
        in read(root / "health_pre.log"),
        "health_post_green": "xpu-health: HEALTHY (cards 0 1)"
        in read(root / "health_post.log"),
        "endpoint_left_down": read(root / "endpoint_state.txt").strip() == "down",
        "inspect_was_single_container": isinstance(inspect, list)
        and len(inspect) == 1,
    }
    passed = all(checks.values())
    summary = {
        "claim": "repaired_lmhead_int8_mechanism",
        "baseline_capacity": BASELINE_CAPACITY,
        "minimum_capacity": MIN_CAPACITY,
        "candidate_capacity": capacity,
        "ready": ready,
        "shared_ranks": shared,
        "routes": routes,
        "fixed_completion_tokens": fixed_tokens,
        "fixed_acceptance": acceptance,
        "fixed_longest_zero_run": longest_zero_run(rates),
        "internal_avg_spec_accept_length": internal_accept,
        "checks": checks,
        "pass": passed,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    lines = [f"VERDICT -> {'PASS' if passed else 'FAIL'}"]
    lines.append(
        f"CAPACITY -> candidate={capacity} minimum={MIN_CAPACITY} "
        f"baseline={BASELINE_CAPACITY}"
    )
    lines.append(
        f"ACCEPTANCE -> samples={acceptance} internal_avg={internal_accept} "
        f"zero_run={longest_zero_run(rates)}"
    )
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
