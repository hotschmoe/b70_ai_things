#!/usr/bin/env python3
"""One fresh-process arm of the Qwen3.6 June/August W8A8 kernel A/B."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch


COMMON_SCHEMAS = {
    "_C::silu_and_mul": (
        "_C::silu_and_mul(Tensor($0! -> ) result, Tensor input) -> ()"
    ),
    "_moe_C::init_expert_map": (
        "_moe_C::init_expert_map(Tensor expert_map, int num_experts, "
        "int ep_rank, int ep_size) -> ()"
    ),
    "_moe_C::remap_hidden_states": (
        "_moe_C::remap_hidden_states(Tensor hidden_states, Tensor? "
        "hidden_states_scales, Tensor remapped_hidden_states, Tensor? "
        "remapped_hidden_states_scales, Tensor? expert_map, Tensor "
        "rows_per_expert, Tensor unpermuted_row_to_permuted_row, Tensor "
        "topk_ids, int total_experts_num, int local_experts_num) -> ()"
    ),
    "_moe_C::moe_gather": (
        "_moe_C::moe_gather(Tensor($0! -> ) output, Tensor moe_output, "
        "Tensor topk_weights, Tensor unpermuted_row_to_permuted_row, "
        "int num_experts) -> ()"
    ),
    "_xpu_C::per_token_quant_int8_xpu": (
        "_xpu_C::per_token_quant_int8_xpu(Tensor x) -> (Tensor, Tensor)"
    ),
    "_xpu_C::int8_gemm_w8a8": (
        "_xpu_C::int8_gemm_w8a8(Tensor A, Tensor A_scale, Tensor B, "
        "Tensor B_scale, ScalarType? out_dtype, Tensor? bias) -> Tensor"
    ),
    "_xpu_C::cutlass_grouped_gemm_w8a8_int8_interface": (
        "_xpu_C::cutlass_grouped_gemm_w8a8_int8_interface(Tensor ptr_A, "
        "Tensor ptr_A_scales, Tensor ptr_B, Tensor ptr_B_scales, "
        "Tensor? ptr_bias, Tensor ptr_D, Tensor rows_per_expert, int N, "
        "int K, int num_experts) -> Tensor"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def seed_for(label: str, base: int) -> int:
    suffix = int.from_bytes(hashlib.sha256(label.encode("ascii")).digest()[:4])
    return (base + suffix) & 0x7FFFFFFF


def round_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value >= 0, torch.floor(value + 0.5), torch.ceil(value - 0.5))


def quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x32 = x.float().reshape(-1, x.shape[-1])
    absmax = x32.abs().amax(dim=-1, keepdim=True).clamp(min=1.0e-10)
    scales = absmax / 127.0
    q = round_away_from_zero(x32 * (127.0 / absmax))
    q = q.clamp(-127, 127).to(torch.int8).reshape_as(x)
    return q, scales.reshape(*x.shape[:-1], 1)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def time_call(
    call: Callable[[], Any], warmup: int = 8, samples: int = 9, batch: int = 20
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.xpu.synchronize()
    timings = []
    for _ in range(samples):
        torch.xpu.synchronize()
        started = time.perf_counter()
        for _ in range(batch):
            call()
        torch.xpu.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0 / batch)
    mean = statistics.mean(timings)
    return {
        "median_ms": statistics.median(timings),
        "p95_ms": percentile(timings, 0.95),
        "mean_ms": mean,
        "cv": 0.0 if mean == 0 else statistics.pstdev(timings) / mean,
    }


def graph_replay(
    call: Callable[[], Any], hashes: Callable[[Any], list[str]], replays: int = 16
) -> dict[str, Any]:
    graph = torch.xpu.XPUGraph()
    torch.xpu.synchronize()
    with torch.xpu.graph(graph):
        static_output = call()
    torch.xpu.synchronize()
    captured = hashes(static_output)
    replay_hashes = []
    for _ in range(replays):
        graph.replay()
        torch.xpu.synchronize()
        replay_hashes.append(hashes(static_output))
    return {
        "captured_hashes": captured,
        "replay_unique_hashes": sorted({tuple(item) for item in replay_hashes}),
        "replays": replays,
        "pass": all(item == captured for item in replay_hashes),
    }


def identify(args: argparse.Namespace) -> dict[str, Any]:
    module = importlib.import_module("vllm_xpu_kernels._xpu_C")
    support_modules = {
        name: importlib.import_module(name)
        for name in (
            "vllm_xpu_kernels._C",
            "vllm_xpu_kernels._moe_C",
        )
    }
    package_root = Path(module.__file__).resolve().parent
    expected_root = Path(args.expected_package_root).resolve()
    if package_root != expected_root:
        raise RuntimeError(f"package root {package_root} != {expected_root}")
    for name, support_module in support_modules.items():
        support_root = Path(support_module.__file__).resolve().parent
        if support_root != expected_root:
            raise RuntimeError(f"{name} package root {support_root} != {expected_root}")
    expected_hashes = json.loads(args.expected_hashes)
    hashes = {}
    for name, expected in expected_hashes.items():
        actual = sha256_file(package_root / name)
        if actual != expected:
            raise RuntimeError(f"{name} hash {actual} != {expected}")
        hashes[name] = actual
    required_by_suite = {
        "identity": {
            "_xpu_C::per_token_quant_int8_xpu",
            "_xpu_C::int8_gemm_w8a8",
        },
        "quant": {"_xpu_C::per_token_quant_int8_xpu"},
        "dense": {"_xpu_C::int8_gemm_w8a8"},
        "quant-dense": {
            "_xpu_C::per_token_quant_int8_xpu",
            "_xpu_C::int8_gemm_w8a8",
        },
        "grouped": {"_xpu_C::cutlass_grouped_gemm_w8a8_int8_interface"},
        "full": set(COMMON_SCHEMAS),
    }
    schemas: dict[str, Any] = {}
    for name, expected in COMMON_SCHEMAS.items():
        namespace, op = name.split("::", 1)
        try:
            found = str(
                torch._C._dispatch_find_schema_or_throw(name, "").schema()
            )
        except RuntimeError:
            schemas[name] = {"present": False}
            if name in required_by_suite[args.suite]:
                raise RuntimeError(f"required operator is absent: {name}")
            continue
        if found != expected:
            raise RuntimeError(f"schema mismatch for {name}: {found}")
        if not torch._C._dispatch_has_kernel_for_dispatch_key(name, "XPU"):
            raise RuntimeError(f"missing XPU dispatch for {name}")
        if not hasattr(getattr(torch.ops, namespace), op):
            raise RuntimeError(f"missing torch.ops binding for {name}")
        schemas[name] = {"present": True, "schema": found, "xpu": True}
    experimental = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("VLLM_XPU_") or key.startswith("B70_")
    }
    if experimental:
        raise RuntimeError(f"non-base experiment environment present: {experimental}")
    return {
        "arm": args.arm,
        "torch": torch.__version__,
        "package_root": str(package_root),
        "hashes": hashes,
        "support_module_origins": {
            name: module.__file__ for name, module in support_modules.items()
        },
        "schemas": schemas,
        "xpu_name": None if args.offdevice else torch.xpu.get_device_name(0),
        "experimental_environment": experimental,
    }


def make_quant_input(
    rows: int, width: int, dtype: torch.dtype, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, width), generator=generator, dtype=torch.float32)
    if rows >= 1:
        x[0].zero_()
        x[0, :8] = torch.tensor(
            [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 1.0e-12, -1.0e-12]
        )
    if rows >= 2:
        x[1].fill_(1.0e-12)
        x[1, 0] = -1.0e-12
    return x.to(dtype)


def run_quant_case(
    rows: int, width: int, dtype: torch.dtype, base_seed: int
) -> dict[str, Any]:
    label = f"quant-{str(dtype).split('.')[-1]}-m{rows}-k{width}"
    host_x = make_quant_input(rows, width, dtype, seed_for(label, base_seed))
    ref_q, ref_scale = quant_reference(host_x)
    x = host_x.to("xpu")

    def call():
        return torch.ops._xpu_C.per_token_quant_int8_xpu(x)

    q, scale = call()
    torch.xpu.synchronize()
    scale_cpu = scale.cpu()
    q_ok = torch.equal(q.cpu(), ref_q)
    scale_error = float((scale_cpu - ref_scale).abs().max())
    repeated = []
    for _ in range(16):
        out = call()
        torch.xpu.synchronize()
        repeated.append([tensor_sha256(out[0]), tensor_sha256(out[1])])
    graph = graph_replay(call, lambda out: [tensor_sha256(out[0]), tensor_sha256(out[1])])
    timing = time_call(call, batch=5 if rows == 8192 else 50)
    passed = q_ok and scale_error <= 1.0e-7 and len({tuple(x) for x in repeated}) == 1 and graph["pass"]
    return {
        "id": label,
        "suite": "quant",
        "shape": [rows, width],
        "dtype": str(dtype),
        "q_matches_round_away_reference": q_ok,
        "scale_max_abs_error": scale_error,
        "q_sha256": tensor_sha256(q),
        "scale_sha256": tensor_sha256(scale),
        "repeat_unique_hashes": sorted({tuple(item) for item in repeated}),
        "graph": graph,
        "timing": timing,
        "pass": passed,
    }


def deterministic_dense_inputs(
    label: str, m: int, k: int, n: int, base_seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed_for(label, base_seed))
    a = torch.randint(-127, 128, (m, k), generator=generator, dtype=torch.int8)
    b = torch.randint(-127, 128, (k, n), generator=generator, dtype=torch.int8)
    a_scale = torch.rand((m, 1), generator=generator, dtype=torch.float32) * 0.02 + 1.0e-4
    b_scale = torch.rand((n,), generator=generator, dtype=torch.float32) * 0.02 + 1.0e-4
    return a, a_scale, b.contiguous(), b_scale


def dense_reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    rows: int = 8,
) -> torch.Tensor:
    take = min(rows, a.shape[0])
    accum = a[:take].to(torch.int32) @ b.to(torch.int32)
    return (accum.float() * a_scale[:take] * b_scale).to(torch.bfloat16)


def validate_float_output(
    output: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    selected = output[: reference.shape[0]].float().cpu()
    error = (selected - reference.float()).abs()
    return float(error.max()), float(error.mean())


def run_dense_case(
    label: str, m: int, k: int, n: int, base_seed: int, quant_first: bool
) -> dict[str, Any]:
    prefix = "quant-dense" if quant_first else "dense"
    case_id = f"{prefix}-{label}-m{m}-k{k}-n{n}"
    if quant_first:
        generator = torch.Generator().manual_seed(seed_for(case_id, base_seed))
        host_x = torch.randn((m, k), generator=generator, dtype=torch.float32).to(torch.bfloat16)
        host_a, host_a_scale = quant_reference(host_x)
        _, _, host_b, host_b_scale = deterministic_dense_inputs(case_id, m, k, n, base_seed)
        x = host_x.to("xpu")
        b = host_b.to("xpu")
        b_scale = host_b_scale.to("xpu")

        def call():
            a, a_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
            out = torch.ops._xpu_C.int8_gemm_w8a8(
                a, a_scale, b, b_scale, torch.bfloat16, None
            )
            return out, out.float().sum()

    else:
        host_a, host_a_scale, host_b, host_b_scale = deterministic_dense_inputs(
            case_id, m, k, n, base_seed
        )
        a = host_a.to("xpu")
        a_scale = host_a_scale.to("xpu")
        b = host_b.to("xpu")
        b_scale = host_b_scale.to("xpu")

        def call():
            out = torch.ops._xpu_C.int8_gemm_w8a8(
                a, a_scale, b, b_scale, torch.bfloat16, None
            )
            return out, out.float().sum()

    reference = dense_reference(host_a, host_a_scale, host_b, host_b_scale)
    output, checksum = call()
    torch.xpu.synchronize()
    max_error, mean_error = validate_float_output(output, reference)
    hashes = []
    for _ in range(16 if m <= 2 else 4):
        repeated_output, repeated_checksum = call()
        torch.xpu.synchronize()
        hashes.append(
            [tensor_sha256(repeated_output), tensor_sha256(repeated_checksum)]
        )
    graph = graph_replay(
        call, lambda out: [tensor_sha256(out[0]), tensor_sha256(out[1])]
    )
    timing = time_call(call, batch=2 if m == 8192 else 20)
    tolerance = 0.25
    passed = max_error <= tolerance and len({tuple(item) for item in hashes}) == 1 and graph["pass"]
    return {
        "id": case_id,
        "suite": prefix,
        "shape": {"M": m, "K": k, "N": n},
        "weight_layout": "contiguous-KN",
        "max_abs_error_vs_int32_reference": max_error,
        "mean_abs_error_vs_int32_reference": mean_error,
        "output_sha256": tensor_sha256(output),
        "checksum_sha256": tensor_sha256(checksum),
        "repeat_unique_hashes": sorted({tuple(item) for item in hashes}),
        "graph": graph,
        "timing": timing,
        "pass": passed,
    }


def routing_counts(total_m: int, experts: int, pattern: str) -> torch.Tensor:
    counts = torch.zeros(experts, dtype=torch.int32)
    if pattern == "spread":
        counts[:total_m] = 1
    elif pattern == "collision":
        counts[0] = total_m
    elif pattern == "skew":
        base = [8, 4, 2, 1, 1] if total_m == 16 else [4, 2, 1, 1]
        counts[: len(base)] = torch.tensor(base, dtype=torch.int32)
    else:
        raise ValueError(pattern)
    if int(counts.sum()) != total_m:
        raise RuntimeError(f"bad routing counts for {pattern}")
    return counts


def grouped_reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty((a.shape[0], b.shape[2]), dtype=torch.bfloat16)
    offset = 0
    for expert, rows in enumerate(counts.tolist()):
        if rows:
            accum = a[offset : offset + rows].to(torch.int32) @ b[expert].to(torch.int32)
            output[offset : offset + rows] = (
                accum.float()
                * a_scale[offset : offset + rows]
                * b_scale[expert]
            ).to(torch.bfloat16)
            offset += rows
    return output


def run_grouped_case(
    label: str,
    total_m: int,
    k: int,
    n: int,
    pattern: str,
    base_seed: int,
) -> dict[str, Any]:
    experts = 256
    case_id = f"grouped-{label}-m{total_m}-k{k}-n{n}-{pattern}"
    generator = torch.Generator().manual_seed(seed_for(case_id, base_seed))
    host_a = torch.randint(-127, 128, (total_m, k), generator=generator, dtype=torch.int8)
    host_a_scale = torch.rand((total_m, 1), generator=generator) * 0.02 + 1.0e-4
    host_b = torch.randint(-127, 128, (experts, k, n), generator=generator, dtype=torch.int8)
    host_b_scale = torch.rand((experts, n), generator=generator) * 0.02 + 1.0e-4
    counts = routing_counts(total_m, experts, pattern)
    reference = grouped_reference(host_a, host_a_scale, host_b, host_b_scale, counts)
    a = host_a.to("xpu")
    a_scale = host_a_scale.to("xpu")
    b = host_b.to("xpu")
    b_scale = host_b_scale.to("xpu")
    rows_per_expert = counts.to("xpu")
    output = torch.empty((total_m, n), dtype=torch.bfloat16, device="xpu")

    def call():
        torch.ops._xpu_C.cutlass_grouped_gemm_w8a8_int8_interface(
            a,
            a_scale,
            b,
            b_scale,
            None,
            output,
            rows_per_expert,
            n,
            k,
            experts,
        )
        return output, output.float().sum()

    result, checksum = call()
    torch.xpu.synchronize()
    max_error, mean_error = validate_float_output(result, reference)
    hashes = []
    for _ in range(16):
        repeated_output, repeated_checksum = call()
        torch.xpu.synchronize()
        hashes.append([tensor_sha256(repeated_output), tensor_sha256(repeated_checksum)])
    graph = graph_replay(
        call, lambda out: [tensor_sha256(out[0]), tensor_sha256(out[1])]
    )
    timing = time_call(call, batch=20)
    passed = max_error <= 0.25 and len({tuple(item) for item in hashes}) == 1 and graph["pass"]
    return {
        "id": case_id,
        "suite": "grouped",
        "shape": {"E": experts, "Total_M": total_m, "K": k, "N": n},
        "routing": pattern,
        "rows_per_expert": counts.tolist(),
        "max_abs_error_vs_int32_reference": max_error,
        "mean_abs_error_vs_int32_reference": mean_error,
        "output_sha256": tensor_sha256(result),
        "checksum_sha256": tensor_sha256(checksum),
        "repeat_unique_hashes": sorted({tuple(item) for item in hashes}),
        "graph": graph,
        "timing": timing,
        "pass": passed,
    }


def execute_suite(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = []
    if args.suite in ("identity", "full"):
        return cases
    if args.suite == "quant":
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rows in (1, 2, 8192):
                for width in (256, 2048):
                    cases.append(run_quant_case(rows, width, dtype, args.seed))
    elif args.suite in ("dense", "quant-dense"):
        shapes = (
            ("gdn_fused_projection", 2048, 6144),
            ("gdn_ba", 2048, 32),
            ("gdn_output", 2048, 2048),
            ("shared_gate_up", 2048, 512),
            ("shared_down", 256, 2048),
            ("attention_qkv", 2048, 4608),
            ("attention_output", 2048, 2048),
        )
        for label, k, n in shapes:
            for m in (1, 2):
                cases.append(
                    run_dense_case(
                        label,
                        m,
                        k,
                        n,
                        args.seed,
                        quant_first=args.suite == "quant-dense",
                    )
                )
        if args.profile:
            cases.append(
                run_dense_case(
                    "profile_hidden",
                    8192,
                    2048,
                    2048,
                    args.seed,
                    quant_first=args.suite == "quant-dense",
                )
            )
    elif args.suite == "grouped":
        for label, k, n in (("gemm1", 2048, 512), ("gemm2", 256, 2048)):
            for total_m in (8, 16):
                for pattern in ("spread", "collision", "skew"):
                    cases.append(
                        run_grouped_case(
                            label, total_m, k, n, pattern, args.seed
                        )
                    )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--suite",
        choices=(
            "identity",
            "full",
            "quant",
            "dense",
            "quant-dense",
            "grouped",
        ),
        required=True,
    )
    parser.add_argument("--expected-package-root", required=True)
    parser.add_argument("--expected-hashes", required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--offdevice", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact: dict[str, Any] = {
        "protocol": "qwen36-june-august-w8a8-kernel-arm-v1",
        "arm": args.arm,
        "suite": args.suite,
        "seed": args.seed,
        "profile": args.profile,
        "identity": None,
        "cases": [],
        "failures": [],
    }
    try:
        artifact["identity"] = identify(args)
        artifact["cases"] = [] if args.offdevice else execute_suite(args)
        artifact["failures"] = [
            case["id"] for case in artifact["cases"] if not case["pass"]
        ]
    except Exception as error:
        artifact["failures"].append(
            {"type": type(error).__name__, "message": str(error)}
        )
    artifact["pass"] = not artifact["failures"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "suite": args.suite,
                "cases": len(artifact["cases"]),
                "failures": artifact["failures"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
