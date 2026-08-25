#!/usr/bin/env python3
"""Isolate the June Qwen3.6 W8A8 fused-MoE stages on one XPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._moe_C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm_xpu_kernels.fused_moe_interface import (
    _normalize_int8_weight_scales,
    _per_token_quant_int8,
    fused_moe_activation,
    xpu_fused_moe,
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def synchronize(stage: str) -> None:
    torch.xpu.synchronize()
    print(f"stage -> {stage}: OK", flush=True)


def make_inputs(rows: int, seed: int) -> dict[str, torch.Tensor]:
    hidden_size = 2048
    inter_size = 256
    experts = 256
    topk = 8
    torch.manual_seed(seed + rows)
    hidden_states = (
        torch.randn((rows, hidden_size), device="xpu", dtype=torch.bfloat16) / 16
    ).contiguous()
    w13 = torch.randint(
        -127,
        128,
        (experts, hidden_size, 2 * inter_size),
        device="xpu",
        dtype=torch.int8,
    ).contiguous()
    w2 = torch.randint(
        -127,
        128,
        (experts, inter_size, hidden_size),
        device="xpu",
        dtype=torch.int8,
    ).contiguous()
    w13_scales = (
        torch.rand((experts, 2 * inter_size), device="xpu") * 0.02 + 0.001
    ).contiguous()
    w2_scales = (
        torch.rand((experts, hidden_size), device="xpu") * 0.02 + 0.001
    ).contiguous()
    topk_ids = (
        torch.arange(rows * topk, device="xpu", dtype=torch.int64) % experts
    ).view(rows, topk).contiguous()
    topk_weights = torch.softmax(
        torch.rand((rows, topk), device="xpu"), dim=-1
    ).contiguous()
    synchronize(f"m{rows}-inputs")
    return {
        "hidden_states": hidden_states,
        "w13": w13,
        "w13_scales": w13_scales,
        "w2": w2,
        "w2_scales": w2_scales,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }


def staged_call(inputs: dict[str, torch.Tensor], rows: int) -> torch.Tensor:
    hidden_size = 2048
    inter_size = 256
    experts = 256
    topk = 8
    routed_rows = rows * topk
    hidden_states = inputs["hidden_states"]
    rows_per_expert = torch.zeros((experts,), dtype=torch.int32, device="xpu")
    remapped = torch.empty(
        (routed_rows, hidden_size), dtype=torch.bfloat16, device="xpu"
    )
    unpermuted_to_permuted = torch.empty(
        (rows, topk), dtype=torch.int32, device="xpu"
    )
    torch.ops._moe_C.remap_hidden_states(
        hidden_states,
        None,
        remapped,
        None,
        None,
        rows_per_expert,
        unpermuted_to_permuted,
        inputs["topk_ids"],
        experts,
        experts,
    )
    synchronize(f"m{rows}-remap")

    gemm1_a, gemm1_a_scales = _per_token_quant_int8(remapped)
    synchronize(f"m{rows}-quant1")
    gemm1_output = torch.empty(
        (routed_rows, 2 * inter_size), dtype=torch.bfloat16, device="xpu"
    )
    torch.ops._xpu_C.cutlass_grouped_gemm_w8a8_int8_interface(
        gemm1_a,
        gemm1_a_scales,
        inputs["w13"],
        _normalize_int8_weight_scales(inputs["w13_scales"], 2 * inter_size),
        None,
        gemm1_output,
        rows_per_expert,
        2 * inter_size,
        hidden_size,
        experts,
    )
    synchronize(f"m{rows}-gemm1")

    activation = torch.empty(
        (routed_rows, inter_size), dtype=torch.bfloat16, device="xpu"
    )
    fused_moe_activation(activation, gemm1_output, "silu")
    synchronize(f"m{rows}-activation")
    gemm2_a, gemm2_a_scales = _per_token_quant_int8(activation.contiguous())
    synchronize(f"m{rows}-quant2")
    gemm2_output = torch.empty(
        (routed_rows, hidden_size), dtype=torch.bfloat16, device="xpu"
    )
    torch.ops._xpu_C.cutlass_grouped_gemm_w8a8_int8_interface(
        gemm2_a,
        gemm2_a_scales,
        inputs["w2"],
        _normalize_int8_weight_scales(inputs["w2_scales"], hidden_size),
        None,
        gemm2_output,
        rows_per_expert,
        hidden_size,
        inter_size,
        experts,
    )
    synchronize(f"m{rows}-gemm2")

    output = torch.empty_like(hidden_states)
    torch.ops._moe_C.moe_gather(
        output,
        gemm2_output,
        inputs["topk_weights"],
        unpermuted_to_permuted,
        experts,
    )
    synchronize(f"m{rows}-gather")
    return output


def fused_call(
    inputs: dict[str, torch.Tensor], output: torch.Tensor
) -> torch.Tensor:
    return xpu_fused_moe(
        hidden_states=inputs["hidden_states"],
        w13=inputs["w13"],
        w13_scales=inputs["w13_scales"],
        w13_bias=None,
        w2=inputs["w2"],
        w2_scales=inputs["w2_scales"],
        w2_bias=None,
        topk_weights=inputs["topk_weights"],
        topk_ids=inputs["topk_ids"],
        n_experts_per_token=8,
        activation="silu",
        num_experts=256,
        output=output,
        is_int8=True,
    )


def run_compiled_case(
    inputs: dict[str, torch.Tensor], eager_output: torch.Tensor, rows: int
) -> dict[str, Any]:
    # The model graph does not inline the June Python function. vLLM exposes
    # the entire expert call as an opaque custom op with a fake implementation.
    # Compile that same boundary so this probe matches the live model route.
    import vllm.model_executor.layers.fused_moe.fused_moe  # noqa: F401

    def body(
        hidden_states,
        w13,
        w13_scales,
        w2,
        w2_scales,
        topk_weights,
        topk_ids,
    ):
        return torch.ops.vllm.outplace_fused_experts(
            hidden_states=hidden_states,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",
            apply_router_weight_on_input=False,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=256,
            w1_scale=w13_scales,
            w2_scale=w2_scales,
        )

    compiled = torch.compile(body, backend="inductor", fullgraph=True, dynamic=False)
    arguments = (
        inputs["hidden_states"],
        inputs["w13"],
        inputs["w13_scales"],
        inputs["w2"],
        inputs["w2_scales"],
        inputs["topk_weights"],
        inputs["topk_ids"],
    )
    output = compiled(*arguments)
    synchronize(f"m{rows}-compiled-first")
    first_hash = tensor_sha256(output)
    max_diff = float((eager_output.float() - output.float()).abs().max().cpu())
    hashes = []
    for _ in range(4):
        output = compiled(*arguments)
        torch.xpu.synchronize()
        hashes.append(tensor_sha256(output))
    print(f"stage -> m{rows}-compiled-repeat: OK", flush=True)
    return {
        "compile_boundary": "vllm::outplace_fused_experts",
        "first_sha256": first_hash,
        "unique_sha256": sorted(set(hashes)),
        "max_abs_diff_vs_eager": max_diff,
        "pass": max_diff == 0.0 and len(set(hashes)) == 1,
    }


def run_case(rows: int, seed: int, compile_case: bool) -> dict[str, Any]:
    inputs = make_inputs(rows, seed)
    staged = staged_call(inputs, rows)
    staged_hash = tensor_sha256(staged)
    output = torch.empty_like(inputs["hidden_states"])
    fused_call(inputs, output)
    synchronize(f"m{rows}-fused-eager")
    fused_hash = tensor_sha256(output)
    max_diff = float((staged.float() - output.float()).abs().max().cpu())

    eager_hashes = []
    for _ in range(4 if rows == 8192 else 16):
        fused_call(inputs, output)
        torch.xpu.synchronize()
        eager_hashes.append(tensor_sha256(output))
    print(f"stage -> m{rows}-fused-repeat: OK", flush=True)

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        fused_call(inputs, output)
    synchronize(f"m{rows}-fused-capture")
    captured_hash = tensor_sha256(output)
    replay_hashes = []
    for _ in range(4 if rows == 8192 else 16):
        graph.replay()
        torch.xpu.synchronize()
        replay_hashes.append(tensor_sha256(output))
    print(f"stage -> m{rows}-fused-replay: OK", flush=True)

    compiled_result = run_compiled_case(inputs, output, rows) if compile_case else None
    passed = (
        max_diff == 0.0
        and len(set(eager_hashes)) == 1
        and len(set(replay_hashes)) == 1
        and replay_hashes[0] == captured_hash
        and (compiled_result is None or compiled_result["pass"])
    )
    return {
        "rows": rows,
        "routed_rows": rows * 8,
        "staged_sha256": staged_hash,
        "fused_sha256": fused_hash,
        "max_abs_diff_staged_vs_fused": max_diff,
        "eager_unique_sha256": sorted(set(eager_hashes)),
        "captured_sha256": captured_hash,
        "replay_unique_sha256": sorted(set(replay_hashes)),
        "compiled": compiled_result,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,48,8192")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    failure = None
    try:
        for rows in (int(value) for value in args.rows.split(",")):
            cases.append(run_case(rows, args.seed, args.compile))
    except Exception as exc:  # noqa: BLE001 - preserve the exact failing stage.
        failure = {"type": type(exc).__name__, "message": str(exc)}
    result = {
        "protocol": "qwen36-june-fused-moe-single-v1",
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(0),
        "cases": cases,
        "failure": failure,
        "pass": failure is None and all(case["pass"] for case in cases),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
