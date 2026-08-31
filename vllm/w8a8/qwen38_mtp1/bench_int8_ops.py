#!/usr/bin/env python3
"""Numerical and latency oracle for Qwen3.8 TP2 native INT8 linear shapes."""

from __future__ import annotations

import argparse
import json
import time

import torch


DEFAULT_SHAPES = (
    (5120, 8704, "mlp_gate_up_tp2"),
    (17408, 2560, "mlp_down_tp2"),
    (5120, 6144, "q_proj_tp2"),
    (6144, 5120, "o_proj_tp2"),
)


def bench(callable_, repeats: int) -> float:
    for _ in range(5):
        callable_()
    torch.xpu.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        callable_()
    torch.xpu.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repeats


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,4,128,512,2048")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--require-quant-exact",
        action="store_true",
        help="Require byte-exact native and Triton quantization for every row.",
    )
    args = parser.parse_args()

    import vllm_xpu_kernels._xpu_C  # noqa: F401
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )

    torch.manual_seed(args.seed)
    activation_dtype = getattr(torch, args.dtype)
    rows = [int(value) for value in args.rows.split(",")]
    results = []

    for k, n, label in DEFAULT_SHAPES:
        weight = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="xpu")
        weight_scale = (
            torch.rand(n, dtype=torch.float32, device="xpu") * 0.02 + 0.001
        )
        for m in rows:
            x = torch.randn((m, k), dtype=activation_dtype, device="xpu")

            q, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
            output = torch.ops._xpu_C.int8_gemm_w8a8(
                q, scale, weight, weight_scale, activation_dtype, None
            )
            torch.xpu.synchronize()

            ref_q, ref_scale = per_token_quant_int8(x)
            q_cpu = q.cpu()
            ref_q_cpu = ref_q.cpu()
            quant_delta = q_cpu.to(torch.int16) - ref_q_cpu.to(torch.int16)
            quant_mismatch_count = int(torch.count_nonzero(quant_delta).item())
            quant_mismatch_fraction = quant_mismatch_count / quant_delta.numel()
            quant_max_abs_delta = int(quant_delta.abs().max().item())
            quant_exact = quant_mismatch_count == 0
            scale_max_abs = (scale - ref_scale).abs().max().item()

            # Limit the dense fp32 reference to decode-sized rows.
            output_cosine = None
            if m <= 4:
                dequant_weight = weight.float() * weight_scale.view(1, -1)
                reference = (q.float() * scale).matmul(dequant_weight)
                output_cosine = cosine(output, reference)

            quant_ms = bench(
                lambda: torch.ops._xpu_C.per_token_quant_int8_xpu(x),
                args.repeats,
            )
            gemm_ms = bench(
                lambda: torch.ops._xpu_C.int8_gemm_w8a8(
                    q, scale, weight, weight_scale, activation_dtype, None
                ),
                args.repeats,
            )

            def combined():
                x_q, x_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
                return torch.ops._xpu_C.int8_gemm_w8a8(
                    x_q,
                    x_scale,
                    weight,
                    weight_scale,
                    activation_dtype,
                    None,
                )

            combined_ms = bench(combined, args.repeats)
            record = {
                "label": label,
                "m": m,
                "k": k,
                "n": n,
                "dtype": args.dtype,
                "quant_exact": quant_exact,
                "quant_mismatch_count": quant_mismatch_count,
                "quant_mismatch_fraction": quant_mismatch_fraction,
                "quant_max_abs_delta": quant_max_abs_delta,
                "scale_max_abs": scale_max_abs,
                "output_cosine": output_cosine,
                "quant_ms": quant_ms,
                "gemm_ms": gemm_ms,
                "combined_ms": combined_ms,
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            results.append(record)

    if args.require_quant_exact:
        if not all(record["quant_exact"] for record in results):
            raise SystemExit("native quantization differs from the reference")
    elif not all(
        record["quant_max_abs_delta"] <= 1
        and record["quant_mismatch_fraction"] <= 0.001
        and record["scale_max_abs"] <= 1e-6
        for record in results
    ):
        raise SystemExit("native quantization exceeded the 1-LSB tolerance")
    checked = [record for record in results if record["output_cosine"] is not None]
    if not all(record["output_cosine"] >= 0.999 for record in checked):
        raise SystemExit("native GEMM cosine fell below 0.999")


if __name__ == "__main__":
    main()
