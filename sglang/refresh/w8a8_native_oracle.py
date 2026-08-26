#!/usr/bin/env python3
"""Numerical and timing oracle for the refreshed SGLang native W8A8 arm."""

from __future__ import annotations

import importlib
import json
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


def round_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value >= 0, torch.floor(value + 0.5), torch.ceil(value - 0.5))


def host_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_float = x.float()
    absmax = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = absmax / 127.0
    quant = round_away_from_zero(x_float * (127.0 / absmax))
    return quant.clamp(-127, 127).to(torch.int8), scale


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def bench(call: Callable[[], Any], warmup: int = 12, batches: int = 9) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.xpu.synchronize()
    samples = []
    for _ in range(batches):
        torch.xpu.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            call()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - started) * 50.0)
    mean = statistics.mean(samples)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "mean_ms": mean,
        "cv": statistics.pstdev(samples) / mean,
    }


def numerical_oracle() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20260826)
    host_x = torch.randn((3, 512), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    host_weight_nk = torch.randint(
        -127, 128, (768, 512), generator=generator, dtype=torch.int8
    )
    host_weight = host_weight_nk.t()
    host_weight_scale = (
        torch.rand((768,), generator=generator, dtype=torch.float32) * 0.009 + 0.001
    )
    expected_q, expected_scale = host_quant(host_x)
    accumulator = torch.matmul(expected_q.int(), host_weight.int())
    expected = (
        accumulator.float() * expected_scale * host_weight_scale.reshape(1, -1)
    ).to(torch.bfloat16)

    x = host_x.to("xpu")
    weight_nk = host_weight_nk.to("xpu")
    weight = weight_nk.t()
    weight_scale = host_weight_scale.to("xpu")
    q, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    output = torch.ops._xpu_C.int8_gemm_w8a8(
        q, scale, weight, weight_scale, torch.bfloat16, None
    )
    torch.xpu.synchronize()

    repeated_quant = []
    repeated_output = []
    for _ in range(16):
        repeat_q, repeat_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
        repeat_output = torch.ops._xpu_C.int8_gemm_w8a8(
            repeat_q,
            repeat_scale,
            weight,
            weight_scale,
            torch.bfloat16,
            None,
        )
        torch.xpu.synchronize()
        repeated_quant.append((repeat_q.cpu(), repeat_scale.cpu()))
        repeated_output.append(repeat_output.cpu())

    actual_q = q.cpu()
    actual_scale = scale.cpu()
    actual = output.cpu()
    error = (actual.float() - expected.float()).abs()
    quant_exact = torch.equal(actual_q, expected_q)
    scale_close = torch.allclose(actual_scale, expected_scale, rtol=1.0e-6, atol=1.0e-8)
    output_close = torch.allclose(actual, expected, rtol=0.02, atol=0.25)
    quant_repeat_exact = all(
        torch.equal(repeat_q, actual_q)
        and torch.equal(repeat_scale, actual_scale)
        for repeat_q, repeat_scale in repeated_quant
    )
    output_repeat_exact = all(
        torch.equal(repeat_output, actual) for repeat_output in repeated_output
    )
    result = {
        "quant_exact": quant_exact,
        "quant_repeat_exact": quant_repeat_exact,
        "scale_max_error": float((actual_scale - expected_scale).abs().max()),
        "output_max_error": float(error.max()),
        "output_mean_error": float(error.mean()),
        "output_close": output_close,
        "output_repeat_exact": output_repeat_exact,
        "weight_shape": list(weight.shape),
        "weight_stride": list(weight.stride()),
    }
    if not (
        quant_exact
        and quant_repeat_exact
        and scale_close
        and output_close
        and output_repeat_exact
        and weight.stride(0) == 1
    ):
        raise RuntimeError(f"native W8A8 numerical oracle failed: {result}")
    return result


def make_case(m: int, k: int, n: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((m, k), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    ).to("xpu")
    weight_nk = torch.randint(
        -127, 128, (n, k), generator=generator, dtype=torch.int8
    ).to("xpu")
    weight_nt = weight_nk.t()
    weight_nn = weight_nt.contiguous()
    weight_scale = (
        torch.rand((n,), generator=generator, dtype=torch.float32) * 0.009 + 0.001
    ).to("xpu")

    def generic_quant():
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min_(1.0e-5)
        scale = absmax * (1.0 / 127.0)
        quant = torch.round(x / scale).clamp_(-127, 127).to(torch.int8)
        return quant, scale

    def native_quant():
        return torch.ops._xpu_C.per_token_quant_int8_xpu(x)

    generic_q, generic_scale = generic_quant()
    native_q, native_scale = native_quant()

    def generic_gemm():
        accumulator = torch._int_mm(generic_q, weight_nn)
        return (
            accumulator.float()
            * generic_scale.float()
            * weight_scale.reshape(1, -1)
        ).to(torch.bfloat16)

    def native_gemm():
        return torch.ops._xpu_C.int8_gemm_w8a8(
            native_q,
            native_scale,
            weight_nt,
            weight_scale,
            torch.bfloat16,
            None,
        )

    def generic_full():
        quant, scale = generic_quant()
        accumulator = torch._int_mm(quant, weight_nn)
        return (
            accumulator.float() * scale.float() * weight_scale.reshape(1, -1)
        ).to(torch.bfloat16)

    def native_full():
        quant, scale = native_quant()
        return torch.ops._xpu_C.int8_gemm_w8a8(
            quant, scale, weight_nt, weight_scale, torch.bfloat16, None
        )

    return {
        "shape": {"m": m, "k": k, "n": n},
        "generic_quant": bench(generic_quant),
        "native_quant": bench(native_quant),
        "generic_gemm_and_scale": bench(generic_gemm),
        "native_gemm": bench(native_gemm),
        "generic_full": bench(generic_full),
        "native_full": bench(native_full),
    }


def main() -> None:
    module = importlib.import_module("vllm_xpu_kernels._xpu_C")
    result = {
        "torch": torch.__version__,
        "extension": module.__file__,
        "device": torch.xpu.get_device_name(0),
        "numerical": numerical_oracle(),
        "cases": [
            make_case(1, 5120, 17408, 1),
            make_case(1, 8704, 5120, 2),
            make_case(128, 5120, 17408, 3),
            make_case(128, 8704, 5120, 4),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
