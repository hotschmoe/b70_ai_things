#!/usr/bin/env python3
"""Card-local XPUGraph oracle for the native W8A8 linear chain."""

from __future__ import annotations

import importlib
import json
import statistics
import time

import torch


def main() -> None:
    module = importlib.import_module("vllm_xpu_kernels._xpu_C")
    generator = torch.Generator().manual_seed(20260826)
    host_x = torch.randn((1, 5120), generator=generator).to(torch.bfloat16)
    host_weight_nk = torch.randint(
        -127, 128, (5120, 5120), generator=generator, dtype=torch.int8
    )
    host_scale = torch.rand((5120,), generator=generator) * 0.009 + 0.001

    x = host_x.to("xpu")
    weight_nk = host_weight_nk.to("xpu")
    weight = weight_nk.t()
    weight_scale = host_scale.to("xpu")

    def forward():
        quant, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
        return torch.ops._xpu_C.int8_gemm_w8a8(
            quant,
            scale,
            weight,
            weight_scale,
            torch.bfloat16,
            None,
        )

    for _ in range(4):
        eager = forward()
    torch.xpu.synchronize()
    expected = eager.cpu()

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        static_output = forward()
    torch.xpu.synchronize()

    replay_exact = []
    for _ in range(16):
        graph.replay()
        torch.xpu.synchronize()
        replay_exact.append(torch.equal(static_output.cpu(), expected))
    if not all(replay_exact):
        raise RuntimeError(f"XPUGraph replay mismatch: {replay_exact}")

    def measure(call, count=200):
        samples = []
        for _ in range(9):
            torch.xpu.synchronize()
            started = time.perf_counter()
            for _ in range(count):
                call()
            torch.xpu.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0 / count)
        return {
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }

    result = {
        "torch": torch.__version__,
        "extension": module.__file__,
        "device": torch.xpu.get_device_name(0),
        "shape": {"m": 1, "k": 5120, "n": 5120},
        "weight_stride": list(weight.stride()),
        "replays": len(replay_exact),
        "replay_exact": all(replay_exact),
        "eager": measure(forward),
        "graph": measure(graph.replay),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
