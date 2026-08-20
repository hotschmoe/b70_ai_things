#!/usr/bin/env python3
"""oneDNN nvfp4_gemm_w4a16 M=1 baseline for the O4 proto shapes.

Run inside int8g-v0260 with the fused _xpu_C mounted. ZE_AFFINITY_MASK selects
the card (use 1 while ornith_o3 holds 0).
"""
from __future__ import annotations

import time

import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401


def bench(tag, n, k, warm=8, iters=30):
    assert hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"), "no nvfp4_gemm_w4a16"
    dev = "xpu:0"
    torch.xpu.set_device(0)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
    scale = (torch.randn(k // 16, n, device=dev) * 0.05).to(torch.bfloat16)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=dev)
    wt = w.transpose(0, 1)
    y = None
    for _ in range(warm):
        y = torch.ops._xpu_C.nvfp4_gemm_w4a16(x, wt, None, scale, 16)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = torch.ops._xpu_C.nvfp4_gemm_w4a16(x, wt, None, scale, 16)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    bytes_ = n * (k / 2) + k * 2 + n * (k / 16) * 2
    gbs = bytes_ / (ms * 1e-3) / 1e9
    print(
        f"onednn {tag} N={n} K={k}  {ms:.3f} ms  {gbs:.1f} GB/s  "
        f"out={tuple(y.shape)} {y.dtype}"
    )
    return ms


def main():
    print("device", torch.xpu.get_device_name(0))
    bench("up", 1024, 2048)
    bench("down", 2048, 512)
    # 8 independent M=1 ups: the current slot loop
    t0 = time.perf_counter()
    warm, iters = 8, 30
    n, k = 1024, 2048
    dev = "xpu:0"
    ws = [
        torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
        for _ in range(8)
    ]
    scales = [
        (torch.randn(k // 16, n, device=dev) * 0.05).to(torch.bfloat16)
        for _ in range(8)
    ]
    x = torch.randn(1, k, dtype=torch.bfloat16, device=dev)
    wts = [w.transpose(0, 1) for w in ws]
    for _ in range(warm):
        for i in range(8):
            torch.ops._xpu_C.nvfp4_gemm_w4a16(x, wts[i], None, scales[i], 16)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for i in range(8):
            torch.ops._xpu_C.nvfp4_gemm_w4a16(x, wts[i], None, scales[i], 16)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    print(f"onednn 8x up (slot loop)  {ms:.3f} ms")
    print("PASS")


if __name__ == "__main__":
    main()
