#!/usr/bin/env python3
"""oneDNN 8x (up + silu_and_mul + down) baseline vs O4e fused layerlet.

ZE_AFFINITY_MASK selects the card (use 1 while ornith_o1 holds 0).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
import vllm_xpu_kernels._xpu_C  # noqa: F401


def silu_and_mul(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def main():
    assert hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"), "no nvfp4_gemm_w4a16"
    dev = "xpu:0"
    torch.xpu.set_device(0)
    print("device", torch.xpu.get_device_name(0))
    S, H, I = 8, 2048, 512
    nup = 2 * I
    warm, iters = 8, 30
    x = torch.randn(1, H, dtype=torch.bfloat16, device=dev)
    w1 = [
        torch.randint(0, 256, (nup, H // 2), dtype=torch.uint8, device=dev)
        for _ in range(S)
    ]
    s1 = [
        (torch.randn(H // 16, nup, device=dev) * 0.05).to(torch.bfloat16)
        for _ in range(S)
    ]
    w2 = [
        torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, device=dev)
        for _ in range(S)
    ]
    s2 = [
        (torch.randn(I // 16, H, device=dev) * 0.05).to(torch.bfloat16)
        for _ in range(S)
    ]
    wt1 = [w.transpose(0, 1) for w in w1]
    wt2 = [w.transpose(0, 1) for w in w2]

    def once():
        ys = []
        for i in range(S):
            gu = torch.ops._xpu_C.nvfp4_gemm_w4a16(x, wt1[i], None, s1[i], 16)
            h = silu_and_mul(gu).to(torch.bfloat16)
            dn = torch.ops._xpu_C.nvfp4_gemm_w4a16(h, wt2[i], None, s2[i], 16)
            ys.append(dn)
        return ys

    for _ in range(warm):
        once()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    print(f"onednn 8x(up+silu+down)  {ms:.3f} ms")
    print("PASS")


if __name__ == "__main__":
    main()
