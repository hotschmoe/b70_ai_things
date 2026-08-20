#!/usr/bin/env python3
"""O4c: b70_nvfp4_m1.gemv vs oneDNN nvfp4_gemm_w4a16 + XPUGraph.

ZE_AFFINITY_MASK selects the card. Requires fused _xpu_C and
B70_NVFP4_M1_SO pointing at b70_nvfp4_m1_gemv.so.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401

SO = os.environ.get("B70_NVFP4_M1_SO", "/work/b70_nvfp4_m1_gemv.so")
torch.ops.load_library(SO)
assert hasattr(torch.ops, "b70_nvfp4_m1"), "library missing"
assert hasattr(torch.ops.b70_nvfp4_m1, "gemv"), "gemv missing"


def bench(tag, n, k, warm=8, iters=30):
    dev = "xpu:0"
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
    scale = (torch.randn(k // 16, n, device=dev) * 0.05).to(torch.bfloat16)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=dev)
    wt = w.transpose(0, 1)
    y_ref = torch.ops._xpu_C.nvfp4_gemm_w4a16(x, wt, None, scale, 16)
    y = torch.ops.b70_nvfp4_m1.gemv(x, w, scale)
    torch.xpu.synchronize()
    a = y.float().flatten().cpu()
    b = y_ref.float().flatten().cpu()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    max_abs = float((a - b).abs().max())
    ref_abs = float(b.abs().max())
    rel = max_abs / max(ref_abs, 1e-3)
    for _ in range(warm):
        y = torch.ops.b70_nvfp4_m1.gemv(x, w, scale)
        y_ref = torch.ops._xpu_C.nvfp4_gemm_w4a16(x, wt, None, scale, 16)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = torch.ops.b70_nvfp4_m1.gemv(x, w, scale)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        y_ref = torch.ops._xpu_C.nvfp4_gemm_w4a16(x, wt, None, scale, 16)
    torch.xpu.synchronize()
    ms_d = (time.perf_counter() - t0) * 1000.0 / iters
    print(
        f"{tag} N={n} K={k} cos={cos:.6f} max_abs={max_abs:.4f} "
        f"rel={rel:.4e} m1={ms:.3f} ms onednn={ms_d:.3f} ms x={ms_d / ms:.2f}"
    )
    # vs oneDNN f4-decompress: not bit-exact. LOOP 12 CPU was 1.6e-7.
    if cos < 0.999 or rel > 2e-2:
        print("FAIL agreement", tag)
        sys.exit(1)
    return ms, ms_d


def graph_check():
    dev = "xpu:0"
    n, k = 1024, 2048
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
    scale = (torch.randn(k // 16, n, device=dev) * 0.05).to(torch.bfloat16)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=dev)
    out_c = torch.empty(1, n, dtype=torch.bfloat16, device=dev)
    g = torch.xpu.XPUGraph()
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            out_c.copy_(torch.ops.b70_nvfp4_m1.gemv(x, w, scale))
        torch.xpu.current_stream().synchronize()
        with torch.xpu.graph(g):
            out_c.copy_(torch.ops.b70_nvfp4_m1.gemv(x, w, scale))
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()
    print("capture OK")
    w.copy_(torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev))
    eager = torch.ops.b70_nvfp4_m1.gemv(x, w, scale)
    g.replay()
    torch.xpu.synchronize()
    ae = out_c.float().cpu().flatten()
    be = eager.float().cpu().flatten()
    cos = float(torch.nn.functional.cosine_similarity(ae, be, dim=0))
    print(f"replay-vs-eager cos={cos:.6f}")
    if cos < 0.999:
        print("FAIL replay")
        sys.exit(2)


def main():
    print("device", torch.xpu.get_device_name(0))
    bench("up", 1024, 2048)
    bench("down", 2048, 512)
    graph_check()
    print("PASS")


if __name__ == "__main__":
    main()
