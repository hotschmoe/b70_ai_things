#!/usr/bin/env python3
"""Slot vs grouped NVFP4 MoE apply agreement + a tiny XPU graph capture.

Run inside int8g-v0260 with the fused _xpu_C mounted and PYTHONPATH on patches/.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/opt/nvfp4_shim")
import fused_moe_apply as fma  # noqa: E402
import vllm_xpu_kernels._xpu_C  # noqa: F401, E402


def pack(E, rows, k_half, dev):
    return torch.randint(0, 256, (E, rows, k_half), dtype=torch.uint8, device=dev)


def main():
    assert hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"), "no nvfp4_gemm_w4a16"
    dev = "xpu:0"
    torch.xpu.set_device(0)
    E, H, I, T, K = 8, 64, 32, 2, 3
    w1 = pack(E, 2 * I, H // 2, dev)
    w2 = pack(E, H, I // 2, dev)
    s13 = (torch.randn(E, H // 16, 2 * I, device=dev) * 0.05).to(torch.bfloat16)
    s2 = (torch.randn(E, I // 16, H, device=dev) * 0.05).to(torch.bfloat16)
    xb = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    ids_share = torch.tensor([[0, 2, 5], [2, 3, 0]], dtype=torch.long, device=dev)
    ids_solo = torch.tensor([[1, 4, 7]], dtype=torch.long, device=dev)
    wts = torch.rand(T, K, dtype=torch.bfloat16, device=dev)
    wts = wts / wts.sum(dim=-1, keepdim=True)

    def agree(tag, ids, x, w):
        ts, hs = x.shape
        out_s = torch.empty(ts, hs, dtype=torch.bfloat16, device=dev)
        out_g = torch.empty(ts, hs, dtype=torch.bfloat16, device=dev)
        fma.apply_slots(out_s, x, w1, w2, s13, s2, w, ids, False)
        fma.apply_grouped(out_g, x, w1, w2, s13, s2, w, ids, None, False)
        torch.xpu.synchronize()
        a = out_s.float().cpu()
        b = out_g.float().cpu()
        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        denom = b.abs().clamp(min=1e-3)
        rel = ((a - b).abs() / denom).max().item()
        print(f"{tag} cos={cos:.6f} relmax={rel:.4e} nrm={b.norm().item():.3f}")
        # Shared-expert rows use M=2 grouped vs 2x M=1 slots; oneDNN NVFP4
        # is not bit-exact across M. Solo (no share) must stay exact.
        if tag == "solo":
            if cos < 0.999999 or rel > 1e-6:
                print("FAIL agreement", tag)
                sys.exit(1)
        elif cos < 0.999:
            print("FAIL agreement", tag)
            sys.exit(1)

    agree("share", ids_share, xb, wts)
    agree("solo", ids_solo, xb[0:1], wts[0:1])

    # Capture the slot path (the GRAPH-sensitive one). Replay with new ids.
    xb1 = xb[0:1].contiguous()
    ids1 = ids_share[0:1].contiguous()
    wts1 = wts[0:1].contiguous()
    out_c = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
    g = torch.xpu.XPUGraph()
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            fma.apply_slots(out_c, xb1, w1, w2, s13, s2, wts1, ids1, False)
        torch.xpu.current_stream().synchronize()
        with torch.xpu.graph(g):
            fma.apply_slots(out_c, xb1, w1, w2, s13, s2, wts1, ids1, False)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()
    print("capture OK")

    ids1.copy_(torch.tensor([[1, 4, 7]], dtype=torch.long, device=dev))
    out_eager = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
    fma.apply_slots(out_eager, xb1, w1, w2, s13, s2, wts1, ids1, False)
    g.replay()
    torch.xpu.synchronize()
    ae = out_c.float().cpu()
    be = out_eager.float().cpu()
    cos2 = torch.nn.functional.cosine_similarity(ae.flatten(), be.flatten(), dim=0)
    mx2 = (ae - be).abs().max().item()
    print(f"replay-vs-eager cos={float(cos2):.6f} max={mx2:.4e}")
    if float(cos2) < 0.999:
        print("FAIL replay used capture-time expert ids (view-not-copy)")
        sys.exit(2)
    print("PASS")


if __name__ == "__main__":
    main()
