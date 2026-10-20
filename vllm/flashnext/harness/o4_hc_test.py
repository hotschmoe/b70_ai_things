#!/usr/bin/env python3
"""O4: fused decode HC (xpu/hc_decode.py) vs the eager HyperConnection.

One card. Real Flash-Next shapes (hc 4, n_embd 2560, lowrank 320).
Random weights scaled like the checkpoint. Reports output differences
(bf16 ulps) and XPUGraph replay time of 48 x (mix + combine) x 2 for
both paths at T = 1, 2, 4.
"""

from __future__ import annotations

import os
import sys
import types

import torch

os.environ.pop("B70_DECODE_HC", None)
from vllm.models.qwen4_exp.xpu import hc_decode  # noqa: E402
from vllm.models.qwen4_exp.xpu.model import HyperConnection  # noqa: E402


def make(has_inject: bool, seed: int):
    cfg = types.SimpleNamespace(hc_count=4, hidden_size=2560, hc_lowrank=320, rms_norm_eps=1e-6)
    hc = HyperConnection(cfg, has_inject=has_inject)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        hc.hc_norm.weight.copy_(torch.randn(10240, generator=g) * 0.05 - 0.06)
        hc.input_mix_weight_down.weight.copy_(torch.randn(320, 10240, generator=g) * 0.01)
        hc.input_mix_weight_up.weight.copy_(torch.randn(10240, 320, generator=g) * 0.05)
        if has_inject:
            hc.block_inject_weight.weight.copy_(torch.randn(4, 10240, generator=g) * 0.01)
    return hc.to(device="xpu", dtype=torch.bfloat16)


def ulps(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float, float]:
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    d = (ai - bi).abs()
    return int(d.max()), float((d > 0).float().mean()), float((a.float() - b.float()).abs().max())


def graph_ms(fn, n=50) -> float:
    fn()
    torch.xpu.synchronize()
    gr = torch.xpu.XPUGraph()
    with torch.xpu.graph(gr):
        fn()
    torch.xpu.synchronize()
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        gr.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / n


def main() -> None:
    torch.manual_seed(0)
    layers = [(make(True, 10 + i), make(True, 100 + i)) for i in range(48)]
    final = make(False, 7)
    ok = True
    for t in (1, 2, 4):
        x = (torch.randn(t, 4, 2560, device="xpu") * 0.5).to(torch.bfloat16)
        blk = (torch.randn(t, 2560, device="xpu") * 0.5).to(torch.bfloat16)
        for name, hc in (("attn", layers[0][0]), ("final", final)):
            ref_m, ref_i = hc.mix(x)
            hc_decode.prepare(hc)
            m2, i2 = hc_decode.mix(hc, x)
            mu = ulps(ref_m, m2)
            line = f"T={t} {name} mix max_ulp={mu[0]} frac_diff={mu[1]:.4f} max_abs={mu[2]:.3e}"
            if ref_i is not None:
                iu = ulps(ref_i, i2)
                line += f" inject max_ulp={iu[0]} frac_diff={iu[1]:.4f}"
                r_ref = hc.combine(x, blk, ref_i)
                r2 = hc_decode.combine(hc, x, blk, ref_i)
                cu = ulps(r_ref, r2)
                line += f" combine(same inject) max_ulp={cu[0]} frac_diff={cu[1]:.4f}"
            print(line, flush=True)
            if mu[0] > 64:
                ok = False

        def eager_stack():
            r = x
            for a, b in layers:
                m, i = a.mix(r)
                r = a.combine(r, blk, i)
                m, i = b.mix(r)
                r = b.combine(r, blk, i)
            final.mix(r)

        def fused_stack():
            r = x
            for a, b in layers:
                m, i = hc_decode.mix(a, r)
                r = hc_decode.combine(a, r, blk, i)
                m, i = hc_decode.mix(b, r)
                r = hc_decode.combine(b, r, blk, i)
            hc_decode.mix(final, r)

        for a, b in layers:
            hc_decode.prepare(a)
            hc_decode.prepare(b)
        te = graph_ms(eager_stack)
        tf = graph_ms(fused_stack)
        print(f"T={t} graph 97 mix + 96 combine: eager {te:.3f} ms fused {tf:.3f} ms saved {te - tf:.3f} ms", flush=True)
    print("O4_HC_TEST", "OK" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
