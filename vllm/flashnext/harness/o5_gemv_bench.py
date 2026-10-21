#!/usr/bin/env python3
"""O5: dense GGUF GEMV at decode/verify widths M=1..8 on real Flash-Next weights.

One card (pin with gpu-run --card N + xpu-run --card N; device xpu:0 here).
For each dense tensor class in two layers (GDN layer 14, attention layer 15)
plus the Q5_K head, reports per M: mean us (graph-free, 200 timed calls),
effective GB/s of weight bytes, and per-row bit equality versus M=1 of the
same row (batch invariance). Also times torch F.linear (oneDNN) for the BF16
router/HC/ssm shapes and checks their invariance.

Env: O5_OUT (json path dir), B70_GGUF_SO (kernel library), O5_LAYERS.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
import gguf  # noqa: E402

import b70_gguf_xpu  # noqa: E402

b70_gguf_xpu.load()
try:
    import b70_o4_xpu

    b70_o4_xpu.load()
except Exception as exc:  # noqa: BLE001
    print("no b70_o4", exc)
GG = "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
OUT = os.environ.get("O5_OUT", ".")
LAYERS = [int(v) for v in os.environ.get("O5_LAYERS", "14,15").split(",")]
dev = "xpu:0"
torch.manual_seed(5)
r = gguf.GGUFReader(GG)
tens = {t.name: t for t in r.tensors}


def timeit(fn, iters=200):
    for _ in range(10):
        fn()
    torch.xpu.synchronize()
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


results = []
names = []
for L in LAYERS:
    for suffix in ("attn_qkv", "attn_gate", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_output",
                   "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"):
        n = f"blk.{L}.{suffix}.weight"
        if n in tens:
            names.append(n)
names.append("output.weight")
for n in names:
    t = tens[n]
    qt = int(t.tensor_type)
    K, N = int(t.shape[0]), int(t.shape[1])
    raw = np.asarray(t.data)
    w = torch.from_numpy(raw.reshape(-1).view(np.uint8).copy()).to(dev)
    nbytes = w.numel()
    if t.tensor_type.name in ("BF16", "F32", "F16"):
        continue
    x = (torch.randn(8, K, device=dev) * 0.5).to(torch.bfloat16)
    ref = torch.ops.b70_gguf.gemv(x[:1], w, qt, N, K)
    rows1 = [torch.ops.b70_gguf.gemv(x[i : i + 1], w, qt, N, K) for i in range(8)]
    for M in (1, 2, 3, 4, 6, 8):
        y = torch.ops.b70_gguf.gemv(x[:M], w, qt, N, K)
        inv = all(torch.equal(y[i : i + 1], rows1[i]) for i in range(M))
        us = timeit(lambda: torch.ops.b70_gguf.gemv(x[:M], w, qt, N, K), 100 if N > 100000 else 200)
        rec = dict(t=n, type=t.tensor_type.name, N=N, K=K, M=M, us=round(us, 2),
                   gbs=round(nbytes / us / 1e3, 1), invariant=inv,
                   h=hashlib.md5(y.cpu().numpy().tobytes()).hexdigest()[:12])
        results.append(rec)
        print(json.dumps(rec), flush=True)
    del w

if os.environ.get("O5_SKIP_BF16") == "1":
    LAYERS_BF = []
else:
    LAYERS_BF = LAYERS
for L in LAYERS_BF:
    for suffix in ("ffn_gate_inp", "ssm_alpha", "ssm_beta", "hc_attn_down", "hc_attn_up", "hc_attn_inject"):
        n = f"blk.{L}.{suffix}.weight"
        if n not in tens:
            continue
        t = tens[n]
        K, N = int(t.shape[0]), int(t.shape[1])
        raw = np.asarray(t.data)
        wt = torch.from_numpy(raw.reshape(-1).view(np.uint16).copy()).view(torch.bfloat16).reshape(N, K).to(dev)
        x = (torch.randn(8, K, device=dev) * 0.5).to(torch.bfloat16)
        rows1 = [F.linear(x[i : i + 1], wt) for i in range(8)]
        for M in (1, 2, 3, 4, 6, 8):
            y = F.linear(x[:M], wt)
            inv = all(torch.equal(y[i : i + 1], rows1[i]) for i in range(M))
            us = timeit(lambda: F.linear(x[:M], wt))
            rec = dict(t=n, type="BF16-onednn", N=N, K=K, M=M, us=round(us, 2), invariant=inv)
            if hasattr(torch.ops, "b70_o4") and K % 8 == 0:
                try:
                    yb = torch.ops.b70_o4.bf_gemv(x[:M].contiguous(), wt, 4)
                    rb = [torch.ops.b70_o4.bf_gemv(x[i : i + 1].contiguous(), wt, 4) for i in range(M)]
                    rec["bf_gemv_invariant"] = all(torch.equal(yb[i : i + 1], rb[i]) for i in range(M))
                    rec["bf_gemv_us"] = round(timeit(lambda: torch.ops.b70_o4.bf_gemv(x[:M].contiguous(), wt, 4)), 2)
                except Exception as exc:  # noqa: BLE001
                    rec["bf_gemv_err"] = str(exc)[:80]
            results.append(rec)
            print(json.dumps(rec), flush=True)

with open(os.path.join(OUT, os.environ.get("O5_BENCH_NAME", "gemv_bench.json")), "w") as fh:
    json.dump(results, fh, indent=1)
print("DONE", flush=True)
