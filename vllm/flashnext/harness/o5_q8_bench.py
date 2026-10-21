#!/usr/bin/env python3
"""O5: Q8_0 GEMV (MTP drafter rt dense weights) dense_k vs KREG (B70_KREG_Q8=1).

One card. Times gemv at M=1..4 for the drafter's Q8_0 tensors (eager, 200
calls) and saves outputs; when O5_Q8_REF names a previous run's .pt, also
reports max relative error against it. Row invariance (M rows vs M=1 calls)
is checked in-process.
"""

import json
import os

import numpy as np
import torch

import b70_gguf_xpu

b70_gguf_xpu.load()
dev = "xpu:0"
OUT = os.environ["O5_OUT"]
TAG = os.environ.get("O5_TAG", "q8")
RT = "/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt"
torch.manual_seed(3)
rows = {}
for line in open(os.path.join(RT, "dense.txt")):
    name, kind, n, k, off, size = line.split()
    if kind == "q8_0" and int(k) % 256 == 0:
        rows[name] = (int(n), int(k), int(off), int(size))


def timeit(fn, iters=200):
    for _ in range(10):
        fn()
    torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


ref = torch.load(os.environ["O5_Q8_REF"]) if os.environ.get("O5_Q8_REF") else None
outs = {}
for name, (n, k, off, size) in rows.items():
    raw = np.fromfile(os.path.join(RT, "dense.bin"), dtype=np.uint8, count=size, offset=off)
    w = torch.from_numpy(raw).to(dev)
    x = (torch.randn(4, k, device=dev) * 0.5).to(torch.bfloat16)
    one = [torch.ops.b70_gguf.gemv(x[i:i + 1], w, 8, n, k) for i in range(4)]
    for M in (1, 2, 3, 4):
        y = torch.ops.b70_gguf.gemv(x[:M], w, 8, n, k)
        inv = all(torch.equal(y[i:i + 1], one[i]) for i in range(M))
        us = timeit(lambda: torch.ops.b70_gguf.gemv(x[:M], w, 8, n, k))
        rec = dict(tag=TAG, t=name, N=n, K=k, M=M, us=round(us, 2), gbs=round(size / us / 1e3, 1), invariant=inv)
        outs[(name, M)] = y.float().cpu()
        if ref is not None:
            r = ref[(name, M)]
            rec["max_rel"] = float((outs[(name, M)] - r).abs().max() / r.abs().max().clamp_min(1e-30))
        print(json.dumps(rec), flush=True)
torch.save(outs, os.path.join(OUT, f"q8-{TAG}.pt"))
print("DONE", flush=True)
