#!/usr/bin/env python3
"""N1: real GDN b|a weights, exact O4 decode parity at prefill widths.

No model load. Requires card pin through xpu-run and bin/gpu-run. Reports
oneDNN repeat/batch behavior diagnostically; gates only N1 fixed arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import time

import gguf
import numpy as np
import torch
import torch.nn.functional as F
import b70_n1_xpu
import b70_o4_xpu

signal.alarm(120)
OUT = Path(os.environ["N1_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
b70_n1_xpu.load()
b70_o4_xpu.load()
GG = Path(os.environ.get("B70_GGUF_PATH", "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"))
reader = gguf.GGUFReader(str(GG))
tensors = {t.name: t for t in reader.tensors}
results = []
start = time.monotonic()


def timed(fn, count=20):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    begin, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    begin.record()
    for _ in range(count):
        fn()
    end.record()
    torch.xpu.synchronize()
    return begin.elapsed_time(end) * 1000 / count


def weight(layer):
    rows = []
    for suffix in ("ssm_beta", "ssm_alpha"):
        t = tensors[f"blk.{layer}.{suffix}.weight"]
        assert t.tensor_type.name == "BF16", (t.name, t.tensor_type)
        k, n = map(int, t.shape)
        rows.append(torch.from_numpy(np.asarray(t.data).reshape(-1).view(np.uint16).copy())
                    .view(torch.bfloat16).reshape(n, k))
    w = torch.cat(rows).contiguous()
    return w.to("xpu:0"), hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()


for layer in (0, 2, 14):
    w, whash = weight(layer)
    n, k = w.shape
    ks = 2 if n >= 256 else 8
    for seed in (5, 17):
        torch.manual_seed(seed)
        x = torch.randn((12288, k), device="xpu:0", dtype=torch.bfloat16) * 0.5
        op = lambda a: torch.ops.b70_n1.bf_gemm(a.contiguous(), w)
        for m in (1, 2, 3, 4, 8, 9, 20, 40, 60, 128, 257, 12288):
            a = x[:m]
            y = op(a)
            rows = list(range(m)) if m <= 60 else sorted({0, 1, 7, 8, m // 2, m - 1})
            assert all(torch.equal(y[i:i+1], torch.ops.b70_o4.bf_gemv(a[i:i+1], w, ks)) for i in rows), ("O4 parity", layer, seed, m)
            assert torch.equal(y, op(a)), ("repeat", layer, seed, m)
            # Shift token-tile boundaries and reverse row order. Both must
            # preserve each row, including partial tails of an eight-row tile.
            if m > 1:
                assert torch.equal(y[1:], op(a[1:])), ("shift", layer, seed, m)
                assert torch.equal(y.flip(0), op(a.flip(0))), ("reverse", layer, seed, m)
            rec = dict(layer=layer, seed=seed, M=m, N=n, K=k, weight_sha256=whash,
                       checked_rows=len(rows), exact_o4=True, repeat=True, shift_reverse=True)
            if seed == 5 and layer == 2:
                rec["n1_us"] = timed(lambda: op(a))
                rec["onednn_us"] = timed(lambda: F.linear(a, w))
            results.append(rec)
            print(json.dumps(rec), flush=True)
        if layer == 2 and seed == 5:
            for m in (1, 3, 8, 20, 40, 257):
                a = x[:m].clone()
                op(a)
                torch.xpu.synchronize()
                stream = torch.xpu.Stream()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.stream(stream):
                    with torch.xpu.graph(graph, stream=stream):
                        got = op(a)
                torch.xpu.synchronize()
                for repeat in range(8):
                    a.copy_(x[repeat:repeat+m])
                    graph.replay()
                    torch.xpu.synchronize()
                    assert torch.equal(got, op(a)), ("graph", m, repeat)
                print(json.dumps(dict(graph_M=m, replays=8, exact=True)), flush=True)
            a = x[:20]
            original = F.linear(a, w)
            print(json.dumps(dict(onednn_repeat_M20=[torch.equal(original, F.linear(a, w)) for _ in range(100)])), flush=True)
            # Numerical change relative to oneDNN is expected: report scale.
            ref = F.linear(a.float(), w.float())
            results.append(dict(layer=layer, maxabs_vs_fp32=(op(a).float()-ref).abs().max().item(),
                                onednn_maxabs_vs_fp32=(original.float()-ref).abs().max().item()))
(OUT / "primitive.json").write_text(json.dumps(dict(results=results, seconds=time.monotonic()-start), indent=2)+"\n")
print("N1_PRIMITIVE_OK", flush=True)
