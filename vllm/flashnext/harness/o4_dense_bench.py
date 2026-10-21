#!/usr/bin/env python3
"""O4: GGUF dense gemv time per type on real Flash-Next tensors (M=1)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness")
from gguf import GGUFReader
from o4_w8_bench_util import gtime
torch.ops.load_library(os.environ["B70_GGUF_SO"])
r = GGUFReader("/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
want = ("attn_gate", "ssm_out", "attn_qkv", "attn_q.", "attn_output")
seen = {}
for t in r.tensors:
    if not any(w in t.name for w in want) or len(t.shape) != 2:
        continue
    key = (t.name.split(".")[2], t.tensor_type.name)
    seen.setdefault(key, []).append(t)
for (kind, ty), ts in sorted(seen.items()):
    ts = ts[:8]
    k, n = int(ts[0].shape[0]), int(ts[0].shape[1])
    qs = [torch.from_numpy(np.ascontiguousarray(t.data).view(np.uint8).reshape(n, -1)).to("xpu") for t in ts]
    x = (torch.randn(1, k) * 0.5).to(torch.bfloat16).to("xpu")
    out = torch.empty(1, n, dtype=torch.bfloat16, device="xpu")
    def run():
        for q in qs:
            torch.ops.b70_gguf.gemv_out(x, q, int(ts[0].tensor_type), n, k, out, 0)
    us = gtime(run) / len(qs) * 1e3
    nb = qs[0].numel()
    ids = torch.zeros(1, 1, dtype=torch.int32, device="xpu")
    def run2():
        for q in qs:
            torch.ops.b70_gguf.moe_gemv(x, q, ids, int(ts[0].tensor_type), n, k)
    us2 = gtime(run2) / len(qs) * 1e3
    ref = torch.ops.b70_gguf.gemv(x, qs[0], int(ts[0].tensor_type), n, k)
    alt = torch.ops.b70_gguf.moe_gemv(x, qs[0], ids, int(ts[0].tensor_type), n, k).reshape(1, n)
    print(f"   row-kernel {us2:7.2f} us {nb/(us2*1e-6)/1e9:5.0f} GB/s rel_err={((alt-ref).abs().max()/ref.abs().max()).item():.2e} bitwise={torch.equal(alt, ref)}", flush=True)
    print(f"{kind:12s} {ty:7s} N={n:6d} K={k:5d} bytes={nb/1e6:6.2f}MB {us:7.2f} us {nb/(us*1e-6)/1e9:5.0f} GB/s (count in file {len(seen[(kind,ty)])})", flush=True)
