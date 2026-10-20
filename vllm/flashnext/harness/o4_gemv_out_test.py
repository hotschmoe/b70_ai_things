#!/usr/bin/env python3
"""O4: b70_gguf::gemv_out (bf16 epilogue, strided) vs gemv(...).to(bf16), real tensors."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
from gguf import GGUFReader
torch.ops.load_library(os.environ["B70_GGUF_SO"])
r = GGUFReader("/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
seen = {}
for t in r.tensors:
    if "_exps" in t.name or t.tensor_type.name in ("BF16", "F32", "F16") or "token_embd" in t.name or "output.weight" == t.name:
        continue
    if t.tensor_type.name in seen or len(t.shape) != 2:
        continue
    seen[t.tensor_type.name] = t
bad = 0
for name, t in seen.items():
    k, n = int(t.shape[0]), int(t.shape[1])
    q = torch.from_numpy(np.ascontiguousarray(t.data).view(np.uint8).reshape(n, -1)).to("xpu")
    for m in (1, 2, 3):
        x = (torch.randn(m, k) * 0.5).to(torch.bfloat16).to("xpu")
        ref = torch.ops.b70_gguf.gemv(x, q, int(t.tensor_type), n, k).to(torch.bfloat16)
        out = torch.full((m, n + 7), 7.0, dtype=torch.bfloat16, device="xpu")
        torch.ops.b70_gguf.gemv_out(x, q, int(t.tensor_type), n, k, out, 3)
        same = torch.equal(out[:, 3:3 + n], ref) and torch.all(out[:, :3] == 7).item() and torch.all(out[:, 3 + n:] == 7).item()
        bad += 0 if same else 1
        print(f"{name:8s} {t.name:28s} N={n} K={k} M={m} identical={same}", flush=True)
print("O4_GEMV_OUT", "OK" if bad == 0 else f"FAIL {bad}")
