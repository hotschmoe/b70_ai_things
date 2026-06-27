#!/usr/bin/env python3
# 124_int_mm_shape_sweep.py -- is torch._int_mm (oneDNN INT8 XMX) CORRECT on XPU for ALL the real
# Qwen3.6-27B W8A8 layer shapes, incl. the FUSED ones (qkv N=14336, gate_up N=34816) and TP=2 half-shards?
# The per-layer validate (123) only exercised down_proj (N=5120). The "!!!!" garbage could be a
# shape-dependent _int_mm correctness bug on the fused/large-N or sharded layers. Reference = exact
# int32 accumulation via float (int8*int8*K stays well within int32 for these K). Any nonzero diff = bug.
import torch

# (label, K, N)  -- full layers + TP=2 shards (col-parallel splits N; row-parallel splits K)
SHAPES = [
    ("q_proj            ", 5120, 12288),
    ("k_proj/v_proj     ", 5120, 1024),
    ("qkv_FUSED         ", 5120, 14336),   # 12288+1024+1024 (sglang fuses q,k,v)
    ("qkv_FUSED tp2-half ", 5120, 7168),
    ("o_proj            ", 6144, 5120),
    ("o_proj tp2-halfK   ", 3072, 5120),
    ("gate/up_proj      ", 5120, 17408),
    ("gate_up_FUSED     ", 5120, 34816),   # 2*17408 (sglang fuses gate,up)
    ("gate_up_FUSED tp2  ", 5120, 17408),
    ("down_proj         ", 17408, 5120),
    ("down_proj tp2-halfK", 8704, 5120),
    ("lm_head-ish bigN   ", 5120, 32768),
    ("odd N=14335        ", 5120, 14335),   # probe non-aligned N
    ("odd N=34815        ", 5120, 34815),
]
MS = [1, 4, 512, 2048]
dev = "xpu"
torch.manual_seed(0)
print(f"{'layer':<20} {'M':>5} {'K':>6} {'N':>6}  {'max|diff|':>10}  {'rel-err':>10}  verdict")
worst = 0.0
for label, K, N in SHAPES:
    wt = torch.randint(-127, 128, (K, N), dtype=torch.int8, device=dev)
    for M in MS:
        x = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
        got = torch._int_mm(x, wt).to(torch.float64)            # [M,N] int32 -> f64
        ref = (x.to(torch.float64) @ wt.to(torch.float64))      # exact (values small enough)
        diff = (got - ref).abs()
        md = diff.max().item()
        rel = (md / ref.abs().max().clamp(min=1).item())
        worst = max(worst, md)
        v = "OK" if md == 0 else (">>> WRONG <<<" if rel > 1e-6 else "rounding?")
        if md != 0 or M in (1, 512):
            print(f"{label:<20} {M:>5} {K:>6} {N:>6}  {md:>10.1f}  {rel:>10.2e}  {v}")
print(f"\nWORST max|diff| across all shapes = {worst}   -> {'_int_mm BUG' if worst>0 else 'torch._int_mm CORRECT for all shapes'}")
