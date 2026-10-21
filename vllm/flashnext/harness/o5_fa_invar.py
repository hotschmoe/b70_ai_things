#!/usr/bin/env python3
"""O5: FA paged decode, MTP verify rows vs sequential 1-token decode.

One card. Flash-Next attention shape (24 q heads? taken from O5_HQ, 2 KV
heads, head 256), KV block 1600 tokens. For each past length, a q_len=3
verify (spec fast path, max_seqlen_k = past+3) against three q_len=1
decodes (max_seqlen_k = past+j+1, as vLLM passes it). Bitwise compare.
Also batch 2 (two sequences) vs batch 1 for the q_len=1 path.
Run with and without B70_FA_KV_SPLITS to see the split heuristic effect.
"""

import json
import os
import time

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

dev = "xpu:0"
torch.manual_seed(7)
HQ = int(os.environ.get("O5_HQ", "16"))
HKV, D, BS = 2, 256, 1600
NB = 40  # 64000 tokens of KV
k = torch.randn(NB, BS, HKV, D, device=dev, dtype=torch.bfloat16)
v = torch.randn_like(k)
q = torch.randn(3, HQ, D, device=dev, dtype=torch.bfloat16)
q2 = torch.randn(6, HQ, D, device=dev, dtype=torch.bfloat16)


def i32(x):
    return torch.tensor(x, device=dev, dtype=torch.int32)


table = i32([list(range(NB))])
table2 = i32([list(range(NB)), list(range(NB))[::-1]])
res = []
for past in (16, 900, 1700, 5000, 12000, 30000, 60000):
    spec = flash_attn_varlen_func(q=q, k=k, v=v, max_seqlen_q=3, cu_seqlens_q=i32([0, 3]),
                                  seqused_k=i32([past + 3]), max_seqlen_k=past + 3,
                                  block_table=table, causal=True)
    seq = torch.cat([
        flash_attn_varlen_func(q=q[j:j + 1], k=k, v=v, max_seqlen_q=1, cu_seqlens_q=i32([0, 1]),
                               seqused_k=i32([past + j + 1]), max_seqlen_k=past + j + 1,
                               block_table=table, causal=True)
        for j in range(3)
    ])
    # batch-2 decode (other sequence much longer) vs batch-1 decode for row 0
    b2 = flash_attn_varlen_func(q=q2[:2], k=k, v=v, max_seqlen_q=1, cu_seqlens_q=i32([0, 1, 2]),
                                seqused_k=i32([past + 1, 63000]), max_seqlen_k=63000,
                                block_table=table2, causal=True)
    b1 = flash_attn_varlen_func(q=q2[:1], k=k, v=v, max_seqlen_q=1, cu_seqlens_q=i32([0, 1]),
                                seqused_k=i32([past + 1]), max_seqlen_k=past + 1,
                                block_table=table, causal=True)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        flash_attn_varlen_func(q=q[:1], k=k, v=v, max_seqlen_q=1, cu_seqlens_q=i32([0, 1]),
                               seqused_k=i32([past + 1]), max_seqlen_k=past + 1,
                               block_table=table, causal=True)
    torch.xpu.synchronize()
    us = (time.perf_counter() - t0) / 20 * 1e6
    rec = dict(past=past, splits=os.environ.get("B70_FA_KV_SPLITS", "auto"),
               verify_rows_equal=[bool(torch.equal(spec[j], seq[j])) for j in range(3)],
               verify_max_abs=float((spec.float() - seq.float()).abs().max()),
               batch2_row0_equal=bool(torch.equal(b2[0], b1[0])),
               decode_us_eager=round(us, 1))
    res.append(rec)
    print(json.dumps(rec), flush=True)
print("DONE", flush=True)
