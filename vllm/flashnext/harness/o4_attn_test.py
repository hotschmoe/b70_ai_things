#!/usr/bin/env python3
"""O4: split-K paged decode (xpu/attn_decode.py) vs fp32 reference and the
XPU FlashAttention decode on Flash-Next shapes (24 q / 2 kv heads, D 256,
page 1600). Graph timing per call."""
import sys
import torch
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness")
from o4_w8_bench_util import gtime
from vllm.models.qwen4_exp.xpu import attn_decode as ad
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

NQH, KVH, D, P = 24, 2, 256, 1600
scale = D ** -0.5
torch.manual_seed(0)
for L in (300, 800, 1700, 4000, 32000):
    pages = (L + P - 1) // P
    nb = pages * 2 + 3
    kv = (torch.randn(nb, KVH, P, 2 * D, device="xpu") * 0.5).to(torch.bfloat16)
    bt = torch.randperm(nb, device="xpu")[:pages].to(torch.int32).view(1, pages)
    sl = torch.tensor([L], dtype=torch.int32, device="xpu")
    q = (torch.randn(1, NQH, D, device="xpu")).to(torch.bfloat16)
    out = torch.empty_like(q)
    ad.decode(q, kv, bt, sl, out, scale, KVH, 32)
    # reference
    kc = kv[bt[0].long()].permute(1, 0, 2, 3).reshape(KVH, pages * P, 2 * D)[:, :L]
    k = kc[..., :D].float(); v = kc[..., D:].float()
    qf = q.float().view(KVH, NQH // KVH, D)
    att = torch.softmax(qf @ k.transpose(1, 2) * scale, -1)
    ref = (att @ v).reshape(1, NQH, D)
    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    key_cache, value_cache = kv.transpose(1, 2).split(D, dim=-1)
    cu = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    fout = torch.empty_like(q)
    def fa():
        flash_attn_varlen_func(q=q, k=key_cache, v=value_cache, out=fout, cu_seqlens_q=cu, max_seqlen_q=1,
                               seqused_k=sl, max_seqlen_k=L, softmax_scale=scale, causal=True, block_table=bt)
    fa()
    ferr = ((fout.float() - ref).abs().max() / ref.abs().max()).item()
    tf = gtime(lambda: [fa() for _ in range(12)]) / 12 * 1e3
    to = gtime(lambda: [ad.decode(q, kv, bt, sl, out, scale, KVH, 32) for _ in range(12)]) / 12 * 1e3
    print(f"L={L:6d} o4 rel_err={err:.2e} fa rel_err={ferr:.2e}  fa {tf:8.2f} us  o4 {to:8.2f} us", flush=True)
