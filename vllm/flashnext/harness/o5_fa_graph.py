#!/usr/bin/env python3
"""O5: FA decode vs MTP-verify rows under XPUGraph capture/replay.

Mirrors the FULL_DECODE_ONLY graphs: the q_len=1 decode graph and the
q_len=k+1 verify graph are captured once (max_seqlen_k fixed at capture, as
vLLM's captured metadata does) and replayed with live seqused_k values.
Compares verify row j against the decode graph replayed at the same
position, bitwise, for several past lengths. Run with/without
B70_FA_KV_SPLITS.
"""

import json
import os

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

dev = "xpu:0"
torch.manual_seed(11)
HQ, HKV, D, BS = 24, 2, 256, 1600
NB = int(os.environ.get("O5_NB", "2"))
CAP_K = int(os.environ.get("O5_CAP_MAXK", str(NB * BS)))
QL = int(os.environ.get("O5_QL", "4"))
k = torch.randn(NB, BS, HKV, D, device=dev, dtype=torch.bfloat16)
v = torch.randn_like(k)
table = torch.arange(NB, device=dev, dtype=torch.int32).view(1, NB)

q1 = torch.zeros(1, HQ, D, device=dev, dtype=torch.bfloat16)
qv = torch.zeros(QL, HQ, D, device=dev, dtype=torch.bfloat16)
cu1 = torch.tensor([0, 1], device=dev, dtype=torch.int32)
cuv = torch.tensor([0, QL], device=dev, dtype=torch.int32)
sk1 = torch.ones(1, device=dev, dtype=torch.int32)
skv = torch.ones(1, device=dev, dtype=torch.int32)


def dec():
    return flash_attn_varlen_func(q=q1, k=k, v=v, max_seqlen_q=1, cu_seqlens_q=cu1, seqused_k=sk1,
                                  max_seqlen_k=CAP_K, block_table=table, causal=True)


def ver():
    return flash_attn_varlen_func(q=qv, k=k, v=v, max_seqlen_q=QL, cu_seqlens_q=cuv, seqused_k=skv,
                                  max_seqlen_k=CAP_K, block_table=table, causal=True)


sk1.fill_(100)
skv.fill_(100)
dec(); ver()
torch.xpu.synchronize()
g1, gv = torch.xpu.XPUGraph(), torch.xpu.XPUGraph()
s = torch.xpu.Stream()
with torch.xpu.stream(s):
    with torch.xpu.graph(g1, stream=s):
        o1 = dec()
    with torch.xpu.graph(gv, stream=s):
        ov = ver()
torch.xpu.synchronize()
qsrc = torch.randn(QL, HQ, D, device=dev, dtype=torch.bfloat16)
bad = 0
for past in (20, 80, 150, 400, 900, 1500, 2500, NB * BS - QL - 1):
    if past + QL > NB * BS:
        continue
    qv.copy_(qsrc)
    skv.fill_(past + QL)
    gv.replay()
    torch.xpu.synchronize()
    vrow = ov.clone()
    eq = []
    eager_eq = []
    for j in range(QL):
        q1.copy_(qsrc[j:j + 1])
        sk1.fill_(past + j + 1)
        g1.replay()
        torch.xpu.synchronize()
        eq.append(bool(torch.equal(vrow[j], o1[0])))
        e = flash_attn_varlen_func(q=qsrc[j:j + 1], k=k, v=v, max_seqlen_q=1, cu_seqlens_q=cu1,
                                   seqused_k=torch.tensor([past + j + 1], device=dev, dtype=torch.int32),
                                   max_seqlen_k=past + j + 1, block_table=table, causal=True)
        eager_eq.append(bool(torch.equal(e[0], o1[0])))
    bad += sum(not x for x in eq)
    print(json.dumps(dict(past=past, splits=os.environ.get("B70_FA_KV_SPLITS", "auto"), cap_maxk=CAP_K, ql=QL,
                          graph_verify_vs_graph_decode=eq, graph_decode_vs_eager_decode=eager_eq)), flush=True)
print("DONE bad", bad, flush=True)
