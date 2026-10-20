"""One-card numeric diagnosis: BF16 projection shapes and GDN verification."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vllm_xpu_kernels._xpu_C  # noqa: F401


def delta(a, b):
    x, y = a.float(), b.float()
    return dict(unequal=int((a != b).sum()), max_abs=float((x - y).abs().max()))


torch.manual_seed(19)
device = "xpu:0"
rt = Path("/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt")
for line in (rt / "dense.txt").read_text().splitlines():
    name, kind, n, k, offset, size = line.split()
    if kind != "bf16" or not (
        name.startswith("hyper_connection_mixer.input_mix") or name == "mlp.gate.weight"
    ):
        continue
    n, k, offset, size = map(int, (n, k, offset, size))
    raw = np.fromfile(rt / "dense.bin", dtype=np.uint8, count=size, offset=offset)
    w = torch.from_numpy(raw).view(torch.bfloat16).reshape(n, k).to(device)
    x = torch.randn(6, k, dtype=torch.bfloat16, device=device)
    rowwise = torch.cat([F.linear(row[None], w) for row in x])
    for m in (2, 3, 6):
        actual = F.linear(x[:m], w)
        print(
            json.dumps(dict(op=name, rows=m, **delta(actual, rowwise[:m]))), flush=True
        )

# Native full Flash-Next GDN shape. Compare one three-row speculative
# verification with the same three inputs run as ordinary single decodes.
nk, nv, dk, dv, width, t = 16, 48, 128, 128, 4, 3
qkv = 2 * nk * dk + nv * dv
qkvz = qkv + nv * dv
proj = torch.randn(t, qkvz, device=device, dtype=torch.bfloat16)
ba = torch.randn(t, 2 * nv, device=device, dtype=torch.bfloat16)
conv = torch.randn(1, width - 1 + t - 1, qkv, device=device, dtype=torch.bfloat16)
ssm = torch.randn(t, nv, dv, dk, device=device)
conv_weights = torch.randn(qkv, width, device=device, dtype=torch.bfloat16)
alog = torch.randn(nv, device=device)
dt = torch.randn(nv, device=device, dtype=torch.bfloat16)
conv_seq = conv[:, : width - 1].clone()
ssm_seq = ssm[:1].clone()
out_spec = torch.zeros(t, nv, dv, device=device, dtype=torch.bfloat16)
z_spec = torch.zeros_like(out_spec)


def i32(values):
    return torch.tensor(values, device=device, dtype=torch.int32)


def run(out, z, p, b, conv_state, ssm_state, spec):
    count = p.shape[0]
    torch.ops._xpu_C.gdn_attention(
        out,
        z,
        p,
        b,
        nk,
        nv,
        dk,
        dv,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=None,
        activation="silu",
        A_log=alog,
        dt_bias=dt,
        num_prefills=0,
        num_decodes=0 if spec else 1,
        num_spec_decodes=1 if spec else 0,
        has_initial_state=None
        if spec
        else torch.ones(1, device=device, dtype=torch.bool),
        non_spec_query_start_loc=None if spec else i32([0, 1]),
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=None if spec else i32([0]),
        spec_query_start_loc=i32([0, count]) if spec else None,
        spec_token_indx=i32(list(range(count))) if spec else None,
        spec_state_indices_tensor=i32([[0, 1, 2]]) if spec else None,
        num_accepted_tokens=i32([1]) if spec else None,
        num_actual_tokens=count,
        tp_size=1,
        reorder_input=True,
    )


run(out_spec, z_spec, proj, ba, conv, ssm, True)
out_seq, z_seq = torch.empty_like(out_spec), torch.empty_like(z_spec)
for j in range(t):
    run(
        out_seq[j : j + 1],
        z_seq[j : j + 1],
        proj[j : j + 1],
        ba[j : j + 1],
        conv_seq,
        ssm_seq,
        False,
    )
print(
    json.dumps(dict(op="gdn_spec_vs_sequential", **delta(out_spec, out_seq))),
    flush=True,
)
print(json.dumps(dict(op="gdn_z", **delta(z_spec, z_seq))), flush=True)
print(json.dumps(dict(op="gdn_final_ssm", **delta(ssm[-1], ssm_seq[0]))), flush=True)

torch.ops.load_library(
    "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/"
    "b70_gguf_xpu/b70_gguf_xpu/_C.o3.so"
)
raw = np.fromfile(rt / "dense.bin", dtype=np.uint8, count=6963200)
packed = torch.from_numpy(raw).to(device)
act = torch.randn(6, 2560, device=device, dtype=torch.bfloat16)
one = torch.cat(
    [torch.ops.b70_gguf.gemv(row[None], packed, 8, 2560, 2560) for row in act]
)
for m in (2, 3, 6):
    result = torch.ops.b70_gguf.gemv(act[:m], packed, 8, 2560, 2560)
    print(
        json.dumps(dict(op="packed_q8_gemv", rows=m, **delta(result, one[:m]))),
        flush=True,
    )

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

k = torch.randn(16, 128, 2, 256, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)
q = torch.randn(3, 24, 256, device=device, dtype=torch.bfloat16)
table = i32([list(range(16))])
for past in (16, 64, 128, 512, 1536):
    options = dict(k=k, v=v, max_seqlen_k=2048, block_table=table, causal=True)
    spec = flash_attn_varlen_func(
        q=q,
        max_seqlen_q=3,
        cu_seqlens_q=i32([0, 3]),
        seqused_k=i32([past + 3]),
        **options,
    )
    seq = torch.cat(
        [
            flash_attn_varlen_func(
                q=q[j : j + 1],
                max_seqlen_q=1,
                cu_seqlens_q=i32([0, 1]),
                seqused_k=i32([past + j + 1]),
                **options,
            )
            for j in range(3)
        ]
    )
    print(
        json.dumps(dict(op="fa_spec_vs_sequential", past=past, **delta(spec, seq))),
        flush=True,
    )
torch.xpu.synchronize()
print("X1_SHAPE_PROBE_DONE", flush=True)
