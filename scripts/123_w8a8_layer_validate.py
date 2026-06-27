#!/usr/bin/env python3
# 123_w8a8_layer_validate.py -- disambiguate the W8A8 "!!!!" garbage: is it the SHIM math or a BAD CKPT?
# For several real layers (incl. fused qkv/gate_up): (1) dequant the int8 ckpt weight and compare to the
# source bf16 weight (checkpoint quant quality + scale alignment); (2) run the shim's EXACT
# quant->torch._int_mm->dequant on XPU and compare to a bf16 matmul reference (shim correctness).
import torch, json
from safetensors import safe_open

W8  = "/models/Qwen3.6-27B-W8A8-sqgptq"
SRC = "/models/Qwen_Qwen3.6-27B"
LAYERS = [
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.0.mlp.gate_up_proj",   # fused (2 logical widths)
    "model.language_model.layers.0.self_attn.qkv_proj",  # fused (q,k,v)
    "model.language_model.layers.0.self_attn.o_proj",
]
src_idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]

# header of the W8A8 single file: which keys exist
with safe_open(f"{W8}/model.safetensors", framework="pt", device="cpu") as f8:
    w8_keys = set(f8.keys())

dev = "xpu"
for L in LAYERS:
    wk, sk = f"{L}.weight", f"{L}.weight_scale"
    if wk not in w8_keys:
        # try unfused names (some ckpts store q/k/v or gate/up separately)
        print(f"\n[{L}] NOT in W8A8 ckpt as fused; keys like it:",
              [k for k in w8_keys if L.rsplit('.',1)[0] in k and 'proj' in k][:6])
        continue
    with safe_open(f"{W8}/model.safetensors", framework="pt", device="cpu") as f8:
        qw = f8.get_tensor(wk)            # [N,K] int8
        ws = f8.get_tensor(sk)           # [N,1] f32 (per-channel)
    # source bf16 reference (handle fused: source may store q/k/v separately -> skip ref if absent)
    if wk in src_idx:
        with safe_open(f"{SRC}/{src_idx[wk]}", framework="pt", device="cpu") as fr:
            wref = fr.get_tensor(wk).float()      # [N,K]
        wdq = qw.float() * ws.float()             # [N,K] dequant
        ckpt_rel = ((wdq - wref).norm() / wref.norm()).item()
    else:
        wref, ckpt_rel = None, None

    # --- shim path on XPU (mirror _orig_pw .t() + our .contiguous() + apply) ---
    M, K, N = 4, qw.shape[1], qw.shape[0]
    x = torch.randn(M, K, dtype=torch.bfloat16)
    wt  = qw.t().contiguous().to(dev)             # [K,N] int8 (== _orig_pw transpose, made contiguous)
    wsr = ws.reshape(1, -1).float().to(dev)       # [1,N]
    xd  = x.to(dev)
    amax = xd.abs().amax(-1, keepdim=True).clamp(min=1e-5)
    xs = amax / 127.0
    xq = torch.round(xd / xs).clamp(-127, 127).to(torch.int8)
    acc = torch._int_mm(xq, wt)                   # [M,N] int32
    out = (acc.float() * xs.float() * wsr).cpu()  # [M,N]
    # reference: full-precision int8-dequant-weight matmul (isolates GEMM+dequant from quant error)
    wdq_dev = (qw.t().float() * ws.reshape(1, -1).float())  # [K,N] dequant weight, fp
    out_dqref = (x.float() @ wdq_dev)
    shim_rel = ((out - out_dqref).norm() / out_dqref.norm()).item()
    bf16_rel = (((out) - (x.float() @ wref.t())).norm() / (x.float() @ wref.t()).norm()).item() if wref is not None else None

    print(f"\n[{L}]  shape int8 qw={tuple(qw.shape)} scale={tuple(ws.shape)}")
    print(f"   CKPT dequant rel-err vs source bf16 : {ckpt_rel}")
    print(f"   SHIM out rel-err vs int8-dequant ref: {shim_rel}   (isolates GEMM+act-quant)")
    print(f"   SHIM out rel-err vs source-bf16 ref : {bf16_rel}   (full path incl quant loss)")
