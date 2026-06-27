#!/usr/bin/env python3
# 125_w8a8_fused_validate.py -- the per-layer validate (123) only checked UNFUSED down_proj. sglang FUSES
# gate_proj+up_proj -> gate_up_proj and q+k+v -> qkv_proj, concatenating each sub-weight's per-channel int8
# scale. If the shim's reshape(1,-1) of the FUSED scale misaligns with the transposed fused weight columns,
# every fused layer dequants with wrong scales -> garbage. This replicates sglang's fusion exactly and checks
# the shim output against a reference that computes each sub-projection separately (the ground truth).
import torch, json
from safetensors import safe_open

W8 = "/models/Qwen3.6-27B-W8A8-sqgptq"
dev = "xpu"

def load(name):  # full-attn layer 3 has q/k/v/o + gate/up/down all int8
    L = f"model.language_model.layers.3.{name}"
    with safe_open(f"{W8}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(f"{L}.weight"), f.get_tensor(f"{L}.weight_scale")  # [N,K] int8, [N,1] f32

def shim_apply(x, qw_NK, ws_N1):
    # mirror the shim: _orig_pw transposes weight [N,K]->[K,N]; weight_t=that.contiguous(); wscale=ws.reshape(1,-1)
    wt = qw_NK.t().contiguous().to(dev)        # [K,N]
    wsr = ws_N1.reshape(1, -1).float().to(dev) # [1,N]
    xd = x.to(dev)
    amax = xd.abs().amax(-1, keepdim=True).clamp(min=1e-5)
    xs = amax / 127.0
    xq = torch.round(xd / xs).clamp(-127, 127).to(torch.int8)
    return (torch._int_mm(xq, wt).float() * xs.float() * wsr).cpu()  # [M,N]

for fused_name, parts in [("gate_up", ["gate_proj", "up_proj"]),
                          ("qkv", ["q_proj", "k_proj", "v_proj"])]:
    ws_parts = [load(p) for p in parts]
    K = ws_parts[0][0].shape[1]
    M = 8
    x = torch.randn(M, K, dtype=torch.bfloat16)
    # REFERENCE: each sub-projection through the shim SEPARATELY, then concat along N (the correct result)
    ref = torch.cat([shim_apply(x, qw, ws) for qw, ws in ws_parts], dim=1)
    # FUSED (as sglang builds it): concat weights along N(=dim0 of [N,K]) and scales along N
    qw_f = torch.cat([qw for qw, _ in ws_parts], dim=0)     # [sumN, K]
    ws_f = torch.cat([ws for _, ws in ws_parts], dim=0)     # [sumN, 1]
    got = shim_apply(x, qw_f, ws_f)
    rel = ((got - ref).norm() / ref.norm()).item()
    Ns = [qw.shape[0] for qw, _ in ws_parts]
    print(f"[{fused_name}] parts N={Ns} fusedN={qw_f.shape[0]} K={K}  fused-vs-separate rel-err={rel:.3e}  "
          f"{'OK' if rel < 1e-5 else '>>> FUSED SCALE MISALIGNED <<<'}")
