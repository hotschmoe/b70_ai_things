#!/usr/bin/env python3
"""O4: fused Q8 HC kernels (w8_down_act, w8_up_collapse) vs the f14 path
(w8_gemv + Triton act / collapse). Bitwise check and graph timing."""
import os, sys, types
import torch
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_o4_xpu")
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness")
import b70_o4_xpu as o4
o4.load()
from o4_w8_bench_util import gtime
from vllm.models.qwen4_exp.xpu import hc_decode as hd
import triton

H, HC, R = 2560, 4, 320
bad = 0
L = 48
Wd = [(torch.randn(R + HC, HC * H, device="xpu") * 0.01).to(torch.bfloat16) for _ in range(L)]
Wu = [(torch.randn(HC * H, R, device="xpu") * 0.05).to(torch.bfloat16) for _ in range(L)]
Qd = [o4.quantize_w8(w) for w in Wd]
Qu = [o4.quantize_w8(w) for w in Wu]
for T in (1, 2, 3, 4, 8):
    flat = (torch.randn(T, HC * H, device="xpu") * 0.7).to(torch.bfloat16)
    for i in range(3):
        lo = torch.ops.b70_o4.w8_gemv(flat, Qd[i][0], Qd[i][1], 8)
        act = torch.empty(T, R, dtype=torch.bfloat16, device="xpu")
        hd._hc_act[(T,)](lo, act, lo.stride(0), R, HC, triton.next_power_of_2(R))
        inj = lo[:, R:]
        act2, inj2 = torch.ops.b70_o4.w8_down_act(flat, Qd[i][0], Qd[i][1], R, float(HC))
        gate = torch.ops.b70_o4.w8_gemv(act, Qu[i][0], Qu[i][1], -8)
        out = torch.empty(T, H, dtype=torch.bfloat16, device="xpu")
        hd._hc_collapse[(T, triton.cdiv(H, 512))](flat, gate, out, H, HC, 512)
        out2 = torch.ops.b70_o4.w8_up_collapse(act, Qu[i][0], Qu[i][1], flat, H)
        ok = torch.equal(act, act2) and torch.equal(inj.contiguous(), inj2) and torch.equal(out, out2)
        if not ok:
            bad += 1
            d1 = (act.view(torch.int16).int() - act2.view(torch.int16).int()).abs().max().item()
            d3 = (out.view(torch.int16).int() - out2.view(torch.int16).int()).abs().max().item()
            print(f"T={T} i={i} mismatch act_ulp={d1} inj_eq={torch.equal(inj.contiguous(), inj2)} out_ulp={d3} frac={(out != out2).float().mean().item():.4f}")
    def old():
        for i in range(L):
            lo = torch.ops.b70_o4.w8_gemv(flat, Qd[i][0], Qd[i][1], 8)
            a = torch.empty(T, R, dtype=torch.bfloat16, device="xpu")
            hd._hc_act[(T,)](lo, a, lo.stride(0), R, HC, triton.next_power_of_2(R))
            g = torch.ops.b70_o4.w8_gemv(a, Qu[i][0], Qu[i][1], -8)
            o = torch.empty(T, H, dtype=torch.bfloat16, device="xpu")
            hd._hc_collapse[(T, triton.cdiv(H, 512))](flat, g, o, H, HC, 512)
    def new():
        for i in range(L):
            a, j = torch.ops.b70_o4.w8_down_act(flat, Qd[i][0], Qd[i][1], R, float(HC))
            torch.ops.b70_o4.w8_up_collapse(a, Qu[i][0], Qu[i][1], flat, H)
    print(f"T={T} 48 HC mixes (no norm): old {gtime(old):.3f} ms new {gtime(new):.3f} ms", flush=True)
print("O4_HCQ8_FUSE", "OK" if bad == 0 else f"MISMATCH {bad}")
