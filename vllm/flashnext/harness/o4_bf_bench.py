#!/usr/bin/env python3
"""O4: b70_o4.bf_gemv (bf16 weights, split-K) vs oneDNN on HC shapes."""
import sys
import torch
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_o4_xpu")
import b70_o4_xpu as o4
o4.load()
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness")
from o4_w8_bench_util import gtime  # noqa: E402

L = 96
for name, N, K, kss in (("down", 324, 10240, (4, 8, 16)), ("up", 10240, 320, (1, 2)), ("router", 512, 2560, (2, 4, 8))):
    Ws = [torch.randn(N, K, device="xpu", dtype=torch.bfloat16) * 0.02 for _ in range(L)]
    for M in (1, 2, 4):
        x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, Ws[0])
        for ks in kss:
            y = torch.ops.b70_o4.bf_gemv(x, Ws[0], ks)
            d = (y.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
            def run():
                for w in Ws:
                    torch.ops.b70_o4.bf_gemv(x, w, ks)
            t = gtime(run) / L * 1e3
            print(f"{name} M={M} ks={ks} {t:.2f} us {N*K*2/(t*1e-6)/1e9:.0f} GB/s ulp_max={int(d.max())} frac_diff={(d>0).float().mean().item():.4f}", flush=True)
        def runb():
            for w in Ws:
                torch.nn.functional.linear(x, w)
        print(f"{name} M={M} onednn {gtime(runb)/L*1e3:.2f} us", flush=True)
    del Ws
    torch.xpu.empty_cache()
