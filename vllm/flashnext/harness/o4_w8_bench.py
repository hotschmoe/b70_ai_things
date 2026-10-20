#!/usr/bin/env python3
"""O4: b70_o4.w8_gemv correctness and graph timing on the HC shapes."""
import sys
import torch
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_o4_xpu")
import b70_o4_xpu as o4
o4.load()


def gtime(fn, n=20):
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    torch.xpu.synchronize()
    s = torch.xpu.Event(enable_timing=True); e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        g.replay()
    e.record(); e.synchronize()
    return s.elapsed_time(e) / n


L = 96
for name, N, K in (("down", 324, 10240), ("up", 10240, 320)):
    Ws = [torch.randn(N, K, device="xpu", dtype=torch.bfloat16) * 0.02 for _ in range(L)]
    Q = [o4.quantize_w8(w) for w in Ws]
    for M in (1, 2, 4):
        x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
        q, s = Q[0]
        deq = (q.float().reshape(N, K // 32, 32) * s.float()[..., None]).reshape(N, K)
        ref = (x.float() @ deq.t())
        bf = torch.nn.functional.linear(x, Ws[0]).float()
        for ks in ((4, 8, 16) if name == "down" else (0, -2, -8)):
            y = o4.w8_gemv(x, q, s, ks).float()
            err = ((y - ref).abs().max() / ref.abs().max()).item()
            qerr = ((ref - bf).abs().max() / bf.abs().max()).item()
            def run():
                for qq, ss in Q:
                    o4.w8_gemv(x, qq, ss, ks)
            t = gtime(run) / L * 1e3
            byt = N * K * (1 + 2 / 32)
            print(f"{name} M={M} ks={ks} {t:.2f} us {byt/(t*1e-6)/1e9:.0f} GB/s kernel_rel={err:.2e} q8_vs_bf16_rel={qerr:.2e}", flush=True)
        def runb():
            for w in Ws:
                torch.nn.functional.linear(x, w)
        print(f"{name} M={M} onednn bf16 {gtime(runb)/L*1e3:.2f} us", flush=True)
    del Ws, Q
    torch.xpu.empty_cache()
