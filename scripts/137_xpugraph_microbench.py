#!/usr/bin/env python3
# 137_xpugraph_microbench.py -- FRONTIER de-risk: does torch.xpu.XPUGraph (PyTorch 2.12+xpu, SYCL-Graph over
# Level-Zero command lists) actually collapse the launch overhead in a DECODE-LIKE (M=1, launch-bound) workload
# on B70 -- AND is replay STABLE, or does it DEGRADE (the "torch-xpu graph dead-end" the journal recorded for the
# old sglang cuda_graph path)? This is the single-stream ceiling-breaker IF it works + is stable. Microbench FIRST
# (cheap, no model) before any sglang integration.
#   ZE_AFFINITY_MASK=1 python 137_xpugraph_microbench.py
import os, time, torch

dev = "xpu"
torch.manual_seed(0)
H = 5120            # qwen3_5 hidden
NLAYERS = 64        # ~model depth
GEMV_PER = 6        # qkv/o/gate_up/down-ish per layer -> ~384 M=1 GEMVs + elementwise = launch-bound proxy
ITERS_EAGER = 300
REPLAYS = 4000      # long replay run to expose any L0/NEO command-stream accumulation (degradation)

print(f"torch {torch.__version__} | xpu avail {torch.xpu.is_available()} | dev {torch.xpu.get_device_name(0) if torch.xpu.is_available() else '?'}")
assert torch.xpu.is_available(), "no XPU"

# static weights (decode: weights fixed) and static activation buffers (graph capture needs fixed addresses)
Ws = [torch.randn(H, H, device=dev, dtype=torch.bfloat16) * (H ** -0.5) for _ in range(GEMV_PER)]
static_h = torch.randn(1, H, device=dev, dtype=torch.bfloat16)

def decode_step(h):
    # mimic a launch-bound decode: per layer = a few M=1 GEMVs + rmsnorm-ish elementwise + residual
    for _ in range(NLAYERS):
        r = h
        h = h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(h.dtype)  # rmsnorm-ish
        for W in Ws:
            h = torch.nn.functional.silu(h @ W)
        h = h + r
    return h

# ---- EAGER baseline ----
for _ in range(20):  # warm
    _ = decode_step(static_h)
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(ITERS_EAGER):
    out = decode_step(static_h)
torch.xpu.synchronize()
eager_per = (time.perf_counter() - t0) / ITERS_EAGER * 1000
print(f"[eager]  {eager_per:.3f} ms/step  ({NLAYERS*GEMV_PER} GEMVs + elementwise/step)")

# ---- XPUGraph capture/replay ----
try:
    # capture into a static output buffer
    static_out = torch.empty_like(static_h)
    g = torch.xpu.XPUGraph()
    # warmup the exact ops on a side stream (mirrors torch.cuda capture protocol)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(5):
            tmp = decode_step(static_h)
    torch.xpu.current_stream().wait_stream(s)

    with torch.xpu.graph(g):
        static_out.copy_(decode_step(static_h))

    # replay correctness vs eager
    static_h.copy_(torch.randn(1, H, device=dev, dtype=torch.bfloat16))
    g.replay(); torch.xpu.synchronize()
    ref = decode_step(static_h)
    err = (static_out.float() - ref.float()).abs().max().item()
    print(f"[graph]  capture OK | replay-vs-eager maxabs err {err:.4f}")

    # warm replay
    for _ in range(20):
        g.replay()
    torch.xpu.synchronize()

    # timed replay + DEGRADATION test (windowed)
    win = REPLAYS // 8
    wtimes = []
    t0 = time.perf_counter()
    for i in range(REPLAYS):
        g.replay()
        if (i + 1) % win == 0:
            torch.xpu.synchronize()
            now = time.perf_counter()
            wtimes.append((now - t0) / win * 1000)
            t0 = now
    graph_per = sum(wtimes) / len(wtimes)
    print(f"[graph]  {graph_per:.3f} ms/step avg over {REPLAYS} replays  -> SPEEDUP {eager_per/graph_per:.2f}x vs eager")
    print(f"[graph]  per-window ms (degradation test, {win}/window): " + " ".join(f"{w:.3f}" for w in wtimes))
    ratio = wtimes[-1] / wtimes[0] if wtimes[0] > 0 else float('nan')
    verdict = "STABLE" if ratio < 1.15 else f"DEGRADES ({ratio:.2f}x last/first)"
    print(f"[graph]  degradation: first {wtimes[0]:.3f} -> last {wtimes[-1]:.3f} ms = {verdict}")
    print(f"=== VERDICT: XPUGraph {'WORKS + ' + verdict if err < 0.5 else 'NUMERICALLY WRONG (err %.3f)'%err} | speedup {eager_per/graph_per:.2f}x ===")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"=== XPUGraph FAILED: {type(e).__name__}: {e} ===")
