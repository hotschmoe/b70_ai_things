"""Grouped INT8 MoE GEMM test on the B70 -- reuses our EXISTING dense oneDNN int8 op
(torch.ops._xpu_C.int8_gemm_w8a8) as a per-expert grouped GEMM (loop over experts).

Goal: prove the int8-XMX expert compute works + correct, and measure:
  (a) kernel correctness vs an exact int8->float reference,
  (b) end-to-end quant fidelity vs the original bf16 (cosine sim),
  (c) int8 grouped-GEMM speedup vs bf16, AND the per-expert launch overhead,
      in both DECODE (1 token, top-8 active) and PREFILL (256 tokens) regimes.
This is the foundation for an XPUExpertsInt8 (native int8 MoE) -- if the loop is fast
enough captured, we may not even need a fused grouped C++ kernel.

Shapes mirror Qwen3.6-35B-A3B: hidden K=2048, expert intermediate N=512, E=256, top_k=8.
"""
import os, time, sys
import torch

print("=== ENV ===")
print("torch", torch.__version__, "| xpu avail", torch.xpu.is_available(), "| count", torch.xpu.device_count())
DEV = "xpu:0"
torch.manual_seed(0)

HAS_OP = hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "int8_gemm_w8a8")
try:
    import vllm._xpu_ops  # noqa: triggers _xpu_C load
    HAS_OP = hasattr(torch.ops._xpu_C, "int8_gemm_w8a8")
except Exception as e:
    print("(vllm._xpu_ops import:", e, ")")
print("int8_gemm_w8a8 present:", HAS_OP,
      "| dynamic_per_token_int8_quant present:",
      hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "dynamic_per_token_int8_quant"))
if not HAS_OP:
    print("FATAL: int8_gemm_w8a8 op not in this image. Use vllm-xpu-env:int8g."); sys.exit(1)

E, K, N, TOPK = 256, 2048, 512, 8

def quant_per_out_channel(W):  # W [N,K] bf16 -> (W_q int8 [N,K], scale [N,1] f32)
    s = (W.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    Wq = (W / s).round().clamp(-127, 127).to(torch.int8)
    return Wq, s.to(torch.float32)

def pt_quant(x):  # per-token sym int8; use our op if present else manual
    if hasattr(torch.ops._xpu_C, "dynamic_per_token_int8_quant"):
        xq, xs, _ = torch.ops._xpu_C.dynamic_per_token_int8_quant(x, True, 8)
        return xq, xs
    s = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-8)
    return (x / s).round().clamp(-127, 127).to(torch.int8), s.to(x.dtype)

def expert_gemm_int8(x_e, Wq_e, ws_e):  # x_e [m,K] bf16 -> [m,N] bf16 via our op
    xq, xs = pt_quant(x_e)
    w_q = Wq_e.t().contiguous()           # [K,N] int8 (op wants transposed weight)
    w_s = ws_e.reshape(1, N).contiguous() # [1,N] per-channel
    return torch.ops._xpu_C.int8_gemm_w8a8(xq, xs, None, w_q, w_s, None, None, torch.bfloat16)

print(f"\n=== build {E} expert weights [N={N},K={K}] (int8 per-channel) ===")
W_true = (torch.randn(E, N, K, device=DEV, dtype=torch.bfloat16) * 0.02)
Wq = torch.empty(E, N, K, device=DEV, dtype=torch.int8)
Ws = torch.empty(E, N, 1, device=DEV, dtype=torch.float32)
for e in range(E):
    q, s = quant_per_out_channel(W_true[e])
    Wq[e], Ws[e] = q, s
print("expert weights int8:", tuple(Wq.shape), "scales:", tuple(Ws.shape),
      "| int8 VRAM ~", round(Wq.numel()/1e6, 1), "MB")

def run_regime(T, label, iters=30):
    print(f"\n=== {label}: T={T} tokens, top_k={TOPK} ===")
    x = torch.randn(T, K, device=DEV, dtype=torch.bfloat16) * 0.1
    # random top-k routing
    gate = torch.randn(T, E, device=DEV)
    topk_w, topk_idx = gate.softmax(-1).topk(TOPK, dim=-1)  # [T,TOPK]
    # flatten token-expert pairs and group by expert
    flat_tok = torch.arange(T, device=DEV).repeat_interleave(TOPK)
    flat_exp = topk_idx.reshape(-1)
    flat_w = topk_w.reshape(-1)
    order = flat_exp.argsort()
    flat_tok, flat_exp, flat_w = flat_tok[order], flat_exp[order], flat_w[order]
    active = torch.unique(flat_exp)
    print(f"  active experts: {active.numel()}/{E}  (pairs={flat_exp.numel()})")

    def grouped_int8():
        out = torch.zeros(T, N, device=DEV, dtype=torch.bfloat16)
        for e in active.tolist():
            sel = (flat_exp == e)
            toks = flat_tok[sel]
            y = expert_gemm_int8(x[toks], Wq[e], Ws[e])
            out.index_add_(0, toks, (y * flat_w[sel].unsqueeze(1).to(torch.bfloat16)))
        return out

    def grouped_bf16():
        out = torch.zeros(T, N, device=DEV, dtype=torch.bfloat16)
        for e in active.tolist():
            sel = (flat_exp == e)
            toks = flat_tok[sel]
            y = x[toks] @ W_true[e].t()
            out.index_add_(0, toks, (y * flat_w[sel].unsqueeze(1).to(torch.bfloat16)))
        return out

    # correctness: kernel (int8 op) vs exact int8->float ref, on one expert
    e0 = int(active[0].item())
    sel0 = (flat_exp == e0); toks0 = flat_tok[sel0]
    xe = x[toks0]
    xq, xs = pt_quant(xe)
    y_op = expert_gemm_int8(xe, Wq[e0], Ws[e0]).float()
    y_ref_int = (xs.float() * (xq.float() @ Wq[e0].float().t())) * Ws[e0].reshape(1, N).float()
    rel = ((y_op - y_ref_int).abs() / (y_ref_int.abs() + 1e-3)).mean().item()
    # end-to-end quant fidelity vs original bf16
    y_true = (xe.float() @ W_true[e0].float().t())
    cos = torch.nn.functional.cosine_similarity(y_op.flatten(), y_true.flatten(), dim=0).item()
    print(f"  kernel correctness (op vs exact int8 ref): mean_rel_err={rel:.2e}")
    print(f"  end-to-end quant fidelity (op vs bf16): cosine={cos:.5f}")

    # timing
    for fn, nm in [(grouped_bf16, "bf16"), (grouped_int8, "int8")]:
        for _ in range(5): fn()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.xpu.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        print(f"  {nm} grouped GEMM: {ms:.3f} ms/iter")
    return

run_regime(1, "DECODE", iters=50)
run_regime(256, "PREFILL", iters=20)

def raw_throughput():
    # Single COMPUTE-BOUND GEMM (no loop, no routing) -> isolates the int8 XMX advantage
    # from per-expert launch overhead. If int8 beats bf16 HERE, the fused grouped kernel is worth building.
    print("\n=== RAW GEMM throughput (compute-bound, single op, no loop) ===")
    for M in (2048, 8192):
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
        W = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02
        Wq, ws = quant_per_out_channel(W)
        w_q = Wq.t().contiguous(); w_s = ws.reshape(1, N).contiguous()
        flops = 2.0 * M * K * N
        def f_bf16(): return x @ W.t()
        def f_int8():
            xq, xs = pt_quant(x)
            return torch.ops._xpu_C.int8_gemm_w8a8(xq, xs, None, w_q, w_s, None, None, torch.bfloat16)
        res = {}
        for fn, nm in [(f_bf16, "bf16"), (f_int8, "int8")]:
            for _ in range(5): fn()
            torch.xpu.synchronize(); t0 = time.perf_counter()
            for _ in range(50): fn()
            torch.xpu.synchronize(); ms = (time.perf_counter() - t0) / 50 * 1e3
            res[nm] = ms
            print(f"  M={M:5d} K={K} N={N}  {nm}: {ms:.3f} ms  ({flops/ms/1e9:.1f} GFLOP/s)")
        print(f"  M={M:5d} int8 speedup vs bf16: {res['bf16']/res['int8']:.2f}x")

raw_throughput()

def gemm_only_compute_bound():
    # int8 GEMM-ONLY (activation PRE-quantized once, as a fused MoE pipeline would deliver it) vs bf16,
    # at genuinely compute-bound shapes (big N too). Decides whether the int8 GEMM itself wins on XMX.
    print("\n=== int8 GEMM-ONLY (pre-quantized, quant op excluded) vs bf16 -- compute-bound shapes ===")
    for (M, Kk, Nn) in [(4096, 2048, 512), (4096, 4096, 4096), (8192, 4096, 11008)]:
        x = torch.randn(M, Kk, device=DEV, dtype=torch.bfloat16) * 0.1
        W = torch.randn(Nn, Kk, device=DEV, dtype=torch.bfloat16) * 0.02
        Wq, ws = quant_per_out_channel(W); w_q = Wq.t().contiguous(); w_s = ws.reshape(1, Nn).contiguous()
        xq, xs = pt_quant(x)  # pre-quantized ONCE (fused into permute in a real MoE pipeline)
        flops = 2.0 * M * Kk * Nn
        def f_bf16(): return x @ W.t()
        def f_int8(): return torch.ops._xpu_C.int8_gemm_w8a8(xq, xs, None, w_q, w_s, None, None, torch.bfloat16)
        res = {}
        for fn, nm in [(f_bf16, "bf16"), (f_int8, "int8")]:
            for _ in range(5): fn()
            torch.xpu.synchronize(); t0 = time.perf_counter()
            for _ in range(50): fn()
            torch.xpu.synchronize(); ms = (time.perf_counter() - t0) / 50 * 1e3
            res[nm] = ms
            print(f"  M={M:5d} K={Kk} N={Nn:5d}  {nm}: {ms:.3f} ms  ({flops/ms/1e9:.0f} GFLOP/s)")
        print(f"  -> int8 GEMM-only speedup vs bf16: {res['bf16']/res['int8']:.2f}x")

gemm_only_compute_bound()
print("\n=== DONE ===")
