#!/usr/bin/env python3
# lmhead_int4_probe.py -- microbench gate for quantizing the BF16 lm_head to int4 g128 sym
# and routing it through torch.ops._xpu_C.int4_gemm_w4a16 (the decode GEMV).
#
# The Lorbus int4 ckpt keeps lm_head.weight BF16 [vocab=248320, hidden=5120] = 2.54 GB,
# read in FULL every decode step (~14% of per-token decode weight bandwidth). Quantizing it
# to int4 should cut that ~4x. This probe validates the PER-OP win + finiteness + quant relerr
# on the REAL lm_head weight BEFORE any serve. Mirrors sglang/w4a8_builtso_test.py loading.
#
# Run (card 0): docker run --rm --device /dev/dri ... sglang-xpu:mtp bash -c \
#   "source /opt/intel/oneapi/setvars.sh --force; \
#    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
#    python /work/lmhead_int4_probe.py"
import os, sys, time, ctypes, torch

DEV = "xpu"
SO = os.environ.get("B70_XPU_C_SO", "/work/w4a8_kernel/_xpu_C.abi3.so")
CKPT = os.environ.get("CKPT", "/models/Lorbus_Qwen3.6-27B-int4-AutoRound")
G = 128

print("torch", torch.__version__, "xpu_avail", torch.xpu.is_available(), flush=True)
try:
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
    print("CDLL RTLD_GLOBAL OK:", SO, flush=True)
except OSError as e:
    print("CDLL FAILED:", str(e)[:400]); sys.exit(1)
has16 = hasattr(torch.ops._xpu_C, "int4_gemm_w4a16")
print("int4_gemm_w4a16 registered:", has16, flush=True)
assert has16


def quant_int4_g128_sym(W, g=G, lo=-7, div=7.0):
    # W: [N, K] float -> qweight [N, K/8] int32 (compressed-tensors/auto_gptq nibble=val+8),
    #                    scales  [N, K/g] (per out-channel, per K-group).  sym, zp=8.
    # lo/div select the quant range: (lo=-7,div=7) restricted-sym; (lo=-8,div=7) full-range.
    N, K = W.shape
    assert K % g == 0
    Wg = W.reshape(N, K // g, g).to(torch.float32)
    amax = Wg.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)     # [N,K/g,1]
    scale = amax / div
    q = torch.round(Wg / scale).clamp_(lo, 7).to(torch.int32).reshape(N, K)
    scales = scale.squeeze(-1).reshape(N, K // g)                   # [N,K/g]
    nib = (q + 8).to(torch.int32).reshape(N, K // 8, 8)             # nibble in [0,15]
    qw = torch.zeros(N, K // 8, dtype=torch.int32, device=W.device)
    for i in range(8):
        qw |= (nib[:, :, i] & 0xF) << (4 * i)
    return qw, scales, q.reshape(N, K)


def sync():
    torch.xpu.synchronize()


def bench(fn, warm=25, iters=60):
    for _ in range(warm):
        fn()
    sync(); s = time.time()
    for _ in range(iters):
        fn()
    sync(); return (time.time() - s) / iters * 1000.0


def main():
    import safetensors.torch as st
    import json
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))
    shard = idx["weight_map"]["lm_head.weight"]
    print("loading lm_head.weight from", shard, flush=True)
    t = st.load_file(f"{CKPT}/{shard}")
    W = t["lm_head.weight"]                    # [N, K] bf16
    N, K = W.shape
    print(f"lm_head.weight {tuple(W.shape)} {W.dtype}  ({W.numel()*2/1e9:.2f} GB bf16)", flush=True)

    Wd = W.to(DEV)
    Wf = Wd.to(torch.float32)
    w_fp16 = Wd.to(torch.float16)                   # bf16 baseline -> fp16 to match op input dtype
    # cheap ranking proxy: random unit hidden vectors, compare top-1/top-5 argmax bf16 vs int4
    torch.manual_seed(0)
    Xr = torch.randn(64, K, device=DEV, dtype=torch.float32)
    Xr = Xr / Xr.norm(dim=-1, keepdim=True)
    ref_logits = (Xr @ Wf.t())                      # [64, N] bf16-weight logits
    ref_top1 = ref_logits.argmax(-1)
    ref_top5 = ref_logits.topk(5, dim=-1).indices

    print("\n==== quant-scheme sweep (relerr + ranking) ====", flush=True)
    best = None
    for (g, lo, div, tag) in [(128, -7, 7.0, "g128 sym[-7,7]"),
                              (128, -8, 7.0, "g128 sym[-8,7]"),
                              (64, -7, 7.0, "g64  sym[-7,7]"),
                              (32, -7, 7.0, "g32  sym[-7,7]")]:
        qw, scales, q = quant_int4_g128_sym(Wd, g, lo, div)
        Wdq = (q.to(torch.float32) * scales.to(torch.float32).repeat_interleave(g, dim=1))
        qre = (Wdq - Wf).norm() / Wf.norm()
        q_logits = Xr @ Wdq.t()
        t1 = (q_logits.argmax(-1) == ref_top1).float().mean().item()
        q5 = q_logits.topk(5, dim=-1).indices
        t5 = sum(len(set(q5[i].tolist()) & set(ref_top5[i].tolist())) for i in range(64)) / (64 * 5)
        sz = qw.numel() * 4 / 1e9 + scales.numel() * 2 / 1e9
        print(f"  {tag}: quant_relerr={qre.item():.4e}  top1_agree={t1:.3f}  top5_overlap={t5:.3f}  "
              f"size={sz:.3f}GB", flush=True)
        if best is None or qre.item() < best[0]:
            best = (qre.item(), g, lo, div, tag)

    # bench the BEST scheme at decode shapes
    _, g, lo, div, tag = best
    print(f"\n==== bench BEST = {tag} ====", flush=True)
    qw, scales, q = quant_int4_g128_sym(Wd, g, lo, div)
    qweight_t = qw.t()                              # [K/8, N] NT view (stride0==1)
    assert qweight_t.stride()[0] == 1, qweight_t.stride()
    wscale_t = scales.t().contiguous().to(torch.float16)
    wzp = torch.tensor([8], dtype=torch.int8, device=DEV)
    Wdq = (q.to(torch.float32) * scales.to(torch.float32).repeat_interleave(g, dim=1))
    print(f"qweight {tuple(qw.shape)} -> NT {tuple(qweight_t.shape)} stride{qweight_t.stride()}; "
          f"wscale_t {tuple(wscale_t.shape)}", flush=True)
    for M in (1, 8):
        x = torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.05
        y16 = torch.ops._xpu_C.int4_gemm_w4a16(x, qweight_t, None, wscale_t, wzp, g, None)
        fin = torch.isfinite(y16).all().item()
        ref_dq = x.to(torch.float32) @ Wdq.t()
        opre = (y16.to(torch.float32) - ref_dq).norm() / ref_dq.norm()
        tb = bench(lambda: x @ w_fp16.t())
        tw = bench(lambda: torch.ops._xpu_C.int4_gemm_w4a16(x, qweight_t, None, wscale_t, wzp, g, None))
        print(f"M={M:>2}  int4 w4a16={tw:.4f}ms  fp16mm={tb:.4f}ms  speedup={tb/tw:.2f}x  "
              f"finite={fin}  op_relerr={opre.item():.2e}", flush=True)

    print("\nGATE: M=1 int4 w4a16 FASTER than fp16 + finite -> per-op win; pick lowest-relerr scheme for serve.",
          flush=True)


if __name__ == "__main__":
    main()
