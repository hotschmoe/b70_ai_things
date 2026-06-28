#!/usr/bin/env python3
# W4A4 MXFP4 SPEED GATE microbench (card 0, synthetic, no serve).
#
# Question: does torch.ops._xpu_C.fp4_gemm (MXFP4 w4a4: mxfp4 weight x mxfp4 act,
# e8m0 block-32 scales) have a FAST compute path on B70, vs the shipped
# int4_gemm_w4a8 (int8 act) / int4_gemm_w4a16 (decode) / bf16 matmul?
# If it is not meaningfully faster than w4a8 at prefill (M=2048), W4A4 is not
# worth an MXFP4 requant on this box.
#
# Real layer shape: down_proj K=17408 (in), N=5120 (out). weight W=[N,K], act=[M,K].
# Warm (>=15 warmup, discard 1st; B70 idle-downclocks). M in {1 (decode), 2048 (prefill)}.
#
# Run inside image sglang-xpu:woq via:
#   ./bin/gpu-run --card 0 docker run --rm --device /dev/dri -e ZE_AFFINITY_MASK=0 \
#     -v /mnt/vm_8tb/b70/w4a8_kernel:/build/w4a8_kernel \
#     -v /mnt/vm_8tb/b70/vllm-xpu-kernels:/build/vllm-xpu-kernels \
#     -v /mnt/vm_8tb/github/b70_ai_things/sglang:/work \
#     -e LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH \
#     sglang-xpu:woq python3 /work/w4a4_probe.py
import os, sys, time, ctypes
import torch

DEV = "xpu"
SO = "/build/w4a8_kernel/_xpu_C.abi3.so"
G = 128  # int4 group size (for the w4a8/w4a16 baselines)
K = 17408  # down_proj in
N = 5120   # down_proj out
WARM = 20
ITERS = 50

print("torch", torch.__version__)
print("loading built so:", SO, "exists:", os.path.exists(SO))
try:
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
    print("CDLL RTLD_GLOBAL OK")
except OSError as e:
    print("CDLL FAILED:", str(e)[:400]); sys.exit(1)

# ---- (1) introspect the fp4 op ----
opname = None
for cand in ("fp4_gemm", "fp4_gemm_w4a4"):
    if hasattr(torch.ops._xpu_C, cand):
        opname = cand; break
print("fp4 op present:", opname)
if opname is None:
    print("ops:", [x for x in dir(torch.ops._xpu_C) if not x.startswith('_')])
    sys.exit(1)
FP4 = getattr(torch.ops._xpu_C, opname)
try:
    print("fp4 schema:", str(FP4.default._schema))
except Exception as e:
    print("fp4 schema read err:", repr(e)[:120])
print("int4_gemm_w4a8 present:", hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"))
print("int4_gemm_w4a16 present:", hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"))

# ---- mxfp4 quantizer (vetted, from the official onednn fp4 test helper) ----
sys.path.insert(0, "/build/vllm-xpu-kernels/tests/ops")
try:
    from mx_utils import to_mxfp  # pure torch; returns (e8m0 scale, fp4 packed)
    HAVE_MXUTILS = True
    print("mx_utils.to_mxfp imported")
except Exception as e:
    HAVE_MXUTILS = False
    print("mx_utils import FAILED:", repr(e)[:200]); sys.exit(1)

def mxfp4(t):
    # t: [.., k] bf16 (contiguous). returns (lp fp4 [..,k/2], scale e8m0 [..,k/32])
    s, q = to_mxfp(t.contiguous(), block_size=32, format="mxfp4")
    return q, s

def sync():
    torch.xpu.synchronize()

def bench(fn):
    fn()  # discard 1st (compile/JIT/alloc)
    for _ in range(WARM):
        fn()
    sync(); t = time.time()
    for _ in range(ITERS):
        fn()
    sync(); return (time.time() - t) / ITERS * 1000.0

# ---- build weights once ----
torch.manual_seed(0)
Wbf = (torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02)   # [N,K]
Wt_bf = Wbf.t().contiguous()                                         # [K,N] for x@Wt

# fp4 weight: quant [N,K] (group along K=last), then transpose -> [K/2,N] NT (stride0==1)
w_lp, w_sc = mxfp4(Wbf)            # w_lp [N,K/2] fp4, w_sc [N,K/32] e8m0
B_fp4 = w_lp.transpose(0, 1)       # [K/2, N], stride[0]==1 -> is_nt
print(f"\nB_fp4 shape={tuple(B_fp4.shape)} dtype={B_fp4.dtype} stride={B_fp4.stride()} "
      f"w_sc shape={tuple(w_sc.shape)} dtype={w_sc.dtype}")

# int4 weight (synthetic packed int32 [N,K/8] -> NT) for the w4a8/w4a16 baselines
qw = torch.randint(-(2**31), 2**31, (N, K // 8), device=DEV, dtype=torch.int32)
qweight = qw.t()                                                     # [K/8,N] NT
wscale = (torch.rand(K // G, N, device=DEV, dtype=torch.float16) * 0.01 + 1e-3)
wzp = torch.tensor([8], dtype=torch.int8, device=DEV)               # symmetric

def act_q_int8(x):  # per-token int8 for w4a8
    amax = x.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
    xs = (amax / 127.0).to(x.dtype)
    xq = (x / xs).round().clamp_(-127, 127).to(torch.int8).contiguous()
    zz = torch.zeros_like(amax, dtype=torch.int32).contiguous()
    return xq, xs.contiguous(), zz

print("\n==== run + bench (warm) ====")
for M in (1, 2048):
    print(f"\n--- M={M} ({'decode' if M==1 else 'prefill'}) ---")
    x_bf = (torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1)
    x_f16 = x_bf.to(torch.float16)

    # bf16 baseline
    tb = bench(lambda: x_bf @ Wt_bf)
    print(f"  bf16 matmul         {tb:8.4f} ms  (1.00x ref)")

    # ---- fp4 (w4a4) ----
    try:
        a_lp, a_sc = mxfp4(x_bf)   # a_lp [M,K/2] fp4, a_sc [M,K/32] e8m0
        y = FP4(a_lp, B_fp4, a_sc, w_sc, torch.bfloat16, None)
        fin = torch.isfinite(y).all().item()
        # op-only (pre-quantized act)
        top = bench(lambda: FP4(a_lp, B_fp4, a_sc, w_sc, torch.bfloat16, None))
        # op + eager act-quant (realistic)
        tfull = bench(lambda: FP4(*( (lambda q,s: (q,B_fp4,s,w_sc,torch.bfloat16,None))(*mxfp4(x_bf)) )))
        print(f"  fp4 w4a4 op-only    {top:8.4f} ms  ({tb/top:5.2f}x bf16)  finite={fin} out={tuple(y.shape)} {y.dtype}")
        print(f"  fp4 w4a4 op+act-q   {tfull:8.4f} ms  ({tb/tfull:5.2f}x bf16)")
    except Exception as e:
        print(f"  fp4 w4a4 FAILED:", repr(e)[:300])

    # ---- int4_gemm_w4a8 (shipped prefill path, int8 act) ----
    try:
        xq, xs, zz = act_q_int8(x_f16)
        y8 = torch.ops._xpu_C.int4_gemm_w4a8(xq, xs, zz, qweight, wscale, wzp, G, None, None)
        fin8 = torch.isfinite(y8).all().item()
        t8 = bench(lambda: torch.ops._xpu_C.int4_gemm_w4a8(xq, xs, zz, qweight, wscale, wzp, G, None, None))
        print(f"  int4 w4a8 op-only   {t8:8.4f} ms  ({tb/t8:5.2f}x bf16)  finite={fin8}")
    except Exception as e:
        print(f"  int4 w4a8 FAILED:", repr(e)[:200])

    # ---- int4_gemm_w4a16 (shipped decode path, fp16 act) ----
    try:
        y16 = torch.ops._xpu_C.int4_gemm_w4a16(x_f16, qweight, None, wscale, wzp, G, None)
        fin16 = torch.isfinite(y16).all().item()
        t16 = bench(lambda: torch.ops._xpu_C.int4_gemm_w4a16(x_f16, qweight, None, wscale, wzp, G, None))
        print(f"  int4 w4a16 op-only  {t16:8.4f} ms  ({tb/t16:5.2f}x bf16)  finite={fin16}")
    except Exception as e:
        print(f"  int4 w4a16 FAILED:", repr(e)[:200])

print("\nDONE")
