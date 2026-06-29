#!/usr/bin/env bash
# Kernel-level microbench for int4_gemm_w4a8 (the decode bottleneck). Times the oneDNN
# int4-weight x int8-act GEMM in ISOLATION at decode (m=1) and prefill (m=512) shapes, so we
# can iterate on the kernel without full-serve noise. Symmetric weights+acts (our checkpoint).
#   Decode m=1 should be ~bandwidth-bound (read int4 weights); if it's far below the card's
#   608 GB/s, the kernel is overhead-bound -> the optimization target.
# GPU run. The CPU GPTQ quant does NOT touch the GPU; expect only minor host-dispatch noise.
#   Env: IMG (vllm-xpu-env:int8), ITERS (100), WARMUP (20).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8}"; ITERS="${ITERS:-100}"; WARMUP="${WARMUP:-20}"
LOG="$ROOT/results/microbench_w4a8_$(date +%Y%m%d_%H%M%S).log"
echo "=== int4_gemm_w4a8 microbench: img=$IMG iters=$ITERS log=$LOG ==="
docker run --rm -i --name w4a8_microbench --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -e ZE_AFFINITY_MASK=0 -e ITERS="$ITERS" -e WARMUP="$WARMUP" \
  -v "$ROOT/vllm_cache:/vllm_cache" -e XDG_CACHE_HOME=/vllm_cache \
  --entrypoint python "$IMG" - <<'PY' 2>&1 | tee "$LOG"
import os, time, torch
import vllm_xpu_kernels._xpu_C  # noqa: registers torch.ops._xpu_C.*
ITERS=int(os.environ.get("ITERS","100")); WARMUP=int(os.environ.get("WARMUP","20"))
BW=608.0  # B70 GB/s
print("torch", torch.__version__, "| xpu", torch.xpu.is_available(),
      "| has int4_gemm_w4a8:", hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"), flush=True)

def quant_pt_int8(x):  # per-token symmetric int8
    s=(x.abs().amax(-1,keepdim=True)/127.0).clamp(min=1e-8)
    q=(x/s).round().clamp(-127,127).to(torch.int8)
    zp=torch.zeros([x.shape[0],1],dtype=torch.int32,device=x.device)
    return q, s.to(torch.float16), zp

def rand_w_packed(k,n):  # int4 weights packed 8-per-int32 -> [k//8, n]
    r=torch.randint(-128,128,[(k*n)//2],device="xpu").to(torch.int8)
    return r.view(dtype=torch.int32).reshape(k//8,n)

def bench(m,k,n):
    g=min(128,k); gn=k//g
    x=torch.randn([m,k],device="xpu",dtype=torch.float16)
    xq,xs,xzp=quant_pt_int8(x)
    w=rand_w_packed(k,n); w_ba=w.transpose(0,1).contiguous().transpose(0,1)
    ws=torch.rand([gn,n],device="xpu",dtype=torch.float16)
    wzp=torch.tensor([8],dtype=torch.int8,device="xpu")  # symmetric int4
    bias=torch.Tensor()
    def call(): return torch.ops._xpu_C.int4_gemm_w4a8(xq,xs,xzp,w_ba,ws,wzp,g,None,bias)
    try:
        for _ in range(WARMUP): call()
        torch.xpu.synchronize()
        t0=time.perf_counter()
        for _ in range(ITERS): call()
        torch.xpu.synchronize()
        dt=(time.perf_counter()-t0)/ITERS
        wbytes=k*n*0.5                      # int4 weight bytes (dominant for m=1)
        gbps=(wbytes + m*k + m*n*2)/dt/1e9
        print(f"  m={m:<4} k={k:<6} n={n:<6} {dt*1e3:8.3f} ms  {2*m*k*n/dt/1e12:6.2f} TFLOP/s  "
              f"~{gbps:6.1f} GB/s ({100*gbps/BW:4.1f}% of {BW:.0f})", flush=True)
    except Exception as e:
        print(f"  m={m} k={k} n={n}  ERROR: {type(e).__name__}: {e}", flush=True)

print("=== decode (m=1) -- the target; should be ~bandwidth-bound ===", flush=True)
for (k,n) in [(4096,11008),(5120,17408),(17408,5120),(5120,5120)]: bench(1,k,n)
print("=== prefill (m=512) -- int8-XMX compute regime ===", flush=True)
for (k,n) in [(5120,17408),(17408,5120)]: bench(512,k,n)
PY
echo "=== exit ${PIPESTATUS[0]} ==="
