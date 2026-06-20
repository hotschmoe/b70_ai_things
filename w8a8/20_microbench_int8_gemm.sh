#!/usr/bin/env bash
# Baseline profile of our oneDNN int8_gemm_w8a8 (s8s8s32) at PREFILL (m=512/2048, compute-bound -> % of
# 367 INT8 TOPS) and DECODE (m=1, BW-bound -> % of 608 GB/s). Grounds the hand-tuning: where is the gap to
# peak? ONEDNN_VERBOSE=2 on one shape shows the jit impl + whether it is leaving XMX util on the table.
# Qwen3-14B shapes. GPU run -- gate via the gpu-run flock lease.
#   Env: IMG (vllm-xpu-env:int8), ITERS (100), WARMUP (20), VERBOSE (0|1).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8}"; ITERS="${ITERS:-100}"; WARMUP="${WARMUP:-20}"; VERBOSE="${VERBOSE:-0}"
LOG="$ROOT/results/microbench_int8gemm_$(date +%Y%m%d_%H%M%S).log"
VENV=(); [ "$VERBOSE" = 1 ] && VENV=(-e ONEDNN_VERBOSE=2)
echo "=== int8_gemm_w8a8 microbench: img=$IMG iters=$ITERS verbose=$VERBOSE log=$LOG ==="
docker run --rm -i --name int8gemm_microbench --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -e ZE_AFFINITY_MASK=0 -e ITERS="$ITERS" -e WARMUP="$WARMUP" "${VENV[@]}" \
  -v "$ROOT/vllm_cache:/vllm_cache" -e XDG_CACHE_HOME=/vllm_cache \
  --entrypoint python "$IMG" - <<'PY' 2>&1 | tee "$LOG"
import os, time, torch
import vllm_xpu_kernels._xpu_C  # noqa: registers torch.ops._xpu_C.*
ITERS=int(os.environ.get("ITERS","100")); WARMUP=int(os.environ.get("WARMUP","20"))
BW=608.0; TOPS=367.0  # B70 peak GB/s + INT8 TOPS
print("torch", torch.__version__, "| has int8_gemm_w8a8:",
      hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"), flush=True)

def bench(m,k,n):
    # per-token int8 src + per-channel int8 weight [k,n] (matches xpu_int8.py apply_weights after process_weights)
    x_q=torch.randint(-127,128,(m,k),device="xpu",dtype=torch.int8)
    x_s=(torch.rand((m,1),device="xpu",dtype=torch.float32)*0.02+0.005).to(torch.float16)
    w_q=torch.randint(-127,128,(k,n),device="xpu",dtype=torch.int8)
    w_s=(torch.rand((1,n),device="xpu",dtype=torch.float32)*0.02+0.002).to(torch.float16)
    bias=torch.Tensor()
    def call(): return torch.ops._xpu_C.int8_gemm_w8a8(x_q,x_s,None,w_q,w_s,None,bias,torch.float16)
    try:
        for _ in range(WARMUP): call()
        torch.xpu.synchronize(); t0=time.perf_counter()
        for _ in range(ITERS): call()
        torch.xpu.synchronize(); dt=(time.perf_counter()-t0)/ITERS
        tflops=2*m*k*n/dt/1e12
        gbps=(k*n + m*k + m*n*2)/dt/1e9     # int8 weights (1B) dominate decode
        tag = "DECODE " if m==1 else "PREFILL"
        if m==1:
            print(f"  {tag} m={m:<4} k={k:<6} n={n:<6} {dt*1e3:8.3f} ms  {gbps:6.1f} GB/s ({100*gbps/BW:4.1f}% of {BW:.0f})", flush=True)
        else:
            print(f"  {tag} m={m:<4} k={k:<6} n={n:<6} {dt*1e3:8.3f} ms  {tflops:7.1f} TFLOP/s ({100*tflops/TOPS:4.1f}% of {TOPS:.0f} TOPS)", flush=True)
    except Exception as e:
        print(f"  m={m} k={k} n={n}  ERROR: {type(e).__name__}: {e}", flush=True)

SHAPES=[(4096,11008),(5120,17408),(17408,5120),(5120,5120)]
print("=== DECODE (m=1) -- BW-bound; int8 weights -> ceiling ~608/14GB ~= 43 t/s ===", flush=True)
for (k,n) in SHAPES: bench(1,k,n)
print("=== PREFILL (m=512) -- compute-bound; target 367 INT8 TOPS ===", flush=True)
for (k,n) in SHAPES: bench(512,k,n)
print("=== PREFILL (m=2048) ===", flush=True)
for (k,n) in [(5120,17408),(17408,5120)]: bench(2048,k,n)
PY
echo "=== exit ${PIPESTATUS[0]} ==="
if [ "$VERBOSE" = 1 ]; then echo "=== unique impls ==="; grep -iE "onednn_verbose.*matmul" "$LOG" | grep -oiE "jit:[a-z0-9_:]+|gemm:[a-z_/:]+|ref:[a-z_]+" | sort | uniq -c | sort -rn | head; fi
