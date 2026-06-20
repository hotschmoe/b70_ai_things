#!/usr/bin/env bash
# PP-1 validation: format_tag::any weight + cached reorder in int8_gemm_w8a8.
# A/B the BASELINE (pre-PP1 baked .so in vllm-xpu-env:int8) vs PP-1 (rebuilt host .so mounted), identical
# deterministic inputs -> (1) correctness fingerprints MUST match (reorder is layout-only, numerically
# identical), (2) prefill % of 367 TOPS should RISE, (3) decode wide-n % of 608 GB/s should rise. Plus a
# short ONEDNN_VERBOSE=2 pass to confirm the weight md is now BLOCKED (not `ab`) + a one-time reorder.
# GPU run -- gate via the gpu-run flock lease.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8}"; ITERS="${ITERS:-200}"; WARMUP="${WARMUP:-40}"
REBUILT_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
BAKED_SO=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so
STAMP="$(date +%Y%m%d_%H%M%S)"
echo "=== rebuilt .so mtime (must be fresh) ==="; ls -la --time-style=+%H:%M:%S "$REBUILT_SO"

cat > "$ROOT/pp1_validate.py" <<'PY'
import os, time, torch
import vllm_xpu_kernels._xpu_C  # noqa
MODE=os.environ.get("MODE","?"); ITERS=int(os.environ.get("ITERS","200")); WARMUP=int(os.environ.get("WARMUP","40"))
BW=608.0; TOPS=367.0
print(f"MODE={MODE} torch={torch.__version__}", flush=True)
def mk(m,k,n,seed):
    g=torch.Generator().manual_seed(seed)
    xq=torch.randint(-127,128,(m,k),generator=g,dtype=torch.int8)
    xs=(torch.rand((m,1),generator=g,dtype=torch.float32)*0.02+0.005).to(torch.float16)
    wq=torch.randint(-127,128,(k,n),generator=g,dtype=torch.int8)
    ws=(torch.rand((1,n),generator=g,dtype=torch.float32)*0.02+0.002).to(torch.float16)
    return xq.to("xpu"),xs.to("xpu"),wq.to("xpu"),ws.to("xpu")
bias=torch.Tensor()
def run(m,k,n):
    xq,xs,wq,ws=mk(m,k,n,seed=4242+k+n)
    def call(): return torch.ops._xpu_C.int8_gemm_w8a8(xq,xs,None,wq,ws,None,bias,torch.float16)
    out=call(); torch.xpu.synchronize(); o=out.float()
    fp=f"sum={o.sum().item():+.6e} absmax={o.abs().max().item():.6e}"
    for _ in range(WARMUP): call()
    torch.xpu.synchronize(); t0=time.perf_counter()
    for _ in range(ITERS): call()
    torch.xpu.synchronize(); dt=(time.perf_counter()-t0)/ITERS
    if m==1:
        gbps=(k*n+m*k+m*n*2)/dt/1e9
        print(f"DECODE  k={k:<6} n={n:<6} | {fp} | {dt*1e3:8.4f} ms {gbps:6.1f} GB/s ({100*gbps/BW:4.1f}%)",flush=True)
    else:
        tf=2*m*k*n/dt/1e12
        print(f"PREFILL m={m:<4} k={k:<6} n={n:<6} | {fp} | {dt*1e3:8.4f} ms {tf:7.1f} TFLOP/s ({100*tf/TOPS:4.1f}%)",flush=True)
# prefill FIRST (the headline pp target) on the safe k=5120 shape, then safe decodes,
# then the k=17408 DEVICE_LOST crasher LAST (isolated) so the safe results always land.
for (k,n) in [(5120,17408)]: run(512,k,n)
for (k,n) in [(5120,17408)]: run(2048,k,n)
for (k,n) in [(4096,11008),(5120,17408),(5120,5120)]: run(1,k,n)
print("--- now the k=17408 shapes (may DEVICE_LOST) ---",flush=True)
for (k,n) in [(17408,5120)]: run(512,k,n)
for (k,n) in [(17408,5120)]: run(2048,k,n)
for (k,n) in [(17408,5120)]: run(1,k,n)
print("DONE",flush=True)
PY

run () { # mode log [extra docker args]
  local mode="$1" log="$2"; shift 2
  docker run --rm --name "pp1_$mode" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0 \
    -e MODE="$mode" -e ITERS="$ITERS" -e WARMUP="$WARMUP" -v "$ROOT/vllm_cache:/vllm_cache" -e XDG_CACHE_HOME=/vllm_cache \
    -v "$ROOT/pp1_validate.py:/pp1_validate.py" "$@" --entrypoint python "$IMG" /pp1_validate.py 2>&1 | tee "$log"
}
echo; echo "############ BASELINE (pre-PP1 baked .so) ############"; run baseline "$ROOT/results/pp1_base_$STAMP.log"
echo; echo "############ PP-1 (rebuilt .so mounted) ############"; run pp1 "$ROOT/results/pp1_new_$STAMP.log" -v "$REBUILT_SO:$BAKED_SO:ro"
echo; echo "############ PP-1 ONEDNN_VERBOSE=2 (weight md blocked? one-time reorder?) ############"
docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0 -e MODE=verb -e ITERS=2 -e WARMUP=1 \
  -e ONEDNN_VERBOSE=2 -v "$ROOT/vllm_cache:/vllm_cache" -e XDG_CACHE_HOME=/vllm_cache \
  -v "$ROOT/pp1_validate.py:/pp1_validate.py" -v "$REBUILT_SO:$BAKED_SO:ro" --entrypoint python "$IMG" /pp1_validate.py 2>&1 \
  | grep -iE "onednn_verbose.*matmul|reorder" | grep -oiE "wei:[a-z0-9_:]+|reorder|jit:[a-z:]+|src_b:s8::[a-z]+" | sort | uniq -c | sort -rn | head
echo; echo "############ VERDICT (correctness must match; % should rise) ############"
paste <(grep -E "^DECODE|^PREFILL" "$ROOT/results/pp1_base_$STAMP.log") <(grep -E "^DECODE|^PREFILL" "$ROOT/results/pp1_new_$STAMP.log") \
  | awk -F'|' '{print "base:"$1$3"   pp1:"$4$6}'
