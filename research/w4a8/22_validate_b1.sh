#!/usr/bin/env bash
# B1 validation: drop-symmetric-src-zp patch to int4_gemm_w4a8.h.
# Runs the SYMMETRIC decode path (s8 acts, zero src-zp -- the real production path) through:
#   (1) BASELINE: the pre-patch .so baked into vllm-xpu-env:int8
#   (2) PATCHED : the freshly rebuilt host .so mounted over the baked path
# Identical deterministic inputs both runs -> outputs MUST match (dropped src-zp was 0), which is a
# stronger correctness check than the existing asym-path unit test. Then a short ONEDNN_VERBOSE=2 run
# proves the `zero_points:src` attr is gone and the impl did not bounce to `ref`. Reports timing delta.
# GPU run -- invoke via the gpu-run flock lease.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8}"
ITERS="${ITERS:-200}"; WARMUP="${WARMUP:-30}"
REBUILT_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
BAKED_SO=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_LOG="$ROOT/results/b1_baseline_${STAMP}.log"
PATCH_LOG="$ROOT/results/b1_patched_${STAMP}.log"
VERB_LOG="$ROOT/results/b1_verbose_${STAMP}.log"

echo "=== rebuilt .so mtime (must be fresh, after the patch) ==="
ls -la --time-style=+%Y-%m-%d_%H:%M:%S "$REBUILT_SO"

cat > "$ROOT/b1_validate.py" <<'PY'
import os, time, torch
import vllm_xpu_kernels._xpu_C  # noqa: registers torch.ops._xpu_C.*
MODE=os.environ.get("MODE","?"); ITERS=int(os.environ.get("ITERS","200")); WARMUP=int(os.environ.get("WARMUP","30"))
BW=608.0
print(f"MODE={MODE} torch={torch.__version__} has_int4={hasattr(torch.ops._xpu_C,'int4_gemm_w4a8')}", flush=True)

def make_inputs(m,k,n,seed):
    g=torch.Generator().manual_seed(seed)
    # symmetric per-token int8 activations (zero src-zp -- the production path)
    xq=torch.randint(-127,128,(m,k),generator=g,dtype=torch.int8)
    xs=(torch.rand((m,1),generator=g,dtype=torch.float32)*0.02+0.005).to(torch.float16)
    xzp=torch.zeros((m,1),dtype=torch.int32)
    # int4 weights packed 8-per-int32 -> [k/8, n], symmetric weight zp=8
    wb=torch.randint(-128,128,((k*n)//2,),generator=g,dtype=torch.int8).view(torch.int32).reshape(k//8,n)
    gsz=min(128,k); gn=k//gsz
    ws=(torch.rand((gn,n),generator=g,dtype=torch.float32)*0.02+0.002).to(torch.float16)
    wzp=torch.tensor([8],dtype=torch.int8)
    return (xq.to("xpu"), xs.to("xpu"), xzp.to("xpu"),
            wb.to("xpu").transpose(0,1).contiguous().transpose(0,1),
            ws.to("xpu"), wzp.to("xpu"), gsz)

SHAPES=[(1,4096,11008),(1,5120,17408),(1,17408,5120),(1,5120,5120)]
bias=torch.Tensor()
for (m,k,n) in SHAPES:
    xq,xs,xzp,wb,ws,wzp,gsz=make_inputs(m,k,n,seed=1234+k+n)
    def call(): return torch.ops._xpu_C.int4_gemm_w4a8(xq,xs,xzp,wb,ws,wzp,gsz,None,bias)
    out=call(); torch.xpu.synchronize()
    # correctness fingerprint (deterministic inputs -> compare across MODE)
    o=out.float()
    fp_sum=o.sum().item(); fp_absmax=o.abs().max().item(); fp_mean=o.mean().item()
    for _ in range(WARMUP): call()
    torch.xpu.synchronize(); t0=time.perf_counter()
    for _ in range(ITERS): call()
    torch.xpu.synchronize(); dt=(time.perf_counter()-t0)/ITERS
    gbps=(k*n*0.5 + m*k + m*n*2)/dt/1e9
    print(f"SHAPE k={k:<6} n={n:<6} | sum={fp_sum:+.6e} absmax={fp_absmax:.6e} mean={fp_mean:+.6e} "
          f"| {dt*1e3:8.4f} ms {gbps:6.1f} GB/s ({100*gbps/BW:4.1f}%)", flush=True)
print("DONE", flush=True)
PY

run () {  # $1=mode  $2=logpath  $3=extra docker args
  local mode="$1" log="$2"; shift 2
  docker run --rm -i --name "b1_val_${mode}" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    -e ZE_AFFINITY_MASK=0 -e MODE="$mode" -e ITERS="$ITERS" -e WARMUP="$WARMUP" \
    -v "$ROOT/vllm_cache:/vllm_cache" -e XDG_CACHE_HOME=/vllm_cache \
    -v "$ROOT/b1_validate.py:/b1_validate.py" "$@" \
    --entrypoint python "$IMG" /b1_validate.py 2>&1 | tee "$log"
}

echo; echo "############ BASELINE (pre-patch baked .so) ############"
run baseline "$BASE_LOG"
echo; echo "############ PATCHED (rebuilt .so mounted) ############"
run patched "$PATCH_LOG" -v "$REBUILT_SO:$BAKED_SO:ro"

echo; echo "############ PATCHED ONEDNN_VERBOSE=2 (proof: no zero_points:src, not ref) ############"
docker run --rm -i --name b1_val_verbose --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -e ZE_AFFINITY_MASK=0 -e MODE=verbose -e ITERS=2 -e WARMUP=1 -e ONEDNN_VERBOSE=2 \
  -v "$ROOT/vllm_cache:/vllm_cache" -e XDG_CACHE_HOME=/vllm_cache \
  -v "$ROOT/b1_validate.py:/b1_validate.py" -v "$REBUILT_SO:$BAKED_SO:ro" \
  --entrypoint python "$IMG" /b1_validate.py 2>&1 | tee "$VERB_LOG" | grep -iE "onednn_verbose.*matmul" | head -3

echo; echo "############ VERDICT ############"
echo "--- impl + zp attr on PATCHED (expect jit:gemm, NO zero-points:src0) ---"
grep -iE "onednn_verbose.*matmul" "$VERB_LOG" | head -1 | grep -oiE "jit:[a-z:]+|ref:[a-z_]+" | head -1
grep -iE "onednn_verbose.*matmul" "$VERB_LOG" | head -1 | grep -oiE "zero-points:src0[^ ]*" && echo "!!! src zp STILL PRESENT (patch did NOT take)" || echo "OK: no src zero-point attr (patch took effect)"
echo "--- correctness: baseline vs patched fingerprints (must match) ---"
paste <(grep '^SHAPE' "$BASE_LOG") <(grep '^SHAPE' "$PATCH_LOG") | \
  awk '{print "  base: "$0}' | sed 's/SHAPE/\nSHAPE/2'
echo "--- timing deltas ---"
python3 - "$BASE_LOG" "$PATCH_LOG" <<'PYC'
import sys,re
def parse(p):
    d={}
    for ln in open(p):
        m=re.search(r'k=(\d+)\s+n=(\d+).*?([\d.]+) ms\s+([\d.]+) GB/s \(([\d.]+)%\).*?sum=([+\-\d.e]+)',ln) or \
          re.search(r'k=(\d+)\s+n=(\d+).*?sum=([+\-\d.e]+).*?([\d.]+) ms\s+([\d.]+) GB/s \(([\d.]+)%\)',ln)
        if 'ms' in ln and 'k=' in ln:
            kk=re.search(r'k=(\d+)\s+n=(\d+)',ln); ms=re.search(r'([\d.]+) ms',ln); pct=re.search(r'\(([\d.]+)%\)',ln); s=re.search(r'sum=([+\-\d.e]+)',ln)
            d[(kk.group(1),kk.group(2))]=(float(ms.group(1)),float(pct.group(1)),s.group(1))
    return d
b=parse(sys.argv[1]); p=parse(sys.argv[2])
print(f"  {'shape':>16} {'base ms':>10} {'patch ms':>10} {'delta%':>8} {'base%pk':>8} {'patch%pk':>8}  sum_match")
for key in b:
    if key in p:
        bm,bp,bs=b[key]; pm,pp,ps=p[key]
        dl=100*(pm-bm)/bm
        match="YES" if bs==ps else f"NO ({bs} vs {ps})"
        print(f"  k={key[0]:>5} n={key[1]:>5} {bm:>10.4f} {pm:>10.4f} {dl:>+7.1f}% {bp:>7.1f}% {pp:>7.1f}%  {match}")
PYC
echo "logs: $BASE_LOG | $PATCH_LOG | $VERB_LOG"
