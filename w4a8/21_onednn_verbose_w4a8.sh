#!/usr/bin/env bash
# Ladder step 1 [FREE]: ONEDNN_VERBOSE=2 diagnostic on the m=1 int4_gemm_w4a8 call.
# Answers: does w4a8 decode land on the optimized 'grouped_micro_gemm' microkernel or the
# slow 'ref' fallback? Shows the zero-point correction term + the bundled oneDNN version.
# A few iters only -- we want the verbose impl strings, not timing. GPU run (gate via flock).
#   Env: IMG (vllm-xpu-env:int8), ITERS (3), WARMUP (1).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8}"; ITERS="${ITERS:-3}"; WARMUP="${WARMUP:-1}"
LOG="$ROOT/results/onednn_verbose_w4a8_$(date +%Y%m%d_%H%M%S).log"
echo "=== int4_gemm_w4a8 ONEDNN_VERBOSE=2: img=$IMG iters=$ITERS log=$LOG ==="
docker run --rm -i --name w4a8_onednn_verbose --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -e ZE_AFFINITY_MASK=0 -e ITERS="$ITERS" -e WARMUP="$WARMUP" \
  -e ONEDNN_VERBOSE=2 -e DNNL_VERBOSE=2 \
  -v "$ROOT/vllm_cache:/vllm_cache" -e XDG_CACHE_HOME=/vllm_cache \
  --entrypoint python "$IMG" - <<'PY' 2>&1 | tee "$LOG"
import os, torch
import vllm_xpu_kernels._xpu_C  # noqa: registers torch.ops._xpu_C.*
ITERS=int(os.environ.get("ITERS","3")); WARMUP=int(os.environ.get("WARMUP","1"))
print("torch", torch.__version__, "| xpu", torch.xpu.is_available(),
      "| has int4_gemm_w4a8:", hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"), flush=True)
try:
    import importlib.metadata as md
    print("oneDNN/dnnl python pkgs:", [d.metadata['Name'] for d in md.distributions()
          if 'dnnl' in d.metadata['Name'].lower() or 'onednn' in d.metadata['Name'].lower()], flush=True)
except Exception as e:
    print("pkg probe err", e, flush=True)

def quant_pt_int8(x):
    s=(x.abs().amax(-1,keepdim=True)/127.0).clamp(min=1e-8)
    q=(x/s).round().clamp(-127,127).to(torch.int8)
    zp=torch.zeros([x.shape[0],1],dtype=torch.int32,device=x.device)
    return q, s.to(torch.float16), zp
def rand_w_packed(k,n):
    r=torch.randint(-128,128,[(k*n)//2],device="xpu").to(torch.int8)
    return r.view(dtype=torch.int32).reshape(k//8,n)

def run(m,k,n,tag):
    print(f"\n##### VERBOSE_SHAPE {tag} m={m} k={k} n={n} #####", flush=True)
    g=min(128,k); gn=k//g
    x=torch.randn([m,k],device="xpu",dtype=torch.float16)
    xq,xs,xzp=quant_pt_int8(x)
    w=rand_w_packed(k,n); w_ba=w.transpose(0,1).contiguous().transpose(0,1)
    ws=torch.rand([gn,n],device="xpu",dtype=torch.float16)
    wzp=torch.tensor([8],dtype=torch.int8,device="xpu")
    bias=torch.Tensor()
    def call(): return torch.ops._xpu_C.int4_gemm_w4a8(xq,xs,xzp,w_ba,ws,wzp,g,None,bias)
    for _ in range(WARMUP): call()
    torch.xpu.synchronize()
    for _ in range(ITERS): call()
    torch.xpu.synchronize()

# decode shapes (the target) + one prefill for contrast
run(1, 5120, 17408, "DECODE_mlp_up")
run(1, 17408, 5120, "DECODE_mlp_down")
run(512, 5120, 17408, "PREFILL_mlp_up")
print("\n##### DONE #####", flush=True)
PY
echo "=== exit ${PIPESTATUS[0]} ==="
echo "=== unique oneDNN matmul impls seen ==="
grep -iE "onednn_verbose|dnnl_verbose" "$LOG" | grep -i matmul | sed -E 's/,[0-9.]+$//' | sort -u | head -40
echo "=== impl/jit strings ==="
grep -iE "onednn_verbose|dnnl_verbose" "$LOG" | grep -ioE "(gemm:[a-z_/:]+|jit:[a-z0-9_:]+|ref:[a-z_]+|micro[a-z_]*|grouped[a-z_]*)" | sort | uniq -c | sort -rn | head
