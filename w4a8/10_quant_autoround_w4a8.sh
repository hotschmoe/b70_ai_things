#!/usr/bin/env bash
# WIN 2 (accuracy) -- AutoRound W4A8-INT 14B, B70-ACCELERATED.
# AutoRound = Intel sign-gradient int4 rounding (what the strong 27B int4 0.927-plus used).
# It optimizes int4 WEIGHTS; we add int8 dynamic activations to get a W4A8 checkpoint that
# routes to the B70's XPUW4A8IntLinearKernel. AutoRound's compressed-tensors export should also
# PACK the int4 weights -> may close Win 1 (packing) at the same time. VERIFY size + dtype after.
#
# Compute placement: DEVICE=xpu runs the rounding on the B70 (much faster than CPU). The repo's
# older note "XPU calibration unreliable" was for llm-compressor SmoothQuant; RETEST for AutoRound.
# Falls back to DEVICE=cpu if the XPU toolchain misbehaves.
#
# [!] FIRST RUN: this is a DRAFT toolchain. Validate (a) auto-round XPU device works, (b) the
#     --act_bits/--act_dynamic/--format flags match the installed auto-round (`auto-round --help`),
#     (c) the export routes to CompressedTensorsW4A8Int on serve. Start with ITERS small.
#
#   Env: DEVICE=xpu|cpu (xpu), ITERS (200; use 50 for a smoke run), NSAMPLES (128), SEQLEN (2048),
#        GROUP (128), OUTNAME (Qwen3-14B-W4A8-autoround), IMAGE (vllm-xpu-env:v0230).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SPECULA=/mnt/vm_8tb/specula-build/models
SRC="${SRC:-/specula_models/Qwen3-14B}"
DEVICE="${DEVICE:-xpu}"; ITERS="${ITERS:-200}"; NSAMPLES="${NSAMPLES:-128}"; SEQLEN="${SEQLEN:-2048}"
GROUP="${GROUP:-128}"; OUTNAME="${OUTNAME:-Qwen3-14B-W4A8-autoround}"; IMAGE="${IMAGE:-vllm-xpu-env:v0230}"
LOG="$ROOT/results/w4a8_autoround_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/models" "$ROOT/pip_cache"
echo "=== AutoRound W4A8: src=$SRC out=$OUTNAME device=$DEVICE iters=$ITERS nsamples=$NSAMPLES image=$IMAGE log=$LOG ==="

# GPU passthrough only when using the B70.
GPUARGS=(); [ "$DEVICE" = xpu ] && GPUARGS=(--device /dev/dri -e ZE_AFFINITY_MASK=0)

docker run --rm --name w4a8_autoround "${GPUARGS[@]}" \
  -v "$SPECULA:/specula_models:ro" -v "$ROOT/models:/models" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/pip_cache:/root/.cache/pip" \
  -v "$ROOT/vllm_cache:/vllm_cache" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/vllm_cache -e OMP_NUM_THREADS=32 \
  -e SRC="$SRC" -e OUTNAME="$OUTNAME" -e DEVICE="$DEVICE" -e ITERS="$ITERS" \
  -e NSAMPLES="$NSAMPLES" -e SEQLEN="$SEQLEN" -e GROUP="$GROUP" \
  "$IMAGE" bash -c '
    set -e
    # auto-round on top of the image torch-xpu build. --no-deps avoids clobbering torch; then
    # pull only the extra deps it needs. (If this misbehaves, pin auto-round and adjust.)
    pip install -q --no-deps auto-round || pip install -q auto-round
    pip install -q "transformers>=4.52" accelerate datasets || true
    python - <<PY
import os, torch
DEV=os.environ["DEVICE"]
try:
    import intel_extension_for_pytorch as ipex  # noqa
except Exception as e:
    print("[diag] ipex import:", e, flush=True)
print("[diag] torch", torch.__version__, "| xpu available:",
      getattr(torch, "xpu", None) is not None and torch.xpu.is_available(), flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
SRC=os.environ["SRC"]; OUT="/models/"+os.environ["OUTNAME"]
G=int(os.environ["GROUP"]); IT=int(os.environ["ITERS"]); NS=int(os.environ["NSAMPLES"]); SQ=int(os.environ["SEQLEN"])
dev = "xpu" if (DEV=="xpu" and torch.xpu.is_available()) else "cpu"
print(f"[load] {SRC} on {dev} ...", flush=True)
model=AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto")
tok=AutoTokenizer.from_pretrained(SRC)

# W4A8: int4 sym group-G weights + int8 dynamic activations. (Flag names may vary by auto-round
# version -- verify with `auto-round --help`; act_bits=8 + act_dynamic is the W4A8 intent.)
ar=AutoRound(model, tok, bits=4, group_size=G, sym=True,
             act_bits=8, act_dynamic=True,
             iters=IT, nsamples=NS, seqlen=SQ, device=dev)
print(f"[quant] AutoRound W4A8 iters={IT} nsamples={NS} ...", flush=True)
ar.quantize()
# Export to a compressed-tensors / llm_compressor format that vLLM-XPU reads as W4A8-int.
print(f"[save] {OUT} (format=llm_compressor) ...", flush=True)
ar.save_quantized(OUT, format="llm_compressor")
tok.save_pretrained(OUT)
print("DONE_AUTOROUND", OUT, flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit: ${PIPESTATUS[0]} ==="
echo "--- size (want PACKED ~9 GB, not 16 GB) ---"; du -sh "$ROOT/models/$OUTNAME" 2>/dev/null
echo "--- format + weight dtype (want I32/packed) ---"
grep -ao '"format":"[a-z-]*"' "$ROOT/models/$OUTNAME/config.json" 2>/dev/null | head -3
head -c 3000000 "$ROOT/models/$OUTNAME"/*.safetensors 2>/dev/null | grep -ao '"dtype":"[A-Z0-9_]*"' | sort | uniq -c
echo "NEXT: serve on B70 + HumanEval+ Tier-1; confirm XPUW4A8IntLinearKernel + resident GiB."
