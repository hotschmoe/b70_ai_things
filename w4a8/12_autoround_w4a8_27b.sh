#!/usr/bin/env bash
# AutoRound W4A8-int8 of the dense 27B (Qwen3.6-27B GDN). Corrected auto_round 0.13.1 recipe
# (driver: _autoround_w4a8.py). int4 sym group-128 weights + per-token dynamic int8 acts ->
# compressed-tensors W4A8Int -> XPUW4A8IntLinearKernel (int4_gemm_w4a8). GPU compute, held via
# gpu-run for the whole run. Source = bf16 Qwen_Qwen3.6-27B (72 GB; fits 125 GB host RAM).
#   Env: ITERS (200; 8 for smoke), NSAMPLES (128; 32 smoke), SEQLEN (2048), OUTNAME, IMAGE (v0230).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
ITERS="${ITERS:-200}"; NSAMPLES="${NSAMPLES:-128}"; SEQLEN="${SEQLEN:-2048}"
OUTNAME="${OUTNAME:-Qwen3.6-27B-W4A8-autoround}"; IMAGE="${IMAGE:-vllm-xpu-env:v0230}"
SRC=/models/Qwen_Qwen3.6-27B; OUT=/models/$OUTNAME
LOG="$ROOT/results/w4a8_ar_27b_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/pip_cache"
echo "=== AutoRound W4A8 27B: ITERS=$ITERS NSAMPLES=$NSAMPLES OUT=$OUT IMG=$IMAGE ==="
echo "=== log -> $LOG ==="
"$ROOT/gpu-run" docker run --rm --name w4a8_ar_27b --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -e ZE_AFFINITY_MASK=0 -e OMP_NUM_THREADS=32 \
  -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache \
  -v "$ROOT/pip_cache:/pipcache" -e PIP_CACHE_DIR=/pipcache \
  -v "$ROOT/_autoround_w4a8.py:/work/_autoround_w4a8.py:ro" \
  -e SRC="$SRC" -e OUT="$OUT" -e ITERS="$ITERS" -e NSAMPLES="$NSAMPLES" -e SEQLEN="$SEQLEN" \
  --entrypoint bash "$IMAGE" -lc '
    set -e
    python -c "import auto_round" 2>/dev/null || pip install -q --no-deps auto-round 2>&1 | tail -3 || pip install -q auto-round 2>&1 | tail -3
    python -c "import accelerate, datasets" 2>/dev/null || pip install -q accelerate datasets 2>&1 | tail -1 || true
    python /work/_autoround_w4a8.py' 2>&1 | tee "$LOG"
echo "QUANT_EXIT=${PIPESTATUS[0]}"
