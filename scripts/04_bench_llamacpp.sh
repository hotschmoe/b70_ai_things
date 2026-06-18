#!/usr/bin/env bash
# Benchmark a GGUF on the B70 via llama.cpp SYCL backend (llama-bench).
# Reports prefill (pp) and decode (tg) tok/s. Results appended to results/.
# Usage: MODEL=/models/<dir>/<file>.gguf [NGL=99] [PP=512] [TG=128] bash 04_bench_llamacpp.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="${MODEL:-/models/unsloth_Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf}"
NGL="${NGL:-99}"; PP="${PP:-512}"; TG="${TG:-128}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/results/llamacpp_${STAMP}.txt"

echo "MODEL=$MODEL NGL=$NGL PP=$PP TG=$TG" | tee "$OUT"
mkdir -p "$ROOT/.sycl_cache"
# Critical Battlemage/Level-Zero env (from literature/01_backends.md):
#  - RELAXED_ALLOCATION_LIMITS: escape 4GB alloc cap, needed for >4GB model residency
#  - IMMEDIATE_COMMANDLISTS: lower-latency submission path
#  - SYCL_CACHE_PERSISTENT + DIR on SSD: skip ~27s JIT recompile each run
# Binaries live in /app and are NOT on PATH; override entrypoint with full path.
docker run --rm --device /dev/dri \
  -v "$ROOT/models:/models" \
  -v "$ROOT/.sycl_cache:/sycl_cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 \
  -e SYCL_CACHE_PERSISTENT=1 \
  -e SYCL_CACHE_DIR=/sycl_cache \
  --entrypoint /app/llama-bench \
  "$IMG" \
  -m "$MODEL" -ngl "$NGL" -p "$PP" -n "$TG" -fa 1 2>&1 | tee -a "$OUT"

echo "===== saved to $OUT =====" | tee -a "$OUT"
