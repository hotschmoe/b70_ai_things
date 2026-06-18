#!/usr/bin/env bash
# Serve Qwen3-14B (existing BF16 at specula-build) on the single B70 for the quant sweep.
# QUANT: none (F16/BF16, tight ~28GB) | fp8 (online, ~15GB, XMX fast path) | path to a
# pre-quantized checkpoint (e.g. self-made W8A8). All caches on SSD. Standard GQA -> default
# (flash) attention backend works; no DeltaNet/multimodal special-casing.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${VLLM_IMG:-vllm-xpu-env:tf}"
SPECULA=/mnt/vm_8tb/specula-build/models
MODEL="${MODEL:-/specula_models/Qwen3-14B}"
NAME="vllm_qwen3"; PORT=18080
QUANT="${QUANT:-fp8}"               # none | fp8 | <checkpoint path under /models or /specula_models>
MAXLEN="${MAXLEN:-16384}"; MAXSEQS="${MAXSEQS:-16}"; UTIL="${UTIL:-0.90}"
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd}
docker rm -f "$NAME" 2>/dev/null || true

QARG=""; case "$QUANT" in none|"") QARG="";; fp8) QARG="--quantization fp8";; *) QARG="";; esac
SERVE_MODEL="$MODEL"; [ "$QUANT" != none ] && [ "$QUANT" != fp8 ] && SERVE_MODEL="$QUANT"

echo "=== serve Qwen3-14B  quant=$QUANT  model=$SERVE_MODEL  maxlen=$MAXLEN seqs=$MAXSEQS util=$UTIL ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} \
  -v "$ROOT/models:/models:ro" -v "$SPECULA:/specula_models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK=0 \
  --entrypoint vllm "$IMG" \
  serve "$SERVE_MODEL" \
    --served-model-name qwen3-14b \
    --host 0.0.0.0 --port ${PORT} \
    $QARG \
    --tensor-parallel-size 1 --enforce-eager \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
    --gpu-memory-utilization "$UTIL" --no-enable-prefix-caching --trust-remote-code

echo "=== waiting for readiness (up to ~10 min) ==="
ok=0
for i in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "exited early"; break; }
  sleep 5
done
echo "=== key startup lines ==="
docker logs "$NAME" 2>&1 | grep -iE 'Resolved architecture|XPUFP8|Fp8|compressed|int8|w8a8|LinearMethod|Using.*Attention|Model loading took|Available KV cache|Maximum concurrency|Application startup complete|error|Traceback|out of memory' | grep -viE 'OperatorEntry|registered|dispatch' | tail -18
[ "$ok" = 1 ] && echo "HEALTHY :$PORT model 'qwen3-14b' quant=$QUANT" || { echo "NOT HEALTHY"; docker logs "$NAME" 2>&1 | tail -25; }
