#!/usr/bin/env bash
# DUAL-CARD (or N-card) vLLM-XPU server for when card #2+ arrives. Adds tensor/pipeline
# parallel + the #41663 Battlemage multi-GPU stability env. Single-card knobs same as 36.
# Env: TP (tensor-parallel, default 2), PP (pipeline-parallel, default 1), plus
#   MODEL, SERVED, QUANT (none|fp8|<ckpt>), KVDTYPE, MAXLEN, MAXSEQS, UTIL, IMG, NAME, EXTRA.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
SPECULA=/mnt/vm_8tb/specula-build/models
MODEL="${MODEL:-/specula_models/Qwen3-14B}"; SERVED="${SERVED:-qwen3}"
QUANT="${QUANT:-fp8}"; KVDTYPE="${KVDTYPE:-auto}"
TP="${TP:-2}"; PP="${PP:-1}"
MAXLEN="${MAXLEN:-16384}"; MAXSEQS="${MAXSEQS:-32}"; UTIL="${UTIL:-0.90}"; EXTRA="${EXTRA:-}"
NAME="${NAME:-vllm_multi}"; PORT=18080
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd}
docker rm -f "$NAME" 2>/dev/null || true

SERVE_TARGET="$MODEL"; QARG=()
case "$QUANT" in none|"") ;; fp8) QARG=(--quantization fp8);; *) SERVE_TARGET="$QUANT";; esac

ARGS=(serve "$SERVE_TARGET" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT"
      "${QARG[@]}" --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
      --distributed-executor-backend mp --enforce-eager
      --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL"
      --kv-cache-dtype "$KVDTYPE" --no-enable-prefix-caching --trust-remote-code)
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

# Battlemage multi-GPU stability env (vLLM #41663): no Arc P2P, CPU-driven oneCCL.
echo "=== MULTI-GPU serve: TP=$TP PP=$PP QUANT=$QUANT model=$SERVE_TARGET ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g -p ${PORT}:${PORT} \
  -v "$ROOT/models:/models:ro" -v "$SPECULA:/specula_models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd \
  -e CCL_ENABLE_SYCL_KERNELS=0 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint vllm "$IMG" "${ARGS[@]}"

echo "=== waiting for readiness (multi-GPU init slower; up to ~15 min) ==="
ok=0
for i in $(seq 1 180); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED"; break; }
  sleep 5
done
docker logs "$NAME" 2>&1 | grep -iE 'Resolved architecture|tensor.parallel|pipeline|world_size|rank|XPUFp8|GDN|Model loading took|Available KV cache|Maximum concurrency|Application startup complete|error|Traceback|gp fault|engine reset|out of memory|CCL|oneccl' | grep -viE 'OperatorEntry|dispatch' | tail -25
[ "$ok" = 1 ] && echo "HEALTHY :$PORT '$SERVED' TP=$TP PP=$PP" || { echo "NOT HEALTHY"; docker logs "$NAME" 2>&1 | tail -25; }
