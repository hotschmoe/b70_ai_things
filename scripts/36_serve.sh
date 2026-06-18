#!/usr/bin/env bash
# Workhorse vLLM-XPU server for the B70 quant/feature sweep. Env knobs:
#   MODEL (default Qwen3-14B path), SERVED (qwen3-14b), QUANT (none|fp8|<ckpt path>),
#   COMPILE (0|1: 1=piecewise cudagraph, drop enforce-eager), KVDTYPE (auto|fp8),
#   MAXLEN, MAXSEQS, UTIL, DRAFT (draft model path for spec-decode), DRAFTN (num spec toks),
#   EXTRA (extra raw args), NAME, IMG.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:tf}"
SPECULA=/mnt/vm_8tb/specula-build/models
MODEL="${MODEL:-/specula_models/Qwen3-14B}"
SERVED="${SERVED:-qwen3-14b}"
QUANT="${QUANT:-fp8}"; COMPILE="${COMPILE:-0}"; KVDTYPE="${KVDTYPE:-auto}"
MAXLEN="${MAXLEN:-16384}"; MAXSEQS="${MAXSEQS:-16}"; UTIL="${UTIL:-0.90}"
DRAFT="${DRAFT:-}"; DRAFTN="${DRAFTN:-3}"; EXTRA="${EXTRA:-}"
NAME="${NAME:-vllm_qwen3}"; PORT=18080
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd}
# free port 18080 + GPU: remove ANY known vllm serving container, not just $NAME (else a stale
# vllm_w4a8/vllm_w8a8 keeps the port and the new server silently fails to bind -> false HEALTHY).
docker rm -f "$NAME" vllm_qwen3 vllm_w4a8 vllm_w8a8 vllm_int8 2>/dev/null || true

# serve target + quant flag
SERVE_TARGET="$MODEL"; QARG=()
case "$QUANT" in
  none|"") ;;
  fp8) QARG=(--quantization fp8) ;;
  *) SERVE_TARGET="$QUANT" ;;   # path to a pre-quantized checkpoint (auto-detected)
esac

ARGS=(serve "$SERVE_TARGET" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT"
      "${QARG[@]}" --tensor-parallel-size 1 --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS"
      --gpu-memory-utilization "$UTIL" --kv-cache-dtype "$KVDTYPE"
      --no-enable-prefix-caching --trust-remote-code)
if [ "$COMPILE" = 1 ]; then
  ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE","use_inductor_graph_partition":true,"compile_sizes":[1]}')
else
  ARGS+=(--enforce-eager)
fi
[ -n "$DRAFT" ] && ARGS+=(--speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFTN}")
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

echo "=== serve: QUANT=$QUANT COMPILE=$COMPILE KVDTYPE=$KVDTYPE MAXLEN=$MAXLEN SEQS=$MAXSEQS UTIL=$UTIL DRAFT=${DRAFT:-none} ==="
echo "vllm ${ARGS[*]}"
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} \
  -v "$ROOT/models:/models:ro" -v "$SPECULA:/specula_models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK=0 \
  --entrypoint vllm "$IMG" "${ARGS[@]}"

echo "=== waiting for readiness (up to ~12 min; compile/quant first-run slower) ==="
ok=0
for i in $(seq 1 144); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; break; }
  sleep 5
done
docker logs "$NAME" 2>&1 | grep -iE 'Resolved architecture|XPUFP8|Fp8|compressed|int8|w8a8|LinearMethod|Using.*Attention|Speculative|draft|Model loading took|Available KV cache|GPU KV cache size|Maximum concurrency|Application startup complete|error|Traceback|out of memory|CUDAGraph|cudagraph|capturing' | grep -viE 'OperatorEntry|registered|dispatch' | tail -22
[ "$ok" = 1 ] && echo "HEALTHY :$PORT '$SERVED'" || { echo "NOT HEALTHY"; docker logs "$NAME" 2>&1 | tail -30; }
