#!/usr/bin/env bash
# Serve 14B W8A8 (vllm-xpu-env:int8) for the ngram spec-decode PoC (HANDOFF step #2).
# Same settled engine config as scripts/48 (int8 linear + FP8 KV cache, eager), but with an optional
# ngram speculative-config so we can bench spec vs no-spec with the SAME harness (scripts/38).
#   SPEC=0 -> plain serve (baseline).  SPEC=1 -> ngram prompt-lookup spec-decode.
#   Env: SPEC (default 1), NSPEC (num_speculative_tokens=4), PLMAX (prompt_lookup_max=3),
#        PLMIN (prompt_lookup_min=2), KVDTYPE (fp8_e4m3), MAXLEN (8192), UTIL (0.90).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG=vllm-xpu-env:int8
MODEL="$ROOT/models/Qwen3-14B-W8A8-INT8"
NAME=vllm_int8; PORT=18080
SPEC="${SPEC:-1}"; NSPEC="${NSPEC:-4}"; PLMAX="${PLMAX:-3}"; PLMIN="${PLMIN:-2}"
KVDTYPE="${KVDTYPE:-fp8_e4m3}"; MAXLEN="${MAXLEN:-8192}"; UTIL="${UTIL:-0.90}"
# GRAPH=1 -> enable vLLM XPU graph capture (VLLM_XPU_ENABLE_XPU_GRAPH=1) and DROP --enforce-eager
# (graph capture is incompatible with eager). Tests whether capture cuts XPU launch overhead.
GRAPH="${GRAPH:-0}"

SPEC_ARGS=()
if [ "$SPEC" = 1 ]; then
  SPEC_JSON="{\"method\":\"ngram\",\"num_speculative_tokens\":${NSPEC},\"prompt_lookup_max\":${PLMAX},\"prompt_lookup_min\":${PLMIN}}"
  SPEC_ARGS=(--speculative-config "$SPEC_JSON")
  echo "=== SPEC ON: $SPEC_JSON ==="
else
  echo "=== SPEC OFF (baseline) ==="
fi

GRAPH_ENV=(); EAGER_ARG=(--enforce-eager)
if [ "$GRAPH" = 1 ]; then
  GRAPH_ENV=(-e VLLM_XPU_ENABLE_XPU_GRAPH=1); EAGER_ARG=()
  echo "=== XPU GRAPH CAPTURE ON (VLLM_XPU_ENABLE_XPU_GRAPH=1, no --enforce-eager) ==="
fi

docker rm -f "$NAME" vllm_qwen3 vllm_w4a8 vllm_w8a8 2>/dev/null || true
echo "=== serve W8A8 + KV=$KVDTYPE from $IMG (spec=$SPEC graph=$GRAPH) ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} -v "$ROOT:$ROOT" \
  -e ZE_AFFINITY_MASK=0 -e VLLM_LOGGING_LEVEL=INFO "${GRAPH_ENV[@]}" \
  --entrypoint vllm "$IMG" \
  serve "$MODEL" --served-model-name qwen3-14b-w8a8 --host 0.0.0.0 --port ${PORT} \
    --dtype float16 --tensor-parallel-size 1 "${EAGER_ARG[@]}" --max-model-len "$MAXLEN" \
    --kv-cache-dtype "$KVDTYPE" --gpu-memory-utilization "$UTIL" \
    --no-enable-prefix-caching --trust-remote-code "${SPEC_ARGS[@]}"

ok=0
for i in $(seq 1 180); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; break; }
  sleep 5
done
echo; echo "===== startup verdict (spec=$SPEC) ====="
docker logs "$NAME" 2>&1 | grep -iE "Selected XPUInt8|kv.cache.dtype|Speculative|ngram|n_gram|draft|GPU KV cache size|Maximum concurrency|XPU Graph|[Cc]apturing|graph capture|Application startup complete|error|KeyError|not support|Traceback|NotImplemented" \
  | grep -viE "OperatorEntry|registered|VLLM_XPU_ENABLE" | tail -24
echo
[ "$ok" = 1 ] && echo "HEALTHY :$PORT  W8A8 spec=$SPEC" || { echo "NOT HEALTHY; last 30:"; docker logs "$NAME" 2>&1 | tail -30; }
