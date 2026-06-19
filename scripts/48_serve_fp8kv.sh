#!/usr/bin/env bash
# Serve our W8A8 (14B) from the committed vllm-xpu-env:int8 image WITH FP8 KV cache, to prove the
# long-context win (KV token budget should ~2x vs the ~71k we saw with auto/fp16 KV). Plain `vllm serve`
# from :int8 -- no graft/patch. Reports the KV cache size + kernel selection.
#   Env: KVDTYPE (default fp8_e4m3), MAXLEN (default 8192), UTIL (0.90).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG=vllm-xpu-env:int8
MODEL="$ROOT/models/Qwen3-14B-W8A8-gptq"
NAME=vllm_int8; PORT=18080
KVDTYPE="${KVDTYPE:-fp8_e4m3}"; MAXLEN="${MAXLEN:-8192}"; UTIL="${UTIL:-0.90}"

docker rm -f "$NAME" vllm_qwen3 vllm_w4a8 vllm_w8a8 2>/dev/null || true
echo "=== serve W8A8 + KV=$KVDTYPE from $IMG (no graft/patch) ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} -v "$ROOT:$ROOT" \
  -e ZE_AFFINITY_MASK=0 -e VLLM_LOGGING_LEVEL=DEBUG \
  --entrypoint vllm "$IMG" \
  serve "$MODEL" --served-model-name qwen3-14b-w8a8-gptq --host 0.0.0.0 --port ${PORT} \
    --dtype float16 --tensor-parallel-size 1 --enforce-eager --max-model-len "$MAXLEN" \
    --kv-cache-dtype "$KVDTYPE" --gpu-memory-utilization "$UTIL" \
    --no-enable-prefix-caching --trust-remote-code

ok=0
for i in $(seq 1 144); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; break; }
  sleep 5
done
echo; echo "===== KV / kernel verdict (KV=$KVDTYPE) ====="
docker logs "$NAME" 2>&1 | grep -iE "Selected XPUInt8|kv.cache.dtype|fp8|GPU KV cache size|Available KV cache|Maximum concurrency|Application startup complete|error|KeyError|not support|Traceback" \
  | grep -viE "OperatorEntry|registered|VLLM_" | tail -16
echo
[ "$ok" = 1 ] && echo "HEALTHY :$PORT  W8A8 + KV=$KVDTYPE  (compare KV tokens vs ~71,040 baseline w/ auto KV)" || { echo "NOT HEALTHY (KV=$KVDTYPE may be unsupported); last 20:"; docker logs "$NAME" 2>&1 | tail -20; }
