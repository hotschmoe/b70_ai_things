#!/usr/bin/env bash
# Serve nameistoken Qwen3.6-35B-A3B Quark W8A8 INT8 on llm-scaler (Steve's proven recipe, adapted TP=2) + bench.
# The REAL target: int8 MoE 35B on our 2x B70. Route via gpu-run.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
NAME=vllm_quark35b
IMG="${IMG:-intel/llm-scaler-vllm:0.14.0-b8.3.1}"
CKPT="${CKPT:-$ROOT/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8}"
TP="${TP:-2}"
docker rm -f "$NAME" 2>/dev/null || true
echo "=== serve Quark-W8A8 35B TP=$TP on $IMG ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -p 8000:8000 --ipc=host --shm-size 32g -v "$ROOT:$ROOT" \
  -e CCL_ENABLE_SYCL_KERNELS="${SYCLKERNELS:-1}" -e SYCL_UR_USE_LEVEL_ZERO_V2=0 --entrypoint vllm \
  "$IMG" serve "$CKPT" --host 0.0.0.0 --port 8000 --trust-remote-code \
  --served-model-name qwen36-35b-quark --dtype auto --quantization quark \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --max-model-len "${MAXLEN:-8192}" --max-num-batched-tokens 8192 --max-num-seqs "${MAXSEQS:-8}" \
  --gpu-memory-utilization "${UTIL:-0.95}" --kv-cache-dtype auto --no-enable-prefix-caching \
  --language-model-only --compilation-config '{"cudagraph_mode":"PIECEWISE"}' --generation-config vllm >/dev/null 2>&1
ok=0
for i in $(seq 1 180); do
  curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "container died"; break; }
  sleep 5
done
if [ "$ok" = 1 ]; then
  SID=$(curl -s --max-time 8 http://localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  echo "=== HEALTHY 35B int8 MoE! served id=$SID -- gen probe ==="
  curl -s --max-time 25 http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"$SID\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" | head -c 500; echo
  echo "=== sweep ctx2048 ==="
  env NAME="$NAME" MODEL="$SID" LABEL="qwen36-35b-quark-w8a8" TOKPATH="$CKPT" PORT=8000 \
    IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 2 4 8}" bash "$ROOT/35_sweep_bench.sh" || true
else
  echo "=== NOT HEALTHY -- 35B int8 serve gap ==="; docker logs "$NAME" 2>&1 | tail -40
fi
docker stop "$NAME" 2>/dev/null || true
echo "=== quark35b bench done ==="
