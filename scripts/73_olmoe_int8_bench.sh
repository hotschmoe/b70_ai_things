#!/usr/bin/env bash
# Serve OLMoE-1B-7B on intel/llm-scaler-vllm (which HAS the int8 MoE kernel, docs/kernel/20) and bench.
# QUANT=experts_int8 (runtime int8 MoE weights, no offline quant -- fast "does it serve" signal) by default;
# pass QUANT=compressed-tensors with a W8A8 ckpt for the true int8-activation path. Route via gpu-run.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
NAME=vllm_olmoe_int8
IMG="${IMG:-intel/llm-scaler-vllm:0.14.0-b8.3.1}"
MDIR="${MDIR:-OLMoE-1B-7B-0924-Instruct}"
QUANT="${QUANT:-experts_int8}"
TP="${TP:-1}"
docker rm -f "$NAME" 2>/dev/null || true
echo "=== serve $MDIR quant=$QUANT TP=$TP on $IMG ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -p 8000:8000 --ipc=host --shm-size 16g -v "$ROOT:$ROOT" -e ZE_AFFINITY_MASK="${MASK:-0}" \
  "$IMG" vllm serve "$ROOT/models/$MDIR" --host 0.0.0.0 --port 18080 --trust-remote-code \
  --served-model-name olmoe-int8 --dtype auto --quantization "$QUANT" --tensor-parallel-size "$TP" \
  --max-model-len "${MAXLEN:-4096}" --max-num-seqs "${MAXSEQS:-8}" --gpu-memory-utilization "${UTIL:-0.90}" \
  --no-enable-prefix-caching --compilation-config '{"cudagraph_mode":"PIECEWISE"}' >/dev/null 2>&1
ok=0
for i in $(seq 1 144); do
  curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "container died early"; break; }
  sleep 5
done
if [ "$ok" = 1 ]; then
  echo "=== HEALTHY -- int8 MoE serves! probe generation ==="
  SID=$(curl -s --max-time 8 http://localhost:8000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)[\"data\"][0][\"id\"])" 2>/dev/null)
  [ -z "$SID" ] && SID="$ROOT/models/$MDIR"; echo "served model id = $SID"
  curl -s --max-time 20 http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"$SID\",\"prompt\":\"The capital of France is\",\"max_tokens\":12,\"temperature\":0}" | head -c 600; echo
  echo "=== sweep (port 8000) ==="
  env NAME="$NAME" MODEL="$SID" LABEL="olmoe-${QUANT}" TOKPATH="/models/$MDIR" PORT=8000 \
    IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 2 4 8}" bash "$ROOT/35_sweep_bench.sh" || true
else
  echo "=== NOT HEALTHY -- the int8 MoE serve gap. logs: ==="
  docker logs "$NAME" 2>&1 | grep -aiE "error|assert|not support|unsupport|quant|moe|expert|int8|Traceback|raise|fail|kernel" | tail -30
fi
docker stop "$NAME" 2>/dev/null || true
echo "=== olmoe int8 bench done (quant=$QUANT) ==="
