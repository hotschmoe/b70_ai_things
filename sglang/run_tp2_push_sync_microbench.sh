#!/usr/bin/env bash
# Run the C2 collective-sync microbench under one both-card gpu-run lease.
# Stops and restores the exact stock Q4_K_M production shelf.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROD="$REPO/rdy_to_serve/llamacpp/qwen38-27b-q4km/serve.sh"
PUSHDIR="$REPO/vllm/contrib/vllm_push_allreduce/prebuilt"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$REPO/results/logs/tp2_push_sync_$STAMP}"
MODES="${MODES:-oneccl current async_safe}"
PROD_NAME="qwen38_stock_q4km_tp2"
RESTORE=0
ACTIVE=""
mkdir -p "$OUT"

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [ -z "$ACTIVE" ] || docker rm -f "$ACTIVE" >/dev/null 2>&1 || true
  if [ "$RESTORE" = 1 ]; then
    if "$REPO/bin/xpu-health" >"$OUT/health_before_restore.log" 2>&1; then
      bash "$PROD" start >"$OUT/production_restore.log" 2>&1 || rc=1
    else
      echo "cards unhealthy; production not restored" | tee "$OUT/restore_refused.txt"
      rc=1
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' | rg -qx "$PROD_NAME"; then
  curl -fsS http://127.0.0.1:18080/v1/models >"$OUT/production_models_before.json"
  RESTORE=1
  bash "$PROD" stop
fi
"$REPO/bin/xpu-health" | tee "$OUT/health_start.log"

for label in $MODES; do
  mode="$label"
  env_args=()
  if [ "$label" = async_safe ]; then
    mode=async
    env_args=(-e ASYNC_HOSTWAIT_INPUT=1)
  elif [ "$label" = async_native ]; then
    mode=async
  fi
  ACTIVE="tp2_push_sync_${label}"
  echo "RUN -> $label"
  timeout --signal=TERM --kill-after=10s 300s docker run --rm --name "$ACTIVE" \
    --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --ipc=host --shm-size 16g \
    -v "$REPO/sglang/tp2_push_sync_microbench.py:/bench.py:ro" \
    -v "$PUSHDIR:/push:ro" \
    -v "$OUT:/out" \
    -e CCL_TOPO_P2P_ACCESS=0 "${env_args[@]}" \
    --entrypoint bash sglang-xpu:mtp -lc \
    "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; python /bench.py --mode $mode --out /out/$label.json" \
    2>&1 | tee "$OUT/$label.log"
  ACTIVE=""
  "$REPO/bin/xpu-health" | tee "$OUT/health_${label}.log"
done
