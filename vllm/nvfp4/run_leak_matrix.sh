#!/usr/bin/env bash
# run_leak_matrix.sh -- drive vllm/nvfp4/leak_matrix.py across collective transports on 2x B70.
# Each COLL mode runs in its OWN throwaway container (a linear_stream SIGABRT in one mode must not
# kill the others). TP env mirrors serve_nvfp4_27b.sh exactly (CCL_ENABLE_SYCL_KERNELS=1 etc).
# Wrap in gpu-run (locks both cards). Usage: ./bin/gpu-run bash vllm/nvfp4/run_leak_matrix.sh
set -u
REPO="/mnt/vm_8tb/github/b70_ai_things"
IMG="${IMG:-vllm-xpu-env:int8g-v0240}"
PUSH_AR_DIR="$REPO/vllm/contrib/vllm_push_allreduce"
REPLAYS="${REPLAYS:-500000}"
MODES="${MODES:-pushar oneccl_ar block3_oneccl block3_pushar oneccl_ag}"
TIMEOUT="${TIMEOUT:-300}"
LOG="$REPO/results/logs/leak_matrix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/results/logs"
echo "=== leak_matrix: MODES='$MODES' REPLAYS=$REPLAYS IMG=$IMG ==="
port=29700
declare -A VERDICT
for coll in $MODES; do
  port=$((port+1))
  name="leakmx_${coll}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  echo; echo "######## COLL=$coll (port $port) ########"
  timeout "$TIMEOUT" docker run --rm --name "$name" --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 32g \
    -v "$PUSH_AR_DIR:/opt/push_ar:ro" -v "$REPO/vllm/nvfp4:/opt/leak:ro" \
    -e CCL_ENABLE_SYCL_KERNELS=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi \
    -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
    -e PYTHONFAULTHANDLER=1 -e PYTHONUNBUFFERED=1 \
    -e COLL="$coll" -e REPLAYS="$REPLAYS" -e MASTER_PORT="$port" \
    -e PUSH_SO=/opt/push_ar/prebuilt/libxpu_push_ar_graph.so \
    --entrypoint python3 "$IMG" /opt/leak/leak_matrix.py 2>&1 | tee "${LOG}_${coll}.log"
  rc=${PIPESTATUS[0]}
  if grep -q "PASS (both ranks clean)" "${LOG}_${coll}.log"; then
    VERDICT[$coll]="CLEAN (survived $REPLAYS)"
  elif grep -q "linear_stream" "${LOG}_${coll}.log"; then
    n=$(grep -oE "replay [0-9]+/" "${LOG}_${coll}.log" | tail -1 | grep -oE "[0-9]+")
    VERDICT[$coll]="LEAK (linear_stream abort near replay ${n:-?})"
  elif [ "$rc" = 124 ]; then
    VERDICT[$coll]="TIMEOUT (${TIMEOUT}s)"
  else
    VERDICT[$coll]="OTHER (rc=$rc, see log)"
  fi
  echo "---- $coll -> ${VERDICT[$coll]} ----"
  docker rm -f "$name" >/dev/null 2>&1 || true
done
echo; echo "================= LEAK MATRIX SUMMARY ================="
for coll in $MODES; do printf "  %-16s %s\n" "$coll" "${VERDICT[$coll]}"; done
echo "logs: ${LOG}_*.log"
