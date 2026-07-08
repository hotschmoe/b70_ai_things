#!/usr/bin/env bash
# run_reclaim.sh -- drive leak_matrix_reclaim.py across RECLAIM modes on 2x B70.
# Decisive test: does a graph-replay command-list reclaim (re-instantiate / stream-rotate) RESET the
# per-replay inst-rate decay? Each mode in its own container. Wrap in gpu-run (locks both cards).
set -u
REPO="/mnt/vm_8tb/github/b70_ai_things"
IMG="${IMG:-vllm-xpu-env:int8g-v0240}"
PUSH_AR_DIR="$REPO/vllm/contrib/vllm_push_allreduce"
COLL="${COLL:-pushar}"
REPLAYS="${REPLAYS:-30000}"
EVERY="${EVERY:-2000}"
MODES="${MODES:-none reinst rotate}"
TIMEOUT="${TIMEOUT:-180}"
LOG="$REPO/results/logs/reclaim_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/results/logs"
echo "=== reclaim test COLL=$COLL MODES='$MODES' EVERY=$EVERY REPLAYS=$REPLAYS ==="
port=29760
for rc in $MODES; do
  port=$((port+1)); name="reclaim_${rc}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  echo; echo "######## RECLAIM=$rc ########"
  timeout "$TIMEOUT" docker run --rm --name "$name" --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 32g \
    -v "$PUSH_AR_DIR:/opt/push_ar:ro" -v "$REPO/vllm/nvfp4:/opt/leak:ro" \
    -e CCL_ENABLE_SYCL_KERNELS=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi \
    -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
    -e PYTHONFAULTHANDLER=1 -e PYTHONUNBUFFERED=1 \
    -e COLL="$COLL" -e RECLAIM="$rc" -e EVERY="$EVERY" -e REPLAYS="$REPLAYS" -e MASTER_PORT="$port" \
    -e PUSH_SO=/opt/push_ar/prebuilt/libxpu_push_ar_graph.so \
    --entrypoint python3 "$IMG" /opt/leak/leak_matrix_reclaim.py 2>&1 | tee "${LOG}_${rc}.log"
  docker rm -f "$name" >/dev/null 2>&1 || true
done
echo; echo "=== logs: ${LOG}_*.log ==="
echo "READ the inst/first column: if it drops toward 0 then JUMPS back to ~1.0 right after each"
echo "'reclaim applied', that reclaim RESETS the accumulation."
