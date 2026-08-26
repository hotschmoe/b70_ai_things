#!/usr/bin/env bash
# Exact June-source TP communicator oracle for push-AR preinitialization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMG="intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94"
JUNE_SOURCE=/mnt/vm_8tb/b70/steve-s2b/vllm-e190
JUNE_RUNTIME=/mnt/vm_8tb/b70/steve-repro/june122-xpuc-regular-20260826/runtime-candidate
PUSH_DIR="$REPO_ROOT/vllm/contrib/vllm_push_allreduce"
PUSH_SO="${PUSH_AR_SO_HOST:-$PUSH_DIR/prebuilt/libxpu_push_ar_graph.so}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/logs/qwen36_push_ar_init_oracle_$STAMP}"
ORACLE_TIMEOUT="${ORACLE_TIMEOUT:-60}"
EXPECT_LOADED_GRAPH_STALL="${EXPECT_LOADED_GRAPH_STALL:-1}"
CONTAINER_NAME="qwen36-push-ar-init-oracle-$STAMP"
mkdir -p "$RESULT_DIR"

case "$ORACLE_TIMEOUT" in
  ''|*[!0-9]*) echo "ORACLE_TIMEOUT must be a positive integer" >&2; exit 2 ;;
esac
[ "$ORACLE_TIMEOUT" -gt 0 ] || {
  echo "ORACLE_TIMEOUT must be a positive integer" >&2
  exit 2
}
case "$EXPECT_LOADED_GRAPH_STALL" in
  0|1) ;;
  *) echo "EXPECT_LOADED_GRAPH_STALL must be 0 or 1" >&2; exit 2 ;;
esac

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "config -> exact June source e190923b, June122 native runtime, TP=2, push-AR preinit strict, 5120 bf16 elements"
echo "config -> push_so=$PUSH_SO sha256=$(sha256sum "$PUSH_SO" | awk '{print $1}')"
echo "config -> timeout=${ORACLE_TIMEOUT}s expect_loaded_graph_stall=$EXPECT_LOADED_GRAPH_STALL"
set +e
timeout --signal=TERM --kill-after=10s "${ORACLE_TIMEOUT}s" \
docker run --rm --name "$CONTAINER_NAME" \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g \
  -e VLLM_USE_V1=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e XPU_GRAPH=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  -e ZE_AFFINITY_MASK=0,1 \
  -e MASTER_ADDR=127.0.0.1 \
  -e MASTER_PORT=29671 \
  -e LOCAL_WORLD_SIZE=2 \
  -e PUSH_AR_SO=/opt/push_ar_runtime/libxpu_push_ar_graph.so \
  -e PUSH_AR_DISABLE=0 \
  -e PUSH_AR_GRAPH=1 \
  -e PUSH_AR_MIN_NUMEL=0 \
  -e PUSH_AR_MAXB=67108864 \
  -e PUSH_AR_PREINIT=1 \
  -e PUSH_AR_PREINIT_STRICT=1 \
  -e PUSH_AR_GRAPH_INPLACE=1 \
  -e PYTHONPATH=/opt/push_ar:/opt/forensic_vllm:/opt/june-runtime \
  -v "$PUSH_DIR:/opt/push_ar:ro" \
  -v "$PUSH_SO:/opt/push_ar_runtime/libxpu_push_ar_graph.so:ro" \
  -v "$JUNE_SOURCE:/opt/forensic_vllm:ro" \
  -v "$JUNE_RUNTIME:/opt/june-runtime:ro" \
  -v "$SCRIPT_DIR/qwen36_push_ar_init_oracle.py:/opt/qwen36_push_ar_init_oracle.py:ro" \
  --entrypoint python "$IMG" /opt/qwen36_push_ar_init_oracle.py \
  2>&1 | tee "$RESULT_DIR/oracle.log"
oracle_rc="${PIPESTATUS[0]}"
set -e

if [ "$oracle_rc" = 0 ]; then
  rg -q 'PUSH_AR_INIT_ORACLE_PASS' "$RESULT_DIR/oracle.log"
  echo "result -> TP communicator preinit and 64 graph replays passed"
  echo "verdict -> loaded June vLLM/XCCL graph context accepts push-AR"
  exit 0
fi

if { [ "$oracle_rc" = 124 ] || [ "$oracle_rc" = 137 ]; } && \
   [ "$EXPECT_LOADED_GRAPH_STALL" = 1 ]; then
  rg -Fq '[push_ar] PREINIT group=tp:0 rank=0 ready=1' "$RESULT_DIR/oracle.log"
  rg -Fq '[push_ar] PREINIT group=tp:0 rank=1 ready=1' "$RESULT_DIR/oracle.log"
  rg -Fq '[oracle r0] capturing=True' "$RESULT_DIR/oracle.log"
  rg -Fq '[oracle r1] capturing=True' "$RESULT_DIR/oracle.log"
  ! rg -q 'PUSH_AR_INIT_ORACLE_PASS|graph capture complete' "$RESULT_DIR/oracle.log"
  echo "result -> both TP ranks imported push-AR IPC before graph capture; native graph submission stalled until timeout"
  echo "verdict -> preinit repairs asymmetric IPC import, but loaded June vLLM/XCCL graph-context compatibility remains blocked"
  exit 0
fi

echo "push-AR loaded-context oracle failed with rc=$oracle_rc" >&2
exit "$oracle_rc"
