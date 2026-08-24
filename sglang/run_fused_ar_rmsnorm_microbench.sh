#!/usr/bin/env bash
# Build and run the TP=2 fused push-AR + residual + Gemma RMSNorm gate.
#
# GPU discipline: invoke this whole script through the shared two-card lease:
#   ./bin/gpu-run bash sglang/run_fused_ar_rmsnorm_microbench.sh
#
# The shared object is a runtime artifact under /mnt/vm_8tb/b70, never repo
# content. This script does not stop/restore a production server; the caller is
# responsible for ensuring the leased cards are otherwise idle.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${IMG:-sglang-xpu:mtp}"
FUSED_SRC="${FUSED_SRC:-$REPO/kernels/xpu_push_ar_fused_rmsnorm.cpp}"
RUNTIME_DIR="${RUNTIME_DIR:-/mnt/vm_8tb/b70/fused_ar_rmsnorm}"
FUSED_SO="${FUSED_SO:-$RUNTIME_DIR/libxpu_push_ar_fused_rmsnorm.so}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNTIME_DIR/results/$STAMP}"
CONTAINER="b70_fused_ar_rmsnorm_microbench"
STRESS_CALLS="${STRESS_CALLS:-256}"
STRESS_CHUNK="${STRESS_CHUNK:-8}"
FAST_MAX_ROWS="${FAST_MAX_ROWS:-2}"
WORKGROUP_SIZE="${WORKGROUP_SIZE:-512}"
ROWS="${ROWS:-1,2}"
FALLBACK_ROWS="${FALLBACK_ROWS-3,11,44,64,128}"

if [ ! -f "$FUSED_SRC" ]; then
  echo "missing candidate source: $FUSED_SRC" >&2
  exit 2
fi
mkdir -p "$RUNTIME_DIR" "$OUT_DIR"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "BUILD -> $FUSED_SRC"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e FAST_MAX_ROWS="$FAST_MAX_ROWS" \
  -e WORKGROUP_SIZE="$WORKGROUP_SIZE" \
  -v "$FUSED_SRC:/src/xpu_push_ar_fused_rmsnorm.cpp:ro" \
  -v "$RUNTIME_DIR:/out" \
  --entrypoint bash "$IMG" -lc \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
   icpx -fsycl -O2 -fPIC -shared \
     -DB70_FUSED_FAST_MAX_ROWS="$FAST_MAX_ROWS" \
     -DB70_FUSED_WORKGROUP_SIZE="$WORKGROUP_SIZE" \
     /src/xpu_push_ar_fused_rmsnorm.cpp \
     -o /out/libxpu_push_ar_fused_rmsnorm.so -lze_loader -lrt'

test -s "$FUSED_SO"
sha256sum "$FUSED_SRC" "$FUSED_SO" | tee "$OUT_DIR/artifacts.sha256"

echo "RUN -> TP=2 correctness/stress/latency (CCL_TOPO_P2P_ACCESS=0)"
docker run --rm --name "$CONTAINER" \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --ipc=host --shm-size 16g \
  -v "$REPO/sglang/fused_ar_rmsnorm_microbench.py:/bench.py:ro" \
  -v "$FUSED_SO:/candidate/libxpu_push_ar_fused_rmsnorm.so:ro" \
  -v "$OUT_DIR:/out" \
  -e CCL_TOPO_P2P_ACCESS=0 \
  -e STRESS_CALLS="$STRESS_CALLS" -e STRESS_CHUNK="$STRESS_CHUNK" \
  -e ROWS="$ROWS" -e FALLBACK_ROWS="$FALLBACK_ROWS" \
  --entrypoint bash "$IMG" -lc \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
   exec python /bench.py \
     --candidate-so /candidate/libxpu_push_ar_fused_rmsnorm.so \
     --rows "$ROWS" --fallback-rows "$FALLBACK_ROWS" --hidden 5120 \
     --stress-calls "$STRESS_CALLS" --stress-chunk "$STRESS_CHUNK" \
     --out /out/result.json' \
  2>&1 | tee "$OUT_DIR/run.log"

echo "RESULT -> $OUT_DIR/result.json"
