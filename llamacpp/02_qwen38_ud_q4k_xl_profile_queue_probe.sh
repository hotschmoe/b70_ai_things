#!/usr/bin/env bash
# Isolate whether the B70 TP=2 failure is the profiling-enabled queue itself.
# Run under ./bin/gpu-run. No timing barriers can be submitted in this probe.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/qwen38_ud_q4k_xl_profile_queue_$STAMP}"
NAME="${NAME:-qwen38_xl_profile_queue_probe}"
PORT="${PORT:-18080}"
IMG="${IMG:-qwen38-b70:quant-timing}"
IMG_ID="${IMG_ID:-sha256:5029a9d394eacd46b48686b564fcc93a410c27a6b1064630008eaec83ef748d1}"
SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-ud-q4-k-xl/serve.sh"
OVERLAY="$REPO/llamacpp/qwen38_ud_q4k_xl_profile_entrypoint.sh"

[ ! -e "$OUT" ] || { echo "refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    docker inspect "$NAME" >"$OUT/container_inspect.json" 2>/dev/null || true
    docker logs "$NAME" >"$OUT/server.log" 2>&1 || true
    docker stop --time 30 "$NAME" >"$OUT/docker_stop.log" 2>&1 || true
    docker rm -f "$NAME" >"$OUT/docker_rm.log" 2>&1 || true
    env -u IMG "$REPO/bin/xpu-health" >"$OUT/health_post.log" 2>&1 || rc=1
    if ss -ltnH "sport = :$PORT" | rg -q .; then
        echo "endpoint still listening on $PORT" >&2
        rc=1
    fi
    echo "exit=$rc endpoint=down artifacts=$OUT"
    exit "$rc"
}
trap cleanup EXIT INT TERM

if ss -ltnH "sport = :$PORT" | rg -q .; then
    echo "port $PORT occupied; refusing to stop unrelated endpoint" >&2
    exit 2
fi
env -u IMG "$REPO/bin/xpu-health" >"$OUT/health_pre.log" 2>&1

# UINT64_MAX makes every callback return in the skip branch. The queue still has
# enable_profiling, but no start/end barrier and no event timestamp read can occur.
set +e
NAME="$NAME" PORT="$PORT" IMG="$IMG" IMG_ID="$IMG_ID" OVERLAY="$OVERLAY" \
    ENABLE_MTP=0 LAB_DOORS=0 RESTART_POLICY=no \
    UR_L0_USE_DRIVER_INORDER_LISTS=0 \
    GGML_SYCL_QUANT_CENSUS=1 GGML_SYCL_QUANT_TIMING_SAMPLE=64 \
    GGML_SYCL_QUANT_TIMING_SKIP=18446744073709551615 \
    bash "$SHELF" start 2>&1 | tee "$OUT/start.log"
start_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$start_rc" >"$OUT/start.rc"

if [ "$start_rc" -ne 0 ]; then
    echo "PROFILE_QUEUE_PROBE FAIL start_rc=$start_rc"
    exit 1
fi
echo "PROFILE_QUEUE_PROBE PASS queue_profiling=1 barriers=0 driver_inorder_lists=0"
