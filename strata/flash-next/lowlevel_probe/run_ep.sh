#!/usr/bin/env bash
# Standalone EP diagnostics, pair lease. Correctness failures remain unresolved (S27).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$D/../../.." && pwd)
MODE=${1:-flag}
POISON=${2:-producer}
HOST_SYNC=${3:-0}
REPS=${4:-20}
WAIT=${5:-native}
EXTRA_READS=${6:-0}
case "$MODE" in copy|flag) ;; *) exit 2;; esac
case "$POISON" in owner|producer) ;; *) exit 2;; esac
case "$HOST_SYNC" in 0|1) ;; *) exit 2;; esac
case "$WAIT" in native|debug) ;; *) exit 2;; esac
case "$EXTRA_READS" in 0|2) ;; *) exit 2;; esac
case "$REPS" in ''|*[!0-9]*) exit 2;; esac
[ "$REPS" -ge 1 ] && [ "$REPS" -le 64 ]
(cd "$D/build/ep-ready" && sha256sum -c <(grep ' ep_ready_probe$' sha256.txt))
mkdir -p "$D/gpu_results" "$D/build/ep-ready/cache"
LBL="ep_${MODE}_${POISON}_sync${HOST_SYNC}_${WAIT}_reads${EXTRA_READS}_$(date +%Y%m%d_%H%M%S)"
timeout 120 "$REPO/bin/gpu-run" docker run --rm --network none \
  --device /dev/dri --cap-add SYS_PTRACE --group-add "$(getent group render | cut -d: -f3)" \
  -u "$(id -u):$(id -g)" -e ZE_AFFINITY_MASK=0,1 -e ONEAPI_DEVICE_SELECTOR='level_zero:*' \
  -e SYCL_CACHE_PERSISTENT=0 -e XDG_CACHE_HOME=/w/build/ep-ready/cache \
  -v "$D:/w" -w /w --entrypoint bash strata-ab-build:ocloc -c \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec ./build/ep-ready/ep_ready_probe --mode $MODE --poison $POISON --host-sync $HOST_SYNC --reps $REPS --wait $WAIT --extra-reads $EXTRA_READS --max-fail 1 --spin 1048576 --timeout-ms 5000" \
  2>&1 | tee "$D/gpu_results/$LBL.txt"
