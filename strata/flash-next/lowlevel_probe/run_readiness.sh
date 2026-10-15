#!/usr/bin/env bash
# run_readiness.sh -- bounded leased launch of build/readiness_probe (ROOT EXECUTES; agents never run GPU modes).
#   ./run_readiness.sh card N CPU [STAGE]   STAGE = smoke (default) | full
#   ./run_readiness.sh quiet                CPU-quiet precheck only (no GPU)
# The binary was AOT-built (bmg-g31) inside strata-ab-build:ocloc and runs INSIDE that image so the SYCL/UR/L0
# libraries match the build. /dev/dri is exposed only inside this leased docker run.
# Env: GPU_RUN, BUILD_IMAGE, EXPECT_PCI0 (0000:0b:00.0), EXPECT_PCI1 (0000:44:00.0), K_USED (10), MAX_LOAD1 (2.0)
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
GPU_RUN=${GPU_RUN:-$D/../../../bin/gpu-run}
BUILD_IMAGE=${BUILD_IMAGE:-strata-ab-build:ocloc}
EXPECT_PCI0=${EXPECT_PCI0:-0000:0b:00.0}
EXPECT_PCI1=${EXPECT_PCI1:-0000:44:00.0}
K_USED=${K_USED:-10}
MAX_LOAD1=${MAX_LOAD1:-2.0}
TS=$(date +%Y%m%d_%H%M%S)
OUT="$D/gpu_results"

quiet_check() {
  local l1; l1=$(cut -d' ' -f1 /proc/loadavg)
  echo "QUIET load1=$l1 (max $MAX_LOAD1) loadavg=\"$(cat /proc/loadavg)\""
  ps -eo psr,pcpu,comm --sort=-pcpu | head -6
  if awk -v a="$l1" -v b="$MAX_LOAD1" 'BEGIN{exit !(a>b)}'; then
    echo "QUIET WARN load1>$MAX_LOAD1: timings DIAGNOSTIC ONLY (correctness still valid)"; return 1
  fi
  echo "QUIET OK"; return 0
}

case "${1:-}" in
  card)
    N=${2:?card N}; CPU=${3:?cpu id}; STAGE=${4:-smoke}
    case "$N" in 0) PCI=$EXPECT_PCI0;; 1) PCI=$EXPECT_PCI1;; *) echo "FAIL card must be 0 or 1"; exit 2;; esac
    case "$CPU" in ''|*[!0-9]*) echo "FAIL cpu must be a single integer"; exit 2;; esac
    [ -x "$D/build/readiness_probe" ] || { echo "FAIL missing build/readiness_probe"; exit 2; }
    (cd "$D" && sha256sum -c <(grep ' build/readiness_probe$' "$D/build/sha256.txt")) || { echo "FAIL binary hash differs from build/sha256.txt"; exit 2; }
    case "$STAGE" in
      smoke) ARGS=(--reps 4 --reps-big 2 --big 64M --k-used "$K_USED");;
      full)  ARGS=(--reps 12 --reps-big 4 --big 64M,256M --k-used "$K_USED");;
      *) echo "FAIL stage smoke|full"; exit 2;;
    esac
    mkdir -p "$OUT"
    LBL="card${N}_cpu${CPU}_${STAGE}_${TS}"
    quiet_check | tee "$OUT/$LBL.quiet.txt" || true
    set -x
    timeout 900 "$GPU_RUN" --card "$N" \
      docker run --rm --network none --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
        --cpuset-cpus "$CPU" -u "$(id -u):$(id -g)" -e HOME=/w/build \
        -e ZE_AFFINITY_MASK="$N" -e ONEAPI_DEVICE_SELECTOR='level_zero:*' \
        -v "$D:/w" -w /w --entrypoint bash "$BUILD_IMAGE" -c \
        "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
         taskset -c $CPU ./build/readiness_probe --expect-pci $PCI --label $LBL --json gpu_results/$LBL.json ${ARGS[*]}" \
      2>&1 | tee "$OUT/$LBL.txt"
    ;;
  quiet) quiet_check ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
