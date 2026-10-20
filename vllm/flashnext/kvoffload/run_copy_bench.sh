#!/usr/bin/env bash
# W8: time b70_kvpark.copy_batch against swap_blocks_batch.
# xpu-run.sh is the gpu-run holder. This script does not wrap it again.
# cpu-run --quiet covers only the timed copies.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CARD="${CARD:-0}"
LOGDIR=/mnt/vm_8tb/b70/build/vllm-flashnext/logs
PLUGIN="$REPO/vllm/flashnext/plugins/b70_kvpark"
FORK=/mnt/vm_8tb/b70/build/vllm-flashnext/vllm
SITE=/mnt/vm_8tb/b70/build/vllm-flashnext/venv/lib/python3.12/site-packages
BENCH="$PLUGIN/tests/bench_copy.py"
NAME="vllm-fn-g4-copy-c${CARD}"
# Default is the 08:41 library. The launcher points this at the ordered
# build once that file exists. The bench prints the path it loaded.
SO="${B70_KVPARK_SO:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_kvpark/libb70_kvpark.so}"

export B70_AGENT=g4
mkdir -p "$LOGDIR"

echo "COPY health before $(date '+%F %T') card=$CARD"
B70_GPU_LOCK_TIMEOUT=90 "$REPO/bin/gpu-run" --card "$CARD" \
  "$REPO/bin/xpu-health" --card "$CARD" \
  > "$LOGDIR/g4-copy-health-before-c${CARD}.txt" 2>&1
echo "COPY health before ok"

echo "COPY bench start $(date '+%F %T') card=$CARD"
set +e
B70_GPU_LOCK_TIMEOUT=120 B70_CPU_LOCK_TIMEOUT=90 \
  "$REPO/bin/cpu-run" --quiet \
  "$REPO/vllm/flashnext/harness/xpu-run.sh" \
  --card "$CARD" --memory 16g --shm-size 4g --name "$NAME" \
  --env "PYTHONPATH=${PLUGIN}:${FORK}:${SITE}" \
  --env "B70_KVPARK_SO=${SO}" \
  -- python3 "$BENCH"
rc=$?
set -e
echo "COPY bench rc=$rc $(date '+%F %T')"

echo "COPY health after $(date '+%F %T') card=$CARD"
B70_GPU_LOCK_TIMEOUT=90 "$REPO/bin/gpu-run" --card "$CARD" \
  "$REPO/bin/xpu-health" --card "$CARD" \
  > "$LOGDIR/g4-copy-health-after-c${CARD}.txt" 2>&1
echo "COPY health after rc=$? $(date '+%F %T')"
exit "$rc"
