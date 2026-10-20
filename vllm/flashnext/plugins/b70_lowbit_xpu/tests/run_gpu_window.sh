#!/usr/bin/env bash
# One xpu-run container: fused-launch bench, then the b70lb1 regression.
# Launch-count 4 is the source default. A timeout (124) means the device
# may be wedged, so the regression bench is skipped.
set -uo pipefail

PLUGIN=/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_lowbit_xpu
cd "$PLUGIN"
unset B70_LOWBIT_MOE_LAUNCHES || true
echo "gpu_window so=$(stat -c %s "${B70_LOWBIT_SO:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so}")"
bash tests/run_launch_bench.sh
lrc=$?
echo "WINDOW_LAUNCH_RC:$lrc"
if [[ "$lrc" -eq 124 ]]; then
  echo "WINDOW_SKIP_BENCH: launch timed out"
  exit 124
fi
unset B70_LOWBIT_MOE_LAUNCHES || true
bash tests/run_xpu.sh
brc=$?
echo "WINDOW_BENCH_RC:$brc"
if [[ "$lrc" -ne 0 ]]; then
  exit "$lrc"
fi
exit "$brc"
