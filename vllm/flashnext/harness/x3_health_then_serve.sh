#!/usr/bin/env bash
# Cached compiled health and serving share one GPU lease.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
AOT=/mnt/vm_8tb/b70/build/vllm-flashnext/c7b-aot-bin
export MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4
echo X3_COMPILED_PRE_HEALTH_BEGIN
PATH="$AOT:$PATH" LD_LIBRARY_PATH="$AOT:${LD_LIBRARY_PATH:-}" \
  timeout --signal=TERM --kill-after=15 180 \
  torchrun --standalone --nproc-per-node=2 "$REPO/bin/xpu-collective-health.py"
echo X3_COMPILED_PRE_HEALTH_PASS
exec "$@"
