#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]]
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
exec env B70_AGENT=QSA-three-steps-private "$REPO/bin/cpu-run" --quiet "$REPO/bin/gpu-run" --card 0 env ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:* python3 "$HERE/leased_run.py" "$1" "$2"
