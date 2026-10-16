#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 ]]
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
exec env B70_AGENT=S146-native-tags "$REPO/bin/cpu-run" --quiet "$REPO/bin/gpu-run" --card 0 env ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:* python3 "$HERE/leased_run.py" "$@"
