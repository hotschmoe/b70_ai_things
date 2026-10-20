#!/usr/bin/env bash
# Inner command for harness/xpu-run.sh --card 0.
# The entrypoint already set ZE_AFFINITY_MASK, /host-umd, and the venv.
# Do not replace LD_LIBRARY_PATH.
set -uo pipefail

PLUGIN=/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_lowbit_xpu
export PYTHONPATH="${PLUGIN}${PYTHONPATH:+:${PYTHONPATH}}"
export B70_LOWBIT_SO="${B70_LOWBIT_SO:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so}"
export B70_LOWBIT_PORT="${B70_LOWBIT_PORT:-29671}"
cd "$PLUGIN"
python3 -u tests/bench_lowbit.py
exit $?
