#!/usr/bin/env bash
# Prefill correctness and bench. Invoke only via: bin/gpu-run --card 0 scripts/prefill_gpu.sh
set -euo pipefail
export ZE_AFFINITY_MASK=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../../../.." && pwd)"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1}"
PY="${PY:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}"
export PYTHONPATH="$VENV/lib/python3.12/site-packages:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:$ONEAPI/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-/mnt/vm_8tb/b70/build/gguf-xpu-sycl-cache}"
# Side build so a concurrent C1 restore of _C.so cannot drop the prefill ops.
export B70_GGUF_SO="${B70_GGUF_SO:-$ROOT/b70_gguf_xpu/_C.full.so}"

health() {
    local tag="$1"
    set +e
    "$REPO/bin/xpu-health" --card 0
    local rc=$?
    set -e
    echo "HEALTH_${tag}:${rc}"
    if [ "$rc" = 1 ]; then
        echo "card 0 wedged; not continuing" >&2
        exit 1
    fi
}

health BEFORE
# Kill a hung launch. Health still runs. -k follows TERM with KILL.
PREFILL_TIMEOUT="${B70_PREFILL_TIMEOUT:-2400}"
set +e
timeout -k 30 "$PREFILL_TIMEOUT" "$PY" "$ROOT/tests/prefill_check.py" "$@"
rc=$?
set -e
if [ "$rc" = 124 ]; then
    echo "FAIL python timed out after ${PREFILL_TIMEOUT}s"
fi
health AFTER
exit "$rc"
