#!/usr/bin/env bash
# C13 expert prefill bench. Invoke only via bin/gpu-run --card 0.
set -euo pipefail
export ZE_AFFINITY_MASK=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../../../.." && pwd)"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1}"
PY="${PY:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}"
GGUF_PY="${GGUF_PY:-/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py}"
export PYTHONPATH="$VENV/lib/python3.12/site-packages:$GGUF_PY:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:$ONEAPI/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-/mnt/vm_8tb/b70/build/gguf-xpu-sycl-cache}"
export PYTHONUNBUFFERED=1
export B70_GGUF_SO="${B70_GGUF_SO:-$ROOT/b70_gguf_xpu/_C.c13.so}"
unset B70_PREFILL_SCHED B70_PREFILL_MT B70_PREFILL_RM || true
export B70_PREFILL_SPLIT=0

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
PREFILL_TIMEOUT="${B70_PREFILL_TIMEOUT:-4200}"
CPU_WAIT="${B70_CPU_LOCK_TIMEOUT:-1800}"
set +e
B70_AGENT="${B70_AGENT:-C13}" B70_CPU_LOCK_TIMEOUT="$CPU_WAIT" \
    "$REPO/bin/cpu-run" --quiet \
    timeout -k 30 "$PREFILL_TIMEOUT" "$PY" "$ROOT/tests/c13_bench.py" "$@"
rc=$?
set -e
if [ "$rc" = 124 ]; then
    echo "FAIL timeout rc=124 cpu_wait=${CPU_WAIT}s python_limit=${PREFILL_TIMEOUT}s"
fi
health AFTER
exit "$rc"
