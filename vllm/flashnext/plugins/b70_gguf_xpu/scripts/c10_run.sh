#!/usr/bin/env bash
# C10 dequant-then-GEMM bench. Invoke only via bin/gpu-run --card 0.
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
export PYTHONUNBUFFERED=1
export B70_GGUF_SO="${B70_GGUF_SO:-$ROOT/b70_gguf_xpu/_C.c10.so}"
# The bench calls fused ops and dq ops directly. Keep the wrapper on fused.
export B70_PREFILL_DQ_M="${B70_PREFILL_DQ_M:-off}"
export B70_PREFILL_DQ_T="${B70_PREFILL_DQ_T:-off}"
unset B70_PREFILL_SCHED B70_PREFILL_MT B70_PREFILL_RM B70_PREFILL_SPLIT || true

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
# Quiet CPU only for the timed window, and only after this script's caller
# holds the card lease. Taking the CPU lock first deadlocks a card holder
# that nests cpu-run --quiet inside gpu-run (C9 on card 0).
PREFILL_TIMEOUT="${B70_PREFILL_TIMEOUT:-3000}"
set +e
B70_AGENT="${B70_AGENT:-C10}" B70_CPU_LOCK_TIMEOUT="${B70_CPU_LOCK_TIMEOUT:-0}" \
    "$REPO/bin/cpu-run" --quiet \
    timeout -k 30 "$PREFILL_TIMEOUT" "$PY" "$ROOT/tests/c10_dq_bench.py" "$@"
rc=$?
set -e
if [ "$rc" = 124 ]; then
    echo "FAIL python timed out after ${PREFILL_TIMEOUT}s"
fi
health AFTER
exit "$rc"
