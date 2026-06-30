#!/usr/bin/env bash
# zml/run_w8a8_sweep_gpu.sh -- W8A8 GEMM/GEMV sweep on ONE B70 (ZML_W8A8.md follow-ups: push int8
# toward the 2x peak, profile the M=1 decode GEMV). SINGLE CARD only (level_zero:0) -- pure GEMM,
# no collectives, so it does NOT touch the TP=2 wedge path. Daily driver must be DOWN; run under the
# gpu-run lease:
#   cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run --card 0 bash zml/run_w8a8_sweep_gpu.sh
#
# Phases:
#   1. full timing sweep (all real qwen3.6 proj shapes x M=1..4096 x {bf16,i8,w8a8,woq}) -> table.
#   2. ONEDNN_VERBOSE + XLA HLO dump for q_proj at M=1 (GEMV) and M=512 (GEMM), i8 + w8a8 -- to
#      confirm the s8/s8/s32 jit:gemm:xe kernel and catch a per-call weight reorder or a standalone
#      slow quantize-reduce (see b70-int8-xmx-roofline memory).
set -uo pipefail
ZML="${ZML:-/mnt/vm_8tb/b70/zml}"
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
BAZELISK="${BAZELISK:-$HOME/.local/bin/bazelisk}"
LOGDIR="${LOGDIR:-/mnt/vm_8tb/b70/w8a8_sweep_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ARGS="${SWEEP_ARGS:---shape=all --m=0 --variant=all --iters=50}"
mkdir -p "$LOGDIR"

echo "=== pre-flight xpu-health ===" && "$REPO/bin/xpu-health" 2>&1 | tail -2
cd "$ZML"
# ONE card. level_zero:0 == card 0 (the non-display card; see ZML_W8A8 M4 gotcha).
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-0}"   # override oneapi.zig:33 garbage default

# Build once up front (compile errors here, not mid-lease).
"$BAZELISK" build //examples/w8a8_sweep --config=release \
  --@zml//platforms:cpu=false --@zml//platforms:oneapi=true 2>&1 | tail -3

echo "=== PHASE 1: full timing sweep -> $LOGDIR/sweep.log  $(date) ==="
# shellcheck disable=SC2086
"$BAZELISK" run //examples/w8a8_sweep --config=release \
  --@zml//platforms:cpu=false --@zml//platforms:oneapi=true \
  -- $SWEEP_ARGS 2>&1 | tee "$LOGDIR/sweep.log"

echo "=== PHASE 2: ONEDNN_VERBOSE + XLA dump (q_proj M=1 GEMV and M=512 GEMM) ==="
for M in 1 512; do
  for V in i8 w8a8; do
    echo "--- q_proj M=$M variant=$V ---"
    ONEDNN_VERBOSE=dispatch,profile_exec \
    XLA_FLAGS="--xla_dump_to=$LOGDIR/hlo_${V}_m${M} --xla_dump_hlo_as_text" \
    "$BAZELISK" run //examples/w8a8_sweep --config=release \
      --@zml//platforms:cpu=false --@zml//platforms:oneapi=true \
      -- --shape=q_proj --m=$M --variant=$V --iters=5 \
      > "$LOGDIR/verbose_${V}_m${M}.log" 2>&1
    echo "  onednn matmul lines:"
    grep -iE "onednn_verbose.*matmul|gemm|reorder" "$LOGDIR/verbose_${V}_m${M}.log" | head -6
  done
done

# Release the bazel daemon's inherited gpu-run flock fds BEFORE returning (else ~3h lease hold).
"$BAZELISK" shutdown >/dev/null 2>&1 || true
echo "=== sweep done; logs in $LOGDIR ; post-run xpu-health ==="
"$REPO/bin/xpu-health" 2>&1 | tail -2 || echo "[!] box may be wedged -- bin/xe-reset"
echo "Summary table:"; grep -E "shape|q_proj|gate_proj|down_proj|k_proj|o_proj|up_proj|v_proj|sq" "$LOGDIR/sweep.log" 2>/dev/null | tail -60