#!/usr/bin/env bash
# O4: compile+run M=1 1D NVFP4 GEMV proto on card 1. Does not stop ornith_o3.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/mnt/vm_8tb/b70/lmx_overnight/o4_m1
LOGDIR=/mnt/vm_8tb/b70/lmx_overnight
IMG=vllm-xpu-env:int8g-v0260
FUSED_SO=/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so
GDN_LIB=/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/libgdn_attn_kernels_xe_2.so
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
STATUS=$LOGDIR/STATUS
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOGDIR/o4_m1_${STAMP}.log"; }
st() { echo "$(date -u +%Y-%m-%dT%H%MZ) O4_STATUS=$*" | tee -a "$STATUS"; }

st "START stamp=$STAMP"
log "build proto"
if ! bash "$REPO/vllm/nvfp4/proto_moe_m1/build.sh" \
    >"$LOGDIR/o4_m1_build_${STAMP}.log" 2>&1; then
  st "BUILD_FAIL log=$LOGDIR/o4_m1_build_${STAMP}.log"
  tail -40 "$LOGDIR/o4_m1_build_${STAMP}.log"
  exit 2
fi
st "BUILD_OK"
log "run proto card 1"
if ! CARD=1 OUT="$OUT" IMG="$IMG" bash "$REPO/vllm/nvfp4/proto_moe_m1/run.sh" \
    >"$LOGDIR/o4_m1_run_${STAMP}.log" 2>&1; then
  st "RUN_FAIL log=$LOGDIR/o4_m1_run_${STAMP}.log"
  tail -40 "$LOGDIR/o4_m1_run_${STAMP}.log"
  exit 3
fi
st "RUN_OK log=$LOGDIR/o4_m1_run_${STAMP}.log"
tail -30 "$LOGDIR/o4_m1_run_${STAMP}.log"

log "onednn M=1 baseline card 1"
docker run --rm --device /dev/dri \
  -e ZE_AFFINITY_MASK=1 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -v "$FUSED_SO:$PKGD/_xpu_C.abi3.so:ro" \
  -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -v "$REPO/vllm/nvfp4/proto_moe_m1:/work:ro" \
  --entrypoint python3 "$IMG" /work/bench_onednn_m1.py \
  >"$LOGDIR/o4_m1_onednn_${STAMP}.log" 2>&1 || {
    st "ONEDNN_FAIL log=$LOGDIR/o4_m1_onednn_${STAMP}.log"
    tail -20 "$LOGDIR/o4_m1_onednn_${STAMP}.log"
  }
tail -20 "$LOGDIR/o4_m1_onednn_${STAMP}.log" || true

log "unit test_fused_moe_apply.py card 1 (must stay XPUGraph PASS)"
docker run --rm --device /dev/dri \
  -e ZE_AFFINITY_MASK=1 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e PYTHONPATH=/opt/nvfp4_shim \
  -v "$FUSED_SO:$PKGD/_xpu_C.abi3.so:ro" \
  -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -v "$REPO/vllm/nvfp4/patches:/opt/nvfp4_shim:ro" \
  -v "$REPO/vllm/nvfp4/test_fused_moe_apply.py:/work/test_fused_moe_apply.py:ro" \
  --entrypoint python3 "$IMG" /work/test_fused_moe_apply.py \
  >"$LOGDIR/o4_m1_unit_${STAMP}.log" 2>&1 || {
    st "UNIT_FAIL log=$LOGDIR/o4_m1_unit_${STAMP}.log"
    tail -30 "$LOGDIR/o4_m1_unit_${STAMP}.log"
    exit 4
  }
st "UNIT_PASS log=$LOGDIR/o4_m1_unit_${STAMP}.log"
tail -15 "$LOGDIR/o4_m1_unit_${STAMP}.log"
st "DONE stamp=$STAMP"
log "DONE"
