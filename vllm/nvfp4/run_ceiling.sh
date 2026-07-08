#!/usr/bin/env bash
# Run an NVFP4 INT8-XMX prefill microbench inside int8g-v0240 on CARD 0, with the
# fused GDN .so (has ALL of nvfp4/int8/int4 gemm ops) mounted. Wrap in `gpu-run --card 0`.
#   ./bin/gpu-run --card 0 bash vllm/nvfp4/run_ceiling.sh [script.py] [extra docker env...]
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:int8g-v0240}"
SCRIPT="${1:-vllm/nvfp4/int8_prefill_ceiling.py}"
NAME="${NAME:-nvfp4_pref_bench}"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
FUSED_SO="${FUSED_SO:-$ROOT/nvfp4_fused_kernel_gdn/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/nvfp4_fused_kernel_gdn/libgdn_attn_kernels_xe_2.so}"
[ -f "$FUSED_SO" ] || { echo "MISSING $FUSED_SO"; exit 1; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  -v "$REPO:/repo:ro" -v "$REPO/models/files:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$FUSED_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK="${CARD:-0}" -e PYTHONUNBUFFERED=1 \
  -e ONEDNN_VERBOSE="${ONEDNN_VERBOSE:-0}" -e MS="${MS:-}" -e DO_I4="${DO_I4:-0}" -e ITERS="${ITERS:-20}" \
  --entrypoint python "$IMG" -u "/repo/$SCRIPT" "${@:2}"
