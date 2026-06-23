#!/usr/bin/env bash
# Serve the 27B W4A8-sqgptq MTP graft single-card (PIECEWISE capture, MTP spec=5, BF16-MTP shim) and READ THE TEXT
# + accept -- to check whether the "W4A8 single-card MTP 2.03x" headline (scripts/90) is real or garbage-benched
# (same ignore-list / capture risk class as the W8A8 headline). Mirrors scripts/90 w4a8-mode serve exactly.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG=vllm-xpu-env:int8g
PORT=18080; NAME=vllm_w4a8graft; SERVED=w4a8graft
MODEL=/models/Qwen3.6-27B-W4A8-sqgptq-prepacked-mtp-graft
SHIM=$MODEL/mtp_bf16_patch
SPEC="${SPEC:-5}"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
KP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py
SP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py
GDN_SO=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so
GDN_LIB=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so
PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"cudagraph_capture_sizes\":[1,2,4,6,8],$PASS}"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} --pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -v "$ROOT/patches/xpu.py:$KP:ro" -v "$ROOT/patches/compressed_tensors_w4a8_int.py:$SP:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache -e TRITON_CACHE_DIR=/vllm_cache/triton \
  -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO -e OMP_NUM_THREADS=8 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTHONPATH="$SHIM" -e VLLM_W4A8_PREPACKED=1 -e ZE_AFFINITY_MASK="${DEVICE:-0}" \
  --entrypoint vllm "$IMG" serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
  --dtype auto --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 4 --gpu-memory-utilization 0.97 \
  --no-enable-prefix-caching --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config "$CC" --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC}" >/dev/null
echo "launched $NAME (W4A8 graft, PIECEWISE, MTP spec=$SPEC)"
