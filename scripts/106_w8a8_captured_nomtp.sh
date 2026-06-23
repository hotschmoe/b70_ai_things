#!/usr/bin/env bash
# 27B W8A8 body, TP=2, PIECEWISE CAPTURE (use_inductor_graph_partition=false), splitting_ops eject collectives,
# NO speculative-config, NO mtp shim. Isolates whether the captured TP=2 hybrid-int8 path corrupts WITHOUT MTP.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
CKPT=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
NAME=vllm_w8a8_capnomtp
PORT=8000
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so
GDN_LIB=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so
ATTN='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":${IGP:-false},\"cudagraph_capture_sizes\":[1,2,4,8],\"splitting_ops\":[$ATTN],$PASS}"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g -p "${PORT}:${PORT}" --pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache -e TMPDIR=/tmp_ssd \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e VLLM_LOGGING_LEVEL=INFO -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e OMP_NUM_THREADS=8 \
  -e CCL_ENABLE_SYCL_KERNELS=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
  --entrypoint vllm vllm-xpu-env:int8g \
  serve "$CKPT" --served-model-name w8a8capnomtp --host 0.0.0.0 --port "$PORT" \
  --dtype auto --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching --trust-remote-code --distributed-executor-backend mp \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config "$CC" >/dev/null
echo "launched $NAME (capture, no-MTP, IGP=${IGP:-false})"
