#!/usr/bin/env bash
# 100 -- MTP-ON KV capacity @ MAXLEN=262144: the recipe config (TP=2, MTP spec=5, splitting_ops, GRAPH=1 capture).
# Answers "with MTP on, what maxlen fits?" directly. MTP-off (scripts/99) = 479,090 tok pool / 1.83x @ 262K, fp16.
# MTP-on shaves the pool by the drafter head + capture. Try fp16 first; if tight, Half-KV (fp8_e4m3).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp100
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp100_mtp262k_${TS}.txt"; : > "$SUMM"
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
MAXLEN="${MAXLEN:-262144}"; UTIL="${UTIL:-0.95}"
echo "=== 100 MTP-ON KV capacity @ MAXLEN=$MAXLEN (recipe: spec5 + splitting_ops + GRAPH=1) ===" | tee -a "$SUMM"

serve() {  # $1 kvdtype("" = fp16)
  local kv="$1" KVARG=(); [ -n "$kv" ] && KVARG=(--kv-cache-dtype "$kv")
  docker rm -f "$NAME" 2>/dev/null || true
  local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  local CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"cudagraph_capture_sizes\":[1,2,4,6,8],\"splitting_ops\":[$SPLIT],$PASS}"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 32g -p ${PORT}:${PORT} \
    --pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
    -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
    -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
    -e OMP_NUM_THREADS=8 -e PYTHONPATH="$SHIM" -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e CCL_ENABLE_SYCL_KERNELS=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
    --entrypoint vllm "$IMG" serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" --dtype auto \
    --tensor-parallel-size 2 --max-model-len "$MAXLEN" --max-num-seqs 4 --gpu-memory-utilization "$UTIL" \
    --no-enable-prefix-caching --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
    --distributed-executor-backend mp --compilation-config "$CC" \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' "${KVARG[@]}" >/dev/null
  local i; for i in $(seq 1 220); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1
}
report() { docker logs "$NAME" 2>&1 | grep -iE "Available KV cache memory|GPU KV cache size|Maximum concurrency|out of memory|less than .* available|Detected MTP|mtp-bf16-shim" | grep -viE "OperatorEntry" | tail -8 | sed 's/^/      /' | tee -a "$SUMM"; }

echo ">>> A: fp16 KV @ MAXLEN=$MAXLEN" | tee -a "$SUMM"
if serve ""; then echo "    HEALTHY (MTP-on fp16 @ $MAXLEN)" | tee -a "$SUMM"; report
else
  echo "    fp16 FAILED -> retry Half-KV (fp8_e4m3)" | tee -a "$SUMM"; report
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 5
  echo ">>> B: Half-KV (fp8_e4m3) @ MAXLEN=$MAXLEN" | tee -a "$SUMM"
  if serve "fp8_e4m3"; then echo "    HEALTHY (MTP-on Half-KV @ $MAXLEN)" | tee -a "$SUMM"; report
  else echo "    Half-KV ALSO failed:" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | tail -12 | tee -a "$SUMM"; fi
fi
docker rm -f "$NAME" >/dev/null 2>&1
echo "=== 100 done ===" | tee -a "$SUMM"
