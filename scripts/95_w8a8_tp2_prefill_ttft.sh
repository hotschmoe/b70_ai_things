#!/usr/bin/env bash
# 95 -- W8A8 27B TP=2 prefill + TTFT at 2048 ctx (the int8xint8 prefill baseline under the MTP recipe).
# Decode is solved (scripts/93: spec5 ~64 t/s, 3.4x via the splitting_ops fix). This measures the OTHER half --
# prefill throughput + TTFT at 2048 input -- so we know the int8 prefill foundation to optimize next.
# Two serves, both TP=2:
#   A = MTP-on  (splitting_ops fix + spec=5)  -> production TTFT a user actually sees.
#   B = MTP-off (default splitting_ops, captured collectives) -> pure int8 prefill ceiling.
# Bench = vLLM's own `vllm bench serve` random IN=2048 OUT=128 at C=1 (single stream) and C=4, via 35_sweep_bench.sh.
#   /mnt/vm_8tb/b70/gpu-run bash 95_w8a8_tp2_prefill_ttft.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp95
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp95_prefill_${TS}.txt"; : > "$SUMM"
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
echo "=== 95 W8A8 TP=2 prefill/TTFT @ 2048 ctx ===" | tee -a "$SUMM"

serve() {  # $1 = label(mtp|off)  $2 = spec("5"|"off")  $3 = use_splitops(1|0)  $4 = caps
  local lab="$1" spec="$2" usesplit="$3" caps="$4" SPECARG=() CSZ='"compile_sizes":[1],' SPL=""
  docker rm -f "$NAME" 2>/dev/null || true
  [ "$spec" != off ] && { SPECARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$spec}"); CSZ=''; }
  [ "$usesplit" = 1 ] && SPL="\"splitting_ops\":[$SPLIT],"
  local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  local CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"cudagraph_capture_sizes\":[$caps],${SPL}${CSZ}$PASS}"
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
    --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
    --distributed-executor-backend mp --compilation-config "$CC" "${SPECARG[@]}" >/dev/null
  local i; for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1
}
prefill_bench() {  # $1 = label
  echo ">>> $1 : vllm bench serve random IN=2048 OUT=128 (C=1 then C=4)" | tee -a "$SUMM"
  env NAME="$NAME" MODEL="$SERVED" LABEL="w8a8tp2_$1" TOKPATH="$MODEL" PORT="$PORT" \
      IN=2048 OUT=128 CONC="1 4" bash "$ROOT/35_sweep_bench.sh" 2>&1 | tee -a "$SUMM"
}

echo ">>> A: MTP-ON (splitting_ops + spec=5, caps incl verify-batch 6)" | tee -a "$SUMM"
if serve mtp 5 1 "1,2,4,6,8"; then prefill_bench "mtp_on"; else echo "A FAIL" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | tail -15 | tee -a "$SUMM"; fi
docker rm -f "$NAME" >/dev/null 2>&1; sleep 5

echo ">>> B: MTP-OFF (default splitting_ops, captured collectives)" | tee -a "$SUMM"
if serve off off 0 "1,2,4,8"; then prefill_bench "mtp_off"; else echo "B FAIL" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | tail -15 | tee -a "$SUMM"; fi
docker rm -f "$NAME" >/dev/null 2>&1

echo "=== 95 prefill done (prefill tok/s ~= 2048 / (mean_ttft_ms/1000)) ===" | tee -a "$SUMM"
