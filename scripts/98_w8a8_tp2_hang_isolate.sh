#!/usr/bin/env bash
# 98 -- isolate the scripts/95 MTP-on hang. 97 proved ctx length alone is NOT it (64..2048 all OK @ max_tokens=16).
# Remaining differences between 97(OK) and 95(HANG): (1) output length 16 vs 128, (2) single curl vs `vllm bench
# serve` 8-prompt path, (3) real text vs --random gibberish tokens. Test each, all on the recipe config (MTP spec5).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp98
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp98_hangiso_${TS}.txt"; : > "$SUMM"
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
echo "=== 98 isolate the MTP-on hang ===" | tee -a "$SUMM"

serve() {
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
    --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
    --distributed-executor-backend mp --compilation-config "$CC" \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' >/dev/null
  local i; for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1
}
mkprompt() { local n="$1" reps=$(( n / 12 )) i p=""; for ((i=0;i<reps;i++)); do p+="The quick brown fox jumps over the lazy dog while the engineer measures latency. "; done; printf '%s' "$p"; }
probe() {  # $1 ctx  $2 max_tokens  $3 timeout  $4 label
  local n="$1" mt="$2" tmo="$3" lab="$4" P t0 t1 resp ct el cls
  P="$(mkprompt "$n")"; t0=$(date +%s.%N)
  resp=$(curl -s --max-time "$tmo" "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    --data-raw "{\"model\":\"$SERVED\",\"prompt\":\"$P\",\"max_tokens\":$mt,\"temperature\":0,\"ignore_eos\":true}" 2>/dev/null)
  t1=$(date +%s.%N); el=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  ct=$(echo "$resp" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'); ct=${ct:-0}
  if [ "$ct" -ge "$mt" ]; then cls="OK"; elif awk -v e="$el" -v t="$tmo" 'BEGIN{exit !(e>=t-2)}'; then cls="HANG(timeout)"; else cls="PARTIAL(ct=$ct)"; fi
  printf "    %-26s ctx~%-5s out=%-4s -> %-14s (ct=%s, %.2fs)\n" "$lab" "$n" "$mt" "$cls" "$ct" "$el" | tee -a "$SUMM"
}

if serve; then
  echo ">>> T1 real prompt, long output (the 95 output length)" | tee -a "$SUMM"
  probe 2048 128 100 "T1a real-2048-out128"
  probe 2048 256 130 "T1b real-2048-out256"
  echo ">>> T2 back-to-back real requests (multi-request state)" | tee -a "$SUMM"
  probe 1024 128 100 "T2a real-1024-out128"
  probe 1024 128 100 "T2b real-1024-out128"
  probe 1024 128 100 "T2c real-1024-out128"
  echo ">>> T3 the 95 path: vllm bench serve --random 2048/128 (timeout-guarded, 2 prompts)" | tee -a "$SUMM"
  if timeout 200 docker exec -i "$NAME" vllm bench serve --backend vllm --model "$SERVED" --tokenizer "$MODEL" \
      --base-url "http://localhost:$PORT" --endpoint /v1/completions --dataset-name random \
      --random-input-len 2048 --random-output-len 128 --num-prompts 2 --max-concurrency 1 --ignore-eos 2>&1 \
      | grep -iE "throughput|TTFT|TPOT|Maximum|error" | tee -a "$SUMM"; then
    echo "    T3 random-bench COMPLETED (no hang)" | tee -a "$SUMM"
  else
    echo "    T3 random-bench HUNG/TIMED OUT (>200s) -> the --random gibberish path is the trigger" | tee -a "$SUMM"
    docker logs "$NAME" 2>&1 | grep -iE "Running:|generation throughput|Avg prompt" | tail -3 | sed 's/^/      /' | tee -a "$SUMM"
  fi
else echo "SERVE-FAIL" | tee -a "$SUMM"; fi
docker rm -f "$NAME" >/dev/null 2>&1
echo "=== 98 done ===" | tee -a "$SUMM"
