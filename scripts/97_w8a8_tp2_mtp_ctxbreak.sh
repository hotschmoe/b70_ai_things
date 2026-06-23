#!/usr/bin/env bash
# 97 -- find WHERE MTP-on TP=2 breaks by context length, and test the chunked-prefill fix.
# Run 95 showed MTP-on @ 2048-ctx prefilled then decode stalled ~0 t/s (real, via in-container vllm bench serve).
# Run 96-B "hang" was a SCRIPT BUG (host has no python3 -> empty request bodies). This rebuilds JSON in pure bash.
# A: MTP-on default (chunked-prefill ON) -> probe ctx 128/512/1024/2048, classify OK/ERROR/HANG, log at first hang.
# B: MTP-on + --no-enable-chunked-prefill (the candidate fix) -> re-probe, does it clear the stall?
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp97
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp97_ctxbreak_${TS}.txt"; : > "$SUMM"
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
echo "=== 97 W8A8 TP=2 MTP context-break probe ===" | tee -a "$SUMM"

serve() {  # $1 extra-args... (e.g. --no-enable-chunked-prefill)
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
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' "$@" >/dev/null
  local i; for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1
}
# pure-bash repeated-sentence prompt of ~N tokens (sentence ~12 tok, no quotes/backslashes)
mkprompt() { local n="$1" reps=$(( n / 12 )) i p=""; for ((i=0;i<reps;i++)); do p+="The quick brown fox jumps over the lazy dog while the engineer measures latency. "; done; printf '%s' "$p"; }
probe() {  # $1 ctx-tokens
  local n="$1" P t0 t1 resp ct el cls
  P="$(mkprompt "$n")"
  t0=$(date +%s.%N)
  resp=$(curl -s --max-time 75 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    --data-raw "{\"model\":\"$SERVED\",\"prompt\":\"$P\",\"max_tokens\":16,\"temperature\":0,\"ignore_eos\":true}" 2>/dev/null)
  t1=$(date +%s.%N); el=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  ct=$(echo "$resp" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'); ct=${ct:-0}
  if [ "$ct" -ge 16 ]; then cls="OK"
  elif echo "$resp" | grep -qiE '"error"|"message"'; then cls="ERROR:$(echo "$resp" | grep -oE '"message":"[^"]{0,60}' | head -1)"
  elif awk -v e="$el" 'BEGIN{exit !(e>=74)}'; then cls="HANG(timeout)"
  else cls="STALL(ct=$ct)"; fi
  printf "    ctx~%-5s -> %-16s (completion_tokens=%s, %.2fs)\n" "$n" "$cls" "$ct" "$el" | tee -a "$SUMM"
  [ "$cls" = OK ] && return 0 || return 1
}

HANG_AT=""
echo ">>> A: MTP-on DEFAULT (chunked-prefill ON)" | tee -a "$SUMM"
if serve; then
  for N in 64 256 512 1024 1536 2048; do
    probe "$N" || { [ -z "$HANG_AT" ] && HANG_AT="$N"; }
  done
  if [ -n "$HANG_AT" ]; then
    echo "    first break at ctx~$HANG_AT -- engine state:" | tee -a "$SUMM"
    docker logs "$NAME" 2>&1 | grep -iE "Running:|generation throughput|Avg prompt|chunk|Waiting:" | tail -4 | sed 's/^/      /' | tee -a "$SUMM"
  fi
else echo "A SERVE-FAIL" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | tail -10 | tee -a "$SUMM"; fi
docker rm -f "$NAME" >/dev/null 2>&1; sleep 5

echo ">>> B: MTP-on + --no-enable-chunked-prefill (candidate fix)" | tee -a "$SUMM"
if serve --no-enable-chunked-prefill; then
  for N in 512 1024 2048; do probe "$N" || true; done
else echo "B SERVE-FAIL (maybe spec requires chunked-prefill):" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | grep -iE "error|chunk|spec|assert" | tail -8 | sed 's/^/      /' | tee -a "$SUMM"; fi
docker rm -f "$NAME" >/dev/null 2>&1

echo "=== 97 done ===" | tee -a "$SUMM"
