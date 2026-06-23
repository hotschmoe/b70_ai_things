#!/usr/bin/env bash
# 94 -- does the scripts/92 "capture the spec-verify batch" lever STACK on the scripts/91 TP=2 splitting_ops fix?
# 93 ran W8A8 TP=2 spec=5 at 63.11 t/s with caps 1,2,4,8 -- MISSING the spec=5 verify batch 6 (ran eager).
# Here: same splitting_ops fix, but caps include the verify batch (6 for spec5, 7 for spec6). spec {5,6}.
# Compare to 93's 63.11 (spec5) and the climbing trend. Honest multiplier vs best MTP-off TP=2 = 18.74 (scripts/91).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp94
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp94_w8a8tp2caps_${TS}.txt"; : > "$SUMM"
CSV="$ROOT/results/mtp94_w8a8tp2caps_${TS}.csv"; echo "spec,caps,decode_tps,mtp_x_vs18.74,accept_len,accept_rate,gen512_s" > "$CSV"
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
PROMPT="Discuss the major causes of the decline of the Roman Empire, weighing economic, military, political, and social factors against each other, and explain which you find most persuasive and why, with specific historical examples."
# spec -> caps (include the 1+spec verify batch)
declare -A CAPS_FOR=( [5]="1,2,4,6,8" [6]="1,2,4,7,8" )
SPECS="${*:-5 6}"
echo "=== 94 W8A8 TP=2 capspec-stacks-on-splitting_ops (93 spec5 caps1,2,4,8 = 63.11) specs={$SPECS} ===" | tee -a "$SUMM"

gen_tok() { curl -s --max-time 360 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }
serve() {  # $1 spec  $2 caps
  local spec="$1" caps="$2"
  docker rm -f "$NAME" 2>/dev/null || true
  local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  local CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"cudagraph_capture_sizes\":[$caps],\"splitting_ops\":[$SPLIT],$PASS}"
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
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$spec}" >/dev/null
}
wait_healthy() { local i; for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1; }
bench() {  # $1 spec $2 caps
  local spec="$1" caps="$2"
  gen_tok 8 >/dev/null
  local s0 s1 l0 l1 ns nl M A D DT
  s0=$(date +%s.%N); ns=$(gen_tok 64); s1=$(date +%s.%N); l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  DT=$(echo "$M" | awk '/num_draft_tokens_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v sp="$spec" -v caps="$caps" \
      -v A="$A" -v D="$D" -v DT="$DT" -v csv="$CSV" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; mx=tps/18.74; al=(D>0)?(A/D)+1:0; ar=(DT>0)?A/DT:0;
      printf "spec=%s caps=%-11s decode_tps=%6.2f  vs-bestoff-x=%.2f  accept_len=%.2f  accept_rate=%.3f  (gen512 %.2fs)\n", sp, caps, tps, mx, al, ar, (tl1-tl0);
      printf "%s,\"%s\",%.2f,%.2f,%.2f,%.3f,%.2f\n", sp, caps, tps, mx, al, ar, (tl1-tl0) >> csv}' | tee -a "$SUMM"
}
for SP in $SPECS; do
  C="${CAPS_FOR[$SP]:-1,2,4,8}"
  echo ">>> spec=$SP caps=$C" | tee -a "$SUMM"
  serve "$SP" "$C"
  if wait_healthy; then bench "$SP" "$C"; else echo "spec=$SP FAIL" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | grep -iE "allgather|RuntimeError|out of memory" | tail -5 | sed 's/^/   /' | tee -a "$SUMM"; fi
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 5
done
echo "=== 94 SUMMARY ===" | tee -a "$SUMM"; cat "$CSV"
echo "=== 94 done ==="
