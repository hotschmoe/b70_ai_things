#!/usr/bin/env bash
# 93 -- W8A8 27B TP=2 MTP spec sweep WITH the splitting_ops fix (scripts/91 variant A revived TP=2 MTP).
# Nails down the overturned M4 verdict: full off + spec{3,4,5} curve, and a HARDER natural-language prompt so
# accept_len is honest (the 91 code prompt gave a 100% artifact). splitting_ops ejects the TP collectives to
# eager (partition boundaries) so the oneCCL allgather is never recorded -> capture succeeds; decode stays captured.
#
#   /mnt/vm_8tb/b70/gpu-run bash 93_w8a8_tp2_mtp_specsweep.sh           # off 3 4 5
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp93
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp93_w8a8tp2_${TS}.txt"; : > "$SUMM"
CSV="$ROOT/results/mtp93_w8a8tp2_${TS}.csv"; echo "spec,decode_tps,mtp_x,accept_len,accept_rate,gen512_s" > "$CSV"
SPECS="${*:-off 3 4 5}"
# 3 collectives appended to this model's attention splitting_ops (the scripts/91 variant-A fix)
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
# harder, higher-entropy prompt -> realistic accept (vs the trivially-predictable LRU-cache code prompt)
PROMPT="Discuss the major causes of the decline of the Roman Empire, weighing economic, military, political, and social factors against each other, and explain which you find most persuasive and why, with specific historical examples."

echo "=== 93 W8A8 TP=2 MTP spec sweep + splitting_ops fix (harder prompt) specs={$SPECS} ===" | tee -a "$SUMM"
gen_tok() { curl -s --max-time 360 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }

serve() {  # $1 spec
  local spec="$1" SPECARG=() CSZ='"compile_sizes":[1],'
  docker rm -f "$NAME" 2>/dev/null || true
  if [ "$spec" != off ]; then SPECARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$spec}"); CSZ=''; fi
  local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  local CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"cudagraph_capture_sizes\":[1,2,4,8],\"splitting_ops\":[$SPLIT],${CSZ}$PASS}"
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
}
wait_healthy() { local i; for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1; }
bench() {  # $1 spec $2 basetps
  local spec="$1" base="$2"
  gen_tok 8 >/dev/null
  local s0 s1 l0 l1 ns nl M A D DT
  s0=$(date +%s.%N); ns=$(gen_tok 64); s1=$(date +%s.%N); l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  DT=$(echo "$M" | awk '/num_draft_tokens_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v sp="$spec" \
      -v A="$A" -v D="$D" -v DT="$DT" -v base="$base" -v csv="$CSV" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; mx=(base>0)?tps/base:0; al=(D>0)?(A/D)+1:0; ar=(DT>0)?A/DT:0;
      printf "spec=%-3s decode_tps=%6.2f  MTPx=%5.2f  accept_len=%.2f  accept_rate=%.3f  (gen512 %.2fs)\n", sp, tps, mx, al, ar, (tl1-tl0);
      printf "%s,%.2f,%.2f,%.2f,%.3f,%.2f\n", sp, tps, mx, al, ar, (tl1-tl0) >> csv; print tps > "/tmp/mtp93_tps"}' | tee -a "$SUMM"
}
BASE=0
for SP in $SPECS; do
  echo ">>> spec=$SP" | tee -a "$SUMM"
  serve "$SP"
  if wait_healthy; then bench "$SP" "$BASE"; [ "$SP" = off ] && BASE=$(cat /tmp/mtp93_tps 2>/dev/null || echo 0)
  else echo "spec=$SP FAIL" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | grep -iE "allgather|sycl_graph|RuntimeError|out of memory" | tail -6 | sed 's/^/   /' | tee -a "$SUMM"; echo "$SP,FAIL,,,," >> "$CSV"; fi
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 5
done
echo "=== 93 SUMMARY ===" | tee -a "$SUMM"; cat "$CSV"
echo "=== 93 w8a8tp2 done ==="
