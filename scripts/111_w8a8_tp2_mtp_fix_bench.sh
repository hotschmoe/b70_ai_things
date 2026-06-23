#!/usr/bin/env bash
# 111 -- W8A8 27B TP=2 MTP bench with THE BUG-B FIX: eject ONLY all_gather (the lone collective oneCCL cannot
# SYCL-graph-record), keep all_reduce + reduce_scatter CAPTURED. The old "63 t/s" (scripts/93) ejected ALL THREE
# collectives -> the ejected per-layer all_reduce broke the captured-piece input-address contract -> GARBAGE that
# the token-count bench scored as a win. This bench fixes the config AND prints the actual generated TEXT for a
# coherence read-out (config -> command -> result -> verdict; never trust token counts again).
#
#   /mnt/vm_8tb/b70/gpu-run bash 111_w8a8_tp2_mtp_fix_bench.sh           # off 5   (add args to change specs)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM="$ROOT/rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/patches"   # BF16-MTP graft shim (proven in scripts/109 cell F)
PORT=18080; NAME=vllm_mtp111
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp111_${TS}.txt"; : > "$SUMM"
CSV="$ROOT/results/mtp111_${TS}.csv"; echo "spec,decode_tps,mtp_x,accept_len,accept_rate,gen512_s,coherence" > "$CSV"
SPECS="${*:-off 5}"
ATTN='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
# THE FIX: eject ONLY all_gather. all_reduce + reduce_scatter stay CAPTURED (they record fine, and ejecting them
# is what corrupted decode). EJECT=none -> capture ALL collectives incl all_gather (needs the csag plan-B shim).
EJECT="${EJECT:-ag}"
case "$EJECT" in
  ag)   SPLIT="$ATTN,\"vllm::all_gather\"" ;;
  none) SPLIT="$ATTN" ;;
  *) echo "bad EJECT=$EJECT"; exit 2 ;;
esac
PROMPT="Discuss the major causes of the decline of the Roman Empire, weighing economic, military, political, and social factors against each other, and explain which you find most persuasive and why, with specific historical examples."
COH_PROMPT="The capital of France is"

echo "=== 111 W8A8 TP=2 MTP fix (eject ONLY all_gather, IGP=false) specs={$SPECS} ===" | tee -a "$SUMM"
gen_tok() { curl -s --max-time 360 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }
coh_text() { curl -s --max-time 60 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$COH_PROMPT\",\"max_tokens\":32,\"temperature\":0}" \
    | grep -oE '"text":"[^"]*"' | head -1 | sed 's/^"text":"//; s/"$//'; }

GRAPH="${GRAPH:-1}"          # 1=PIECEWISE capture (the fix), 0=eager (accept baseline)
SHIM_OVERRIDE="${SHIM_OVERRIDE:-}"   # point PYTHONPATH at a different shim dir (e.g. the csag plan-B shim)
[ -n "$SHIM_OVERRIDE" ] && SHIM="$SHIM_OVERRIDE"
serve() {  # $1 spec
  local spec="$1" SPECARG=() CAP="1,2,4,8" CCARG=()
  docker rm -f "$NAME" 2>/dev/null || true
  if [ "$spec" != off ]; then SPECARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$spec}"); CAP="1,2,4,$((spec+1)),8"; fi
  local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  local CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":false,\"cudagraph_capture_sizes\":[$CAP],\"splitting_ops\":[$SPLIT],$PASS}"
  if [ "$GRAPH" = 1 ]; then CCARG=(--compilation-config "$CC"); else CCARG=(--enforce-eager); fi
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 32g -p ${PORT}:${PORT} \
    --pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
    -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
    -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
    -e OMP_NUM_THREADS=8 -e PYTHONPATH=/opt/mtp_shim -v "$SHIM:/opt/mtp_shim:ro" -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e CCL_ENABLE_SYCL_KERNELS=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
    --entrypoint vllm "$IMG" serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" --dtype auto \
    --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
    --distributed-executor-backend mp "${CCARG[@]}" "${SPECARG[@]}" >/dev/null
}
wait_healthy() { local i; for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1; }
bench() {  # $1 spec $2 basetps
  local spec="$1" base="$2"
  gen_tok 8 >/dev/null
  local s0 s1 l0 l1 ns nl M A D DT TXT COH
  s0=$(date +%s.%N); ns=$(gen_tok 64); s1=$(date +%s.%N); l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  TXT=$(coh_text); echo "   COHERENCE TEXT: '$TXT'" | tee -a "$SUMM"
  COH=$(printf '%s' "$TXT" | awk '{s=$0; gsub(/ /,"",s); n=length(s); if(n<8){print "SHORT"; exit} for(i=1;i<=n;i++){c[substr(s,i,1)]++} m=0; for(k in c) if(c[k]>m)m=c[k]; print (m/n>0.55)?"GARBAGE":"OK"}')
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  DT=$(echo "$M" | awk '/num_draft_tokens_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v sp="$spec" \
      -v A="$A" -v D="$D" -v DT="$DT" -v base="$base" -v csv="$CSV" -v coh="$COH" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; mx=(base>0)?tps/base:0; al=(D>0)?(A/D)+1:0; ar=(DT>0)?A/DT:0;
      printf "spec=%-3s decode_tps=%6.2f  MTPx=%5.2f  accept_len=%.2f  accept_rate=%.3f  (gen512 %.2fs)  coherence=%s\n", sp, tps, mx, al, ar, (tl1-tl0), coh;
      printf "%s,%.2f,%.2f,%.2f,%.3f,%.2f,%s\n", sp, tps, mx, al, ar, (tl1-tl0), coh >> csv; print tps > "/tmp/mtp111_tps"}' | tee -a "$SUMM"
}
BASE=0
for SP in $SPECS; do
  echo ">>> spec=$SP" | tee -a "$SUMM"
  serve "$SP"
  if wait_healthy; then bench "$SP" "$BASE"; [ "$SP" = off ] && BASE=$(cat /tmp/mtp111_tps 2>/dev/null || echo 0)
  else echo "spec=$SP FAIL" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | grep -iE "allgather|sycl_graph|RuntimeError|out of memory|weight_scale" | tail -8 | sed 's/^/   /' | tee -a "$SUMM"; echo "$SP,FAIL,,,,," >> "$CSV"; fi
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 30   # let the XPU driver release ~30 GiB before the next serve (else OOM)
done
echo "=== 111 SUMMARY ===" | tee -a "$SUMM"; cat "$CSV"
echo "=== 111 done ==="
