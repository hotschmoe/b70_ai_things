#!/usr/bin/env bash
# 96 -- W8A8 TP=2 prefill/TTFT @ ctx, DEFENSIVE rewrite of 95 (which HUNG: MTP-on + 2048-ctx random prompt ->
# prefill OK but decode stalled at ~0 t/s for 30 min, a TP collective/spec-decode long-ctx deadlock).
# Defenses: cleanup trap (always docker rm -f), curl --max-time on every call, short probes, keep logs.
# Goals:
#   A = MTP-OFF int8 prefill/TTFT baseline @ IN=2048 (the number for the int8-prefill optimization). Robust (no spec).
#   B = MTP-ON characterization: real prompts at 256 / 1024 / 1800 tokens, max_tokens=24, timed -> where does it break?
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp96
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp96_prefill_${TS}.txt"; : > "$SUMM"
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
echo "=== 96 W8A8 TP=2 prefill/TTFT (defensive) ===" | tee -a "$SUMM"

serve() {  # $1 spec(off|5) $2 usesplit(0|1) $3 caps
  local spec="$1" usesplit="$2" caps="$3" SPECARG=() CSZ='"compile_sizes":[1],' SPL=""
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
# build a real prompt of ~N tokens by repeating a sentence (~12 tok each)
mkprompt() { local n="$1" reps=$(( n / 12 )) s="The quick brown fox jumps over the lazy dog while the engineer measures prefill latency. "; local p=""; local i; for ((i=0;i<reps;i++)); do p+="$s"; done; printf '%s' "$p"; }
# timed completion: $1 prompt, $2 max_tokens, $3 max_time -> prints "completion_tokens elapsed_s http_ok"
timed_gen() {
  local prompt="$1" mt="$2" tmo="$3" t0 t1 out ct
  t0=$(date +%s.%N)
  out=$(curl -s --max-time "$tmo" "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "$(python3 - "$prompt" "$mt" <<'PY'
import json,sys
print(json.dumps({"model":"qwen36-27b-w8a8-sqgptq-mtp","prompt":sys.argv[1],"max_tokens":int(sys.argv[2]),"temperature":0,"ignore_eos":True}))
PY
)" 2>/dev/null)
  t1=$(date +%s.%N)
  ct=$(echo "$out" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'); ct=${ct:-0}
  awk -v c="$ct" -v e="$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')" 'BEGIN{printf "%d %.2f\n", c, e}'
}

echo ">>> A: MTP-OFF int8 prefill/TTFT @ IN=2048 (vllm bench serve, timeout-guarded)" | tee -a "$SUMM"
if serve off 0 "1,2,4,8"; then
  timeout 420 env NAME="$NAME" MODEL="$SERVED" LABEL="w8a8tp2_mtpoff_pf" TOKPATH="$MODEL" PORT="$PORT" \
    IN=2048 OUT=128 CONC="1 4" bash "$ROOT/35_sweep_bench.sh" 2>&1 | tee -a "$SUMM" || echo "A: bench timed out/failed" | tee -a "$SUMM"
else echo "A SERVE-FAIL" | tee -a "$SUMM"; fi
docker rm -f "$NAME" >/dev/null 2>&1; sleep 5

echo ">>> B: MTP-ON characterization (real prompts 256/1024/1800 tok, max_tokens=24, --max-time 90)" | tee -a "$SUMM"
if serve 5 1 "1,2,4,6,8"; then
  P_WARM="The capital of France is"; r=$(timed_gen "$P_WARM" 8 60); echo "    warmup(short): tokens elapsed = $r" | tee -a "$SUMM"
  for N in 256 1024 1800; do
    P="$(mkprompt "$N")"
    # TTFT proxy: max_tokens=1 (prefill + 1) ; decode: max_tokens=25 -> rate from (25-1)/(t25-t1)
    r1=$(timed_gen "$P" 1 90);  c1=${r1% *}; e1=${r1#* }
    r25=$(timed_gen "$P" 25 90); c25=${r25% *}; e25=${r25#* }
    awk -v N="$N" -v c1="$c1" -v e1="$e1" -v c25="$c25" -v e25="$e25" \
      'BEGIN{ ttft=e1; dt=e25-e1; dn=(c25-c1); dr=(dt>0&&dn>0)?dn/dt:0;
        hung=(c25<=1)?" [HANG/STALL]":"";
        printf "    ctx~%-5s TTFT~%.2fs (prefill, %d tok) | decode %.2f t/s (%d tok in %.2fs)%s\n", N, ttft, c1, dr, dn, dt, hung}' | tee -a "$SUMM"
  done
  echo "    --- engine log tail (spec/collective/stall signals) ---" | tee -a "$SUMM"
  docker logs "$NAME" 2>&1 | grep -iE "spec|allgather|stall|chunk|prefill|generation throughput|Running:" | tail -6 | sed 's/^/      /' | tee -a "$SUMM"
else echo "B SERVE-FAIL" | tee -a "$SUMM"; fi
docker rm -f "$NAME" >/dev/null 2>&1

echo "=== 96 done (prefill tok/s ~= 2048 / (mean_ttft_ms/1000)) ===" | tee -a "$SUMM"
