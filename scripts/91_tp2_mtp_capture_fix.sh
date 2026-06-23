#!/usr/bin/env bash
# 91 -- TP=2 MTP PIECEWISE capture-crash FIX probe (follow-up to scripts/90 w8a8tp2).
# Baseline crash (90): MTP spec=4 PIECEWISE dies at capture with
#   oneCCL coll.cpp:1204 ccl_allgather_impl: |CCL_SYCL| sched algorithms do not support sycl_graph recording.
# The spec verify adds a model-forward `vllm::all_gather` (a real registered custom op) that oneCCL records
# into the SYCL graph via its scheduler ("sched") algorithm, which has no graph-recordable impl.
# oneCCL in the image = 2021.17. `vllm::all_gather`/`all_reduce`/`reduce_scatter` are vllm:: custom ops.
#
# Two orthogonal fix families, tested as named variants (each = one full TP=2 MTP spec=4 PIECEWISE serve):
#   A split   -- put the collectives in splitting_ops -> inductor partitions at them -> they run EAGER
#                (never recorded into the SYCL graph), while decode GEMMs/attention stay CAPTURED.
#   B agvring -- codex oneCCL knob: force allgatherv onto a sycl-recordable scaleout algo (ring + big thresh).
#   C topo    -- codex oneCCL knob: force allgather/allgatherv onto the topo (Level-Zero) sycl path.
#   D both    -- A splitting_ops + B env (belt and suspenders).
# WIN = serve reaches /health (crash cleared) AND, if so, bench decode_tps + accept_len vs 90's off=18.74.
#
#   /mnt/vm_8tb/b70/gpu-run bash 91_tp2_mtp_capture_fix.sh            # all variants
#   /mnt/vm_8tb/b70/gpu-run bash 91_tp2_mtp_capture_fix.sh A C        # subset
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp91; SPEC=4
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp91_tp2fix_${TS}.txt"; : > "$SUMM"
VARIANTS="${*:-A B C D}"
# default attention splitting_ops for this model + the 3 collectives (variant A/D)
SPLIT_WITH_COLL='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, docstrings, and an example. Then explain the time complexity step by step in detail."

echo "=== 91 TP=2 MTP capture-fix probe  model=$MODEL  oneCCL=2021.17  variants={$VARIANTS} ===" | tee -a "$SUMM"
echo "    (90 baseline: off PIECEWISE 18.74 t/s healthy; MTP spec4 PIECEWISE CRASH; MTP spec4 eager 14.27=0.76x)" | tee -a "$SUMM"

gen_tok() { curl -s --max-time 360 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }

run_variant() {  # $1 = variant letter
  local V="$1" CCL=() SPLITOPS=""
  case "$V" in
    A) SPLITOPS="\"splitting_ops\":[$SPLIT_WITH_COLL]," ;;
    B) CCL=(-e CCL_SYCL_ALLGATHERV_SCALEOUT=ring -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=4294967296
            -e CCL_SYCL_ALLGATHERV_TMP_BUF=1 -e CCL_SYCL_ALLGATHERV_SCALEOUT_COMM_SIZE=1024) ;;
    C) CCL=(-e CCL_ALLGATHER=topo -e CCL_ALLGATHERV=topo -e CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=1) ;;
    D) SPLITOPS="\"splitting_ops\":[$SPLIT_WITH_COLL],"
       CCL=(-e CCL_SYCL_ALLGATHERV_SCALEOUT=ring -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=4294967296
            -e CCL_SYCL_ALLGATHERV_TMP_BUF=1 -e CCL_SYCL_ALLGATHERV_SCALEOUT_COMM_SIZE=1024) ;;
    *) echo "unknown variant $V" | tee -a "$SUMM"; return ;;
  esac
  echo ">>> VARIANT $V" | tee -a "$SUMM"
  docker rm -f "$NAME" 2>/dev/null || true
  local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  local CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"cudagraph_capture_sizes\":[1,2,4,8],${SPLITOPS}$PASS}"
  local ARGS=(serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" --dtype auto
        --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.90
        --no-enable-prefix-caching --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}'
        --distributed-executor-backend mp --compilation-config "$CC"
        --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC}")
  echo "    CC=$CC" | tee -a "$SUMM"
  [ ${#CCL[@]} -gt 0 ] && echo "    extra-env: ${CCL[*]}" | tee -a "$SUMM"
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
    -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd "${CCL[@]}" \
    --entrypoint vllm "$IMG" "${ARGS[@]}" >/dev/null

  local ok=0 i
  for i in $(seq 1 170); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && break
    sleep 5
  done
  if [ "$ok" = 1 ]; then
    echo "    VARIANT $V: CRASH CLEARED -- HEALTHY. benching..." | tee -a "$SUMM"
    gen_tok 8 >/dev/null
    local s0 s1 l0 l1 ns nl M A D
    s0=$(date +%s.%N); ns=$(gen_tok 64); s1=$(date +%s.%N); l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
    M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
    A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
    D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
    awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v V="$V" -v A="$A" -v D="$D" \
      'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; mx=tps/18.74; al=(D>0)?(A/D)+1:0;
       printf "    VARIANT %s RESULT: decode_tps=%6.2f  vs-off-x=%.2f  accept_len=%.2f  (acc=%d drafts=%d, gen512 %.2fs)\n",
         V, tps, mx, al, A, D, (tl1-tl0)}' | tee -a "$SUMM"
  else
    echo "    VARIANT $V: STILL FAILS -- crash signature:" | tee -a "$SUMM"
    docker logs "$NAME" 2>&1 | grep -iE "allgather|sycl_graph|sched algorithm|RuntimeError|allreduce|out of memory|Traceback" | tail -8 | sed 's/^/      /' | tee -a "$SUMM"
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true; sleep 5
}

for V in $VARIANTS; do run_variant "$V"; done
echo "=== 91 SUMMARY ===" | tee -a "$SUMM"; grep -E "VARIANT|RESULT|CLEARED|STILL FAILS" "$SUMM"
echo "=== 91 tp2fix done ==="
