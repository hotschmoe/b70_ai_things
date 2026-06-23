#!/usr/bin/env bash
# 90 -- MTP feasibility sweep for the compressed-tensors BF16-MTP GRAFTS (W4A8, W8A8).
# Extends the validated W4A16-graft method (JOURNAL 2026-06-23) to the two remaining CT quants.
#
# The grafts add 15 BF16 mtp.* tensors (model-mtp-graft.safetensors) + a mtp_bf16_patch/sitecustomize.py
# that forces ONLY the Qwen3_5MultiTokenPredictor drafter to instantiate unquantized/BF16 (else vLLM builds
# the drafter through the CT quantized/fused path and skips the BF16 MTP linears -> bogus 0% accept).
# This script puts that shim on PYTHONPATH (the one piece the stock 30_serve engine does not wire) and
# benches MTP-off vs MTP-on across spec tokens with the proven TTFT-cancelled decode + /metrics accept-len.
#
# MODES:
#   w4a8     -- single-card (card 0). Qwen3.6-27B-W4A8-sqgptq-prepacked + GDN .so + PREPACK loader. spec {off,3,4,5}.
#   w8a8tp2  -- TP=2 (both cards, SYCLKERNELS=1). Qwen3.6-27B-W8A8-sqgptq (34G, needs 2 cards) + GDN .so.
#               Re-tests the M4 "TP=2 MTP DEAD (spec-allgather not graph-capturable)" verdict with a REAL
#               grafted MTP head. spec {off,4}.
#
# GPU lease (CLAUDE.md): run UNDER the lease, e.g.
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash 90_mtp_graft_sweep.sh w4a8
#   /mnt/vm_8tb/b70/gpu-run            bash 90_mtp_graft_sweep.sh w8a8tp2
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
MODE="${1:-w4a8}"
IMG="${IMG:-vllm-xpu-env:int8g}"
PORT="${PORT:-18080}"
NAME="vllm_mtp90"
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
KP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py
SP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py
TS="$(date +%Y%m%d_%H%M%S)"

case "$MODE" in
  w4a8)
    MODEL=/models/Qwen3.6-27B-W4A8-sqgptq-prepacked-mtp-graft
    SERVED=qwen36-27b-w4a8-sqgptq-mtp
    SHIM=$MODEL/mtp_bf16_patch
    TP=1; UTIL="${UTIL:-0.97}"; MAXLEN="${MAXLEN:-2048}"; MAXSEQS="${MAXSEQS:-4}"; CAPS="${CAPS:-1,2,4}"
    SPECS="${SPECS:-off 3 4 5}"
    EXTRA_MOUNTS=( -v "$ROOT/patches/xpu.py:$KP:ro" -v "$ROOT/patches/compressed_tensors_w4a8_int.py:$SP:ro" )
    EXTRA_ENV=( -e VLLM_W4A8_PREPACKED=1 )
    ;;
  w8a8tp2)
    MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
    SERVED=qwen36-27b-w8a8-sqgptq-mtp
    SHIM=$MODEL/mtp_bf16_patch
    TP=2; UTIL="${UTIL:-0.90}"; MAXLEN="${MAXLEN:-4096}"; MAXSEQS="${MAXSEQS:-8}"; CAPS="${CAPS:-1,2,4,8}"
    # off=PIECEWISE baseline; 4=MTP PIECEWISE (M4 says spec-capture crashes); 4e=MTP eager fallback (no capture)
    SPECS="${SPECS:-off 4 4e}"
    EXTRA_MOUNTS=()
    EXTRA_ENV=()
    ;;
  *) echo "usage: 90_mtp_graft_sweep.sh [w4a8|w8a8tp2]"; exit 2 ;;
esac

SUMM="$ROOT/results/mtp90_${MODE}_${TS}.txt"
CSV="$ROOT/results/mtp90_${MODE}_${TS}.csv"
: > "$SUMM"
echo "spec,decode_tps,mtp_x,accept_len,accept_rate,accepted,drafts,draft_tok,gen512_s" > "$CSV"
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, docstrings, and an example usage. Then explain the time complexity and walk through an example step by step in detail."

echo "=== 90 MTP graft sweep MODE=$MODE  model=$MODEL  IMG=$IMG  TP=$TP ===" | tee -a "$SUMM"
echo "    UTIL=$UTIL MAXLEN=$MAXLEN MAXSEQS=$MAXSEQS CAPS=$CAPS specs={$SPECS}" | tee -a "$SUMM"

gen_tok() { curl -s --max-time 360 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }

serve() {  # $1 = spec: "off" | <int> (PIECEWISE) | <int>e (eager, --enforce-eager)
  local spec="$1" SPECARG=() CSZ='"compile_sizes":[1],' EAGER=0 num="$1"
  docker rm -f "$NAME" 2>/dev/null || true
  case "$spec" in *e) EAGER=1; num="${spec%e}";; esac    # trailing 'e' -> run eager (no graph capture)
  if [ "$num" != off ]; then
    SPECARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$num}")
    CSZ=''   # spec decode batch pads to 1+spec -> compile_sizes [1] is rejected; omit
  fi
  local CAPARG=()
  if [ "$EAGER" = 1 ]; then
    CAPARG=(--enforce-eager)                               # eager: no XPU graph capture (dodges spec-capture crash)
  else
    local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
    local CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"cudagraph_capture_sizes\":[$CAPS],${CSZ}$PASS}"
    CAPARG=(--compilation-config "$CC")
  fi
  local ARGS=(serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT"
        --dtype auto --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS"
        --gpu-memory-utilization "$UTIL" --no-enable-prefix-caching --trust-remote-code
        --limit-mm-per-prompt '{"image":0,"video":0}'
        "${CAPARG[@]}" "${SPECARG[@]}")
  [ "$TP" -gt 1 ] && ARGS+=(--distributed-executor-backend mp)
  local GENV=(-e VLLM_XPU_ENABLE_XPU_GRAPH=1); [ "$EAGER" = 1 ] && GENV=()

  # GDN-enabled kernel + any sibling lib it dlopens
  local KSO=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
  # multi-GPU vs single-card env
  local MGPU SHM
  if [ "$TP" -gt 1 ]; then
    MGPU=(-e CCL_ENABLE_SYCL_KERNELS="${SYCLKERNELS:-1}" -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
          -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn
          -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd); SHM=32g
  else
    MGPU=(-e ZE_AFFINITY_MASK="${DEVICE:-0}"); SHM=16g
  fi

  echo "vllm ${ARGS[*]}" | tee -a "$SUMM"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "$SHM" -p ${PORT}:${PORT} \
    --pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
    "${KSO[@]}" "${EXTRA_MOUNTS[@]}" \
    -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
    -e OMP_NUM_THREADS=8 -e PYTHONPATH="$SHIM" "${GENV[@]}" \
    "${EXTRA_ENV[@]}" "${MGPU[@]}" --entrypoint vllm "$IMG" "${ARGS[@]}" >/dev/null
}

wait_healthy() {
  local i
  for i in $(seq 1 200); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1
    sleep 5
  done
  return 1
}

bench() {  # $1 = spec label
  local spec="$1"
  # confirm the MTP BF16 shim actually loaded
  docker logs "$NAME" 2>&1 | grep -iE "mtp-bf16-shim|Detected MTP|SpeculativeConfig|num_spec" | tail -4 | sed 's/^/    [log] /' | tee -a "$SUMM"
  gen_tok 8 >/dev/null   # warmup (compile/capture)
  local s0 s1 l0 l1 ns nl
  s0=$(date +%s.%N); ns=$(gen_tok 64);  s1=$(date +%s.%N)
  l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  local M A D DT
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  DT=$(echo "$M" | awk '/num_draft_tokens_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v sp="$spec" \
      -v A="$A" -v D="$D" -v DT="$DT" -v base="${BASETPS:-0}" -v csv="$CSV" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; mx=(base>0)?tps/base:0;
      al=(D>0)?(A/D)+1:0; ar=(DT>0)?A/DT:0;
      printf "spec=%-3s decode_tps=%6.2f  MTPx=%5.2f  accept_len=%.2f  accept_rate=%.3f  (acc=%d drafts=%d dtok=%d, gen512 %.2fs)\n",
        sp, tps, mx, al, ar, A, D, DT, (tl1-tl0);
      printf "%s,%.2f,%.2f,%.2f,%.3f,%d,%d,%d,%.2f\n", sp, tps, mx, al, ar, A, D, DT, (tl1-tl0) >> csv;
      print tps > "/tmp/mtp90_lasttps"}' | tee -a "$SUMM"
}

BASETPS=0
for SP in $SPECS; do
  echo ">>> $MODE  spec=$SP" | tee -a "$SUMM"
  serve "$SP"
  if wait_healthy; then
    bench "$SP"
    if [ "$SP" = off ]; then BASETPS=$(cat /tmp/mtp90_lasttps 2>/dev/null || echo 0); fi
  else
    echo "spec=$SP  SERVE-FAIL / CRASH -- log tail:" | tee -a "$SUMM"
    docker logs "$NAME" 2>&1 | grep -iE "error|spec_query_start_loc|allgather|allreduce|ccl|out of memory|work_group_scratch|Traceback|RuntimeError|mtp-bf16-shim|cannot allocate" | tail -25 | sed 's/^/    /' | tee -a "$SUMM"
    echo "$SP,FAIL,,,,,,," >> "$CSV"
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 5
done

echo "=== 90 $MODE SUMMARY ===" | tee -a "$SUMM"; cat "$SUMM"
echo "=== CSV: $CSV ===" ; cat "$CSV"
echo "=== 90 $MODE done ==="
