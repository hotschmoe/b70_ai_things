#!/usr/bin/env bash
# 92 -- single-card W4A8 27B MTP decode-rate levers (push past the scripts/90 winner: spec=5 = 42.03 t/s, 2.03x).
# Two hypotheses surfaced by the 90 run, tested as variants (all MTP spec=5 PIECEWISE, single card):
#   base    -- reproduce the 90 winner (dtype auto=bf16, caps 1,2,4).      [sanity re-baseline]
#   fp16    -- --dtype float16. The 90 log warned "int4_gemm_w4a8 produces float16 output, recommend --dtype
#              float16" -- bf16 forces a per-op convert; fp16 should remove it.
#   capspec -- caps include the spec-verify batch 1+spec (spec=5 -> 6): caps 1,2,4,6,8. The winner's 1,2,4
#              miss batch 6 -> the verify decode falls back to EAGER; capturing 6 should speed the verify.
#   combo   -- fp16 + capspec.
# Also runs ONE MTP-off baseline per dtype so each variant gets a real multiplier.
#
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash 92_w4a8_mtp_levers.sh           # all variants
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W4A8-sqgptq-prepacked-mtp-graft
SERVED=qwen36-27b-w4a8-sqgptq-mtp
SHIM=$MODEL/mtp_bf16_patch
PORT=18080; NAME=vllm_mtp92; SPEC=5
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
KP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py
SP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp92_w4a8levers_${TS}.txt"; : > "$SUMM"
CSV="$ROOT/results/mtp92_w4a8levers_${TS}.csv"; echo "variant,spec,dtype,caps,decode_tps,mtp_x,accept_len,gen512_s" > "$CSV"
VARIANTS="${*:-base fp16 capspec combo}"
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, docstrings, and an example. Then explain the time complexity step by step in detail."

echo "=== 92 W4A8 MTP levers (90 winner: spec5 bf16 caps1,2,4 = 42.03 t/s / off 20.74 = 2.03x) variants={$VARIANTS} ===" | tee -a "$SUMM"

gen_tok() { curl -s --max-time 360 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }

serve() {  # $1=dtype  $2=caps  $3=spec("off"|int)
  local DT="$1" CAPS="$2" spec="$3" SPECARG=() CSZ='"compile_sizes":[1],'
  docker rm -f "$NAME" 2>/dev/null || true
  if [ "$spec" != off ]; then SPECARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$spec}"); CSZ=''; fi
  local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  local CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"cudagraph_capture_sizes\":[$CAPS],${CSZ}$PASS}"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p ${PORT}:${PORT} \
    --pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
    -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
    -v "$ROOT/patches/xpu.py:$KP:ro" -v "$ROOT/patches/compressed_tensors_w4a8_int.py:$SP:ro" \
    -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
    -e OMP_NUM_THREADS=8 -e PYTHONPATH="$SHIM" -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e VLLM_W4A8_PREPACKED=1 -e ZE_AFFINITY_MASK="${DEVICE:-0}" \
    --entrypoint vllm "$IMG" serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
    --dtype "$DT" --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 4 --gpu-memory-utilization 0.97 \
    --no-enable-prefix-caching --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
    --compilation-config "$CC" "${SPECARG[@]}" >/dev/null
}
wait_healthy() { local i; for i in $(seq 1 170); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1; }
bench() {  # $1=variant $2=dtype $3=caps $4=spec $5=basetps
  local lab="$1" DT="$2" CAPS="$3" spec="$4" base="$5"
  gen_tok 8 >/dev/null
  local s0 s1 l0 l1 ns nl M A D
  s0=$(date +%s.%N); ns=$(gen_tok 64); s1=$(date +%s.%N); l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v lab="$lab" -v DT="$DT" \
      -v CAPS="$CAPS" -v spec="$spec" -v A="$A" -v D="$D" -v base="$base" -v csv="$CSV" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; mx=(base>0)?tps/base:0; al=(D>0)?(A/D)+1:0;
      printf "%-10s spec=%-3s dtype=%-7s caps=%-9s decode_tps=%6.2f  MTPx=%5.2f  accept_len=%.2f  (gen512 %.2fs)\n",
        lab, spec, DT, CAPS, tps, mx, al, (tl1-tl0);
      printf "%s,%s,%s,\"%s\",%.2f,%.2f,%.2f,%.2f\n", lab, spec, DT, CAPS, tps, mx, al, (tl1-tl0) >> csv;
      print tps > "/tmp/mtp92_tps"}' | tee -a "$SUMM"
}

# per-dtype off baseline (caps 1,2,4), cached
declare -A OFFTPS
off_baseline() { local DT="$1"; [ -n "${OFFTPS[$DT]:-}" ] && return
  echo ">>> off baseline dtype=$DT" | tee -a "$SUMM"
  serve "$DT" "1,2,4" off; if wait_healthy; then bench "off" "$DT" "1,2,4" off 0; OFFTPS[$DT]=$(cat /tmp/mtp92_tps); else echo "off($DT) FAIL" | tee -a "$SUMM"; OFFTPS[$DT]=0; fi
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 4; }

for V in $VARIANTS; do
  case "$V" in
    base)    DT=auto;    CAPS="1,2,4" ;;
    fp16)    DT=float16; CAPS="1,2,4" ;;
    capspec) DT=auto;    CAPS="1,2,4,6,8" ;;
    combo)   DT=float16; CAPS="1,2,4,6,8" ;;
    *) echo "unknown variant $V" | tee -a "$SUMM"; continue ;;
  esac
  off_baseline "$DT"
  echo ">>> variant $V (dtype=$DT caps=$CAPS spec=$SPEC)" | tee -a "$SUMM"
  serve "$DT" "$CAPS" "$SPEC"
  if wait_healthy; then bench "$V" "$DT" "$CAPS" "$SPEC" "${OFFTPS[$DT]:-0}"; else
    echo "variant $V FAIL:" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | grep -iE "error|out of memory|Traceback|capture" | tail -8 | sed 's/^/   /' | tee -a "$SUMM"; fi
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 4
done
echo "=== 92 SUMMARY ===" | tee -a "$SUMM"; grep -E "decode_tps|FAIL" "$SUMM"
echo "=== CSV $CSV ==="; cat "$CSV"
echo "=== 92 w4a8levers done ==="
