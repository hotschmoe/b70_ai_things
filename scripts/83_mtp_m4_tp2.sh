#!/usr/bin/env bash
# M4 -- TP=2 27B MTP (confirm old Lorbus TP2-MTP [NEG] is stale; measure allreduce tax vs single-card 55.28).
# v0230 TP=2 + CCL_ENABLE_SYCL_KERNELS=1 (capture-safe path, P2P_GPU H.5) + PIECEWISE. A=MTP-off, B=MTP-on spec=4.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
MODEL=/models/Lorbus_Qwen3.6-27B-int4-AutoRound; SERVED=qwen36-27b-int4; IMG=vllm-xpu-env:v0230; PORT=18080
SUMM="$ROOT/results/m4_tp2_${SERVED}.txt"; : > "$SUMM"
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, docstrings, and an example. Then explain the time complexity step by step in detail."
echo "=== M4 TP=2 27B MTP (single-card ref: MTP-off 30.84, MTP-on spec4 55.28) ===" | tee -a "$SUMM"
gen_tok() { curl -s --max-time 300 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
  | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }
bench() { local label="$1"; gen_tok 8 >/dev/null
  local s0 s1 l0 l1 ns nl; s0=$(date +%s.%N); ns=$(gen_tok 64); s1=$(date +%s.%N); l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  local M A D; M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}'); D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v lab="$label" -v A="$A" -v D="$D" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; al=(D>0)?(A/D)+1:0; printf "%-22s decode_tps=%6.2f  accept_len=%.2f  (gen %d tok %.2fs)\n", lab, tps, al, nl, (tl1-tl0)}' | tee -a "$SUMM"; }
serve() { docker rm -f vllm_m4 2>/dev/null || true
  env IMG="$IMG" MODEL="$MODEL" SERVED="$SERVED" TP=2 SYCLKERNELS=1 GRAPH=1 CGMODE=PIECEWISE DTYPE=auto UTIL=0.90 \
      MAXLEN=8192 MAXSEQS=8 CAPSIZES=1,2,4,8 NOMM=1 NAME=vllm_m4 "$@" bash ./30_serve_w4a8_graph.sh >/dev/null 2>&1; }
echo ">>> A: TP=2 MTP-OFF PIECEWISE" | tee -a "$SUMM"; serve
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then bench "A_TP2_MTPoff"; else echo "A SERVE-FAIL" | tee -a "$SUMM"; docker logs vllm_m4 2>&1 | grep -iE "error|ccl|oneccl|allreduce|traceback" | tail -15 | tee -a "$SUMM"; fi
docker stop vllm_m4 >/dev/null 2>&1; sleep 5
echo ">>> B: TP=2 MTP-ON spec=4" | tee -a "$SUMM"; serve COMPILESZ= SPEC='{"method":"mtp","num_speculative_tokens":4}'
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then bench "B_TP2_MTPon_s4"; else echo "B SERVE-FAIL -- crash:" | tee -a "$SUMM"; docker logs vllm_m4 2>&1 | grep -iE "error|spec|ccl|allreduce|traceback|RuntimeError" | tail -20 | tee -a "$SUMM"; fi
docker stop vllm_m4 >/dev/null 2>&1
echo "=== M4 SUMMARY ===" | tee -a "$SUMM"; cat "$SUMM"; echo "=== M4 done; $SUMM ==="
