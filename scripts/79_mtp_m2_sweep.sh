#!/usr/bin/env bash
# M2 -- MTP spec-token sweep (MTP_TODO M2). Hold everything fixed, sweep num_speculative_tokens; per spec:
# serve MTP-on PIECEWISE, TTFT-cancelled decode bench, pull /metrics spec_decode counters (accept len + per-pos).
# WINNER = max decode_tps (not max accept). Baseline MTP-off = 30.84 t/s (M1).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
MODEL="${MODEL:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"
SERVED="${SERVED:-qwen36-27b-int4}"
IMG="${IMG:-vllm-xpu-env:v0230}"
SPECS="${SPECS:-2 3 4 5 6}"
BASE="${BASE:-30.84}"
PORT=18080
SUMM="$ROOT/results/m2_specsweep_${SERVED}.txt"
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, including docstrings and an example usage. Then explain the time complexity and walk through an example step by step in detail."
: > "$SUMM"
echo "=== M2 spec sweep: $SERVED on $IMG  (MTP-off baseline=$BASE t/s from M1) ===" | tee -a "$SUMM"
gen_tok() { curl -s --max-time 300 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }
for SP in $SPECS; do
  echo ">>> spec=$SP" | tee -a "$SUMM"
  docker rm -f vllm_m2 2>/dev/null || true
  env IMG="$IMG" MODEL="$MODEL" SERVED="$SERVED" GRAPH=1 CGMODE=PIECEWISE DTYPE=auto UTIL=0.92 MAXLEN=8192 MAXSEQS=8 \
      CAPSIZES=1,2,4,8 COMPILESZ= NOMM=1 NAME=vllm_m2 \
      SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$SP}" bash ./30_serve_w4a8_graph.sh >/dev/null 2>&1
  if ! curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then echo "spec=$SP SERVE-FAIL" | tee -a "$SUMM"; docker stop vllm_m2 >/dev/null 2>&1; continue; fi
  gen_tok 8 >/dev/null   # warmup
  s0=$(date +%s.%N); ns=$(gen_tok 64);  s1=$(date +%s.%N)
  l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  DT=$(echo "$M" | awk '/num_draft_tokens_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v sp="$SP" -v A="$A" -v D="$D" -v DT="$DT" -v base="$BASE" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; mx=(base>0)?tps/base:0; al=(D>0)?(A/D)+1:0; ar=(DT>0)?A/DT:0;
     printf "spec=%s  decode_tps=%6.2f  MTPx=%.2f  accept_len=%.2f  accept_rate=%.3f  (accepted=%d drafts=%d draft_tok=%d, gen %d tok %.2fs)\n", sp, tps, mx, al, ar, A, D, DT, nl, (tl1-tl0)}' | tee -a "$SUMM"
  echo "    per-position accept (if present):" | tee -a "$SUMM"
  echo "$M" | grep -i "per_pos\|per_position" | sed 's/^/      /' | tee -a "$SUMM"
  docker stop vllm_m2 >/dev/null 2>&1 || true; sleep 4
done
echo "=== M2 SUMMARY ===" | tee -a "$SUMM"; cat "$SUMM"
echo "=== M2 done; $SUMM ==="
