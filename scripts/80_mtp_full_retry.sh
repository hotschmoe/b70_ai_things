#!/usr/bin/env bash
# MTP FULL-capture frontier RETRY (codex recipe): FULL_DECODE_ONLY + TRITON_ATTN + capture sizes that INCLUDE
# the spec-verify len 1+spec (M1-C omitted 6 -> crashed). If it serves, decode should beat PIECEWISE 52.95.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
MODEL="${MODEL:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"
SERVED="${SERVED:-qwen36-27b-int4}"
IMG="${IMG:-vllm-xpu-env:v0230}"
SPECTOK="${SPECTOK:-5}"
CGM="${CGM:-FULL_DECODE_ONLY}"
CAPS="${CAPS:-1,2,4,6,8,16,32}"   # MUST include 1+spec (=6 for spec=5)
PORT=18080
SUMM="$ROOT/results/mtp_fullretry_${SERVED}_${CGM}_s${SPECTOK}.txt"
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, docstrings, and an example. Then explain the time complexity step by step."
: > "$SUMM"
echo "=== FULL retry: $CGM spec=$SPECTOK caps=$CAPS (baseline MTP-off 30.84, PIECEWISE-MTP 52.95) ===" | tee -a "$SUMM"
docker rm -f vllm_full 2>/dev/null || true
env IMG="$IMG" MODEL="$MODEL" SERVED="$SERVED" GRAPH=1 CGMODE="$CGM" ATTN=TRITON_ATTN TRITONSHIM=1 \
    DTYPE=auto UTIL=0.92 MAXLEN=8192 MAXSEQS=8 CAPSIZES="$CAPS" COMPILESZ= NOMM=1 NAME=vllm_full \
    SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECTOK}" bash ./30_serve_w4a8_graph.sh 2>&1 | tail -25 | tee -a "$SUMM"
gen_tok() { curl -s --max-time 300 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
  echo "=== FULL_DECODE_ONLY MTP SERVES! Triton/capture confirmation ===" | tee -a "$SUMM"
  docker logs vllm_full 2>&1 | grep -iE "Using Triton|Disabling Triton|decode, FULL|FULL_DECODE|Capturing|cudagraph|TRITON_ATTN" | tail -10 | tee -a "$SUMM"
  gen_tok 8 >/dev/null
  s0=$(date +%s.%N); ns=$(gen_tok 64);  s1=$(date +%s.%N)
  l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v A="$A" -v D="$D" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; al=(D>0)?(A/D)+1:0; printf "FULL_DECODE_ONLY MTP: decode_tps=%.2f  MTPx=%.2f (vs30.84)  vs_PIECEWISE=%.2f  accept_len=%.2f\n", tps, tps/30.84, tps/52.95, al}' | tee -a "$SUMM"
else
  echo "=== FULL retry STILL FAILS -- crash signature ===" | tee -a "$SUMM"
  docker logs vllm_full 2>&1 | grep -iE "error|spec_query_start_loc|num_spec|work_group_scratch|assert|RuntimeError|Triton" | tail -25 | tee -a "$SUMM"
fi
docker stop vllm_full 2>/dev/null || true
echo "=== full retry done; $SUMM ==="
