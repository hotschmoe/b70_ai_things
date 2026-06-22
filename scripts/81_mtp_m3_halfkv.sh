#!/usr/bin/env bash
# Combined lease: (1) FULL_DECODE_ONLY frontier retry at M2 winner spec=4, (2) M3 Half-KV accept isolation.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
SERVED=qwen36-27b-int4; MODEL=/models/Lorbus_Qwen3.6-27B-int4-AutoRound; IMG=vllm-xpu-env:v0230; PORT=18080
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, docstrings, and an example. Then explain the time complexity step by step in detail."
gen_tok() { curl -s --max-time 300 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
  | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }

echo "########## PART 1: FULL_DECODE_ONLY retry (spec=4, caps incl 5) ##########"
SPECTOK=4 CAPS=1,2,4,5,8,16,32 CGM=FULL_DECODE_ONLY bash ./mtp_full_retry.sh || echo "(full retry returned nonzero)"

echo "########## PART 2: M3 Half-KV (PIECEWISE spec=4 + fp8 KV) vs full-KV accept 3.25 ##########"
SUMM="$ROOT/results/m3_halfkv_${SERVED}.txt"; : > "$SUMM"
docker rm -f vllm_m3 2>/dev/null || true
env IMG="$IMG" MODEL="$MODEL" SERVED="$SERVED" GRAPH=1 CGMODE=PIECEWISE DTYPE=auto UTIL=0.92 MAXLEN=8192 MAXSEQS=8 \
    CAPSIZES=1,2,4,8 COMPILESZ= NOMM=1 NAME=vllm_m3 KVDTYPE=fp8_e4m3 \
    SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":4}" bash ./30_serve_w4a8_graph.sh >/dev/null 2>&1
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
  gen_tok 8 >/dev/null
  s0=$(date +%s.%N); ns=$(gen_tok 64);  s1=$(date +%s.%N)
  l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v A="$A" -v D="$D" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; al=(D>0)?(A/D)+1:0; full=3.25;
     printf "M3 Half-KV(fp8_e4m3) spec=4: decode_tps=%.2f  accept_len=%.2f  vs_full_KV_accept=3.25  delta=%+.2f  verdict=%s\n", tps, al, al-full, (full-al>0.2)?"HALF-KV COSTS ACCEPT":"Half-KV OK (keep)"}' | tee -a "$SUMM"
else
  echo "M3 Half-KV SERVE-FAIL" | tee -a "$SUMM"; docker logs vllm_m3 2>&1 | grep -iE "error|fp8|kv|assert" | tail -15 | tee -a "$SUMM"
fi
docker stop vllm_m3 2>/dev/null || true
echo "=== m3_and_full done; FULL summary in results/mtp_fullretry_*, M3 in $SUMM ==="
