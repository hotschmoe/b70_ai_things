#!/usr/bin/env bash
# M5 -- 35B-A3B int4 MoE + MTP captured (MTP_TODO M5, NOVEL: no community row has MoE+capture+MTP). Single-card
# on :v0230moe (the int4 MoE already captures ~56.8 t/s, avoids the int8-MoE dequant-linear capture blocker).
# A=MTP-off PIECEWISE (baseline), B=MTP-on PIECEWISE spec=4. TTFT-cancelled decode + /metrics accept.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
MODEL="${MODEL:-/models/Intel_Qwen3.6-35B-A3B-int4-AutoRound}"
SERVED="${SERVED:-qwen36-35b-a3b-int4}"
IMG="${IMG:-vllm-xpu-env:v0230moe}"
SPECTOK="${SPECTOK:-4}"; PORT=18080
SUMM="$ROOT/results/m5_moe_mtp_${SERVED}.txt"; : > "$SUMM"
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, docstrings, and an example. Then explain the time complexity step by step in detail."
echo "=== M5: 35B int4 MoE + MTP captured on $IMG (spec=$SPECTOK) ===" | tee -a "$SUMM"
gen_tok() { curl -s --max-time 300 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
  | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }
bench() { # $1=label ; uses global accept
  local label="$1"
  gen_tok 8 >/dev/null
  local s0 s1 l0 l1 ns nl
  s0=$(date +%s.%N); ns=$(gen_tok 64);  s1=$(date +%s.%N)
  l0=$(date +%s.%N); nl=$(gen_tok 512); l1=$(date +%s.%N)
  local M A D
  M=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "vllm:spec_decode" | grep -vE "^#")
  A=$(echo "$M" | awk '/num_accepted_tokens_total/{v=$NF} END{print v+0}')
  D=$(echo "$M" | awk '/num_drafts_total/{v=$NF} END{print v+0}')
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v lab="$label" -v A="$A" -v D="$D" \
    'BEGIN{dt=(tl1-tl0)-(te-ts); dn=nl-ns; tps=(dt>0)?dn/dt:0; al=(D>0)?(A/D)+1:0; printf "%-22s decode_tps=%6.2f  accept_len=%.2f  (gen %d tok %.2fs)\n", lab, tps, al, nl, (tl1-tl0)}' | tee -a "$SUMM"
}
serve() { docker rm -f vllm_m5 2>/dev/null || true
  env IMG="$IMG" MODEL="$MODEL" SERVED="$SERVED" GRAPH=1 CGMODE=PIECEWISE DTYPE=auto UTIL=0.90 KVDTYPE=fp8_e5m2 \
      MAXLEN=8192 MAXSEQS=8 CAPSIZES=1,2,4,8 NOMM=1 NAME=vllm_m5 "$@" bash ./30_serve_w4a8_graph.sh >/dev/null 2>&1; }

echo ">>> A: MoE MTP-OFF PIECEWISE" | tee -a "$SUMM"
serve
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then bench "A_MoE_MTPoff"; else echo "A SERVE-FAIL" | tee -a "$SUMM"; docker logs vllm_m5 2>&1 | grep -iE "error|traceback|moe|oom|assert" | tail -15 | tee -a "$SUMM"; fi
docker stop vllm_m5 >/dev/null 2>&1; sleep 5

echo ">>> B: MoE MTP-ON PIECEWISE spec=$SPECTOK" | tee -a "$SUMM"
serve COMPILESZ= SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECTOK}"
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then bench "B_MoE_MTPon_s$SPECTOK"; else echo "B SERVE-FAIL -- crash:" | tee -a "$SUMM"; docker logs vllm_m5 2>&1 | grep -iE "error|traceback|spec|mtp|moe|assert|RuntimeError" | tail -20 | tee -a "$SUMM"; fi
docker stop vllm_m5 >/dev/null 2>&1
echo "=== M5 SUMMARY ===" | tee -a "$SUMM"; cat "$SUMM"
echo "=== M5 done; $SUMM ==="
