#!/usr/bin/env bash
# Q8 validation: serve the Qwable W4A16 int4-AutoRound (daily-driver recipe, inc auto-detected) + gen probe +
# TTFT-cancelled decode bench. Run AFTER the full Q8 produces /models/Qwable-5-27B-Coder-int4-AutoRound.
# One gpu-run lease (serve+probe+stop). gsm8k/HumanEval+ is a separate eval-harness follow-up.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
MODEL=/models/Qwable-5-27B-Coder-int4-AutoRound; SERVED=qwable-27b-int4; PORT=18080
[ -d "$MODEL" ] || { echo "[!] $MODEL not produced yet -- run after Q8 full DONE"; exit 2; }
LOGF="$ROOT/results/q8_validate.log"; mkdir -p "$ROOT/results"
docker rm -f vllm_q8val 2>/dev/null || true
echo "=== serve $SERVED (v0230 GRAPH=1 NOMM=1, inc auto) ==="
env IMG=vllm-xpu-env:v0230 MODEL="$MODEL" SERVED="$SERVED" GRAPH=1 DTYPE=auto UTIL=0.92 NOMM=1 \
    MAXLEN=8192 MAXSEQS=8 CAPSIZES=1,2,4,8,16,32,64 NAME=vllm_q8val \
    bash ./30_serve_w4a8_graph.sh 2>&1 | tee "$LOGF"
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
  echo "=== HEALTHY -- coherence + decode bench ===" | tee -a "$LOGF"
  echo "--- coherence (greedy) ---" | tee -a "$LOGF"
  curl -s --max-time 90 http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"Write a Python function that merges two sorted lists into one sorted list, with a docstring.\",\"max_tokens\":160,\"temperature\":0}" | head -c 900 | tee -a "$LOGF"; echo
  echo "--- decode bench (TTFT-cancelled) ---" | tee -a "$LOGF"
  P="Write a detailed thread-safe LRU cache in Python with get/put and an example."
  gt(){ curl -s --max-time 200 http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"prompt\":\"$P\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }
  gt 8 >/dev/null; s0=$(date +%s.%N); ns=$(gt 64); s1=$(date +%s.%N); l0=$(date +%s.%N); nl=$(gt 320); l1=$(date +%s.%N)
  awk -v ts=$s0 -v te=$s1 -v tl0=$l0 -v tl1=$l1 -v ns="${ns:-0}" -v nl="${nl:-0}" 'BEGIN{dt=(tl1-tl0)-(te-ts);dn=nl-ns;printf "qwable-27b-int4 decode_tps=%.2f (target ~30.8 = Lorbus 27B int4)\n",(dt>0)?dn/dt:0}' | tee -a "$LOGF"
else echo "=== NOT HEALTHY ===" | tee -a "$LOGF"; docker logs vllm_q8val 2>&1 | grep -iE "error|traceback|inc|quant|XPUwNa16|dim" | tail -20 | tee -a "$LOGF"; fi
docker stop vllm_q8val 2>/dev/null || true
echo "=== q8 validate done; $LOGF ==="
