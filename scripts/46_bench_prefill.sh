#!/usr/bin/env bash
# Prefill / large-batch + decode bench against a running vLLM server. Captures, per (in:out:conc) profile:
# req/s, PREFILL tok/s (~ in_len*1000/mean_TTFT for prefill-dominated rows), output(decode) tok/s,
# total tok/s, mean TTFT, mean TPOT. Purpose: quantify where INT8 W8A8 beats FP8 (prefill/large-batch =
# compute-bound, native s8s8s32 systolic ~2x; decode = bandwidth-bound, ties). Env or positional:
#   NAME (container), MODEL (served name), TOKPATH (tokenizer), LABEL, CONFIGS="in:out:conc ...".
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
NAME="${1:-${NAME:-vllm_int8}}"; MODEL="${2:-${MODEL:-qwen3-14b-w8a8-gptq}}"; LABEL="${3:-${LABEL:-w8a8-gptq}}"
TOKPATH="${4:-${TOKPATH:-/mnt/vm_8tb/b70/models/Qwen3-14B-W8A8-gptq}}"
PORT=18080
CONFIGS="${CONFIGS:-512:128:1 4096:8:1 8192:8:1 2048:64:8 4096:8:8 512:128:32}"
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT_FILE="$ROOT/results/prefill_${LABEL}_${STAMP}.csv"

curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || { echo "server $NAME not healthy on :$PORT"; exit 1; }
echo "in,out,conc,req_s,prefill_tok_s,decode_out_tok_s,total_tok_s,ttft_ms,tpot_ms" | tee "$OUT_FILE"

for cfg in $CONFIGS; do
  IN="${cfg%%:*}"; rest="${cfg#*:}"; OUTL="${rest%%:*}"; C="${rest##*:}"
  N=$((C*6)); [ "$C" = 1 ] && N=6
  raw=$(docker exec -i "$NAME" vllm bench serve \
    --backend vllm --model "$MODEL" --tokenizer "$TOKPATH" --base-url "http://localhost:${PORT}" \
    --endpoint /v1/completions --dataset-name random \
    --random-input-len "$IN" --random-output-len "$OUTL" \
    --num-prompts "$N" --max-concurrency "$C" --ignore-eos 2>&1)
  reqs=$(echo "$raw" | grep -iE 'Request throughput'      | grep -oE '[0-9.]+' | head -1)
  otps=$(echo "$raw" | grep -iE 'Output token throughput' | grep -oE '[0-9.]+' | head -1)
  ttps=$(echo "$raw" | grep -iE 'Total Token throughput'  | grep -oE '[0-9.]+' | head -1)
  ttft=$(echo "$raw" | grep -iE 'Mean TTFT'               | grep -oE '[0-9.]+' | head -1)
  tpot=$(echo "$raw" | grep -iE 'Mean TPOT'               | grep -oE '[0-9.]+' | head -1)
  # prefill tok/s ~ in_len / TTFT (TTFT ~ prefill time when output is tiny); x concurrency for aggregate
  pf=$(awk -v i="$IN" -v t="$ttft" -v c="$C" 'BEGIN{ if(t>0) printf "%.1f", i*1000.0/t*c; else print "nan" }')
  echo "${IN},${OUTL},${C},${reqs:-NA},${pf},${otps:-NA},${ttps:-NA},${ttft:-NA},${tpot:-NA}" | tee -a "$OUT_FILE"
done
echo "=== saved $OUT_FILE ==="
