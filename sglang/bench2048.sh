#!/usr/bin/env bash
# bench2048.sh -- standard PP/TTFT/TG bench for an SGLang serve at ~2048 ctx, matching the
# vLLM scripts/121 methodology (random IN=2048 OUT=128, concurrency 1 and 4, ignore-eos default).
# Reports per the repo convention:
#   PP (prefill tok/s) = IN*1000/TTFT_ms ; TTFT (ms) ; TG/decode tok/s = 1000/TPOT_ms
# Usage: bash sglang/bench2048.sh [NAME] [PORT] [SERVED] [TOK] [IN] [OUT] [CONC...]
#   NAME=container (default sglang_test), runs sglang.bench_serving INSIDE it.
set -uo pipefail
NAME="${1:-sglang_test}"; PORT="${2:-30000}"; SERVED="${3:-qwen36-27b-bf16-sglang}"
TOK="${4:-/models/Qwen_Qwen3.6-27B}"; IN="${5:-2048}"; OUT="${6:-128}"; shift 6 2>/dev/null || true
CONC=("$@"); [ "${#CONC[@]}" -eq 0 ] && CONC=(1 4)

one() { # conc num
  docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
    --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
    --random-input-len $IN --random-output-len $OUT --num-prompts $2 --max-concurrency $1 2>&1"
}

printf 'conc\treq_s\tout_tps\tttft_ms\ttpot_ms\tdecode_tps\tprefill_tps\n'
for C in "${CONC[@]}"; do
  N=$((C*4)); [ "$C" = 1 ] && N=6
  raw="$(one "$C" "$N")"
  reqs=$(echo "$raw" | grep -i 'Request throughput'      | grep -oE '[0-9.]+' | head -1)
  otps=$(echo "$raw" | grep -i 'Output token throughput' | grep -oE '[0-9.]+' | head -1)
  ttft=$(echo "$raw" | grep -i 'Mean TTFT'               | grep -oE '[0-9.]+' | head -1)
  tpot=$(echo "$raw" | grep -i 'Mean TPOT'               | grep -oE '[0-9.]+' | head -1)
  dec=$(awk -v t="$tpot" 'BEGIN{ if(t>0) printf "%.2f",1000.0/t; else print "NA" }')
  pp=$(awk -v t="$ttft" -v i="$IN" 'BEGIN{ if(t>0) printf "%.0f",i*1000.0/t; else print "NA" }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$C" "${reqs:-NA}" "${otps:-NA}" "${ttft:-NA}" "${tpot:-NA}" "$dec" "$pp"
done
