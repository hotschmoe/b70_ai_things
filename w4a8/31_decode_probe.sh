#!/usr/bin/env bash
# Single-stream decode throughput probe against a running vLLM server on :18080.
# Sends fixed-length generations (ignore_eos -> exactly N tokens) and reports tok/s = completion_tokens
# / wall_elapsed. Short prompt so TTFT is ~1% of a 256-tok decode -> the number is decode-dominated and
# directly comparable eager-vs-PIECEWISE in the SAME harness (the metric that averages out microbench noise).
# Pure curl + bash (the Unraid host has no python3). Run on the host (curl to localhost) after HEALTHY.
#   Env: SERVED (qwen3-14b-w4a8-gptq), N (256 max_tokens), TRIALS (4), PORT (18080), PROMPT.
set -uo pipefail
PORT="${PORT:-18080}"; SERVED="${SERVED:-qwen3-14b-w4a8-gptq}"; N="${N:-256}"; TRIALS="${TRIALS:-4}"
PROMPT="${PROMPT:-Write a detailed technical explanation of how GPUs accelerate matrix multiplication.}"
URL="http://localhost:${PORT}/v1/completions"
req () {
  curl -s "$URL" -H 'Content-Type: application/json' -d "{
    \"model\":\"${SERVED}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${1},
    \"temperature\":0,\"ignore_eos\":true,\"stream\":false}"
}
echo "=== decode probe: SERVED=$SERVED N=$N TRIALS=$TRIALS ==="
echo "served models:"; curl -s "http://localhost:${PORT}/v1/models" | grep -oE '"id":"[^"]*"' | head -2
echo "warmup..."; req 16 >/dev/null 2>&1
best=0; sum=0; cnt=0
for t in $(seq 1 "$TRIALS"); do
  t0=$(date +%s.%N)
  resp=$(req "$N")
  t1=$(date +%s.%N)
  ct=$(printf '%s' "$resp" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*' | head -1)
  [ -z "$ct" ] && { echo "  trial $t: PARSE FAIL -> $(printf '%s' "$resp" | head -c 200)"; continue; }
  el=$(awk "BEGIN{print $t1-$t0}")
  ts=$(awk "BEGIN{printf \"%.2f\", $ct/$el}")
  echo "  trial $t: ${ct} tok in ${el}s -> ${ts} tok/s"
  sum=$(awk "BEGIN{print $sum+$ts}"); cnt=$((cnt+1))
  awk "BEGIN{exit !($ts>$best)}" && best=$ts
done
[ "$cnt" -gt 0 ] && echo "=== mean $(awk "BEGIN{printf \"%.2f\", $sum/$cnt}") tok/s | best ${best} tok/s over $cnt trials ==="
