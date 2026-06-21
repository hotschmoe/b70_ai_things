#!/usr/bin/env bash
# Concurrency sweep against a running vLLM server using vLLM's own bench tool.
# Produces a table: concurrency -> req/s, output tok/s (aggregate), mean TTFT, mean TPOT,
# per-stream decode (=1000/TPOT). Args: [container] [served_model] [label]
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
# positional args OR env overrides (runremote.ps1 passes KEY=VALUE as env, not positional)
NAME="${1:-${NAME:-vllm_qwen3}}"; MODEL="${2:-${MODEL:-qwen3-14b}}"; LABEL="${3:-${LABEL:-qwen3-14b}}"
TOKPATH="${4:-${TOKPATH:-/specula_models/Qwen3-14B}}"
PORT="${PORT:-18080}"; IN=512; OUT=128   # PORT override -> bench a 2nd replica (data-parallel) on 18081
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT_FILE="$ROOT/results/sweep_${LABEL}_${STAMP}.csv"

curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || { echo "server not healthy"; exit 1; }
echo "concurrency,req_s,out_tok_s,mean_ttft_ms,mean_tpot_ms,per_stream_decode_tok_s" | tee "$OUT_FILE"

for C in ${CONC:-1 4 8 16 32}; do
  N=$((C*8)); [ "$C" = 1 ] && N=8
  raw=$(docker exec -i "$NAME" vllm bench serve \
    --backend vllm --model "$MODEL" --tokenizer "$TOKPATH" --base-url "http://localhost:${PORT}" \
    --endpoint /v1/completions --dataset-name random \
    --random-input-len $IN --random-output-len $OUT \
    --num-prompts $N --max-concurrency $C --ignore-eos 2>&1)
  reqs=$(echo "$raw"  | grep -iE 'Request throughput'        | grep -oE '[0-9.]+' | head -1)
  otps=$(echo "$raw"  | grep -iE 'Output token throughput'   | grep -oE '[0-9.]+' | head -1)
  ttft=$(echo "$raw"  | grep -iE 'Mean TTFT'                 | grep -oE '[0-9.]+' | head -1)
  tpot=$(echo "$raw"  | grep -iE 'Mean TPOT'                 | grep -oE '[0-9.]+' | head -1)
  pst=$(awk -v t="$tpot" 'BEGIN{ if(t>0) printf "%.2f", 1000.0/t; else print "nan" }')
  echo "${C},${reqs:-NA},${otps:-NA},${ttft:-NA},${tpot:-NA},${pst}" | tee -a "$OUT_FILE"
done
echo "=== saved $OUT_FILE ==="
