#!/usr/bin/env bash
# perf_regime.sh -- the decode-perf TESTING REGIME (task #1). Uniform measurement against an
# ALREADY-RUNNING sglang serve, so every lever (MTP / eager / fusion / quant) is compared the same way.
# Separates serve-launch (per-config: scripts/122, serve_sglang.sh, ...) from MEASUREMENT (this script).
#
# Three gates, always in this order:
#   1. COHERENCE  -- a real single answer must not be "!!!!" (correctness before speed; numbers on garbage are fake)
#   2. WARM bench -- discard the 1st bench (B70 idle-downclocks; cold ~2x slow), then record warm c1 + c4
#   3. SOAK       -- windowed decode t/s over a long single stream -> catches DEGRADATION (graph/state
#                    accumulation: the 26->7 t/s failure an aggregate-TPOT bench hides) + re-checks coherence
#
#   usage: perf_regime.sh <container> <port> <served> <tokenizer> [label]
set -uo pipefail
NAME="${1:?container}"; PORT="${2:?port}"; SERVED="${3:?served}"; TOK="${4:?tokenizer}"; LABEL="${5:-$SERVED}"
REPO=/mnt/vm_8tb/github/b70_ai_things
say(){ echo "[regime $(date +%H:%M:%S)] $*"; }

curl -fsS -o /dev/null "http://localhost:$PORT/health" || { echo "serve not healthy on :$PORT"; exit 1; }

# --- 1. COHERENCE gate ---
g=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":80,\"temperature\":0}")
coh=$(echo "$g" | python3 -c "import sys,json;from collections import Counter
try:
 m=json.load(sys.stdin)['choices'][0]['message']
 # --reasoning-parser puts short answers entirely in reasoning_content (content empty) -- fall back
 t=((m.get('content') or '') or (m.get('reasoning_content') or '')).strip()
except Exception as e: print('PARSE_FAIL'); sys.exit()
v='OK'
if len(t)>=16:
 c,n=Counter(t).most_common(1)[0]
 if n/len(t)>=0.6: v='GARBAGE'
print(v if t else 'EMPTY', '::', repr(t[:90]))")
say "COHERENCE[$LABEL]: $coh"
case "$coh" in GARBAGE*|EMPTY*|PARSE_FAIL*) say "FAILED coherence gate -- skipping perf (numbers would be fake)"; exit 2;; esac

# --- 2. WARM bench (discard 1st) ---
bench(){ docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
  python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
  --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
  --random-input-len 2048 --random-output-len 128 --num-prompts $2 --max-concurrency $1 2>&1"; }
row(){ local raw="$2" t o; t=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
  o=$(echo "$raw"|grep -i 'Output token throughput'|grep -oE '[0-9.]+'|head -1)
  local ttft; ttft=$(echo "$raw"|grep -i 'Mean TTFT'|grep -oE '[0-9.]+'|head -1)
  say "WARM[$1] decode=$(awk -v t="$t" 'BEGIN{if(t>0)printf"%.2f",1000/t;else print"NA"}') t/s  agg_out=${o:-NA}  TTFT=${ttft:-NA}ms"; }
say "warming (discard)..."; bench 1 4 >/dev/null 2>&1 || true
row c1 "$(bench 1 6)"
row c4 "$(bench 4 16)"

# --- 3. SOAK (windowed decode + degradation + coherence) ---
say "soak (2000-tok single stream, 400-tok windows)..."
python3 "$REPO/sglang/soak_probe.py" "$PORT" "$SERVED" 2000 400 localhost 2>&1 | sed 's/^/[regime] /'
say "DONE[$LABEL]"
