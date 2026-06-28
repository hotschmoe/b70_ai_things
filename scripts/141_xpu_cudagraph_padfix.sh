#!/usr/bin/env bash
# 141_xpu_cudagraph_padfix.sh -- find a graph config that gives BOTH ~23 single-stream AND concurrency.
# scripts/140 showed bs[1 2 4]/maxreq=4 tanks c1 to 11.88 (a single decode pads UP to the bs=4 graph).
# Test --disable-cuda-graph-padding (single req uses the bs=1 graph, not padded). A=multi-bucket+nopad,
# B=bs1/maxreq1 reference (reproduce 139's 23). C=multi-bucket WITH pad (re-confirm the 140 regression).
#   ./bin/gpu-run --card 0 bash scripts/141_xpu_cudagraph_padfix.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_xpucg; PORT=30000; SERVED=qwen36-27b-int4-xpucg
CKPT=/models/Lorbus_Qwen3.6-27B-int4-AutoRound; TOK=/models/Qwen_Qwen3.6-27B
SP=/opt/venv/lib/python3.12/site-packages
LOG="$REPO/sglang/xpu_cudagraph_padfix.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# label | bs-decode list | max-bs | maxreq | extra
declare -a CFG=(
  "A_multibucket_nopad|1 2 4|4|4|--disable-cuda-graph-padding"
  "B_bs1_ref|1|1|1|"
  "C_multibucket_pad|1 2 4|4|4|"
)

run_cfg(){ local lbl="$1" bslist="$2" bsmax="$3" mreq="$4" extra="$5"
  say "================= $lbl (bs[$bslist] max=$bsmax maxreq=$mreq extra='${extra:-none}') ================="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$REPO/sglang/patches/woq_shim.py:$SP/woq_shim.py:ro" \
    -v "$REPO/sglang/patches/xpu_cudagraph.py:$SP/xpu_cudagraph.py:ro" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_CUDAGRAPH=1 \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend triton --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
      --cuda-graph-bs-decode $bslist --cuda-graph-max-bs-decode $bsmax $extra \
      --max-running-requests $mreq \
      --tp 1 --context-length 4096 --mem-fraction-static 0.90 --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

  local ok=0
  for i in $(seq 1 150); do
    docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "[$lbl] EXITED"; docker logs "$NAME" 2>&1|grep -iE "error|scratch|capture|assert|Traceback"|tail -8|tee -a "$LOG"; return 0; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { say "[$lbl] not healthy"; docker rm -f "$NAME">/dev/null 2>&1; return 0; }

  local g coh
  g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":64,\"temperature\":0}")
  coh=$(echo "$g"|python3 -c "import sys,json
from collections import Counter
try: t=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip()
except Exception: print('FAIL'); raise SystemExit
v='OK'
if len(t)>=16:
 c,n=Counter(t).most_common(1)[0]
 if n/len(t)>=0.6: v='GARBAGE'
print(v if t else 'EMPTY')")
  say "[$lbl] coherence: $coh"
  case "$coh" in GARBAGE*|EMPTY*|FAIL*) say "[$lbl] FAILED coherence"; docker rm -f "$NAME">/dev/null 2>&1; return 0;; esac

  local bench raw t o ttft
  bench(){ docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
    --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
    --random-input-len 2048 --random-output-len 128 --num-prompts $2 --max-concurrency $1 2>&1"; }
  row(){ local raw="$2" t o ttft; t=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
    o=$(echo "$raw"|grep -i 'Output token throughput'|grep -oE '[0-9.]+'|head -1)
    ttft=$(echo "$raw"|grep -i 'Mean TTFT'|grep -oE '[0-9.]+'|head -1)
    say "[$lbl] $1 decode=$(awk -v x="$t" 'BEGIN{if(x>0)printf"%.2f",1000/x;else print"NA"}') t/s agg=${o:-NA} TTFT=${ttft:-NA}ms"; }
  bench 1 3 >/dev/null 2>&1 || true   # warm
  row c1 "$(bench 1 6)"
  [ "$mreq" -gt 1 ] && row c4 "$(bench 4 16)"
  docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c | sed "s/^/[$lbl] /" | tee -a "$LOG"
  docker rm -f "$NAME" >/dev/null 2>&1
}
for c in "${CFG[@]}"; do IFS='|' read -r l b m r e <<< "$c"; run_cfg "$l" "$b" "$m" "$r" "$e"; done
say "================= SUMMARY ================="; grep -E "decode=|coherence|EXITED" "$LOG" | tee -a "$LOG.summary"
say "=== padfix sweep DONE -> $LOG ==="
