#!/usr/bin/env bash
# 136_mtp_maxreq_memfrac.sh -- PERF PUSH lever 3b retry: the MAXREQ wall is the SPEC MAMBA CACHE, which is
# reserved OUTSIDE --mem-fraction-static ("draft weights are now counted"; even ctx2048/MAXREQ6 OOMed at
# memfrac 0.92). So LOWER memfrac to leave room for a bigger spec cache -> maybe MAXREQ>4 fits -> higher
# per-card aggregate (x2 via DP=2). Sweep (ctx, MAXREQ, memfrac). Single card.
#   ./bin/gpu-run --card 0 bash scripts/136_mtp_maxreq_memfrac.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_mtp_mf; PORT=30000; SERVED=qwen36-27b-int4-mtp-nextn
CKPT=/models/Lorbus_Qwen3.6-27B-int4-mtp; TOK=/models/Qwen_Qwen3.6-27B
LOG="$REPO/sglang/mtp_maxreq_memfrac.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# (ctx, maxreq, memfrac) -- lower memfrac frees room for the out-of-static spec mamba cache.
declare -a CFG=( "2048 8 0.80" "2048 8 0.70" "2048 12 0.65" "2048 6 0.85" )

run_cfg(){ local ctx="$1" mreq="$2" mf="$3"
  say "================= ctx=$ctx MAXREQ=$mreq memfrac=$mf ================="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_MTP=1 \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
      --speculative-algorithm NEXTN --speculative-num-steps 7 --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 8 --speculative-draft-attention-backend triton --disable-cuda-graph \
      --max-running-requests $mreq --skip-server-warmup \
      --tp 1 --context-length $ctx --mem-fraction-static $mf --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

  local ok=0
  for i in $(seq 1 90); do
    docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "[c$ctx/m$mreq/mf$mf] EXITED"; docker logs "$NAME" 2>&1|grep -iE "no gpu memory|leave no|out of memory|mamba|kv cache|ValueError"|tail -3|tee -a "$LOG"; return 0; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { say "[c$ctx/m$mreq/mf$mf] not healthy"; docker rm -f "$NAME">/dev/null 2>&1; return 0; }
  say "[c$ctx/m$mreq/mf$mf] *** HEALTHY at MAXREQ=$mreq (fits!) ***"

  curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi in one word.\"}],\"max_tokens\":16,\"temperature\":0}" >/dev/null 2>&1 || true
  local g coh
  g=$(curl -s --max-time 120 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
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
  say "[c$ctx/m$mreq/mf$mf] coherence: $coh"
  case "$coh" in GARBAGE*|EMPTY*|FAIL*) say "[c$ctx/m$mreq/mf$mf] FAILED coherence"; docker rm -f "$NAME">/dev/null 2>&1; return 0;; esac

  local bench raw o t ttft al
  bench(){ docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
    --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
    --random-input-len 2048 --random-output-len 128 --num-prompts $2 --max-concurrency $1 2>&1"; }
  bench "$mreq" "$mreq" >/dev/null 2>&1 || true
  raw="$(bench "$mreq" $((mreq*3)))"
  o=$(echo "$raw"|grep -i 'Output token throughput'|grep -oE '[0-9.]+'|head -1)
  t=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
  ttft=$(echo "$raw"|grep -i 'Mean TTFT'|grep -oE '[0-9.]+'|head -1)
  al=$(echo "$raw"|grep -i 'Accept length'|grep -oE '[0-9.]+'|head -1)
  say "[c$ctx/m$mreq/mf$mf] MC$mreq AGG=${o:-NA} tok/s | perstream=$(awk -v x="$t" 'BEGIN{if(x>0)printf"%.2f",1000/x;else print"NA"}') t/s | TTFT=${ttft:-NA}ms | accept=${al:-NA}"
  docker rm -f "$NAME" >/dev/null 2>&1
}
for c in "${CFG[@]}"; do run_cfg $c; done
say "================= SUMMARY ================="; grep -E "AGG=|HEALTHY|EXITED|FAILED" "$LOG"
say "=== DONE -> $LOG ==="
