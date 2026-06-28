#!/usr/bin/env bash
# 135_mtp_concurrency_ceiling.sh -- PERF PUSH lever 3b: find the per-card max-running-requests ceiling.
# MTP's spec mamba INTERMEDIATE-STATE cache scales with --max-running-requests; MAXREQ=8 OOMed the KV at
# ctx=4096 -> the driver caps at 4. But at LOWER ctx the KV pool needs less, freeing memory for more spec
# cache -> more in-flight streams -> higher per-card AGGREGATE (which then ~2x's through DP=2). Sweep
# (ctx, MAXREQ); for each, serve, coherence-gate, bench at MC=MAXREQ (saturate), record aggregate tok/s.
# Single card -> ./bin/gpu-run --card 0 bash scripts/135_mtp_concurrency_ceiling.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_mtp_cc; PORT=30000; SERVED=qwen36-27b-int4-mtp-nextn
CKPT=/models/Lorbus_Qwen3.6-27B-int4-mtp; TOK=/models/Qwen_Qwen3.6-27B
LOG="$REPO/sglang/mtp_concurrency_ceiling.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# (ctx, maxreq) configs -- push concurrency at lower ctx where the KV pool frees memory for the spec cache.
declare -a CFG=( "2048 6" "2048 8" "2048 12" "2048 16" "4096 6" )

run_cfg(){ local ctx="$1" mreq="$2"
  say "================= ctx=$ctx MAXREQ=$mreq ================="
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
      --tp 1 --context-length $ctx --mem-fraction-static 0.92 --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

  local ok=0
  for i in $(seq 1 90); do
    docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "[ctx$ctx/m$mreq] CONTAINER EXITED (likely KV OOM)"; docker logs "$NAME" 2>&1|grep -iE "no gpu memory|out of memory|oom|kv cache|leave no"|tail -4|tee -a "$LOG"; return 0; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { say "[ctx$ctx/m$mreq] not healthy"; docker logs "$NAME" 2>&1|grep -iE "memory|oom|error"|tail -4|tee -a "$LOG"; docker rm -f "$NAME">/dev/null 2>&1; return 0; }

  # coherence + JIT warmup
  curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello in one word.\"}],\"max_tokens\":16,\"temperature\":0}" >/dev/null 2>&1 || true
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
print((v if t else 'EMPTY'))")
  say "[ctx$ctx/m$mreq] coherence: $coh"
  case "$coh" in GARBAGE*|EMPTY*|FAIL*) say "[ctx$ctx/m$mreq] FAILED coherence -> skip"; docker rm -f "$NAME">/dev/null 2>&1; return 0;; esac

  local bench raw o t ttft al
  bench(){ docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
    --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
    --random-input-len 2048 --random-output-len 128 --num-prompts $2 --max-concurrency $1 2>&1"; }
  bench "$mreq" "$mreq" >/dev/null 2>&1 || true   # warm/discard
  raw="$(bench "$mreq" $((mreq*4)))"
  o=$(echo "$raw"|grep -i 'Output token throughput'|grep -oE '[0-9.]+'|head -1)
  t=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
  ttft=$(echo "$raw"|grep -i 'Mean TTFT'|grep -oE '[0-9.]+'|head -1)
  al=$(echo "$raw"|grep -i 'Accept length'|grep -oE '[0-9.]+'|head -1)
  say "[ctx$ctx/m$mreq] MC$mreq AGG=${o:-NA} tok/s | perstream=$(awk -v x="$t" 'BEGIN{if(x>0)printf"%.2f",1000/x;else print"NA"}') t/s | TTFT=${ttft:-NA}ms | accept=${al:-NA}"
  docker rm -f "$NAME" >/dev/null 2>&1
}

for c in "${CFG[@]}"; do run_cfg $c; done
say "================= SUMMARY (per-card aggregate ceiling) ================="
grep -E "AGG=|EXITED|FAILED|coherence" "$LOG"
say "=== concurrency-ceiling sweep DONE -> $LOG ==="
