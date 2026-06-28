#!/usr/bin/env bash
# 134_mtp_cheap_flags.sh -- PERF PUSH lever 3: cheap CPU/scheduler launch-overhead flags on int4+MTP
# (no cuda-graph needed -- the only path left after torch.compile NO-GO). Single-stream decode is
# launch-bound; with --disable-overlap-schedule the CPU scheduler does NOT overlap the GPU worker, so
# every decode step pays scheduling latency. Test flags that amortize/hide that:
#   - num-continuous-decode-steps N : run N GPU decode steps per scheduler poll (amortize CPU overhead)
#   - overlap schedule ON           : remove --disable-overlap-schedule (hide CPU sched behind GPU) -- may
#                                     be rejected/garble with spec-decode; coherence-gated, reverts if so.
# A/B each vs the known baseline c1=15.31 t/s. Single card -> ./bin/gpu-run --card 0 bash scripts/134_*.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_mtp_flags; PORT=30000; SERVED=qwen36-27b-int4-mtp-nextn
CKPT=/models/Lorbus_Qwen3.6-27B-int4-mtp; TOK=/models/Qwen_Qwen3.6-27B
LOG="$REPO/sglang/mtp_cheap_flags.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# config label -> extra launch flags (overlap default OFF = --disable-overlap-schedule present)
declare -a LABELS=( "baseline" "contsteps2" "contsteps4" "overlap_on" )
declare -a FLAGS=(
  "--disable-overlap-schedule"
  "--disable-overlap-schedule --num-continuous-decode-steps 2"
  "--disable-overlap-schedule --num-continuous-decode-steps 4"
  ""   # overlap ON: omit --disable-overlap-schedule
)

run_cfg(){ local lbl="$1" extra="$2"
  say "================= CONFIG: $lbl  (extra: ${extra:-<overlap ON>}) ================="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_MTP=1 \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --page-size 64 --disable-radix-cache \
      --speculative-algorithm NEXTN --speculative-num-steps 7 --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 8 --speculative-draft-attention-backend triton --disable-cuda-graph \
      $extra --max-running-requests 4 --skip-server-warmup \
      --tp 1 --context-length 4096 --mem-fraction-static 0.92 --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

  local ok=0
  for i in $(seq 1 90); do
    docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "[$lbl] CONTAINER EXITED (flags rejected?)"; docker logs "$NAME" 2>&1|grep -iE "error|assert|not support|spec|overlap|exception"|tail -8|tee -a "$LOG"; return 0; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { say "[$lbl] not healthy; skip"; docker rm -f "$NAME">/dev/null 2>&1; return 0; }

  # coherence + warmup (JIT spec path)
  local g coh
  g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":80,\"temperature\":0}")
  coh=$(echo "$g"|python3 -c "import sys,json
from collections import Counter
try: t=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip()
except Exception: print('FAIL'); raise SystemExit
v='OK'
if len(t)>=16:
 c,n=Counter(t).most_common(1)[0]
 if n/len(t)>=0.6: v='GARBAGE'
print((v if t else 'EMPTY')+' :: '+repr(t[:70]))")
  say "[$lbl] coherence: $coh"
  case "$coh" in GARBAGE*|EMPTY*|FAIL*) say "[$lbl] FAILED coherence -> skip bench (this flag breaks GDN/MTP)"; docker rm -f "$NAME">/dev/null 2>&1; return 0;; esac

  # warm c1 bench (discard 1st)
  local bench raw t o ttft al
  bench(){ docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
    --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
    --random-input-len 2048 --random-output-len 128 --num-prompts $2 --max-concurrency $1 2>&1"; }
  bench 1 4 >/dev/null 2>&1 || true
  raw="$(bench 1 6)"
  t=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
  o=$(echo "$raw"|grep -i 'Output token throughput'|grep -oE '[0-9.]+'|head -1)
  ttft=$(echo "$raw"|grep -i 'Mean TTFT'|grep -oE '[0-9.]+'|head -1)
  al=$(echo "$raw"|grep -i 'Accept length'|grep -oE '[0-9.]+'|head -1)
  say "[$lbl] WARM c1: decode=$(awk -v x="$t" 'BEGIN{if(x>0)printf"%.2f",1000/x;else print"NA"}') t/s  agg=${o:-NA}  TTFT=${ttft:-NA}ms  accept=${al:-NA}"
  docker rm -f "$NAME" >/dev/null 2>&1
}

for i in "${!LABELS[@]}"; do run_cfg "${LABELS[$i]}" "${FLAGS[$i]}"; done
say "================= SUMMARY ================="
grep -E "WARM c1|coherence|EXITED|FAILED" "$LOG" | tee -a "$LOG.summary"
say "=== cheap-flags sweep DONE -> $LOG ==="
