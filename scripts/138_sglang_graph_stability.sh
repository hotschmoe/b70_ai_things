#!/usr/bin/env bash
# 138_sglang_graph_stability.sh -- FRONTIER decisive test: now that torch.xpu.XPUGraph is proven STABLE
# (scripts/137), is sglang's own XPU cuda_graph path (which gave 9.4->23.6 t/s server-side before being
# abandoned for DEGRADATION) now STABLE on this torch-2.12 image -- and what is the CLIENT-side decode (the
# old run saw server 23 but client only 12.57 due to detok/stream overhead)? Tests int4 NO-MTP (simplest;
# a stable graph win here would also restore SAMPLING + lift the MTP MAXREQ=4 cap). The perf_regime soak is
# the degradation test (windowed decode t/s, first/last ratio). Single card.
#   ./bin/gpu-run --card 0 bash scripts/138_sglang_graph_stability.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_graph; PORT=30000; SERVED=qwen36-27b-int4-graph
CKPT=/models/Lorbus_Qwen3.6-27B-int4-AutoRound; TOK=/models/Qwen_Qwen3.6-27B
LOG="$REPO/sglang/sglang_graph_stability.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# label -> extra flags. graph_off = eager baseline (~9.4). graph_on = remove --disable-cuda-graph (default
# decode backend). graph_si8 = graph + stream-interval 8 (close the detok/stream client gap the journal found).
declare -a LABELS=( "graph_off" "graph_on" "graph_on_si8" )
declare -a FLAGS=(
  "--disable-cuda-graph --disable-overlap-schedule"
  "--disable-overlap-schedule"
  "--disable-overlap-schedule --stream-interval 8"
)

run_cfg(){ local lbl="$1" extra="$2"
  say "================= CONFIG: $lbl  (extra: $extra) ================="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --page-size 64 --disable-radix-cache \
      $extra --max-running-requests 8 --skip-server-warmup \
      --tp 1 --context-length 4096 --mem-fraction-static 0.90 --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

  local ok=0
  for i in $(seq 1 120); do
    docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "[$lbl] CONTAINER EXITED (graph capture crash?)"; docker logs "$NAME" 2>&1|grep -iE "error|assert|graph|capture|sycl|level.zero|runtime|exception"|tail -10|tee -a "$LOG"; return 0; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "[$lbl] /health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { say "[$lbl] not healthy; skip"; docker logs "$NAME" 2>&1|tail -15|tee -a "$LOG"; docker rm -f "$NAME">/dev/null 2>&1; return 0; }

  # coherence + graph-capture warmup (first decode of each captured bs triggers capture)
  local g coh
  g=$(curl -s --max-time 240 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
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
  case "$coh" in GARBAGE*|EMPTY*|FAIL*) say "[$lbl] FAILED coherence -> skip"; docker rm -f "$NAME">/dev/null 2>&1; return 0;; esac

  # full regime: warm c1/c4 + SOAK (the degradation test)
  bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "$lbl" 2>&1 | tee -a "$LOG"
  # confirm graph engaged
  docker logs "$NAME" 2>&1 | grep -iE "capture|cuda graph|graph.*decode|XPUGraph|replay" | tail -4 | tee -a "$LOG" || true
  docker rm -f "$NAME" >/dev/null 2>&1
}

for i in "${!LABELS[@]}"; do run_cfg "${LABELS[$i]}" "${FLAGS[$i]}"; done
say "================= SUMMARY ================="; grep -E "WARM|SOAK|OVERALL|coherence|EXITED|FAILED" "$LOG" | tee -a "$LOG.summary"
say "=== sglang graph stability test DONE -> $LOG ==="
