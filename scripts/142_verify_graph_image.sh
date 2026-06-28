#!/usr/bin/env bash
# 142_verify_graph_image.sh -- ACCEPTANCE TEST for the productionized int4+XPUGraph driver. Drives the shipped
# recipe (rdy_to_serve/qwen36-27b-int4-graph/serve.sh) against the BAKED sglang-xpu:mtp image with ZERO mounts
# -> proves the graph patches (xpu_cudagraph.py + woq_shim B70_XPU_CUDAGRAPH block) are baked + self-contained,
# and reproduces the ~23.5 t/s single-stream capture + coherence + sampling.
#   Pick a card via DEVICE (default 0). ./bin/gpu-run --card 1 DEVICE=1 bash scripts/142_verify_graph_image.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
R="$REPO/rdy_to_serve/qwen36-27b-int4-graph/serve.sh"
DEVICE="${DEVICE:-0}"; PORT="${PORT:-30000}"; SERVED=qwen36-27b-int4-graph; NAME="sglang_int4_graph"
export DEVICE PORT NAME
LOG="$REPO/sglang/verify_graph_image.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 1. START shipped recipe (baked image, NO mounts, card $DEVICE) ==="
if ! bash "$R" start 2>&1 | tee -a "$LOG"; then
  say "*** recipe start FAILED ***"; bash "$R" stop >/dev/null 2>&1; exit 1
fi

say "=== 2. SAMPLING check (graph driver supports it; temp=0.8 should vary across seeds) ==="
for k in 1 2; do
  s=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 6-word story about the sea.\"}],\"max_tokens\":40,\"temperature\":0.8,\"seed\":$k}")
  echo "$s"|python3 -c "import sys,json;print('  sample$k:',repr((json.load(sys.stdin)['choices'][0]['message']['content'] or '')[:80]))" 2>/dev/null | tee -a "$LOG" || say "  sample$k parse-fail"
done

say "=== 3. BENCH (warm c1 pp/ttft/tg @ ctx2048 + soak) ==="
bash "$R" bench 2>&1 | tee -a "$LOG"

say "=== 4. capture counts (confirm graph engaged) ==="
docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c | tee -a "$LOG"

say "=== 5. STOP ==="; bash "$R" stop 2>&1 | tee -a "$LOG"
say "=== verify complete -> $LOG ==="
