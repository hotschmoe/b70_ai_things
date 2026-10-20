#!/usr/bin/env bash
# Short, bounded one-card MTP validation window. The serve owns the GPU lease.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT=/mnt/vm_8tb/b70/build/vllm-flashnext/x1
export B70_AGENT=X1 B70_GPU_LOCK_TIMEOUT=7200 B70_CPU_LOCK_TIMEOUT=7200
export CARD=${CARD:-0} PORT=18121 MTP_TOKENS=${MTP_TOKENS:-2}
export NAME=vllm-fn-x1-proxy-mtp${MTP_TOKENS}
export GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY ATTN=FLASH_ATTN
export CAPTURE_SIZES=1,2,3,4,6,8 MAX_SEQS=2 MAX_LEN=2048 MAX_BATCHED=1024
export GPU_MEM=0.90 B70_XPU_CG_RECLAIM=500 PREFIX_CACHE=0
export B70_GGUF_REF=0 B70_EAGER_QK=0 B70_PREFILL=0
export B70_GGUF_SO=$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.o3.so
PY=/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
RUN=(uv run --no-project --python "$PY" python)
mkdir -p "$OUT"
LOG=$OUT/proxy-mtp${MTP_TOKENS}.log
date -u
sha256sum "$B70_GGUF_SO"
"$REPO/bin/gpu-run" --card "$CARD" "$REPO/bin/xpu-health" --card "$CARD"
"$REPO/vllm/flashnext/harness/x1_serve_proxy.sh" > "$LOG" 2>&1 &
serve_pid=$!
(
  while kill -0 "$serve_pid" 2>/dev/null; do
    available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if ((available < 8388608)); then
      echo "ABORT MemAvailable=$available"
      docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
      break
    fi
    sleep 2
  done
) &
watchdog_pid=$!
cleanup() {
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
  docker stop -t 15 "$NAME" >/dev/null 2>&1 || true
  wait "$serve_pid" 2>/dev/null || true
}
trap cleanup EXIT
ready=0
for ((i=0;i<2600;i++)); do
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if ((available < 8388608)); then
    echo "ABORT MemAvailable=$available"
    exit 4
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/models-mtp${MTP_TOKENS}.json"; then
    ready=1
    break
  fi
  if ! kill -0 "$serve_pid" 2>/dev/null; then break; fi
  sleep 3
done
if ((ready)); then
  # Continue to latency diagnostics after a mismatch, recording the gate rc.
  extra=()
  if [[ "$MTP_TOKENS" != 0 ]]; then extra=(--baseline "$OUT/greedy-mtp0.json"); fi
  set +e
  "$REPO/bin/cpu-run" --quiet "${RUN[@]}" \
    "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" \
    --url "http://127.0.0.1:$PORT" --out "$OUT/greedy-mtp${MTP_TOKENS}.json" \
    "${extra[@]}"
  echo GREEDY_RC=$?
  "$REPO/bin/cpu-run" --quiet "${RUN[@]}" \
    "$REPO/vllm/flashnext/harness/bench_latency.py" \
    --base-url "http://127.0.0.1:$PORT" --model hotschmoe-dd \
    --alias "qwen3.6-35b-a3b-ud-iq3-xxs-mtp${MTP_TOKENS}" \
    --out "$OUT/bench-mtp${MTP_TOKENS}.json" --decode-tokens 96 \
    --skip-prefill --tag "X1-proxy-full-mtp${MTP_TOKENS}"
  echo BENCH_RC=$?
  set -e
else
  echo SERVE_NOT_READY
  tail -n 65 "$LOG"
fi
cleanup
trap - EXIT
"$REPO/bin/gpu-run" --card "$CARD" "$REPO/bin/xpu-health" --card "$CARD"
echo "X1_WINDOW_DONE ready=$ready"
test "$ready" = 1
