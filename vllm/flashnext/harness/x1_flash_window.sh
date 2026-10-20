#!/usr/bin/env bash
# Bounded Flash-Next on/off diagnostic, TP=1 DP=2 EP, both GPU leases.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT=${X1_OUT:-/mnt/vm_8tb/b70/build/vllm-flashnext/x1}
export B70_AGENT=X1 B70_GPU_LOCK_TIMEOUT=7200 B70_CPU_LOCK_TIMEOUT=7200
export PORT=18121 MTP_TOKENS=${MTP_TOKENS:-2}
export NAME=vllm-fn-x1-flashnext-mtp${MTP_TOKENS}
PY=/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
RUN=(uv run --no-project --python "$PY" python)
LOG=$OUT/flash-mtp${MTP_TOKENS}.log
mkdir -p "$OUT"
if [[ -e "$LOG" ]]; then
  echo "Refusing to replace evidence: $LOG (choose X1_OUT)"
  exit 2
fi
date -u
if [[ "${X1_REUSE_POST_HEALTH:-0}" != 1 ]]; then
  "$REPO/bin/gpu-run" "$REPO/bin/xpu-health"
else
  echo REUSING_RECORDED_POST_HEALTH
fi
"$REPO/vllm/flashnext/harness/x1_serve_flashnext.sh" > "$LOG" 2>&1 &
serve_pid=$!
cleanup() {
  docker stop -t 20 "$NAME" >/dev/null 2>&1 || true
  wait "$serve_pid" 2>/dev/null || true
}
trap cleanup EXIT
ready=0
failed=0
started=0
for ((i=0;i<2800;i++)); do
  if [[ "$started" = 0 ]] && grep -Fq 'gpu-run: [X1] acquired card(s)' "$LOG"; then
    started=$SECONDS
  fi
  if ((started > 0 && SECONDS - started > 600)); then echo STARTUP_TIMEOUT; break; fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/flash-models-mtp${MTP_TOKENS}.json"; then
    ready=1
    break
  fi
  if ! kill -0 "$serve_pid" 2>/dev/null; then break; fi
  sleep 3
done
if ((ready)); then
  extra=()
  if [[ "$MTP_TOKENS" != 0 ]]; then extra=(--baseline "$OUT/flash-greedy-mtp0.json"); fi
  set +e
  "$REPO/bin/cpu-run" --quiet "${RUN[@]}" \
    "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" --respect-eos \
    --url "http://127.0.0.1:$PORT" --out "$OUT/flash-greedy-mtp${MTP_TOKENS}.json" \
    "${extra[@]}"
  greedy_rc=$?
  echo GREEDY_RC=$greedy_rc
  if ((greedy_rc)); then failed=1; fi
  "$REPO/bin/cpu-run" --quiet "${RUN[@]}" \
    "$REPO/vllm/flashnext/harness/bench_latency.py" \
    --base-url "http://127.0.0.1:$PORT" --model hotschmoe-dd \
    --alias "qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-vllm-rt-mtp${MTP_TOKENS}" \
    --out "$OUT/flash-bench-mtp${MTP_TOKENS}.json" --decode-tokens 96 \
    --skip-prefill --tag "X1-flashnext-full-mtp${MTP_TOKENS}"
  bench_rc=$?
  echo BENCH_RC=$bench_rc
  if ((bench_rc)); then failed=1; fi
  if [[ "$MTP_TOKENS" = 0 ]]; then
    "$REPO/bin/cpu-run" --quiet "${RUN[@]}" \
      "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" --respect-eos \
      --url "http://127.0.0.1:$PORT" --out "$OUT/flash-repeat-mtp0.json" \
      --baseline "$OUT/flash-greedy-mtp0.json"
    repeat_rc=$?
    echo CONTROL_REPEAT_RC=$repeat_rc
    if ((repeat_rc)); then failed=1; fi
  fi
  set -e
else
  echo SERVE_NOT_READY
  tail -n 80 "$LOG"
fi
cleanup
trap - EXIT
"$REPO/bin/gpu-run" "$REPO/bin/xpu-health"
echo "X1_FLASH_WINDOW_DONE ready=$ready"
test "$ready" = 1
test "$failed" = 0
