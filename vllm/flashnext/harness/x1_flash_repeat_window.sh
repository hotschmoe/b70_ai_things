#!/usr/bin/env bash
# Repeat the same no-MTP gate to distinguish target repeatability from MTP.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT=/mnt/vm_8tb/b70/build/vllm-flashnext/x1
export B70_AGENT=X1 B70_GPU_LOCK_TIMEOUT=7200 B70_CPU_LOCK_TIMEOUT=7200
export PORT=18121 MTP_TOKENS=0 NAME=vllm-fn-x1-repeat
PY=/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
LOG=$OUT/flash-repeat.log
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
started=0
for ((i=0;i<2800;i++)); do
  if [[ "$started" = 0 ]] && grep -Fq 'gpu-run: [X1] acquired card(s)' "$LOG"; then started=$SECONDS; fi
  if ((started > 0 && SECONDS - started > 600)); then echo STARTUP_TIMEOUT; break; fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=1; break; fi
  if ! kill -0 "$serve_pid" 2>/dev/null; then break; fi
  sleep 3
done
if ((ready)); then
  baseline=$OUT/flash-greedy-mtp0.json
  for pass in 1 2; do
    result=$OUT/flash-repeat-pass${pass}.json
    set +e
    "$REPO/bin/cpu-run" --quiet uv run --no-project --python "$PY" python \
      "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" --respect-eos \
      --url "http://127.0.0.1:$PORT" --out "$result" --baseline "$baseline"
    echo REPEAT_PASS_${pass}_RC=$?
    set -e
    baseline=$result
  done
else
  echo SERVE_NOT_READY
  tail -n 60 "$LOG"
fi
cleanup
trap - EXIT
"$REPO/bin/gpu-run" "$REPO/bin/xpu-health"
echo "X1_REPEAT_WINDOW_DONE ready=$ready"
test "$ready" = 1
