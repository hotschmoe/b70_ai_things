#!/usr/bin/env bash
# Short leased numeric/captured Q8 gate. Quiet starts after GPU acquisition.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT=$B/x5/q8-fast/probe-${PROBE_TAG:-r1}
NAME=vllm-fn-x5-q8probe
mkdir -p "$OUT"
test ! -e "$OUT/started" || exit 2
touch "$OUT/started"
export B70_AGENT=X5 B70_GPU_LOCK_TIMEOUT=21600 B70_CPU_LOCK_TIMEOUT=21600
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt"
sha256sum "$B/x5/q8-fast/libb70_x5_xpu.so" "$REPO/vllm/flashnext/plugins/b70_x5_xpu/csrc/q8_linear.cpp" > "$OUT/identity.txt"
"$REPO/vllm/flashnext/harness/xpu-run.sh" --card 1 --name "$NAME" \
  --env "LD_PRELOAD=$B/x5/eventless-lazy/libx5_eventless_stub.so" \
  --env "PYTHONPATH=$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu:$REPO/vllm/flashnext/plugins/b70_x5_xpu" \
  --env "B70_GGUF_SO=$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.o3g.so" \
  --env B70_XPU_CG_RECLAIM=0 --env SYCL_CACHE_PERSISTENT=0 --env "X5_Q8_OUT=$OUT" \
  -- timeout --signal=TERM --kill-after=10 240 /opt/venv/bin/python "$REPO/vllm/flashnext/harness/x5_q8_probe.py" > "$OUT/probe.log" 2>&1 &
SPID=$!
cleanup() { docker stop -t 10 "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
for ((i=0;i<7200;i++)); do
  if [[ -f "$OUT/ready" ]] || ! kill -0 "$SPID" 2>/dev/null; then break; fi
  sleep 2
done
QUIET_RC=0
if [[ -f "$OUT/ready" ]]; then
  set +e
  "$REPO/bin/cpu-run" --quiet python3 - "$OUT" "$SPID" <<'PY'
import os, sys, time
from pathlib import Path
out=Path(sys.argv[1]); (out/'go').touch()
for _ in range(150):
    if (out/'done').exists(): break
    try: os.kill(int(sys.argv[2]),0)
    except ProcessLookupError: break
    if int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))) < 8388608:
        raise SystemExit('ABORT_LOW_RAM')
    time.sleep(1)
else: raise SystemExit('X5_Q8_TIMEOUT')
PY
  QUIET_RC=$?
  set -e
  if [[ "$QUIET_RC" != 0 ]]; then cleanup; fi
fi
RC=0
wait "$SPID" || RC=$?
cleanup
trap - EXIT
echo "$RC" > "$OUT/process-exit-code.txt"
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-after.txt"
echo "X5_Q8_WINDOW_DONE rc=$RC quiet_rc=$QUIET_RC"
if [[ "$QUIET_RC" != 0 ]]; then exit "$QUIET_RC"; fi
exit "$RC"
