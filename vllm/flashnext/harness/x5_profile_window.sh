#!/usr/bin/env bash
# Finite proxy profile; queue without holding CPU quiet until card is available.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
export B70_AGENT=X5 B70_GPU_LOCK_TIMEOUT=21600 B70_CPU_LOCK_TIMEOUT=21600
QUANT=${QUANT:-UD-IQ4_XS}
TAG=${TAG:-${QUANT,,}-profile}
OUT=$B/x5/$TAG
mkdir -p "$OUT/trace"
test ! -e "$OUT/started" || { echo "Refusing to overwrite $OUT"; exit 2; }
touch "$OUT/started"
NAME=vllm-fn-x5-profile
UT=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt"
# Keep CPU quiet out of the queue. The profile process waits for a trigger
# after loading; the host holds quiet from that trigger through completion.
CMD=(/opt/venv/bin/python "$REPO/vllm/flashnext/harness/x5_profile.py")
RECLAIM=500
EXTRA=()
EXTRA+=(--env "B70_X5_PACK_LINEAR=${B70_X5_PACK_LINEAR:-0}")
if [[ "${UT_TRACE:-0}" == 1 ]]; then
  CMD=(bash -c "export LD_LIBRARY_PATH=$B/o4/zelayers:\$LD_LIBRARY_PATH; exec $UT -d --chrome-kernel-logging --teardown-on-signal 10 --session x5$TAG --follow-child-process 0 --output-dir-path $OUT/trace -o $OUT/trace/timing.txt /opt/venv/bin/python $REPO/vllm/flashnext/harness/x5_profile.py")
  EXTRA+=(--env "O4_UT_SESSION=x5$TAG")
fi
if [[ "${B70_X5_GDN_BF16:-0}" == 1 ]]; then
  EXTRA+=(--env B70_X5_GDN_BF16=1 --env "B70_GDN_SO=$B/x5/gdn-bf16/libb70_gdn_xpu.so")
fi
if [[ "${B70_X5_EVENTLESS:-0}" == 1 ]]; then
  RECLAIM=0
  CMD=("$REPO/vllm/flashnext/harness/x5_eventless_serve.sh" "${CMD[@]}")
fi
rm -f "$OUT/profile-ready" "$OUT/profile-go"
"$REPO/vllm/flashnext/harness/xpu-run.sh" --card 1 --name "$NAME" \
  --env B70_GGUF=1 --env "B70_GGUF_PATH=/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-$QUANT.gguf" \
  --env "B70_GGUF_SO=$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.o3g.so" \
  --env "PYTHONPATH=$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu:$REPO/vllm/flashnext/harness" \
  --env B70_FUSE_SHARED=1 --env B70_D3_GDN=0 --env "B70_XPU_CG_RECLAIM=$RECLAIM" \
  --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 --env B70_PREFILL=0 \
  --env SYCL_CACHE_PERSISTENT=0 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env "O4_OUT=$OUT" --env "X5_ALIAS=qwen3.6-35b-a3b-${QUANT,,}-x5-mtp0" \
  "${EXTRA[@]}" -- "${CMD[@]}" > "$OUT/run.log" 2>&1 &
SPID=$!
cleanup() { docker stop -t 15 "$NAME" >/dev/null 2>&1 || true; wait "$SPID" 2>/dev/null || true; }
trap cleanup EXIT
for ((i=0;i<7200;i++)); do
  a=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if ((a < 8388608)); then cleanup; break; fi
  if [[ -f "$OUT/profile-ready" ]] || ! kill -0 "$SPID" 2>/dev/null; then break; fi
  sleep 3
done
if [[ -f "$OUT/profile-ready" ]]; then
  set +e
  "$REPO/bin/cpu-run" --quiet python3 - "$OUT" "$SPID" <<'PY'
import os, sys, time
from pathlib import Path
out = Path(sys.argv[1])
(out / 'profile-go').touch()
deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    if (out / 'profile-done').exists():
        break
    try:
        os.kill(int(sys.argv[2]), 0)
    except ProcessLookupError:
        break
    status = Path('/proc') / sys.argv[2] / 'stat'
    if status.exists() and status.read_text().split()[2] == 'Z':
        break
    if int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) < 8388608:
        raise SystemExit('ABORT_LOW_RAM')
    time.sleep(1)
else:
    raise SystemExit('PROFILE_TIMEOUT')
PY
  QUIET_RC=$?
  set -e
  if [[ "$QUIET_RC" != 0 ]]; then cleanup; fi
fi
wait "$SPID" || RC=$?
cleanup
trap - EXIT
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-after.txt"
echo "X5_PROFILE_WINDOW_DONE rc=${RC:-0}"
if [[ "${QUIET_RC:-0}" != 0 ]]; then exit "$QUIET_RC"; fi
exit "${RC:-0}"
