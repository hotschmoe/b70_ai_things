#!/usr/bin/env bash
# One-card QSA unit test.
#
# Host Python sees the card through Level Zero Sysman and then SYCL init
# reports no device: the host process does not have the IGC mount the
# container uses. harness/xpu-run.sh is the supported B70 entry for this
# venv. It calls bin/gpu-run --card N, bind-mounts the host UMD, and
# checks torch.xpu before the command. Do not wrap this script in a
# second gpu-run; the lease is already inside xpu-run.
set -euo pipefail

ROOT="${ROOT:-/mnt/vm_8tb/github/b70_ai_things}"
PLUGIN="$ROOT/vllm/flashnext/plugins/b70_qsa_xpu"
REF="$ROOT/vllm/flashnext"
OUT="${OUT:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_qsa_xpu}"
SO="$OUT/libb70_qsa_xpu.so"
CARD="${CARD:-0}"

test -f "$SO"

if [[ "${B70_QSA_INNER:-}" == "1" ]]; then
  cd "$PLUGIN"
  export B70_QSA_SO="$SO"
  export PYTHONPATH="$PLUGIN:$REF${PYTHONPATH:+:$PYTHONPATH}"
  # A hung kernel must not keep the card. 600s covers first-time JIT.
  timeout --signal=KILL 600 python -u tests/test_qsa_xpu.py
  exit $?
fi

echo "=== xpu-run card $CARD ==="
set +e
B70_AGENT="${B70_AGENT:-G5}" \
B70_GPU_LOCK_TIMEOUT="${B70_GPU_LOCK_TIMEOUT:-7200}" \
  "$ROOT/vllm/flashnext/harness/xpu-run.sh" \
  --card "$CARD" \
  --name "vllm-fn-g5-qsa" \
  --memory 16g \
  --shm-size 4g \
  --env "B70_QSA_INNER=1" \
  --env "B70_QSA_SO=$SO" \
  --env "PYTHONPATH=$PLUGIN:$REF" \
  --workdir "$PLUGIN" \
  -- bash "$PLUGIN/tests/run_gpu.sh"
bench_rc=$?
set -e
echo "qsa unit rc=$bench_rc"

echo "=== xpu-health after card $CARD ==="
set +e
"$ROOT/bin/xpu-health" --card "$CARD"
health_after=$?
set -e
echo "xpu-health after rc=$health_after"
if [[ "$health_after" -eq 1 ]]; then
  echo "card $CARD wedged after the test"
  exit 1
fi
exit "$bench_rc"
