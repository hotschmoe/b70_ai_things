#!/usr/bin/env bash
# zml/test_sharding.sh -- THE first GPU test for ZML on the dual B70: the oneAPI //examples/sharding
# SPMD smoke test across both cards. Validates the whole stack (PJRT enumerates 2 B70s, oneAPI platform
# not CPU fallback, num_partitions=2, collectives complete without DEVICE_LOST / box wedge) BEFORE any LLM.
#
# MUST run under the GPU lease (both cards):
#   cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash zml/test_sharding.sh
#
# Critical env (REVIEW_intel_arch.md sec 1/5):
#   CCL_TOPO_P2P_ACCESS=0  -- set EXPLICITLY. zml's oneapi.zig:33 has a bug that otherwise defaults this
#                            knob from CCL_ATL_TRANSPORT (= "ofi", a garbage value). =0 also aligns with our
#                            P2P-wedge discipline (CLAUDE.md / docs/P2P_GPU.md). Override to 1 only knowingly.
#   ZE_FLAT_DEVICE_HIERARCHY=FLAT  -- each B70 = one PJRT device (no tile/sub-device composite).
#   ONEAPI_DEVICE_SELECTOR=level_zero:gpu  -- all L0 GPUs (both cards).
#
# --mesh=mock needs >=8 devices -> NOT usable on 2 cards; use --mesh=auto (a 2-wide .bus axis = fine for TP=2).
set -euo pipefail
ZML="${ZML:-/mnt/vm_8tb/b70/zml}"
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
BAZELISK="${BAZELISK:-$HOME/.local/bin/bazelisk}"

echo "=== pre-flight xpu-health ===" && "$REPO/bin/xpu-health" 2>&1 | tail -2
echo "=== /dev/dri ===" && ls -l /dev/dri/renderD* 2>/dev/null
cd "$ZML"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-0}"
echo "=== zml oneAPI sharding smoke (CCL_TOPO_P2P_ACCESS=$CCL_TOPO_P2P_ACCESS) $(date) ==="
set +e   # capture rc even on failure so the bazel-shutdown below ALWAYS runs (else set -e skips it and
         # the bazel daemon keeps the gpu-run flock held ~3h, blocking later gpu-runs incl. DD restore)
"$BAZELISK" run //examples/sharding \
  --config=release \
  --@zml//platforms:cpu=false \
  --@zml//platforms:oneapi=true \
  -- \
  --partitioner=shardy \
  --mesh=auto
rc=$?
# Shut down the bazel DAEMON before returning -- it inherits the gpu-run flock fds and would otherwise
# keep the GPU lease HELD for ~3h, blocking every later gpu-run (incl. the daily-driver restore).
"$BAZELISK" shutdown >/dev/null 2>&1 || true
echo "=== sharding exit rc=$rc ; post-run xpu-health ==="
"$REPO/bin/xpu-health" 2>&1 | tail -2 || echo "[!] box may be wedged -- bin/xe-reset"
exit $rc
