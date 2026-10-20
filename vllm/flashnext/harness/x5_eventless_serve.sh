#!/usr/bin/env bash
# X1's Torch-08187d9e eventless replay, inside the existing single-card lease.
set -euo pipefail
SHIM=/mnt/vm_8tb/b70/build/vllm-flashnext/x1/eventless/libx1_eventless.so
if [[ "${B70_X5_EVENTLESS_LAZY:-0}" == 1 ]]; then
  SHIM=/mnt/vm_8tb/b70/build/vllm-flashnext/x5/eventless-lazy/libx5_eventless_stub.so
fi
test -f "$SHIM"
echo X5_EVENTLESS_SHIM
sha256sum "$SHIM"
# Cold model inspection is a fresh process: prime both classes BEFORE
# globally preloading the shim. X1 found the opposite order can spin in UMD.
timeout --signal=TERM --kill-after=10 90 /opt/venv/bin/python - <<'PY'
from vllm.model_executor.models import ModelRegistry
for arch in ('Qwen3_5MoeForConditionalGeneration', 'Qwen3_5MoeMTP'):
    ModelRegistry.models[arch].inspect_model_cls()
    print('X5_MODELINFO_PRIMED', arch, flush=True)
PY
export LD_PRELOAD="$SHIM${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$@"
