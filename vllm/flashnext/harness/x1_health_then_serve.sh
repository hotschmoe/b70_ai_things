#!/usr/bin/env bash
# Same leased container: cached compiled health, then the requested serve.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
AOT=/mnt/vm_8tb/b70/build/vllm-flashnext/c7b-aot-bin
export MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4
echo X1_COMPILED_PRE_HEALTH_BEGIN
PATH="$AOT:$PATH" LD_LIBRARY_PATH="$AOT:${LD_LIBRARY_PATH:-}" \
  timeout --signal=TERM --kill-after=15 180 \
  torchrun --standalone --nproc-per-node=2 "$REPO/bin/xpu-collective-health.py"
echo X1_COMPILED_PRE_HEALTH_PASS
if [[ -n "${X1_EVENTLESS_SHIM:-}" ]]; then
  # Registry inspection is a fresh subprocess. Finish it before preload
  # initializes the Torch/XPU runtime in the API parent.
  timeout --signal=TERM --kill-after=10 90 python - <<'PYPRIME'
from vllm.model_executor.models import ModelRegistry
for arch in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpMTP"):
    ModelRegistry.models[arch].inspect_model_cls()
    print("X1_MODELINFO_PRIMED", arch, flush=True)
PYPRIME
  export LD_PRELOAD="$X1_EVENTLESS_SHIM${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec "$@"
