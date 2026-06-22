#!/usr/bin/env bash
# Q5 prepack: Qwable W4A8 sqgptq (33G) -> int4-packed prepacked (~25G). CPU ONLY (no --device, no gpu-run lease).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SRC="${SRC:-/models/Qwable-5-27B-Coder-W4A8-sqgptq}"
DST="${DST:-/models/Qwable-5-27B-Coder-W4A8-sqgptq-prepacked}"
LOGF="$ROOT/results/q5_prepack.log"; mkdir -p "$ROOT/results"
docker rm -f q5_prepack 2>/dev/null || true
echo "=== Q5 prepack (CPU): $SRC -> $DST ==="
docker run --rm --name q5_prepack -v "$ROOT/models:/models" -v "$ROOT/w4a8_prepack.py:/work/prepack.py:ro" \
  -e SRC="$SRC" -e DST="$DST" --entrypoint bash vllm-xpu-env:v0230 -lc \
  'python -c "import torch,safetensors;print(\"deps ok\")" && python /work/prepack.py' > "$LOGF" 2>&1 || echo "(prepack returned nonzero)"
echo "=== q5 prepack done; log $LOGF ==="; tail -8 "$LOGF"
du -sh "$DST" 2>/dev/null || echo "(no DST)"
