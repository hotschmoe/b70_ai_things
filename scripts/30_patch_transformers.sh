#!/usr/bin/env bash
# Gemma 4 (gemma4_unified) needs a newer Transformers than vllm-xpu-env bundles.
# Build a thin derived image vllm-xpu-env:tf with upgraded transformers.
set -uo pipefail
BUILD=/mnt/vm_8tb/b70/build/tfpatch
mkdir -p "$BUILD"; cd "$BUILD"

echo "=== current transformers in vllm-xpu-env ==="
docker run --rm --entrypoint python vllm-xpu-env -c 'import transformers; print("transformers", transformers.__version__)' 2>&1 | tail -1

cat > Dockerfile <<'EOF'
FROM vllm-xpu-env
# Upgrade transformers for Gemma 4 (gemma4_unified). Try latest release first.
RUN pip install --no-cache-dir -U "transformers>=4.58" || \
    pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git"
EOF

echo "=== building vllm-xpu-env:tf ==="
docker build -t vllm-xpu-env:tf . 2>&1 | tail -15

echo "=== new transformers version ==="
docker run --rm --entrypoint python vllm-xpu-env:tf -c 'import transformers; print("transformers", transformers.__version__); from transformers.models.auto import configuration_auto as c; print("gemma4 in registry:", any("gemma4" in k for k in c.CONFIG_MAPPING_NAMES))' 2>&1 | tail -3
echo "=== verify vllm still imports ==="
docker run --rm --entrypoint python vllm-xpu-env:tf -c 'import vllm; print("vllm", vllm.__version__, "OK")' 2>&1 | tail -2
echo "=== DONE ==="
