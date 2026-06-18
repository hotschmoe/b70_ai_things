#!/usr/bin/env bash
# Download Qwen/Qwen3.6-27B full BF16 safetensors (~54GB) to SSD. This is the
# universal source for vLLM-XPU online FP8 (8-bit) and sym_int4 (4-bit) quant.
# 128GB DDR4 + VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 handle the online-quant memory.
set -euo pipefail
ROOT=/mnt/vm_8tb/b70
REPO="Qwen/Qwen3.6-27B"
DEST="$ROOT/models/${REPO//\//_}"

echo "Downloading $REPO BF16 -> $DEST"
docker run --rm \
  -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache \
  python:3.11 bash -c '
    set -e
    pip install -q huggingface_hub
    python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="'"$REPO"'",
    allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model","merges*","vocab*"],
    local_dir="/models/'"${REPO//\//_}"'",
)
print("Downloaded to", p)
PY'
echo "=== size ==="
du -sh "$DEST" 2>/dev/null
ls -lh "$DEST" | grep -iE 'safetensors|config|index' | head -20
echo "=== DONE ==="
