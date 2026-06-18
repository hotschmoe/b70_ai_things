#!/usr/bin/env bash
# Download OFFICIAL Qwen/Qwen3.6-27B-FP8 (~27GB, near-lossless 8-bit, block-128).
# Pre-quantized -> no online-quant needed; loads directly on llm-scaler-vllm.
# Fits the single 32GB B70 (tight KV = short-context testing now; long ctx w/ card #2).
set -euo pipefail
ROOT=/mnt/vm_8tb/b70
REPO="Qwen/Qwen3.6-27B-FP8"
DEST="$ROOT/models/${REPO//\//_}"

echo "Downloading $REPO -> $DEST"
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
echo "=== size ==="; du -sh "$DEST" 2>/dev/null
ls -lh "$DEST" | grep -iE 'safetensors|config|index' | head -20
echo "=== DONE ==="
