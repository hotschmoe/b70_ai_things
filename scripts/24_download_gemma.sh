#!/usr/bin/env bash
# Download google/gemma-4-12B-it (BF16) to SSD using the HF token stored at
# /mnt/vm_8tb/b70/.hf_token (600 perms). Token is never printed. ~24GB.
# Source for online FP8 (8-bit fast path) and self W8A8 INT8 quantization.
set -euo pipefail
ROOT=/mnt/vm_8tb/b70
REPO="${REPO:-google/gemma-4-12B-it}"
DEST="$ROOT/models/${REPO//\//_}"
TOKFILE="$ROOT/.hf_token"
[ -s "$TOKFILE" ] || { echo "ERROR: no token at $TOKFILE"; exit 1; }

echo "Downloading $REPO -> $DEST (token hidden)"
docker run --rm \
  -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache \
  -e HF_TOKEN="$(cat "$TOKFILE")" \
  python:3.11 bash -c '
    set -e
    pip install -q huggingface_hub
    python - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="'"$REPO"'",
    allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model","merges*","vocab*"],
    local_dir="/models/'"${REPO//\//_}"'",
    token=os.environ["HF_TOKEN"],
)
print("Downloaded to", p)
PY'
echo "=== size ==="; du -sh "$DEST" 2>/dev/null
ls -lh "$DEST" | grep -iE 'safetensors|config|index' | head -20
echo "=== DONE ==="
