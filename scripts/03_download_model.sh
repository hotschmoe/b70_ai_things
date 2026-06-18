#!/usr/bin/env bash
# Download a Qwen3.6-27B GGUF quant to the SSD via a throwaway python container.
# HF cache + target both on /mnt/vm_8tb so nothing touches docker.img.
# Usage: REPO=unsloth/Qwen3.6-27B-GGUF PATTERN='*Q4_K_M*' bash 03_download_model.sh
set -euo pipefail
ROOT=/mnt/vm_8tb/b70
REPO="${REPO:-unsloth/Qwen3.6-27B-GGUF}"
PATTERN="${PATTERN:-*Q4_K_M*}"

echo "Downloading $REPO  (pattern: $PATTERN)  ->  $ROOT/models"
docker run --rm \
  -v "$ROOT/models:/models" \
  -v "$ROOT/hf_cache:/hf_cache" \
  -e HF_HOME=/hf_cache \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  python:3.11 bash -c '
    set -e
    pip install -q "huggingface_hub[hf_transfer]"
    python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="'"$REPO"'",
    allow_patterns=["'"$PATTERN"'"],
    local_dir="/models/'"$(echo "$REPO" | tr "/" "_")"'",
)
print("Downloaded to", p)
PY
  '
echo "===== models dir ====="
ls -lhR "$ROOT/models" | grep -iE "gguf|/models|total" | head -40
