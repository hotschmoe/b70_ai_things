#!/usr/bin/env bash
# Generic non-gated HF safetensors model downloader. REPO=<id> [DEST_NAME=<dir>].
# Used for draft models (Qwen3-0.6B) and small test models.
set -euo pipefail
ROOT=/mnt/vm_8tb/b70
REPO="${REPO:?set REPO=org/name}"
DEST="$ROOT/models/${REPO//\//_}"
TOK=""; [ -s "$ROOT/.hf_token" ] && TOK="$(cat "$ROOT/.hf_token")"
echo "Downloading $REPO -> $DEST"
docker run --rm -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache \
  -e HF_TOKEN="$TOK" python:3.11 bash -c '
    set -e; pip install -q huggingface_hub
    python - <<PY
import os
from huggingface_hub import snapshot_download
tok=os.environ.get("HF_TOKEN") or None
p=snapshot_download(repo_id="'"$REPO"'",
  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model","merges*","vocab*"],
  local_dir="/models/'"${REPO//\//_}"'", token=tok)
print("Downloaded to", p)
PY'
du -sh "$DEST" 2>/dev/null; echo DONE
