#!/usr/bin/env bash
# Download DJLougen/Qwable-5-27B-Coder (BF16, 28B, ~55.6GB / 15 shards, public Apache-2.0) to the 8TB SSD
# for later quant to w4a16 / w4a8 / w8a8 once the second B70 lands. Pure disk I/O -- no GPU touch, so NO
# gpu-run lease needed. Runs as a DETACHED named container so it survives an ssh/session handoff;
# snapshot_download RESUMES from the HF cache on relaunch (re-run this script to resume). Check progress:
#   ssh root@192.168.10.5 'docker logs -f qwable27b_dl; du -sh /mnt/vm_8tb/b70/models/DJLougen_Qwable-5-27B-Coder'
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
REPO="DJLougen/Qwable-5-27B-Coder"
DEST_NAME="DJLougen_Qwable-5-27B-Coder"
DEST="$ROOT/models/$DEST_NAME"
TOK=""; [ -s "$ROOT/.hf_token" ] && TOK="$(cat "$ROOT/.hf_token")"   # public repo; token harmless if present

echo "=== (re)launch download DETACHED: $REPO -> $DEST (resumes from cache) ==="
docker rm -f qwable27b_dl 2>/dev/null || true
docker run -d --name qwable27b_dl \
  -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache -e HF_TOKEN="$TOK" \
  python:3.11 bash -c '
    pip install -q huggingface_hub hf_transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
  repo_id="DJLougen/Qwable-5-27B-Coder",
  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model","merges*","vocab*","*.jinja"],
  local_dir="/models/DJLougen_Qwable-5-27B-Coder",
  token=os.environ.get("HF_TOKEN") or None)
print("DOWNLOAD_COMPLETE", p)
PY'
sleep 6
echo "=== status ==="
docker ps --format "{{.Names}} {{.Status}}" | grep qwable27b_dl || { echo "FAILED to start"; docker logs qwable27b_dl 2>&1 | tail -20; exit 1; }
du -sh "$DEST" 2>/dev/null || true
echo "tail logs with: ssh root@192.168.10.5 'docker logs -f qwable27b_dl'"
