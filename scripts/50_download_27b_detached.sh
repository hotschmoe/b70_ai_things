#!/usr/bin/env bash
# Make the Qwen3.6-27B BF16 download robust across a session/host handoff: free the GPU, kill the
# ssh-held --rm downloader, and relaunch it as a DETACHED named container that survives. snapshot_download
# RESUMES from the HF cache (no re-download of the ~17GB already fetched). Check later with:
#   docker logs -f qwen27b_dl   ;   du -sh /mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
echo "=== free GPU + kill ssh-held downloader ==="
docker rm -f vllm_int8 2>/dev/null || true
DL=$(docker ps -q --filter ancestor=python:3.11); [ -n "$DL" ] && docker rm -f $DL 2>/dev/null || true
docker rm -f qwen27b_dl 2>/dev/null || true
TOK=""; [ -s "$ROOT/.hf_token" ] && TOK="$(cat "$ROOT/.hf_token")"
echo "=== relaunch download DETACHED (resumes from cache) ==="
docker run -d --name qwen27b_dl \
  -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache -e HF_TOKEN="$TOK" \
  python:3.11 bash -c '
    pip install -q huggingface_hub
    python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3.6-27B",
  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model","merges*","vocab*"],
  local_dir="/models/Qwen_Qwen3.6-27B",
  token=os.environ.get("HF_TOKEN") or None)
print("DOWNLOAD_COMPLETE")
PY'
sleep 6
echo "=== status ==="; docker ps --format "{{.Names}} {{.Status}}" | grep qwen27b_dl || { echo "FAILED to start"; docker logs qwen27b_dl 2>&1 | tail -10; }
du -sh "$ROOT/models/Qwen_Qwen3.6-27B" 2>/dev/null
