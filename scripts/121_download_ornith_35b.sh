#!/usr/bin/env bash
# Download deepreinforce-ai/Ornith-1.0-35B (BF16 source) to the 8TB SSD.
#
# Ornith-1.0-35B: a Qwen3.5-based 35B MoE coder model. Public, ungated, BF16, 16 safetensors shards
# (~70 GB params; HF "usedStorage" 140.5 GB counts LFS revision history, not the live download).
# It is a VLM (ships processor/preprocessor/video_preprocessor configs) like the Qwen3.5 family, so it
# needs trust-remote-code + our VLM-config handling to serve. Maker claims it beats Qwen3.5-35B on the
# coding evals we care about (SWE-bench Verified 75.6 vs 70, Terminal-Bench 64.2 vs 41.4, NL2Repo 34.6
# vs 20.5). We download it now to feed the W4A16/W8A8/W4A8 quant queue (QUANTS_TODO.md "Ornith" block).
#
# Pure disk/network I/O -- NO GPU touch, so NO gpu-run lease (safe to run while another agent holds the
# cards). DETACHED named container survives session handoff; snapshot_download RESUMES from the HF cache
# on relaunch -- re-run this script to resume a partial download.
#   Check:  docker logs -f ornith35b_dl
#           du -sh /mnt/vm_8tb/b70/models/deepreinforce-ai_Ornith-1.0-35B
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
REPO="deepreinforce-ai/Ornith-1.0-35B"
DEST_NAME="deepreinforce-ai_Ornith-1.0-35B"
DEST="$ROOT/models/$DEST_NAME"

echo "=== (re)launch download DETACHED: $REPO -> $DEST (resumes from cache) ==="
docker rm -f ornith35b_dl 2>/dev/null || true
# NOTE: $ROOT/.hf_token is root-only (0600); we run as hotschmoe and cannot cat it on the host. The
# container runs as root, so bind-mount it read-only and read it INSIDE the container. Ornith is a
# public repo -- the token is optional and harmless if absent.
docker run -d --name ornith35b_dl \
  -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" \
  -v "$ROOT/.hf_token:/hf_token_ro:ro" \
  -e HF_HOME=/hf_cache \
  python:3.11 bash -c '
    pip install -q huggingface_hub hf_transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    TOK=""; [ -s /hf_token_ro ] && TOK="$(cat /hf_token_ro 2>/dev/null)"
    HF_TOKEN="$TOK" python - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
  repo_id="deepreinforce-ai/Ornith-1.0-35B",
  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model","merges*","vocab*","*.jinja",
                  "preprocessor*","*processor*","generation_config*"],
  local_dir="/models/deepreinforce-ai_Ornith-1.0-35B",
  token=os.environ.get("HF_TOKEN") or None)
print("DOWNLOAD_COMPLETE", p)
PY'
sleep 6
echo "=== status ==="
docker ps --format "{{.Names}} {{.Status}}" | grep ornith35b_dl || { echo "FAILED to start"; docker logs ornith35b_dl 2>&1 | tail -20; exit 1; }
du -sh "$DEST" 2>/dev/null || true
echo "tail logs with: docker logs -f ornith35b_dl"
