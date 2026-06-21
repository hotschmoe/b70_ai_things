#!/usr/bin/env bash
# Download Qwen/Qwen3.6-35B-A3B (BF16 source, qwen3_5_moe VLM: 40 layers, 256 experts/8-per-tok, ~3B active,
# +vision +MTP head) to the 8TB SSD. This is the full-precision SOURCE we lacked -- needed to ever produce an
# int8-act (W8A8/W4A8) MoE checkpoint once an XPU int8 MoE kernel exists (see QUANTS_TODO.md / docs/kernel/15+18).
# Confirmed base repo from Intel/Qwen3.6-35B-A3B-int4-AutoRound README: base_model: Qwen/Qwen3.6-35B-A3B.
# Pure disk I/O -- NO GPU touch, so NO gpu-run lease. DETACHED container survives session handoff; resumes from
# the HF cache on relaunch (re-run this script to resume). Check:
#   ssh root@192.168.10.5 'docker logs -f qwen35b_dl; du -sh /mnt/vm_8tb/b70/models/Qwen_Qwen3.6-35B-A3B'
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
REPO="Qwen/Qwen3.6-35B-A3B"
DEST_NAME="Qwen_Qwen3.6-35B-A3B"
DEST="$ROOT/models/$DEST_NAME"
TOK=""; [ -s "$ROOT/.hf_token" ] && TOK="$(cat "$ROOT/.hf_token")"   # public repo; token harmless if present

echo "=== (re)launch download DETACHED: $REPO -> $DEST (resumes from cache) ==="
docker rm -f qwen35b_dl 2>/dev/null || true
docker run -d --name qwen35b_dl \
  -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache -e HF_TOKEN="$TOK" \
  python:3.11 bash -c '
    pip install -q huggingface_hub hf_transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
  repo_id="Qwen/Qwen3.6-35B-A3B",
  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model","merges*","vocab*","*.jinja",
                  "preprocessor*","*processor*","generation_config*"],
  local_dir="/models/Qwen_Qwen3.6-35B-A3B",
  token=os.environ.get("HF_TOKEN") or None)
print("DOWNLOAD_COMPLETE", p)
PY'
sleep 6
echo "=== status ==="
docker ps --format "{{.Names}} {{.Status}}" | grep qwen35b_dl || { echo "FAILED to start"; docker logs qwen35b_dl 2>&1 | tail -20; exit 1; }
du -sh "$DEST" 2>/dev/null || true
echo "tail logs with: ssh root@192.168.10.5 'docker logs -f qwen35b_dl'"
