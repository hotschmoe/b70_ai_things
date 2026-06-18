#!/usr/bin/env bash
# Pull the Intel-recommended llm-scaler vLLM tag for Qwen3.6-27B. Clean up the dead
# FP8 container first. Plenty of docker.img room now (200G).
set -uo pipefail
docker rm -f vllm_fp8 2>/dev/null || true
IMG="intel/llm-scaler-vllm:0.14.0-b8.3.1"
echo "=== pulling $IMG ==="
docker pull "$IMG" || { echo "PULL FAILED for $IMG (tag may not exist; will fall back to b8.3)"; exit 1; }
echo "=== headroom ==="; df -h /var/lib/docker | tail -1
docker images intel/llm-scaler-vllm --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
echo "=== DONE ==="
