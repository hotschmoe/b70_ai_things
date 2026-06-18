#!/usr/bin/env bash
# Pull the Intel LLM-Scaler vLLM image (working Qwen3.6 B70 path per user data points).
# Check docker.img headroom first (50GB loop). vLLM-XPU images are large.
set -uo pipefail
echo "=== docker.img headroom BEFORE ==="
df -h /var/lib/docker | tail -1
docker system df

IMG="intel/llm-scaler-vllm:0.14.0-b8.3"
echo "=== pulling $IMG ==="
docker pull "$IMG" || { echo "PULL FAILED for $IMG — will need exact tag/registry from research"; exit 1; }

echo "=== docker.img headroom AFTER ==="
df -h /var/lib/docker | tail -1
docker images "intel/llm-scaler-vllm" --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
echo "=== DONE ==="
