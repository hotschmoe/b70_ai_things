#!/usr/bin/env bash
# Hold both GPU leases across pre-health and the foreground daily-driver serve.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
IMAGE=b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a

cd "$REPO"
exec env B70_AGENT=hotschmoe-dd-systemd B70_GPU_LOCK_TIMEOUT=120 \
  "$REPO/bin/gpu-run" bash -lc '
    set -euo pipefail
    cd /mnt/vm_8tb/github/b70_ai_things
    IMG=b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a bin/xpu-health
    IMG=b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a \
      bin/xpu-collective-health --p2p 0 --ccl-root /opt/venv
    exec env \
      NAME=hotschmoe-dd \
      PORT=18080 \
      PUBLISH_HOST=0.0.0.0 \
      API_KEY_FILE=/mnt/vm_8tb/b70/secrets/dd_api_key \
      SERVED=hotschmoe-dd \
      bash rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh start
  '
