#!/usr/bin/env bash
# Build the two 0.5.15 shim layers on top of sglang-xpu:bmg-0515:
#   sglang-xpu:bmg-0515 -> sglang-xpu:woq-0515 -> sglang-xpu:mtp-0515
# GPU-FREE (COPY + pip install auto-round-lib only; no device). Run AFTER sglang-xpu/build_0515.sh.
# See ../SGLANG_0515_UPGRADE.md.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/2] build sglang-xpu:woq-0515 ==="
docker build -t sglang-xpu:woq-0515 -f "$SCRIPT_DIR/sglang-xpu-woq-0515/Dockerfile" \
  --build-arg http_proxy="${http_proxy:-}" --build-arg https_proxy="${https_proxy:-}" \
  --build-arg no_proxy="${no_proxy:-}" "$SCRIPT_DIR/sglang-xpu-woq-0515"

echo "=== [2/2] build sglang-xpu:mtp-0515 ==="
docker build -t sglang-xpu:mtp-0515 -f "$SCRIPT_DIR/sglang-xpu-mtp-0515/Dockerfile" \
  --build-arg http_proxy="${http_proxy:-}" --build-arg https_proxy="${https_proxy:-}" \
  --build-arg no_proxy="${no_proxy:-}" "$SCRIPT_DIR/sglang-xpu-mtp-0515"

echo "=== done: sglang-xpu:{woq-0515,mtp-0515} ==="
docker images | grep -E 'sglang-xpu:(bmg|woq|mtp)-0515' || true
