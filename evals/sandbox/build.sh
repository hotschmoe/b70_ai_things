#!/usr/bin/env bash
# Build the Tier-1 code-execution sandbox image. One-time (rebuild only to bump evalplus).
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${EVALPLUS_VERSION:-0.3.1}"
TAG="evalplus-sandbox:${VERSION}"
echo "=== building ${TAG} ==="
docker build --build-arg "EVALPLUS_VERSION=${VERSION}" -t "${TAG}" .
echo "=== done: ${TAG} ==="
docker run --rm "${TAG}" python -c "import evalplus; print('evalplus', evalplus.__version__, 'OK in sandbox')"
