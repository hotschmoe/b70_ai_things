#!/usr/bin/env bash
# Build the vLLM v0.26.0 XPU base image from the upstream tag.
# v0.26.0 still pins torch 2.12 for XPU, so the existing custom kernel ABI is
# unchanged. The source checkout must be clean and exactly at v0.26.0.
#
# Result tag: vllm-xpu-env:v0260
set -uo pipefail

SRC="${SRC:-/mnt/vm_8tb/b70/build24/vllm}"
TAG="${TAG:-vllm-xpu-env:v0260}"
LOG="${LOG:-/mnt/vm_8tb/b70/build24/build_v0260.log}"

cd "$SRC" || { echo "no src $SRC"; exit 1; }
HEAD="$(git rev-parse HEAD)"
EXPECTED="$(git rev-list -n1 v0.26.0 2>/dev/null)"
[ -n "$EXPECTED" ] || { echo "missing local v0.26.0 tag"; exit 1; }
[ "$HEAD" = "$EXPECTED" ] || {
  echo "source HEAD $HEAD is not v0.26.0 $EXPECTED"
  echo "checkout the clean upstream tag before building"
  exit 1
}
[ -z "$(git status --short)" ] || {
  echo "source checkout is dirty; refusing an unreproducible image"
  git status --short
  exit 1
}

echo "=== vLLM $(git describe --tags --exact-match) -> $TAG ==="
echo "=== docker build -f docker/Dockerfile.xpu ==="
date
time docker build -f docker/Dockerfile.xpu -t "$TAG" --shm-size=8g . \
  2>&1 | tee "$LOG" | tail -1
rc=${PIPESTATUS[0]}
echo "=== build rc=$rc ==="
docker images "$TAG" --format '{{.Repository}}:{{.Tag}} {{.Size}} {{.CreatedAt}}'
echo "=== tail of log ==="
tail -25 "$LOG"
exit "$rc"
