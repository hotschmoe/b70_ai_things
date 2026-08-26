#!/usr/bin/env bash
# Overlay the tracked native W8A8 dispatcher on the exact extension image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_TAG="${BASE_TAG:-b70-sglang-xpu-int8:20260826-2dd55f3}"
BASE_ID=sha256:aeb939facd982c65227aa097c29842529f67840d0a114337fa2ea93803e7c6e9
TAG="${TAG:-b70-sglang-xpu-int8-runtime:20260826-2dd55f3}"

[ "$(docker image inspect --format '{{.Id}}' "$BASE_TAG")" = "$BASE_ID" ] || {
  echo "Local native extension tag does not resolve to exact image: $BASE_TAG" >&2
  exit 1
}

echo "Building $TAG"
echo "Native extension base $BASE_TAG at $BASE_ID"

docker build --progress=plain \
  --build-context "b70-refresh=$SCRIPT_DIR" \
  --tag "$TAG" \
  --file "$SCRIPT_DIR/Dockerfile.int8-runtime" \
  "$SCRIPT_DIR"

docker image inspect --format \
  'image={{.Id}} created={{.Created}} repo_digests={{json .RepoDigests}}' \
  "$TAG"
