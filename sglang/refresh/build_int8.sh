#!/usr/bin/env bash
# Build the native INT8 comparison layer on the exact qualified SGLang image.
# GPU qualification is deliberately separate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS_SRC="${VLLM_XPU_KERNELS_SRC:-/mnt/vm_8tb/b70/steve-s2b/vllm-xpu-kernels}"
TAG="${TAG:-b70-sglang-xpu-int8:20260826-2dd55f3}"
BUILD_JOBS="${BUILD_JOBS:-8}"

KERNELS_COMMIT=2dd55f380df753a10a88fcd9e96192561066e713
KERNELS_TREE=2416da2ad02ff58717edb864fa839442a15ca3d2
BASE_TAG=b70-sglang-xpu:20260826-bede6bc-2d10888-torch213-umd2622
BASE_ID=sha256:8678399dce536377f67760868b166744eb149ff9146e344476bb124e0c5933cd

[ -d "$KERNELS_SRC/.git" ] || {
  echo "Missing exact source tree: $KERNELS_SRC" >&2
  exit 1
}
[ -z "$(git -C "$KERNELS_SRC" status --porcelain)" ] || {
  echo "Refusing dirty source tree: $KERNELS_SRC" >&2
  exit 1
}
[ "$(git -C "$KERNELS_SRC" rev-parse HEAD)" = "$KERNELS_COMMIT" ]
[ "$(git -C "$KERNELS_SRC" rev-parse HEAD^{tree})" = "$KERNELS_TREE" ]
[ "$(docker image inspect --format '{{.Id}}' "$BASE_TAG")" = "$BASE_ID" ] || {
  echo "Local base tag does not resolve to exact image: $BASE_TAG" >&2
  exit 1
}

BUILD_ROOT="$(mktemp -d /tmp/b70-sglang-int8-build.XXXXXX)"
cleanup() {
  rm -rf -- "$BUILD_ROOT"
}
trap cleanup EXIT

# A local clone transfers only tracked source and its small Git identity. It
# excludes the retained tree's ignored build products and dependency cache.
git clone --local --no-hardlinks "$KERNELS_SRC" \
  "$BUILD_ROOT/vllm-xpu-kernels"
git -C "$BUILD_ROOT/vllm-xpu-kernels" checkout --detach "$KERNELS_COMMIT"

echo "Building $TAG"
echo "Base $BASE_TAG at $BASE_ID"
echo "vllm-xpu-kernels $KERNELS_COMMIT tree $KERNELS_TREE"

docker build --progress=plain \
  --build-arg "BUILD_JOBS=$BUILD_JOBS" \
  --build-context "vllm-xpu-kernels=$BUILD_ROOT/vllm-xpu-kernels" \
  --tag "$TAG" \
  --file "$SCRIPT_DIR/Dockerfile.int8" \
  "$SCRIPT_DIR"

docker image inspect --format \
  'image={{.Id}} created={{.Created}} repo_digests={{json .RepoDigests}}' \
  "$TAG"
