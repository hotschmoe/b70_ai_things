#!/usr/bin/env bash
# Build a source-only MTP8 RMSNorm overlay on the qualified F07 image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
BUILD_ROOT="${BUILD_ROOT:?set BUILD_ROOT to a dedicated new build directory}"
SOURCE_REPO="${SOURCE_REPO:-$ROOT/steve-s2b/vllm}"
SOURCE_COMMIT=ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9
SOURCE_DIR="$BUILD_ROOT/vllm-$SOURCE_COMMIT"
PATCH="${PATCH:-$ROOT/steve-repro/qwen38-fp8-neural-20260829/source/experiments/qwen38-27b-b70/patches/vllm-qwen38-xpu-gemma-rmsnorm-mtp1-mtp8-serial-exact-r34-20260828.patch}"
PATCH_SHA256=98c26561926abfcfa7b057eb83cda3c2774dff908c3641f09586f748c7dbff44
BASE_IMAGE="${BASE_IMAGE:-neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local}"
BASE_IMAGE_ID="${BASE_IMAGE_ID:-sha256:8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81}"
IMAGE="${IMAGE:-b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a}"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.qwen38-fp8-mtp8-rms-overlay"

[ ! -e "$BUILD_ROOT" ] || {
  echo "BUILD_ROOT must be new: $BUILD_ROOT" >&2
  exit 1
}
[ -d "$SOURCE_REPO/.git" ] || { echo "missing source repo: $SOURCE_REPO" >&2; exit 1; }
[ "$(sha256sum "$PATCH" | awk '{print $1}')" = "$PATCH_SHA256" ] || {
  echo "source patch digest mismatch" >&2
  exit 1
}
[ "$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')" = "$BASE_IMAGE_ID" ] || {
  echo "base image identity mismatch" >&2
  exit 1
}

mkdir -p "$BUILD_ROOT"
git clone --shared --no-checkout "$SOURCE_REPO" "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --detach "$SOURCE_COMMIT"
git -C "$SOURCE_DIR" apply --check "$PATCH"
git -C "$SOURCE_DIR" apply "$PATCH"
git -C "$SOURCE_DIR" diff --check
LAYERNORM_SHA256="$(sha256sum "$SOURCE_DIR/vllm/model_executor/layers/layernorm.py" | awk '{print $1}')"

docker build --pull=false \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "LAYERNORM_SHA256=$LAYERNORM_SHA256" \
  --build-arg "SOURCE_COMMIT=$SOURCE_COMMIT" \
  --build-arg "SOURCE_PATCH_SHA256=$PATCH_SHA256" \
  --file "$DOCKERFILE" --tag "$IMAGE" "$SOURCE_DIR"

echo "image=$IMAGE"
echo "image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')"
echo "layernorm_sha256=$LAYERNORM_SHA256"
echo "source_commit=$SOURCE_COMMIT"
echo "source_patch_sha256=$PATCH_SHA256"
