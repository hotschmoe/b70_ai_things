#!/usr/bin/env bash
# Add the publisher's pinned mixed spec/non-spec GDN kernel to the qualified
# local deterministic MTP1 image. This is a build-only operation; it does not
# touch XPU devices.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
BUILD_ROOT="${BUILD_ROOT:-$ROOT/build/qwen38-fp8-mixed-gdn}"
BASE_IMAGE="${BASE_IMAGE:-neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-serial-f04-local}"
EXPECTED_BASE_IMAGE_ID="${EXPECTED_BASE_IMAGE_ID:-sha256:338ef5e2c956471a195a0644e2acb38288ac6837ebf5282671794be5329a7e2b}"
IMAGE="${IMAGE:-neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local}"
RUN_ID=32798686770
ARTIFACT_NAME=vllm-xpu-kernels--20260825-014754
KERNEL_HEAD=1e90ffa672ba02f17a909da11838a4c55b199783
WHEEL_NAME=vllm_xpu_kernels-0.1.dev1+g1e90ffa67-cp38-abi3-manylinux_2_28_x86_64.whl
WHEEL_SHA256=f3d999060c11ad6db5b4033d50d19c6b665492380075480d041ec4ee58fdfeb6
ARTIFACT_DIR="$BUILD_ROOT/vllm-xpu-kernels-$KERNEL_HEAD"

for command_name in docker gh sha256sum; do
  command -v "$command_name" >/dev/null || {
    echo "$command_name is required" >&2
    exit 1
  }
done
[ "$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')" = "$EXPECTED_BASE_IMAGE_ID" ] || {
  echo "base image identity mismatch" >&2
  exit 1
}
mkdir -p "$ARTIFACT_DIR"

if [ ! -f "$ARTIFACT_DIR/$WHEEL_NAME" ]; then
  gh run download "$RUN_ID" \
    --repo vllm-project/vllm-xpu-kernels \
    --name "$ARTIFACT_NAME" \
    --dir "$ARTIFACT_DIR"
  wheel_source="$(find "$ARTIFACT_DIR" -type f -name "$WHEEL_NAME" -print -quit)"
  [ -n "$wheel_source" ] || {
    echo "downloaded artifact omitted $WHEEL_NAME" >&2
    exit 1
  }
  if [ "$wheel_source" != "$ARTIFACT_DIR/$WHEEL_NAME" ]; then
    cp --reflink=auto --no-preserve=ownership \
      "$wheel_source" "$ARTIFACT_DIR/$WHEEL_NAME"
  fi
fi
[ "$(sha256sum "$ARTIFACT_DIR/$WHEEL_NAME" | awk '{print $1}')" = "$WHEEL_SHA256" ] || {
  echo "kernel wheel digest mismatch" >&2
  exit 1
}

docker build --pull=false \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$SCRIPT_DIR/Dockerfile.qwen38-fp8-mixed-gdn" \
  --tag "$IMAGE" "$ARTIFACT_DIR"
[ "$(docker image inspect "$IMAGE" --format '{{ index .Config.Labels "b70.kernel.head" }}')" = "$KERNEL_HEAD" ] || {
  echo "built image omitted the pinned kernel identity" >&2
  exit 1
}
docker image inspect "$IMAGE" --format '{{.Id}}'
