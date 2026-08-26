#!/usr/bin/env bash
# Build the exact-source SGLang/XPU refresh from detached trees outside the
# repository. This is a CPU-only image build; GPU qualification is separate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFRESH_ROOT="${B70_REFRESH_ROOT:-/mnt/vm_8tb/b70/refreshed}"
SGLANG_SRC="$REFRESH_ROOT/sglang"
SGL_KERNEL_SRC="$REFRESH_ROOT/sgl-kernel-xpu"
TAG="${TAG:-b70-sglang-xpu:20260826-bede6bc-2d10888-torch213-umd2622}"
BUILD_JOBS="${BUILD_JOBS:-8}"

SGLANG_COMMIT=bede6bc37c5d9638099ebb948d93b9e2a7799f10
SGLANG_TREE=938cf2b1b71bbc60e5d18d8388f1388ca0eff5a7
SGL_KERNEL_COMMIT=2d10888c069350ff20a192338d568dec945c9594
SGL_KERNEL_TREE=3f975153d4d430535c759e57cb176e141a1b25c8

for src in "$SGLANG_SRC" "$SGL_KERNEL_SRC"; do
  [ -d "$src/.git" ] || {
    echo "Missing exact source tree: $src" >&2
    exit 1
  }
  [ -z "$(git -C "$src" status --porcelain)" ] || {
    echo "Refusing dirty source tree: $src" >&2
    exit 1
  }
done

[ "$(git -C "$SGLANG_SRC" rev-parse HEAD)" = "$SGLANG_COMMIT" ]
[ "$(git -C "$SGLANG_SRC" rev-parse HEAD^{tree})" = "$SGLANG_TREE" ]
[ "$(git -C "$SGL_KERNEL_SRC" rev-parse HEAD)" = "$SGL_KERNEL_COMMIT" ]
[ "$(git -C "$SGL_KERNEL_SRC" rev-parse HEAD^{tree})" = "$SGL_KERNEL_TREE" ]

echo "Building $TAG"
echo "SGLang $SGLANG_COMMIT tree $SGLANG_TREE"
echo "sgl-kernel-xpu $SGL_KERNEL_COMMIT tree $SGL_KERNEL_TREE"

docker build --progress=plain \
  --build-arg "BUILD_JOBS=$BUILD_JOBS" \
  --build-context "b70-refresh=$SCRIPT_DIR" \
  --tag "$TAG" \
  --file "$SCRIPT_DIR/Dockerfile" \
  "$REFRESH_ROOT"

docker image inspect --format \
  'image={{.Id}} created={{.Created}} repo_digests={{json .RepoDigests}}' \
  "$TAG"
