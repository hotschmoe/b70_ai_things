#!/usr/bin/env bash
# Stage the pinned 0xSero production inputs plus repo-owned instrumentation.
# This is a compilation-only workflow; it does not mount /dev/dri or touch a GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_RECIPE="${UPSTREAM_RECIPE:-/mnt/vm_8tb/b70/qwen38-b70}"
IMAGE="${IMAGE:-qwen38-b70:quant-timing}"
TP2_SHA256=f21e9b557c3d024527ac98d5f189cf7ea72fa8c38a5faf2a22ee339fd1988998
Q4K_SHA256=0a27858525f6a402cf9c92d1b93daee0a80e2ffaef9137bf7bce784a549b58b6

for path in \
    "$UPSTREAM_RECIPE/entrypoint.sh" \
    "$UPSTREAM_RECIPE/patches/tp2-full-stack.patch" \
    "$UPSTREAM_RECIPE/patches/q4k-increment.patch" \
    "$SCRIPT_DIR/patches/quant-census.patch" \
    "$SCRIPT_DIR/patches/quant-timing.patch"; do
    [ -f "$path" ] || { printf 'missing build input: %s\n' "$path" >&2; exit 2; }
done

printf '%s  %s\n' "$TP2_SHA256" "$UPSTREAM_RECIPE/patches/tp2-full-stack.patch" | sha256sum -c -
printf '%s  %s\n' "$Q4K_SHA256" "$UPSTREAM_RECIPE/patches/q4k-increment.patch" | sha256sum -c -

BUILD_CONTEXT="$(mktemp -d /tmp/qwen38-b70-build.XXXXXX)"
trap 'rm -rf -- "$BUILD_CONTEXT"' EXIT
mkdir -p "$BUILD_CONTEXT/patches"
cp "$SCRIPT_DIR/Dockerfile" "$BUILD_CONTEXT/Dockerfile"
cp "$UPSTREAM_RECIPE/entrypoint.sh" "$BUILD_CONTEXT/entrypoint.sh"
cp "$UPSTREAM_RECIPE/patches/tp2-full-stack.patch" "$BUILD_CONTEXT/patches/"
cp "$UPSTREAM_RECIPE/patches/q4k-increment.patch" "$BUILD_CONTEXT/patches/"
cp "$SCRIPT_DIR/patches/quant-census.patch" "$BUILD_CONTEXT/patches/"
cp "$SCRIPT_DIR/patches/quant-timing.patch" "$BUILD_CONTEXT/patches/"

printf 'building %s with pinned base patches plus quant instrumentation\n' "$IMAGE"
QUANT_TIMING_PATCH_SHA="$(sha256sum "$SCRIPT_DIR/patches/quant-timing.patch" | awk '{ print $1 }')"
docker build \
    --label "ai.b70.quant_timing_patch_sha=$QUANT_TIMING_PATCH_SHA" \
    --tag "$IMAGE" "$BUILD_CONTEXT"
