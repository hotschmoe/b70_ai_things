#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BUILD_ROOT="${BUILD_ROOT:-$RUNTIME_ROOT/steve-repro/qwen38-w8a8-int8-1e90-$STAMP}"
SOURCE_REPO="${SOURCE_REPO:-$RUNTIME_ROOT/steve-s2b/vllm-xpu-kernels}"
SOURCE_COMMIT=1e90ffa672ba02f17a909da11838a4c55b199783
EXPECTED_PATCHED_TREE=6c944faae2af17ada2123acacfdf540ce43b2255
PATCH="$REPO/kernels/vllm_xpu_kernels_1e90_w8a8.patch"
INT8_HEADER="$REPO/kernels/int8_gemm_w8a8.h"
INT8_QUANT="$REPO/kernels/int8_quant_xpu.cpp"
BUILD_IMAGE="${BUILD_IMAGE:-b70-sglang-xpu:20260826-bede6bc-2d10888-torch213-umd2622}"
BASE_IMAGE="${BASE_IMAGE:-b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a}"
EXPECTED_BASE_IMAGE_ID="${EXPECTED_BASE_IMAGE_ID:-sha256:9ae697d4bbe64338518e8b139ec69e1d101d26bb6766c501c6ef83b022a9d5df}"
IMAGE="${IMAGE:-b70-local/vllm-openai-xpu:qwen38-w8a8-int8-mtp1-r03}"
BUILD_JOBS="${BUILD_JOBS:-8}"

[ ! -e "$BUILD_ROOT" ] || {
  echo "BUILD_ROOT must be new: $BUILD_ROOT" >&2
  exit 1
}
[ -d "$SOURCE_REPO/.git" ] || {
  echo "missing source repository: $SOURCE_REPO" >&2
  exit 1
}
[ "$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')" = \
    "$EXPECTED_BASE_IMAGE_ID" ] || {
  echo "base image identity mismatch" >&2
  exit 1
}

mkdir -p "$BUILD_ROOT/source" "$BUILD_ROOT/wheels" \
  "$BUILD_ROOT/extracted" "$BUILD_ROOT/artifacts" "$BUILD_ROOT/logs" \
  "$BUILD_ROOT/evidence"

git -C "$BUILD_ROOT/source" init -q
git -C "$BUILD_ROOT/source" fetch --no-tags "$SOURCE_REPO" \
  refs/remotes/upstream/main 2>&1 | tee "$BUILD_ROOT/logs/fetch.log"
git -C "$BUILD_ROOT/source" checkout --detach "$SOURCE_COMMIT"
git -C "$BUILD_ROOT/source" apply --check "$PATCH"
git -C "$BUILD_ROOT/source" apply "$PATCH"
install -D -m 0644 "$INT8_HEADER" \
  "$BUILD_ROOT/source/csrc/xpu/onednn/int8_gemm_w8a8.h"
install -D -m 0644 "$INT8_QUANT" \
  "$BUILD_ROOT/source/csrc/xpu/quantization/int8_quant.cpp"
git -C "$BUILD_ROOT/source" add --all
git -C "$BUILD_ROOT/source" diff --cached --check

PATCHED_TREE="$(git -C "$BUILD_ROOT/source" write-tree)"
[ "$PATCHED_TREE" = "$EXPECTED_PATCHED_TREE" ] || {
  echo "patched tree mismatch: $PATCHED_TREE" >&2
  exit 1
}
PATCH_SHA256="$(sha256sum "$PATCH" | awk '{print $1}')"
git -C "$BUILD_ROOT/source" diff --cached --binary \
  >"$BUILD_ROOT/evidence/applied-source.patch"
printf '%s\n' "$SOURCE_COMMIT" >"$BUILD_ROOT/evidence/source-commit.txt"
printf '%s\n' "$PATCHED_TREE" >"$BUILD_ROOT/evidence/patched-tree.txt"
sha256sum "$PATCH" "$INT8_HEADER" "$INT8_QUANT" \
  >"$BUILD_ROOT/evidence/source-input-sha256.txt"
docker image inspect "$BUILD_IMAGE" >"$BUILD_ROOT/evidence/build-image.json"
docker image inspect "$BASE_IMAGE" >"$BUILD_ROOT/evidence/base-image.json"

docker run --rm --user "$(id -u):$(id -g)" --entrypoint bash \
  -e VLLM_XPU_AOT_DEVICES=bmg-g21-a0 \
  -e VLLM_XPU_XE2_AOT_DEVICES=bmg-g21-a0 \
  -e B70_BUILD_JOBS="$BUILD_JOBS" \
  -v "$BUILD_ROOT/source:/src" \
  -v "$BUILD_ROOT:/out" \
  "$BUILD_IMAGE" -lc '
set -eo pipefail
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
MAX_JOBS="$B70_BUILD_JOBS" \
CMAKE_BUILD_PARALLEL_LEVEL="$B70_BUILD_JOBS" \
VLLM_VERSION_OVERRIDE=0.1.dev1+g1e90ffa67.b70w8a8 \
BUILD_SYCL_TLA_KERNELS=ON \
VLLM_XPU_ENABLE_XE2=ON \
VLLM_XPU_ENABLE_XE_DEFAULT=ON \
BASIC_KERNELS_ENABLED=OFF \
FA2_KERNELS_ENABLED=OFF \
MOE_KERNELS_ENABLED=ON \
GDN_KERNELS_ENABLED=ON \
MQA_LOGITS_KERNELS_ENABLED=ON \
MHC_KERNELS_ENABLED=ON \
XPU_SPECIFIC_KERNELS_ENABLED=ON \
XPUMEM_ALLOCATOR_ENABLED=OFF \
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir /out/wheels /src 2>&1 | tee /out/logs/build.log
'

WHEEL="$(find "$BUILD_ROOT/wheels" -maxdepth 1 -type f \
  -name 'vllm_xpu_kernels-*.whl' -print -quit)"
[ -n "$WHEEL" ] || { echo "built wheel is missing" >&2; exit 1; }
python3 -m zipfile -e "$WHEEL" "$BUILD_ROOT/extracted"
XPU_C="$(find "$BUILD_ROOT/extracted" -type f -name '_xpu_C*.so' -print -quit)"
RUNTIME_LIBS=(
  libgdn_attn_kernels_xe_2.so
  libgrouped_gemm_xe_2.so
  libgrouped_gemm_xe_default.so
  libmhc_kernels_xe_2.so
  libmqa_logits_kernels_xe_2.so
)
[ -n "$XPU_C" ] || {
  echo "wheel is missing the XPU extension" >&2
  exit 1
}
install -m 0755 "$XPU_C" "$BUILD_ROOT/artifacts/_xpu_C.abi3.so"
for library in "${RUNTIME_LIBS[@]}"; do
  source_library="$(find "$BUILD_ROOT/extracted" -type f \
    -name "$library" -print -quit)"
  [ -n "$source_library" ] || {
    echo "wheel is missing runtime library: $library" >&2
    exit 1
  }
  install -m 0755 "$source_library" "$BUILD_ROOT/artifacts/$library"
done
XPU_C_SHA256="$(sha256sum "$BUILD_ROOT/artifacts/_xpu_C.abi3.so" | awk '{print $1}')"
RUNTIME_LIBS_SHA256="$(sha256sum \
  "$BUILD_ROOT/artifacts/libgdn_attn_kernels_xe_2.so" \
  "$BUILD_ROOT/artifacts/libgrouped_gemm_xe_2.so" \
  "$BUILD_ROOT/artifacts/libgrouped_gemm_xe_default.so" \
  "$BUILD_ROOT/artifacts/libmhc_kernels_xe_2.so" \
  "$BUILD_ROOT/artifacts/libmqa_logits_kernels_xe_2.so" | \
  awk '{print $1}' | sha256sum | awk '{print $1}')"
SITE_CUSTOMIZE_SHA256="$(sha256sum "$SCRIPT_DIR/sitecustomize.py" | awk '{print $1}')"
sha256sum "$WHEEL" "$BUILD_ROOT/artifacts/"*.so \
  >"$BUILD_ROOT/evidence/artifact-sha256.txt"

docker build --pull=false \
  --build-context "int8-artifacts=$BUILD_ROOT/artifacts" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "BASE_IMAGE_ID=$EXPECTED_BASE_IMAGE_ID" \
  --build-arg "KERNEL_SOURCE_COMMIT=$SOURCE_COMMIT" \
  --build-arg "KERNEL_PATCH_SHA256=$PATCH_SHA256" \
  --build-arg "KERNEL_PATCHED_TREE=$PATCHED_TREE" \
  --build-arg "XPU_C_SHA256=$XPU_C_SHA256" \
  --build-arg "RUNTIME_LIBS_SHA256=$RUNTIME_LIBS_SHA256" \
  --build-arg "SITE_CUSTOMIZE_SHA256=$SITE_CUSTOMIZE_SHA256" \
  --file "$SCRIPT_DIR/Dockerfile.runtime" \
  --tag "$IMAGE" "$SCRIPT_DIR"

docker image inspect "$IMAGE" >"$BUILD_ROOT/evidence/runtime-image.json"
printf '%s\n' \
  "image=$IMAGE" \
  "image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')" \
  "source_commit=$SOURCE_COMMIT" \
  "patched_tree=$PATCHED_TREE" \
  "wheel_sha256=$(sha256sum "$WHEEL" | awk '{print $1}')" \
  "xpu_c_sha256=$XPU_C_SHA256" \
  "runtime_libs_manifest_sha256=$RUNTIME_LIBS_SHA256" \
  "sitecustomize_sha256=$SITE_CUSTOMIZE_SHA256"
