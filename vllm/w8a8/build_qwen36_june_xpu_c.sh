#!/usr/bin/env bash
# Reconstruct the minimal June 9 Qwen3.6 Quark W8A8 XPU extension.
#
# This is adapted from Steve Seguin's xpu-C-only build loop. It fetches the
# official upstream base and applies the attributed patch stored in this repo;
# it does not read or mount Steve's checkout. No GPU device is mounted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$RUNTIME_ROOT/steve-repro/june-xpuc-bmg-g21-a0-$STAMP}"
IMAGE="${IMAGE:-intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94}"
UPSTREAM="${UPSTREAM:-https://github.com/vllm-project/vllm-xpu-kernels.git}"
BASE_COMMIT="${BASE_COMMIT:-28e1f5e74c15744b69cf3b760f6160ceabd15de0}"
PATCH_FILE="$REPO_ROOT/kernels/steve_qwen36_quark_w8a8_20260609.patch"
PATCH_SHA256="14c2e801da02a7b46e63940dbe41f5c0c45fabb98b3ee4c5bd03d7dc7d0b1266"
PATCHED_TREE="c882c446d8ea47b6a9cee8aa5d16ee5121b8cd1f"
AOT_DEVICES="${AOT_DEVICES:-bmg-g21-a0}"
JOBS="${JOBS:-2}"

if [[ ! "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "IMAGE must be pinned by a 64-character sha256 digest: $IMAGE" >&2
  exit 2
fi
case "$JOBS" in
  ''|*[!0-9]*)
    echo "JOBS must be a positive integer, got: $JOBS" >&2
    exit 2
    ;;
esac
if [ "$JOBS" -le 0 ]; then
  echo "JOBS must be positive, got: $JOBS" >&2
  exit 2
fi
case "$AOT_DEVICES" in
  ''|*[!A-Za-z0-9,._-]*)
    echo "AOT_DEVICES contains unsupported characters: $AOT_DEVICES" >&2
    exit 2
    ;;
esac

if [ ! -f "$PATCH_FILE" ]; then
  echo "Missing owned patch: $PATCH_FILE" >&2
  exit 2
fi

mkdir -p "$(dirname "$RUN_DIR")"
if ! mkdir "$RUN_DIR"; then
  echo "Refusing to overwrite existing RUN_DIR: $RUN_DIR" >&2
  exit 2
fi
mkdir "$RUN_DIR/evidence" "$RUN_DIR/logs" "$RUN_DIR/install"
git -C "$RUN_DIR" init -q source
git -C "$RUN_DIR/source" remote add origin "$UPSTREAM"
git -C "$RUN_DIR/source" fetch --depth=1 origin "$BASE_COMMIT" \
  2>&1 | tee "$RUN_DIR/logs/fetch.log"
git -C "$RUN_DIR/source" checkout --detach FETCH_HEAD \
  2>&1 | tee "$RUN_DIR/logs/checkout.log"
git -C "$RUN_DIR/source" submodule update --init --recursive \
  2>&1 | tee "$RUN_DIR/logs/submodules.log"

actual_patch_sha="$(sha256sum "$PATCH_FILE" | awk '{print $1}')"
if [ "$actual_patch_sha" != "$PATCH_SHA256" ]; then
  echo "Patch hash mismatch: $actual_patch_sha" >&2
  exit 3
fi
git -C "$RUN_DIR/source" apply --check "$PATCH_FILE"
git -C "$RUN_DIR/source" apply --index "$PATCH_FILE"
git -C "$RUN_DIR/source" diff --cached --check
cp "$PATCH_FILE" "$RUN_DIR/evidence/"
git -C "$RUN_DIR/source" show -s --format='%H%n%T%n%cI%n%s' HEAD \
  >"$RUN_DIR/evidence/base-source-identity.txt"
git -C "$RUN_DIR/source" diff --cached --binary \
  >"$RUN_DIR/evidence/applied-source.patch"
actual_tree="$(git -C "$RUN_DIR/source" write-tree)"
printf '%s\n' "$actual_tree" >"$RUN_DIR/evidence/patched-tree.txt"
if [ "$actual_tree" != "$PATCHED_TREE" ]; then
  echo "Patched tree mismatch: $actual_tree" >&2
  exit 3
fi
sha256sum "$RUN_DIR/evidence/applied-source.patch" \
  >"$RUN_DIR/evidence/applied-source.patch.sha256"
docker pull "$IMAGE" 2>&1 | tee "$RUN_DIR/logs/image-pull.log"
docker image inspect "$IMAGE" >"$RUN_DIR/evidence/docker-image-inspect.json"

docker run --rm --user "$(id -u):$(id -g)" \
  -e VLLM_XPU_AOT_DEVICES="$AOT_DEVICES" \
  -e VLLM_XPU_XE2_AOT_DEVICES="$AOT_DEVICES" \
  -e B70_BUILD_JOBS="$JOBS" \
  -v "$RUN_DIR:/repro" --entrypoint /bin/bash "$IMAGE" -lc '
set -eo pipefail
set +u
source /opt/intel/oneapi/setvars.sh --force > /repro/logs/oneapi-setvars.log 2>&1
set -u
python_path="$(python3 - <<'"'"'PY'"'"'
import sys
print(":".join(sys.path))
PY
)"

cmake -S /repro/source -B /repro/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/repro/source/cmake/toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=/repro/install \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVLLM_TARGET_DEVICE=xpu \
  -DVLLM_PYTHON_EXECUTABLE="$(command -v python3)" \
  -DVLLM_PYTHON_PATH="$python_path" \
  -DFETCHCONTENT_BASE_DIR=/repro/source/.deps \
  -DBUILD_SYCL_TLA_KERNELS=ON \
  -DVLLM_XPU_ENABLE_XE2=ON \
  -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF \
  -DBASIC_KERNELS_ENABLED=OFF \
  -DFA2_KERNELS_ENABLED=OFF \
  -DMOE_KERNELS_ENABLED=ON \
  -DGDN_KERNELS_ENABLED=ON \
  -DMQA_LOGITS_KERNELS_ENABLED=OFF \
  -DXPU_SPECIFIC_KERNELS_ENABLED=ON \
  -DXPUMEM_ALLOCATOR_ENABLED=OFF \
  2>&1 | tee /repro/logs/configure.log

cmake -E copy /repro/build/CMakeCache.txt /repro/evidence/CMakeCache.txt
cmake -E copy /repro/build/compile_commands.json \
  /repro/evidence/compile_commands.json
cmake --build /repro/build -j "$B70_BUILD_JOBS" --target _xpu_C \
  2>&1 | tee /repro/logs/build.log
cmake --install /repro/build --prefix /repro/install --component _xpu_C \
  2>&1 | tee /repro/logs/install.log

# The June patch changes this Python dispatcher as well as native code. Keep it
# beside the native replacement set. A complete importable package is
# materialized below because fused_moe_interface imports both _C and _xpu_C;
# a partial PYTHONPATH package would silently disable fused MoE.
install -D -m 0644 /repro/source/vllm_xpu_kernels/fused_moe_interface.py \
  /repro/install/vllm_xpu_kernels/fused_moe_interface.py
install -D -m 0644 /repro/source/LICENSE /repro/install/LICENSE

for library in libgdn_attn_kernels_xe_2.so libgrouped_gemm_xe_2.so; do
  source_library="/repro/build/$library"
  if [ ! -f "$source_library" ]; then
    echo "Enabled sibling library is missing: $source_library" >&2
    exit 4
  fi
  install -D -m 0755 "$source_library" \
    "/repro/install/vllm_xpu_kernels/$library"
done

base_package=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
runtime_root=/repro/install/runtime_package
runtime_package="$runtime_root/vllm_xpu_kernels"
if [ ! -d "$base_package" ]; then
  echo "Pinned-image base package is missing: $base_package" >&2
  exit 4
fi
find "$base_package" -type f -print0 | sort -z | xargs -0 sha256sum \
  > /repro/evidence/base-package-sha256.txt
mkdir -p "$runtime_root"
cp -a "$base_package" "$runtime_package"
find "$runtime_package" -type f -name '*.pyc' -delete
for replacement in \
  /repro/install/vllm_xpu_kernels/_xpu_C*.so \
  /repro/install/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so \
  /repro/install/vllm_xpu_kernels/libgrouped_gemm_xe_2.so; do
  if [ ! -f "$replacement" ]; then
    echo "Runtime-package replacement is missing: $replacement" >&2
    exit 4
  fi
  install -m 0755 "$replacement" "$runtime_package/$(basename "$replacement")"
done
install -m 0644 /repro/install/vllm_xpu_kernels/fused_moe_interface.py \
  "$runtime_package/fused_moe_interface.py"

find /repro/install -type f -name "*.so*" -print0 | sort -z | \
  xargs -0 sha256sum > /repro/evidence/artifact-sha256.txt
find /repro/install -type f -name "*.so*" -printf "%s %p\n" | sort -n \
  > /repro/evidence/artifact-sizes.txt
extension="$(find "$runtime_package" -maxdepth 1 -type f -name "_xpu_C*.so" -print -quit)"
if [ -z "$extension" ]; then
  echo "Built _xpu_C extension is missing" >&2
  exit 4
fi
artifact_lib="$runtime_package"
torch_lib=/opt/venv/lib/python3.12/site-packages/torch/lib
export LD_LIBRARY_PATH="$artifact_lib:$torch_lib:${LD_LIBRARY_PATH:-}"
readelf -n "$extension" > /repro/evidence/xpu-c-notes.txt
readelf -d "$extension" > /repro/evidence/xpu-c-dynamic.txt
ldd "$extension" > /repro/evidence/xpu-c-ldd.txt
if grep -q 'not found' /repro/evidence/xpu-c-ldd.txt; then
  echo "Built extension has unresolved dynamic dependencies" >&2
  cat /repro/evidence/xpu-c-ldd.txt >&2
  exit 4
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$runtime_root" python3 - "$extension" \
  /repro/evidence/operator-census.json <<'"'"'PY'"'"'
import hashlib
import json
from pathlib import Path
import sys

import torch
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _moe_C, _xpu_C, fused_moe_interface


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

extension = sys.argv[1]
required_rebuilt = [
    "_xpu_C::cutlass_grouped_gemm_w8a8_int8_interface",
    "_xpu_C::int8_gemm_w8a8",
    "_xpu_C::per_token_quant_int8_xpu",
]
required_inherited = [
    "_C::silu_and_mul",
    "_moe_C::init_expert_map",
    "_moe_C::remap_hidden_states",
    "_moe_C::moe_gather",
]
required = required_rebuilt + required_inherited
ops = sorted(
    name for name in torch._C._dispatch_get_all_op_names()
    if name.startswith(("_C::", "_moe_C::", "_xpu_C::"))
)
operator_details = {}
for name in required:
    if name not in ops:
        continue
    schema = torch._C._dispatch_find_schema_or_throw(name, "").schema()
    operator_details[name] = {
        "schema": str(schema),
        "has_xpu_kernel": torch._C._dispatch_has_kernel_for_dispatch_key(
            name, "XPU"
        ),
    }
missing = sorted(set(required) - set(ops))
missing_xpu = sorted(
    name for name in required
    if name in operator_details and not operator_details[name]["has_xpu_kernel"]
)
native_modules = {
    "_C": file_identity(_C.__file__),
    "_moe_C": file_identity(_moe_C.__file__),
    "_xpu_C": file_identity(_xpu_C.__file__),
}
expected_package = Path(extension).resolve().parent
native_origins_match = {
    name: Path(identity["path"]).parent == expected_package
    for name, identity in native_modules.items()
}
xpu_origin_matches = (
    Path(native_modules["_xpu_C"]["path"]) == Path(extension).resolve()
)
python_origins_match = (
    Path(vllm_xpu_kernels.__file__).resolve().parent == expected_package
    and Path(fused_moe_interface.__file__).resolve().parent == expected_package
)
document = {
    "extension": extension,
    "torch_version": torch.__version__,
    "package_origin": vllm_xpu_kernels.__file__,
    "fused_moe_origin": fused_moe_interface.__file__,
    "fused_moe_available": fused_moe_interface.FUSEDMOE_AVAILABLE,
    "fused_moe_unavailable_reason": (
        fused_moe_interface.FUSEDMOE_UNAVAILABLE_REASON
    ),
    "native_modules": native_modules,
    "native_origins_match_runtime_package": native_origins_match,
    "python_origins_match_runtime_package": python_origins_match,
    "xpu_origin_matches_selected_extension": xpu_origin_matches,
    "operators": ops,
    "operator_details": operator_details,
    "required_rebuilt": required_rebuilt,
    "required_inherited": required_inherited,
    "missing": missing,
    "missing_xpu_dispatch": missing_xpu,
}
Path(sys.argv[2]).write_text(json.dumps(document, indent=2) + "\n")
if (
    missing
    or missing_xpu
    or not all(native_origins_match.values())
    or not python_origins_match
    or not xpu_origin_matches
    or not fused_moe_interface.FUSEDMOE_AVAILABLE
):
    raise SystemExit(
        "runtime-package census failed: "
        f"missing={missing} missing_xpu={missing_xpu} "
        f"native_origins_match={native_origins_match} "
        f"python_origins_match={python_origins_match} "
        f"xpu_origin_matches={xpu_origin_matches} fused_moe_available="
        f"{fused_moe_interface.FUSEDMOE_AVAILABLE} reason="
        f"{fused_moe_interface.FUSEDMOE_UNAVAILABLE_REASON}"
    )
PY

find "$runtime_package" -type f -print0 | sort -z | xargs -0 sha256sum \
  > /repro/evidence/runtime-package-sha256.txt
find /repro/install -type f -print0 | sort -z | xargs -0 sha256sum \
  > /repro/evidence/complete-install-sha256.txt

printf "oneDNN %s\n" \
  "$(git -C /repro/source/third_party/oneDNN rev-parse HEAD)" \
  > /repro/evidence/fetchcontent-revisions.txt
for dependency in /repro/source/.deps/*-src; do
  if [ -d "$dependency/.git" ]; then
    printf "%s %s\n" "$(basename "$dependency")" \
      "$(git -C "$dependency" rev-parse HEAD)"
  fi
done | sort >> /repro/evidence/fetchcontent-revisions.txt
' 2>&1 | tee "$RUN_DIR/logs/container.log"

echo "result -> $RUN_DIR"
