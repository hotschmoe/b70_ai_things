#!/usr/bin/env bash
# images/int8g/build.sh -- (re)build vllm-xpu-env:int8g. CANONICAL home for this image recipe.
# Supersedes scripts/52_bake_int8_graph.sh (now lab-notebook history). Runs ON THE GPU HOST.
#
# LINEAGE (see ../README.md):
#   vllm-xpu-env:v0230            base (vLLM 0.23.0+xpu)
#     -> :int8                    bake the oneDNN INT8 W8A8 GEMM .so (contrib/vllm_int8_xpu) + register it
#                                 as XPUInt8ScaledMMLinearKernel (apply_patches.py). scripts/47_build_int8_image.sh.
#     -> :int8g  (THIS)           + register_fake on the custom int8 ops so XPU graph capture can trace
#                                 them. Purely additive over :int8 (swaps in the fake-enabled xpu_int8.py).
#
# [!] ORGANIZATION.md image contract: tags are IMMUTABLE. This builds a DATED tag and also moves the
#     convenience tag :int8g to it. Record the digest in ../README.md + any rdy_to_serve dir that pins it.
#     (Full Dockerfile-ization of the whole chain incl. the compiled .so is a tracked follow-up; today the
#     base + :int8 are commit-built and the .so is a host binary from vllm-xpu-kernels/.)
set -uo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
BASE="${BASE:-vllm-xpu-env:int8}"
SRC="${SRC:-$ROOT/contrib_int8/xpu_int8.py}"   # register_fake-enabled kernel class (repo: contrib/vllm_int8_xpu/xpu_int8.py)
DATE="${DATE:-$(date +%Y%m%d 2>/dev/null || echo manual)}"
TAG="vllm-xpu-env:int8g-$DATE"

[ -f "$SRC" ] || { echo "MISSING $SRC"; exit 1; }
grep -q register_fake "$SRC" || { echo "FAIL: $SRC has no register_fake (wrong file?)"; exit 1; }
docker image inspect "$BASE" >/dev/null 2>&1 || { echo "FAIL: base $BASE not present (build :int8 first, scripts/47)"; exit 1; }

echo "=== bake $TAG (FROM $BASE, swap in fake-enabled xpu_int8.py) ==="
docker rm -f int8g_build 2>/dev/null || true
docker run --name int8g_build -v "$ROOT:$ROOT" --entrypoint bash "$BASE" -c '
  set -e
  # >1 vllm install (editable /workspace/vllm AND site-packages) -> write to EVERY copy (import-resolution gotcha).
  N=0
  for DST in $(find /workspace /opt/venv -path "*/vllm/model_executor/kernels/linear/scaled_mm/xpu_int8.py" 2>/dev/null); do
    cp -f '"$SRC"' "$DST"; echo "copied -> $DST"; N=$((N+1))
  done
  [ "$N" -ge 1 ] || { echo "FAIL: found no xpu_int8.py to overwrite"; exit 1; }
  python -c "import torch, vllm._xpu_ops; print(\"int8_gemm:\", hasattr(torch.ops._xpu_C,\"int8_gemm_w8a8\"), \"fused_quant:\", hasattr(torch.ops._xpu_C,\"dynamic_per_token_int8_quant\"))"
  echo BAKE_OK
'
[ "$(docker inspect -f '{{.State.ExitCode}}' int8g_build)" = 0 ] || { echo "BAKE FAILED"; docker logs int8g_build 2>&1 | tail -20; docker rm -f int8g_build; exit 1; }
docker commit --change 'ENTRYPOINT []' \
  -m "int8 + fake/meta kernels for _xpu_C int8 ops (XPU graph capture)" int8g_build "$TAG"
docker rm -f int8g_build
docker tag "$TAG" vllm-xpu-env:int8g            # convenience tag the recipes reference
echo "=== built ==="; docker images "$TAG" --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
echo "digest (record in README): $(docker image inspect --format '{{.Id}}' "$TAG")"
