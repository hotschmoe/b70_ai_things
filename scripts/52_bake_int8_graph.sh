#!/usr/bin/env bash
# Bake vllm-xpu-env:int8g = :int8 + the fake/meta registration for our custom int8 ops, so vLLM's
# XPU graph capture (VLLM_XPU_ENABLE_XPU_GRAPH=1) can trace through them. Purely additive: only swaps
# in the updated scaled_mm/xpu_int8.py (which lazily calls torch.library.register_fake for
# _xpu_C.dynamic_per_token_int8_quant + _xpu_C.int8_gemm_w8a8). Leaves :int8 untouched.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SRC="$ROOT/contrib_int8/xpu_int8.py"
BASE=vllm-xpu-env:int8; IMG=vllm-xpu-env:int8g
[ -f "$SRC" ] || { echo "MISSING $SRC"; exit 1; }
grep -q register_fake "$SRC" || { echo "FAIL: $SRC has no register_fake (wrong file?)"; exit 1; }

echo "=== bake $IMG (FROM $BASE, swap in fake-enabled xpu_int8.py) ==="
docker rm -f int8g_build 2>/dev/null || true
docker run --name int8g_build -v "$ROOT:$ROOT" --entrypoint bash "$BASE" -c '
  set -e
  # IMPORTANT: there are >1 vllm installs (editable /workspace/vllm/vllm AND site-packages). `import vllm`
  # resolves differently in the bake vs the serve process (HANDOFF gotcha) -> write to EVERY copy.
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
  -m "int8 + fake/meta kernels for _xpu_C int8 ops (unblocks VLLM_XPU_ENABLE_XPU_GRAPH=1 capture)" \
  int8g_build "$IMG"
docker rm -f int8g_build
echo "=== built ==="; docker images "$IMG"
