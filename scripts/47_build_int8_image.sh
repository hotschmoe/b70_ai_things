#!/usr/bin/env bash
# CONSOLIDATE: bake our INT8 W8A8 kernel into a reusable image `vllm-xpu-env:int8`, so any
# compressed-tensors W8A8-INT8 checkpoint serves with a plain `vllm serve` (no graft/patch dance).
# Bakes: our _xpu_C.so (int8_gemm_w8a8 + fused dynamic_per_token_int8_quant), XPUInt8ScaledMMLinearKernel,
# and the _POSSIBLE_INT8_KERNELS[XPU] + .get() chooser-hardening registry patch. Then serve-verifies.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SOPATH="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
D="$ROOT/contrib_int8"
MODEL="$ROOT/models/Qwen3-14B-W8A8-INT8"
BASE=vllm-xpu-env:v0230; IMG=vllm-xpu-env:int8
NAME=vllm_int8; PORT=18080

[ -f "$SOPATH" ] || { echo "MISSING $SOPATH"; exit 1; }
[ -f "$D/apply_patches.py" ] || { echo "MISSING $D/apply_patches.py (run scripts/45 first)"; exit 1; }

echo "=== bake the kernel into $IMG ==="
docker rm -f int8_img_build "$NAME" 2>/dev/null || true
docker run --name int8_img_build -v "$ROOT:$ROOT" --entrypoint bash "$BASE" -c '
  set -e
  cp -f '"$SOPATH"' /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so
  python '"$D"'/apply_patches.py
  python -c "import torch, vllm._xpu_ops; print(\"int8_gemm:\", hasattr(torch.ops._xpu_C,\"int8_gemm_w8a8\"), \"fused_quant:\", hasattr(torch.ops._xpu_C,\"dynamic_per_token_int8_quant\"))"
  echo BAKE_OK
'
[ "$(docker inspect -f '{{.State.ExitCode}}' int8_img_build)" = 0 ] || { echo "BAKE FAILED"; docker logs int8_img_build 2>&1 | tail -20; docker rm -f int8_img_build; exit 1; }
docker commit --change 'ENTRYPOINT []' \
  -m "INT8 W8A8 on Battlemage: int8_gemm_w8a8 + fused per-token int8 quant + XPUInt8ScaledMMLinearKernel + registry/.get hardening" \
  int8_img_build "$IMG"
docker rm -f int8_img_build
echo "=== image built ==="; docker images "$IMG"

echo "=== verify: serve W8A8 from the baked image (plain vllm serve, NO graft/patch) ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} -v "$ROOT:$ROOT" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e ZE_AFFINITY_MASK=0 -e VLLM_LOGGING_LEVEL=DEBUG \
  --entrypoint vllm "$IMG" \
  serve "$MODEL" --served-model-name qwen3-14b-w8a8 --host 0.0.0.0 --port ${PORT} \
    --dtype float16 --tensor-parallel-size 1 --enforce-eager --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --no-enable-prefix-caching --trust-remote-code

ok=0
for i in $(seq 1 144); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; break; }
  sleep 5
done
echo "=== verdict ==="
docker logs "$NAME" 2>&1 | grep -iE "Selected XPUInt8|Application startup complete|KeyError|error" | grep -viE "OperatorEntry|registered" | tail -8
[ "$ok" = 1 ] && echo "OK: $IMG serves W8A8 via our kernel with a plain vllm serve." || { echo "NOT HEALTHY"; docker logs "$NAME" 2>&1 | tail -20; }
echo "(leaving $NAME up for any check; stop with: docker rm -f $NAME)"
