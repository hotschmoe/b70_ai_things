#!/usr/bin/env bash
# Clean A/B for B1: rebuilt-UNPATCHED .so vs rebuilt-PATCHED .so, SAME scripts/44 toolchain
# (isolates the patch from the baked-0.1.9-vs-rebuilt-0.1.11 build confound). Reuses the host
# b1_validate.py (deterministic symmetric inputs). 2 interleaved rounds to expose run-to-run noise.
# NOTE: no `docker run -i` (python is a mounted file, not piped) -- `-i` would slurp this script's stdin.
# GPU run -- invoke via the gpu-run flock lease.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; IMG="${IMG:-vllm-xpu-env:int8}"
BAKED=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so
UNPATCHED="$ROOT/b1_unpatched.so"; PATCHED="$ROOT/b1_patched.so"
# refresh the unpatched .so from whatever scripts/44 last built (must be the unpatched control)
cp "$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so" "$UNPATCHED"
echo "unpatched=$(stat -c%s "$UNPATCHED")B  patched=$(stat -c%s "$PATCHED")B  (sizes equal is fine)"
runso(){  # $1=label $2=so
  docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0 \
    -e MODE="$1" -e ITERS="${ITERS:-200}" -e WARMUP="${WARMUP:-40}" \
    -v "$ROOT/vllm_cache:/vllm_cache" -e XDG_CACHE_HOME=/vllm_cache \
    -v "$ROOT/b1_validate.py:/b1_validate.py" -v "$2:$BAKED:ro" \
    --entrypoint python "$IMG" /b1_validate.py 2>&1 | grep -E "^SHAPE"
}
for r in 1 2; do
  echo "### round $r UNPATCHED"; runso "u$r" "$UNPATCHED"
  echo "### round $r PATCHED";   runso "p$r" "$PATCHED"
done
echo "### DONE"
