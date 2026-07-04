#!/usr/bin/env bash
# serve_nvfp4.sh -- EXPERIMENT: nvidia/Qwen3-8B-NVFP4 (ModelOpt NVFP4 W4A4 checkpoint)
# on ONE B70 card via the XPU shim in patches/sitecustomize.py.
#
#   MODE=dequant ./vllm/nvfp4/serve_nvfp4.sh    # (default) dequant-at-load -> bf16 F.linear
#   MODE=emul    ./vllm/nvfp4/serve_nvfp4.sh    # true per-forward fp4 emulation (slow, reference)
#   CARD=0|1  PORT=8077  MAXLEN=8192  GPU lease: caller runs this under
#   `gpu-run --card $CARD` or holds the lease with `docker wait` like the DD does.
set -euo pipefail
ROOT=/mnt/vm_8tb/b70
REPO=/mnt/vm_8tb/github/b70_ai_things
DIR="$REPO/vllm/nvfp4"

IMG="${IMG:-vllm-xpu-env:int8g-v0240}"
NAME="${NAME:-nvfp4_xpu}"
PORT="${PORT:-8077}"
CARD="${CARD:-0}"
MODE="${MODE:-dequant}"          # dequant | emul | int8xmx
MAXLEN="${MAXLEN:-8192}"
UTIL="${UTIL:-0.90}"
MAXSEQS="${MAXSEQS:-4}"
SERVED="qwen3-8b-NVFP4-modelopt-${MODE}"

# int8xmx rides the K-group-fixed oneDNN int8 kernel; mount it over the pkg .so.
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
KERN_MOUNTS=()
if [ "$MODE" = int8xmx ]; then
  SO="${SO:-$ROOT/nvfp4_kernel/_xpu_C.abi3.so}"
  GDN_LIB="${GDN_LIB:-$ROOT/w8a8_kernel_v0240/libgdn_attn_kernels_xe_2.so}"
  [ -f "$SO" ] || { echo "MISSING $SO -- run build_nvfp4_kernel.sh first"; exit 1; }
  KERN_MOUNTS=( -v "$SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$REPO/models/files:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$DIR/patches:/opt/nvfp4_shim:ro" \
  ${KERN_MOUNTS[@]+"${KERN_MOUNTS[@]}"} \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTHONPATH=/opt/nvfp4_shim -e NVFP4_XPU_MODE="$MODE" \
  -e ZE_AFFINITY_MASK="$CARD" \
  --entrypoint vllm "$IMG" \
  serve /models/qwen3-8b/nvfp4-modelopt --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL" \
  --enforce-eager --no-enable-prefix-caching --trust-remote-code

echo "container $NAME up; follow with: docker logs -f $NAME"
echo "health:    curl -s http://localhost:$PORT/health"
echo "smoke:     curl -s http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' \\"
echo "             -d '{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":16}'"
