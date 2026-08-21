#!/usr/bin/env bash
# Qwen3.8-27B W4A8-gptq-gdn on sglang Path H (NOT a shelf).
# Campaign K: docs/20260820_qwen38_w4a8_campaign.md. compressed-tensors
# two-group (INT4 Linear + INT8 GDN). GRAPH=0 first. Do NOT torch.compile
# act-quant (D05). P2PACCESS=0. Keep vLLM NOMTP :18082 if running.
#
#   GRAPH=0 PORT=18083 NAME=qwen38_w4a8_sgl DEVICE=1 \
#     ./bin/gpu-run --card 1 bash sglang/serve_qwen38_w4a8.sh start
#   NAME=qwen38_w4a8_sgl bash sglang/serve_qwen38_w4a8.sh stop
set -uo pipefail
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
ACTION="${1:-start}"
NAME="${NAME:-qwen38_w4a8_sgl}"
IMG="${IMG:-sglang-xpu:mtp}"
CKPT="${CKPT:-/models/qwen3.8-27b/w4a8-gptq-gdn}"
SERVED="${SERVED:-qwen3.8-27b-W4A8-gptq-sglang}"
PORT="${PORT:-18083}"
DEVICE="${DEVICE:-${CARD:-1}}"
GRAPH="${GRAPH:-0}"
CTX="${CTX:-4096}"
MEMFRAC="${MEMFRAC:-0.85}"
KERNEL_DIR="${KERNEL_DIR:-$ROOT/w4a8_kernel}"
SHIMS="${SHIMS:-$REPO/sglang/patches}"
SITE=/opt/venv/lib/python3.12/site-packages
LOG="${LOG:-$REPO/results/logs/sglang_qwen38_w4a8_card${DEVICE}.log}"

if [ "$ACTION" = stop ]; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

[ -f "$KERNEL_DIR/_xpu_C.abi3.so" ] || { echo "MISSING $KERNEL_DIR/_xpu_C.abi3.so"; exit 1; }
[ -f "$SHIMS/woq_shim.py" ] || { echo "MISSING $SHIMS/woq_shim.py"; exit 1; }
[ -f "$SHIMS/w4a8_shim.py" ] || { echo "MISSING $SHIMS/w4a8_shim.py"; exit 1; }
[ -f "$SHIMS/w8a8_shim.py" ] || { echo "MISSING $SHIMS/w8a8_shim.py"; exit 1; }

gflags=()
genv=(-e B70_XPU_W4A8=1 -e B70_XPU_W8A8=1 -e B70_W4A8_COMPILE=0)
if [ "$GRAPH" = 1 ]; then
  genv+=(-e B70_XPU_CUDAGRAPH=1)
  gflags=(--cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1 --max-running-requests 1)
  echo "=== GRAPH=1 XPUGraph bs=1 ==="
else
  echo "=== GRAPH=0 eager ==="
fi

echo "=== sglang 3.8 W4A8 Path H SERVED=$SERVED GRAPH=$GRAPH DEVICE=$DEVICE PORT=$PORT ==="
echo "=== IMG=$IMG CKPT=$CKPT COMPILE=0 P2PACCESS=0 ==="
mkdir -p "$REPO/results/logs" "$ROOT/hf_cache" "$ROOT/sgl_cache"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -e ZE_AFFINITY_MASK="$DEVICE" \
  -e CUDA_VISIBLE_DEVICES="" \
  -v "$REPO/models/files:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -v "$KERNEL_DIR:/work/w4a8_kernel:ro" \
  -v "$SHIMS/woq_shim.py:$SITE/woq_shim.py:ro" \
  -v "$SHIMS/w4a8_shim.py:$SITE/w4a8_shim.py:ro" \
  -v "$SHIMS/w8a8_shim.py:$SITE/w8a8_shim.py:ro" \
  -v "$SHIMS/w4a8_actquant_triton.py:$SITE/w4a8_actquant_triton.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache \
  -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_C_SO=/work/w4a8_kernel/_xpu_C.abi3.so \
  "${genv[@]}" \
  --entrypoint bash \
  "$IMG" \
  -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH
      exec python -m sglang.launch_server \
        --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
        --device xpu --dtype "${DTYPE:-bfloat16}" --attention-backend triton --linear-attn-backend triton \
        --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 \
        --disable-radix-cache --skip-server-warmup \
        ${gflags[*]:-} \
        --tp 1 --context-length $CTX --mem-fraction-static $MEMFRAC \
        --host 0.0.0.0 --port $PORT"

echo "waiting /health :$PORT (ceiling ~900s)"
ok=0
for i in $(seq 1 180); do
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || {
    echo "CONTAINER EXITED"
    docker logs "$NAME" > "$LOG" 2>&1
    tail -40 "$LOG"
    exit 1
  }
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = 200 ]; then ok=1; echo "HEALTHY :$PORT $SERVED (~$((i*5))s)"; break; fi
  sleep 5
done
docker logs "$NAME" > "$LOG" 2>&1
[ "$ok" = 1 ] || { echo "NOT healthy; see $LOG"; tail -40 "$LOG"; exit 1; }
echo "serving $SERVED on :$PORT log=$LOG"
