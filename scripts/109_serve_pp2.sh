#!/usr/bin/env bash
# 109_serve_pp2.sh -- PIPELINE-PARALLEL (PP=2) serve of 27B-W8A8 on dual B70, the J.13 bet.
#
# WHY: on our push-fast / read-slow cross-die fabric (J.2), TP=2 pays a per-layer all_reduce tax (~64-128
# collectives/forward). PP=2 needs only ONE point-to-point activation handoff per microbatch at the stage
# boundary -- a single posted push, the primitive our fabric is fastest at. See P2P_GPU.md J.13.
#
# STATUS: UNTESTED -- written 2026-06-24 while the box was H.13-wedged (J.15); MUST run on a freshly xe-reset
# box. lib.sh hardcodes --tensor-parallel-size and gates the #41663 multi-GPU env on TP>1, so PP can't go
# through it -- this is a self-contained serve (does NOT edit rdy_to_serve). No MTP first (PP+MTP is a later
# step); EAGER (GRAPH=0) to avoid the capture questions. Goal: confirm PP=2 serves coherently, then compare
# TTFT/decode/c8 vs TP=2 (H.7) -- does dropping 128 allreduces to 1 push/microbatch win?
#
# Usage: ./bin/gpu-run bash scripts/109_serve_pp2.sh start|stop|smoke
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-vllm-xpu-env:int8g}"
CKPT="${CKPT:-/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft}"
SERVED="${SERVED:-qwen36-27b-w8a8-pp2}"
PORT="${PORT:-8000}"
NAME="vllm_${SERVED}"
PP="${PP:-2}"; MAXLEN="${MAXLEN:-4096}"; MAXSEQS="${MAXSEQS:-8}"; UTIL="${UTIL:-0.90}"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so}"

# graceful teardown (wedge-guard L2): docker stop -t (SIGTERM+grace) before rm -f (SIGKILL), so a
# PP=2 worker is not force-killed mid-collective/init (the wedge trigger; P2P_GPU.md J.17).
stop() { docker stop -t "${STOP_GRACE:-30}" "$NAME" >/dev/null 2>&1 || true; docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME (graceful -t${STOP_GRACE:-30})"; }
case "${1:-start}" in stop) stop; exit 0 ;; esac

stop
echo "=== serve PP=$PP (TP=1) $SERVED  IMG=$IMG  port=$PORT  EAGER ==="
# PP=2 across 2 cards still needs the #41663 multi-GPU stability env + mp executor + spawn.
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g -p "${PORT}:${PORT}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  -e CCL_ENABLE_SYCL_KERNELS=0 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
  --entrypoint vllm "$IMG" serve "$CKPT" --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" --dtype auto \
  --pipeline-parallel-size "$PP" --tensor-parallel-size 1 --distributed-executor-backend mp \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL" \
  --no-enable-prefix-caching --trust-remote-code \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enforce-eager >/dev/null

echo "=== waiting for /health (up to ~15 min) ==="
for i in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "=== HEALTHY :$PORT $SERVED (PP=$PP) ==="; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[!] container died:"; docker logs "$NAME" 2>&1 | tail -25; exit 1; fi
  sleep 5
done

echo "--- gen probe ---"
curl -sf "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":24,\"temperature\":0}" \
  2>/dev/null | python3 -c 'import sys,json; print("gen:",json.load(sys.stdin)["choices"][0]["text"])' || echo "(probe failed)"

case "${1:-start}" in
  smoke) stop ;;
  start) echo "Serving. Bench: env NAME=$NAME MODEL=$SERVED LABEL=${SERVED} bash $ROOT/35_sweep_bench.sh ; stop: bash $0 stop" ;;
esac
