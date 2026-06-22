#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B Quark W8A8 INT8 on vllm-xpu-env:v0230 (vLLM 0.23.0 -- NEWER than steve's
# 0.20.2rc1) at TP=2. Why v0230 (not the llm-scaler image): 0.23 ALREADY ships QuarkW8A8Int8MoEMethod
# AND a dynamic-per-token int8 LINEAR dispatch, AND it has the proven Triton fused-MoE path on XPU
# (our int4 35B-A3B serves here -- contrib/vllm_moe_xpu). The llm-scaler 0.14.1 image lacked the XPU
# MoE op suite (_moe_C) entirely. The ONLY gap left on v0230 is the int8 LINEAR scaled-mm kernel (no
# XPU entry -> KeyError), so we mount ONE patched quark.py that reroutes the int8 linear dispatch to a
# weight-only int8->bf16 dequant scheme (contrib/llm_scaler_quark_int8_moe/v0230/quark.py). The 256
# routed experts stay TRUE int8 via v0230's native QuarkW8A8Int8MoEMethod + Triton fused_experts.
# Route every GPU touch via gpu-run.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
NAME=vllm_quark35b_v0230
IMG="${IMG:-vllm-xpu-env:v0230}"
CKPT="${CKPT:-/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8}"   # CONTAINER path (models bind-mounted at /models)
SERVED="${SERVED:-qwen36-35b-a3b-quark-w8a8-int8}"
TP="${TP:-2}"; PORT="${PORT:-8000}"
PATCH_LIN="${PATCH_LIN:-$ROOT/patches/quark_v0230.py}"
# v0230 imports vLLM from /workspace/vllm/vllm (live) but a real copy also sits in site-packages;
# mount over BOTH so the patch applies regardless of which path resolves.
Q1=/workspace/vllm/vllm/model_executor/layers/quantization/quark/quark.py
Q2=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/quark/quark.py
LOGF="$ROOT/results/quark35b_v0230_tp${TP}.log"
mkdir -p "$ROOT/results" "$ROOT/hf_cache" "$ROOT/vllm_cache" "$ROOT/tmp_ssd"
[ -f "$PATCH_LIN" ] || { echo "[!] missing patched quark.py at $PATCH_LIN"; exit 2; }
docker rm -f "$NAME" 2>/dev/null || true

# TP=2 Battlemage multi-GPU stability env (vLLM #41663): no Arc P2P, pidfd IPC exchange, OFI, spawn.
MGPU=(-e CCL_ENABLE_SYCL_KERNELS="${SYCLKERNELS:-0}" -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
      -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -e CCL_TOPO_P2P_ACCESS="${P2PACCESS:-0}" -e CCL_ZE_IPC_EXCHANGE="${IPCX:-pidfd}")
MOUNT=(-v "$PATCH_LIN:$Q1:ro" -v "$PATCH_LIN:$Q2:ro")
[ "$TP" -gt 1 ] || MGPU=(-e ZE_AFFINITY_MASK="${DEVICE:-0}")   # TP=1 fallback (won't fit 1 card for 35GB int8)

echo "=== serve Quark-W8A8 INT8 35B  TP=$TP  on $IMG  (native int8 MoE + dequant-linear patch) ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g -p ${PORT}:${PORT} --pids-limit=-1 \
  --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  "${MOUNT[@]}" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  "${MGPU[@]}" --entrypoint vllm "$IMG" \
  serve "$CKPT" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
  --quantization quark --dtype auto --tensor-parallel-size "$TP" \
  $([ "$TP" -gt 1 ] && echo "--distributed-executor-backend mp") \
  --max-model-len "${MAXLEN:-8192}" --max-num-seqs "${MAXSEQS:-8}" --gpu-memory-utilization "${UTIL:-0.92}" \
  --no-enable-prefix-caching --trust-remote-code --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"video":0}' >/dev/null 2>&1

ok=0
for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY (see log)"; break; }
  sleep 5
done
docker logs "$NAME" > "$LOGF" 2>&1 || true

if [ "$ok" = 1 ]; then
  SID="$SERVED"   # host has no python3; served id == --served-model-name we set
  echo "=== HEALTHY -- int8 MoE 35B serves on v0230 at TP=$TP! served=$SID ==="
  echo "--- load + MoE-kernel confirmation ---"
  sed 's/\x1b\[[0-9;]*m//g' "$LOGF" | grep -iE "Model loading took|fused_moe|int8|Triton|world_size|tensor.parallel|Available KV|GiB|QuarkW8A8Int8" | tail -16
  echo "--- gen probe (greedy) ---"
  curl -s --max-time 40 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SID\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" | head -c 600; echo
  echo "=== concurrency sweep (in 2048 / out 128, c=1 2 4) ==="
  env NAME="$NAME" MODEL="$SID" LABEL="qwen36-35b-quark-w8a8-int8-v0230-tp${TP}" TOKPATH="$CKPT" PORT="$PORT" \
    IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 2 4}" bash "$ROOT/35_sweep_bench.sh" || true
else
  echo "=== NOT HEALTHY -- root cause ==="
  sed 's/\x1b\[[0-9;]*m//g' "$LOGF" | grep -iE "error|traceback|raise |exception|topk_softmax|_moe_C|Unsupported|No quark|KeyError|out of memory|assert|sycl|terminate|version counter" | tail -45
  echo "--- last 25 ---"; tail -25 "$LOGF"
  echo "(full log: $LOGF)"
fi
docker stop "$NAME" 2>/dev/null || true
echo "=== quark35b v0230 TP=$TP done -- log $LOGF ==="
