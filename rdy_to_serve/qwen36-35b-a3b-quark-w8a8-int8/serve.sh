#!/usr/bin/env bash
# Self-contained serve: Qwen3.6-35B-A3B Quark W8A8 INT8 (int8 MoE) on 2x B70, TP=2.
# Everything needed is in THIS directory (serve.sh + patches/quark.py). Run ON THE GPU HOST.
#
#   # from the host (ssh root@192.168.10.5), with this dir synced to the host:
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh            # acquire GPU lease, start, wait healthy, gen-probe
#   bash serve.sh stop                                # stop + remove container (release GPU)
#   bash serve.sh logs                                # follow logs
#   bash serve.sh bench                               # concurrency sweep vs the running server
#   GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run # serve + bench + stop in one lease-held measurement
#
# [!] IMAGE: vllm-xpu-env:v0230 = vLLM 0.23.0. DO NOT downgrade to intel/llm-scaler-vllm:0.14.x
#     (ancient 0.14 fork, no _moe_C -> int8 MoE hard-fails). Prefer any NEWER vLLM-XPU image if it exists.
#
# Why this works (short): vLLM 0.23 already ships QuarkW8A8Int8MoEMethod + the dynamic-per-token int8
# LINEAR dispatch, and routes the 256 int8 experts through the Triton fused_moe_kernel on XPU. The ONE
# gap is the int8 LINEAR scaled-mm kernel (no XPU entry -> KeyError); patches/quark.py reroutes the int8
# linear layers (linear_attn.*, mlp.shared_expert.*) to a weight-only int8->bf16 dequant GEMM. Experts
# stay TRUE int8. See ./README.md.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"                         # GPU host: models, gpu-run, 35_sweep_bench, caches
IMG="${IMG:-vllm-xpu-env:v0230}"                        # vLLM 0.23.0  [!] do not downgrade
CKPT="${CKPT:-/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8}" # container path (models bind-mounted at /models)
SERVED="${SERVED:-qwen36-35b-a3b-quark-w8a8-int8}"
NAME="${NAME:-vllm_quark35b_v0230}"
TP="${TP:-2}"; PORT="${PORT:-8000}"
GRAPH="${GRAPH:-0}"                                     # 1 = graph capture (decode perf lever)
CGMODE="${CGMODE:-PIECEWISE}"                           # PIECEWISE works on v0230. FULL_DECODE_ONLY is BLOCKED on
                                                        # stock v0230: SYCL-Graph scratch-memory limit (needs a patched image).
PATCH="$SCRIPT_DIR/patches/quark.py"
Q1=/workspace/vllm/vllm/model_executor/layers/quantization/quark/quark.py
Q2=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/quark/quark.py
mkdir -p "$ROOT/hf_cache" "$ROOT/vllm_cache" "$ROOT/tmp_ssd" "$ROOT/results"

case "${1:-start}" in
  stop) docker rm -f "$NAME" 2>/dev/null; echo "stopped $NAME (GPU released)"; exit 0;;
  logs) exec docker logs -f "$NAME";;
  bench)
    env NAME="$NAME" MODEL="$SERVED" LABEL="${SERVED}-tp${TP}$([ "$GRAPH" = 1 ] && echo -graph)" \
      TOKPATH="$CKPT" PORT="$PORT" IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 2 4}" \
      bash "$ROOT/35_sweep_bench.sh"
    exit 0;;
esac

[ -f "$PATCH" ] || { echo "[!] missing patch: $PATCH"; exit 2; }
docker rm -f "$NAME" 2>/dev/null || true

# TP>1: Battlemage multi-GPU stability env (vLLM #41663) -- no Arc P2P, pidfd IPC, OFI, spawn. For GRAPH
# capture at TP=2, CCL_ENABLE_SYCL_KERNELS=1 makes the oneCCL allreduce graph-capturable (eager uses 0).
if [ "$TP" -gt 1 ]; then
  SK="${SYCLKERNELS:-$([ "$GRAPH" = 1 ] && echo 1 || echo 0)}"
  MGPU=(-e CCL_ENABLE_SYCL_KERNELS="$SK" -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
        -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e CCL_TOPO_P2P_ACCESS="${P2PACCESS:-0}" -e CCL_ZE_IPC_EXCHANGE="${IPCX:-pidfd}")
  DEB=(--distributed-executor-backend mp)
else
  MGPU=(-e ZE_AFFINITY_MASK="${DEVICE:-0}"); DEB=()
fi

GDOCK=(--pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556)
if [ "$GRAPH" = 1 ]; then
  # Graph capture. pass_config disables CUDA-only fusion passes that NameError on XPU.
  # FULL_DECODE_ONLY: capture ONLY the decode step; prefill runs eager -> no mid-serve prefill
  # recompile stall at concurrency>1 (PIECEWISE breaks at c>1). PIECEWISE: capture whole step.
  GENV=(-e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e OMP_NUM_THREADS="${OMP:-8}")
  PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  CAP=""; [ -n "${CAPSIZES:-}" ] && CAP="\"cudagraph_capture_sizes\":[$CAPSIZES],"
  if [ "$CGMODE" = PIECEWISE ]; then
    EXTRA="\"use_inductor_graph_partition\":true,\"compile_sizes\":[1],"
  else
    EXTRA=""   # FULL_DECODE_ONLY: let vLLM auto-split (gdn_attention_core etc.); no inductor partition
  fi
  EAGER=(); CC=(--compilation-config "{\"cudagraph_mode\":\"$CGMODE\",${EXTRA}${CAP}$PASS}")
else
  GENV=(); EAGER=(--enforce-eager); CC=()
fi

echo "=== serve $SERVED  TP=$TP  GRAPH=$GRAPH  IMG=$IMG ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g -p ${PORT}:${PORT} "${GDOCK[@]}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$PATCH:$Q1:ro" -v "$PATCH:$Q2:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  "${MGPU[@]}" "${GENV[@]}" --entrypoint vllm "$IMG" \
  serve "$CKPT" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
  --quantization quark --dtype auto --tensor-parallel-size "$TP" "${DEB[@]}" \
  --max-model-len "${MAXLEN:-8192}" --max-num-seqs "${MAXSEQS:-8}" --gpu-memory-utilization "${UTIL:-0.92}" \
  --no-enable-prefix-caching --trust-remote-code "${EAGER[@]}" "${CC[@]}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' >/dev/null 2>&1

echo "=== waiting for /health (up to ~15 min; first run JIT-compiles the Triton MoE kernel) ==="
ok=0
for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; docker logs "$NAME" 2>&1 | tail -30; exit 1; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "NOT HEALTHY"; docker logs "$NAME" 2>&1 | tail -30; exit 1; }

echo "=== HEALTHY -- $SERVED on http://<host>:$PORT/v1 (TP=$TP GRAPH=$GRAPH) ==="
echo "--- gen probe ---"
curl -s --max-time 60 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":24,\"temperature\":0}" | head -c 600; echo

if [ "${1:-start}" = run ]; then
  # WARMUP (graph capture only): a throwaway sweep so the one-time lazy torch.compile per batch
  # shape happens here (server finishes the compile even if the warmup's client requests time out),
  # so the MEASURED sweep below doesn't stall at c>1. Spoof for the PIECEWISE c>1 recompile break.
  if [ "$GRAPH" = 1 ] && [ "${WARMUP:-1}" = 1 ]; then
    echo "=== warmup sweep (absorb per-batch-shape recompiles) ==="
    bash "${BASH_SOURCE[0]}" bench >/dev/null 2>&1 || true
  fi
  echo "=== bench (measured) ==="; bash "${BASH_SOURCE[0]}" bench
  docker stop "$NAME" 2>/dev/null; echo "stopped (GPU released)"
else
  echo "Serving. Stop with: bash serve.sh stop  (holds the GPU until then)."
fi
