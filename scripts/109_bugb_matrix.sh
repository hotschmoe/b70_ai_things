#!/usr/bin/env bash
# Bug B isolation matrix -- parameterized launcher + COHERENCE READ-OUT (reads the actual text,
# never token metrics; that false-positive is the whole reason Bug B existed).
#
# The captured-garbage repros so far (scripts/106,108) ALL eject the TP collectives via splitting_ops.
# Nobody tested 27B W8A8 TP=2 captured with the collectives LEFT INSIDE the captured graph after the
# ignore-list fix (the old 18.74 "baseline" was measured pre-fix and never read). This script fills
# every missing cell.
#
# Knobs (env):
#   CKPT     model dir under /models           (default 27B W8A8 graft body)
#   SERVED   served-model-name                 (default derived)
#   TP       tensor-parallel-size              (default 2)
#   GRAPH    1=PIECEWISE capture, 0=eager      (default 1)
#   EJECT    1=eject collectives to eager via splitting_ops, 0=leave them captured (default 1)
#   IGP      use_inductor_graph_partition      (default false)
#   MAXLEN   max-model-len                     (default 4096)
#   UTIL     gpu-memory-utilization            (default 0.90)
#   PORT     (default 8000)
#   NAME     container name                    (default vllm_bugb)
# Run under the GPU lease:  /mnt/vm_8tb/b70/gpu-run bash scripts/109_bugb_matrix.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
CKPT="${CKPT:-/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft}"
SERVED="${SERVED:-bugb}"
TP="${TP:-2}"
GRAPH="${GRAPH:-1}"
EJECT="${EJECT:-1}"
IGP="${IGP:-false}"
MAXLEN="${MAXLEN:-4096}"
UTIL="${UTIL:-0.90}"
PORT="${PORT:-8000}"
NAME="${NAME:-vllm_bugb}"
DTYPE="${DTYPE:-auto}"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so
GDN_LIB=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so

# attention + GDN custom ops -- ALWAYS split points (not capturable)
ATTN_BASE='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
COLL='"vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
if [[ "$EJECT" == "1" ]]; then
  SPLIT="$ATTN_BASE,$COLL"
else
  SPLIT="$ATTN_BASE"
fi
PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'

if [[ "$GRAPH" == "1" ]]; then
  CC="{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":${IGP},\"cudagraph_capture_sizes\":[1,2,4,8],\"splitting_ops\":[$SPLIT],$PASS}"
  CC_ARGS=(--compilation-config "$CC")
else
  CC_ARGS=(--enforce-eager)
fi

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g -p "${PORT}:${PORT}" --pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache -e TMPDIR=/tmp_ssd \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e VLLM_LOGGING_LEVEL=INFO -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e OMP_NUM_THREADS=8 \
  -e CCL_ENABLE_SYCL_KERNELS=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
  --entrypoint vllm vllm-xpu-env:int8g \
  serve "$CKPT" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
  --dtype "$DTYPE" --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --max-num-seqs 8 --gpu-memory-utilization "$UTIL" \
  --no-enable-prefix-caching --trust-remote-code --distributed-executor-backend mp \
  --limit-mm-per-prompt '{"image":0,"video":0}' "${CC_ARGS[@]}" >/dev/null

echo "launched $NAME :: CKPT=$CKPT TP=$TP GRAPH=$GRAPH EJECT=$EJECT IGP=$IGP MAXLEN=$MAXLEN"
echo "waiting for /health (up to 600s)..."
ok=0
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "CONTAINER DIED. last logs:"; docker logs --tail 40 "$NAME" 2>&1; exit 1
  fi
  sleep 5
done
if [[ "$ok" != "1" ]]; then echo "HEALTH TIMEOUT. last logs:"; docker logs --tail 60 "$NAME" 2>&1; exit 1; fi
echo "HEALTHY. coherence probe (temp=0):"

probe() {
  # host has NO python3 -> dump raw JSON (content is inline + readable) and try docker-exec python for clean extract
  local p="$1"
  local resp
  resp=$(curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED}\",\"messages\":[{\"role\":\"user\",\"content\":\"${p}\"}],\"temperature\":0,\"max_tokens\":160}")
  # clean extract via the container's python (always present in the vllm image)
  echo "$resp" | docker exec -i "$NAME" python3 -c 'import sys,json; d=json.load(sys.stdin); print("CONTENT:",repr(d["choices"][0]["message"]["content"]))' 2>/dev/null \
    || { echo "RAW: ${resp:0:1500}"; }
}
echo "--- Q1 capital of France ---"
probe "What is the capital of France? Answer in one sentence."
echo "--- Q2 fibonacci ---"
probe "Write a Python function that returns the nth Fibonacci number."
echo "=== END PROBE ($NAME) ==="
if [[ "${KEEP:-0}" != "1" ]]; then
  echo "tearing down $NAME (KEEP=1 to keep it running)"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi
