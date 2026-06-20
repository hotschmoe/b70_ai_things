#!/usr/bin/env bash
# Serve Qwen3-14B-W4A8-gptq on the B70, EAGER (GRAPH=0) or PIECEWISE XPU graph capture (GRAPH=1).
# A1: measure whether PIECEWISE capture lifts w4a8 decode like it did w8a8 (+16.7%). The int4 decode
# path uses the custom op `int4_gemm_w4a8`; the rebaked :int8g now carries its register_fake (folded
# into xpu_int8.py), so dynamo can trace through it. GRAPH=1 mirrors the env that banked the w8a8 win:
# VLLM_XPU_ENABLE_XPU_GRAPH=1 + OMP + the pids/ulimit ceiling fix (capture spawns many threads).
# GPU run -- invoke via the gpu-run flock lease (long-lived server holds the lease for its lifetime).
#   Env: GRAPH (0|1), IMG (:int8g), MAXLEN (4096), MAXSEQS (4), UTIL (0.90), DTYPE (float16), OMP (8).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8g}"
MODEL="${MODEL:-/models/Qwen3-14B-W4A8-gptq}"
SERVED="${SERVED:-qwen3-14b-w4a8-gptq}"
GRAPH="${GRAPH:-0}"; MAXLEN="${MAXLEN:-4096}"; MAXSEQS="${MAXSEQS:-4}"; UTIL="${UTIL:-0.90}"
DTYPE="${DTYPE:-float16}"; OMP="${OMP:-8}"
# A2 knobs: CGMODE (PIECEWISE|FULL|FULL_AND_PIECEWISE), ATTN (attention backend, e.g. TRITON_ATTN -> FULL
# capture, since flash-attn FULL is blocked by SYCL-Graph work_group_scratch_memory). Default PIECEWISE.
CGMODE="${CGMODE:-PIECEWISE}"; ATTN="${ATTN:-}"
SPEC="${SPEC:-}"   # optional --speculative-config JSON (e.g. MTP: {"method":"qwen3_5_mtp","num_speculative_tokens":3})
NAME="${NAME:-vllm_w4a8}"; PORT=18080
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd}
docker rm -f vllm_qwen3 vllm_w4a8 vllm_w8a8 vllm_int8 "$NAME" 2>/dev/null || true

GRAPH_ENV=(); GRAPH_DOCKER=(); EAGER=(--enforce-eager); CC=(); ATTN_ENV=()
[ -n "$ATTN" ] && ATTN_ENV=(-e VLLM_ATTENTION_BACKEND="$ATTN")
if [ "$GRAPH" = 1 ]; then
  GRAPH_ENV=(-e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e OMP_NUM_THREADS="$OMP")
  GRAPH_DOCKER=(--pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556)
  EAGER=()
  # pass_config: force-disable the CUDA/ROCm-only inductor fusion passes. On XPU these classes are
  # NOT imported (vllm/compilation/passes/pass_manager.py gates the imports on is_cuda_alike()), but
  # under torch.compile their flags resolve None->True unguarded, so configure() references an
  # undefined class -> `NameError: MLARoPEKVCacheCatFusionPass is not defined` and the engine aborts.
  # These fusions can't run on XPU regardless; the graph CAPTURE (the decode lever) is independent of them.
  PASSCFG='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  CC=(--compilation-config "{\"cudagraph_mode\":\"$CGMODE\",\"use_inductor_graph_partition\":true,\"compile_sizes\":[1],$PASSCFG}")
fi

ARGS=(serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT"
      --dtype "$DTYPE" --tensor-parallel-size 1 --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS"
      --gpu-memory-utilization "$UTIL" --no-enable-prefix-caching --trust-remote-code "${EAGER[@]}" "${CC[@]}")
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")

echo "=== serve W4A8 GRAPH=$GRAPH cgmode=$([ "$GRAPH" = 1 ] && echo $CGMODE || echo eager) attn=${ATTN:-default} IMG=$IMG dtype=$DTYPE MAXLEN=$MAXLEN SEQS=$MAXSEQS ==="
echo "vllm ${ARGS[*]}"
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} "${GRAPH_DOCKER[@]}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK=0 -e VLLM_LOGGING_LEVEL=INFO \
  "${GRAPH_ENV[@]}" "${ATTN_ENV[@]}" --entrypoint vllm "$IMG" "${ARGS[@]}"

echo "=== waiting for readiness (up to ~14 min; first compile/capture slower) ==="
ok=0
for i in $(seq 1 168); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; break; }
  sleep 5
done
echo "=== model-id check (CLAUDE.md: verify served model) ==="
curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -oE '"id":"[^"]*"' | head -2
echo "=== capture + kernel confirmation ==="
docker logs "$NAME" 2>&1 | grep -iE 'registered fake.*int4|XPUW4A8|CompressedTensorsW4A8|Model loading took|saved AOT compiled|captur|cudagraph|Application startup complete|UnsupportedOperator|fake impl|cannot allocate memory|work_group_scratch|error|Traceback|out of memory' | grep -viE 'OperatorEntry|dispatch' | tail -28
[ "$ok" = 1 ] && echo "HEALTHY :$PORT '$SERVED' GRAPH=$GRAPH" || { echo "NOT HEALTHY"; docker logs "$NAME" 2>&1 | tail -30; }
