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
# CAPSIZES: explicit cudagraph_capture_sizes (comma list, e.g. "1,2,4,8,16,32"). Default capture tops out at
# 8 -> batches >8 fall back to eager (the N=16 serving-throughput cliff). Capturing 16/32 is a FREE capacity bump.
CAPSIZES="${CAPSIZES:-}"
NAME="${NAME:-vllm_w4a8}"; PORT=18080
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd}
docker rm -f vllm_qwen3 vllm_w4a8 vllm_w8a8 vllm_int8 "$NAME" 2>/dev/null || true

GRAPH_ENV=(); GRAPH_DOCKER=(); EAGER=(--enforce-eager); CC=(); ATTN_ENV=()
[ -n "$ATTN" ] && ATTN_ENV=(-e VLLM_ATTENTION_BACKEND="$ATTN")
# TRITONSHIM=1: inject a sitecustomize.py that warms torch.xpu.device_count() in EVERY spawned process,
# so triton's is_active() (= torch.xpu.is_available()) returns True in the engine worker -> Triton enabled
# -> TRITON_ATTN + the Triton rejection sampler work (unblocks FULL capture + MTP-positive).
SHIM_ARGS=()
[ -n "${TRITONSHIM:-}" ] && SHIM_ARGS=(-v "$ROOT/triton_shim:/opt/triton_shim:ro" -e PYTHONPATH=/opt/triton_shim)
# PREPACK=1: serve an offline-prepacked W4A8 model (int32-packed weights on disk) -- mount the patched
# loader+kernel (env-gated VLLM_W4A8_PREPACKED) so vLLM loads the small packed weights directly (no 28 GiB
# unpacked-I8 GPU transient). Makes the quality (GDN-bf16) 27B-W4A8 a true 1-card model.
PREPACK_ARGS=()
if [ -n "${PREPACK:-}" ]; then
  _KP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py
  _SP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py
  PREPACK_ARGS=(-e VLLM_W4A8_PREPACKED=1 -v "$ROOT/patches/xpu.py:$_KP:ro" -v "$ROOT/patches/compressed_tensors_w4a8_int.py:$_SP:ro")
fi
# KERNEL_SO: mount a rebuilt _xpu_C.abi3.so over the baked one (e.g. the GDN-enabled build for the 27B
# gated-delta-net decode op, which the fast int8-only build omits).
KERNEL_SO_ARGS=()
if [ -n "${KERNEL_SO:-}" ]; then
  _PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
  KERNEL_SO_ARGS=(-v "$KERNEL_SO:$_PKGD/_xpu_C.abi3.so:ro")
  # also mount any sibling lib*.so the extension dlopens (e.g. libgdn_attn_kernels_xe_2.so from GDN=ON build)
  for _lib in "$(dirname "$KERNEL_SO")"/lib*.so; do
    [ -f "$_lib" ] && KERNEL_SO_ARGS+=(-v "$_lib:$_PKGD/$(basename "$_lib"):ro")
  done
fi
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
  CAPCFG=""; [ -n "$CAPSIZES" ] && CAPCFG="\"cudagraph_capture_sizes\":[$CAPSIZES],"
  # compile_sizes default [1]; with MTP/spec-decode the decode batch pads to 1+num_spec, so [1] is rejected
  # ("would be padded to 2") -> set COMPILESZ=2 (or empty to omit) for spec-decode serves.
  _CS="${COMPILESZ-1}"; COMPILECFG=""; [ -n "$_CS" ] && COMPILECFG="\"compile_sizes\":[$_CS],"
  CC=(--compilation-config "{\"cudagraph_mode\":\"$CGMODE\",\"use_inductor_graph_partition\":true,${CAPCFG}${COMPILECFG}$PASSCFG}")
fi

ARGS=(serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT"
      --dtype "$DTYPE" --tensor-parallel-size 1 --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS"
      --gpu-memory-utilization "$UTIL" --no-enable-prefix-caching --trust-remote-code "${EAGER[@]}" "${CC[@]}")
# ATTN as a CLI flag (the VLLM_ATTENTION_BACKEND env var is deprecated/ignored on current vLLM main, which
# is likely why TRITON_ATTN never engaged before -> it silently fell back to flash-attn = PIECEWISE-only).
# --attention-backend TRITON_ATTN is the verified path to FULL cudagraph capture on XPU (vLLM PR #34482).
[ -n "$ATTN" ] && ARGS+=(--attention-backend "$ATTN")
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
# NOMM=1: text-only serve of a VLM (Qwen3_5) -- disallow image/video so vLLM skips the vision-encoder
# dummy profiling (which crashes on XPU: qwen2_5_vl.py "not enough values to unpack"). The vision tower is
# still loaded but unused; this is the right mode for text inference on the 27B.
[ -n "${NOMM:-}" ] && ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0}')
# KVDTYPE: store the KV cache in fp8 (fp8_e5m2 = no scales, simplest) -> halves KV BW (long-ctx decode win)
# + 2x context/batch capacity. B70 has no FP8 ALU, so this is fp8-STORAGE + dequant-on-read in attention.
[ -n "${KVDTYPE:-}" ] && ARGS+=(--kv-cache-dtype "$KVDTYPE")

echo "=== serve W4A8 GRAPH=$GRAPH cgmode=$([ "$GRAPH" = 1 ] && echo $CGMODE || echo eager) attn=${ATTN:-default} IMG=$IMG dtype=$DTYPE MAXLEN=$MAXLEN SEQS=$MAXSEQS ==="
echo "vllm ${ARGS[*]}"
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} "${GRAPH_DOCKER[@]}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK=0 -e VLLM_LOGGING_LEVEL=INFO \
  "${GRAPH_ENV[@]}" "${ATTN_ENV[@]}" "${SHIM_ARGS[@]}" "${PREPACK_ARGS[@]}" "${KERNEL_SO_ARGS[@]}" --entrypoint vllm "$IMG" "${ARGS[@]}"

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
