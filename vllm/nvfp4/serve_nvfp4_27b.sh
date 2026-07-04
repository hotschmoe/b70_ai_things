#!/usr/bin/env bash
# serve_nvfp4_27b.sh -- nvidia/Qwen3.6-27B-NVFP4 (ModelOpt MIXED_PRECISION checkpoint:
# NVFP4 W4A4 MLP + FP8 attention + bf16 norms/conv/vision/mtp, FP8 KV) on ONE B70 card.
#
# This is the GDN-hybrid VLM (Qwen3_5ForConditionalGeneration), unlike the dense 8B.
# So it ALWAYS needs the GDN attention kernel mounted (the stock image ships GDN off).
# vLLM v0.24.0 dispatches per-layer via ModelOptMixedPrecisionConfig: NVFP4 layers ->
# our XPU shim (patches/sitecustomize.py), FP8 layers -> vLLM's XPUFP8ScaledMMLinearKernel.
#
# EXACT single-card VRAM (measured from the real checkpoint):
#   keep-4bit resident (emul / fused): 21.9 GB  -> FITS one card + KV headroom
#   dequant NVFP4->int8 at load:       31.1 GB  -> does NOT fit (> ~30 GB card)
#   full bf16 dequant:                 56.7 GB  -> no
# So the ONLY viable FAST single-card path is keeping weights 4-bit in VRAM.
#
# Modes (NVFP4_XPU_MODE):
#   emul   - 4-bit resident, per-forward fp4 emulation. FITS (~22GB), COHERENT, but SLOW
#            (re-dequants every weight every forward). The fits+coherence reference.
#   fused  - 4-bit resident + custom E2M1 LUT dequant-GEMM kernel (the fast small-footprint
#            target; wired once the kernel lands).
#
#   CARD=0  PORT=8078  MAXLEN=2048   Run under `gpu-run --card 0` or hold the lease.
set -euo pipefail
ROOT=/mnt/vm_8tb/b70
REPO=/mnt/vm_8tb/github/b70_ai_things
DIR="$REPO/vllm/nvfp4"

IMG="${IMG:-vllm-xpu-env:int8g-v0240}"
NAME="${NAME:-nvfp4_27b}"
PORT="${PORT:-8078}"
CARD="${CARD:-0}"
MODE="${MODE:-emul}"
MAXLEN="${MAXLEN:-2048}"
UTIL="${UTIL:-0.92}"
MAXSEQS="${MAXSEQS:-4}"
SERVED="qwen3.6-27b-NVFP4-modelopt-${MODE}"

# GRAPH=1 -> PIECEWISE XPU graph capture (M6, 2026-07-04). Needs the register_fake
# for _xpu_C.nvfp4_gemm_w4a16 (in patches/sitecustomize.py, MODE=fused only) so
# dynamo can trace the custom op. IGP=false = the legacy piecewise splitter, the
# proven setting on this GDN hybrid (the inductor partitioner KeyErrors on mixed
# quant/no-quant regions -- see rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh).
# VLLM_USE_AOT_COMPILE=0 = the vision+capture fix (AOT serialize mishandles the
# optional mm inputs). Single card -> no TP collectives -> no SPLITOPS needed.
GRAPH="${GRAPH:-0}"
CGMODE="${CGMODE:-PIECEWISE}"
IGP="${IGP:-false}"
CAPSIZES="${CAPSIZES:-}"   # e.g. 1,2,4,8 -- REQUIRED with MTPTOK (spec decode balloons the
                           # default size list to [1..64]; the drafter + 24.1GiB resident
                           # weights then OOM at capture, UR err 40. c1 MTP only needs
                           # size (1+MTPTOK); larger batches fall back to eager).
                           # PROVEN fused+GRAPH+MTP combo (2026-07-04): UTIL=0.85 MAXLEN=4096
                           # MAXSEQS=8 CAPSIZES=1,2,4,8 -> 38.7 t/s c1. UTIL=0.88 loads +
                           # captures but OOMs (err 40) on the FIRST 2048-token prefill --
                           # do not raise UTIL past 0.85 on this 24.1GiB-resident serve.
GRAPH_ARGS=( --enforce-eager )
GRAPH_ENV=( )
if [ "$GRAPH" = 1 ]; then
  CAP=""; [ -n "$CAPSIZES" ] && CAP="\"cudagraph_capture_sizes\":[$CAPSIZES],"
  GRAPH_ARGS=( --compilation-config "{${CAP}\"cudagraph_mode\":\"$CGMODE\",\"use_inductor_graph_partition\":$IGP}" )
  GRAPH_ENV=( -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e VLLM_USE_AOT_COMPILE=0 )
  SERVED="${SERVED}-graph"
fi

# MTPTOK=N -> NEXTN MTP spec decode (M7). The ModelOpt ckpt natively carries the
# 15 bf16 mtp.* tensors and the quantized_layers map EXCLUDES mtp -> the drafter
# loads unquantized without the w8a8 shelf's graft/shim. Empty = MTP off.
# SWEEP (M9, 2026-07-04, card 0, code-probe t/s / random-c1 t/s): spec3 58.1/38.7-41.8,
# spec5 67.4/40.7-44.1 (WINNER both workloads), spec7 63.4/42.7. MTPTOK=5 is the best
# config -- unlike w8a8 (spec3-optimal @ 48% accept), the bf16 head on NVFP4 numerics
# drafts well enough that 5 pays (code accept ~1.00/1.00/0.97 at spec3).
MTPTOK="${MTPTOK:-}"
SPEC_ARGS=( )
if [ -n "$MTPTOK" ]; then
  SPEC_ARGS=( --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTPTOK}}" )
  SERVED="${SERVED}-mtp${MTPTOK}"
fi

# GDN attention kernel: required for the qwen3.6 hybrid (linear_attn layers). The
# w8a8_kernel_v0240 .so carries gdn_attention_core_xpu + int8_gemm_w8a16 + the GDN lib.
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/w8a8_kernel_v0240/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/w8a8_kernel_v0240/libgdn_attn_kernels_xe_2.so}"
[ -f "$GDN_SO" ] || { echo "MISSING GDN .so $GDN_SO"; exit 1; }
KERN_MOUNTS=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )

# fused mode: the GDN-ON .so carrying the custom nvfp4_gemm_w4a16 op (bit-exact NVFP4
# weight-decompression matmul: weights stay 4-bit/f4_e2m1 resident, dequant in the
# oneDNN JIT gemm -> 2.85x bf16 at decode). Same source tree as the GDN kernel, so it
# has BOTH gdn_attention_core AND nvfp4_gemm_w4a16. GDN_LIB sidecar from w8a8_kernel_v0240.
if [ "$MODE" = fused ]; then
  FUSED_SO="${FUSED_SO:-$ROOT/nvfp4_fused_kernel_gdn/_xpu_C.abi3.so}"
  [ -f "$FUSED_SO" ] || { echo "MISSING fused GDN kernel $FUSED_SO -- run the GDN-ON build first"; exit 1; }
  KERN_MOUNTS=( -v "$FUSED_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$REPO/models/files:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$DIR/patches:/opt/nvfp4_shim:ro" \
  "${KERN_MOUNTS[@]}" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTHONPATH=/opt/nvfp4_shim -e NVFP4_XPU_MODE="$MODE" \
  -e ZE_AFFINITY_MASK="$CARD" "${GRAPH_ENV[@]}" \
  --entrypoint vllm "$IMG" \
  serve /models/qwen3.6-27b/nvfp4-modelopt --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL" \
  "${GRAPH_ARGS[@]}" "${SPEC_ARGS[@]}" --no-enable-prefix-caching --trust-remote-code --skip-mm-profiling

echo "container $NAME up; follow with: docker logs -f $NAME"
echo "health: curl -s http://localhost:$PORT/health"
