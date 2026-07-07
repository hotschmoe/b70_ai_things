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
TP="${TP:-1}"              # TP=2 -> both cards, oneCCL multicard env (lib.sh recipe), 2x KV pool.
MODE="${MODE:-emul}"
MAXLEN="${MAXLEN:-2048}"
UTIL="${UTIL:-0.92}"
# MAXBATCH: --max-num-batched-tokens (chunked-prefill chunk size). Empty = vLLM default (8192 online).
# A LONG cold prefill is sliced into ceil(prompt/MAXBATCH) forward steps, each firing all 128 TP
# all-reduces; a larger chunk cuts the AR-launch count (prefill is per-collective-overhead-bound) at the
# cost of a bigger prefill activation transient (bounded by MAXBATCH). Stacks on the push-AR overlay for
# 32K/128K prefills. Leave empty for IN<=8192 (already a single chunk). Track 11g.
MAXBATCH="${MAXBATCH:-}"
MAXBATCH_ARG=(); [ -n "$MAXBATCH" ] && MAXBATCH_ARG=( --max-num-batched-tokens "$MAXBATCH" )
# MAXSEQS: TP>1 defaults to 8. [!] CORRECTNESS (2026-07-04): on the TP=2 + MTP + GRAPH path, MAXSEQS
# MUST be >= the largest cudagraph capture size (see the guard by CAPSIZES below). MAXSEQS=4 with
# CAPSIZES=1,2,4,8 emits "!!!!" NaN garbage on ~half of concurrent streams (gate 8-10/18 FAIL) while
# MAXSEQS=8 is clean (gate 18/18) at the SAME MAXLEN=131072 -- the small cap starves the size-8 spec
# graph the MTP verify batch pads to. Single-card default stays 4 (no spec-verify all_gather batch).
_MAXSEQS_DEF=$([ "$TP" != 1 ] && echo 8 || echo 4)
MAXSEQS="${MAXSEQS:-$_MAXSEQS_DEF}"
# PREFIXCACHE: --enable-prefix-caching on this hybrid GDN model needs the mamba align-mode int64
# pointer fix (shim block 6, VLLM_XPU_MAMBA_PTR_FIX, default on) or engine init overflows on XPU USM
# addresses. Default OFF to preserve the single-card shelf behavior; the TP=2 long-context daily driver
# sets PREFIXCACHE=1 for warm-prefix reuse (skips the full re-prefill on repeated long prefixes).
PREFIXCACHE="${PREFIXCACHE:-0}"
if [ "$PREFIXCACHE" = 1 ]; then PC_ARG="--enable-prefix-caching"; else PC_ARG="--no-enable-prefix-caching"; fi
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
# SPLITOPS (TP>1 + capture only): list the model's genuinely non-capturable attention/GDN custom ops
# as compilation splitting_ops so inductor cuts the piecewise graph there. Needed for MTP-on-TP2: the
# spec-verify all_gather (which oneCCL cannot record) is handled by the capture-safe all-reduce-of-padded
# shim in patches/sitecustomize.py block (3), so EJECT NOTHING here -- split only at the attn/GDN ops.
# Single-card (TP=1) keeps the byte-identical no-splitting_ops config that M6-M9 gated in.
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
GRAPH_ARGS=( --enforce-eager )
GRAPH_ENV=( )
if [ "$GRAPH" = 1 ]; then
  # CORRECTNESS guard: TP>1 + MTP + capture emits NaN "!!!!" if MAXSEQS < the largest capture size
  # (the size-8 spec-verify graph gets starved). Bump MAXSEQS up to cover it. Verified 2026-07-04.
  if [ "$TP" != 1 ] && [ -n "$MTPTOK" ] && [ -n "$CAPSIZES" ]; then
    _MAXCAP=$(echo "$CAPSIZES" | tr ',' '\n' | sort -n | tail -1)
    if [ "$MAXSEQS" -lt "$_MAXCAP" ] 2>/dev/null; then
      echo "[guard] TP>1+MTP+capture: raising MAXSEQS $MAXSEQS -> $_MAXCAP (largest CAPSIZES; <it emits NaN garbage)" >&2
      MAXSEQS="$_MAXCAP"
    fi
  fi
  CAP=""; [ -n "$CAPSIZES" ] && CAP="\"cudagraph_capture_sizes\":[$CAPSIZES],"
  SPLIT=""; PASSCFG=""
  if [ "$TP" != 1 ]; then
    SPLIT="\"splitting_ops\":[${SPLITOPS:-$_ATTN_OPS}],"
    # XPU bug (v0.24.0 pass_manager.py): under the TP>1 fusion path pass_config.fuse_rope_kvcache_cat_mla
    # resolves True (raw default is None -> skipped, which is why the single-card path never hits it), but
    # MLARoPEKVCacheCatFusionPass is imported only under is_cuda_alike() -> NameError at compile. Qwen3.6 is
    # not MLA, so this pass is a no-op for it -> disable it explicitly on the TP>1 capture path.
    PASSCFG="\"pass_config\":{\"fuse_rope_kvcache_cat_mla\":false},"
  fi
  GRAPH_ARGS=( --compilation-config "{${CAP}${SPLIT}${PASSCFG}\"cudagraph_mode\":\"$CGMODE\",\"use_inductor_graph_partition\":$IGP}" )
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

# Agentic parity (daily-driver: pi.dev / omp.sh / hermes harnesses). Qwen3.6 emits XML <tool_call>
# (qwen3_coder parser, NOT hermes JSON) + <think> reasoning (qwen3 parser -> reasoning_content). Same
# parsers as the vLLM/sglang W8A8 shelf entries, validated on v0.24.0. Default OFF -> single-card shelf
# + bare research serve unchanged; the DD sets TOOLCALL=1. API_KEY (or VLLM_API_KEY): if set, vLLM
# ENFORCES it on /v1/* (--api-key); /health stays open. Used by the DD (DD_API_KEY) for WAN exposure.
AGENTIC_ARGS=( )
if [ "${TOOLCALL:-0}" = 1 ]; then
  AGENTIC_ARGS+=( --enable-auto-tool-choice --tool-call-parser "${TOOLPARSER:-qwen3_coder}" )
fi
[ -n "${REASONPARSER:-}" ] && AGENTIC_ARGS+=( --reasoning-parser "$REASONPARSER" )
_APIKEY="${API_KEY:-${VLLM_API_KEY:-}}"
[ -n "$_APIKEY" ] && AGENTIC_ARGS+=( --api-key "$_APIKEY" )

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

# push all-reduce PREFILL overlay (TP>1 only, Track 11g). Routes big EAGER prefill all-reduces through the
# L0-IPC posted-write (push) collective (vllm/contrib/vllm_push_allreduce) instead of oneCCL to cut cold
# TTFT (J.21: 3.8x prefill TTFT on the sibling W8A8 TP=2 serve). Host-barrier .so + PUSH_AR_MIN_NUMEL above
# the largest CAPTURED all-reduce (max(CAPSIZES)*hidden = 8*5120 = 40960; the MTP all_gather shim block(3)
# uses dist.all_reduce directly so it is NOT intercepted) and below the eager prefill all-reduce (IN*5120)
# -> captured decode stays on graph-recordable oneCCL, only big eager prefill all-reduces use push. The .so
# is pure C-ABI SYCL/L0 (no libtorch link) so the committed prebuilt works on int8g-v0240. P2PACCESS stays
# 0 (push-ar does its OWN L0-IPC peer write, independent of CCL_TOPO_P2P_ACCESS -> dodges the H.13 wedge).
# Loaded once via sitecustomize block (8). Default OFF -> single-card shelf + default TP=2 byte-identical.
PUSH_AR="${PUSH_AR:-0}"
PUSH_AR_MOUNTS=( ); PUSH_AR_ENV=( )
if [ "$PUSH_AR" = 1 ] && [ "$TP" != 1 ]; then
  PUSH_AR_DIR="$REPO/vllm/contrib/vllm_push_allreduce"
  # PUSH_AR_GRAPH=1 (Track 11g, ported from the W8A8 shelf 2026-07-05): route the DECODE all-reduce through
  # the CAPTURABLE push path too (libxpu_push_ar_graph.so ar_allreduce_graph -> device-side L0-event sync
  # recorded into torch's XPUGraph, P2P_GPU.md K.6/K.8), MIN_NUMEL=0 so every all-reduce (prefill AND
  # captured decode) uses the 11 GB/s posted-write fabric -- no oneCCL decode fallback. PUSH_AR_GRAPH=0
  # (default here) = the proven prefill-only overlay (host-barrier .so; decode-sized all-reduces -> oneCCL).
  # HARD GATE: the graph path records INTO an XPU graph, so it REQUIRES GRAPH=1 + CGMODE!=NONE (a real
  # capture). Without a captured graph the captured-decode push reads a stale/uninit buffer -> input-
  # dependent "!!!!" garbage (W8A8 J.17: NONE+GRAPH=1 emitted garbage on some prompts). So force
  # PUSH_AR_GRAPH=0 unless capture is on; the patch itself only takes the graph branch while _is_capturing().
  PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-0}"
  if [ "$PUSH_AR_GRAPH" = 1 ] && { [ "$GRAPH" != 1 ] || [ "$CGMODE" = NONE ]; }; then
    echo "[guard] PUSH_AR_GRAPH=1 needs GRAPH=1 + CGMODE!=NONE (captured decode); GRAPH=$GRAPH CGMODE=$CGMODE -> forcing PUSH_AR_GRAPH=0 (prefill-only)" >&2
    PUSH_AR_GRAPH=0
  fi
  if [ "$PUSH_AR_GRAPH" = 1 ]; then
    _PA_SO_NAME="libxpu_push_ar_graph.so"; _PA_MIN_DEF=0
  else
    _PA_SO_NAME="libxpu_push_ar_torch.so"; _PA_MIN_DEF=65536
  fi
  PUSH_AR_SO_HOST="$PUSH_AR_DIR/prebuilt/$_PA_SO_NAME"   # committed prebuilt (graph OR host-barrier)
  [ -f "$PUSH_AR_SO_HOST" ] || { echo "MISSING push-ar .so $PUSH_AR_SO_HOST (set PUSH_AR=0 for oneCCL)"; exit 1; }
  PUSH_AR_MOUNTS=( -v "$PUSH_AR_DIR:/opt/push_ar:ro" )
  PUSH_AR_ENV=( -e PUSH_AR_PATCH=/opt/push_ar/_push_ar_patch.py
                -e PUSH_AR_SO="/opt/push_ar/prebuilt/$_PA_SO_NAME"
                -e PUSH_AR_DISABLE=0
                -e PUSH_AR_GRAPH="$PUSH_AR_GRAPH"
                -e PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-$_PA_MIN_DEF}"
                -e PUSH_AR_MAXB="${PUSH_AR_MAXB:-134217728}" )
  # PUSH_AR_MAXB (scratch bytes, default 128 MiB): CAPS the prefill chunk that push can accelerate. An
  # all-reduce whose bytes exceed MAXB falls back to oneCCL. chunk_tokens*hidden*2 <= MAXB -> with hidden
  # 5120 the default 128 MiB fits chunks up to ~13,107 tokens; the default 8192 chunk (83.9 MB) is inside
  # it, but raising --max-num-batched-tokens past ~13k WITHOUT also raising PUSH_AR_MAXB silently DEFEATS
  # push-AR (measured 2026-07-04: MAXBATCH=16384 -> 167 MB > 128 MiB -> oneCCL -> 32K PP 734 vs 1772).
  if [ "$PUSH_AR_GRAPH" = 1 ]; then
    echo "=== PUSH_AR overlay ON (GRAPH decode push: graph .so, MIN_NUMEL=${PUSH_AR_MIN_NUMEL:-0}, P2PACCESS=0) ==="
    SERVED="${SERVED}-pushargraph"
  else
    echo "=== PUSH_AR prefill overlay ON (host-barrier .so, MIN_NUMEL=${PUSH_AR_MIN_NUMEL:-65536}, P2PACCESS=0) ==="
    SERVED="${SERVED}-pushar"
  fi
fi

# KV_FP8=0 -> DISABLE the checkpoint's fp8 KV cache and run bf16 KV (kv_cache_dtype=auto).
# WHY (JOURNAL 2026-07-06, Track 11h root-cause): the ModelOpt NVFP4 checkpoint declares
# quantization_config.kv_cache_scheme={num_bits:8,type:float} in config.json AND ships NO calibrated
# k/v scales -- vLLM logs "Using KV cache scaling factor 1.0 for fp8_e4m3". On this hybrid GDN/attention
# model that uncalibrated fp8 KV loses precision that ACCUMULATES with generation length -> coherent
# early, degenerate repetition late (the pi/openwebui "...community community..." collapse). Proven by
# bisection: bf16 KV ran 3500 forced tokens clean where fp8 KV looped by ~tok985. We overlay a patched
# config.json (kv_cache_scheme stripped; config_groups/NVFP4 weight quant untouched). Default 1 = stock fp8.
# Manual override: KV_CONFIG_OVERRIDE=/host/config.json (takes precedence).
KV_MOUNTS=( )
if [ -n "${KV_CONFIG_OVERRIDE:-}" ]; then
  [ -f "$KV_CONFIG_OVERRIDE" ] || { echo "MISSING KV_CONFIG_OVERRIDE $KV_CONFIG_OVERRIDE"; exit 1; }
  KV_MOUNTS=( -v "$KV_CONFIG_OVERRIDE:/models/qwen3.6-27b/nvfp4-modelopt/config.json:ro" )
  echo "=== KV_CONFIG_OVERRIDE ON -> $KV_CONFIG_OVERRIDE ==="
elif [ "${KV_FP8:-1}" = 0 ]; then
  _CFG_SRC="$REPO/models/files/qwen3.6-27b/nvfp4-modelopt/config.json"
  _CFG_PATCH="${KV_PATCH_DIR:-/tmp}/b70_config.nvfp4.nokvfp8.json"
  python3 -c "import json; d=json.load(open('$_CFG_SRC')); d.get('quantization_config',{}).pop('kv_cache_scheme',None); json.dump(d,open('$_CFG_PATCH','w'))" \
    || { echo "failed to generate $_CFG_PATCH"; exit 1; }
  KV_MOUNTS=( -v "$_CFG_PATCH:/models/qwen3.6-27b/nvfp4-modelopt/config.json:ro" )
  echo "=== KV_FP8=0: fp8 KV cache DISABLED (bf16 KV, kv_cache_scheme stripped) ==="
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

# card/multicard env: TP=1 pins one card; TP=2 drives both through CPU-staged oneCCL
# (Battlemage stability set from rdy_to_serve/_common/lib.sh: no Arc P2P, OFI transport,
# spawn workers, LZ v1 adapter; CCL_ENABLE_SYCL_KERNELS=1 when capturing).
TP_ARGS=( )
if [ "$TP" = 1 ]; then
  MGPU=( -e ZE_AFFINITY_MASK="$CARD" ); SHM=16g
else
  SK=$([ "$GRAPH" = 1 ] && echo 1 || echo 0)
  MGPU=( -e CCL_ENABLE_SYCL_KERNELS="$SK" -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
         -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi
         -e CCL_TOPO_P2P_ACCESS="${P2PACCESS:-0}" -e CCL_ZE_IPC_EXCHANGE="${IPCX:-pidfd}" )
  SHM=32g
  TP_ARGS=( -tp "$TP" )
fi

# Diagnostic + arbitrary-env passthrough (mirrors the w8a8 recipe, added 2026-07-07 for the NEO
# abort fix campaign): B70_DEBUG=1 -> faulthandler (dumps a py+C traceback on SIGABRT, which is how
# this crash dies silently); B70_EXTRA_ENV="A=1 B=2" -> inject arbitrary -e (e.g. B70_XPU_DRAFTER_EAGER=1,
# B70_XPU_CG_RECYCLE_STEPS=N, NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=...). Default-off => the
# shelf serve is byte-identical.
EXTRA_ENV=( )
if [ "${B70_DEBUG:-0}" != 0 ]; then EXTRA_ENV+=( -e PYTHONFAULTHANDLER=1 -e PYTHONUNBUFFERED=1 ); echo "=== B70_DEBUG=${B70_DEBUG} -> faulthandler ===" >&2; fi
if [ -n "${B70_EXTRA_ENV:-}" ]; then for kv in ${B70_EXTRA_ENV}; do EXTRA_ENV+=( -e "$kv" ); done; echo "=== B70_EXTRA_ENV -> ${B70_EXTRA_ENV} ===" >&2; fi
# B70_EXTRA_MOUNTS="host:container[:ro] ..." -> extra -v binds (e.g. overlay the patched
# libtorch_xpu.so over the prebuilt one to validate the submit_without_event NEO-abort fix).
EXTRA_MOUNTS=( )
if [ -n "${B70_EXTRA_MOUNTS:-}" ]; then for m in ${B70_EXTRA_MOUNTS}; do EXTRA_MOUNTS+=( -v "$m" ); done; echo "=== B70_EXTRA_MOUNTS -> ${B70_EXTRA_MOUNTS} ===" >&2; fi

docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size "$SHM" -p "${PORT}:${PORT}" \
  -v "$REPO/models/files:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$DIR/patches:/opt/nvfp4_shim:ro" \
  "${KERN_MOUNTS[@]}" "${PUSH_AR_MOUNTS[@]}" "${KV_MOUNTS[@]}" "${EXTRA_MOUNTS[@]}" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTHONPATH=/opt/nvfp4_shim -e NVFP4_XPU_MODE="$MODE" \
  "${MGPU[@]}" "${GRAPH_ENV[@]}" "${PUSH_AR_ENV[@]}" "${EXTRA_ENV[@]}" \
  --entrypoint vllm "$IMG" \
  serve /models/qwen3.6-27b/nvfp4-modelopt --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL" "${MAXBATCH_ARG[@]}" \
  "${TP_ARGS[@]}" "${GRAPH_ARGS[@]}" "${SPEC_ARGS[@]}" "${AGENTIC_ARGS[@]}" "$PC_ARG" --trust-remote-code --skip-mm-profiling

echo "container $NAME up; follow with: docker logs -f $NAME"
echo "health: curl -s http://localhost:$PORT/health"
