#!/usr/bin/env bash
# Qwen3.6-27B W8A8 (compressed-tensors INT8 w x INT8 a, SmoothQuant+GPTQ) + BF16 MTP graft, TP=2 + MTP spec=5.
# Coherent single-stream 27B int8-activation serve. EAGER ~9 t/s MTP-on (2.3x vs ~4 t/s MTP-off), 48% accept.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
# [!! 2026-07-03 -- vLLM UN-PAUSED, rebased to v0.24.0 (torch 2.12). The 5 hybrid mixed-prefill+decode PRs
#     (#44700 split mixed -> recurrent GDN, #43990/#42430/#43961/#43556) FIX the "!!!!" concurrent garbage
#     that paused vLLM: gate 40/40 + 32/32 coherent under staggered mixed load on BOTH CGMODE=NONE and
#     PIECEWISE, and PIECEWISE is now STABLE under sustained MTP (restarts=0) + DETERMINISTIC (3/3 greedy).
#     USAGE-BASED A/B (best config = CGMODE=PIECEWISE + PUSH_AR=1, spec=3, MAXLEN=8192, text): single-stream
#     40.6 tok/s, N=8 agg 163.8 -- 2.3x the sglang daily driver's 18.0 (captured 74ms fwd-pass vs eager 267ms).
#     Defaults below now point at the v0.24.0 stack. Kernel .so rebuilt vs torch 2.12 (int8+GDN combined),
#     ABI-verified. Build: ../../../vllm/build_v0240_base.sh + build_v0240_int8gdn_so.sh + images/int8g/bake_v0240.sh.
#     Gate: ../../../vllm/gate_concurrent_coherence.py + perf_probe.py. JOURNAL 2026-07-03. Daily-driver SWITCH
#     still needs agentic parity (MAXLEN 131072, tool/reason parsers, VISION NOMM=0) + a 131K sweep -- TODO.]
#
# [!!! 2026-06-23 CORRECTION -- scripts/101-106]: the OLD "63-64 t/s / 3.4x captured" headline was a FALSE POSITIVE
#   benched on GARBAGE. Two stacked bugs were found+fixed/characterized:
#   (A) ROOT CAUSE of the garbage (FIXED, config-only, no requant): the checkpoint config.json `ignore` list had 336
#       ENUMERATED leaf names with the wrong flat prefix `model.layers.N.linear_attn.*` -- they do NOT match the
#       VLM-nested keys `model.language_model.layers.N.*`, so the 48 GDN linear_attn layers (correctly stored BF16)
#       were NOT exempted -> built as W8A8, BF16 silently shape-matched int8, scale missing -> 48 recurrent garbage
#       layers -> degenerate output. The degenerate body made draft==target -> trivial ~98% accept / accept_len 5.9
#       (the bench counted tokens, never read text). FIX = regex ignore (scripts/104); weights were always good
#       (dequant cos 0.97-0.9999 vs base). Eager now COHERENT.
#   (B) CAPTURED PATH STILL BROKEN: with correct BF16 GDN, PIECEWISE capture is numerically broken on this TP=2 hybrid
#       (IGP=true -> KeyError weight_scale crash; IGP=false -> serves but garbage even WITHOUT MTP; clean-cache
#       confirmed). 14B W8A8 captures fine single-card, so it's a TP=2 + BF16-GDN-in-captured-pieces + int8 capture
#       bug. -> recipe DEFAULTS TO EAGER (GRAPH=0) for correctness. `GRAPH=1` reproduces the broken captured path.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh          # start (both cards), wait healthy, COHERENCE-GATED gen-probe
#   bash serve.sh stop                              # stop + release both GPUs
#
# [!] IMAGE: vllm-xpu-env:int8g  (custom oneDNN s8s8s32 INT8 W8A8 GEMM = XPUInt8ScaledMMLinearKernel, auto-selected
#     from the compressed-tensors W8A8-int8 config; plus register_fake so XPU graph capture can trace the int8 ops).
# [!] 35 GiB int8 weights -> needs TP=2 (does NOT fit one 32 GB card). TP collectives run CPU-staged oneCCL over
#     PCIe (no Battlemage P2P) -- LATENCY-bound, which is exactly what MTP hides (1 collective round per ~6 verified
#     tokens instead of per token), so MTP is a BIGGER win at TP=2 than single-card.
#
# THREE things beyond a plain W8A8 serve, all wired below:
#  (1) GDN: Qwen3.6-27B is gated-delta-net; the :int8g baked kernel ships GDN OFF -> mount the GDN-enabled
#      _xpu_C.abi3.so (+ sibling libgdn_attn_kernels_xe_2.so) over the baked one (host kernel build).
#  (2) MTP graft: CKPT is the *-mtp-graft dir (15 BF16 mtp.* tensors added). Mount patches/sitecustomize.py on
#      PYTHONPATH so ONLY the MTP drafter is built unquantized/BF16 (else it loads through the W8A8 quant path -> 0% accept).
#  (3) splitting_ops THE FIX: TP+MTP records the spec all_gather into the SYCL graph, but oneCCL 2021.17's sched
#      allgather has no graph-recordable impl -> capture crash. Listing the 3 collectives in SPLITOPS makes inductor
#      partition at them (run EAGER) while decode stays captured. This is what overturned the old "TP=2 MTP DEAD".
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="${ROOT:-/mnt/vm_8tb/b70}"     # needed to reference the host GDN .so before sourcing lib.sh

export IMG="${IMG:-vllm-xpu-env:int8g-v0240}"   # v0.24.0/torch 2.12 (2026-07-03). Old v0.23 = vllm-xpu-env:int8g (rollback).
export CKPT="${CKPT:-/models/qwen3.6-27b/w8a8-sqgptq}"
export SERVED="${SERVED:-qwen36-27b-w8a8-sqgptq-mtp}"
export DTYPE="${DTYPE:-auto}"
export TP="${TP:-2}"
export GRAPH="${GRAPH:-1}"                  # PIECEWISE capture (2026-06-24: Bug B FIXED). The old garbage was caused
                                            # by EJECTING the TP collectives to eager (they are out-of-place; the
                                            # ejected output does not land at the captured-piece's capture-time
                                            # address -> stale read -> garbage). Fix = eject NOTHING (see SPLITOPS)
                                            # + a capture-safe all_gather (patches/, all-reduce-of-padded) so the
                                            # spec-verify all_gather records into the SYCL graph instead of crashing.
export CGMODE="${CGMODE:-PIECEWISE}"        # DEFAULT PIECEWISE (2026-07-03, v0.24.0): graph REPLAY on -> the fast
                                            # captured decode (usage-based 44 t/s WITH vision, 2.4x the sglang daily
                                            # driver's 18). On v0.23 PIECEWISE CRASHED under sustained MTP (graph-replay
                                            # accumulation) so NONE was the stable default; v0.24.0 FIXES that --
                                            # validated STABLE (restarts=0 under sustained concurrent MTP+capture) +
                                            # coherent (gate 40/40) + deterministic. Fall back to CGMODE=NONE (torch.compile,
                                            # no replay; ~half the decode) only if a replay regression ever recurs.
export IGP="${IGP:-false}"                  # legacy piecewise splitter: REQUIRED on this hybrid (the inductor
                                            # partitioner KeyErrors on the mixed W8A8(scale)+BF16-GDN(no-scale) region).
export NOMM="${NOMM-}"                       # VISION ON by default (2026-07-03: v0.24.0 vision profiling no longer
                                            # crashes -> the old NOMM=1 text-only workaround is retired). Pass NOMM=1
                                            # to force text-only (e.g. a pure-text perf bench). Empty = vision retained.
# Agentic parity (daily-driver): tool-calling + reasoning split, same parsers as the sglang shelf. lib.sh maps
# TOOLCALL -> --enable-auto-tool-choice --tool-call-parser; REASONPARSER -> --reasoning-parser. Validated on v0.24.0
# (qwen3_coder emits proper tool_calls; qwen3 splits <think> into reasoning_content).
export TOOLCALL="${TOOLCALL:-1}"; export TOOLPARSER="${TOOLPARSER:-qwen3_coder}"
export REASONPARSER="${REASONPARSER:-qwen3}"
# VISION + torch.compile (GRAPH=1) on v0.24.0 XPU: enabling vision crashed at init with 'NoneType' object
# has no attribute 'size' (torch/_dynamo call_size, in determine_available_memory _dummy_run). ROOT CAUSE =
# the STANDALONE AOT COMPILE (serialize+reload of the compiled graph) mishandles the optional/None multimodal
# inputs. THE FIX = VLLM_USE_AOT_COMPILE=0 (env): use the normal captured graph, not the serialized artifact.
# NO runtime perf cost (AOT is only a startup-serialization optimization; the captured graph is identical --
# validated 44 tok/s WITH vision+capture, gate 40/40 coherent, restarts=0). --skip-mm-profiling is also kept
# (skips the vision-encoder memory-profiling dummy run; harmless, was in the validated combo). Auto-applied
# when vision is on (NOMM empty) AND capture is on (GRAPH=1). ALSO clear a STALE text-only AOT compile cache
# when flipping text<->vision (torch_compile_cache is keyed WITHOUT mm-mode -> they collide -> same crash).
if [ -z "$NOMM" ] && [ "${GRAPH:-1}" = 1 ]; then
  export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--skip-mm-profiling"
  export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }VLLM_USE_AOT_COMPILE=0"
fi
# [2026-07-03 launch-path A/B, docs/20260703_faster_dd_plan.md Tier A + JOURNAL incl the CORRECTION entry]
# MRV2 (VLLM_USE_V2_MODEL_RUNNER=1, async runner: overlaps schedule N+1 with GPU step N) is a measured
# decode win on the launch/python-bound captured path: TG c1 30.24 -> 32.27/33.60 across two runs
# (+7-11%, run-to-run noise +-4%), c4 15.89, TTFT unchanged, gen-probe + gate 24/24 coherent,
# engagement log-verified. CAVEATS: (a) INCOMPATIBLE with prefix caching on this hybrid (mamba align
# mode): upstream asserts "Model Runner V2 has not yet supported mamba_cache_mode='align'" at init ->
# MRV2 defaults ON only when PREFIXCACHE=0 (cold-bench/single-shot); the PREFIXCACHE=1 daily driver
# keeps the prefix cache (6.97x warm TTFT >> +7% decode). (b) no thinking_token_budget support (unused
# by us). Force with B70_MRV2=0|1. (Gate applied after the PREFIXCACHE default below.)
# NOTE: SYCL_UR_USE_LEVEL_ZERO_V2=1 (L0 V2 adapter) was NOT actually A/B-able from here -- lib.sh's
# multicard MGPU block pins -e SYCL_UR_USE_LEVEL_ZERO_V2=0 AFTER DOCKER_ENV in the docker run (last -e
# wins), a deliberate vllm#41663 stability pin. Testing V2=1 requires a lib.sh edit; untested, no claim.

export UTIL="${UTIL:-0.90}"                 # 17 GiB/card -> plenty of KV headroom at TP=2
export MAXLEN="${MAXLEN:-4096}"
export MAXSEQS="${MAXSEQS:-8}"
export PREFIXCACHE="${PREFIXCACHE:-1}"      # DEFAULT ON (2026-07-03): --enable-prefix-caching, the big win for long
                                            # agentic/coding sessions (warm reprompt reuses the cached prefix instead
                                            # of re-prefilling). Overrides lib.sh's 0 default for THIS entry only.
                                            # It was PREVIOUSLY forced off because vLLM's mamba "align" mode (which
                                            # prefix-caching auto-selects on the hybrid GDN) crashed at init on XPU:
                                            # "Overflow when unpacking long long" -- it packs Level-Zero USM device
                                            # pointers (>=2**63) into SIGNED int64 tensors (mamba_utils.py
                                            # MambaSpecDecodeGPUContext state_base_addrs/block_table_ptrs). FIXED by
                                            # patches/sitecustomize.py block (4): re-exec initialize_from_forward_context
                                            # with the two data_ptr() assignments wrapped to a two's-complement signed
                                            # int64 (bit-identical to what CUDA stores; the triton kernel reinterprets
                                            # the bits back to a pointer). VALIDATED 2026-07-03 @ MAXLEN=253952 TP=2 MTP
                                            # spec=3 + PIECEWISE capture + vision: init clean, gate 24/24 coherent, warm
                                            # prefix reuse 6.97x TTFT (cold 3620ms -> warm 519ms, identical output), and
                                            # miss-path decode 32 t/s >= the PREFIXCACHE=0 baseline (27.1). PREFIXCACHE=0
                                            # to force the clean re-prefill baseline (e.g. a cold-perf bench).
# MRV2 gate (see the L0-V2/MRV2 block above): default ON only when prefix caching is OFF.
_MRV2_DEF=$([ "$PREFIXCACHE" = 1 ] && echo 0 || echo 1)
[ "${B70_MRV2:-$_MRV2_DEF}" = 1 ] && export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }VLLM_USE_V2_MODEL_RUNNER=1"

export MTPTOK="${MTPTOK:-3}"                # MTP spec tokens. spec=3 = WINNER on the fixed captured path (scripts/111
                                            # coherence-gated: spec3 34.82 t/s @51% > spec4 30.56 @37% > spec5 26.10
                                            # @26% -- MONOTONIC DECREASING; the 1-layer MTP head over-drafts past ~3.
                                            # (The old "spec=5 winner / climbing 50/57/63" was on degenerate garbage.)
export CAPSIZES="${CAPSIZES:-1,2,4,6,8}"    # covers the spec-verify batch 1+spec for spec 3/4/5 (=4/5/6)
export COMPILESZ="${COMPILESZ-}"           # MUST be empty for spec-decode (compile_sizes [1] is rejected; pads to 1+spec)
# THE FIX (3) [2026-06-24, REVISED]: eject NOTHING. splitting_ops = the model's attention + GDN custom ops ONLY
# (the genuine non-capturable ops). Do NOT add the TP collectives: ejecting them breaks the captured-piece input-
# address contract -> garbage (the old config ejected all 3 -> the ejected per-layer all_reduce corrupted decode).
# all_reduce + reduce_scatter record fine inside the graph. all_gather (which oneCCL CANNOT record) is handled by
# the capture-safe all-reduce-of-padded shim in patches/sitecustomize.py, so it too stays captured. Net: every
# collective is captured + correct -> coherent body AND a numerically-correct spec-verify (real accept), all fast.
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
export SPLITOPS="${SPLITOPS:-${_ATTN_OPS}}"

PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
# v0.24.0/torch-2.12 combined int8+GDN .so (Stage 2, ABI-verified: gdn_attention + int8_gemm_w8a8/w8a16 +
# dynamic_per_token_int8_quant). Old torch-2.11 .so at $ROOT/vllm-xpu-kernels/vllm_xpu_kernels/ (rollback w/ :int8g).
GDN_SO="${GDN_SO:-$ROOT/w8a8_kernel_v0240/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/w8a8_kernel_v0240/libgdn_attn_kernels_xe_2.so}"
MOUNTS=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro"
         -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro"
         -v "$SCRIPT_DIR/patches:/opt/mtp_shim:ro" )
DOCKER_ENV=( -e PYTHONPATH=/opt/mtp_shim )

# push-allreduce overlay -- DEFAULT ON (PUSH_AR=1, PUSH_AR_GRAPH=1): replaces oneCCL's TP all-reduce with the
# custom L0-IPC push transport (contrib/vllm_push_allreduce). With PUSH_AR_GRAPH=1 the DECODE all-reduce is also
# graph-captured (device-side L0-event sync recorded into torch's XPUGraph, P2P_GPU.md K.6/K.8), so EVERY
# all-reduce (prefill AND decode) uses the 11 GB/s posted-write fabric -- no oneCCL fallback. Push-ar's
# sitecustomize CHAINS the MTP shim (PUSH_AR_CHAIN_SITECUSTOMIZE) so both patches run. P2PACCESS stays 0
# (push-ar uses its own L0-IPC peer write, NOT oneCCL's wedge-prone CCL_TOPO_P2P_ACCESS). Measured vs oneCCL on
# this 27B-W8A8 TP=2 serve: 3.8x prefill TTFT, +80-126% agg throughput (J.21), +8-10% per-stream decode (K.8).
# OPT-OUT to the plain oneCCL baseline with PUSH_AR=0. Set PUSH_AR_GRAPH=0 for the prefill-only push (decode on
# oneCCL, J.17/J.21). The .so is prebuilt + committed; rebuild via scripts/118 (graph) or scripts/108 (eager).
if [ "${PUSH_AR:-1}" = 1 ]; then
  PUSH_AR_DIR="$SCRIPT_DIR/../../../vllm/contrib/vllm_push_allreduce"
  # [!] CRITICAL (2026-06-26): PUSH_AR_GRAPH=1 records the DECODE all-reduce INTO the XPU graph. CGMODE=NONE has
  # NO graph -> the captured-decode push reads a stale/uninit buffer -> INPUT-DEPENDENT GARBAGE ("!!!!" on some
  # prompts; coherent on others -> a single gen-probe misses it). So on NONE the decode all-reduce MUST stay on
  # the host-barrier/oneCCL path: default PUSH_AR_GRAPH=0 here. (verified: NONE+PUSH_AR_GRAPH=0 -> 0/8 garbage.)
  _PAG_DEF=$([ "${CGMODE:-PIECEWISE}" = NONE ] && echo 0 || echo 1)
  PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-$_PAG_DEF}"
  # PUSH_AR_GRAPH=1 -> CAPTURABLE decode path: graph .so + MIN_NUMEL=0 (push ALL all-reduces incl. decode; needs
  # a captured graph). PUSH_AR_GRAPH=0 -> prefill-only push (host-barrier .so, decode-sized -> oneCCL fallback).
  if [ "$PUSH_AR_GRAPH" = 1 ]; then
    _DEF_SO="$PUSH_AR_DIR/prebuilt/libxpu_push_ar_graph.so"; _DEF_MIN=0
  else
    _DEF_SO="$PUSH_AR_DIR/prebuilt/libxpu_push_ar_torch.so"; _DEF_MIN=65536
  fi
  SO_HOST="${SO_HOST:-$_DEF_SO}"  # self-contained; rebuild via scripts/108 REBUILD_SO=1 or scripts/118
  [ -f "$SO_HOST" ] || { echo "[!] PUSH_AR=1 but push-ar .so missing at $SO_HOST (set PUSH_AR=0 for oneCCL)" >&2; exit 1; }
  MOUNTS+=( -v "$PUSH_AR_DIR:/opt/push_ar:ro"
            -v "$SO_HOST:/opt/push_ar_so/libxpu_push_ar_torch.so:ro" )
  DOCKER_ENV=( -e PYTHONPATH=/opt/push_ar:/opt/mtp_shim
               -e PUSH_AR_CHAIN_SITECUSTOMIZE=/opt/mtp_shim/sitecustomize.py
               -e PUSH_AR_SO=/opt/push_ar_so/libxpu_push_ar_torch.so
               -e PUSH_AR_DISABLE=0
               -e PUSH_AR_GRAPH="$PUSH_AR_GRAPH"
               -e PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-$_DEF_MIN}" )
  export P2PACCESS="${P2PACCESS:-0}"
  echo "=== PUSH_AR overlay ON [default] (GRAPH=${PUSH_AR_GRAPH:-1} MIN_NUMEL=${PUSH_AR_MIN_NUMEL:-$_DEF_MIN}, P2PACCESS=0, .so=$SO_HOST) ==="
else
  echo "=== PUSH_AR overlay OFF (PUSH_AR=0 -> plain oneCCL TP all-reduce baseline) ==="
fi

# --- DIAGNOSTIC / BISECT TOGGLES (opt-in; default-off => the serve command is byte-identical) -------
# Added 2026-06-25 to root-cause the TP=2 long-load engine crash (JOURNAL/FINDINGS 2026-06-25).
#   B70_NOMTP=1  drop --speculative-config (run with MTP OFF) -- the bisect variable.
#   B70_DEBUG=1  faulthandler only: near-zero overhead, dumps a worker py+C traceback on a FATAL signal
#                (SIGSEGV/SIGABRT/SIGBUS/SIGFPE) -- the worker dies with no traceback today, this catches it.
#   B70_DEBUG=2  + Level-Zero validation layer + UR/oneCCL/vLLM debug logging (HEAVIER, perturbs timing;
#                use to characterize a crash, NOT for a clean timing bisect).
if [ "${B70_NOMTP:-0}" = 1 ]; then export MTPTOK=""; export SPEC=""; echo "=== B70_NOMTP=1 -> MTP OFF (no --speculative-config) ===" >&2; fi
if [ "${B70_DEBUG:-0}" != 0 ]; then
  DOCKER_ENV+=( -e PYTHONFAULTHANDLER=1 -e PYTHONUNBUFFERED=1 )
  # B70_DEBUG=2 adds the Level-Zero validation layer + basic LEAK CHECKER (prints per-handle create-vs-
  # destroy counts at exit -> confirms WHICH handle type accumulates) + UR/oneCCL/vLLM debug logging.
  [ "${B70_DEBUG}" = 2 ] && DOCKER_ENV+=( -e VLLM_LOGGING_LEVEL=DEBUG -e ZE_ENABLE_VALIDATION_LAYER=1 \
        -e ZE_ENABLE_PARAMETER_VALIDATION=1 -e ZEL_ENABLE_BASIC_LEAK_CHECKER=1 -e UR_L0_LEAKS_DEBUG=1 \
        -e UR_LOG_LEVEL=info -e CCL_LOG_LEVEL=info )
  echo "=== B70_DEBUG=${B70_DEBUG} -> diagnostic env injected (faulthandler$([ "${B70_DEBUG}" = 2 ] && echo ' + L0 validation/leak-checker + UR/CCL/vLLM debug')) ===" >&2
fi
# B70_EXTRA_ENV: space-separated NAME=VAL list injected as -e flags (test any env -- UR_L0_* event knobs,
# CCL_ENABLE_SYCL_KERNELS, etc. -- without further recipe edits). e.g. B70_EXTRA_ENV="UR_L0_REUSE_DISCARDED_EVENTS=1".
if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for kv in ${B70_EXTRA_ENV}; do DOCKER_ENV+=( -e "$kv" ); done
  echo "=== B70_EXTRA_ENV -> injected: ${B70_EXTRA_ENV} ===" >&2
fi

# B70_EXTRA_MOUNTS: space-separated host:container[:ro] bind specs appended to MOUNTS -- test file
# overlays without recipe edits (mirrors B70_EXTRA_ENV). e.g. the fusedq xpu_int8.py routing overlay
# (research/w8a8/FUSEDQ_NOTES.md section 2).
if [ -n "${B70_EXTRA_MOUNTS:-}" ]; then
  for m in ${B70_EXTRA_MOUNTS}; do MOUNTS+=( -v "$m" ); done
  echo "=== B70_EXTRA_MOUNTS -> injected: ${B70_EXTRA_MOUNTS} ===" >&2
fi

# --- DFLASH daily-driver toggle (2026-07-03 session 3; JOURNAL + vllm/DFLASH_XPU.md) -----------------
# DFLASH=1 swaps the NEXTN MTP head for vLLM's in-tree DFlash block-diffusion drafter (a SEPARATE draft
# model, z-lab Qwen3.6-27B-DFlash, W8A8-RTN). DFTOK=8 speculative tokens (sweep-gated peak). Two modes:
#   DFSWA=1 (DEFAULT, the daily driver): all 5 drafter layers windowed (sliding_window=2048, uniform ->
#     ONE kv cache group, assertion-safe) via the qwen3_dflash_swa.py model overlay. accept_len HOLDS at
#     deep context (3.9@4k -> 3.6@100k) where the stock drafter COLLAPSES, and it fits the FULL 253952
#     context (291k KV tokens, 1.15x conc). Reliable two-point decode t/s BEATS MTP at depth (40k: 21.3
#     vs 17.7) and is competitive shallow. This is why the DD runs SWA + full context.
#   DFSWA=0: the STOCK full-attention drafter. Faster at SHALLOW context (<8k, up to +75% on short
#     coding bursts) BUT its sliding layers run out-of-distribution at depth -> accept COLLAPSES
#     (4.5@4k -> 1.6@100k) and it caps ctx at ~186k. Use only for short-context-dominated workloads.
# Both modes need the gs=8 KV-group padding patch (byte-identical for MTP: argmin-padding leaves
# {16 full, 48 mamba} -> group_size 16 unchanged). Validated: gate 12/12, prefix cache warm-reuse intact.
if [ "${DFLASH:-0}" = 1 ]; then
  export DFTOK="${DFTOK:-8}"
  export DFDRAFT="${DFDRAFT:-/models/qwen3.6-27b/dflash-draft-w8a8-rtn}"
  export SPEC="${SPEC:-{\"method\":\"dflash\",\"model\":\"$DFDRAFT\",\"num_speculative_tokens\":$DFTOK}}"
  export MTPTOK=""                                  # DFlash replaces the NEXTN MTP head
  [ "$CAPSIZES" = "1,2,4,6,8" ] && export CAPSIZES="1,2,4,6,9"   # spec-verify batch = 1+DFTOK(8)
  [ "$SERVED" = "qwen36-27b-w8a8-sqgptq-mtp" ] && export SERVED="qwen36-27b-w8a8-sqgptq-dflash"
  MOUNTS+=( -v "$SCRIPT_DIR/../../../vllm/patches/kv_cache_utils_gcd.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py:ro" )
  if [ "${DFSWA:-1}" = 1 ]; then     # DEFAULT: all-sliding drafter -- holds accept at depth + full ctx
    DOCKER_ENV+=( -e B70_DFLASH_SWA=1 )
    MOUNTS+=( -v "$SCRIPT_DIR/../../../vllm/patches/qwen3_dflash_swa.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py:ro" )
    echo "=== DFLASH=1 DFSWA=1 -> all-sliding drafter spec=$DFTOK, served=$SERVED (deep-accept + full ctx) ===" >&2
  else
    echo "=== DFLASH=1 DFSWA=0 -> STOCK full drafter spec=$DFTOK, served=$SERVED (shallow-only; caps ~186k ctx) ===" >&2
  fi
fi

source "$SCRIPT_DIR/../../_common/lib.sh"
b70_dispatch "$@"
