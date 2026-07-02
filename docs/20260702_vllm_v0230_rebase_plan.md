# vLLM-XPU rebase plan: :v0230 / :int8g -> verified upstream vLLM v0.23.0

Status: PLAN (read-only; nothing built/modified/run). Drafted 2026-07-02. vLLM is the PAUSED baseline
(sglang is primary); this is a maintained-baseline exercise. Goal: pick up hybrid mixed prefill+decode
correctness fixes that map to our "!!!!" concurrent-batch garbage.

## Executive summary + two load-bearing corrections

Target vLLM v0.23.0 (torch-xpu 2.11) is sound; all five target fix PRs are confirmed present in the v0.23.0
tag by git ancestry. But:

1. The #431 oneDNN->FetchContent break does NOT hit a v0.23.0-matched build. v0.23.0 pins vllm-xpu-kernels
   v0.1.9; the two GDN NaN fixes we want (#411, #399) are in v0.1.10; #431 (FetchContent) only lands in
   v0.1.10.1. So building our custom .so from the v0.1.10 SOURCE TAG banks both NaN fixes AND stays pre-#431
   (submodule oneDNN) -- our kernels/int8_gemm_kernel.patch build plumbing is preserved. #431 only bites at
   the future v0.24.0 / torch-2.12 jump.
2. The current :v0230 image has unresolved provenance. int8g docs claim :v0230 = "vLLM 0.23.0+xpu / torch
   2.11", but the only from-source build script, scripts/28_build_vllm_xpu.sh, checks out c51df4300 =
   vLLM 0.20.2rc1.dev2 (torch 2.10). :v0230/:int8 are docker commit-built, not reproducible from a
   checked-in Dockerfile. So we cannot assume the running baseline contains the five fix PRs -- Stage 0
   resolves this first.

## Confirmed upstream facts (via gh, read-only)

Tag v0.23.0 (commit 0fc695fc6d1d, 2026-06-15). From requirements/xpu.txt + docker/Dockerfile.xpu @ v0.23.0:
- torch==2.11.0 (+xpu), torchaudio, torchvision
- triton-xpu==3.7.0 (installed post-requirements)
- vllm-xpu-kernels wheel v0.1.9 (cp38-abi3, manylinux_2_28), published 2026-05-29
- auto_round_lib >=0.13.0 ; Python 3.12 (wheel is abi3/py3.12)
- Base image intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04 (oneAPI 2025.3 incl dnnl-devel 2025.3,
  oneCCL 2021.15.9.14 BMG-enabled, Level-Zero 1.26.0, compute-runtime 25.48.36300.8, IGC 2.24.8)

Target fix PRs -- all confirmed IN v0.23.0 (behind_by:0 vs tag): #44700 (split mixed prefill+decode: route
decodes to recurrent GDN kernel), #43990 (zero freshly-allocated KV blocks for hybrid+fp8 KV), #42430 (mamba:
single-token extends as decodes), #43961 (corrupted MLA+linear attn fix), #43556 (mamba linear-attn refactor).

vllm-xpu-kernels PRs -> tags: #399 (GQA-ratio-3 conv1d z-drop) and #411 (SLM write-after-read NaN >=32K on
XE2) first in v0.1.10 (NOT v0.1.9); #431 (oneDNN FetchContent) first in v0.1.10.1. Timeline: v0.1.9 (05-29) ->
v0.1.9.1 (06-01) -> v0.1.10 (06-18) -> v0.1.10.1 (06-24).

## Our custom kernel/patch layer (what a rebase must re-apply/rebuild)

- kernels/int8_gemm_kernel.patch -- adds f16_int8/bf16_int8 joint_dtypes + onednn_types_mapper specializations
  + dispatch in csrc/xpu/onednn/onednn_ext.h; int8_gemm_w8a16() in onednn_matmul.cpp; op reg in
  torch_bindings.cpp; decl in ops.h. Action: re-apply against v0.1.10 csrc; the enum/mapper/dispatch block is
  the likeliest to have shifted.
- kernels/int8_gemm_w8a16.h, kernels/int8_gemm_w8a8.h -- decode (s8 x f16) + prefill (s8xs8s32 XMX) op bodies.
  Action: drop in; verify they compile vs v0.1.10 oneDNN wrappers (check dnnl_matmul_w8a16_int8 signature).
- vllm/contrib/vllm_int8_xpu/xpu_int8.py -- XPUInt8ScaledMMLinearKernel. Action: re-diff vs v0.23.0
  ScaledMMLinearKernel ABC (captured surface: vllm/contrib/vllm_int8_xpu/v0230/_refs.txt).
- Registry patch (scripts/45_patch_serve_int8.sh -> apply_patches.py) -- registers XPU in
  _POSSIBLE_INT8_KERNELS[XPU] + hardens chooser; patches model_executor/kernels/linear/__init__.py.
  Action: re-verify registry module path + dict at v0.23.0 (classic per-release break point).
- Fake/meta xpu_int8.py (:int8g step, vllm/images/int8g/build.sh) -- register_fake for cudagraph PIECEWISE.
  Action: verify direct_register_custom_op/register_fake at torch 2.11.
- GDN .so runtime-mount (rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh) mounts _xpu_C.abi3.so +
  libgdn_attn_kernels_xe_2.so (GDN_KERNELS_ENABLED=ON). This is where #399/#411 land -> rebuild from v0.1.10.

On #431: our patch edits vllm-xpu-kernels' OWN C++ wrappers, not third-party oneDNN, so #431 (how oneDNN dep
is fetched) does not conflict with patched file text; its risk is include/link (headers relocated, CMake target
renamed). Because v0.1.10 is pre-#431, building the custom .so from v0.1.10 sidesteps this for v0.23.0.

## Staged plan

Stage 0 -- Resolve current provenance (no build): docker image inspect vllm-xpu-env:v0230 digest vs the digest
in vllm/images/int8g/README.md (sha256:04e26c1c7f89...); inside, read vllm.__version__ + torch.__version__.
If genuinely v0.23.0/torch 2.11 -> rebase is torch-ABI-neutral (verification+provenance hardening). If actually
c51df4300/0.20.2rc1/torch 2.10 -> real torch 2.10->2.11 ABI bump (#37947), every custom .so must rebuild.
Record in JOURNAL. Keep current image as rollback.

Stage 1 -- Build base :v0230-successor from the pinned TAG (reproducible): copy scripts/28 to a NEW number
(append-only rule), git checkout v0.23.0, docker build -f docker/Dockerfile.xpu -t <dated tag>. Keep
torch 2.11.0 / triton-xpu 3.7.0 / xpu-kernels v0.1.9 as pinned. Decision: (A, recommended) keep base wheel
v0.1.9 and get #399/#411 only via the runtime-mounted custom GDN .so from v0.1.10 (Stage 2) -- cleanest
canonical v0.23.0, and the 27B path already mounts its GDN .so; or (B) bump base wheel to v0.1.10 (deviates
from pin, needs import+op smoke test). Tag immutable dated (vllm-xpu-env:v0230-YYYYMMDD), move convenience tag,
record digest.

Stage 2 -- Rebuild custom _xpu_C.abi3.so from v0.1.10 source (pre-#431, has #399+#411): checkout
vllm-xpu-kernels v0.1.10; apply kernels/int8_gemm_kernel.patch + drop the two .h (per research/w8a8/
W8A8_BUILD.md); resolve hunk drift; build inside the Stage-1 base (torch 2.11 ABI) with the minimal scope from
scripts/44 (XPU_SPECIFIC_KERNELS_ENABLED=ON) for dense/14B .so, then a second build GDN_KERNELS_ENABLED=ON for
the 27B mount .so (+ libgdn_attn_kernels_xe_2.so). Honor the stale-cache rm from W8A8_BUILD.md. Op-presence
check: hasattr(torch.ops._xpu_C, "int8_gemm_w8a8") and dynamic_per_token_int8_quant True.

Stage 3 -- Bake :int8-successor (scripts/47): FROM Stage-1 base, copy Stage-2 .so over
site-packages/vllm_xpu_kernels/_xpu_C.abi3.so, run apply_patches.py (registry, re-verified in Stage 0/1), copy
in xpu_int8.py (re-verified vs v0.23.0 ABC), docker commit -> :int8-<date>.

Stage 4 -- Bake :int8g-successor (vllm/images/int8g/build.sh): FROM :int8, swap register_fake xpu_int8.py into
EVERY resolvable scaled_mm/xpu_int8.py, commit -> :int8g-<date>, move convenience tag, record digest.

## Sweep-gating checklist (before it replaces the paused baseline)

1. Provenance/identity: /v1/models id + vllm.__version__==0.23.0, torch 2.11, hasattr _xpu_C.int8_gemm_w8a8,
   GDN .so mounted (27B). Served id encodes method+scheme (...-W8A8-gptq) per Model Identity rules.
2. Concurrent prefill+decode coherence (headline gate): mixed concurrent load, NO "!!!!"/garbage on any stream
   (what #44700/#43990/#42430/#43961/#43556 target). Pass/fail, non-negotiable.
3. W8A8 int8 correctness: HumanEval+ within tolerance of baseline; GDN long-context >=32K coherence (#411) +
   GQA-ratio-3 conv1d path (#399).
4. fp8-KV hybrid: if exercised, confirm freshly-allocated KV blocks zeroed (no NaN first-tokens) -- #43990.
5. TTFT/decode parity: best-config-vs-best-config >= baseline; capture PIECEWISE at its own settings.
6. Shelf smoke: if any bin/ or rdy_to_serve/_common/ glue changes, bin/serve-sweep --smoke green.
7. GPU discipline: all under gpu-run; single-card xpu-health before TP=2; never chain TP=2 worker-init crashes.

Only when 1-6 green does :int8g-<date> become the shelf image (update digest in shelf serve.sh/README.md);
keep old digest as rollback.

## Risks
- Provenance ambiguity (highest): if current :v0230 != real v0.23.0, budget a full from-source build + torch
  2.10->2.11 ABI bump (all custom .so rebuild).
- xpu-kernels skew: base v0.1.9 lacks #399/#411; option B (v0.1.10 base) risks vLLM<->kernels API mismatch ->
  import+op smoke test, fall back to option A.
- Patch drift: int8_gemm_kernel.patch may not apply cleanly to v0.1.10 csrc; hand-resolve (small, additive).
- #431 FetchContent (deferred): do NOT build custom .so from v0.1.10.1+/main for the v0.23.0 image.
- Registry-hook refactor: _POSSIBLE_INT8_KERNELS/ScaledMMLinearKernel is per-release churn; re-diff vs _refs.txt.
- XPU graph-replay decay + TP=2 wedge (carried): keep cudagraph_mode=NONE for 27B TP=2; push-AR overlay unchanged.
- Rollback: immutable dated tags -> re-point shelf serve.sh at prior digest + docker restart. Keep current
  :v0230/:int8/:int8g digests until the successor passes gates 1-6 in a production soak.

Key repo files: scripts/28_build_vllm_xpu.sh (checks out the wrong commit today), scripts/44_build_int8_kernel.sh
+ research/w8a8/W8A8_BUILD.md, scripts/45_patch_serve_int8.sh, scripts/47_build_int8_image.sh,
vllm/images/int8g/build.sh + README.md, kernels/int8_gemm_kernel.patch + kernels/int8_gemm_w8a16.h +
kernels/int8_gemm_w8a8.h, vllm/contrib/vllm_int8_xpu/xpu_int8.py + v0230/_refs.txt,
rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh.
