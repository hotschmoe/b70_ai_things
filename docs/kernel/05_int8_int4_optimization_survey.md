# 05 - int8/int4 GPU matmul optimization survey (decode m=1 focus)

> **[!] DEPRIORITIZED (2026-06-22) -- the m=1 GEMV lever this survey chases is largely FUTILE on the B70.** The real
> decode win turned out to be PIECEWISE graph capture (~doubles int4 decode; FINDINGS/kernel/04), NOT a faster GEMV:
> oneDNN's int4/int8 GEMV already meets/beats llama.cpp (kernel/04 step 4 + microbench kernel/19), so there's little
> headroom for a custom kernel. Kept as a reference survey; do not treat the GEMV-tuning levers as open work.

Current (2025-2026) literature/web sweep on int8/int4 matmul optimizations relevant to our **decode
(m=1)** bottleneck on the B70 (Xe2/Battlemage). Companion to `04_decode_optimization.md` (the execution
ladder + levers A/B/C). This doc only adds VERIFIED deltas + status updates; it does not restate 04.

**Every URL below was actually fetched (WebFetch/WebSearch/`gh`), not invented.** Dates are as returned
by the source. Findings are split VERIFIED vs SPECULATIVE. Each carries a one-line ACTION.

Snapshot date: 2026-06-20.

---

## 0. TL;DR -- top actionable findings

1. **vLLM XPU graph support has LANDED upstream** (issue #26970 closed COMPLETED 2026-06-05; PRs
   #34482/#38193/#41344/#43043 merged; torch RFC pytorch#162143 closed). The torch-side `XPUGraph` API
   now exists. -> ACT: our PIECEWISE path is on the supported track; chase FULL via TRITON_ATTN.
2. **PR #43092 (the AOT-compile Dynamo crash fix) is still OPEN.** It fixes the exact
   `AssertionError: Handler already registered for ... current_stream` we'd hit enabling graph+AOT.
   -> ACT: cherry-pick the one-file patch if `:int8g` enable fails at startup.
3. **oneAPI 2026.0 added `sycl_ext_oneapi_work_group_scratch_memory` inside graph nodes** -- the exact
   restriction blocking FULL capture w/ flash-attn -- "requested by the PyTorch community." -> ACT:
   upgrading the toolchain in our image unblocks FULL capture with flash-attn, no vLLM change.
4. **IPEX `QMatmul.h` confirms BOTH our lever-B deltas**: conditional src zero-points
   (`m1_need_zp = q_zero_point() != 0`) AND `format_tag::any` weights with a cached reorder.
   -> ACT: this validates lever B1 (drop symmetric zp) and B4 (format_tag::any reorder) -- do them.
5. **oneDNN v3.8 (rel May 2025) "improved int8 matmul perf with zero-points for src+weight tensors" +
   improved Battlemage convolution.** v3.9-rc exists but its GPU matmul note is Lunar-Lake-only.
   -> ACT: confirm the image bundles >= v3.8 (`ONEDNN_VERBOSE=1`); the zp-perf item is real, do B1.

---

## 1. oneDNN int8/int4 matmul on Intel GPU (versions + zero-point cost)

### 1.1 v3.6 first enabled the int8-act + int4/int8-weight GPU matmul [VERIFIED]
- CLAIM: v3.6 (rel **2024-10-15**) "Enabled support for `int8` activations with grouped scales and
  `int8` or `int4` compressed weights in matmul primitive. This functionality is implemented on Intel
  GPUs." This is the primitive our `int4_gemm_w4a8` rides.
- URL: https://github.com/uxlfoundation/oneDNN/releases/tag/v3.6 (fetched 2026-06-20)
- ACTION: baseline fact -- our W4A8 path requires >= v3.6. No action beyond version-confirm.

### 1.2 v3.8 improved int8 matmul perf WITH zero-points + Battlemage conv [VERIFIED]
- CLAIM (exact bullets, v3.8 rel **2025-05-10**):
  - "Improved `int8` matmul performance with zero-points support for source and weight tensors."
  - "Improved convolution performance on: Intel Arc B-series discrete graphics (formerly Battlemage)."
  - "Enabled `int8`/`int4` compressed weights support in matmul primitive."
  - Graph API: "Scaled Dot Product Attention (SDPA) with `int4` and `int8` compressed key and value."
- URL: https://github.com/uxlfoundation/oneDNN/releases/tag/v3.8 (fetched 2026-06-20)
- INTERPRETATION: oneDNN explicitly TRACKS zero-points as a perf item it has to optimize around. That
  confirms our standing claim in 04 (B1) that src+weight zp carries real cost on Intel GPU at m=1.
- ACTION: (a) confirm the image's bundled oneDNN >= v3.8 via `ONEDNN_VERBOSE=1` (lever B5).
  (b) The fact that v3.8 had to "improve" int8-matmul WITH zp -- and our checkpoints are SYMMETRIC
  (zp=0) -- means dropping zp entirely (lever B1) should beat even the v3.8-improved-zp path.

### 1.3 v3.9-rc exists but no Battlemage int8-matmul note [VERIFIED]
- CLAIM: v3.9-rc release page lists only "Improved matmul performance for Intel Arc Graphics for Intel
  Core Ultra processors (Series 2) (formerly Lunar Lake)" and an int8-conv-with-plain-weights item.
  NO new Battlemage / int4-decode / grouped-micro-gemm matmul bullet on the v3.9-rc page.
- URL: https://github.com/uxlfoundation/oneDNN/releases/tag/v3.9-rc (fetched 2026-06-20)
- ACTION: no rush to v3.9 for our shapes -- the relevant int8-matmul-zp win is already in v3.8. Target
  v3.8.x in the image; revisit v3.9 final notes later.

### 1.4 issue #3323 -- B60/Battlemage u4 G=64 GEMM memory growth [VERIFIED, status corrected]
- CLAIM: "Get unexpected GPU memory usage on B60 platform." Repro: u4 weights, **group size 64**,
  100-iter GEMM loop; "GPU memory usage is increased a little bit and never get released," not seen on
  A770. Opened **2025-05-23**, repro on **oneDNN v3.6.1**. **Status: CLOSED.**
- URL: https://github.com/uxlfoundation/oneDNN/issues/3323 (fetched 2026-06-20)
- CORRECTION to 04: 04 framed #3323 as a "u4 G=64 decompression memory" perf concern. It is actually a
  **memory-LEAK / non-release** bug on B60, and it is now CLOSED (likely fixed in a >= v3.7/3.8 oneDNN).
- ACTION: low priority. If we run long-lived W4A8 serves and see VRAM creep, confirm our oneDNN post-
  dates the fix. Otherwise no action.

### 1.5 `grouped_micro_gemm` vs `ref` [VERIFIED as a diagnostic, value unverified]
- Our 04 lever B2 stands: `ONEDNN_VERBOSE=2` on the m=1 call tells you whether W4A8 lands on the
  optimized `grouped_micro_gemm` microkernel or the slow `ref` fallback. The web sources do NOT publish
  a per-shape "which impl fires" table -- this is empirical, must be read off our own verbose log.
- ACTION: unchanged from 04 -- run `ONEDNN_VERBOSE=2` FIRST (free). No new web finding changes this.

---

## 2. IPEX (intel-extension-for-pytorch, branch xpu-main) -- what they do that we don't

Source files fetched from `raw.githubusercontent.com/.../xpu-main/...` on 2026-06-20.

### 2.1 `QMatmul.h` (int8 path): conditional src-zp + format_tag::any weights [VERIFIED]
- CLAIM: IPEX sets src zero-points ONLY when nonzero, and uses oneDNN-blocked weight layout for 2D:
  ```cpp
  bool m1_need_zp = (m1.q_zero_point() != 0);
  if (m1_need_zp) pattr.set_zero_points_mask(DNNL_ARG_SRC, mask_ac);
  // ... and only then insert DNNL_ARG_ATTR_ZERO_POINTS|DNNL_ARG_SRC into args
  if (is_onednn_layout_suggested && dims == 2) {
    m1_md = memory::desc(m1_dims, m1_dt, memory::format_tag::any);
    m2_md = memory::desc(m2_dims, m2_dt, memory::format_tag::any);   // weight -> oneDNN picks blocked
    dst_md = memory::desc(dst_dims, dst_dt, memory::format_tag::any);
  }   // weights reordered once, cached in inference mode
  ```
- URL: https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/csrc/gpu/oneDNN/QMatmul.h
- ACTION: **directly validates lever B1 + B4.** Mirror exactly: gate the src-zp on `zp != 0` (our
  symmetric checkpoints -> zp dropped), and feed `format_tag::any` + offline reorder for the weight.
  These are the two highest-confidence kernel edits because a shipping Intel library does precisely this.

### 2.2 `DnnlMatmulQuant.h` (W4A8 path): act_quant_mode, conditional zp, stride-based int4 [VERIFIED]
- CLAIM: the W4A8 entry takes an `act_quant_mode` selecting PER_TENSOR / PER_M (per-token) x SYM/asym.
  Zero-points set conditionally per mode (e.g. PER_M sets a row+col mask only when that mode is active),
  not unconditionally. The int4 weight uses STRIDE-based descriptors here (NOT format_tag::any -- that
  optimization is in the int8 `QMatmul.h` path), and dequant is internal to oneDNN
  (`matmul_primitive_create_and_cache`), no explicit XeTLA call in this header.
- URL: https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/csrc/gpu/oneDNN/DnnlMatmulQuant.h
- INTERPRETATION: even IPEX's own W4A8 oneDNN path does NOT apply format_tag::any to the int4 weight --
  so B4 for int4 is greenfield, higher risk than B4-for-int8. The clean, proven win is the SYM-zp drop
  (an `act_quant_mode` PER_M_SYM selects no src-zp), which is exactly lever B1.
- ACTION: implement an explicit symmetric mode in `int4_gemm_w4a8.h` (PER_M_SYM analogue) that emits NO
  src/weight zp. Treat int4 format_tag::any as a separate, later experiment (B4), not bundled with B1.

### 2.3 `act_dynamic_quant.cpp`: the per-token int8 quant is STANDALONE in IPEX too [VERIFIED]
- CLAIM: IPEX's `dynamic_per_token_quant(input, use_sym_quant)` / `dynamic_per_tensor_quant(...)` are
  single-kernel absmax+scale+quantize ops returning (q, scale, zp). They are **standalone** -- NOT
  fused into a preceding RMSNorm/SiLU. Symmetric path -> int8; else uint8.
- URL: https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/csrc/gpu/aten/operators/act_dynamic_quant.cpp
- CORRECTION/NUANCE to 04 (B3): 04 implies IPEX may have a fused norm+quant we're missing. It does NOT
  -- IPEX's act-quant is standalone, same shape as our `dynamic_per_token_int8_quant.cpp`. So fusing
  quant into the norm epilogue (B3) is a genuine NET-NEW optimization beyond IPEX, not a port. Higher
  value (no one upstream has it) but also no reference impl to copy -- write it ourselves.
- ACTION: keep B3 (fused rmsnorm+quant / silu+quant) as a real lever, but reclassify it from "port IPEX"
  to "net-new kernel." Sequence it AFTER the free wins (graph capture, B1).

> NOTE: `XEGEMM_INT4.cpp` and `int4_dequantize_xe.hpp` (the XeTLA int4 path) were NOT individually
> fetched/verified this round -- their existence is asserted by 04 but the specific optimizations are
> UNVERIFIED here. The oneDNN W4A8 path (DnnlMatmulQuant.h) is what our kernel actually mirrors.

---

## 3. vLLM XPU graph capture -- PR/issue status (mid-2026)

All via `gh ... --repo vllm-project/vllm` on 2026-06-20. **This is the biggest status correction set.**

| PR / issue | title | state | date |
|---|---|---|---|
| #34482 | [XPU] Support CUDAGraph on XPU Platform | **MERGED** | 2026-02-25 |
| #38193 | [XPU] Disable xpu graph by default | **MERGED** | 2026-03-26 |
| #41344 | [XPU] Disable CUDA graph memory estimate on XPU | **MERGED** | 2026-05-06 |
| #43043 | [XPU] update xpu graph usage | **MERGED** | 2026-05-19 |
| #43092 | [XPU] Fix CUDA API shims breaking Torch Dynamo during AOT compile | **OPEN** | upd 2026-05-19 |
| #26970 | [Feature][XPU] XPU graph support (roadmap) | **CLOSED / COMPLETED** | 2026-06-05 |
| pytorch#162143 | [RFC] XPU Graph (torch-side dep) | **CLOSED** | 2026-04-29 |

- VERIFIED: the XPU-graph roadmap issue #26970 is CLOSED as **COMPLETED** -- XPU graph capture is now a
  landed, supported feature line in vLLM (not "young/experimental defaulting"). The torch dependency
  (pytorch#162143 [RFC] XPU Graph) is also CLOSED. -> This CONFIRMS, from upstream, our FINDINGS that
  PIECEWISE works and the "no XPU graph capture" premise is dead.
- VERIFIED: **#43092 is still OPEN.** Its body describes exactly the crash we'd hit: on XPU,
  `_torch_cuda_wrapper()` aliases `torch.cuda.current_stream = torch.xpu.current_stream` (same object),
  and during `profile_run()` AOT compile Dynamo throws `AssertionError: Handler already registered for
  <function current_stream ...>`, killing EngineCore at KV-cache init. The fix wraps each shim in its
  own `functools.partial`. Triggered when `VLLM_USE_AOT_COMPILE=1` (default on torch >= 2.10).
- URLs:
  - https://github.com/vllm-project/vllm/pull/34482 (merged)
  - https://github.com/vllm-project/vllm/pull/38193 (merged)
  - https://github.com/vllm-project/vllm/pull/41344 (merged)
  - https://github.com/vllm-project/vllm/pull/43043 (merged)
  - https://github.com/vllm-project/vllm/pull/43092 (OPEN -- the AOT/Dynamo fix)
  - https://github.com/vllm-project/vllm/issues/26970 (closed COMPLETED)
  - https://github.com/pytorch/pytorch/issues/162143 (closed)
- ACTION (high): if enabling graph + AOT on `:int8g` fails at startup with "Handler already registered,"
  cherry-pick #43092's one-file change to `vllm/v1/worker/xpu_model_runner.py` (functools.partial the
  cuda->xpu shims). Cheap, known-good.
- ACTION (matrix unchanged): 04's support matrix (FLASH_ATTN -> PIECEWISE only; TRITON_ATTN -> ALL incl
  FULL) was set by #34482 and is NOT contradicted by anything fetched. Lever A2 (FULL via TRITON_ATTN)
  stands as the top untested experiment.

### 3.1 oneAPI 2026.0 SYCL-Graph `work_group_scratch_memory` -- unblocks FULL w/ flash-attn [VERIFIED]
- CLAIM: oneAPI DPC++ 2026.0 "added work-group scratch memory size control within graph nodes via
  `sycl_ext_oneapi_work_group_scratch_memory` ... requested by the PyTorch community." Plus handler-less
  graph submission, graph-owned allocations, graph recording for handler-less kernel submission.
- URL: https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html
  (via WebSearch result summary; the .intel.com HTML 403s on direct fetch -- treat the quote as
  search-surfaced, re-confirm the exact bullet before quoting verbatim in a contribution)
- ACTION: this is the exact restriction 04 flags as blocking FULL capture with flash-attn. Upgrading the
  image's oneAPI to 2026.0 should unblock FULL capture WITHOUT switching to TRITON_ATTN (lever A3).
  Cheaper than A2 if the toolchain bump is clean. Sequence: try A2 (TRITON_ATTN, no rebuild) first; A3
  (toolchain) if we want FULL on flash-attn specifically.

---

## 4. Triton-XPU (intel/intel-xpu-backend-for-triton) -- Battlemage int8 status

- VERIFIED: **v3.7.1 added "Intel G31 GPU support" (PR #6133)** -- G31 = Big Battlemage = our B70 die.
  v3.7.1 also "Enable block-scale DPAS (`bdpas`) for `tl.dot_scaled`" (PR #5909). (Release-date strings
  came back garbled in the fetch; cross-checked against 04/doc-06 which dates v3.7.1 to June 2026.)
- URL: https://github.com/intel/intel-xpu-backend-for-triton/releases (fetched 2026-06-20)
- VERIFIED (from doc 06_xpu_kernel_fastpaths, prior round): the DPAS lowering enumerates int8 engine
  types `S32_S32_S8_S8` / `U32_U32_U8_U8` on Xe2 -> an int8 `tl.dot` with int32 accum CAN map to
  Battlemage XMX in principle.
- **UNVERIFIED this round: the "FIXME: INT8 hangs" note -- could not confirm whether it is fixed.** No
  fetchable issue/PR surfaced confirming an int8-tl.dot-hang fix on BMG. The upstream Triton README I
  fetched is the generic OpenAI Triton README (lists only NVIDIA/AMD), not Intel's -- not evidence either
  way. Treat int8 `tl.dot` on BMG as PLAUSIBLE-BUT-UNPROVEN.
- ACTION (speculative): a Triton fused quant-prologue + int8 matmul on BMG is attractive (avoids
  hand-SYCL for lever C) BUT must be gated by a tiny standalone `tl.dot` int8 smoke test on the B70
  first -- do NOT build on it until the hang question is settled empirically. Lower priority than the
  oneDNN levers (B1/B4) which use a library that already ships an optimized path.

---

## 5. Community forks / others' Battlemage int8/int4 work worth porting

### 5.1 sgl-kernel-xpu (SGLang Intel-XPU backend) [VERIFIED scope, status partial]
- CLAIM: the SGLang XPU roadmap (issue #8309) lists "INT4 AWQ/GPTQ weight-only quant," "FP8 GEMM weight-
  only (E4M3/E5M2)," and "decode and extend attention for RadixAttention" for Battlemage/Battlematrix.
  Completion status not explicitly marked (issue created Jul 2025, marked inactive).
- URL: https://github.com/sgl-project/sglang/issues/8309 (via WebSearch summary; direct fetch 403'd)
- ACTION: monitor sgl-kernel-xpu for a published int4 W4A16/W4A8 GEMV or decode-attention kernel we
  could port. Their int4 is weight-only (W4A16), not our W4A8 -- different act path -- so not a drop-in.
  Low priority; revisit if they publish a decode-optimized int4 GEMV.

### 5.2 llm-scaler (Intel's B70-validated vLLM distro) -- oneDNN INT4 +25% [VERIFIED claim, single-source]
- CLAIM: llm-scaler-vllm 0.14.0-b8 "Thanks to Intel oneDNN optimizations, INT4 performance saw up to a
  **25% throughput improvement** vs the prior release"; "1.49x performance with BMG-G31"; decode-phase
  attention optimizations ">10% end-to-end on 10+ models"; MLA decode via FA2 varlen paged decode
  (head_size 576) on Xe2/Battlemage. Latest image `0.14.0-b8.3.1` (2026.06) adds FP8 KV cache.
- URLs:
  - https://www.phoronix.com/news/Intel-llm-scaler-vllm-0.14-b8 (Phoronix, ~Mar 2026)
  - https://github.com/intel/llm-scaler/blob/main/README.md (latest image 0.14.0-b8.3.1, 2026.06)
- CAVEAT: the "INT4 +25%" is attributed to a oneDNN bump, NOT a llm-scaler-specific kernel. It is the
  SAME oneDNN improvement (v3.8 int8/int4 matmul) we get by bumping oneDNN in OUR image -- i.e. it
  CORROBORATES lever B5 (newer oneDNN = free int4 perf) from an independent source.
- ACTION: this independently supports B5. Confirm our `vllm-xpu-env:int8` image's oneDNN version; if it
  predates the v3.8 matmul improvements, bumping oneDNN is a near-free int4-decode win (their 25%).
  NOTE: llm-scaler wraps an OLDER upstream vLLM for dense models (per doc 06) -- port the oneDNN bump,
  not the whole image.

### 5.3 ggml-sycl int4 GEMV / llama.cpp [no NEW finding this round]
- 04 lever C1/C2 already covers llama.cpp `ggml-sycl/mmvq.cpp` as the BW-ceiling reference + the only
  real int4 GEMV for Xe2 (WARP_SIZE=16). No new web evidence changes that; the relevant prior refs
  (PR #21517/#21527 MMVQ reorder for B70; #21893 BMG SYCL corruption) are already in doc 06.
- ACTION: unchanged -- C1 (bench mmvq.cpp on B70) remains the BW reference experiment.

---

## Contradicts our current docs:

1. **docs/literature/06_vllm_latest_xpu.md is STALE on XPU graph capture (whole section 2).** It states
   "XPU has *no* CUDA-graph-equivalent capture," "`cudagraph_mode=PIECEWISE` ... silently **no-ops**,"
   and "keep `--enforce-eager`." This is now FALSE: XPU graph support landed (issue #26970 CLOSED
   COMPLETED 2026-06-05; #34482/#43043 merged; torch RFC pytorch#162143 closed), and our own FINDINGS
   show PIECEWISE = +16.7% decode. Doc 06's section 2 and its env-var table row calling
   `VLLM_XPU_ENABLE_XPU_GRAPH` "unverified, leave default" are both outdated. -> FIX doc 06 section 2 +
   the env-var table; point to docs/kernel/04 + FINDINGS as the authority.

2. **docs/kernel/04 framing of oneDNN issue #3323 is slightly wrong.** 04 calls it "u4 G=64
   decompression memory" (implying a perf/decompression cost). It is actually a **memory-non-release
   (leak) bug on B60**, opened on v3.6.1, and is **now CLOSED**. -> downgrade it in 04 from a tracked
   perf item to a closed leak bug (only matters for long-lived serves on old oneDNN).

3. **docs/kernel/04 lever B3 implies a fused norm+quant exists in IPEX to port -- it does NOT.** IPEX's
   `act_dynamic_quant.cpp` is STANDALONE per-token int8 quant, same shape as our
   `dynamic_per_token_int8_quant.cpp`. Fusing quant into the RMSNorm/SiLU epilogue is NET-NEW (nobody
   upstream ships it), not a port. -> reclassify B3 in 04 as "net-new kernel, no reference impl,"
   which RAISES its value (unique) but also its effort (write from scratch).

4. **docs/kernel/04 lever B4 (format_tag::any) is verified for int8 but NOT for int4.** IPEX applies
   `format_tag::any` + cached reorder only in its INT8 `QMatmul.h`; its W4A8 `DnnlMatmulQuant.h` uses
   stride-based int4 descriptors. -> in 04, split B4: "B4a int8 weight reorder (proven by IPEX)" vs
   "B4b int4 weight reorder (greenfield, higher risk)." Do not assume the int4 reorder is a copy.

(No contradiction found for 04's lever A support matrix, the 52-64%-of-peak microbench, or the
PIECEWISE +16.7% result -- those remain consistent with everything fetched.)

---

## Verified-vs-speculative ledger

- VERIFIED (fetched source): oneDNN v3.6/v3.8/v3.9-rc release bullets; issue #3323 closed/leak; IPEX
  QMatmul conditional-zp + format_tag::any; IPEX DnnlMatmulQuant act_quant_mode + stride int4; IPEX
  act_dynamic_quant standalone; vLLM PR #34482/#38193/#41344/#43043 merged + #43092 OPEN + #26970
  closed-COMPLETED + pytorch#162143 closed; Triton-XPU v3.7.1 G31 support + bdpas.
- SEARCH-SURFACED (summary only, .intel.com 403'd direct -- re-confirm exact wording before quoting):
  oneAPI 2026.0 `work_group_scratch_memory` in graph nodes; llm-scaler INT4 +25% / 1.49x / decode +10%;
  sglang #8309 roadmap items.
- UNVERIFIED / still open: whether Triton-XPU int8 `tl.dot` "INT8 hangs" is fixed on BMG (no fix found);
  whether OUR image's bundled oneDNN is >= v3.8 (must read `ONEDNN_VERBOSE=1` on the box -- GPU-gated).
