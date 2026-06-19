# 04 - Decode (m=1) optimization research: graph capture, oneDNN, fused quant, SYCL GEMV

Synthesis of a 4-agent literature+code sweep (2026-06-20) on speeding up int8-activation DECODE
on the B70. **Every lever here lifts BOTH w8a8 and w4a8** (shared per-token-int8-act + GEMM path).
All code pointers were fetched/verified by the agents (paths confirmed, not invented).

## The problem, decomposed
- Microbench (`w4a8/20_microbench_w4a8_decode.sh`): `int4_gemm_w4a8` at m=1 already hits **52-64% of
  peak BW** in isolation -> the GEMM is not the disaster.
- Full-model w4a8 decode = 16.5 t/s (~25% effective) -> **~half the time is NON-GEMM**: eager-mode
  per-op dispatch (hundreds of ops/token) + the separate per-token act-quant. w8a8 has the same
  structure (decode 22.6-23.8 vs fp8 ~29).
- So two independent levers: **(A) kill dispatch overhead = graph capture**; **(B) raise GEMM BW +
  cut the act-quant/zp overhead = kernel/oneDNN work**.

---

## LEVER A — Graph capture (the dominant decode lever; partially DONE)

**The "--enforce-eager mandatory / no XPU graph capture" premise is STALE.** Confirmed against our
own `FINDINGS.md:19-23`: **PIECEWISE XPU graph capture already gives +16.7% w8a8 decode**
(23.33 -> 27.23 t/s), via image `vllm-xpu-env:int8g` + `register_fake` meta-kernels for our custom
int8 ops + `cudagraph_mode=PIECEWISE`. (Our `docs/literature/06_vllm_latest_xpu.md` still says
PIECEWISE is "inert on B70" -- that doc is outdated; fix it.)

Verified facts (vLLM source + PRs):
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` enables it; gated on torch >= 2.11.0.dev (`supports_xpu_graph()`).
- PR #34482 (merged): support matrix = **distributed NO; FLASH_ATTN -> PIECEWISE only; TRITON_ATTN
  -> ALL modes (incl FULL)**. So the eager-only guidance is conservative defaulting, NOT a single-card
  defect. The TP2 disablement is a SEPARATE issue.
- PR #38193 (merged): off-by-default because FULL needs "a specific driver version, not stable yet".
- FULL capture w/ flash-attn blocked by SYCL-Graph `work_group_scratch_memory` restriction ->
  **oneAPI DPC++ 2026.0 release notes lift exactly this** (toolchain upgrade unblocks it).
- PR #43092 (open): fixes `torch.cuda.*`->`torch.xpu.*` shim breaking Dynamo during AOT compile
  (`AssertionError: Handler already registered`) -- cherry-pick if enable fails at startup.

Actions (ranked):
- A1. **Extend PIECEWISE to w4a8** [hours]. The +16.7% was on w8a8; w4a8 uses `int4_gemm_w4a8` (a
  different op) -> add its `register_fake` meta-kernel (mirror `contrib/vllm_int8_xpu/xpu_int8.py`),
  serve on `:int8g` `cudagraph_mode=PIECEWISE`, measure. Expect a similar lift, free.
- A2. **FULL capture via `--attention-backend TRITON_ATTN`** [~hours]. PR #34482 says TRITON_ATTN
  supports FULL. Our repo already runs TRITON_ATTN OOB on Xe. FULL capture also flips ngram/MTP
  spec-decode positive (PIECEWISE leaves attention eager -> spec-decode stayed negative). **Highest-
  leverage untested experiment.** Bench TRITON_ATTN base-attn perf vs flash-attn first.
- A3. **oneAPI 2026.0 toolchain** [medium] -> FULL capture with flash-attn, no vLLM change.
- A4. Package the `register_fake`/meta-kernel registrations as the upstream contribution (Python-only).

Code: `vllm/platforms/xpu.py::check_and_update_config()`, `vllm/envs.py:277,1903`,
`torch/xpu/graphs.py` (XPUGraph, new in torch 2.11). PRs: vllm #34482, #38193, #43092, #43043, #41344;
issue #26970. SYCL Graph spec: `intel/llvm .../sycl_ext_oneapi_graph.asciidoc`.

---

## LEVER B — Kernel / oneDNN (raise GEMM BW + cut overhead)

### B1. Drop the symmetric zero-point attrs in `int4_gemm_w4a8.h` [~1 day, HIGH value, LOW risk]
`int4_gemm_w4a8.h` sets src `s32` zp AND weight `u4` zp **even though our checkpoints are symmetric**
(zp data = 0). oneDNN then carries a zero-point correction (per-group reduction sums via
`DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS`) on every decode = pure overhead at m=1. **IPEX's int8 path
(`QMatmul.h`) omits src-zp for symmetric; ours doesn't.** oneDNN v3.8 notes call zp a tracked perf
cost. Remove the zp attrs for the symmetric case (like our clean `int8_gemm_w8a8.h`); validate it
doesn't bounce to `ref` and that `test_int4_gemm_onednn.py -k w4a8` (SYM) still passes.

### B2. `ONEDNN_VERBOSE=2` diagnostic [free, do FIRST]
Run it on the m=1 call. Confirms whether w4a8 lands on the optimized `grouped_micro_gemm` microkernel
vs the slow `ref` fallback, shows the zp-correction term, and the oneDNN version. Answers our standing
"is the kernel actually optimized?" risk in one run. (w8a8 lands on the jit GEMM `gemm.cpp`.)

### B3. Fuse the per-token act-quant into the preceding RMSNorm / SiLU [days, helps w8a8+w4a8]
oneDNN **cannot** fuse f16->int8 act-quant into the GEMM prologue (src must be pre-quantized). The
universal fix is to fuse the quant into the op that PRODUCES the activation:
- `silu_and_mul_quant` wrapper EXISTS (`tests/register_ops.py`) -> use it for MLP down_proj input.
- A fused **rmsnorm+quant** for the qkv/gate-up input: `dynamic_per_token_int8_quant.cpp` is the only
  quant kernel in our `csrc/xpu/sycl/` -- **VERIFY** whether an XPU `layernorm_quant`/`rmsnorm_quant`
  exists (the agent cited a CUDA path); if not, write one (fuse the absmax+quant into the norm
  epilogue). Then wire the model's int8/int4 linear inputs to consume the fused output.
- Note: per-token (per-M) dst dequant must use `DNNL_ARG_DST` runtime scales (per-row mask), NOT a
  binary post-op (those broadcast per-N).

### B4. Weights as `format_tag::any` + offline reorder [2-4 days]
`int8_gemm_w8a8.h` feeds explicit strides; IPEX feeds `format_tag::any` and lets oneDNN pick a
blocked/VNNI XMX-packed weight layout (once, in process_weights). At m=1 the bottleneck is weight
bytes read -> a coalesced packed layout is the lever most likely to move 52-64% toward peak.

### B5. oneDNN >= v3.8 in the image [low if image allows]
v3.6 first enabled int8-act + int4/int8 grouped-weight matmul on Intel GPU; v3.8 "improved int8 matmul
perf with src+weight zero-points" + Battlemage. Confirm bundled version (`ONEDNN_VERBOSE=1`); upgrade
may recover decode perf for free. Watch oneDNN issue #3323 (B60/Battlemage u4 G=64 decompression mem).

---

## LEVER C — Custom SYCL int4 GEMV (the real GEMM->GEMV fix) [1-2 weeks, biggest kernel win]

oneDNN/IPEX-XeTLA/sycl-tla are ALL GEMM-tuned with no m=1 GEMV path (oneDNN's `grouped_micro_gemm`
just split-K-emulates GEMV -> the 52-64% ceiling). The ONLY real int4 GEMV for Xe2 is **llama.cpp
ggml-sycl**. At m=1 (BW-bound) the systolic array is ~1/16 utilized -> a vectorized-FMA SYCL GEMV with
sub-group-shuffle reductions can target near-peak BW.

- C1. **Benchmark llama.cpp `ggml-sycl/mmvq.cpp` int4 GEMV on the B70** [days] -> a BW-ceiling
  reference + competitive baseline BEFORE writing anything. Tells us how much BW is actually on the table.
- C2. **Write a custom SYCL int4 GEMV** [1-2 wk]: one sub-group (size **16** on Battlemage -- ggml uses
  WARP_SIZE=16 for Intel) per output row; vectorized FMA (skip DPAS); `reduce_over_group`; inline
  register int4->int8 unpack; deferred ZP; double-buffered SLM loads; reordered weight layout for
  coalesced loads. Algorithm template = AWQ `gemv_cuda.cu`; Xe mixed-dtype building blocks =
  sycl-tla `02_bmg_gemm_f16_u4_s8.cpp`; SYCL idioms + reordered layout = ggml-sycl `mmvq.cpp`.
- C3. (later) split-K + shape-specialized dispatch (IPEX `hgemm_policy.h` pattern).
- Alt: Triton-XPU (`intel/intel-xpu-backend-for-triton` lists B70, `tl.dot`->DPAS) for a fused
  quant-prologue+matmul without hand-SYCL -- but has a `FIXME: INT8 hangs` note; verify int8 on BMG first.

---

## Verified code pointers (port-these-first)
- llama.cpp ggml-sycl int4 GEMV: `ggml/src/ggml-sycl/{mmvq.cpp,dmmv.cpp,vecdotq.hpp,dequantize.hpp}`;
  `CMakeLists.txt` sets `GGML_SYCL_WARP_SIZE=16` for Intel. PR #12035 (reorder path).
- intel/sycl-tla: `examples/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_u4_{f16,s8}.cpp` (Battlemage, int8 variant).
- AWQ GEMV (algorithm): `mit-han-lab/llm-awq` `awq/kernels/csrc/quantization/gemv_cuda.cu`.
- IPEX (branch `xpu-main`): `csrc/gpu/oneDNN/{QMatmul.h (int8, conditional src-zp),DnnlMatmulQuant.h
  (W4A8 act_quant_mode)}`; `.../xetla/kernels/include/experimental/group/gemm/impl/int4_dequantize_xe.hpp`;
  `csrc/gpu/aten/operators/{XEGEMM_INT4.cpp,act_dynamic_quant.cpp}`.
- oneDNN: `src/gpu/intel/matmul/{gemm.cpp,grouped_micro_gemm.cpp,ref.cpp}` + `src/gpu/intel/gemm/jit/`;
  example `examples/tutorials/matmul/matmul_with_weight_only_quantization.cpp` (CPU/s8 only).
- Our kernels: `vllm-xpu-kernels/csrc/xpu/onednn/{int4_gemm_w4a8.h,int8_gemm_w8a8.h}`,
  `csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp`; tests `tests/test_int4_gemm_onednn.py`.

## Recommended order when the GPU frees
1. `ONEDNN_VERBOSE=2` on the w4a8 m=1 call (B2) -- free, tells us impl + zp overhead + version.
2. Drop symmetric zp (B1) -> rebuild (`scripts/44`) -> microbench: does 52-64% improve?
3. Extend PIECEWISE graph capture to w4a8 (A1) -> serve `:int8g` -> measure decode lift.
4. (parallel) benchmark llama.cpp ggml-sycl mmvq (C1) as the BW ceiling.
5. Decide the big bet: FULL capture via TRITON_ATTN (A2) vs custom SYCL GEMV (C2).
