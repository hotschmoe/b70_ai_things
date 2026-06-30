# Patch / kernel / env-knob applicability to llama.cpp and zml

Audit date: 2026-06-30. Read-only inventory of the custom Intel-Arc-B70 patches, kernels, shims,
and stability env-knobs that the two EXISTING serving backends (vLLM, sglang) depend on, scored for
the two NEW backends we are adding:

- **llama.cpp (SYCL/GGML)** -- C++/GGML with its own SYCL backend. Weight-only GGUF quant
  (Q8_0, Q4_K_M, ...) with fp16/fp32 compute. NO int8 ACTIVATIONS. Its SYCL GEMM kernels are its
  own (dequant-on-the-fly mmvq/dmmv), NOT our oneDNN ops. Multi-GPU via `--split-mode row/layer`
  + `--tensor-split`, NOT oneCCL / torch-distributed. Its own server + tool-call parser.
- **zml (Zig + MLIR/XLA/PJRT-oneAPI)** -- runs bf16/f16 through XLA. Collectives are
  compiler-visible XLA ops, NOT oneCCL-knob-driven. Does NOT consume compressed-tensors. No torch,
  no custom int8 XPU kernel path. Shipping LLM server is single-GPU today.

Scoring: **DIRECT** = the code or `.so` could be reused as-is; **CONCEPT** = the idea transfers but
must be re-implemented in the new backend's stack; **N/A** = irrelevant (different quant model /
different collective stack / solved differently upstream or by design).

Headline: almost everything is **N/A** for both. The int8/int4 kernels and every quant-scheme shim
are torch + compressed-tensors/Quark + oneDNN-op specific; llama.cpp uses a different (weight-only,
fp16-compute) quant model and its own kernels, and zml has neither int8 nor torch. The oneCCL env
block and the all-reduce surgery are oneCCL/torch-distributed specific; llama.cpp doesn't use oneCCL
and zml's collectives are compiler IR. What DOES transfer is a short list of **hard-won lessons**
(no Arc P2P / route over host, coherence-gating, vision retention, the L0-graph-replay decay, the
single-file bind-mount gotcha) plus a few genuinely cross-stack items (the `bin/xpu-health` /
`bin/xe-reset` wedge tooling, the Level-Zero-v1 SYCL runtime knob).

---

## Table 1 -- llama.cpp (SYCL/GGML)

| Artifact | Class | Why (one line) |
|---|---|---|
| `kernels/int8_gemm_w8a8.h` (prefill s8xs8 XMX) | N/A | Torch C++ extension (`torch::Tensor`, `_xpu_C` op reg, `c10::xpu` stream); GGML has no torch and no per-token int8 activation tensor to feed s8s8s32. |
| `kernels/int8_gemm_w8a16.h` (decode s8 weight x f16 act) | N/A (weak CONCEPT) | The "int8 weight, fp16 act, fused dequant epilogue" idea matches GGML's Q8_0 weight-only model, but GGML already does this in its OWN SYCL mmvq/dmmv kernels and does not link oneDNN for matmul -- a port would be a from-scratch GGML kernel, not reuse. |
| `kernels/int8_gemm_kernel.patch` (oneDNN ext + joint_dtypes) | N/A | Patches `vllm-xpu-kernels` C++; nothing in GGML to patch. |
| `vllm-xpu-env:int8g` `XPUInt8ScaledMMLinearKernel` | N/A | vLLM linear-kernel class over the same torch ops. |
| `sglang/patches/w8a8_shim.py` | N/A | Wires compressed-tensors W8A8-int8 -> torch oneDNN ops; no compressed-tensors, no int8 acts, no torch in GGML. |
| `sglang/patches/w4a8_shim.py` | N/A | Defines a torch compressed-tensors W4A8 scheme -> `int4_gemm_w4a8`. Irrelevant to GGUF. |
| `sglang/patches/woq_shim.py` (woqgemm route + lm_head int4 + cuda->xpu redirects) | N/A | sglang GPTQ-scheme + auto_round_kernel + `torch.cuda.*`->`torch.xpu.*` redirects; all torch/sglang. |
| `sglang/patches/quark_moe_int8.py` | N/A | sglang Quark int8 MoE loader -> in-tree Triton int8 fused_moe; GGUF MoE is weight-only with GGML's own kernels. |
| vLLM `compressed_tensors_w4a8_int.py` / `xpu.py` (XPUwNa16, XPUW4A8Int) | N/A | vLLM scheme + kernel wiring to the same oneDNN int4 ops. |
| vLLM 35B `quark.py` (int8 MoE dispatch + XPU dequant) | N/A | vLLM Quark config monkeypatch. |
| `sglang/patches/int8_actquant_xpu.py` (libdevice.round -> floor/ceil) | N/A (CONCEPT note) | Fixes a CUDA-libdevice intrinsic that won't link on triton-xpu. GGML SYCL is hand-written, has no CUDA intrinsics, and llama.cpp doesn't do int8 act-quant at all. The general "CUDA-isms don't port to Intel" lesson holds but is sidestepped by design. |
| `sglang/patches/w4a8_actquant_triton.py` (1-launch per-token int8) | N/A | Triton act-quant; no int8 acts, no Triton in llama.cpp. |
| `sglang/patches/xpu_cudagraph.py` (torch.xpu.XPUGraph wiring) | CONCEPT | GGML's SYCL backend has NO graph capture today (only the CUDA backend has CUDA graphs). Capturing the decode path for a 2.5x win is a valid idea, but would be a fresh GGML-SYCL implementation -- AND inherits the hard caveat that XPU graph REPLAY decays over a soak (see lessons). |
| `sglang/patches/mtp_tree_xpu.py` (EAGLE/NEXTN torch fallbacks + None-index fix) | N/A (weak CONCEPT) | Pure-torch EAGLE tree kernels + sglang device-gate fixes. llama.cpp has its OWN C++ draft-model speculative path; it does not implement qwen3-next NEXTN self-spec. "MTP is a stable speedup worth wiring" is a research note, not portable code. |
| `sglang/patches/fused_gdn_gating.py` (Triton GDN gating) | N/A (weak CONCEPT) | Triton kernel. IF llama.cpp ever supports qwen3-next/GDN it needs its own GGML SYCL gated-delta kernels; fusing the gating is a generic optimization but a clean-room reimplementation. |
| `sglang/patches/gdn_fused_conv.py` (fuse causal_conv1d into GDN recurrence) | N/A (weak CONCEPT) | Same: Triton, B=1 decode; relevant only as an optimization pattern for a future GGML GDN kernel, not as code. |
| `sglang/patches/qwen3_coder_detector.py` (incremental tool-arg streaming) | CONCEPT | The code is a copy of sglang's qwen3_coder detector. But the BUG is generic: a server that buffers a long string tool-arg until `</parameter>` emits zero bytes for minutes and trips client idle timeouts. llama.cpp's server has its OWN tool-call/grammar parser -- re-verify it streams arg deltas incrementally (likely fine, but worth a check for qwen/Hermes-style calls). |
| vLLM `qwen35_text_hybrid.py` + arch-reg sitecustomize | N/A (CONCEPT note) | vLLM model-loading shim (load text-only LM, graft GDN-state classmethods, mrope). llama.cpp loads GGUF; vision is a separate `mmproj`. The "don't silently drop the vision tower" discipline transfers (see lessons). |
| vLLM `sitecustomize.py` capture-safe all_gather (all-reduce of padded buf) | N/A | Surgery to make oneCCL all_gather graph-recordable under torch piecewise capture. llama.cpp has no oneCCL and no in-graph collective. |
| vLLM cg-recycle (recapture PIECEWISE every N steps) | N/A (CONCEPT note) | Bounds L0 command-stream accumulation in vLLM graphs; the underlying L0-decay lesson transfers if GGML adds SYCL graphs. |
| MGPU env: `CCL_ENABLE_SYCL_KERNELS` | N/A | oneCCL allreduce-capturability knob; llama.cpp `--split-mode` uses GGML device copies, not oneCCL. |
| MGPU env: `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` | N/A | oneCCL topology knob. |
| MGPU env: `CCL_ATL_TRANSPORT=ofi` | N/A | oneCCL transport selection. |
| MGPU env: `CCL_TOPO_P2P_ACCESS` | N/A (CONCEPT note) | oneCCL knob has no effect, BUT the no-Arc-P2P / host-staged-fabric REALITY hits `--split-mode row` cross-card traffic just as hard (see lessons). |
| MGPU env: `CCL_ZE_IPC_EXCHANGE=pidfd` | N/A | oneCCL IPC-handle exchange method. |
| MGPU env: `VLLM_WORKER_MULTIPROC_METHOD=spawn` | N/A | vLLM/torch multiproc; llama.cpp is single-process. |
| MGPU env: `SYCL_UR_USE_LEVEL_ZERO_V2=0` | CONCEPT (near-DIRECT) | This is a SYCL/Unified-Runtime adapter knob, NOT oneCCL -- it selects the Level-Zero v1 backend for ANY oneAPI SYCL program. llama.cpp's SYCL backend runs on the same UR/L0; worth setting + A/B-ing for B70 stability. |
| Wedge guard: `bin/xpu-health` / `bin/xe-reset` | DIRECT | Backend-agnostic shared tools: per-card matmul probe + xe reload/reboot recovery. Wrap ANY multi-GPU llama.cpp run with them. |
| Wedge guard: `lib.sh` pre-flight/teardown/health-wait integration | CONCEPT | The torch/oneCCL-specific glue (graceful `docker stop` not SIGKILL-mid-collective, stall-aware wait) is the right OPERATIONAL pattern, re-expressed around the llama.cpp server. |

## Table 2 -- zml (Zig + MLIR/XLA/PJRT-oneAPI)

| Artifact | Class | Why (one line) |
|---|---|---|
| `kernels/int8_gemm_w8a8.h` / `int8_gemm_w8a16.h` / `int8_gemm_kernel.patch` | N/A | zml runs bf16/f16 via XLA; no int8 path, no torch ops, no oneDNN-op surface to register into. |
| `vllm-xpu-env:int8g` `XPUInt8ScaledMMLinearKernel` | N/A | vLLM/torch class. |
| `sglang/patches/w8a8_shim.py` / `w4a8_shim.py` / `woq_shim.py` / `quark_moe_int8.py` | N/A | All wire compressed-tensors/Quark + torch oneDNN ops; zml consumes neither compressed-tensors nor torch. |
| vLLM `compressed_tensors_w4a8_int.py` / `xpu.py` / 35B `quark.py` | N/A | vLLM scheme/kernel/config wiring. |
| `sglang/patches/int8_actquant_xpu.py` / `w4a8_actquant_triton.py` | N/A | Triton int8 act-quant kernels; XLA has no Triton and no int8 acts. |
| `sglang/patches/xpu_cudagraph.py` | N/A | XLA/PJRT compiles the whole executable; "graph capture" is owned by the compiler, not wired by hand. No manual capture to port. |
| `sglang/patches/mtp_tree_xpu.py` | N/A | sglang EAGLE/torch internals; zml has no equivalent spec-decode layer in the shipping server. |
| `sglang/patches/fused_gdn_gating.py` / `gdn_fused_conv.py` | N/A (weak CONCEPT) | Triton GDN kernels. In zml, GDN would be StableHLO ops the XLA compiler fuses/schedules -- the fusion is the compiler's job, not a hand kernel. |
| `sglang/patches/qwen3_coder_detector.py` | N/A | zml's shipping server is single-GPU bf16 and is not the OpenAI tool-call frontend we'd run agentic calls through; no qwen3_coder parser to patch. |
| vLLM `qwen35_text_hybrid.py` + arch-reg / MTP-unquant sitecustomize | N/A (CONCEPT note) | vLLM model-loading internals; zml has its own Zig model defs + weight conversion. Vision-retention discipline still applies. |
| vLLM capture-safe all_gather (all-reduce of padded buffer) | N/A (this is the CONCEPT argument FOR zml) | The whole problem -- an opaque host-launched collective forces a graph break, so you hand-fuse around it -- is exactly what a StableHLO/XLA stack solves BY DESIGN (collective-as-IR, SPMD partitioner inserts + overlaps them). Don't port the surgery; the surgery's absence is zml's structural advantage. |
| vLLM cg-recycle | N/A (CONCEPT note) | Bounds torch-graph L0 accumulation; the L0-decay risk could still surface under PJRT's compiled execution -- watch it, don't port the code. |
| MGPU env: `CCL_ENABLE_SYCL_KERNELS` / `CCL_TOPO_*` / `CCL_ATL_TRANSPORT` / `CCL_ZE_IPC_EXCHANGE` / `CCL_TOPO_P2P_ACCESS` | N/A | zml collectives go through XLA->PJRT, not oneCCL env knobs. The no-P2P fabric reality still bounds any cross-card collective (see lessons). |
| MGPU env: `VLLM_WORKER_MULTIPROC_METHOD=spawn` | N/A | vLLM-specific; zml is a single Zig binary. |
| MGPU env: `SYCL_UR_USE_LEVEL_ZERO_V2=0` | CONCEPT | zml's PJRT-oneAPI plugin runs on Level Zero; the same v1-vs-v2 stability concern may apply at the runtime layer (verify whether the PJRT plugin honors the UR env). |
| Wedge guard: `bin/xpu-health` / `bin/xe-reset` | DIRECT | Backend-agnostic card probe + recovery; wrap any zml mesh/multi-GPU run with them. |
| Wedge guard: `lib.sh` integration | CONCEPT | Operational pattern (probe before/after, never kill mid-collective, reboot-recovery) re-expressed around zml. |

---

## Portable lessons (NOT code)

These are the hard-won insights that transfer regardless of backend. They cost the most to learn and
are the real deliverable of porting to a new stack.

1. **No working Arc-B70<->B70 P2P on this box's history; route collectives over host.** On kernel
   <7.0 / cross-die PCIe Gen3 there is NO peer DMA (`canAccessPeer=False`); torch d2d copy is
   launch/sync-bound, not peer-direct. Kernel 7.0 + IOMMU-off opens P2P at the allreduce layer
   (~9.7 GB/s, 8.4x host-staged) but it is a **prefill/TTFT + concurrency** win, NOT a single-stream
   decode win, AND `CCL_TOPO_P2P_ACCESS=1` inside a TP>1 serve WEDGES the box (reboot-only recovery).
   Implication for the new backends: llama.cpp `--split-mode row` and any zml mesh pay the
   host-staged fabric ceiling for every cross-card hop. Prefer single-card if the model fits; treat
   any GPU-to-GPU number as host-staged until measured otherwise.

2. **The first-order TP cost is the graph break around an opaque collective, not the wire.** vLLM's
   pain (eject the collective -> capture-address contract breaks -> garbage; keep it captured ->
   oneCCL `sched` algo can't be recorded) drove the all-reduce-surgery and `CCL_ENABLE_SYCL_KERNELS`
   work. zml/XLA dissolves this structurally (collective is compiler IR, fused + overlapped
   automatically) -- a genuine architectural reason to prefer it for TP. llama.cpp sidesteps it by
   not doing in-graph collectives at all. Either way: do NOT recreate the surgery; pick a stack
   where the collective is visible to the compiler or absent.

3. **XPU graph REPLAY decays over a soak (Level-Zero command-stream/command-list accumulation).**
   Capture gives ~2.5x initially (23 t/s) then degrades to ~8 t/s with multi-second stalls on both
   sglang-XPUGraph and vLLM-PIECEWISE; no env knob fixed it; the only mitigations are recapture/recycle
   or stay eager. If llama.cpp ever adds SYCL graph capture, or zml's PJRT compiled execution shows
   the same decay, this is the trap to expect.

4. **Vision-tower-retention discipline.** Some quant pipelines silently drop the ~333 `visual.*`
   tensors despite a `vision_config`. Every Qwen3.6 conversion (GGUF `mmproj` for llama.cpp, zml
   weight conversion) must be checked for a surviving vision tower, not assumed.

5. **Coherence-gate the "best" config; the failure mode that matters is concurrent prefill+decode.**
   "Best" = coherent-under-load FIRST, then fast (vLLM's "!!!!" garbage under mixed load is the canonical
   trap). Any llama.cpp/zml "best" serve config must be validated under CONCURRENT load, not just
   single-stream benched.

6. **Single-file Docker bind-mount inode gotcha.** Editing a single-file bind mount (a config, a
   patched parser) needs `docker restart`, not an in-process reload -- atomic-write editors create a
   new inode the running mount can't see (reload reports OK, serves the stale file). Applies to any
   containerized llama.cpp/zml serve mounting single-file configs.

7. **CUDA-isms don't port to Intel.** A large fraction of the sglang work is monkeypatching
   `torch.cuda.*`, `is_cuda()` device gates, `tl.extra.cuda.libdevice.*` intrinsics, and CUDA-only
   sgl-kernel ops. Both new backends avoid this CLASS of problem by design (llama.cpp writes SYCL
   directly; zml lowers to XLA) -- a real maintenance win, and a reason their B70 enablement should
   be far thinner than the torch backends' shim pile.

8. **Tool-call argument streaming.** Stream string tool-arg deltas incrementally; do not buffer a
   long string param until its close token (minutes of zero bytes trips client idle timeouts). When
   wiring llama.cpp's server for agentic tool-calling, verify its parser streams arg deltas.

9. **Cross-stack runtime knobs worth carrying.** `SYCL_UR_USE_LEVEL_ZERO_V2=0` (Level-Zero v1) is a
   SYCL/Unified-Runtime knob, not oneCCL -- it applies to any oneAPI SYCL program (llama.cpp SYCL,
   zml PJRT-oneAPI). And `bin/xpu-health` + `bin/xe-reset` are already backend-agnostic card-probe /
   recovery tools -- reuse them DIRECTLY as the operational guard around any multi-GPU run on this box,
   independent of which serving stack triggers a wedge (including the firmware-level BCS copy-engine
   job-timeout wedge, which is hardware, not torch).
