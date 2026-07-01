# zml fused int8 GEMM optimization -- integration-path study (decode w8a16 + prefill w8a8)

Goal: get a FUSED int8 decode matmul into the zml (oneAPI PJRT / XLA / StableHLO) qwen3.6-27b W8A8
serve on dual B70 -- s8 weight x f16/bf16 activation, dequant in the matmul epilogue, NO bf16 weight
materialization -- and confirm the w8a8 prefill XMX path is already saturated. This doc ranks four
integration paths with feasibility verdicts grounded in the ACTUAL prebuilt plugin, the zml source,
and the reference kernels. Read alongside `zml/ZML_INT8_PERF_HANDOFF.md` (mission/roadmap) and
`zml/W8A8_SWEEP_RESULTS.md` (the numbers).

Investigation date: 2026-07-01. Sources verified: `kernels/*.h`, the zml checkout at
`/mnt/vm_8tb/b70/zml`, and the prebuilt plugin binary (symbols/strings).

--------------------------------------------------------------------------------------------------
## 0. TL;DR / recommendation

- **The prebuilt plugin is XLA's generic GPU PJRT plugin retargeted to SYCL/Level-Zero.** It fully
  exposes the XLA **typed FFI** custom-call registry AND an FFI **stream getter that returns the
  device SYCL queue**. This is the single most important finding: it makes an FFI-to-native-oneDNN
  custom call (Path 2) genuinely feasible, and it is exactly how zml's own flashattn CUDA kernel is
  wired. The handoff doc listed this as "UNKNOWN"; it is now CONFIRMED possible.
- **The plugin's bundled oneDNN is CPU-only.** The XPU int8 GEMM does NOT run on oneDNN today; it
  runs on MKL SYCL BLAS and/or the plugin's built-in **Intel Triton (SPIR-V)**. So there is no
  "oneDNN weight-decompression pass" to pattern-match into on XPU -- Path 1 must instead target
  XLA's **subchannel-dequantisation -> Triton GEMM fusion**, which IS compiled in but is OFF for the
  oneapi target by default.
- **HLO dumping and XLA flags DO work via PJRT compile-options overrides** (not the env var). This
  contradicts the handoff's "prebuilt plugin ignores dumps": you can dump optimized/per-pass HLO and
  flip experimental fusion flags for the oneapi target. This de-risks Path 1's A/B considerably.
- **Prefill (w8a8, large-M dotAcc) is already at the realistic ceiling** (~67-69% of 367 TOPS, 2.1-
  2.6x bf16). It needs NOTHING structural. Decode (M=1/2) is the sole headroom.

**Recommended order: Path 1 first (one-day flag+HLO-dump A/B), then Path 2 (the durable win),
keep Path 4-Triton as the "milk the M=1 GEMV" endgame, and DROP Path 3 (build-your-own-plugin) as
not worth it now that FFI + stream + HLO dump all work with the prebuilt binary.**

--------------------------------------------------------------------------------------------------
## 1. Ground truth about the prebuilt oneAPI plugin

Binary: `libpjrt_oneapi.so` (462 MB), release `manual-2026-06-23` pinned in
`platforms/oneapi/oneapi.bzl:4` (`PJRT_ONEAPI_RELEASE`), fetched as prebuilt in
`platforms/oneapi/libpjrt_oneapi.BUILD.bazel:4`. Loaded at `platforms/oneapi/oneapi.zig:36` (`load`)
via `pjrt.Api.loadFrom`. It bundles MKL SYCL BLAS, oneCCL 2022, the DPC++/SYCL runtime, and
Level-Zero adapters (`platforms/oneapi/oneapi.bzl:110-148`).

What the binary actually contains (from symbol/string inspection):

- **It is `pjrt::gpu_plugin`** -- symbol `pjrt::CreatePjrtApi(...)` and `pjrt::gpu_plugin::
  PJRT_Register_Custom_Partitioner`. i.e. Intel-extension-for-openxla is built on XLA's standard GPU
  PJRT, with `stream_executor::sycl::SyclStream` / `SyclStreamPool` / `SyclBlasSupport` replacing the
  CUDA backend. Consequence: everything true of the CUDA GPU plugin's FFI/Triton/stream machinery is
  structurally available here, just over SYCL.

- **Typed FFI is present and complete.** C-API symbols: `XLA_FFI_Handler_Register`,
  `XLA_FFI_Stream_Get`, `XLA_FFI_ExecutionContext_Get_Args`, `XLA_FFI_DeviceMemory_Allocate_Args` /
  `_Free_Args`, `XLA_FFI_State_Get/Set`, `XLA_FFI_Future_*`, `XLA_FFI_DeviceOrdinal_Get`; plus
  `xla::ffi::GetXlaFfiApi()` and `pjrt::CreateFfiExtension(PJRT_Extension_Base*)`. String
  `"API_VERSION_TYPED_FFI"` and `"Called ResolveLegacyCustomCall with API_VERSION_TYPED_FFI"`.
  -> `PJRT_Api` advertises the `PJRT_FFI` extension, so `pjrt.Api.ffi()` (`pjrt/pjrt.zig:271`) returns
  non-null on oneAPI and `Platform.registerFfi` (`zml/platform.zig:551`) will succeed.

- **FFI handlers receive a device stream that is a SYCL queue.** Internal handler symbols are bound
  with `xla::ffi::Ctx<Stream>` + `Buffer` and take a `stream_executor::Stream*` (e.g. the
  `kBufferDebugChecksumLogInitHandler` handler signature). The stream concrete type is
  `stream_executor::sycl::SyclStream`, whose platform handle is a `sycl::queue`. The plugin's own GEMM
  path proves this: `stream_executor::sycl::SyclBlasSupport::DoBlasGemm(...)` and
  `DispatchOneMklGemm(...)` pull the `sycl::queue` off the stream ("SYCL GEMM stream has null queue"
  guard). So `call_frame.api.stream(call_frame.ctx)` (`pjrt/ffi.zig:84`, `XLA_FFI_Stream_Get`) yields
  the queue we need for `dnnl::sycl_interop`.

- **Bundled oneDNN is CPU-ONLY.** All dnnl matmul symbols/strings are `xla::cpu` +
  `external/onednn/src/cpu/matmul/*` (`gemm_x8s8s32x_matmul.cpp`, `ref_matmul_int8.cpp`,
  `x64:gemm_s8s8s32:jit`, avx512 brgemm) and `xla::cpu::OneDnnMatMulConfig` / `onednn_matmul.cc`.
  There is **no GPU/SYCL oneDNN matmul** in the plugin. So on XPU the int8 `dot_general` does NOT go
  through oneDNN; it lowers to MKL SYCL BLAS or Triton. (This is why the b70-int8-xmx-roofline note
  "int8 dispatches to oneDNN" is a CPU-side statement; on this XPU plugin the executor is MKL/Triton.)

- **Intel Triton is compiled in.** `TritonIntelGPUAccelerateMatmulPass`,
  `TritonIntelGPULowerTo2DBlockLoadPass`, `IntelGPUPipelinePass`, `TritonIntelGPURemoveLayoutConversions`,
  `Subgroup2DBlockLoadOp` (the Xe 2D-block load feeding XMX/DPAS), SPIR-V codegen, and the
  `PJRT_Triton_Extension` (`pjrt/pjrt.zig:200`). The custom-call target `__gpu$xla.gpu.triton` (the
  one zml's `ops.triton` emits, `zml/ops.zig:682`) is handled. zml already tunes a Triton attention
  kernel for Intel decode (`zml/attention/triton_attention.zig:96` "Intel decode needs more warps"),
  which is strong evidence the zml Triton path runs on this plugin.

- **A subchannel-dequant -> Triton-GEMM fusion pass exists** but is experimental/off by default:
  strings `dequantize_dot.cc`, `xla_gpu_experimental_enable_subchannel_dequantisation_fusion`
  ("Enable fusion for the subchannel dequantisation sequences like [x,z]param -> broadcast ->
  bitcast -> multiply -> dot"), `gemm_fusion.cc`, `+xla_gpu_experimental_scaled_dot_with_triton`,
  `DynamicDequantize`, `dequantize_mode`. This is the ONLY automatic weight-decompression-into-matmul
  mechanism on the XPU side, and it is a Triton GEMM under the hood.

- **XLA flags + HLO dumps go through compile-options overrides, and the plugin honors them.** zml
  builds a `CompileOptionsProto` and writes `env_option_overrides` (`zml/module.zig:626`), setting
  per-target flags for `.oneapi` today: `xla_gpu_autotune_level=0`, `xla_gpu_enable_command_buffer=""`,
  `xla_gpu_enable_cublaslt=false` (`zml/module.zig:648-652`). It ALSO wires `xla_dump_to` +
  `xla_dump_hlo_as_proto` + `xla_dump_hlo_pass_re` through the same map (`zml/module.zig:656-665`).
  So: (a) you CAN dump the real optimized/per-pass HLO from the prebuilt binary, and (b) you CAN flip
  the subchannel-dequant fusion flag for oneapi. The "plugin ignores XLA_FLAGS" caveat is about the
  ENV var only.

--------------------------------------------------------------------------------------------------
## 2. What zml emits today, and why decode is slow

- The int8 math is `Tensor.dotAcc(.i32, w, tag)` (`zml/tensor.zig:1418`) -> `dotGeneralAcc`
  (`zml/tensor.zig:1353`) -> `stablehlo.dot_general` with genuine s8 operands, result element type =
  s32 (`preferred_element_type`, `zml/tensor.zig:1399-1413`), `.dot_precision = .fast`. This is the
  correct int8-XMX trigger and is why prefill hits XMX.
- Decode column-parallel (q/k/v/gate/up) uses `forwardQuant` (`common_quant.zig:201`): per-token
  act-quant prologue + `dotAcc(.i32)` s8xs8 + f32 dequant epilogue. At M=1 this is the oneDNN/MKL
  int8-GEMV trap: 361 GB/s vs bf16's 460 GB/s = only 1.57x raw, 1.20x with the act-quant prologue
  (`W8A8_SWEEP_RESULTS.md` rows M=1). The act-quant is pure overhead at M=1 (no XMX benefit when
  bandwidth-bound).
- Decode row-parallel (o_proj/down_proj) uses `weightOnly` (`common_quant.zig:91,133-137`) which
  calls `dequantWeight` (`common_quant.zig:222`): `weight.convert(.f32).mul(scale).convert(dtype)`
  then `x.dot(wf)`. This MATERIALIZES a full bf16 weight -> reads i8 but writes+reads bf16 -> no
  bandwidth win (0.5-1.2x, dead end -- `W8A8_SWEEP_RESULTS.md` finding 7). It is kept only because it
  is TP-coherent, not for speed.

The missing piece is exactly `kernels/int8_gemm_w8a16.h:20` (`dnnl_matmul_w8a16_int8`): pass the s8
weight straight in, set `pattr.set_scales(DNNL_ARG_WEIGHTS, ...)` (`int8_gemm_w8a16.h:87-110`) +
`pattr.set_fpmath_mode(f16/bf16, allow_downconvert=true)` (`int8_gemm_w8a16.h:111-114`) so the
dequant happens IN the matmul epilogue on the f16/bf16 XMX path -- one launch, reads only int8 bytes.
Every path below is a different way to realize that fusion under zml/XLA.

--------------------------------------------------------------------------------------------------
## 3. The four integration paths, ranked

### PATH 1 -- Emit HLO the plugin auto-fuses (subchannel-dequant -> Triton GEMM). Verdict: TRY FIRST (cheap), MODERATE confidence.

Feasibility: PLAUSIBLE now that we know (a) the fusion pass exists in the binary and (b) we can flip
its flag and dump HLO via compile options. The current `weightOnly` "materialize" result (0.5-1.2x)
is consistent with the fusion NOT firing by default -- because the flag is off AND the emitted shape
(`i8 -> convert f32 -> mul f32 -> convert bf16 -> dot`, `common_quant.zig:222`) has a double convert
and an f32 multiply that likely fail the matcher, which wants `s8 param -> convert(compute) ->
broadcast(scale) -> multiply -> dot`.

Concrete steps:
1. Add the fusion flag for oneapi in `zml/module.zig:648` (the `.oneapi` switch arm), next to the
   existing `setXlaOverrideFlag` calls:
   `setXlaOverrideFlag(overrides_map, "xla_gpu_experimental_enable_subchannel_dequantisation_fusion", true, ...)`
   and ensure `xla_gpu_enable_triton_gemm` is true (the fusion output is a Triton GEMM; do not set it
   false for oneapi).
2. Rewrite `dequantWeight` (`common_quant.zig:222`) to the canonical single-convert form the pass
   matches: `weight.convert(bf16).mul(weight_scale.convert(bf16).broad(...))` (no f32 hop), fed
   straight into `x.dot(...)`. Keep it behind a variant flag so the current path stays as fallback.
3. Turn on HLO dump via the compile options (`zml/module.zig:656`, i.e. set `opts.xla_dump_to`) and
   confirm in the optimized HLO that the `multiply`/`convert` folded INTO the dot fusion (a
   `__triton_gemm`/`fusion` with an s8 parameter and no standalone bf16 `multiply` buffer). If a
   materializing `multiply` remains, the fusion did not fire.
4. A/B microbench in `//examples/w8a8_sweep` (add a `woq_fused` variant next to `woq` at
   `examples/w8a8_sweep/main.zig` Variant enum) at the real shapes, M=1,2,4,8. Success = int8-weight
   decode >= bf16-weight bandwidth AND ~2x today's woq; must stay byte-identical.

Files to touch: `zml/module.zig:648-652` (flag), `examples/llm/models/common_quant.zig:222` (canonical
HLO), `examples/w8a8_sweep/main.zig` (bench variant). Effort: ~1-2 days. Risk: LOW (pure flag + HLO
shape; falls back cleanly). Expected win: IF it fires, decode matmul becomes bandwidth-bound (~2x);
the fusion help-text warns "performance can be worse ... split-k > 1 not considered", so the M=1 GEMV
win is NOT guaranteed even when it fires -- this is a probe, not a commitment. Ceiling is the same
Intel-Triton GEMM as Path 4-Triton, so a null result here also informs Path 4.

### PATH 2 -- XLA typed-FFI custom call to a native oneDNN-SYCL w8a16 kernel. Verdict: RECOMMENDED (the durable win), HIGH confidence it can work.

Feasibility: CONFIRMED possible. The plugin exposes the FFI registry + an FFI stream getter that
returns the device SYCL queue (section 1), and zml already has the full typed-FFI wrapper AND a
working precedent (flashattn) that does precisely "register FFI handler -> get stream -> call a native
GPU library on raw device buffers":
- `zml/attention/flashattn.zig:128` `register` -> `platform.registerFfi(...)` (CUDA today,
  `.platform_name = "cuda"`).
- `zml/attention/flashattn.zig:188` `const stream = call_frame.api.stream(call_frame.ctx);` then
  calls the native `flashattn.fa2_mha_varlen_fwd(...)` on the FFI buffers.
- The generic machinery: `zml/ops.zig:2014` `register()` -> `platform.registerFfi`; `zml/ops.zig:2050`
  `handler(call_frame)`; `zml/ops.zig:2053` `customCallArgsFromPjrtCallFrame`; the emit side is
  `zml/ops.zig:1508` `customCall(...)` (typed_ffi backend_config). Buffers arrive as
  `pjrt/ffi.zig:257` `Buffer{ dtype, data:[*]u8, rank, dims }`; device scratch via
  `pjrt/ffi.zig:101` `allocateDeviceMemory`.

This is the sglang/vLLM approach (`kernels/int8_gemm_w8a16.h`) adapted to XLA: same oneDNN primitive
attrs, but the torch plumbing (`c10::xpu::getCurrentXPUStream`, `GpuEngineManager`, `at::Tensor`) is
replaced by `dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context())` +
`make_stream(engine, queue)` built from the FFI `sycl::queue`, operating on the raw USM pointers in
the FFI Buffers.

Concrete steps:
1. Native lib `platforms/oneapi/int8gemm/int8_gemm_xpu.cpp` (+ BUILD, mirroring
   `platforms/cuda/flashattn/`): a C-ABI entry `zml_onednn_w8a16(queue*, dst, src, wgt_s8, wscale,
   m,n,k, lda,ldb,ldc, dtype, scale_kind)` that ports the body of `kernels/int8_gemm_w8a16.h:87-155`
   (set_scales WEIGHTS + fpmath_mode f16/bf16 + create-and-cache the primitive + execute on the SYCL
   stream). Link against a **GPU-enabled oneDNN** (DNNL_GPU_RUNTIME=SYCL, DPC++ 2026.0 -- same
   toolchain the plugin bundles). sglang already builds this exact oneDNN-GPU (`kernels/README.md:22`,
   `research/w8a8/W8A8_BUILD.md`), so the kernel math is proven; only the engine/stream glue is new.
   Fastest bootstrap: dlopen the already-built sglang `_xpu_C.abi3.so`-adjacent oneDNN, like flashattn
   dlopens its lib -- but cleanest is a bazel `cc_library` dep on oneDNN-GPU.
2. zml wrapper `zml/int8_gemm.zig`: an `ops.CustomCall`-style type (Inputs `{src, weight_s8,
   weight_scale}`, Output `{y}`, Attributes `{m,n,k,transpose,scale_kind}`) whose `register` targets
   the oneapi platform (`platform_name = null` -> defaults to the XPU platform name via
   `zml/platform.zig:552`) and whose handler mirrors `flashattn.zig:137-200`. Register it at platform
   init next to `flashattn` (`zml/platform.zig:319`), gated on `target == .oneapi`.
3. Wire it into `QuantizedLinear` as the DECODE path: replace `forwardQuant`'s dotAcc
   (`common_quant.zig:205`) and `weightOnly`'s materialize (`common_quant.zig:133-137`) with a call
   to the custom op when M is small (decode) and a `weight_scale` is present; keep `dotAcc` for
   large-M prefill. Start with the 5 column-parallel + the GDN projections (replicated contract axis,
   no collective); for row-parallel o/down use the `sharding_aware = true` custom call (`ops.zig:1997`
   / `shardingAwareTypedCustomCall`, `ops.zig:2079`) so it runs per-shard inside a manual block, or
   keep weightOnly there initially.
4. One-time weight VNNI/blocked reorder cached at load (oneDNN wants its packed weight format for
   best XMX); the primitive create-and-cache already keys on shapes.

Files to touch/create: `platforms/oneapi/int8gemm/*` (new native lib + BUILD), `zml/int8_gemm.zig`
(new), `zml/platform.zig:319-341` (register for oneapi), `examples/llm/models/common_quant.zig:133,
201-215` (route decode through it), `examples/w8a8_sweep/main.zig` (bench variant). Effort: ~1-2
weeks (the oneDNN-GPU build + sycl_interop glue is the bulk; kernel math is copy-from-kernels/*.h).
Risk: MEDIUM -- build/link of oneDNN-GPU against the plugin's SYCL runtime, USM pointer/stream
lifetime, command-buffer compatibility (set `command_buffer_compatible=false` first, matching
`zml/module.zig:650` which disables command buffers for oneapi anyway). Expected win: the real target
-- reads only int8 weight bytes on the f16/bf16 XMX path -> decode matmul bandwidth-bound, ~2x today
and >= bf16 bandwidth; this is the same kernel that gave sglang 23.8 t/s on this box.

### PATH 3 -- Build our own oneAPI PJRT plugin / XLA oneDNN-GPU pass. Verdict: DROP for now.

Feasibility: possible but unjustified. The two reasons the handoff floated it -- "can't dump HLO" and
"can't register custom calls on XPU" -- are BOTH false (sections 1 and 2: HLO dump via compile opts
works; typed FFI + stream work). Building intel-extension-for-openxla from source (SYCL + Triton +
XLA + oneDNN-GPU) is a multi-week toolchain slog, replaces a working pinned artifact
(`oneapi.bzl:4-8`), and its only unique benefit (a native oneDNN-GPU weight-decompression fusion pass)
is obtained more cheaply by Path 2 (same oneDNN kernel, via FFI, no plugin fork). Keep as a last
resort ONLY if both Path 1 and Path 2 fail for a plugin-internal reason. Effort: 3-6 weeks. Risk:
HIGH. Win: same ceiling as Path 2.

### PATH 4 -- Hand-written kernel as a zml custom op. Verdict: KEEP as the M=1 endgame; prefer the Triton variant over XeTLA/ESIMD.

Two sub-variants:

- **4-Triton (preferred):** write the int8 GEMV/GEMM as a Triton kernel via zml's existing
  `zml.kernel.triton.Kernel` (`zml/kernel.zig:96-159`) -> `ops.triton` -> `__gpu$xla.gpu.triton`
  (`zml/ops.zig:642-698`), which the plugin's Intel Triton JITs to SPIR-V (section 1;
  `TritonIntelGPUAccelerateMatmulPass` maps `tl.dot(s8,s8)->i32` to DPAS, `Subgroup2DBlockLoad` feeds
  XMX). Precedent that zml Triton runs on Intel: `zml/attention/triton_attention.zig:96`. This lets
  us hand-tune the exact M=1/2 decode GEMV (block sizes, warps, 2D-block weight load, epilogue dequant
  with per-token+per-channel scales) that oneDNN/MKL leaves ~30% of bandwidth on the table for
  (`W8A8_SWEEP_RESULTS.md` finding 3). Effort: ~1-2 weeks per shape family; reuses a tested zml
  surface. Risk: MEDIUM (Intel-Triton int8-GEMV maturity at M=1). Win: potentially BEST at M=1 (closes
  the 361->~460+ GB/s gap) since it is purpose-built for the trap case.

- **4-native (XeTLA/ESIMD/Level-Zero/Gen-asm):** same FFI plumbing as Path 2 but the native kernel is
  hand-written XeTLA/ESIMD instead of oneDNN. Only pursue if BOTH oneDNN (Path 2) and Triton
  (Path 4-Triton) leave the M=1 GEMV short. Effort: multi-week, deep. Risk: HIGH. Win: the theoretical
  max, for the specific shapes q_proj[12288,5120], o_proj[5120,6144], gate/up[17408,5120],
  down[5120,17408], GDN in_proj[10240,5120] at M=1,2,4,8.

Files (either variant): `zml/int8_gemm.zig` (or a `triton` Kernel spec), `common_quant.zig` decode
route, `examples/w8a8_sweep/main.zig` bench.

--------------------------------------------------------------------------------------------------
## 4. Prefill (w8a8) status -- confirmed saturated, do nothing

From `W8A8_SWEEP_RESULTS.md` (median timing, real shapes) and memory `b70-int8-xmx-roofline`:
- int8 `dotAcc(.i32)` at large M saturates ~245-253 TFLOP/s = **67-69% of the 367-TOPS peak**, i.e.
  the realistic oneDNN/MKL/Xe2 int8 ceiling (feeding/occupancy-bound, not systolic-peak-bound).
- int8 vs bf16 = 2.1-2.62x at M=512-2048, converging to ~1.7-1.8x at M>=8192 as bf16 also scales.
- `nk` weight `{n,k}` layout is optimal (kn is 1.7x slower -- do not transpose); i32-store beats
  bf16-store; the model already stores `{n,k}`. These are already in place in `dotAcc`.
Verdict: prefill needs NO structural change. Two watch-items only: (a) the M>=8192 act-quant "STALL"
(hard 10s stalls; `W8A8_SWEEP_RESULTS.md` caveat) -- avoid the largest-M act-quant path until root-
caused; (b) re-run `//examples/w8a8_sweep --shape=all` after any tensor.zig change to confirm no
regression. The prefill lever is spent; all remaining headroom is DECODE (M=1/2), addressed by
Paths 1/2/4.

--------------------------------------------------------------------------------------------------
## 5. Recommended path + first experiment

Recommended sequence:
1. **Path 1 as a 1-day probe** (flag + canonical HLO + dump). It is nearly free and its result also
   tells you what the Intel-Triton GEMM can do (informing Path 4-Triton). If it fires AND beats bf16
   at M=1/2, you may be done for the column-parallel/GDN projections.
2. **Path 2 as the durable landing** (the fused oneDNN-SYCL w8a16 via FFI) regardless of Path 1's
   outcome -- it is the proven-on-this-box kernel and the most robust. Land it for the 5 column-
   parallel + GDN projections first (biggest decode-byte share, no collective), then extend.
3. **Path 4-Triton** only if the M=1 GEMV still lags the ~460 GB/s bf16 bandwidth after Path 1/2.
4. **Drop Path 3.**

FIRST EXPERIMENT (do this before writing any kernel -- it decides Path 1 vs Path 2 and validates the
whole HLO-dump assumption):

a. In `zml/module.zig` `.oneapi` arm (line 648), add
   `setXlaOverrideFlag(overrides_map, "xla_gpu_experimental_enable_subchannel_dequantisation_fusion",
   true, ...)`; keep `xla_gpu_enable_triton_gemm` at default (do not disable).
b. In `common_quant.zig:222` (`dequantWeight`) emit the single-convert canonical form
   `weight.convert(bf16).mul(weight_scale.convert(bf16).broad(...))` behind a `woq_fused` variant.
c. Compile the `//examples/w8a8_sweep` harness with `opts.xla_dump_to` set (wire it through, per
   `zml/module.zig:656`) and CONFIRM from the dumped optimized HLO whether the dequant `multiply`
   folded into the dot fusion (s8 param into a `__triton_gemm`/`fusion`, no standalone bf16 multiply).
d. Bench `woq_fused` vs `bf16`/`i8`/`woq` at q/o/gate/down, M=1,2,4,8; require byte-identical output.

Outcomes:
- Fusion fires AND >= bf16 bandwidth at M=1 -> adopt Path 1 for column-parallel/GDN; still build
  Path 2 for row-parallel/robustness.
- Fusion doesn't fire OR is slower -> proceed straight to Path 2 (and note the Triton-GEMM ceiling
  for Path 4-Triton).
Either way you will have PROVEN the HLO-dump + XLA-flag-override capability, which is the tool you
need to instrument every subsequent kernel change on the "blindfolded" prebuilt plugin.

--------------------------------------------------------------------------------------------------
## 6. Key file:line index

- Reference kernels: `kernels/int8_gemm_w8a16.h:20` (w8a16 op), `:87-114` (set_scales WEIGHTS +
  fpmath_mode f16/bf16), `:153-155` (execute); `kernels/int8_gemm_w8a8.h:18` (w8a8 op), `:76-112`
  (SRC per-token + WEIGHTS per-channel scales); `kernels/README.md:22-26` (sglang oneDNN-GPU build).
- zml int8 emit: `zml/tensor.zig:1418` (`dotAcc`), `:1353` (`dotGeneralAcc`), `:1399-1413`
  (dot_general s8 operands + s32 result + `.dot_precision=.fast`).
- zml quant module: `common_quant.zig:91` (weightOnly), `:121` (forward), `:133-137` (woq path),
  `:201-215` (forwardQuant s8xs8), `:222-226` (dequantWeight = materialize).
- zml custom-call / FFI: `zml/kernel.zig:96` (triton Kernel), `:367-388` (TPU custom_call, "tpu_custom_call");
  `zml/ops.zig:642-698` (triton -> `__gpu$xla.gpu.triton`), `:1508` (customCall), `:2014` (register),
  `:2050-2056` (handler), `:1997` (`sharding_aware`), `:2079` (shardingAwareTypedCustomCall);
  `zml/platform.zig:551` (registerFfi), `:552` (platform_name default), `:331-339` (zml$print FFI for
  all targets), `:319-327` (flashattn for cuda only); `pjrt/ffi.zig:84` (Api.stream / XLA_FFI_Stream_Get),
  `:101` (allocateDeviceMemory), `:257` (Buffer), `:274`/`:290` (Args/Rets); `pjrt/pjrt.zig:271`
  (ffi() extension), `:198-201` (extension union incl ffi/triton).
- Template (native GPU custom call): `zml/attention/flashattn.zig:128` (register),
  `:188` (`call_frame.api.stream(call_frame.ctx)` then native lib); Intel Triton in use:
  `zml/attention/triton_attention.zig:96`.
- Compile options / flags: `zml/module.zig:626` (env_option_overrides), `:648-652` (.oneapi flags),
  `:656-665` (xla_dump_to via compile opts).
- Plugin pin/load: `platforms/oneapi/oneapi.bzl:4-8`, `platforms/oneapi/libpjrt_oneapi.BUILD.bazel:4`,
  `platforms/oneapi/oneapi.zig:36`.
- Numbers: `zml/W8A8_SWEEP_RESULTS.md` (prefill 67-69% peak; M=1 int8 361 vs bf16 460 GB/s; woq dead
  end); memory `b70-int8-xmx-roofline` (367 TOPS / 608 GB/s; s32 mandatory; M=1 GEMV trap).

--------------------------------------------------------------------------------------------------
## 7. Research update (2026-07-01, deep-session): M=1 decode is BANDWIDTH/LAYOUT-bound, not XMX

A dedicated web+source research pass (Intel-XPU Triton, oneDNN release notes, XeTLA/ESIMD, llama.cpp
+ vLLM-XPU issues, arXiv) reframes the whole decode-kernel plan. The load-bearing conclusions are
CONFIRMED against primary sources (adversarially verified). This section SUPERSEDES the "make it hit
XMX" framing in sections 2-3 for the M=1 case.

### 7.1 The reframing (the single most important finding)
- **At M=1..8, weight-only int8 matmul is 30-300x below the compute roofline -- it is PURELY
  weight-bandwidth-bound.** The entire win over bf16 comes from reading FEWER WEIGHT BYTES (int8=2x,
  int4=4x) and hitting PEAK BANDWIDTH via a good weight MEMORY LAYOUT -- NOT from the systolic array.
  "Reaching DPAS at M=1" is the wrong goal: DPAS can be *issued* at M=1 (Intel Triton emits an rcount=1
  DPAS tile, no FMA fallback) but it does not help throughput there.
- Roofline crossover (B70, 367 TOPS / 608 GB/s = 603 ops/byte; int8 weight-only AI = 2*M ops/byte):
  int8 becomes compute-bound only near **M~=300**, int4 near **M~=150**. So M=1..8 (decode + MTP
  verify) is DEEP in the bandwidth-bound regime.
- **On-THIS-box proof:** llama.cpp issue #21517 (created 2026-04-06, Qwen3.5-27B dense, a B70):
  Q8_0 decode = 4.88 t/s (130 GB/s = **21% of 608**) vs Q4_K_M = 20.56 t/s (53%). Root cause was
  KERNEL/LAYOUT DISPATCH (Q8_0 fell to the generic DMMV path: 2 values/thread, stride 64), NOT XMX.
  Fix PR #21527 (merged 2026-04-07, `dequantize_q8_0_reorder`) -> **15.24 t/s = 3.1x, 21%->66% BW, via
  a WEIGHT LAYOUT REORDER ALONE, still no XMX.** Same regression on Vulkan; absent on Xe1 A770 => it is
  a Xe2-specific int8 kernel/layout gap. THIS is the existence proof that the decode win is a
  layout-optimized weight load, and it was measured on our exact GPU + model.

### 7.2 Consequence for the four paths (revised verdicts)
- **PATH 1 (subchannel-dequant fusion flag)** -- unchanged: still the cheapest probe. CODE LANDED this
  session (env-gated `ZML_SUBCHANNEL_FUSION` in `module.zig` .oneapi arm, flag string verified against
  the plugin .so; HLO dump via `ZML_DUMP_HLO` in the sweep). GPU A/B pending. Its ceiling is the same
  Intel-Triton GEMM as Path 4-Triton, and per 7.1 a Triton `tl.dot(s8,s8)` stays bandwidth-bound at
  M=1 -- so even if it fires it may not beat a layout-optimized load. Run it, but do not expect a
  ceiling result.
- **PATH 2 (oneDNN-SYCL w8a16 via FFI)** -- STILL WORTH DOING, but with a corrected expectation: the
  oneDNN GPU matmul dispatches to `jit:gemm:any` (GemmStone) which has **NO dedicated GPU GEMV kernel**;
  at M=1 it picks a small-unroll/FMA strategy and likely leaves BW on the table. HOWEVER at **M=2..8 it
  is a real tiled GEMM that reads the weight ONCE** -- so it should deliver the MTP-verify amortization
  (7.4) even if it under-serves pure M=1 decode. Recommended as the low-effort BASELINE: benchmark it
  at M=1,2,4,8 first (confirm native s8s8 vs u8s8+compensation via `ONEDNN_VERBOSE=dispatch`), then
  decide whether the layout-first kernel (7.3) is needed for M=1. oneDNN GPU weight-decompression is
  v3.5+ (int8/int4 weights), int8-activations-with-grouped-scales is v3.6+, Battlemage perf in v3.7.
  The bundled plugin oneDNN is CPU-only, so link our own DNNL_GPU_RUNTIME=SYCL build -- OR reuse the
  already-built sglang GPU oneDNN in `_xpu_C.abi3.so` / `vllm-xpu-kernels-w8a8/csrc/xpu/onednn/`
  (kernels-mining agent confirmed it exists and is dlopen-able, like flashattn dlopens its lib).
- **PATH 4 (hand kernel) -- PROMOTED to the real decode ceiling.** The layout-first weight-only kernel
  (7.3) is THE lever, and a fused ESIMD/XeTLA kernel is the ceiling. Two concrete reference
  implementations to study/port:
  1. **llama.cpp `dequantize_q8_0_reorder` (PR #21527)** -- the minimal proven 3.1x: reorder the int8
     weight for coalesced Xe2 loads (16 values/thread, stride 512), dequant-to-f16 + vector-ALU, no XMX.
  2. **Intel arXiv:2508.06753 "Pushing the Envelope of LLM Inference on ... Intel GPUs"** (v2 Jan 2026)
     -- fused ESIMD/XeTLA kernels with an autotuner, weights offline in **VNNI16**, activation
     quant/dequant fused INSIDE the GEMM, shipped as a **vLLM quantization plugin**. Measured GEMV
     within 10% of the int8 roofline (380 of 456 GB/s on B580), 6.3x end-to-end vs bf16. This is the
     "hand-write the kernel" endgame recipe.
  - Xe2 DPAS int8 shape (verified from the SYCL joint_matrix spec): **s8xs8->s32 native, M<=8, N=16,
    K=32** (N=8 on older Alchemist). Variable rcount=M 1..8 with NO M-padding -- unlike NVIDIA's fixed
    16x16 fragments -- which is exactly why batching MTP tokens (7.4) is free on Xe2.

### 7.3 The recommended decode-kernel lever (revised): layout-first weight-only w8a16
int8 weight + bf16 activation + per-channel weight dequant in the epilogue, with the WEIGHT
PRE-SHUFFLED OFFLINE (VNNI16 / reorder) so the M=1 load is fully coalesced at ~peak BW. This is the
llama.cpp-proven 3.1x on this box and "the ~2x-over-bf16 that is actually on the table." Whether it
emits DPAS(rcount=1) or an FMA reduction is second-order -- both are bandwidth-bound; pick whichever
the compiler coalesces best. Secondary M=1 occupancy lever: Split-K (4-8) with an i32 partial-sum
reduce (a single GEMV tile under-occupies the 32 Xe cores). Keep symmetric zero-point-free per-channel
int8 (Xe2 s8xs8 is native, so the old "+128 tax" is the asymmetric zero-point compensation + act-quant
prologue, NOT signedness -- our scheme is already symmetric, good).

### 7.4 The MTP amortization is a batched-M GEMM (structural, may not even need a new kernel)
The measured "verify(s=Kv=2) ~= 2x decode(s=1), nothing amortizes" is because the current small-M int8
path is GEMV-per-token that RE-READS the weight M times. The fix: run the projection as ONE GEMM over
the M=Kv verify tokens so the weight is read ONCE for all M (Xe2 rcount absorbs M=1..8 with no padding
waste). zml's `dotAcc` over M=Kv is already a single dot, but it dispatches to the inefficient small-M
kernel; a proper w8a16 GEMM (Path 2 at M=2..8, or the 7.3 kernel) makes MTP amortize "for free." This
is the biggest structural MTP win and it is orthogonal to the M=1 decode problem.

### 7.5 P2P / TP collectives -- verdict (from the kernels/patches mining + ZML_TP_ALLREDUCE.md)
- DO NOT port the vLLM push-all-reduce kernel. zml's TP all_reduce is an in-graph StableHLO
  `all_reduce` (`zml/ops.zig:142`) that Shardy/SPMD schedules and OVERLAPS with the matmul -- a
  PJRT custom-call would be opaque to SPMD and lose that. The push-AR host-barrier/graph-break problem
  simply does not exist in zml.
- int8/int32-partial reduce does NOT help the row-parallel o_proj/down_proj: the contracting axis is
  `.model`-SHARDED with PER-SHARD act-scales, so reducing int32 partials before the per-shard scale is
  WRONG. The bf16 dequantized `add`-reducer all_reduce Shardy already emits is the correct minimal
  collective. (Verify on GPU: dump StableHLO, confirm down_proj has an `add`-reducer all_reduce and NO
  `maximum`-reducer.)
- The ONE transferable P2P lever: an attended `CCL_TOPO_P2P_ACCESS=1` A/B (zml single-process PJRT is
  the H.12 topology that worked with P2P=1). Expect a PREFILL/TTFT win, not decode. Respect the wedge
  constraints: export `CCL_TOPO_P2P_ACCESS` explicitly (oneapi.zig:33 garbage-defaults it), oneCCL
  2022 bundled = same wedge risk, reboot-only recovery, run attended.

### 7.6 Revised execution order (int8 kernel track)
1. Path 1 GPU A/B (coded) -- cheap, informs the Triton ceiling. Batch into the next attended run.
2. Path 2 oneDNN w8a16 via FFI as the BASELINE -- benchmark M=1,2,4,8; expect it to unlock MTP
   (M=2..8) even if it under-serves M=1. Reuse the sglang GPU-oneDNN .so to skip a 14-min build.
3. If M=1 still lags peak BW: the 7.3 layout-first weight-only kernel (port llama.cpp #21527's reorder;
   then the arXiv:2508.06753 fused ESIMD/XeTLA recipe as the ceiling).
4. In parallel (independent of the kernel): the chunked GDN scan (C=32; ZML_GDN_OPT.md) for prefill +
   to remove the verify's sequential-scan dependency so 7.4 can amortize.
