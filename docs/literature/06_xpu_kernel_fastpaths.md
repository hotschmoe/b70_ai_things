# 06 - Writing Low-Level Kernels for Intel Arc Pro B70 (Battlemage / Xe2): Fast-Path Surfaces and Ranked Contribution Targets

**Research snapshot: 2026-06-18.**
**Scope:** A literature review for a small team that wants to WRITE/FIX its own low-level GPU kernels for the Intel Arc Pro B70 (Xe2 "Battlemage", BMG-G31, 32 GB) to close concrete gaps in vLLM-XPU and llama.cpp. Covers the compute microarchitecture (DPAS/XMX), the programming surfaces (SYCL joint_matrix, ESIMD, oneDNN, Triton-XPU, oneMKL, sycl-tla), the vLLM-XPU and vllm-xpu-kernels extension seams, the xe/Level-Zero runtime stack, and a ranked list of kernels to write first.
**Builds on:** [`05_w8a8_recipe.md`](./05_w8a8_recipe.md) (the W8A8/FP8 dispatch gap) and [`06_vllm_latest_xpu.md`](./06_vllm_latest_xpu.md) (no XPU cudagraph capture; FP8 is the working 8-bit path). This doc is the kernel-authoring companion to those.

**Confidence markers used throughout:** `[WELL-SOURCED]` = 2+ primary sources or a direct spec/source quote, usually cross-confirmed; `[SINGLE-SOURCE]` = one primary source; `[INFERRED]` = derived from a formula/enum/design, not a verbatim statement; `[UNVERIFIED]` = could not confirm, flagged honestly.

> **Premise corrections up front (this project's prior notes need three fixes).** The deep research overturned three working assumptions:
> 1. **The "Xe2 warptile bug" in llama.cpp does not exist as stated.** There is no `warptile`/`mmq` knob in the SYCL backend (those are CUDA-backend terms). The real Battlemage correctness bug is a reorder/DPAS-sync corruption (llama.cpp #21893), worked around with `GGML_SYCL_DISABLE_OPT=1`. [WELL-SOURCED]
> 2. **llama.cpp DOES now have Gated-DeltaNet.** PR #16095 (~Nov 2025) added `GGML_OP_GATED_DELTA_NET` + `GGML_OP_CUMSUM`, and `docs/ops.md` marks `GATED_DELTA_NET` as supported on SYCL. Qwen3-Next / Qwen3.5/3.6 run on it (CPU-first, correctness-focused). The "build 9680 segfaults" report does not match a locatable build; the real DeltaNet crashes are #19728 (Metal assert at `delta-net-base.cpp:316`) and #17586 (ROCm). [WELL-SOURCED]
> 3. **`torch.xpu.XPUGraph` now exists upstream** (PyTorch 2.11, PR #174046, backed by SYCL-Graph over Level-Zero command lists). The "no CUDA-graph-equivalent on XPU" statement is now only true *in vLLM*, which has not yet wired its XPU path to the new API. [WELL-SOURCED]
> Two more nuances corrected inline: (a) llama.cpp Vulkan DOES use `VK_KHR_cooperative_matrix` on Xe2 (so "matrix cores: none" is the pre-Xe2 / misdetection case, not B70); (b) on the B70 specifically, SYCL beats Vulkan for decode (the gap is SYCL's MMVQ+reorder path, not matrix cores).

---

## A. Battlemage Xe2 Compute Microarchitecture for Kernel Authors

### A.1 Xe-core and engine counts

| Item | Value | Confidence |
|---|---|---|
| Vector engines (XVE) per Xe-core | 8 (8 x SIMD16 = 128 FP32 lanes/core) | [WELL-SOURCED] |
| XMX (DPAS/matrix) engines per Xe-core | 8 (each 2048-bit wide) | [WELL-SOURCED] |
| Native SIMD width | **SIMD16** (Xe2 dropped Alchemist's SIMD8 preference) | [WELL-SOURCED] |
| Native/preferred sub-group size | **16** (32 also valid) | [WELL-SOURCED] |
| Xe-cores on BMG-G31 (Arc Pro B70) | **32** (8 render slices x 4) | [WELL-SOURCED] |
| Total XMX engines on B70 | **256** (32 x 8) | [WELL-SOURCED] |
| Xe-cores on B580 (BMG-G21, consumer) | 20 | [WELL-SOURCED] |

The **single most kernel-relevant Xe1->Xe2 change is SIMD8 -> SIMD16** (native sub-group 8 -> 16; DPAS N went 8 -> 16). Code tuned for Alchemist's SIMD8 / N=8 can misbehave on Battlemage; this is plausibly related to the llama.cpp reorder corruption (#21893). [WELL-SOURCED]

### A.2 The DPAS instruction (the systolic primitive every XMX kernel maps onto)

DPAS = Dot Product Accumulate Systolic. It computes `D = C + A x B` per instruction with shape **M x K x N**:
- **M = repeat count (RC)**, 1..8.
- **N = execution size = 16** on Xe2 (was 8 on DG2/Alchemist).
- **K = SystolicDepth x OPS_PER_CHAN**, with SystolicDepth fixed at **8** on all XEHP+/Xe2 hardware.

From the IGC vISA `DPAS.md` spec (primary, fetched directly): `[WELL-SOURCED]`

| Input dtype | Encoding | OPS_PER_CHAN | K (SD=8) | Full tile (M=8, N=16) | Accumulator |
|---|---|---|---|---|---|
| TF32 | tf32 | 1 | 8 | 8 x 8 x 16 | FP32 |
| FP16 (hf) | hf | 2 | 16 | 8 x 16 x 16 | FP32 |
| BF16 (bf) | bf | 2 | 16 | 8 x 16 x 16 | FP32 |
| **INT8 (s8/u8)** | u8/s8 | 4 | **32** | **8 x 32 x 16** | **INT32** |
| **FP8 (E4M3=hf8 / E5M2=bf8)** | bf8/hf8 | 4 | 32 | 8 x 32 x 16 | FP32 |
| **INT4 (s4/u4)** | u4/s4 | 8 | **64** | **8 x 64 x 16** | **INT32** |
| INT2 (s2/u2) | u2/s2 | 8 | 64 | 8 x 64 x 16 | INT32 |

**B matrix must be VNNI-packed:** elements along the contraction (K) dim are packed vertically into 32-bit channels (4 int8/dword, 8 int4/dword). This is a hard requirement for feeding DPAS; weight pre-processing kernels must produce this layout. [WELL-SOURCED]

**INT8/INT4 GEMM mapping:** inputs s8/u8 (or s4/u4), accumulator **int32**, then dequantize the int32 to float post-accumulation (multiply by per-channel weight scale x per-token activation scale). The XeTLA arxiv paper (arXiv:2508.06753) does exactly this: quantize BF16 activations to int8 in-register, run int8 DPAS, accumulate int32, dequantize the int32 to BF16 before 2D-store. [WELL-SOURCED]

**Throughput per Xe-core/clock:** ~2048 FP16 ops, ~4096 INT8 ops (INT8 = 2x FP16). INT4 ~2x INT8 again (K doubles). The B70's ~367 INT8 TOPS reflects this. [WELL-SOURCED for the per-core rates; INT4:INT8 = 2x is INFERRED from K-doubling]

### A.3 The FP8-on-DPAS caveat (important for this project)

The IGC vISA spec **encodes** bf8/hf8 (FP8) as valid DPAS precisions, but Battlemage architecture briefs list only INT2/INT4/INT8/FP16/BF16 for the XMX units, and oneDNN's data-type table marks FP8 on Xe2 as **supported via up-conversion (`.`), not native systolic (`+`)**. Triton-XPU's DPAS lowering enumerates native FP8 DPAS engine types only for the next-gen **Xe3P** tier, not base Xe2. [CONFLICTING at ISA vs implementation level]

**Conclusion:** On B70, treat **FP8 matmul as conversion-based (up-converted to bf16/fp16 then run on the FP16 systolic path), NOT as a native FP8 systolic op.** INT8 and INT4, by contrast, are genuinely native integer DPAS on Battlemage. This is a meaningful asymmetry: an INT8 W8A8 kernel hits the systolic array directly at int8; the existing FP8 kernel does not get a native-FP8 systolic speedup, it rides the FP16 datapath. [WELL-SOURCED at the oneDNN/Triton implementation level; the "FP8 rides FP16 path" framing is INFERRED-but-corroborated]

### A.4 Confirmed B70 headline numbers

All confirmed from launch coverage (ServeTheHome, TheFPSReview, TechPowerUp) and arch analysis: 367 INT8 TOPS, ~22.9 FP32 TFLOPS (= 4096 x 2 x 2.8 GHz), 256 XMX engines, 32 Xe-cores, 608 GB/s (256-bit @ 19 Gbps GDDR6), 32 GB GDDR6 ECC, 2.8 GHz boost, PCIe 5.0 x16, BMG-G31. [WELL-SOURCED; FP32 TFLOPS is SINGLE-SOURCE math-validated]

---

## B. Programming Surfaces for Writing B70 Kernels (tradeoffs and when to use each)

### B.1 SYCL `joint_matrix` (portable XMX/DPAS)

- **Header:** `sycl/ext/oneapi/matrix/matrix.hpp`. **Namespace:** `sycl::ext::oneapi::experimental::matrix`.
- **Types/ops:** `joint_matrix<Group, T, use, Rows, Cols, layout>`; `use::a / use::b / use::accumulator`; `joint_matrix_load / _store / _mad / _fill / _apply`. `joint_matrix_mad` is the op that lowers to DPAS. A/B need explicit row/col-major; C/D must be `layout::dynamic`. [WELL-SOURCED]
- **Battlemage IS supported** (`intel_gpu_bmg_g21`, `bmg_g31` in the spec tables) on the N=16 (PVC-class) path. Query combinations at runtime via `dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>()`; branch on `nsize` (16 on BMG). [WELL-SOURCED]
- **Queryable Intel combinations** (verbatim from the spec + `static-query-use.hpp`):

  | A/B type | C/D | M | N | K |
  |---|---|---|---|---|
  | sint8/uint8 (all 4 sign combos) | sint32 | <=8 | 16 | 32 |
  | fp16 | fp32 | 16/1/32 | 16/64/32 | 16 |
  | bf16 | fp32 | 16/1/32 | 16/64/32 | 16 |
  | tf32 | fp32 | <=8 | 16 | 8 |

- **CRITICAL LIMITATION:** portable `joint_matrix` exposes **only int8, fp16, bf16, tf32** on Intel. There is **no `precision::s4`/`u4`, no int2, no FP8** row in any Intel `matrix_combinations`. The DPAS *hardware* does int4/int2; the portable API does not surface it. [WELL-SOURCED - confirmed by direct read of the asciidoc and the header]
- **When to use:** an **INT8 W8A8** or fp16/bf16 GEMM where portability and compiler-managed register allocation are wanted. This is the cleanest surface for the INT8 W8A8 kernel this project needs.
- **When NOT to use:** anything int4/int2/fp8 (drop to sycl-tla); anything needing exact LSC/cache/fence control (drop to ESIMD). Note llama.cpp's SYCL maintainers explicitly **declined** joint_matrix ("not a good fit due to data layout") and plan direct DPAS - so for llama.cpp specifically this is not the house style. [WELL-SOURCED]

### B.2 ESIMD (Explicit SIMD) - for custom collectives and hand-tuned kernels

- **Header:** `sycl/ext/intel/esimd.hpp`. **Namespace:** `sycl::ext::intel::esimd`. Sub-group forced to 1; you vectorize explicitly with `simd<T,N>` over the register file. [WELL-SOURCED]
- **Loads/stores:** `block_load/block_store` (contiguous), `gather/scatter`, `load_2d/store_2d/prefetch_2d` (2D block messages, ideal for GEMM tiles), plus legacy `lsc_*` forms. [WELL-SOURCED]
- **Cache hints:** `cache_hint` enum = `uncached, cached, streaming, read_invalidate, write_through, write_back`. Per-level via property list (`cache_hint_L1<...>, cache_hint_L2<...>`). **L1-uncached reads** are the primitive for peer reads that must bypass a stale local cache line. [WELL-SOURCED]
- **Fences:** `fence<memory_kind, fence_flush_op, fence_scope>()`. `fence_scope` = `group, local, tile, gpu, gpus, system, system_acquire`. The verbatim spec rationale for `system_acquire`: *"for GPUs that do not follow PCIe Write ordering for downstream writes targeting device memory, this op will commit to device memory all downstream and peer writes that have reached the device."* This is the cross-device acquire primitive. [WELL-SOURCED]
- **Atomics:** `atomic_update<atomic_op::...>` (add/sub/inc/cmpxchg/...). Can target peer memory - but reliability of *remote* PCIe atomics is the crux below.
- **When ESIMD beats plain SYCL:** the capture-safe / PCIe-only collective case. Shipping proof is **oneCCL's PCIe low-latency ("LL") protocol**, gated for Arc-B (`CCL_ENABLE_ARCB` -> `CCL_SYCL_ENABLE_ARCB`): it uses **L1-uncached reads (`L1UC_L3C`) + an in-band sequence-number/flag handshake, with NO remote atomics** (`proto_rt64.hpp`, `ring_transmit.hpp`, `allreduce_pcie.cpp`). This is exactly the local-write/remote-read pattern. [WELL-SOURCED]

> **Premise refinement on "PCIe D2D atomics unreliable -> local-write/remote-read":** This is accurate **for the PCIe path Battlemage uses** (oneCCL's ARCB LL protocol avoids remote atomics, uses uncached reads + flag polling). But oneCCL's *other* path - the Xe-Link "topo" ESIMD kernel for PVC/Max - deliberately DOES use remote atomics (`lsc_atomic_update<atomic_op::add>`). So the two strategies coexist; the no-atomics/uncached one is specifically the **PCIe-without-fabric** path that B70 is forced onto. A flat "Intel consumer PCIe atomics are unreliable" Intel statement was **not** found; the evidence is the `system_acquire` spec wording + oneCCL's separate non-atomic PCIe path + the dual-B70 crash report. [INFERRED, strongly corroborated]

### B.3 oneDNN (the matmul library that vLLM-XPU/IPEX actually call)

- **ukernel/brgemm is CPU-ONLY** - "the oneDNN micro-kernel API ... is a low-level, sequential abstraction for CPU only." **Not an option for the B70 GPU.** GPU GEMM goes through the standard `matmul` primitive (SYCL/OpenCL engine). [WELL-SOURCED]
- **oneDNN GPU matmul dtype support on Xe2-HPG ("Intel Arc B-Series Graphics"):** `[WELL-SOURCED]`

  | Path | Status on Battlemage |
  |---|---|
  | **s8/u8 x s8 -> s32 (INT8 W8A8)** | **Hardware-native (`+`) on XMX** |
  | bf16 / fp16 | Native (`+`) |
  | FP8 (f8_e4m3/e5m2) | Supported via conversion (`.`), not native systolic |
  | int4 weight + fp16/bf16 act (W4A16) | Supported (weight-only quant, `.`) |
  | **int4 weight + int8 act (W4A8)** | **Supported on Intel GPU** (oneDNN v3.6: "int8 activations with grouped scales and int8 or int4 compressed weights ... implemented on Intel GPUs") |

  Integer accumulation rule (verbatim): *"if src, weights ... are integral datatypes (s8, u8, s32), then the Op outputs s32 elements."* So **oneDNN already has everything needed for an INT8 W8A8 GPU matmul** - the primitive accepts s8 x s8 -> s32 natively. [WELL-SOURCED]
- **How vLLM/IPEX call it:** all real XPU matmuls are `torch.ops._xpu_C.*` ops compiled in the out-of-tree `vllm-xpu-kernels` repo (`csrc/xpu/onednn/*.h` -> oneDNN `dnnl::matmul`). IPEX (`csrc/gpu/oneDNN/`) already has a true W8A8 int8 GPU path via `QMatmul.h::quantized_matmul`. [WELL-SOURCED]
- **Adding an INT8xINT8 path is wrapper plumbing,** not new math: oneDNN does the systolic work. Open risk: whether oneDNN's Intel-GPU engine ships an *optimized* s8xs8 kernel vs a reference fallback for the target shapes - **verify empirically**. [oneDNN capability WELL-SOURCED; optimization-level UNVERIFIED]

### B.4 Triton-XPU (intel-xpu-backend-for-triton)

- **Maturity:** actively maintained, latest v3.7.1 (June 2026), explicitly added "Intel G31 GPU support" (Big Battlemage). Arc Pro B-Series listed as supported hardware. [WELL-SOURCED]
- **`tl.dot` lowers to XMX/DPAS** (dedicated MMA layout). The DPAS lowering enumerates engine types **`S32_S32_S8_S8` and `U32_U32_U8_U8`** in the Xe2 set - so an **int8 `tl.dot` with int32 accumulator maps to XMX int8 on Battlemage.** You *can* write an INT8 W8A8 matmul in Triton-XPU that hits the systolic array. [WELL-SOURCED for the enum; a ready-made int8 example was not found by name]
- **2D block load** is the default operand-feed path (`tt.make_tensor_ptr`, >2x over tensor-of-pointers). `tl.dot_scaled` + FP8 supported, but native FP8 DPAS is Xe3P-gated; on Xe2 FP8 dot_scaled is upcast-emulated. [WELL-SOURCED enum; Xe2-upcast INFERRED]
- **Known Battlemage bugs:** the gated-delta-rule recurrent kernel `UR_RESULT_ERROR_DEVICE_LOST` (#6658, see Section E.3); B580 PassManager/OOM/NaN (#4838). Stability on Xe2 needs validation. [WELL-SOURCED]

### B.5 oneMKL and sycl-tla (CUTLASS for Intel)

- **oneMKL GPU INT8 GEMM: YES**, named `gemm_bias` in DPC++ (not `gemm_s8u8s32`): (s8/u8) x (s8/u8) -> **s32** with integer offsets, wired to the Intel GPU (`mklgpu`) backend. The GPU analogue of cuBLASLt int8. **oneMKL has NO FP8 GEMM** (FP8 lives in oneDNN/sycl-tla). [WELL-SOURCED; one per-page "GPU supported" line UNVERIFIED but carried by open-repo wiring]
- **sycl-tla = `github.com/intel/sycl-tla`** ("CUTLASS for Intel GPUs", formerly Codeplay CUTLASS-SYCL). Header-only CUTLASS+CuTe port. **Targets Battlemage** (`intel_gpu_bmg_g21/g31`, validated on Arc B580). Dtypes (verbatim): *"FP16, BF16, 8b floating point (E5M2/E4M3 FP8), narrow integer types (4 and 8b signed/unsigned with zero-point quantization)."* So it covers **int8, int4, fp8** with zero-point - the only surface that exposes int4/fp8 in a CUTLASS-like template form. Ships `06_bmg_flash_attention`, `08_bmg_gemm_f8`, mixed-dtype GEMM examples. **vLLM-XPU's flash-attention is built on sycl-tla** (pulled via CMake FetchContent into vllm-xpu-kernels). Pre-1.0 (v0.9.1, June 2026). [WELL-SOURCED; a dedicated INT8 example was not found by name]

### B.6 Surface selection cheat-sheet

| You want to write... | Best surface | Why |
|---|---|---|
| INT8 W8A8 GEMM (portable, lowest effort) | **oneDNN matmul** (s8xs8->s32 native) or `joint_matrix` | hardware-native int8; oneDNN already wired into vLLM-XPU |
| INT8 W8A8 in Triton | Triton-XPU `tl.dot` int8 | lowers to `S32_S32_S8_S8` DPAS; but validate Xe2 stability |
| INT4 / FP8 custom GEMM | **sycl-tla** | only surface exposing s4/u4/e4m3/e5m2 |
| Capture-safe all-reduce / custom collective | **ESIMD** | L1-uncached peer reads + `system_acquire` fence + flag handshake |
| Flash-attention variants | sycl-tla (`06_bmg_flash_attention`) | what vLLM-XPU already uses |

---

## C. vLLM-XPU and vllm-xpu-kernels Structure (the exact extension seams)

### C.1 Where the registry and selector live

The selector, the `_POSSIBLE_*_KERNELS` dicts, and the `init_*_linear_kernel` helpers all live in the **package file** `vllm/model_executor/kernels/linear/__init__.py` (NOT in `scaled_mm/`). The file docstring states the contract: *"If you are adding a new kernel selector or kernel implementation, add it to this `__init__.py` to maintain import stability."* [WELL-SOURCED - corrects the prior note that pointed at `scaled_mm/`]

The KeyError seam (your prior finding, confirmed verbatim):

```python
def choose_scaled_mm_linear_kernel(config, possible_kernels, compute_capability=None, force_kernel=None):
    ...
    platform_kernels = possible_kernels[current_platform._enum]   # <-- bare subscript, no try/except
    ...
    for kernel in platform_kernels:
        ok, reason = is_supported_and_can_implement_kernel(kernel, config, compute_capability)
        if ok: return kernel
    raise ValueError("Failed to find a kernel that can implement the ScaledMM linear layer...")
```

On XPU, `_POSSIBLE_INT8_KERNELS` has no `PlatformEnum.XPU` key, so `possible_kernels[current_platform._enum]` raises **`KeyError(PlatformEnum.XPU)`** *before* the loop - the advertised `ValueError` is never reached. [WELL-SOURCED]

### C.2 The registry contents (XPU wired everywhere except int8)

```python
_POSSIBLE_INT8_KERNELS = {
    PlatformEnum.CPU:  [ZentorchInt8..., CPUInt8...],
    PlatformEnum.CUDA: [CutlassInt8..., TritonInt8...],
    PlatformEnum.ROCM: [AiterInt8..., TritonInt8...],
}   # NO PlatformEnum.XPU  <-- the gap

_POSSIBLE_FP8_KERNELS[PlatformEnum.XPU]       = [XPUFP8ScaledMMLinearKernel]
_POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.XPU] = [XPUFp8BlockScaledMMKernel, TritonFp8BlockScaledMMKernel]
_POSSIBLE_WFP8A16_KERNELS[PlatformEnum.XPU]   = [XPUFP8ScaledMMLinearKernel]
_POSSIBLE_KERNELS[PlatformEnum.XPU]           = [XPUW4A8IntLinearKernel, XPUwNa16LinearKernel]
```

FP8 works on XPU because `_POSSIBLE_FP8_KERNELS` has the XPU entry; INT8 raises KeyError because `_POSSIBLE_INT8_KERNELS` does not. The gap is INT8-specific. [WELL-SOURCED]

**There is also a public programmatic seam** (new finding) - an out-of-tree plugin can avoid editing the dict literal:
```python
register_linear_kernel(XPUInt8ScaledMMLinearKernel, PlatformEnum.XPU, "int8")
```
(`register_linear_kernel(kernel_class, platform, kernel_type)`, `kernel_type in {"mp","int8","fp8","mxfp8","nvfp4","mxfp4"}`, exported in `__all__`.) [WELL-SOURCED]

### C.3 The interface a new XPU INT8 kernel must implement

From `scaled_mm/ScaledMMLinearKernel.py`:
```python
@dataclass
class Int8ScaledMMLinearLayerConfig(MMLinearLayerConfig):
    is_static_input_scheme: bool
    is_channelwise: bool
    input_symmetric: bool

class ScaledMMLinearKernel(Generic[_ConfigT, _ParamsT], ABC):
    @classmethod @abstractmethod
    def is_supported(cls, compute_capability=None) -> tuple[bool, str | None]: ...
    @classmethod @abstractmethod
    def can_implement(cls, c) -> tuple[bool, str | None]: ...
    @abstractmethod
    def process_weights_after_loading(self, layer) -> None: ...
    @abstractmethod
    def apply_weights(self, layer, x, bias=None) -> torch.Tensor: ...
```
`Int8ScaledMMLinearKernel` adds `_get_layer_params` returning the 5-tuple `(weight, weight_scale, input_scale, input_zero_point, azp_adj)`. **Reference templates to copy:** `CutlassInt8ScaledMMLinearKernel` (weight-processing logic: transpose, per-tensor->per-channel scale, azp_adj) and `XPUFP8ScaledMMLinearKernel` (the SYCL op-call pattern: `return torch.ops._xpu_C.fp8_gemm_w8a16(...)`). [WELL-SOURCED]

### C.4 The call chain (how compressed_tensors W8A8-INT8 reaches the selector)

`CompressedTensorsW8A8Int8.create_weights` -> `init_int8_linear_kernel(is_channelwise, is_static_input_scheme, input_symmetric, ...)` -> `choose_scaled_mm_linear_kernel(config, _POSSIBLE_INT8_KERNELS)` -> KeyError on XPU. `process_weights_after_loading`/`apply_weights` just delegate to the chosen kernel. [WELL-SOURCED]

### C.5 vllm-xpu-kernels (out-of-tree native repo)

- `github.com/vllm-project/vllm-xpu-kernels`: SYCL/DPC++ + oneDNN native kernels. Layout: `csrc/` (kernels), `csrc/xpu/onednn/` (GEMM headers), `csrc/xpu/gdn_attn/` (DeltaNet), `cmake/`, `vllm_xpu_kernels/` (bindings), `tests/`, `benchmark/`. Importing `vllm_xpu_kernels._C` auto-registers all custom ops with the PyTorch dispatcher. [WELL-SOURCED]
- **Build:** `requires-python >=3.9,<3.14`; hard-pins `torch==2.12.0+xpu`; oneAPI 2025.3 (`icx`/`icpx`), C++20, CMake >=3.26. Wheels are `cp38-abi3` from v0.1.3 on (the "Python 3.12-only" memory is only true for v0.1.0-v0.1.2). Latest release v0.1.10 (June 2026); v0.1.9 (2026-05-29) is the one with "Support Xe2 MTP for QWEN." [WELL-SOURCED - corrects the "3.12-only" note]
- **Ops registered** (`csrc/xpu/torch_bindings.cpp`, namespace `xpu_ops` -> `_xpu_C`): `fp8_gemm`, `fp8_gemm_w8a16`, `fp4_gemm`, `int4_gemm_w4a16`, `int4_gemm_w4a8`, `gdn_attention`, `cutlass_grouped_gemm_interface`, rope/LoRA/sampler ops. oneDNN headers in `csrc/xpu/onednn/`: `fp8_gemm_w8a16.h`, `fp8_gemm_w8a8.h` (this is **FP8**, not int8), `fp4_gemm_w4a4.h`, `int4_gemm_w4a16.h`, `int4_gemm_w4a8.h`, `onednn_ext.h`. [WELL-SOURCED]
- **CRITICAL GAP:** there is **NO integer s8xs8 / W8A8-INT8 GEMM op.** The `joint_dtypes_t` enum in `onednn_ext.h` has only int4/fp8/mxfp4 weight combos - no `s8_s8`/`s8_u8`. `int4_gemm_w4a8` is int8-activation x int4-weight, not int8xint8. [WELL-SOURCED]
- **This IS the right place to add an INT8 W8A8 kernel** (two-repo work, Section E).

### C.6 Gated-DeltaNet (GDN) XPU kernels + the projection break

- **Attention backend/metadata:** `vllm/v1/attention/backends/gdn_attn.py` (`GDNAttentionBackend`). **Compute layer + dispatch:** `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (`QwenGatedDeltaNetAttention`; XPU -> `forward_xpu` -> `torch.ops.vllm.gdn_attention_core_xpu`). **Native SYCL kernel:** `csrc/xpu/gdn_attn/` in vllm-xpu-kernels (`gated_delta_rule.hpp`, `causal_conv1d.hpp`, `xe_2/`), registering `torch.ops._xpu_C.gdn_attention`. [WELL-SOURCED]
- The SYCL `gdn_attention` op covers **only conv1d + the recurrent core**; its docstring: *"input/output projections are performed by the caller."* Those projections (`in_proj_qkvz`, `in_proj_ba`, `out_proj`) are ordinary ParallelLinear layers.
- **Why the KeyError breaks GDN:** when a Qwen3-Next/GDN model is W8A8-INT8 quantized, each projection gets `CompressedTensorsW8A8Int8` -> `init_int8_linear_kernel` -> `choose_scaled_mm_linear_kernel(_POSSIBLE_INT8_KERNELS)` -> **KeyError on XPU**, killing model load before the SYCL GDN core ever runs. FP8 projections succeed only because `_POSSIBLE_FP8_KERNELS` has the XPU entry. **So fixing the INT8 registry gap (target #2) directly unblocks the GDN projection path.** [WELL-SOURCED architecture; the precise int8-quantized-projection chain is INFERRED-but-corroborated]
- **PRs:** #43565 (XPU GDN-attention MTP, needs companion vllm-xpu-kernels#368); #43534 is **CPU**, not XPU. The Triton/FLA fallback (`vllm/model_executor/layers/fla/ops/`) is reachable on XPU but hits the device-lost bug (#6658), which is why the SYCL kernel exists. [WELL-SOURCED]

### C.7 Exact files to patch for the two registry targets

Both seams are in `vllm/model_executor/kernels/linear/__init__.py`:
1. **(target #1) Add `_POSSIBLE_INT8_KERNELS[PlatformEnum.XPU] = [XPUInt8ScaledMMLinearKernel]`** + a top-of-file import; the kernel class goes in `scaled_mm/xpu.py`. (Or call `register_linear_kernel(...)` from a plugin.)
2. **(target #2) Harden `choose_scaled_mm_linear_kernel`:** change `possible_kernels[current_platform._enum]` to `possible_kernels.get(current_platform._enum, [])` so a missing platform yields the clean `ValueError` instead of an unhandled `KeyError`. Strictly, once #1 adds the key, the function already works - this is defensive. [WELL-SOURCED]
3. **Native op:** add `csrc/xpu/onednn/int8_gemm_w8a8.h` (modeled on `int4_gemm_w4a8.h`, weights s8, per-channel s8 scales) + `joint_dtypes_t::s8_s8` in `onednn_ext.h` + `ops.def/impl("int8_gemm_w8a8")` in `csrc/xpu/torch_bindings.cpp`. (First grep `onednn_ext.h` for an existing s8 helper.) [structure WELL-SOURCED; design INFERRED from int4 template]

---

## D. Driver / Runtime Stack

### D.1 The `xe` driver

Battlemage uses the `xe` DRM driver **only** (no i915 path; "Xe kernel driver is only used by default beginning with Lunar Lake and Battlemage"). xe adds VM_BIND, explicit sync, GuC execution queues, and recoverable page faults / GPU SVM (xe-specific `DRM_XE_*` paths). Pairs with compute-runtime (NEO) at Level-Zero 1.15 / OpenCL 3.0 (auto-detects the KMD; no userspace flag picks xe vs i915). **Target kernel 6.12+** (Xe2 enabled by default). **Resizable BAR is mandatory** - "the GPU driver does not support configurations without Resizable BAR when using the xe-kmd module" (watch this with Unraid passthrough). [WELL-SOURCED]

### D.2 Level-Zero: command lists, immediate lists, L0 v2

- **Immediate command lists** (`zeCommandListCreateImmediate`, takes a *queue* descriptor) are "both a command list and an implicit command queue" - commands execute on append, no close/execute round-trip -> lower launch latency. **Caveat:** more host overhead per append; Intel warns that for single-queue sub-10us kernels, immediate lists can *regress*. [WELL-SOURCED]
- **L0 v2 = the Unified Runtime Level-Zero v2 adapter** (default in oneAPI 2025.3 for Xe2/B-series; **only supports immediate command lists**; "significantly reduces host runtime overhead"). The B70 falls in this default group. **But it is currently fragile on dual B70:** L0 v2 copy-offload triggers `xe` BCS copy-engine resets; the stable workaround is `SYCL_UR_USE_LEVEL_ZERO_V2=0`. [WELL-SOURCED]
- **IPC peer buffers:** `zeMemGetIpcHandle` / `zeMemOpenIpcHandle` (64-byte opaque handle, carries a process-local fd -> passed via Unix-socket `SCM_RIGHTS` or pidfd). No guaranteed address equivalence across processes - collectives keep a per-peer base-pointer table and translate offsets. [WELL-SOURCED]

### D.3 No P2P on B70 - the wall for collective authors

The dual-B70 report (vLLM #41663, **your exact SKU 8086:e223**) shows **`p2p_access:0`** and **"No XeLink between GPUs."** Xe Link is Max-series-only; consumer/Pro Arc has none and must go over host PCIe. Consequence: `zeDeviceCanAccessPeer` is **false** across cards, so the IPC peer-pointer / direct-remote-read kernel pattern (NCCL/Intel-SHMEM style) **does not work across two B70s** - you are forced into **host-staged** copies (device -> host-pinned -> device). IPEX states directly: *"oneCCL Bindings for Pytorch allreduce primitive does not support PCIe for cross-cards communication"* (-> `TORCH_LLM_ALLREDUCE=0`). Tuned oneCCL still works (~140 tok/s TP=2, ~540 TP=4) but host-routed. [WELL-SOURCED]

> **This materially downgrades the "ESIMD IPC peer-buffer all-reduce" idea on B70.** The local-write/remote-read kernel collective works on parts with P2P/Xe-Link; on dual B70 with `p2p_access:0`, a hand-rolled kernel cannot read peer device memory directly and must host-stage anyway. The realistic win on B70 is a *capture-safe* host-staged all-reduce (compatible with XPUGraph), not a P2P-kernel all-reduce. [INFERRED from the P2P-absent + host-stage facts; "IPC peer kernel across two B70s" is UNVERIFIED / likely unavailable]

- **`ZE_AFFINITY_MASK`:** decimal `device[.sub-device]`, comma-separated; hides + re-indexes survivors. On single-tile B70 use whole-device `0,1`. [WELL-SOURCED]

### D.4 torch.xpu graph capture (XPUGraph) now exists

PyTorch 2.11 added **`torch.xpu.XPUGraph`** (PR #174046, merged Feb 2026): capture/replay an XPU op sequence to cut launch overhead - a CUDA-graph equivalent. API mirrors `torch.cuda` (`XPUGraph`, `torch.xpu.graph`, `graph_pool_handle`, `make_graphed_callables`). Underpinned by **SYCL-Graph** (`sycl_ext_oneapi_graph`), which maps to **Level-Zero command lists** (finalize once, replay cheap). PyTorch 2.12 added device-agnostic `torch.accelerator.Graph`. **But vLLM-XPU still runs eager** - it disables cudagraph_mode when PyTorch lacks XPU graph support and has not yet been observed to adopt the new `XPUGraph` API. [WELL-SOURCED; "vLLM integrated XPUGraph" = UNVERIFIED, likely not yet]

---

## E. Ranked Contribution Targets

Difficulty: S (days, plumbing) / M (1-2 weeks, one kernel) / L (multi-week, new math or cross-backend) / XL (research-grade).

### #1 - XPU INT8 W8A8 scaled-MM kernel + registry entry  [RANK 1, difficulty M]

- **Gap:** no `_POSSIBLE_INT8_KERNELS[XPU]`; no `int8_gemm_w8a8` native op. (Section C.2/C.5)
- **Surface:** **oneDNN matmul** (s8xs8->s32 is hardware-native on Battlemage XMX - B.3). Lowest-risk because the systolic work is library-provided; this is wrapper plumbing, not new math.
- **Where:** (a) vllm-xpu-kernels: `csrc/xpu/onednn/int8_gemm_w8a8.h` + `joint_dtypes_t::s8_s8` + `torch_bindings.cpp` op. (b) vLLM: `XPUInt8ScaledMMLinearKernel` in `scaled_mm/xpu.py` calling `torch.ops._xpu_C.int8_gemm_w8a8`, + the registry entry.
- **Risk:** confirm oneDNN GPU ships an *optimized* (not reference) s8xs8 kernel for the target shapes. Reuse the per-token int8 activation quant already in the W4A8 path.

### #2 - Fix `choose_scaled_mm_linear_kernel` to have an XPU branch  [RANK 2, difficulty S]

- **Gap:** bare-subscript KeyError on missing platform; also breaks the **GDN W8A8 projection path** (Section C.6).
- **Surface:** pure Python. Add the `_POSSIBLE_INT8_KERNELS[XPU]` entry (from #1) and harden the subscript to `.get(..., [])`.
- **Why high-leverage despite tiny size:** it converts a hard crash into either a working kernel (with #1) or a clean, debuggable `ValueError`, and it's a prerequisite for INT8-quantized Qwen3-Next/GDN models loading on XPU at all. **Do #2 together with #1.**

### #3 - Faster XPU Gated-DeltaNet decode kernel  [RANK 3, difficulty L]

- **Gap:** the Triton/FLA `fused_recurrent_gated_delta_rule_fwd_kernel` causes `UR_RESULT_ERROR_DEVICE_LOST` on BMG (#6658); the sequential loop-carried state recurrence stresses the Xe2 runtime. Decode reportedly ~20% of bandwidth ceiling.
- **Surface:** **ESIMD or sycl-tla**, written into `csrc/xpu/gdn_attn/xe_2/` alongside the existing `gated_delta_rule.hpp` (the native SYCL path that already exists for prefill/MTP). Avoid the recurrent-Triton path. The chunked formulation (reusing the SSD/segsum machinery) parallelizes better than the per-timestep recurrence.
- **Risk:** highest-effort of the top targets (real kernel design + numerics). The native SYCL GDN op already exists (#43565/#368), so this is *optimizing* an existing kernel rather than writing from zero - which lowers risk somewhat. Decode is bandwidth-bound, so target memory-access patterns (coalescing, 2D block loads) over raw FLOPs.

### #4 - XPU graph capture / capture-safe collectives  [RANK 4, difficulty L]

- **Gap:** vLLM-XPU runs eager (high launch overhead; spec-decode goes net-negative). `torch.xpu.XPUGraph` now exists (D.4) but vLLM hasn't adopted it.
- **Surface:** (a) wire vLLM's XPU platform to `torch.xpu.XPUGraph` (Python + platform plumbing). (b) a capture-safe all-reduce - **but on dual B70 this must be host-staged, not a P2P kernel** (D.3). The ESIMD local-write/remote-read pattern needs P2P, which B70 lacks; so the realistic B70 contribution is a host-staged collective that is graph-capture-safe (no dynamic allocation/sync inside capture).
- **Risk:** graph integration depends on PyTorch 2.11+ and on vLLM's XPU runner; the P2P-less constraint caps the collective upside. Single-GPU B70 users get the bigger win (launch-overhead reduction) and don't need the collective at all.

### #5 - llama.cpp SYCL DeltaNet / quant-GEMM ops  [RANK 5, difficulty M-L]

- **Reframed:** llama.cpp **already has** `GGML_OP_GATED_DELTA_NET` with SYCL support (premise correction). The real llama.cpp gaps are: (a) a true `joint_matrix`/DPAS **quantized** GEMM to engage XMX INT8 for prefill instead of the current dequant->fp16 path (maintainers signaled DPAS is the intended direction, discussion #12570, but it's unwritten); (b) extend the MMVQ+reorder decode path to more quant types (the #21527 pattern); (c) finish Vulkan SSM/DeltaNet shaders (#19957) if Vulkan matters.
- **Surface:** ggml-sycl (auto-globbed `*.cpp`; add op dispatch + `supports_op` case; CPU reference enables auto-fallback). Adding a ggml op is a well-trodden ~12-16 file change (PR #17063 as template).
- **Risk:** the maintainers' "no joint_matrix" stance means a DPAS GEMM must be hand-written; numerically validated against `test-backend-ops`. Lower priority for *this project* since the B70 LLM stack is vLLM-XPU-centric and llama.cpp DeltaNet already works (just slowly).

---

## F. Is a true INT8 W8A8 XMX kernel even worth it? (quantitative)

**At batch-1 decode: NO.** Decode is bandwidth-bound; throughput is set by bytes-of-weights-read-per-token. **INT8 and FP8 are both ~1 byte/weight**, so identical memory traffic and identical decode speed - the INT8 TOPS advantage (a compute-side win) is invisible at small batch. On B70 (608 GB/s, decode-bound for a single stream), FP8 already saturates the same bandwidth. (This matches `05_w8a8_recipe.md` and the llm-compressor #2549 H20 datapoint where W8A8 was *slower* than FP16 at batch-1.) [WELL-SOURCED]

**Where INT8 W8A8 actually wins over FP8 on B70:**
1. **Prefill (compute-bound).** Long prompts make the GEMMs arithmetic-bound. Here INT8 is **native systolic** on Battlemage (B.2: ~367 INT8 TOPS, 4096 int8 ops/clk/core) while **FP8 is conversion-based, riding the FP16 datapath** (~2048 ops/clk/core, A.3). So INT8 prefill GEMM has up to **~2x the systolic throughput of the (up-converted) FP8 path** on this specific hardware. That is the real, B70-specific argument for INT8 W8A8 - not memory, but native-int8-vs-emulated-fp8 compute. [INFERRED from the DPAS rates + oneDNN FP8-conversion finding; the 2x is an upper bound, real kernels lose some to dequant overhead]
2. **Large-batch / high-concurrency decode.** As batch grows, decode crosses from bandwidth-bound to compute-bound; the same native-int8 advantage reappears.
3. **W4A8 already exploits this** - the one int8-activation path on XPU feeds native int8 DPAS with int4 weights, getting both smaller weights *and* the native-int8 compute.

**Net:** worth it specifically for **prefill-heavy and large-batch** B70 serving, because of the native-INT8-vs-conversion-FP8 systolic asymmetry, not for single-stream decode. For batch-1 chat, FP8 remains the right choice (`06_vllm_latest_xpu.md`). The strongest combined play is **W4A8** (native int8 compute + 4-bit weight bandwidth), which already exists and just needs the registry unblocked.

---

## Bottom line for our project

**Write these first, in this order:**

1. **XPU INT8 W8A8 scaled-MM kernel + the registry/selector fix (targets #1 + #2 together).** Toolchain: **oneDNN matmul** (native s8xs8->s32 on Battlemage XMX) wrapped as `torch.ops._xpu_C.int8_gemm_w8a8` in **vllm-xpu-kernels** (`csrc/xpu/onednn/`, modeled on `int4_gemm_w4a8.h`), plus `XPUInt8ScaledMMLinearKernel` + the `_POSSIBLE_INT8_KERNELS[XPU]` entry in vLLM. Difficulty M+S; lowest risk (library does the systolic work); **also unblocks the Qwen3.6 GDN W8A8 projection KeyError.** Highest leverage because it's plumbing over a hardware-native path and clears a hard crash. Caveat: verify oneDNN ships an *optimized* s8xs8 GPU kernel for your shapes, and remember the win is prefill/large-batch, not batch-1 decode (Section F).

2. **Faster XPU Gated-DeltaNet decode kernel (target #3).** Toolchain: **ESIMD or sycl-tla**, written into the existing native `csrc/xpu/gdn_attn/xe_2/` (optimize the chunked path; do NOT use the recurrent Triton kernel that device-losts in #6658). Difficulty L, but you're optimizing an existing SYCL op, not starting cold. This is the actual decode bottleneck for the Qwen3.6 workload; target memory-access patterns since it's bandwidth-bound.

3. **(If pursuing collectives/graphs) wire vLLM-XPU to `torch.xpu.XPUGraph` (target #4).** Toolchain: PyTorch 2.11+ `torch.xpu.XPUGraph` (SYCL-Graph over L0 command lists) in vLLM's XPU runner. **Skip the P2P-kernel all-reduce on B70** - `p2p_access:0` forces host-staging; build a capture-safe *host-staged* collective instead. The single-GPU launch-overhead reduction is the real prize here.

**Toolchain summary by job:** native INT8/fp16/bf16 GEMM -> oneDNN (or portable `joint_matrix`); INT4/FP8 custom GEMM or flash-attn -> **sycl-tla**; quantized Triton matmul -> Triton-XPU `tl.dot` (int8 lowers to `S32_S32_S8_S8` DPAS, but validate Xe2 stability); custom collectives / hand-tuned kernels -> **ESIMD**; graph capture -> `torch.xpu.XPUGraph`. brgemm/ukernel is CPU-only - not usable on B70.

---

## Sources

Microarchitecture / DPAS:
- IGC vISA DPAS spec (K/depth/dtype table): https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
- SYCL joint_matrix extension spec: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc
- Intel ESIMD ISA doc (DPAS K formula, VNNI, s4/u4): https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md
- Intel oneAPI GPU Optimization Guide - Xe arch / XMX joint_matrix: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html
- arXiv 2508.06753 "Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs" (XeTLA int2xint8 DPAS, B580 numbers): https://arxiv.org/html/2508.06753v2
- Chips and Cheese, Battlemage architecture: https://old.chipsandcheese.com/2025/02/10/intels-battlemage-architecture/
- HWCooling Xe2 analysis: https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/
- ServeTheHome B70/B65 launch (367 TOPS / 32GB / 608 GB/s): https://www.servethehome.com/intel-announces-arc-pro-b70-and-b65-video-cards-big-battlemage-brings-big-memory-for-ai-workstations/
- Phoronix B70 Linux review: https://www.phoronix.com/review/intel-arc-pro-b70-linux

Programming surfaces:
- ESIMD function reference (block_load/cache_hint/fence/atomic, verbatim): https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd_functions.md
- ESIMD enums header (fence_scope::system_acquire, cache_hint, atomic_op): https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/common.hpp
- Intel guide - Optimizing Explicit SIMD Kernels: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-explicit-simd-kernels.html
- oneDNN ukernel basic concepts (CPU-only): https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_basic_concepts.html
- oneDNN GPU data types (Xe2-HPG s8 native, FP8 conversion): https://raw.githubusercontent.com/uxlfoundation/oneDNN/main/doc/programming_model/data_types.md
- oneDNN matmul primitive: https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html
- oneDNN v3.6 release notes (int8-act + int4/int8-weight on Intel GPU): https://github.com/uxlfoundation/oneDNN/releases/tag/v3.6
- intel-xpu-backend-for-triton v3.7.1 (Big Battlemage / G31): https://github.com/intel/intel-xpu-backend-for-triton/releases/tag/v3.7.1
- Triton-XPU architecture (tt.dot -> XMX): https://github.com/intel/intel-xpu-backend-for-triton/blob/main/docs/ARCHITECTURE.md
- intel/sycl-tla (CUTLASS for Intel GPUs, BMG, int8/int4/fp8): https://github.com/intel/sycl-tla
- oneMKL gemm_bias (int8 GPU GEMM): https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-0/gemm-bias.html

vLLM-XPU structure:
- Linear kernel registry/selector (`__init__.py`): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/__init__.py
- ScaledMMLinearKernel interface: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py
- XPU scaled_mm (FP8-only): https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/kernels/linear/scaled_mm/xpu.py
- compressed_tensors W8A8 int8 scheme: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
- vllm-xpu-kernels repo: https://github.com/vllm-project/vllm-xpu-kernels
- vllm-xpu-kernels v0.1.9 release: https://github.com/vllm-project/vllm-xpu-kernels/releases
- GDN attention backend: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/gdn_attn.py
- Qwen GDN linear-attn (projections-by-caller): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- RFC: XPU kernel migration to vllm-xpu-kernels: https://github.com/vllm-project/vllm/issues/33214
- PR #43565 (XPU GDN-attention MTP): https://github.com/vllm-project/vllm/pull/43565

Driver / runtime:
- kernel.org xe driver RFC: https://www.kernel.org/doc/html/v6.8/gpu/rfc/xe.html
- kernel.org GPU SVM / recoverable page faults: https://docs.kernel.org/gpu/rfc/gpusvm.html
- Intel compute-runtime README (Battlemage, L0 1.15): https://github.com/intel/compute-runtime/blob/master/README.md
- compute-runtime FAQ (ResizableBAR mandatory on xe): https://raw.githubusercontent.com/intel/compute-runtime/master/FAQ.md
- Level-Zero spec - command lists / immediate / IPC: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html
- Level-Zero header (ze_ipc_mem_handle_t): https://raw.githubusercontent.com/oneapi-src/level-zero/master/include/ze_api.h
- oneCCL PCIe LL protocol (ARCB, L1-uncached + flag handshake): https://raw.githubusercontent.com/uxlfoundation/oneCCL/master/src/coll/algorithms/utils/protocol/proto_rt64.hpp
- oneCCL allreduce_pcie / ESIMD topo kernel: https://raw.githubusercontent.com/uxlfoundation/oneCCL/master/src/coll/algorithms/allreduce/sycl/allreduce_pcie.cpp
- IPEX known issues (oneCCL no PCIe cross-card allreduce): https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/tutorials/performance_tuning/known_issues.html
- vLLM #41663 (dual-B70 p2p_access:0, no XeLink, BCS resets, env workarounds): https://github.com/vllm-project/vllm/issues/41663
- PyTorch 2.11 release blog (torch.xpu.XPUGraph): https://pytorch.org/blog/pytorch-2-11-release-blog/
- PyTorch torch.xpu docs: https://docs.pytorch.org/docs/main/xpu.html
- PyTorch PR #174046 (XPUGraph frontend): https://github.com/pytorch/pytorch/pull/174046
- Codeplay SYCL-Graph -> Level-Zero blog: https://codeplay.com/portal/blogs/2024/01/22/sycl-graphs

DeltaNet / llama.cpp:
- intel-xpu-backend-for-triton #6658 (gated_delta_rule DEVICE_LOST on BMG): https://github.com/intel/intel-xpu-backend-for-triton/issues/6658
- llama.cpp PR #16095 (Qwen3 Next / GGML_OP_GATED_DELTA_NET): https://github.com/ggml-org/llama.cpp/pull/16095
- llama.cpp PR #17063 (CUMSUM/TRI/SOLVE_TRI/SOFTPLUS/EXPM1 - new-op template): https://github.com/ggml-org/llama.cpp/pull/17063
- llama.cpp #21893 (Battlemage SYCL corruption; GGML_SYCL_DISABLE_OPT=1): https://github.com/ggml-org/llama.cpp/issues/21893
- llama.cpp #19728 (Metal DeltaNet assert at delta-net-base.cpp:316): https://github.com/ggml-org/llama.cpp/issues/19728
- llama.cpp #19957 (Vulkan missing SSM_CONV/SSM_SCAN shaders): https://github.com/ggml-org/llama.cpp/issues/19957
- llama.cpp #21517 / PR #21527 (Q8_0 slow -> MMVQ reorder, B70): https://github.com/ggml-org/llama.cpp/issues/21517
- llama.cpp #12690 (Vulkan coopmat disabled pre-Xe2): https://github.com/ggml-org/llama.cpp/issues/12690
- llama.cpp discussion #12570 (SYCL maintainers: no joint_matrix, DPAS later): https://github.com/ggml-org/llama.cpp/discussions/12570
- ggml SYCL backend docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md

> **Honesty flags / loose ends:** (1) "FP8 rides the FP16 systolic path on Xe2, so INT8 prefill is up to ~2x" is INFERRED from the DPAS rate table + oneDNN's FP8-conversion legend; not a single verbatim Intel sentence - benchmark before quoting the 2x. (2) Whether oneDNN's Intel-GPU engine ships an *optimized* s8xs8 kernel (vs reference) for your shapes is UNVERIFIED - test empirically before assuming the W8A8 kernel is fast. (3) "Intel consumer PCIe D2D atomics unreliable" is INFERRED from the `system_acquire` spec wording + oneCCL's separate non-atomic PCIe path; no flat Intel statement found. (4) "IPC peer-buffer kernel all-reduce across two B70s" is likely UNAVAILABLE (`p2p_access:0`) - host-staging is the documented reality. (5) The "build 9680 segfault" could not be matched; the real DeltaNet crashes are #19728 / #17586. (6) vLLM has NOT yet been observed to adopt `torch.xpu.XPUGraph` - treat graph-mode-in-vLLM-XPU as not-yet-landed.
