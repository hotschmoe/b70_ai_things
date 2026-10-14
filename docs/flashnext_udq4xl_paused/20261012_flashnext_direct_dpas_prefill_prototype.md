# Direct F16 DPAS grouped-prefill prototype

CONFIG -> Source/CPU only. Existing live actors, SDKs, original quantized
weights/dequantizers and failed grouped-oneMKL evidence unchanged. Actual
compiler is2026.1; no compiler, GPU or model payload access by this author.

COMMAND -> Read source45/47 grouped patches and current SDK local examples;
check Intel primary matrix/ESIMD documentation. Prepare a bounded real kernel,
linked synthetic harness and explicit NoDri compile recipe. One CPU packing/
tail-routing control passes; it is not a compile or device qualification.

RESULT -> The smallest substitution is the body of Gemm::f16_grouped. Retain
existing eight decoded GU/down slots, native dequantization, row offsets,
SwiGLU/FP16 H storage, combine/scatter and in-order slot reuse. Direct kernel
captures eight pointer/shape descriptors by value; it allocates no metadata,
submits no host_task and adds no normal wait. Pointee buffers remain caller
owned until the returned event completes. Original CPU pointer arrays may die
immediately after launch. It requires the same in-order compute queue.

The prepared prototype computes Y[ne][N]=X[ne][K]*W[N][K]^T, FP16 operands,
FP32 accumulator/output, alpha1/beta0. Fixed actual GU N1280/K2560 and down
N2560/K640, group1..8, each ne1..64. One ESIMD work-item owns8x16 outputs;
K steps16. A is128 FP16 values row-major; B is256 FP16 values packed in vertical
pairs at (k/2)*32+column*2+k%2. Invalid last-tile input rows are zeroed without
loading them; output stores are limited to real rows. Existing W layout stays
unchanged: register packing is performed after loads, not in the model files.

Intel's [DPAS header](https://raw.githubusercontent.com/intel/llvm/sycl/sycl/include/sycl/ext/intel/esimd/xmx/dpas.hpp)
defines half precision deduction and FP32 accumulation with16-wide execution.
Its [official example](https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/dpas.md)
describes row-major A/output and vertically packed B. These support the source
packing/API choice, not actual2026.1 compiler or B70 execution success.

For eight experts,16 host BLAS entry calls become two explicit SYCL GEMM kernel
submissions; eight existing SwiGLU calls remain. Dequantization and scatter
calls remain. Compared with source45/47 grouped oneMKL, two explicit metadata
retirement host_tasks disappear. Internal oneMKL GPU kernel counts were not
measured. This is a static submission argument, not measured latency or proof
that2000tok/s is reachable. Basic8-row tiling also wastes work for small ne;
memory loads/register packing and occupancy may dominate until measured.

Local source references in actual SDK47, retained read-only:

- `sycl/include/strata/kernels/xmx_gemm.hpp` and
  `sycl/src/kernels/xmx_gemm_bench.cpp`: existing quantized joint-matrix GEMM
  candidate plus original dequant/single-oneMKL reference pattern.
- `sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp`: local joint-matrix FP16
  staging example. Its16-row tile is not automatically a suitable BMG target.
- Existing grouped patch caller `sycl/src/prefill/prefill.cpp`: the eight-slot
  pending group, variable ne, final partial flush and unchanged SwiGLU/scatter.

Intel's [joint-matrix issue22926](https://github.com/intel/llvm/issues/22926)
reports B70 capability/launch failure on2025.3.3 despite working ESIMD DPAS.
Our2026.1 stack may differ: joint_matrix is not declared unavailable here.
The [matrix specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)
lists BMG shapes, but capability query and actual operation are still required.
Intel's [large-project DPAS issue21741](https://github.com/intel/llvm/issues/21741)
reports correct standalone output and wrong subsequent work-items in a B70
large backend. Its older stack is not evidence of our current failure, but
standalone success must not be treated as full-model integration success.

Linked fixture uses actual admitted SDK49 individual Gemm::f16 only, never the
faulted grouped oneMKL API. Twelve cases per single-/two-queue mode cover
group1/3/8, tiny16/32, actual GU/down and mixed ne1/2/7/16/32/48/64/64. It checks
every output against independent exact sparse/dyadic values, three asynchronous
DPAS repeats, copied descriptor lifetime and eight untouched guard rows per
output. Mod29 input values avoid modulus17 column symmetry. Compute and copy
queues drain before buffers free, including failures. Explicit caps128MiB
device/96MiB pinned host exclude driver/oneMKL internal allocations.

Owned files:

- `kernels/prefill_grouped_f16_esimd_v1.hpp`
- `kernels/prefill_grouped_f16_esimd_harness_v1.cpp`
- `kernels/prefill_grouped_f16_esimd_harness_v1.txt` (actual49 NoDri recipe)
- `kernels/test_prefill_grouped_f16_esimd_layout_cpu_v1.py`

VERDICT -> Concrete source prepared. Root may compile without exposing GPUs,
then qualify the linked fixture under its lease/health/teardown discipline.
No production patch before actual linked results pass; no unchanged source47
grouped-oneMKL retry. FP32 accumulation order may differ from oneMKL, so dyadic
operator success does not settle original-model fidelity or dense-input error.

ROOT UPDATE -> NoDri compile79791 subsequently passed in29.3s. Actual ELF:
`/mnt/vm_8tb/b70/build/prefill-f16-esimd-linked49-v1/prefill_grouped_f16_esimd_harness_v1`;
SHA256 `f2d895a55a45d5e2a8e12035fd85f1ce71cbb70f4f2da6d939cdbd370d174c9f`.
Compiled header SHA `284dd2905f8bc30d6d930568e33ca62659d267ef383b0def197d33fbf27527c8`,
harness SHA `67f2fd8e175190ad0d3232821fe491b70761e269f1b32f333fac6fdcc296e4da`.
Frozen source plan SHA `b16c6e485b477e12828c336c5c94ad257658d6b1e29a536ff42c7ee1e7ccc0e7`.
CPU layout control passes1.124s; exact frozen hashes rechecked. Single-queue
actual GPU fixture42438 is queued behind source50 build; no device verdict yet.
No compiled source, header, harness or recipe changed after this compile.
