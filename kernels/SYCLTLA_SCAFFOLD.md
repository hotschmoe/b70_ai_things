# SYCL-TLA scaffold: small-M int8 GEMM microbench (plan item C1)

Foundation for the fused small-M int8 decode kernel per Intel arXiv:2508.06753 v2
(VNNI16-packed int8 weights + rectangular subgroup tiles reusing each dequant weight
register ~8x to keep the DPAS array fed at M<8). This doc covers the SCAFFOLD only:
what builds, on which toolchain, how to run the harness, and the exact source files
to start the real kernel from. It is NOT the optimized kernel.

Source tree (git-ignored, external): `/mnt/vm_8tb/b70/sycl-tla` (github.com/intel/sycl-tla,
v0.9.1-dev, commit 6601377). Bench harness + build script: `/mnt/vm_8tb/b70/sycl-tla-bench`.
Build output: `/mnt/vm_8tb/b70/sycl-tla/build_bmg`. Nothing here is compiled into the repo.

## 1. Toolchain / environment (chosen)

Build INSIDE the container `vllm-xpu-env:v0240` -- it is the same image that already
compiles sycl-tla for the GDN kernels (`vllm/build_v0240_int8gdn_so.sh`,
`BUILD_SYCL_TLA_KERNELS=ON`). It has the full toolchain; the host does NOT (host has only
`ocloc`, no `icpx`).

| tool  | in `vllm-xpu-env:v0240` |
|-------|--------------------------|
| icpx  | oneAPI DPC++ 2025.3.3 (meets sycl-tla's ">= 2025.1") |
| cmake | 4.3.4 (`/opt/venv/bin`) |
| ninja | 1.13 (`/opt/venv/bin`) |
| ocloc | 26.18.38308 (AOT) |
| oneMKL| 2025.3 (`MKLROOT` set; `libmkl_sycl_rng` + `MKLConfig.cmake` present -- needed for example RNG) |
| torch | 2.12.0+xpu |

Host runtime for the GPU run: Intel Compute Runtime 26.22 (newer than sycl-tla's stated
25.13 minimum). Build is compile-only (no `--device`); the GPU run is left to the orchestrator.

GOTCHA (cost me a wall): oneAPI `setvars.sh` dies under `set -u` (it dereferences unset
vars). The build script uses `set -o pipefail` WITHOUT `-u`. Do not add `-u`.

## 2. Build

Reproducible, one command from the host (no GPU):

```
docker run --rm -v /mnt/vm_8tb/b70/sycl-tla:/src -v /mnt/vm_8tb/b70/sycl-tla-bench:/bench \
  --entrypoint bash vllm-xpu-env:v0240 \
  -c 'bash /bench/build_in_container.sh <targets...>'
```

`build_in_container.sh` configures (once, cached via `build_bmg/build.ninja`) then `ninja`-builds
each named target. CMake config (from the SYCL build doc, BMG-targeted):

```
CC=icx CXX=icpx cmake /src -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \   # B70 = Battlemage BMG-G31 / Xe2
  -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=Release
```

AOT perf hints exported before ninja (from the build doc, so the orchestrator's GPU run does
NOT JIT): `SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file`,
`IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread`, `IGC_VISAOptions=-perfmodel`,
`IGC_VectorAliasBBThreshold=10000`. `-DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31` triggers ocloc
AOT at link, so the binaries carry BMG-G31 machine code (confirmed by the ocloc
"# Asm Insts / Spill Size" report in the build log).

### What built (targets)

Built 2026-07-03, all AOT for `intel_gpu_bmg_g31` (ocloc asm report in the build log,
zero register spills):

| target | binary | size | RC |
|--------|--------|------|----|
| `00_bmg_gemm` | `build_bmg/examples/00_bmg_gemm/00_bmg_gemm` | 918 KB | 0 |
| `02_bmg_gemm_bf16_s8_bf16` | `build_bmg/examples/02_bmg_gemm_mixed_dtype/02_bmg_gemm_bf16_s8_bf16` | 1.3 MB | 0 |
| `02_bmg_gemm_f16_s8_f16_tensorwise` | `.../02_bmg_gemm_f16_s8_f16_tensorwise` | 1.0 MB | 0 |

`02_bmg_gemm_bf16_s8_bf16` is the int8 (mixed-precision, s8 weight) GEMM the mission asked
for. CMake configure ~4s; example 00 AOT compile ~66s; the two mixed-dtype examples ~2.5 min
each (template-heavy). NOT yet run on the GPU (that is the orchestrator's step 3, section 3).

## 3. Microbench harness

`/mnt/vm_8tb/b70/sycl-tla-bench/bench.py` drives the built example binaries over the
Qwen3.6-27B decode shapes and reports us/iter, GB/s, weight-only GB/s, % of the 608 GB/s
B70 read roofline, and TFLOP/s. The sycl-tla examples already accept
`--m --n --k --l --iterations --mode` and print `Disposition: Passed` +
`Cutlass GEMM Performance: [X]TFlop/s (Y)ms`; the harness parses those.

Shapes (hidden K=5120; heads=24, kv_heads=4, head_dim=256, inter=17408):

| label      | N      | K      | note |
|------------|--------|--------|------|
| qkv_fused  | 8192   | 5120   | q 6144 + k 1024 + v 1024 |
| q_proj     | 6144   | 5120   | |
| o_proj     | 5120   | 6144   | K = heads*head_dim |
| gate_up    | 34816  | 5120   | 2 x 17408 |
| down       | 5120   | 17408  | |
| kv_small   | 1024   | 5120   | small-N KV proj = the int8 GEMV-trap shape |

M in {1, 2, 4, 8, 16}. Bytes for GB/s are dtype-aware per example (act szA, weight szB,
out szC): bf16 example = 2/2/4, mixed bf16xs8 = 2/1/4 (weight is the int8 operand by default).

### Run it (ORCHESTRATOR, one card, needs the GPU + the shared lease)

```
cd /mnt/vm_8tb/github/b70_ai_things && \
  ./bin/gpu-run --card 0 bash /mnt/vm_8tb/b70/sycl-tla-bench/run_bench.sh
```

`run_bench.sh` wraps the container GPU run: `--device /dev/dri`,
`ONEAPI_DEVICE_SELECTOR=level_zero:0`, then `python3 bench.py`. Env overrides:
`CARD` (default 0), `EXAMPLES` (default `bf16,bf16_s8`), `ITERS` (default 100).
oneDNN reference numbers for the same shapes already live in
`docs/kernel/19_int8_microbench_results.md` and `docs/kernel/23_b70_gemv_gemm_roofline.md`
(int8 GEMV: 59% BW @ M=1, ~1.1x over bf16 at small-N) -- compare sycl-tla against those.

## 4. Path to the arXiv:2508.06753 kernel -- start-from files

sycl-tla already ships the int8 building blocks; the paper's kernel is a small-M
specialization of them, not a from-scratch write.

- **DPAS int8 atoms**: `include/cute/arch/mma_xe.hpp` -- `XE_DPAS_TT<M, s32, s8, s8, s32>`
  (s8xs8->s32, line ~151) plus s8xs4 / s8xu4 / s4xs8 / u4xs8 for W4A8. **M is a template
  parameter** = exactly the rectangular-subgroup-tile knob the paper varies at M<8.
- **Pure int8 (W8A8) mainloop**: `include/cutlass/gemm/collective/xe_mma_w8a8.hpp`
  (dispatch policy `MainloopIntelW8A8`, `include/cutlass/gemm/dispatch_policy.hpp:1279`).
  Currently exercised via the fp8 example `examples/08_bmg_gemm_f8`; there is NO dedicated
  s8s8s32 example -- writing one is the first concrete deliverable after this scaffold.
- **Mixed-precision (bf16 act x int8 weight) mainloop** = the paper's decode config
  (wide activation dodges the per-token act-quant that is our real bottleneck --
  docs/kernel/23): `include/cutlass/gemm/collective/xe_mma_mixed_input.hpp`
  (policy `MainloopIntelXeXMX16MixedPrecision`, dispatch_policy.hpp:1275), builder
  `include/cutlass/gemm/collective/builders/xe_mma_builder.inl`, dequant helpers
  `include/cutlass/detail/collective/mixed_input_utils.hpp`. Example to copy:
  `examples/02_bmg_gemm_mixed_dtype/02_bmg_gemm_bf16_s8_bf16.cpp`.
- **Grouped/MoE variant** (for the 35B path later):
  `include/cutlass/gemm/collective/xe_array_mma_mixed_input.hpp`
  (`MainloopIntelXeXMX16GroupMixedPrecision`), example
  `examples/10_bmg_grouped_gemm_mixed_dtype`.
- **VNNI16 weight pre-pack / subbyte reorder**: sycl-tla 0.9.1 added "subbyte reorder"
  (PR #793). Reorder utilities live under `include/cutlass/util` and the DPAS operand
  layouts in `mma_xe.hpp`; the paper's VNNI16 pre-pack maps to instantiating the B-operand
  copy atom with the DPAS-native packed layout.

### Concrete next steps
1. Add a dedicated s8s8s32 example (copy example 08's `MainloopIntelW8A8` wiring, swap fp8
   operands for `int8_t`), build it, add key `s8s8` to `bench.py::EXAMPLES` (2/1/4 -> for
   pure int8 activation use 1/1/4). This gives the W8A8 apples-to-apples vs our oneDNN op.
2. Baseline the STOCK tiles (`TileShape <256,256,32>`, subgroup 8x4x1) at M in {1..16} and
   compare to the oneDNN roofline numbers in docs/kernel/19+23. The stock tiles are large-M
   tuned; expect them to under-fill at small M (the gap the paper closes).
3. Instantiate a rectangular small-M `TiledMMA` (`XE_DPAS_TT<M=8,...>`, subgroup layout tuned
   so each subgroup owns a tall-N/thin-M slab) and measure the register-reuse win.
4. Fuse per-token int8 activation quant into the mainloop prologue (kills the
   `dynamic_per_token_int8_quant` op that is 35% of down_proj decode -- docs/kernel/23).
5. When it beats oneDNN at M=2..16, wrap as a torch custom op and upstream to
   vllm-xpu-kernels (which has NO int8 GEMM -- plan C1/C3).
