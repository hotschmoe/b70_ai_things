# C5: Strata routed kernel vs the plugin moe_layer, toolchain A/B

Card 0, 2026-10-20. Hypothesis: C1's fused `moe_layer` at 216 us (T=1 a2) versus Strata's ~93 us routed call is build and runtime, not source. Three expert-dot rewrites had not moved it.

Verdict: the hypothesis is false. Every toolchain knob left Strata's kernel at 93-97 us and the plugin kernel at ~198 us (graph) / ~210 us (AOT). IGC 2.41.5 does not matter, so `harness/xpu-run.sh` was not changed. The plugin `_C.so` was not rebuilt. Bit-exact dequant and the tolerance tests were not retouched.

## What the two numbers are

Strata's published ~93 us is `native_expert_grouped` at IQ3_S/Q2_0, scenario "C1 card0" (T=4, 10 groups / 20 entries: 2x3, 6x2, 2x1), graph replay, 48 layers, median of 5. Source: `sycl/src/kernels/cuda/iq_kernels.dp.cpp` (`rows_dot_multi_iq` ~2263, `native_gu_kernel_rows` ~2302, `reqd_sub_group_size(32)`), bench `ep_night_bench.cpp` `grouped_bench`. Worktree pin `1c360526`. Activation quant is outside the timer. Gate and up are the same ggml type. Columns: R2 = `STRATA_NIGHT_EP_ROWS=2` (the ~93 us column); IQR2 = R2 plus `STRATA_NIGHT_IQ=1` (the served kernel, ~70 us here).

C1's 216 us is the whole fused layer at T=1, topk=10, plus one shared expert, IQ2_XS / IQ2_XXS / IQ4_NL, E=32, I=640, K=2560 (`row_moe.hpp` `b70_launch_gu` ~289, `reqd_sub_group_size(16)`, R=2). It includes Q8(x), gate+up+SiLU, Q8(hidden), down, and the route reduce. At T=1 each expert is used once, so Strata's multi-entry weight reuse has nothing to group.

Shared expert, separate binary: `night_shexp_bench`, nc=1, MMVQ level 2. Published IQ4_XS/IQ3_S/IQ4_NL is 24.9 -> 18.7 us.

## Strata source, same objects or same TU

us per call, IQ3_S/Q2_0, C1 card0, bitwise yes (153600 floats, 0 diffs). Health HEALTHY before and after.

| build | runtime IGC | R2 all | IQR2 all |
|---|---|---:|---:|
| historical AOT, night objects, ocloc of IGC 2.36.3 | 2.36.3 | 96.87 | 72.64 |
| same objects relinked in strata-sycl-dev:020 (ocloc 26.35, IGC 2.41.5) | 2.41.5 | 95.86 | 72.66 |
| that 020 binary | 2.36.3 | 96.12 | 72.96 |
| historical AOT, immediate command lists = 1 | 2.36.3 | 96.50 | 73.14 |
| JIT rebuild, Strata flags (`-fp-model=precise`, sg default 32, spir64) | 2.36.3 | 93.47 | 69.26 |
| same JIT binary | 2.41.5 | 93.25 | 69.10 |
| JIT, `IGC_OverrideGRFSize=256` | 2.36.3 | 93.84 | 69.40 |
| JIT, immediate command lists = 1 | 2.36.3 | 93.95 | 69.35 |
| JIT, plugin flags (no precise, no sg-default 32, `-O3`, spir64) | 2.36.3 | 93.84 | 69.44 |

Published R2 is 53.5 / 40.8 / 93.2. Today's JIT R2 all is 93.3-93.9 us. AOT is about 3 us slower. Plugin flags on this source match Strata flags. IQ2_XS/IQ4_NL on the 020 AOT binary (gate matches C1, up does not) is IQR2 47.17 / 38.46 / 85.64.

A link of the existing spir64_gen objects with `-fsycl-targets=spir64` dropped device code (`linked binaries do not contain expected spir64`, 10 MB, no SPIR-V). That binary was not timed. The JIT rows above are a recompile of `iq_kernels.dp.cpp` plus a grouped-only bench, inside strata-sycl-dev:020. Host icpx 2026.1.1 cannot compile that TU against the image's dpcpp-ct 2025.3 headers (`get_tangle_group` is missing from the steve-repro SYCL headers). The plugin-flag row uses icpx 2026.1.0 with the plugin flag set, then runs on host IGC 2.36.3.

`reqd_sub_group_size(32)` is in the expert kernels, so `-fsycl-default-sub-group-size` does not retarget them. GRF 256 is a source attribute on the XMX path (`grf_size<256>` near line 6399), not on the night decode path. `IGC_OverrideGRFSize=256` did not move the night kernel. SLM size is the kernel's `group_local_memory`, not a build flag. Shader `-allow_caching` is in both AOT links, which already match.

Shared expert on card 0, historical binary, image `e41d59c0`: IQ4_XS/IQ3_S/IQ4_NL nc1 orig 24.88 us, night 18.69 us, 0 differing bits. Published 24.9 -> 18.7.

## Plugin moe_layer, both toolchains

Standalone `ours_bench` (no torch). Graph column is Strata's timer (8 launches per graph, median of 5, us/call). Eager is min of 8 host `q.wait` calls. Profile is a device-event window; it fails on the AOT binaries (`program was built for 1 devices`) and the graph number is the one to use. Shape c1 is T=1, topk=10, shared expert, IQ2_XS/IQ2_XXS/IQ4_NL. Shape s20 is T=2, topk=10, no shared, IQ3_S/IQ3_S/Q2_0 (20 entries, not grouped). Finite outputs, nonzero.

| binary | c1 graph us | s20 graph us |
|---|---:|---:|
| plugin JIT, host icpx 2026.1.1, `-O3`, spir64, host IGC 2.36.3 | 198.39 | 313.24 |
| plugin JIT + `-fp-model=precise` | 200.81 | 312.69 |
| plugin JIT + `-fsycl-default-sub-group-size=32` | 198.44 | 312.86 |
| plugin JIT + `-ffast-math` | 198.33 | 313.15 |
| plugin JIT + `-O2` | 213.82 | 333.72 |
| 020 icpx JIT, no precise, no sg default | 198.47 | 313.46 |
| 020 icpx AOT bmg-g31, precise, sg default 32 | 209.70 | 366.96 |
| 020 icpx AOT only | 212.55 | 366.61 |

C1's torch a2 at this T=1 shape was 216 us. The graph timer on the same source is 198 us. Eager on the first JIT run was 487 us and on the later 020 JIT run was 207 us, so the eager min still carries launch noise. `-O2` is the only compile flag that moved the graph number, and it made it worse. AOT was a few microseconds slower. Default subgroup 32 does not override `reqd_sub_group_size(16)` (`b70_acc_pair` strides by 16).

## What was not applied

No edit to `plugins/b70_gguf_xpu` or `harness/xpu-run.sh`. Mounting IGC 2.41.5 would not change this kernel. Rebuilding `_C.so` would race F1's card-0 UMD campaign, which is measuring the existing binary.

A paste of Strata's kernel bodies behind `b70_moe_layer_launch` is not a flag change. Strata's Q8 block is 36 bytes and the plugin's `B70Q8` is 48. Strata's grouped API takes one type for gate and up and a group plan; the plugin op takes separate gate and up types and a token-by-slot id list. At T=1 the cited shape has one entry per expert, so the multi-entry weight reuse in `rows_dot_multi_iq` does not fire. That schedule cannot turn the 198 us fused layer into the 93 us routed number.

## Next step

Measure a grouped multi-entry dot, with Strata's lane map (subgroup 32, `STRATA_EXPERT_LANES` 8) and an attribution comment, at the C1 card0 plan (T=4, group sizes 3/2/1) against today's JIT R2 of 93.5 us. Keep the T=1 fused path on the current kernel until that plan is faster and `tests/gpu_check.py` (atol/rtol 5e-2) and `tests/moe_layer_bench.py` (8e-2) still pass. Dequant bit-exact stays on `tests/compare_cpu.py`.

Logs: `/mnt/vm_8tb/b70/build/vllm-flashnext/c5/meas_*.log`, `repro_020_*.log`, `repro_ocloc_iq3s_c1c0.log`.
