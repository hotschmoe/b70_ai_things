# S62: grouped GU selected-cell CPU reference (2026-10-15)

CONFIG -> Bounded CPU diagnostic after S59 released its quiet lease. Preserve
[S42's failed GU byte gate](20261015_strata_grouped_gemm_probe.md). No GPU,
server change, GGUF/model-file read, cloud review or performance experiment.

COMMAND -> Under bin/cpu-run, audit only S42's saved compact quantized blobs and
actual gathered activation bytes. Two tiny host helpers call pinned existing
ggml to_float -> ggml_fp32_to_fp16_row RN -> ggml_fp16_to_fp32_row. No new quant
math or host sycl::half conversion. Build/run image is immutable e41d59c0,
network disabled, no DRI mount; linked dependencies exclude SYCL/LevelZero/MKL.

RESULT -> Full compact GU operand-class enumeration, followed by36 deliberately
selected normal-scale reference cells covering all ten GU/down format pairs.
Actual GPU-dequantized FP16 weight bytes remain **UNMEASURED**. This is a
source-conditional CPU reference, not an empirical GPU operand-identity proof.

| Population | Normal | Subnormal | Zero / Inf / NaN |
| --- | ---: | ---: | ---: |
| Raw GU block scales |204223|577|0|
| CPU reconstructed GU FP16 |52428800|0|0|
| Actual saved activation FP16 |4474880|0|0|

Subnormal raw scales occur in IQ3_XXS/IQ3_S fixtures. CPU dequantization can
produce normal FP16 weights from them; that does not establish GPU half-load
behavior. The selected diagnostic excludes every row containing a subnormal
raw scale. Its reconstructed weights and saved activation rows are normal.

The selection, recorded before reconstruction, includes a normal gate/up row
for each of ten real format pairs, the first S42 mismatch, the global largest
interalgorithm difference, per-fixture largest differences and cancellation
candidates. Deduplication yields36 cells; none of the required selections was
blocked. Minimum scalar magnitude is only a cancellation-selection proxy.
These samples are neither random nor representative of the whole wave.

The CPU operands come from existing ggml routines. Finite FP16 values are
integer multiples of2^-24. Python arbitrary-precision integer products/sums on
the2^-48 lattice give the exact stored-operand dot, followed by one FP64
rounding. All36 actual sums are exactly representable in binary64, with final
rounding error0. This method adopts no GPU accuracy tolerance or pass bound.

| Selected36-cell observation | Scalar oneMKL | Grouped oneMKL |
| --- | ---: | ---: |
| Maximum absolute error to exact CPU-operand dot |4.208413884e-8|7.497146726e-8|
| RMS error |1.111258075e-8|1.393494962e-8|
| Relative L2 error |2.599286194e-7|3.259451872e-7|
| Closer-to-reference count |8|9|

Nineteen cells tie. At the first mismatch, the reference is
0.016924131428822875; scalar/grouped signed errors are3.958120942e-9 and
2.095475793e-9. Largest selected A/abs(exact_dot) is42976269.08. These are
observations against CPU-reconstructed operands, not an accuracy winner,
whole-wave qualification, model-quality result or serving-speed evidence.

The remaining weight-operand assumptions are explicit: S45's matching tables
and exact finite dyadic FP32 dequant formulas, normal finite GPU half-load
behavior, and RN FP32-to-FP16 store conversion under the recorded precise AOT
compilation. CPU FE_TONEAREST and FTZ/DAZ-off state were checked. GPU weight
bytes were not read back; do not infer that all52428800 CPU values match GPU.
The saved activation bytes themselves came from S42's actual readback.

VERDICT -> Keep S42 FAILED and grouped serving unchanged. No acceptance gate,
accuracy winner, full-wave or speed claim follows. If a later prefill stage is
chosen, the cheapest missing empirical proof is only these36 GPU-dequantized
GU rows:184320bytes (180KiB), compared bytewise to their CPU RN rows. Include
affected subnormal-scale rows separately before expanding scope. Do not default
to a100MiB full-plane export. No new GPU probe is requested by this result.

Two avoidable setup assumptions are preserved: host g++ was absent (0s stop),
then optional /usr/bin/time was absent in the pinned image after a successful
13s helper-build wrapper. The same compiled binary was reused directly,
without rebuilding, for the11.8227s operand audit. The selected-row wrapper was
necessary because that first enumerator exposes no row/cell mode; its full
selection/build/reference pipeline took13.3762s. CPU operand exports total
921600bytes. Preflight tools, writable outputs and archive/ABI/flags next time;
omit optional utilities. No AOT build, GPU health cycle or server restore.

Exact CPU source root: `/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/ggml/`.
The GPU source is `src-ep/sycl/src/kernels/cuda/iq_kernels.dp.cpp` and the
resolved GPU table header is `src-ep/third_party/ggml/ggml-common.h`, under the
same strata-ab root. The CPU archive is
`src-ep/build-sycl-ep/ggml/src/libggml-base.a`. Key SHA256 pins:

| Input/source | SHA256 |
| --- | --- |
| src/ggml-quants.c | `07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d` |
| src/ggml-impl.h | `43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3` |
| CPU/GPU ggml-common.h | `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c` |
| Existing libggml-base.a | `293516f3e8a101a7e6b732347094047ea905e40204fa5489c7202e0e5c2d0f8d` |
| GPU iq_kernels.dp.cpp | `acaa7da63733af9f3831b3a74b99e6176bf46a06e498e0a33c0591f7f4a29851` |
| select_cells.cpp | `f18a8f8a4c20f73b60f6232d4092f44bc70528f7c20f89996f5a210bc7e237b6` |
| selected_reference.py | `a8ef2e31dd1d6f044bc404089c7e8a78543c7be8c97fa18f2b5a1271d1641e78` |

Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
`grouped-gu-reference-s62-prep/` preserves exact helper sources/binaries,
commands, original setup failures, input/source pins, class counts, selection,
method, CPU operands and every sampled result. See `s62-report.json`,
`cpu/audit-run1/preflight.json`, `cells/preflight.json` and
`cells/reference-observations.json`. Original S42/S45 outputs are unchanged.

Frozen S62:43files/3731339bytes. External
`grouped-gu-cpu-reference-s62-manifest.json` SHA256
`e30dbe633599574d04008490116c2a2e173ac1bdd0d697f18148aa5859445570`.
The external manifest excludes itself.
