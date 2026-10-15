# S61: Strata fusion triage (2026-10-15)

CONFIG -> Hold a new fusion build/GPU campaign. Compare the retained S42 grouped
peer GEMM, S45 CPU-reference feasibility, S44 independent accuracy/warm results,
a concrete RMSNorm-plus-Q8 pair and current SYCL MMQ/XMX reachability.

COMMAND -> Read current pinned source and S31's mapped159-operation evidence
from five traced cycles5..9. No new model review, test, build, GPU operation,
weight payload read, numerical reference, timing or runtime-source change.

RESULT -> Grouped oneMKL is a distinct prefill candidate with unresolved operand
provenance and no speed result. The concrete decode RMS/Q8 seam has a small
retained service pool. Existing MMQ/XMX code is not a ready decode replacement.

| Historical S31 GPU0 pool | Calls/cycle | GPU service ms/cycle |
| --- | ---: | ---: |
| Dense projections |468|13.8990344|
| Hyper-connections |447|7.1990958|
| Native primary experts |144|7.7380890|
| GDN, all operations |336|2.8479270|
| Activation quantization, all formats |281|0.7135206|
| Native Q8_1, grid144 |48|0.1155412|

These are operation service sums, not additive wall time or recoverable savings.
S31 predates the later HC/head experiments. Grid144 matches C6*K6144 and includes
GDN output and QSA attention output; it cannot all be assigned to GDN fusion.
The peer's144calls/8.0659412ms use a separate device clock. S42's prefill oneMKL
calls are different from these native decode expert entries.

[S42](20261015_strata_grouped_gemm_probe.md) stopped at its declared GU byte gate:
902568 of2237440 values differed, maxabs7.4505806e-8, with structural and immutable
checks passing before comparison. Candidate SiLU/down and timing never ran.
That strict gate remains FAILED. Differences do not establish invalid arithmetic
or poor speed. Vendor grouped FP16 GEMM is distinct from the earlier11-15% slower
custom primary grouped-XMX path and rejected per-layer async submission.

Current peer IPC prefill still submits per-expert DQ/GU/SiLU/down. Grouped waves
could address those submissions, with immutable shared-USM metadata and explicit
event dependencies; one library call need not mean one device kernel. Fifteen
extra DQ slots cost147456000B (140.625MiB), requiring actual headroom/residency
checks. This mechanism does not alter the native decode expert dispatcher.

If a later prefill stage is selected, the cheapest prerequisite is S45's20-64
selected FP64 dot cells from saved compact quantized blobs and actual FP16 input.
Use existing ggml dequantization and RN FP16 conversion. Host sycl::half truncates
subnormals and is unsuitable. S42 saved no GPU-dequantized FP16 weight planes;
exact CPU/device operand identity remains conditional on conversion/denorm
provenance. A few cells assess plausibility, not full-wave correctness.

Any new accuracy contract must be declared before execution: independent actual-
operand reference, justified accumulation/summation bounds, finite/zero handling,
and separate errors for both algorithms. Require strict direct/graph/repeat and
changed-input equality within each algorithm, full guards/immutable checks, then
complete DQ/GU/SiLU/down/metadata timing. Do not retrofit an epsilon onto S42.
[S44](20261015_strata_q6_collection_and_warm_probe.md) illustrates the distinction:
its independent accuracy gate passed, but warm packed-Q6 integration was rejected.
That result supplies no grouped-GEMM accuracy or speed qualification.

A concrete RMS/Q8 pair exists at MtpDrafter::record_front, mtp.cpp822..824:
weighted RMSNorm of the N2560 embedding, Q8_1 quantization, then projection.
A new fused producer would need identical normalized-FP32 and Q8 bytes, existing
finite/clamp helpers, changed-input/graph/guard gates and complete consumer timing.
The retained weighted-RMS1024 grids1/3 total8calls/0.0388726ms; Q8 grid10 totals
14calls/0.0355352ms with other consumers. Neither pool isolates this pair. These
small shared pools do not justify a larger150tok/s-per-stream C2 claim.

Dormant STRATA_QFUSE has source blockers, not an observed runtime failure here.
In verify.cpp1053..1069, batch_rec_ GDN calls omit xq_out while g_qfuse skips the
standalone quantizer afterward. Separately, gdn_q8_1_store at
verify_kernels.dp.cpp306..337 lacks the current finite scale/sum and int8 clamp
helpers used by native_mmvq.dp.cpp191..205. Global QFUSE also changes HC producer
work/counters. Enabling it is not a clean one-pair fusion experiment. The current
OFF path is unaffected by this source audit; no flag or code was changed.

SYCL CMake245..248 builds moe_fused_stub/xmx_moe, not moe_mmq.dp.cpp;
prefill.cpp59..64 supplies mmq::built=false. A migrated source file is not a
built DPAS route. Existing xmx_gemm_iq at iq_kernels.dp.cpp5756..5797 uses FP16
activations rather than canonical Q8 and excludes types16/17/23, including
IQ4_XS. It uses16-row tiles; the grouped variant starts at64 rows. Changing
operands and small-C geometry requires a new numerical/performance case.

VERDICT -> HOLD. Preserve grouped GU as prefill research, with the tiny saved-cell
CPU reference as its prerequisite if prioritized later. Do not queue another
fusion GPU arm, enable global QFUSE, repeat rejected grouped-XMX/GRF/packed-Q6
mechanisms or forecast whole-model gains from these pools. The separately
prepared IQ4_XS C6 primitive remains a different mechanism.

Exact source root: `/mnt/vm_8tb/b70/build/strata-ab/src-ep/`. Pins below refer to
`sycl/` paths; source ranges above belong to these versions.

| Path under sycl/ | SHA256 |
| --- | --- |
| src/core/mtp.cpp | `fcd5d2e4b029985a6e7768707a5aa489e4b5724a78db50fc2313d9c64e67ec31` |
| src/core/verify.cpp | `e97ce93a3434fa1d4e9a3a3dce99a3fd41d86aac8662fbd3c959a3c78ce0d4b2` |
| src/prefill/prefill.cpp | `d2861c5c8573fc5dd57742bde1c13f5fb6738ca5c975d273fa3ec189dd34f346` |
| CMakeLists.txt | `af2c33703f0fcb53b7f0f9b0aa823ce3fef7c9209ea24036e8b9f597fb07592f` |
| src/kernels/cuda/native_mmvq.dp.cpp | `81558e95f721413a75c3d3ec6ed7cb5eee766e4f662080cf500c7985090f03e2` |
| src/kernels/cuda/native_qsa.dp.cpp | `a935d8edbad66eb341d6e4a1c45fa51a81b9dc8606a062a0b1e95ba0e1318f37` |
| src/kernels/cuda/verify_kernels.dp.cpp | `5faf60ccd6e8d0665281253374bd97d4d874deb86f0aad479ce94f8565b2673e` |
| src/kernels/cuda/iq_kernels.dp.cpp | `acaa7da63733af9f3831b3a74b99e6176bf46a06e498e0a33c0591f7f4a29851` |
| include/strata/kernels/q8_1_finite.hpp | `6a3c49cabd87bf2b0b2f163c03986066f13c708c96adc34a83ad0234918e115b` |

Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
`fusion-next-s61-prep/source-trace-evidence.json` records all exact input paths,
source pins and43 selected operations; `findings.md` retains the full reasoning.
The retained S31 `head-pair-s31/analysis-prep/result/stage-rollups.json` is SHA256
`66befff3466a37dcd5a5b275be2587a80b25f40bc1b8e830d6fe1a4c43bb95bd`;
its reviewed-summary/operations.csv is
`1d58cbb36c8b683b990fb5e30ae831cbae961a4159182aa9650e335b1e1bcd67`.

Frozen S61:3files/33281bytes. External `fusion-triage-s61-manifest.json` SHA256
`78ee0f4ae59a7f4173078fbd353ce353b69c40f367e8b21cf84ece98174169a7`.
The external manifest excludes itself; original S42/S44/S45 records are unchanged.
