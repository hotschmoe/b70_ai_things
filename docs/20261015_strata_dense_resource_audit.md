# S56: dense-format and existing AOT resource audit (2026-10-15)

CONFIG -> Keep global Q6 RPW2 rejected and hold its remaining actual-weight
C6 corner. S15 already tested this mechanism; existing compiled metadata
adds a concrete C6 spill liability. No new GPU window, build, benchmark,
model-weight read or serving change was needed.

COMMAND -> Read current source and retained S15/S24/S40/S41/S44 evidence,
aggregate the existing S31 operation CSV, inspect pack-index shape metadata,
then extract four `.ze_info` records from the already frozen S55 executable.
The successful ELF metadata scan reads 2,623,300 bytes in 0.037 s. It traverses
1,600 embedded images without reading their ISA/SPIR-V payloads or copying
whole images. An initial selector missed SYCL's suffixed RPW2 class names;
the corrected metadata-only pass retains exact emitted names and offsets.

RESULT -> S31's primary non-head dense pool is heterogeneous. These are
GPU0 device-service sums per cycle over five retained traced cycles, not an
additive client-wall budget or a prediction of recoverable latency.

| Format, resolved from kernel/source | Calls/cycle | Service ms/cycle |
| --- | ---: | ---: |
| Q6_K | 71 | 3.8256372 |
| Q4_K | 45 | 2.3121056 |
| IQ4_XS | 67 | 2.0693470 |
| IQ3_S | 55 | 1.9335818 |
| BF16 | 130 | 1.2114892 |
| Q5_K, non-head | 14 | 1.0027652 |
| Q8_0, draft projections | 38 | 0.7442326 |
| IQ4_NL | 40 | 0.5820252 |
| Q2_0 | 8 | 0.2178506 |
| Total | 468 | 13.8990344 |

Q6 accounts for 27.52% of that service; other formats account for
10.0733972 ms. Q4_K, IQ4_XS and non-head Q5_K together account for
5.3842178 ms through the separate `launch_wide<F,6>` path. IQ3_S uses
`launch_mmvq_multi_nc<21,false>`. BF16 uses FP32 activations and ordered
BF16-weight FMAs, rather than a quant/dequant path. IQ4_NL and draft Q8_0
use their separate 32-element-block wide kernels. Source resolves the CSV's
`Q8_0?` and unclassified `Q20Traits` labels here; the CSV remains unchanged.

Observed primary projection shapes include K2560 to rows10240/6144/12288/
512/640 and K6144 to rows2560; shared down is K640 to rows2560. Source and
pack metadata establish those roles. Kernel names establish quant formats,
but the trace does not join every call to a specific layer/tensor or K.
The dense stage also includes drafter calls; it is not exclusively target
verification. All observed S31 Q6 calls are C6. C1 is source-reachable,
without a new claim that Q6 C1 appeared in that steady C2 trace.

S15 already exercised the exact `STRATA_MMVQ_WIDE_RPW=2` mechanism on five
synthetic model shapes at C1..4, with matching output hashes. Its overall
median latency ratio was 1.1056915, so global RPW2 was rejected. Only the
6144->2560 out-projection improved: ratios 0.811920/0.827888/0.913942/
0.911908 at C1/2/3/4. Its S31 Q6 pool is 1.750572 ms/29 calls, not all
3.826 ms. Missing actual-weight C6 coverage alone does not justify a rerun.

Q6 uses `native_mmvq_q6k_wide_kernel`, not a `WideQ6K` format class. Its
kernel/launcher section remains byte-identical to upstream296b59d2, S15's
base. RPW2 shares activation loads between adjacent rows while preserving
each row's lane assignment, k order, dp4a chain and butterfly reduction.
It also increases live row state; source order alone is not a compiled
bitwise pass. The Q6 knob is `STRATA_MMVQ_WIDE_RPW`, distinct from the
head/general-K knob `STRATA_MMVQ_WIDE_K_RPW`.

S41/S44 rejected a different packed-plane algorithm, with changed arithmetic
and extra storage. S44's original control stayed RPW1. Its arithmetic bound
and warm timings do not test RPW2 or justify relaxing a new strict gate.
S24's GR/readiness work and S40's MTP Q4 ownership fix are also separate.

The four existing AOT records provide the cheaper discriminator:

| Q6 variant | GRF count | SIMD | spill_size field | Scratch-buffer size |
| --- | ---: | ---: | ---: | ---: |
| C1 RPW1 | 128 | 32 | absent | absent |
| C1 RPW2 | 128 | 32 | absent | absent |
| C6 RPW1 | 128 | 32 | absent | absent |
| C6 RPW2 | 128 | 32 | 7936 | 8192 |

C6 RPW2 declares per-thread scratch, usage `single_space`. Absent fields
are not newly measured runtime zeros. All four declare `eu_thread_count=8`;
this is compiler metadata, not observed occupancy. Source local range is
SYCL(1,4,32), 128 work-items. These `.ze_info` records declare subgroup32
but no explicit required workgroup size; walk order is not workgroup shape.

The executable is `src-ep/build-sycl-ep/strata-head-rpw2-s55`, SHA256
`bb0669002bff48f28340cd413b576fe905ffbc327a332bebd63ff12e3066272f`.
The audit reuses root's existing S55 build-closure hash and checks unchanged
file stat, rather than rehashing the full 92 MB executable. Source/object
pins and exact embedded-image offsets bind the metadata. Its compiler
comment records DPC++2026.1.0; the build pins image
`e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`.
All selected images embed `-cl-fp32-correctly-rounded-divide-sqrt`.
This establishes embedded compiled resources, not actual dispatch,
occupancy, scratch traffic, bandwidth, latency or serving benefit.

VERDICT -> Do not repeat global Q6 RPW2 or revive packed Q6. Keep the
actual-weight C6 out-projection corner on hold: its compiled RPW2 image has
explicit spill/scratch absent from the corresponding RPW1 record. No new
GPU experiment follows. The larger non-Q6 pools remain separate source
leads. This stage replaces a duplicate measurement with a cheap inspection
of already available evidence; no user performance target is newly met.

Raw root R:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
Evidence is under `R/primary-dense-next-s56-prep/`: `findings.md`,
`source-audit.json`, `s15-q6-rpw2-rejection.json`,
`s31-dense-quant-pool.json`, `s31-wide-observed-shapes.json`,
`existing-pack-shapes.json` and `existing-object-symbols.json`.
The `aot-resources/` subdirectory contains four exact `.ze_info.yaml` files,
`embedded-resource-index.json`, `resource-summary.json` and `manifest.json`.
Executable/source binding reuses
`R/decode-head-serving-s55/root-build-closure.json`.
Original S15 controls are under
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_dense_20261014/`.

Frozen whole-stage evidence: 16 files, 96267 bytes. External
`primary-dense-s56-manifest.json` SHA256
`8e5a894d4e5462473c4e74b7c015e85dcf7fbf74d30a34ebad99fd4f2a1b767f`.
