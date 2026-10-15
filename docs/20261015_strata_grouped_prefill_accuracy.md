# Grouped prefill accuracy and performance, 2026-10-15 (S80)

The full GU numerical target passes for both scalar and grouped execution.
The exact grouped mapping is rejected on performance: complete-wave medians
1.051034ms scalar versus4.404047ms grouped. Production stays scalar; no
serving change, rerun, promotion or model-quality claim.

CONFIG -> Saved real S42 quantized weights, deterministic synthetic half
activations and compact routing,16 experts,
ten GU/down format pairs,1748 rows, K2560/GU1280. One physical B70 card1,
PCI0000:44:00.0, immutable e41 image. Native authors mtp_q4_share (CPU
reference), strata_ep_s27 (leaf/controller) and ple_upstream (versioned
geometry fix/reviews); root reads sources, builds, audits and runs.

COMMAND -> Independent CPU RN-half reconstruction and FP64 GEMM reference
plus absolute-product sum for all2,237,440 GU values. S77 predeclares the
componentwise FP32 gamma2560 target plus outward FP64 reference uncertainty,
finite/normal/zero guards and exact sign-flipped input. This is an empirical
acceptance target, not a vendor arithmetic theorem. Shared bound-header SHA
468471b8ed5c972de11318bf19e7449be23fe48df7b5fbd21bcf3916de5a6ab0.
[Contract and scope](20261015_strata_decode_overlap_and_prefill_contract.md).

The first CPU helper built12s but stopped0.523s after expert0: type42 Q2_0
uses64-value blocks, not the assumed32. Its partial results remain evidence,
not a reference PASS. Version2 changes three geometry/row-stride lines only;
GU math, bounds, header and schema are unchanged. Rebuild11s; CPU run16s
including identity/finalization, numeric helper3.051s,4threads, peak124004KiB
(121.10MiB). All36 exact half-lattice checks, reference uncertainty, zero
DGEMM and positive/corruption controls pass. All52,428,800 GU half operands
are normal; raw scales include577 subnormals. No GPU had been touched.

Actual fresh Opus5.5 (151.962s) and Fable5.1 (437.759s) source reviews gave
conditional GO; root read both. A separate root recipe adds final-link
-fp-model=precise, retaining frozen original sources/scripts. This is a
conservative setting, not evidence about default FTZ. Actual map/link log
contain no crtfastmath.o. Original eight frozen archive hashes match copies
and build pre/post. Selected engine members are layer/native_head/vmm;
verify/spec objects are not selected. Existing runtime build path is an
explicit symlink to the separate actual build, recorded in root closure.
Build139s: adapter12.373s/AOT112.725s; no DRI/network or main-source mutation.

One quiet49s card1 window, leaf21s, pre/posthealth and own createdCID+NAME
removal before release. Actual production prefill is eager, so no irrelevant
graph requirement or graph run is added. Three warm waves per arm, followed
by four balanced AB/BA/BA/AB full-wave pairs, only after correctness gates.

RESULT -> Full100MiB actual GPU GU plane equals the CPU RN-half plane,
including raw-subnormal-scale cases. Full50MiB GPU down planes agree across
algorithms. CPU down planes are diagnostic; no ideal down target is claimed.
All four algorithm/input GU comparisons pass2,237,440 values each:

| Algorithm | Max absolute GU error | Worst residual/bound |
| --- | ---: | ---: |
| Scalar, initial and changed | 5.0640665e-8 | 0.000253533 |
| Grouped, initial and changed | 8.2422048e-8 | 0.000406271 |

Equal summary errors do not prove a whole-array sign-bit relation. Each
algorithm's eager/repeat/changed/restored complete GU/H/down state remains
strict; same-GU packed/segmented production SiLU producers agree. Original
S42 scalar GU/H/down anchor, metadata, guards, tails and immutable inputs
pass. Sixteen saved binary artifacts total309432320B. Historical S42 strict
cross-algorithm byte failure remains FAILED_UNCHANGED.

| Pair | Scalar complete wave, ms | Grouped complete wave, ms |
| --- | ---: | ---: |
| 0 | 1.019624 | 4.150444 |
| 1 | 1.049130 | 5.420274 |
| 2 | 1.071533 | 4.451648 |
| 3 | 1.052938 | 4.356446 |

All four grouped samples are slower; median paired change+314.594percent.
Timing includes gather, real DQ, GU, production SiLU/F16, down, dependency
barriers and grouped metadata fill. Reset/readback stays outside. Scalar
timed GU/H assertions cover final expert scratch only, plus full down;
untimed correctness exports cover every expert. Metadata fill0.101-0.300ms
and host-submit/complete-wall observations do not isolate a hardware or CPU
cause; library calls can wait internally. Four nested pairs in one process
are descriptive, not population statistics or full-model throughput.

VERDICT -> Retain the reusable independent reference; reject this exact
mapping and keep production scalar. No new serve, retry, format sweep,
ideal H/down accuracy, native-exp guarantee,2000tok/s forecast or promotion.
Actual resumed post Opus24.734s/Fable22.687s both ACCEPT; root read both
full responses and corrected unit/causal/scope overstatements. Compact
12467B post packet avoids full trace/log repetition. Post manifest SHA
9dbe7f8c10f83c6c3f1d5d4cda4c0a60d9121ce99603f5f3852807be77638685.

Raw evidence root:
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/
CPUv1/v2: grouped-prefill-cpu-reference-s80[-v2]-prep/.
GPU: grouped-prefill-accuracy-s80-prep/run-clean/.
Actual build: grouped-prefill-accuracy-s80-root-build/build/.
[Small exact execution records](../strata/flash-next/lowlevel_probe/grouped_prefill_accuracy_s80/README.md).

Completed whole-stage manifest233files757155438B, plus explicit build-path
symlink, SHA256
`5f0f2262cef1699098066fb71f5e8bbabecf4842631bdcb800f9d211fa5f728f`.
