# Strata next mechanisms, 2026-10-15

S76 parks a decoded-weight prototype. S78 selects source preparation for
batch-commit overlap; S77 selects a new grouped-prefill accuracy experiment.
These stages record source findings, not performance qualifications. The
following S79 stage applies a separate default-OFF patch for a bounded
serving comparison; the qualified RAM-recovery result is unchanged.

## S76: native decode cost and parallelism

CONFIG -> Native IQ3_S C6 versus the rejected S75 local-matrix mapping.
Native author: `/root/mtp_q4_share`; root read the full findings and checked
the current source/retained profile context. No new GPU or model payload read.

COMMAND -> Source accounting and retained S31/S75 evidence only. No compiler,
new timing, source mutation or inference from missing runtime counters.

RESULT -> For sixteen output rows the native path uses sixteen SG32 groups
(512 work-items), distributing K32 segments as p=lane+32*j. S75 uses one
SG16 group and iterates all80/192 segments serially. Both decode each row/
segment once. S75 amortizes logical activation reads across16 outputs but
adds local staging/ledger references and barriers. These source facts do not
isolate the measured slowdown's cause or establish physical transactions.

One source-eligible alternative preserves the original SG32/DP4A/FP order
and replaces only weight loading with temporary decoded signed-code planes.
Codes/odd scales/raw half-d require16,343,040B per tested shape,31.171875MiB
extra for both while compressed originals remain. Expanded storage is2.41818x
compressed storage. It removes full IQ lookup/sign decoding, unlike S65 SWAR,
but retains the expanded-weight risk seen in rejected Q6 packing.

Historical S31 has five IQ3_S C6 operation rows,55calls/1.9335818ms service
per cycle. Grids matching the two tested output geometries account for13calls/
1.1693698ms; grids do not identify exact input dimensions or tensors. Service
sums are not recoverable wall time. The lead is parked while stronger paths
are prepared; no decoded-plane implementation or GPU job is selected.

VERDICT -> Preserve the source lead and its memory tradeoff. Do not refine
S75 barriers or forecast an all-format benefit from these counts.

Frozen `iq3s-native-cost-s76-manifest.json` SHA256:
`ffa9599c6fc3dd9fa5cc84af6e647a29e381fdcb504a831c4e5838ca75ad2b2b`.

## S78: replay boundaries and commit overlap

CONFIG -> Existing S31/S32 reports and current verifier/drafter ownership.
Native author: `/root/ple_upstream`; root read the full findings, scalar/batch
commit, captured commit work and the batch-step caller.

COMMAND -> Reuse existing event summaries; no duplicate270MB trace parse,
new union computation, cross-clock join, profiling run or serving start.

RESULT -> Nine source replays reconcile with the observed motif: primary
and peer verification, batch commit, and six ordered paired-draft graphs.
This is source reconciliation, not a trace-handle proof. Pair-graph coalescing
targets S32's small45us submission region; it does not explain the old gaps.
Keep it parked. S31's uncovered intervals and S32's separate CPU spans remain
diagnostic; neither establishes physical idle or removable overhead.

Batch commit currently waits unconditionally. Its graph updates main GDN,
QSA indexer and PLE history; copied final residuals and private drafter state
are distinct. Existing scalar commit already has a completion-event path.
This supports investigating overlap on existing queues, then joining before
the batch call returns. It does not prove safety or a1.59ms saving from the
old CPU commit region. Both workloads still use the primary card.

A separately numbered S79 preparation must preserve mutable commit records,
host PLE advancement, terminal/solo fallback, publication/error ordering and
draining on every exit. Layer-stage chains retain synchronous behavior. No
new queue, asynchronous slot dispatcher or broad cache lifecycle change is
selected. S78 itself changes no production source; the separate S79 stage
applies and builds a default-OFF candidate.

VERDICT -> Prepare one exact bounded fork/join patch for review. Do not
repeat S22's rejected independent-slot dispatch or revive global QFUSE.

Frozen `decode-idle-criticalpath-s78-prep-manifest.json` SHA256:
`cc6eb10aca6c6ea54f84181a34e71fa8f21d2622af196c6a4e9259980b9dce4b`.

## S77: grouped prefill numerical contract

CONFIG -> The retained16-expert/10-format S42 wave,1748 routed rows. Native
author: `/root/strata_ep_s27`; root read the full contract and checked the
precision/production-path scope. S42 remains a failed cross-algorithm byte
gate. S62 remains a conditional selected-cell CPU diagnostic.

COMMAND -> Source/API and reference-cost review, no new operands, reference
calculation, build or GPU. Official API/math references supply background;
installed headers/source and actual output gates remain necessary evidence.

RESULT -> A full GU reference covers2,237,440 outputs and5.72785G MAC per
input, not trillions. Existing CPU CBLAS double GEMM can compute reference
and absolute-product sums before GPU execution. Host and pinned-image Python
lack NumPy; the pinned image has icpx and CPU MKL. No installation is needed.
Use four threads under cpu-run and stream experts within a bounded working set.

Declare the dimension-derived FP32 gamma2560 envelope before measurement,
including outward FP64 reference uncertainty and zero/finite handling. This
is an independent acceptance target, not a promise about opaque vendor kernel
internals. Full actual GPU/CPU RN-half operand equality must establish the
reference premise. An exact FP16 sign-bit flip supplies changed input without
recreating random inputs. [Intel's grouped API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-0/gemm-batch.html)
supports half inputs/float outputs; that signature alone is not a theorem
about intermediate rounding. [Higham and Mary, sections3-4](https://eprints.maths.manchester.ac.uk/2841/1/paper_eprint.pdf)
describe the arithmetic model and matrix-product error bounds.

Keep the actual production SiLU/F16 producer. Its native exponential has
implementation-defined accuracy, so the minimal experiment does not claim
an ideal full-pipeline error bound. Require strict same-algorithm complete
GU/H/down repeat/changed-input checks and same-GU producer equivalence, plus
full memory/metadata checks. Cross-algorithm H/down differences remain
diagnostics. [Khronos native math reference](https://github.khronos.org/SYCL_Reference/iface/native-prec-math-func.html).

Root scope correction for S80: actual peer prefill enqueues eager DQ/GEMM/
SiLU/GEMM, without graph capture. Grouped graph support is not a prerequisite
for this eager production path and is omitted as an irrelevant mandatory
test. Full GU operand identity is required; a full down CPU reference is not
claimed. Time complete gather/DQ/GU/SiLU/down plus descriptor fill only after
the declared numerical/coherence gates, with no intermediate timing readbacks.

VERDICT -> Prepare a separately numbered S80 CPU reference and eager grouped
leaf. No fitted epsilon, changed quant format, ideal H/down qualification,
model-quality claim, serving promotion or2000tok/s forecast follows.

Frozen proposal `grouped-prefill-contract-s77-prep/manifest.json` SHA256:
`5c0fbc6a0816508ed2eb72e56fbb82b2cea4e0131e98addc70a947cb6cde57ed`.
The original proposal remains immutable; S80 records the eager-scope correction.

All external evidence is under
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
Upstream refreshed2026-10-15T15:02Z: main296b59d2 unchanged; tracked author
PRs2511,recutita1/2 and draft2623 remain open, not merged. Refresh record:
`source-refresh-s76-root/upstream-refresh.json`. No source change or runtime
qualification is inferred from contribution status.


## S79: bounded commit/draft comparison, HOLD

CONFIG -> Same new03bb07f9 executable, two fresh two-card server arms,
commit overlap0/1 only. HC common retained, paired heads/depth2, globalRPW1,
headRPW2/S49OFF. STRATA_TRACE1 supplies native prompt-start admission; CPU
boundary/profile keys absent. One warm plus three fresh paired requests per
arm,1500 completion tokens per stream, then arithmetic/reasoning/tool checks.
Native author ple_upstream; root implements/audits the actual build/harness.

COMMAND -> Three exact candidate source files; one CPU build168s including
158.876s incremental work. Actual fresh Opus5.5/Fable5.1 source reviews took
60.254/158.009s. Root read both and closed actual recipe filename/scalar handoff
conditions. Both-card prehealth, OFF arm, normal QUIT/posthealth, ON arm,
normal QUIT/posthealth. No daily restore, long-context repeat or GPU profiler.

RESULT -> All eight pairs match request/full text/SSE content/usage. All
functionals, model/image identity, activation, normal teardown and individual/
compiled collective health checks pass. ON asked=deferred=joined for every
window. These counters do not establish hardware overlap or generic safety.

| Pair | OFF client wall, s | ON client wall, s | Native work |
| --- | ---: | ---: | --- |
| Warm, diagnostic | 28.998804 | 29.066701 | Matched |
| Repeat1 | 28.972413 | 29.095658 | Matched |
| Repeat2 | 28.903272 | 28.964655 | Matched |
| Repeat3 | 29.511710 | 29.573993 | Unmatched |

Repeat3 has556/557 windows,553 paired in each,2/3 solo. Slot1 offers1138/1140
drafts while accepting933 in both. Outputs remain identical, but this fails
the preregistered native-work gate. No aggregate timing median or speed claim
is reported. The first two matched measured pairs are descriptively slower;
no inference of a general regression follows from two nested observations.
Do not omit the third pair or normalize its extra work to manufacture a win.

ON buffers publication until its commit join, so native BDONE/commit/emit
partitions have changed meaning. Client wall is primary; SSE content-event
gaps are not token ITL. No CPU ring means exact per-window layouts/cache hits
are unobserved. PLE totals are process cumulative, not coding-only.

VERDICT -> HOLD_UNMATCHED_NO_LATENCY_PROMOTION. Park exact patch, no rerun,
no promotion or long-context qualification. Restore only the three exact
pre-S79 source files without rebuilding/restarting a server. Frozen03bb07f9
and the generic build executable still contain the candidate and do not
represent restored current source. The derived context-free patch reproduces
the candidate exactly from pinned bases; raw patch/full sources remain frozen.
[Small source/build/runtime records](../strata/flash-next/lowlevel_probe/batch_commit_overlap_s79/README.md).

Closed runtime: `batch-commit-overlap-s79-serving/comparison.json` SHA256
`6180f1fca4f88d2d72fd55f1dafacec0c317c3c1acc5a487de2169e01344269f`.
Root decision: `root-final-runtime-decision.json`; restoration:
`batch-commit-overlap-s79-execution/root-source-restoration.json` SHA256
`de87f816dad20d6cbf6ea25d814ca6a24db3b17375cc08a2df359ba74626bf08`.

Actual resumed post consultations Opus5.5 (26.109s) and Fable5.1 (29.027s)
accept HOLD/no promotion. Root read both full responses. Their restoration
record ambiguity is resolved by the separate current-source readback:
`root-source-restoration-readback.json` SHA256
`ad051e5ba6cfda984ff1946244a97ef56a1a4e1677f517db1ac37aa63b8df92a`.
The historical before/after fields describe application, not restoration.
No distribution-equivalence or causal model-review wording is adopted.
Frozen post-review manifest SHA256
`b750dfe0dd806e39ff050e0682d15b2230e44b4658a8f2f82d506bdde2331ade`.

S79 completed whole-stage manifest214files196711034B, SHA256
`292aee57cb7b1b9ab2a48cedfc461cbc0396ff5cb0708d6e61e68feec1211cd6`.
