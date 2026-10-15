# S59: real IQ4_XS C6 rows-per-workgroup screen (2026-10-15)

CONFIG -> Keep RPW1. The existing public IQ4_XS RPW2 branch passes this strict
primitive screen, but its timing difference is too small to justify integration
or a serving run. This is separate from the Q5_K vocabulary-head experiments.

One actual main-model tensor, `blk.0.attn_qkv.weight`, supplies type23 weights
for N2560/M10240/C6. It occupies13,926,400bytes at offset741126016 in the S53
NVMe primary shard. Slice SHA256 is
`03064eacd49919859095d3eac38484ce65465b848c2c75a744b0d0f12ccf9e91`;
FNV1a64 is `e408de02a955877f`. Inputs are two finite seeded FP32 arrays,
not captured model activations. No model or live serving configuration changed.

COMMAND -> Actual Claude Opus5.5 and Fable5.1 source reviews approved the narrow
screen in30.668/40.818seconds, resuming the original S52 review sessions.
Root read the full reviews, source and harness delta. The S60 supported
[prelink build](20261015_strata_aot_prelink_reuse.md) compiled only the adapter;
main kernel objects and frozen archives were reused. Build/runtime image is
`sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`.
Executable SHA256 is
`56e551afe6faf58d377340dd8a248674ef98e7cebb2e4f07011629ee6bfc4814`.

One quiet `cpu-run` lease enclosed `gpu-run --card 0`, physical0000:0b:00.0,
for72seconds including pre/post-health; the four leaves took46seconds.
Fresh processes ran A1 RPW1, B1 RPW2, B2 RPW2, A2 RPW1 because the environment
selector is latched per process. Global WideK was1 in each process; only its
RPW selector changed. No profiler, server, other card, reset, retry or wider sweep.

RESULT -> All four leaves, host full-byte comparator, pre/post-health and
container-removal checks exit0. Direct and own initial/changed graph outputs,
Q8 bytes, guards, finite outputs, changed-input observability, full inputs and
weights pass. All48saved binary files total4,838,400bytes; all36cross-process
files/3,628,800bytes exactly equal A1. Each leaf completes nine checked phases.
C6 fills the allocated payload, so inactive-tail checks are vacuous; external
4096-byte guards are nonempty. Full-file equality is checked, not only hashes.

Each leaf records a quantize-plus-projection graph and a20-call batch graph.
An independent host audit verifies all eight actual DOT files: exact directed
Q->projection->Q chains,168kernel nodes/160edges,84quantizers/84projections.
A1/A2 names identify WideIQ4XS C6 original; B1/B2 identify RPW2scalar2.
Four device-pointer bindings remain constant within each process across the
single and batch graphs, and each projection reads the preceding Q8 output.
This is capture metadata, not independent dispatch/ISA or physical-traffic proof.
N/M bind to pinned source/run identity; opaque DOT scalars are not decoded.

After three warm batch replays, each process supplies three timed20-call batch
means. Primary timing is completed barrier-span/20 for quantization PLUS
projection, not isolated IQ4_XS kernel service. All recorded means are retained:

| Process | RPW | Three means, microseconds/call | Process median |
| --- | ---: | --- | ---: |
| A1 |1|68.83855,68.59375,68.30730|68.59375|
| B1 |2|68.29170,68.00520,68.10935|68.10935|
| B2 |2|68.32815,68.12500,68.09375|68.12500|
| A2 |1|68.66665,68.29685,68.52600|68.52600|

Pooled nested medians are68.559875->68.117175us (-0.645713percent), a0.4427us
difference; baseline/candidate ranges are0.5417/0.32295us. Both candidate
process medians are lower, but the ranges overlap and the difference is tiny
and smaller than the baseline batch-mean range. These are three nested observations from only two
processes per variant, not six independent trials. Round/order effects remain;
no significance, p95, model gain, occupancy or bandwidth claim follows.
No new numerical, percent or statistical gate was invented after execution.

VERDICT -> Keep RPW1. Preserve this negative/inconclusive optimization result
without another serving, profiler or standalone retry. The successful numerical
screen also supplies S60's first bounded runtime canary for the invoked public
Q8/IQ4_XS/RPW branches. It does not qualify every common image, byte identity to
old AOT, model logits/quality, concurrency, context or cache behavior. Qualified
frozen152055a6 remains recoverable; HC stays common in future experimental arms
and the independent S55 head flag remainsOFF. Absolute serving goals are open.

Actual post-result reviews resume the same original S52 sessions; full texts
and responding model IDs are retained. Root read both complete responses.

| Reviewer | Elapsed | Finding and recommendation |
| --- | ---: | --- |
| Claude Opus5.5 |23.972s|Counts/identity/numerical closure reconcile; both candidate process medians lower but ranges overlap. Agree keep RPW1/no serving/no retry; prelink canary limited to invoked paths.|
| Claude Fable5.1 |49.426s|Recomputed timing arithmetic matches; small favorable direction remains unresolved. Agree keep RPW1 and bounded canary; old-AOT equality and most common images untested.|

Root excludes unsupported serving-resolution/no-visible-model-gain forecasts,
allocator-determinism attribution, a claim that drift was ruled out, and
duplicate-helper speculation. Pointer equality is a per-process gate only;
launch signatures alone do not prove source hashes. Final-link phase timings
are measured, but second-adapter build savings are not. Earlier CPU-only
RUNTIME_GATE_PENDING records remain intact, superseded by S60's new separate
root-runtime-canary record. No further review or GPU repetition is needed for
the unchanged KEEP_RPW1_NO_SERVING_OR_RETRY decision.

Exact source:
[iq4xs_rpw_leaf_s59.cpp](../strata/flash-next/lowlevel_probe/iq4xs_rpw_leaf_s59.cpp),
[full-byte analyzer](../strata/flash-next/lowlevel_probe/analyze_iq4xs_rpw_s59.py),
[independent DOT audit](../strata/flash-next/lowlevel_probe/audit_iq4xs_dots_s59.py).
Leaf SHA256 is
`9755445400226b7108bf98552411fd603d9489fa2288c42425902a8dfcf61797`.
Native source is81558e95; frozen kernel/core archives aref2b53eb4/34e91c40,
with full pins in raw build records and the S60 report.

Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.
`iq4xs-rpw2-s59-prep/` retains reviewed inputs, original ordinary-build proposal,
actual `build/`, `run-clean/`, and `s59-root-report.json`. The executed alternate
build scripts are in `aot-reuse-s60-execution/`; the unexecuted original ordinary
link is not misrepresented as run. `iq4xs-rpw2-s59-dot-audit.json` is the separate
chain/binding audit, SHA256
`038fec7de504abeb936c2b3f32d9ed0c953f2f63c10d09a41ea25899ce8ff97c`.
Source reviews are in `iq4xs-rpw2-reviews-s59-20261015/`; actual post reviews
belong under `iq4xs-rpw2-postreviews-s59-20261015/`.

Frozen S59:173files/157830430bytes. External `iq4xs-rpw-s59-manifest.json`
SHA256 `a2f52861937b1956f1c3e9eda15cf28086d6416bd5950758af1735f9b75192c5`.
This includes original/source/post reviews, exact source/build/runtime outputs,
independent DOT helper/audit and final root closure; the manifest excludes itself.
