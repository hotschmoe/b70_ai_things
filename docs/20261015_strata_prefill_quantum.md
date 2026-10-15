# Strata prefill scheduling quantum

Date: 2026-10-15. S33; bounded mixed-request state and latency comparison.

CONFIG -> Same frozen native `strata-prefill-quantum-s33`, SHA256
`32ff11a202408c818ca9c6fd37bf2028cb4b65a5bd02098ea7f9a040db5a3803`,
for both arms. HeadpairON/depth2, CLI spec4, two200k slots,32GiB/4 parking,
all24576 experts7152/17424; S30 Python unchanged. Startup allocation stays
12288 tokens. `STRATA_BATCH_PREFILL_QUANTUM=0` uses that width;4096 bounds
each contiguous prefill run when another batch slot is decoding. Both arms
use FIRST0, existing STRATA_TRACE and prefill phase timers. S32 buffered
CPU markers are OFF. This is a logged diagnostic comparison, one workload
per arm, control then ON, without controlled clocks.

The opt-in patch changes only the SYCL program's existing read_part loop.
Eligibility requires text, no pipeline/layer stages and an active batch
slot at entry. It snapshots the width for that part, accepts0 or a positive
64-aligned divisor of the allocated chunk, and preserves allocation,
expert ownership, loans, state callbacks and decode_share. The BYIELD
remaining-history guard still uses the allocation; checks now occur at
the smaller boundaries. No waiting-request BYIELD oracle is claimed.
Malformed values fail at the batched read_part path, not at startup.

COMMAND -> Refresh upstream/PR status; actual independent Opus5.5/Fable5.1
source reviews before a single122.7s offline SYCL build without DRI.
Two leased starts, one identical fixture per arm: retained107-input/
1500-output coding begins, then a fresh32769-token five-anchor request
arrives after first content. Follow with the actual assistant reply and a
new short user turn. Run existing math/reasoning/tool checks, normal stop,
lease exit0, both-card and ten compiled two-rank4x5120 BF16/P2P0 health
checks for each arm. All pass, no reset. No extra8K preflight, third arm,
full200K rerun, soak or daily restoration.

RESULT -> Both decoders exactly match the full retained1500-token S16
reference. All five ordered values are correct on both initial59-token
answers and follow-ups. Initial prompts are fresh. Both returns reuse
32827 tokens and read32 new tokens; the previous API total is32828,
consistent with the engine retaining all but its last emitted token.
The API oracle alone does not inspect saved state bytes or prove RAM
recovery. This stage tests resident prefix continuity across smaller runs;
S30 remains the separate full-length RAM qualification.

| Client observation | Q0 / width12288 | Q4096 |
| --- | ---: | ---: |
| Largest decoder content-event gap during incoming request | 7.824 s | 4.059 s |
| Incoming TTFT | 29.526 s | 37.581 s |
| Incoming completion | 30.990 s | 39.062 s |
| Decoder completion | 41.619 s | 46.612 s |
| Decoder client decode rate, including interruption | 37.022 tok/s | 32.955 tok/s |
| Cached follow-up TTFT | 0.975 s | 0.982 s |
| Decoder content-event gap p50 / p95 | 0.087 / 36.214 ms | 0.099 / 36.357 ms |

SSE events can contain multiple tokens. Each role has one TTFT observation;
do not manufacture per-role p95 estimates or combine the heterogeneous
roles into a latency distribution. All event timestamps, gap distributions,
completion timings, request bodies and hashes remain in the raw rows.
The reduced maximum gap is48.1%, but incoming TTFT grows27.3% and decoder
completion grows12.0%. This is a fairness tradeoff, not a throughput win.

## Native execution and cost buckets

The long batched part covers0..32762; the native final token/window path
accounts for the other seven prompt tokens. Both keep allocation12288.
Control runs12288/12288/8186 with two decode handoffs. ON reports effective
4096/eligible1 and runs seven4096-token pieces plus4090, with seven
handoffs. Coverage is contiguous and monotone, including the unaligned
tail. Two checkpoints are reported for the long read and follow-up in each
arm. Both have zero taken/declined yield messages; waiting-request behavior
and checkpoint branching remain unqualified.

| Native sequential wall accounting | Q0 | Q4096 |
| --- | ---: | ---: |
| Sum of prefill-call walls | 21.117 s | 25.639 s |
| Measured interleaved decode wall | 7.718 s | 11.255 s |
| Whole long batched read | 28.835 s | 36.897 s |

Printed decode walls are observations, not exact budgets: decode_share0.5
uses the preceding call's elapsed time and can overshoot by one window.
Smaller pieces also change how much work occurs before the final piece,
after which this loop grants no decode share. Do not attribute the whole
8.062s increase to kernel work. The two sequential wall categories explain
it within rounded logging precision. Primary event-region timeline sums
20.963/25.489s are separate; never add them to those walls or peer times.

| Primary event-region label, summed across long prefill | Q0 | Q4096 |
| --- | ---: | ---: |
| gather | 2.348 s | 5.174 s |
| dequant | 0.472 s | 1.056 s |
| host grouping | 1.206 s | 1.657 s |
| GEMM gate/up | 0.671 s | 1.087 s |
| GEMM down | 0.414 s | 0.751 s |
| combine | 1.742 s | 1.043 s |
| QSA attention | 6.442 s | 6.287 s |

These are region labels, not isolated kernel service. In particular, the
primary gather interval includes activation gathering and the subsequent
host-side enqueueing of the peer's expert work before primary dequant.
The live SYCL build has no STRATA_PREFILL_MMQ; native weight-group gather
branches are unreachable stubs. Do not choose a gather_native microprobe
from the label alone. The actual EP IPC route gathers activation rows,
dequantizes IQ weights and executes FP16 GEMMs per peer expert. Source and
operation-level measurement must separate that work from primary waiting.
No physical bandwidth, utilization, GPU idle or exact client/device clock
join is established by this experiment.

## Decision and development loop

Both actual reviewers approved the supplied patch for a bounded build and
state oracle. Opus identified the more frequent BYIELD checks; root accepts
them as deliberate opt-in behavior and records their untested waiter scope.
Fable's default-path byte-identity wording is not adopted: the default width
and underlying GPU work remain, while the executable and host code differ.
Reviews finish before the single build, avoiding S32's incremental rebuild.

Both reviewers also recommend stopping the sweep after the observed
tradeoff. Root agrees because no maximum-gap requirement would select a
midpoint and the measured cost buckets give a better next investigation.
Their stronger claims that6144 must interpolate or that a1-2s difference
is noise are unsupported: shape nonlinearity and noise amplitude were not
measured. No6144 run is performed. Preserve Q0;4096 is an experimental
fairness option, without daily or full-context promotion.

A first CPU fixture estimate used one filler repetition, whose marginal
13-token count differed from the steady repeated12-token count. Its fifth
anchor missed the intended neighborhood. Preserve the failed count-only
transcript, amortize over64 repetitions, then prepare the immutable fixture
with21 CPU count calls in0.29s. No generation, GPU rerun or restart was
needed. Approximate32K and honest prefix-rendered anchor counts suffice;
those counts are not exact anchor token IDs in the final chat template.
The same fixture/request bodies are reused byte-for-byte in both arms.

VERDICT -> State continuity and the fairness tradeoff are measured; no
new speed winner, daily promotion or150tok/s claim. Do not rerun full200K
for a nonwinning parameter choice. Next isolate the actual EP expert
submission/gather/dequant path and primary dense/HC work; preserve the
prior rejected asynchronous-submission findings when considering changes.

Patch: [batch-prefill-quantum-s33.patch](../strata/flash-next/patches/batch-prefill-quantum-s33.patch).
Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/prefill-quantum-s33/`.
See inputs.json, model-review-closure.json, tradeoff-review-closure.json,
fixture/, api-comparison.json, analysis/comparison-q0-q4096.json,
q{0,4096}/{state-screen/,functional/,normal-stop.json,post-health.json},
experiment-closure.json and s33-report.json. The full goal remains active.
