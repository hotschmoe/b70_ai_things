# Paired-head serving profile

Date: 2026-10-15. S31; one ON trace, diagnostic only.

CONFIG -> Frozen native152055a6, paired headsON, batch depth2, CLI spec4,
two200k slots, INT8 KV,32GiB/4 parking, all24576 experts7152/17424.
Python includes S30's accounting correction. No engine/source/build/stack
change. Ordinary Intel server entry; no S30 completion observer or
snapshot readback. S30's completed full-cache/functional/health evidence
is retained; this stage changes instrumentation only.

COMMAND -> Launch the native executable through installed unitrace with
kernel/API collection ON before graph capture. One concurrent32/32-token
coding pair, no warmup and no extra generation. Verify identity, registry,
residency, output prefixes and actual paired activation. Pause/stop the
named profiler session, normally stop the server, then per-card and ten
compiled two-rank collective health checks. All pass; no reset.

RESULT -> Both replies match S16 prefixes and each completes32 tokens.
Native9paired/2solo. Stable valid trace270756565B,1722051 events,
SHA256 `eb7ac30d484ea7120198c4f61b89e93be08483016447779f2700462a6ae52985`.
Fresh BOOTTIME-minus-MONOTONIC_RAW correction10627611.213us, selected
from bracketed samples; never reuse this offset for another run.

## Measured replay structure

There are18 full-vocabulary C2 heads: exactly two for each of the nine
native paired windows. Each paired cycle has two input quantizers per
head, the C2 head, private per-row sampling/probability kernels and two
private MTP selections. The main C6 verifier head is separate. Earlier
C1/C3 work belongs to capture/admission/fallback context, not the reviewed
paired cycles. This confirms C2 execution on the paired path; it does not
measure an OFF-to-ON driver-call delta.

Cycles0..4 contain module/event/command-list creation and capture work.
Review cycles5..9 after that construction; conservatively exclude final
candidate10. This selects by API evidence, not by fastest timings. Five
reviewed C6-completion-to-C6-completion spans cover287.335ms, with
median57.699ms. These traced spans do not replace S29's clean benchmark.

| Primary device family | Calls per cycle | Summed service ms per cycle |
| --- | ---: | ---: |
| Paired Q5_K full-vocab C2 heads | 2 | 2.416 |
| Main Q5_K C6 head | 1 | 2.262 |
| Dense projections | 468 | 13.899 |
| Hyper-connections | 447 | 7.199 |
| Primary routed experts | 144 | 7.738 |
| Primary EP waits | 48 | 3.076 |

Each C2 head's median service is1.208ms. Family sums can overlap and are
not an additive request budget. The peer's27.100ms/cycle EP wait is a
separate device timeline; it is not added to the primary's wait or work.
Primary/peer busy unions are233.960/184.590ms over this bounded range,
including spin and instrumentation-sensitive work, not hardware utilization.

The [159-row operation CSV](../strata/flash-next/data-map/20261015_s31_replay_operations.csv)
retains every measured operation family, quant/template-derived labels,
grid/work-group, calls, median/p95/range and summed service per cycle.
It uses full-contained intervals, as the existing summary tool does.
The S22 reference map rows remain unchanged; S31 is a separate profile.

## Host boundaries and waits

Assign API calls once by their start in each cycle's half-open interval;
retain full-contained counts and boundary-crossing records separately.

| API | Start-attributed calls per cycle |
| --- | ---: |
| zeCommandListImmediateAppendCommandListsExp | 9 each |
| zeCommandListAppendMemoryCopy | 2 each |
| zeEventHostSynchronize | 11 each |
| zeCommandListHostSynchronize | 319 / 324 / 334 / 324 / 349 |

The45 AppendCommandListsExp calls take1.596ms total,0.319ms/cycle;
median11.812us per invocation. This is the measured ON count, not an
inferred count of logical SYCL graphs or extra calls versus OFF.

Of1650 HostSynchronize invocations,1625 take at most5us and15 exceed1ms.
Median2.034us; the longest is43.205ms. Chrome arguments are empty and
the CSV supplies no handles, timeouts or return values. Invocation count
does not establish1650 logical barriers; polling is plausible, unproven.
Wait durations overlap GPU execution and are not CPU overhead to subtract
or optimize away wholesale. Start-attributed sums include durations
crossing cycle boundaries and differ from full-contained summaries.

GPU-category EventReset records occur2436 times/cycle,2.245ms summed
service. Their runtime/instrumentation provenance is unresolved; this is
not evidence of removable production GPU work.

Primary-only interval analysis leaves53.377ms without a recorded GPU-op
span over the five cycles,18.58% of their range. This includes all
GPU-op categories, including reset/barrier records. The five largest
repeated gaps run from final private MTP-select to next ep_bump,
3.300-3.941ms, once per cycle. The next five are barrier-to-mapped-input
copy,1.555-1.648ms, with instrumentation-sensitive barrier endpoints.
Top ten account for25.943ms of the uncovered intervals. Overlapping host
AppendCommandListsExp calls are associations, not proof of the cause.
These are absence-of-recorded-activity intervals, not physical GPU idle
or production savings. The raw-domain union differs from the summary's
shifted-epoch union by1.75us due to floating timestamp rounding.

Before tuning these gaps, audit the corresponding host handoff and
establish whether they survive without unitrace. A coarse, buffered
boundary measurement would be more useful than another large traced
benchmark. No such uninstrumented boundary result is claimed here.

## Decisions and next work

The actual sliced-GR-down to reduce gap is median0.209us/p950.438us
across545 same-queue adjacencies. Norm-to-down and reduce-to-up are
similarly sub-microsecond. Small negative gaps are timestamp precision
artifacts. The older standalone9.5us gap does not describe this serving
boundary. Do not run a direct-GR serving arm based on that old gap;
different reduction order and historical sliced advantage remain.

Independent actual Opus5.5 reviewed the supplied trace facts successfully.
It accepts the timing/count limits and recommends scheduler source review.
Root uses the narrower claim that paired C2 execution is observed, without
a counterfactual OFF replacement count; an OFF trace solely to strengthen
wording is unnecessary. Actual Fable5.1's source review hit its turn cap
but left a preliminary scheduling-quantum report. It did not inspect
Prefill::run; root read that body, PLE persistence, callbacks and drains.

The multi-second mixed-prefill pause motivates studying a smaller runtime
scheduling quantum while retaining the12288-token allocation and expert
partition. Existing decode_share only grants decode time after a chunk.
No quantum code or test is introduced here. Callback checkpoint positions
and floating-point grouping can change, so arbitrary bitwise cross-chunk
checkpoint/logit equality is not adopted as a correctness gate. First
establish bounded state continuity before a matched mixed-arrival test.

VERDICT -> Current paired-tail kernels, ON submission counts and operation
service are now measured. No new speed/stability claim, OFF delta, daily
promotion or throughput-target completion. One start/no rebuild/no repeat
200k test/no daily restore; reuse existing parsers and small summaries.
Raw root: cachefix_20261014/head-pair-s31/; see window.json, timebase.json,
analysis-prep/result/final-analysis.json, reviewed-summary/, post-health.json
and the actual model review folders.
