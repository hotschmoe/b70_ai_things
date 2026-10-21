# Flash Next C4 decode tests and time budget

Next research item: measure whether MTP helps vLLM at four active streams,
then tune the operation that limits completion time. Budget 2-3 hours for
the decision pass, with a first actionable result in approximately one hour.
Kernel implementation is separate work.

## Measured baseline

Two B70s, identical four cold coding-review prompts of 19798 rendered
tokens, 512 output tokens each, greedy, thinking off. One warmup excluded;
medians of two measured batches. All rates below are aggregate tok/s.

| Recipe | Cold effective PP | Steady C4 TG | Cold output per wall second | Cold batch wall s |
|---|---:|---:|---:|---:|
| Strata Flash-Next IQ3_XXS, MTP depth1 | 1739.5 | 188.5 | 38.84 | 52.74 |
| vLLM Flash-Next IQ3_XXS, MTP1 | 2262.3 | 121.4 | 39.44 | 51.92 |
| vLLM Qwen3.8-27B AutoRound INT4, MTP3 | 722.4 | 161.4 | 16.75 | 122.29 |

Effective PP includes scheduling and overlapping generation until every
stream emits its first content. Steady TG covers only the interval when
all four streams are decoding. Strata's cold run had no such overlap, so
its steady TG uses a separate cached replay at approximately the same 20K
decode contexts. Cached replay walls12.85/13.12s do not replace cold wall
or PP. These are finite recipe measurements, not isolated kernel rates.

Complete configuration, coherence, identity, health and teardown evidence:
[comparison report](../vllm/flashnext/reports/20261021_c4_compare.md),
[machine-readable results](../vllm/flashnext/reports/20261021_c4_compare.json).
Raw root: `/mnt/vm_8tb/b70/results/c4_compare_20261021/`.

## Ordered tests

| Stage | Budget | Decision |
|---|---:|---|
| C4 MTP0 versus MTP1 and critical-path profile | 45-75 min | Does speculation repay target verification and draft cost? |
| Fixed attention splits16 versus8, if attention is material | 20-35 min | Can the existing split setting reduce attention time? |
| Dynamic depth3/2/1 atC1/C2/C3-4 | 45-75 min | Does greater depth as requests finish reduce batch completion tails? |

The previous complete vLLM arm cost522s including load, JIT, health and
measurement. Reuse disk caches and prepared fixtures; preparation and
interpretation account for much of the remaining budget. Graph or
instrumentation repairs can extend the estimate.

1. Freeze the Oct21 vLLM recipe: maxlen131072, maxseq4, chunk12288, prefix
   off, dense FA, route/Q8F/O5c, fixed16 attention splits and placement.
   Compare MTP off (targetT4) with depth1 (targetT8, draftT4). Keep GDN spec
   decode enabled in both arms. Collect exact four-way acceptance, target,
   draft, attention, HC, sampling/commit, CPU gaps and readiness/waits on
   both cards in a separate short diagnostic. Measure normal throughput
   without per-operation host synchronization; check observer overhead.
2. If attention is material, first check width2 causal verification versus
   sequential rows at each fixed split count, including inactive/padded
   rows and19798-20310 histories. Then compare16 versus8 with every other
   setting fixed. If attention is minor, skip this test and investigate
   the measured HC or peer bottleneck. Shared-load width2 attention is a
   later implementation candidate, contingent on this profile.
3. Compare static depth1 with dynamic tiers[[1,1,3],[2,2,2],[3,4,1]], using
   capture sizes[1,2,3,4,6,8] in both arms. Check transitions and rejected
   drafts, then replay the frozen burst and a staged-completion workload.
   Report steady C4 separately from whole-batch wall and completion tails.

C4/depth1 already fills the eight-row fast path. C4/depth2 creates12 rows
and changes several kernel families. The proposed dynamic schedule keeps
target width at most8; it does not increase steady C4 speculation depth.
Async scheduling and the major KREG/Q8/route/EP fusions are already enabled.
Native QSA multirow MTP needs separate transactional and graph integration.

## Execution and acceptance

The daily Strata service owns both GPU leases when running. Stop it before
research, use `bin/cpu-run --quiet` outside `bin/gpu-run`, and retain live
model identity, coherence, normal teardown, per-card and compiled-pair
health checks. Do not restore it between individual arms. Report aggregate
PP/TG, cold wall/output throughput, per-stream TTFT/ITL, acceptance and
critical-path timings with the exact configuration. No speedup or parity
is assumed in advance.

Source audit and detailed component gates:
[decode research](../vllm/flashnext/reports/20261021_decode_parity_research.md).
User direction 2026-10-21: save this as the next research item and restore
Strata for daily serving now; resume these tests when research restarts.
