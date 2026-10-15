# Strata graph capture and current bottlenecks, 2026-10-15

Strata already captures SYCL command graphs for MTP, verification, peer work
and accepted-prefix commit. S106 now tests coarser paired MTP replay: main
submission counts fall, but the matched serving result is effectively flat.
Keep the new option default OFF and move to the next measured kernel lead.

## Current capture boundaries

At draft depth two, `MtpPair::draft` submits six cached executable graphs on
one in-order pair queue: round A, round B, shared head zero, step A, step B,
shared head one. No host decision occurs between these submissions. Fresh
mapped inputs are staged before them; output polling and PLE prefetch follow.

Re-recording that same kernel sequence into one cached graph could reduce six
submissions to one. Its cache key must include both input row counts and draft
depth. Input freshness, scratch ownership, output publication, queue draining
and the existing prohibition on redrafting after partial execution must hold.
The accepted-prefix commit remains after the host acceptance decision.

Source locations in the runtime checkout are `sycl/src/core/mtp.cpp` around
2076-2265, `sycl/include/strata/core/mtp.hpp`, and
`sycl/src/core/verify.cpp`. These paths are outside this repository.

The instrumented S95 trace has nine Level Zero graph submissions per cycle,
consistent with six paired-draft, two verifier and one commit submissions.
This is not a definitive per-call trace join. Its host submit spans overlap
device work and profiler activity; they are not removable latency budgets.

## Measured work to investigate

The retained S95 census reports the following primary-card median service
times per reviewed batch cycle. These are instrumented family totals, not
per-token latency, additive critical-path budgets or predicted speedups.

| Operation family | Median service, ms |
| --- | ---: |
| All non-vocabulary quantized matrix-vector projections | 12.680 |
| Routed expert gate/up and down | 7.504 |
| HC norm/down/reduce/up | 6.719 |
| Vocabulary head | 4.664 |
| QSA attention-related work | 3.936 |
| Expert-parallel wait kernels | 3.068 |
| GDN-related work | 2.835 |

The native MMVQ subset is 10.745 ms; the broader row also includes the
IQ3_S MMVQ path. See the [extended family census](../strata/flash-next/lowlevel_probe/current_c2_profile_s95/analysis/extended-mmvq-and-cold-limits.json)
and the [S95 census](../strata/flash-next/lowlevel_probe/current_c2_profile_s95/analysis/facts.json)
and [data map](../strata/flash-next/data-map/index.html).
Unknowns include which graph boundaries expose GPU idle time, which waits
are on the request critical path, and how much earlier polling could improve
data readiness. Existing derived records retain selected tail motifs and
family aggregates, insufficient to identify all five draft boundaries.

Long prefill interfering with another stream's decode is also unresolved.
The earlier mixed-load case had a 7.753-second decode gap and 29.564-second
arrival TTFT. Strong isolated kernel throughput does not settle scheduling.

## Agent recommendations and root decision

Actual `claude-opus-5-5` responded in 53.064 seconds. It recommends inspecting
same-card graph-boundary gaps before an expensive serving change, then testing
executable-subgraph composition only after SDK and submission-count proof.
Root accepts the gap question; existing derived records must be checked before
another raw trace parse. Empty GPU gaps are not the only possible benefit:
earlier host polling and lower driver CPU work are also unmeasured candidates.

Actual `claude-fable-5-1` responded in 81.616 seconds. It recommends directly
re-recording the existing operations into one graph, avoiding unproved nested
executable-graph behavior. Root selects this simpler route for a small API
probe. Same-card GPU gaps do not require a CPU/GPU clock join. A single graph
can still partially execute before failure; drain and no-redraft rules remain.

S103 now passes the tiny six-operation fixture: actual graph submissions
18 versus 3 across three replays per mode, each of six kernels executing
three times. Fresh mapped A/B/A inputs, all intermediate outputs and guards
match exactly. The whole graph DOT has six nodes and five ordered edges.
Owned teardown and card-0 health pass. This proves the bounded API mechanism;
it does not test host output polling or MTP state. Production MTP parity
and clean serving comparisons remain required before a model speed claim.
See [S103 evidence](../strata/flash-next/lowlevel_probe/paired_graph_api_s103/README.md).

In parallel, the Q6 C6 projection audit resolved the actual S100 executable's
host metadata, registered kernel and selected device image. Its first 16,384
target instruction bytes exactly match the existing S85 original disassembly;
all eight differing target bytes are after end-of-thread. Execution metadata,
options, constant data and relocations match. This avoids another decoder.
The real instructions retain aligned-load reconstruction compares, selects
and funnels. A dword-aligned scalar-load alternative is worth preparing;
its load cost, register pressure and exact floating-point contraction still
need compiled evidence. No candidate was built or run in this audit.

The latest routed IQ2 reuse experiment passed bounded serving checks but
changed aggregate median throughput only from 103.016 to 103.333 tok/s
(+0.308 percent, one fixed OFF-then-ON comparison). It remains default OFF.
Current streams are roughly 54-56 tok/s against the 150 tok/s target.
See [S100 results](../strata/flash-next/lowlevel_probe/routed_iq2_multi_serving_s100/README.md)
and [independent next-kernel designs](../strata/flash-next/lowlevel_probe/next_kernel_designs_s101/README.md).

## Development-loop decision

Keep the next tests small: use existing traces and exact compiled artifacts,
then a one-card graph fixture before a full engine build and model startup.
The Q6 binding audit took under one second under `bin/cpu-run`, required no
GPU, and reused existing instructions. Do not repeat cache recovery or the
daily-server restoration for these source and API questions. True RAM restore
without recomputation already has bounded S30 evidence; concurrency, decode
and MTP remain the priority.

S103 used a 20-second offline build and a 43-second card-0 window. Since
only exactness and submission counts were assessed, it ran during another
session's CPU build without taking the quiet-CPU lease. Instrumented timing
columns are retained raw and do not support a speed comparison.

## S106: production graph comparison closed

One incremental host build added a default-OFF exact-key graph for paired
draft depth two. It records the same existing operations in order. The kernel
archive and prefill archive remain byte-exact with S100. Both arms use the same
new executable, fixed suffix/lookup/confidence settings, one C2 warm pair and
three fresh 107-prompt/1500-output C2 pairs. This configuration differs from
S100 and is not its speed control.

All four pairs match full text, SSE content, usage, native work/drafts,
windows, paired/solo counts, printed rows and logged captures. Functionals,
normal QUIT, full container-ID/name removal and full pinned GPU/collective
health pass. Aggregate medians are 102.914658 -> 102.976265 tok/s (+0.059862%).
Stream medians are 53.429/55.807 -> 53.456/55.809 tok/s. This is one fixed
OFF-then-ON sequence with three nested pairs, without a resolved serving gain,
independent replication or stability claim.

A separate host-API-only count experiment closes with the same strict
observable work: 24 windows, 21 paired and two solo. Actual graph submissions
fall 231 -> 126; event host synchronizes fall 1195 -> 1090. The reduction of
105 is consistent with five fewer submissions across 21 depth-two paired
calls under the pinned source/configuration bounds. There is no independent
per-call/per-key event mirror or device-idle proof. Command-list host
synchronizes fall 19728 -> 19374; that delta remains unattributed. All seven
other actual API counts match, including 85001 kernel appends in each arm.
These totals include admission, prefill and tail work; they are not a GPU
dispatch census or removable-time budget.

The source audit also corrects a possible census misreading: the steady motif
is two C2 MTP heads plus one C6 verifier head. Whole-profile 24 C6 versus 22
batch C6 calls is consistent with two explicitly logged ordinary prompt T6
reads. Exact GPU-event ownership remains unresolved; there is no evidence of
two extra steady C6 heads to eliminate.

Actual claude-opus-5-5 (17.572s) and claude-fable-5-1 (17.268s) accept keeping
default OFF with no repeat or full-cache campaign. Root does not adopt the
unproved explanation that the pair queue was already fed ahead of the GPU.
The next source review is the separate Q6 aligned-load proposal. An existing
profile may also resolve first-C2-head -> post-head/select -> dependent j1
readiness; check small retained records before another raw trace parse.

The API summaries are about 2KB per arm, replacing a potential roughly 80MB
Chrome API trace per arm. No new kernel decoder, 75GB model hash, daily-server
restore, profiling functionals or flush request was needed. The bounded S30
RAM-recovery result and unmet throughput targets remain unchanged.

[S106 commands, exact records, derived summaries and actual model reviews](../strata/flash-next/lowlevel_probe/paired_whole_graph_serving_s106/README.md).
