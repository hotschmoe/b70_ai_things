# Strata graph capture and current bottlenecks, 2026-10-15

Strata already captures SYCL command graphs for MTP, verification, peer work
and accepted-prefix commit. The next graph opportunity is coarser paired MTP
replay. It is a source-supported candidate, with no measured speed result yet.

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
| Non-vocabulary quantized matrix-vector projections | 10.745 |
| Routed expert gate/up and down | 7.504 |
| HC norm/down/reduce/up | 6.719 |
| Vocabulary head | 4.664 |
| QSA attention-related work | 3.936 |
| Expert-parallel wait kernels | 3.068 |
| GDN-related work | 2.835 |

See the [S95 census](../strata/flash-next/lowlevel_probe/current_c2_profile_s95/analysis/facts.json)
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

First prepare a tiny six-operation fixture comparing six graphs with one
graph. Require fresh mapped A/B/A inputs, exact outputs and guards, preserved
dependencies, and actual Level Zero submission counts. This proves API and
memory behavior only. Production MTP parity and clean serving comparisons
would still be required before claiming a model speed improvement.

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
