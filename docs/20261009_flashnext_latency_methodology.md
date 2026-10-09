# Flash-Next latency investigation from the Hexagon research

Date: 2026-10-09 UTC. Status: CPU source review and proposed experiment plan.
No new GPU run, performance claim, runtime change or weight download accompanies
this document. Current model correctness gates remain independent and mandatory.

## Reference identity and discovery

CONFIG -> User-specified current repository `hotschmoe/x2-nvfp4-lab`, read-only
CPU research. The repository name predates its current Hexagon llama.cpp work.
COMMAND -> `gh repo clone hotschmoe/x2-nvfp4-lab /mnt/vm_8tb/b70/research/x2-nvfp4-lab -- --depth 1`;
read source and coordination rules; record `git rev-parse HEAD`.
RESULT -> Reference checkout at `/mnt/vm_8tb/b70/research/x2-nvfp4-lab`, revision
`e0756d2115dd22af0b7399515de41d9390004422`. No AGENTS.md found; read the repository's
AGENT_COORDINATION.md. No reference code, model weights or device workloads ran.
VERDICT -> This is the user's authoritative newer reference. This document uses
its Hexagon llama.cpp campaign and its separately identified native OpenCL trace
work. Old Specula is not substituted for the requested methodology.

Pinned sources (all at the revision above):

- [Campaign correctness and interleaved measurements](https://github.com/hotschmoe/x2-nvfp4-lab/blob/e0756d2115dd22af0b7399515de41d9390004422/AGENT_COORDINATION.md)
- [Hexagon per-op framework attribution and negative control](https://github.com/hotschmoe/x2-nvfp4-lab/blob/e0756d2115dd22af0b7399515de41d9390004422/npu_campaign/OPS650_LOG.md)
- [Hexagon stage-duplication critical-path probes](https://github.com/hotschmoe/x2-nvfp4-lab/blob/e0756d2115dd22af0b7399515de41d9390004422/npu_campaign/PREFILL2_LOG.md)
- [Counter observability, wraparound, replay and profiling caveats](https://github.com/hotschmoe/x2-nvfp4-lab/blob/e0756d2115dd22af0b7399515de41d9390004422/npu_campaign/STATUS.md)
- [Barrier-free full-token traces and unmeasured request stages](https://github.com/hotschmoe/x2-nvfp4-lab/blob/e0756d2115dd22af0b7399515de41d9390004422/BENCHMARKS.md)
- [Native trace ABI and event ownership](https://github.com/hotschmoe/x2-nvfp4-lab/blob/e0756d2115dd22af0b7399515de41d9390004422/native_nvfp4/runtime/README.md)

## What transfers from the actual work

AGENT_COORDINATION.md requires same-build repeats before blaming a candidate,
full-vocabulary exact logit rows for pure performance changes, prompt lengths
around microbatch boundaries, frozen per-op CPU/device inputs, and a negative
control that must fail. Find the first divergent tensor rather than guessing
from final text. Approximate changes require a CPU quality reference, not just
agreement with the accelerated baseline. Its model-specific PPL constants,
NMSE threshold and Hexagon alignment requirements are not B70 acceptance values.
Freeze the appropriate B70 lane and oracle tolerances before measuring. Current
Flash-Next history-dependent drift therefore blocks equivalence and promotion;
diagnostic profiling remains allowed with that failure clearly retained.

OPS650_LOG.md (September 29, 21:30 through September 30, 00:05) demonstrates the
needed latency reasoning. A whole token used one DSP queue message, so hundreds
of short operations were not hundreds of FastRPC calls. The device operation
interval included framework preparation and cache bookkeeping; nearly absent
inter-op gaps did not prove absent overhead. Segment timers localized a large
cost to dirty-range merging. An incremental merge candidate then passed
128-row full-vocabulary equality, and deliberately swapping a convolution tap
caused the correctness gate to fail. This establishes a useful investigation
pattern, not evidence that the B70 has the same cache implementation or cost.

PREFILL2_LOG.md (September 29, 09:40 and 10:10) adds a complementary perturbation:
duplicate one idempotent pipeline stage while retaining the correct result,
then measure the change in whole-pass latency using interleaved controls. The
reported dequant/HMX/DMA/gather/scatter probes had different marginal costs even
where nominal utilization was low. Such deltas measure sensitivity at that
operating point; they are not an additive or unique decomposition of a changing
pipeline. Some probes had only one run, explicitly limiting evidence. For B70,
use a bounded, default-off duplication or delay only where ownership and output
equality are proven; avoid doubling cache updates, recurrent state writes or
other non-idempotent operations. Never install diagnostic work inflation.

The same repository's native OpenCL work supplies a concrete tracing pattern:
logical scopes plus queued/submit/start/end events, without inserting scope
barriers or changing graph order. BENCHMARKS.md records three exact replays,
keeps cross-sample distributions, and chooses the trace nearest the median
kernel total for its illustrative timeline. Each traced replay matched
untraced logits. The trace ABI replaces completed records at the next
synchronize, so the consumer must copy them before another measured graph.
This is separate from the Hexagon execution path; transfer its instrumentation
principle deliberately to SYCL/Level Zero, not its API or device conclusions.

The sources also identify important limits. BENCHMARKS.md explicitly lacks a
complete HTTP/admission/tokenization/sampler/streaming timeline. STATUS.md warns
that a compiled timer without an observable return path supplies no evidence;
missing counters are not zero. Old 32-bit timestamp wrap caused invalid negative
gaps. Its profiling mode can disable replay, and profiling can change device
state. HMX issue/export spans are not automatically pure HMX execution. Therefore
validate timestamp units, overflow, collector version, record coverage and
interval semantics, then compare matched clean runs. Do not call a profile-mode
result a production latency improvement.

Finally, the coordination rules report 5-10 percent session timing drift and
require interleaved comparisons over several rounds. Adopt balanced A/B or ABBA
ordering with fixed configuration rather than comparing yesterday's baseline.
The transferable target is reduced end-to-end blocking latency with correct
outputs, not a desired memory-bandwidth utilization number.

## B70 request and event ledger

Run all GPU work through the existing leased controller. Keep source/image,
model/quantization/PLE identity, topology, precision, graph mode, MTP, host thread
counts, context capacity and memory settings in each receipt. Preserve the
failed strict-screen status when collecting diagnostic data.

For each request, record monotonic client send, server arrival, admission,
prefill start/end, first generated token, each decoded token, enqueue-to-client,
and completion. Record request ID, token position, actual prompt token IDs,
output token IDs, batch size, microbatch size, selected experts and device.

For each relevant host operation, record begin/end and correlation ID:

- routing-ID readback and synchronization;
- cache lookup/LRU planning, fill selection and slot-map update;
- host tensor preparation, quantized row packing and page-fault handling;
- command submission, explicit waits, sampling and response serialization.

For each device command, record correlated queue/device identity, dependency
edges, transfer direction/bytes or kernel shape, and device start/end timestamps.
Collect timestamps without adding per-operation waits. Preserve event and buffer
ownership until completion; copy records before their backing storage is reused.
Calibrate host/device clock correspondence before subtracting cross-domain
timestamps. Until calibrated, show separate lanes; do not invent queue-wait time
by subtracting unrelated clocks. Event creation/completion timestamps and their
measurement overhead belong in the receipt. Missing, wrapped, dropped or invalid
records must fail trace validation rather than be converted to zero-duration work.

Build a dependency timeline from request input through its next observable token.
Attribute only non-overlapped blocking intervals to that token's critical path.
Report host work, waiting, device compute, transfers, inter-card synchronization,
and unexplained residual separately. Summed device durations can exceed wall time;
report busy-time union and overlap explicitly. Queue submission is not completion.

Existing native SSE captures contain cumulative engine timings, probabilities
and token IDs. Their client timestamps measure receipt of events, potentially
buffered. Use them as labeled event gaps until server token timestamps and
correlation are captured. VTune software hotspots describe CPU work; they alone
cannot distinguish PCIe latency, blocked GPU dependencies or critical-path waits.

## Workload and user-facing metrics

Concurrency is limited to 1, 2, 4 and 6 users. Use the frozen useful code, prose
and operations prompts with natural completion and declared output caps. Add a
fixed mixed short/long prompt workload to expose prefill interference with an
already decoding request. Keep exact prompts, arrival schedules, context lengths
and caps matched between control and candidate. No hidden repeated-prefix reuse;
explicit reuse experiments are a separate lane with cache counts disclosed.

For each user and workload class, report:

- TTFT P50/P95, including queueing; separately show admission and prefill time.
- Inter-token latency P50/P95 using verified token events, plus maximum stall.
- Completion latency, generated length, achieved per-stream output rate and
  aggregate throughput over one common wall window.
- Per-stream distributions, slowest stream, minimum/median rate ratio, and each
  stream's slowdown against its matching solo request. Do not present aggregate
  throughput divided by user count as a measured individual rate.
- Failures, cancellations, incomplete responses and correctness results; never
  silently drop a slow or failed stream from statistics.

State the quantile estimator and sample counts. A four-request history probe
cannot support a reliable P95 service claim. Proposed service characterization:
at least 100 completed requests per concurrency cell across three independent
server launches, with class/per-stream counts reported. Initial bounded screens
may be much smaller but must remain exploratory, with all raw samples retained.
Do not pool thousands of token gaps as if they were independent request trials;
use request/launch grouping for uncertainty and paired comparisons.

## Controlled memory-tier experiments

Use one changed factor per arm and retain the original model bytes and arithmetic
where the comparison claims equivalence. Memory budgets must come from measured
host/VRAM availability with startup and transient headroom, not card capacities.

1. Static expert placement versus dynamic expert caching at the admitted cache
   budget. At equal prompt/step geometry, measure critical-path routing readback,
   planning, miss upload, hit-copy and compute intervals. Separate initialization,
   first-use and steady operation. Existing lifetime cache counters cannot supply
   per-request latency attribution.
2. Within caching, vary only cache budget across feasible admitted sizes. Record
   per-layer misses, miss bytes, eviction and consecutive-row copy counts. Decide
   whether lower miss traffic actually shortens the critical path; a higher hit
   rate alone is not a win.
3. Within a fixed placement, compare admitted host preparation/copy mechanisms
   such as reusable pinned buffers versus the existing path. Preserve buffer
   ownership until transfer completion. Diagnose fixed transfer latency and host
   submission gaps as well as bytes per second; do not optimize for link
   saturation as an end in itself.
4. PLE and model-page behavior: distinguish initial faults, repeated resident
   accesses and bounded-cache misses using fault/I/O counters and correlated
   intervals. Do not drop system caches or alter swap implicitly. NVMe streaming
   is a separate, explicitly configured arm; filesystem cache warmth is recorded.
5. Compare overlap only after byte/state correctness: issue earlier permissible
   copies without changing routing results, then prove whether their completion
   moves off the token's critical path. Additional queued work that delays another
   stream is a latency/fairness regression even if aggregate bandwidth rises.

## Experiment acceptance and next actions

CONFIG -> Freeze one qualified or explicitly unqualified diagnostic control,
workload manifest, placement and source identities; preregister the changed
factor and expected blocking interval reduction.
COMMAND -> Run a bounded instrumented control/candidate pair, then matched clean
runs without heavyweight profiling. Use identical useful requests and retain
first-use observations separately. Repeat balanced control/candidate ordering
when screening results justify further GPU work.
RESULT -> Preserve outputs, probability drift on shared prefixes, event traces,
clock calibration, memory peaks, health and teardown. Reconcile critical-path
intervals against token/request wall time and label unresolved residual.
VERDICT -> A faster invalid response is not a promoted improvement. Current
history-dependent output/logit drift remains a separate blocker. A profiler-only
win, unmatched geometry or traffic reduction without lower matched latency is
inconclusive. Promote only after correctness, deterministic/fresh-server repeats,
concurrent coherence, clean teardown and post-health pass.

Before attributing an optimization, validate the collector with a known bounded
CPU delay and a known device dependency in isolated fixtures; confirm trace-on
versus trace-off output equivalence and measure collector perturbation. Retain
raw per-request records and an explicit event schema with timestamp domains.

The next implementation task is a correlation ledger around the observed MoE
cache callbacks and host sampling path, with matching device transfer/kernel
intervals. Use it first at one user; expand to two, four and six only after the
single-user trace and controller lifecycle are trustworthy. This is an
implementation proposal, not evidence that such tracing already exists.
