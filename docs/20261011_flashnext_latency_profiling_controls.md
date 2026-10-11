# Latency-first Intel profiling controls

CONFIG -> User steering on 2026-10-11: prioritize request/token latency and
device-idle gaps between operations. A kernel reporting high memory bandwidth
does not establish a good end-to-end critical path. Synthetic inputs and
component substitutions are authorized as explicitly labeled diagnostics.
The selected artifact, original fidelity and full campaign remain unchanged.

COMMAND -> Start with an unchanged OFF/ON bounded source40 host trace, matching
actual prompt/output IDs and normal lifecycle. Existing H36 and decode timing
separate host preparation, submission, waits, commit and streaming. They do
not supply physical device execution or bandwidth measurements. VERIFY_DEBUG
adds readbacks/waits and is excluded from the first normal-path control.

Acquire a separate calibrated device timeline covering actual graph replays,
kernel/copy intervals, queues, contexts and dependencies. Compute busy interval
unions and idle gaps rather than summing overlapping operation durations.
Join host and device clocks explicitly. Measure profiler overhead with matched
OFF/ON or ABBA runs; graph profiling may disable graph optimizations, and tool
instrumentation may add synchronization. Missing coverage stays unobserved.

Use loaded-context synthetic controls to isolate expert source, transfer,
launch, shape or arithmetic costs. Keep allocation sizes, dependency edges and
completion semantics explicit. Resident-only or no-op substitutes change the
workload and must retain synthetic provenance; captured inputs are comparison
targets or conditional diagnostics, never an original-model reference.

RESULT -> Initial chunk64/512 diagnostics improve observed long-prompt prefill,
but warm single-card decode stays near8-9tok/s and long answers differ. Neither
a qualified optimization nor a measurement of the two-card target follows.
VTune2026.2 CLI is present in the current C388 image; actual device-timeline
capability and observer overhead remain untested.

VERDICT -> Optimize latency across the complete dependency chain. Evaluate
launch reduction, fusion, useful batch/tile shapes, GEMV/GEMM and matmul kernels
against measured critical-path cost. Validate promising changes on unchanged
real-model inputs with declared numerical/output gates and natural completion,
TTFT/token-gap/completion distributions and concurrency fairness. Synthetic
speed, peak bandwidth and summed kernel durations cannot qualify the serving
shelf or replace correctness, health, identity and teardown evidence.
