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

For each measured token, report wall time, covered GPU busy union, uncovered
intervals, idle-gap distribution and host preparation/submission/wait time.
Report per-card intervals separately: serial layer stages cannot be treated
as concurrent work merely because both devices are allocated. Distinguish
device-memory traffic from host-mirror PCIe traffic and count useful bytes
separately from repeated reads. Bandwidth estimates need an explicit byte
model; allocation inventories do not provide one.

Rank candidates by removable critical-path milliseconds per token. Compare
real replay with dependency-preserving no-op work to isolate scheduling cost;
compare captured real operands in resident and host-mirror placements to
isolate access cost. Sweep actual matrix shapes, chunk sizes and alignment
before choosing GEMV/GEMM tiles. Evaluate fusion by both saved launches and
saved intermediate traffic, including any occupancy or scratch penalty.
Keep output and dependency checks for each conditional control. Run timing
arms without competing model scans, builds or audits, and use repeated
matched runs to distinguish gains from observer overhead and cache warming.

## Existing source controls to measure first

CONFIG -> Read-only inspection of the fresh source41 SDK's SYCL
src/kernels/cuda/shared_expert.dp.cpp on 2026-10-11. This is source evidence;
the active real-model call path and its cost still need a trace.

COMMAND -> Inspect shared_expert_multi and shared_expert before proposing new
fusion. The existing STRATA_FUSED_SWIGLU_Q81 switch defaults ON when absent.
Its OFF branch executes SwiGLU and Q8_1 quantization separately; ON produces
the quantized activation in one operation. The multi-token route also has
lfuse bit2 for a supported same-type gate/up projection pair and accepts an
already-produced input Q8_1 activation. Confirm whether these branches execute
in our decode and prefill paths, rather than counting a default as a gain.

RESULT -> Concrete first comparisons: existing fused/unfused SwiGLU-Q8_1;
supported paired/separate gate-up projections; input-quantization reuse versus
duplication. Keep the original activation rounding and weight formats. Gate/up
types can differ, so paired dispatch must retain its explicit fallback. The
multi-token route accepts only1..8 tokens; do not substitute that for arbitrary
prefill shapes. No actual timing or numerical improvement is claimed here.

VERDICT -> Capture branch and operation counts at the actual model shapes,
then use loaded-context, matched uninstrumented controls to measure saved
launch latency and intermediate traffic. Validate intermediate quantized bytes
and original-reference outputs before accepting a change. Prior single-code
Q8_1 sensitivity makes numerical validation necessary even for a small fusion.

## Request and observer costs to separate

CONFIG -> Read-only inspection of actual source41 generate.cpp, server_intel.py
and host_critical_path_trace.hpp. Existing bounded model controls request
logprobs20. This differs from ordinary serving without log probabilities.

COMMAND -> Inspect the request parser and token-emission branch. The default
req_logprobs is-1. Every nonnegative setting, including logprobs0, copies a
full vocabulary logit row to host and computes a CPU log-sum-exp using double
exp. Positive settings also partially sort vocabulary indices and print the
requested alternatives. The server omits the GEN logprobs field when the
feature is not requested. A disabled control must omit that field; setting0
does not disable the work.

RESULT -> H36 explicitly exposes LPReadback and LPCPU spans. Its records use
CLOCK_MONOTONIC and a bounded memory buffer, without SYCL queries or device
waits. RequestScope records its final event before exporting the JSONL file;
export I/O is outside the recorded request span and can delay the next GEN.
The model actor also leaves an ARM marker when numeric observers are OFF;
the inherited protocol then polls up to1s after DONE for a missing numeric
finish marker. The new host-only control will omit that unused ARM marker.

VERDICT -> Keep LP20 as an explicitly matched diagnostic, then compare a
separately declared log-probability-disabled workload with the same inputs and
output gates. Account for export and protocol time separately. No measured
gain follows from these source observations; H36 cannot establish physical
GPU execution, bandwidth or idle time.
