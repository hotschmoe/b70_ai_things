# Native HC standalone write numerical preregistration

CONFIG -> Source patches0001 through0004, selected D=10240/N=2560/H=4.
Original fixture files remain unchanged. Build native_hc_write_gpu_fixture.cpp
against current overlay headers plus original source headers, primitive and
composition sources, with -DSTRATA_SYCL_Q8_HC_BUILT=1. The build includes the
actual native_hc_write_rows helper from native_hc_dispatch.hpp.
Runtime image39992d70 is retained. GPU execution belongs to the parent under
bin/gpu-run with an explicit one-card device pin, health, teardown and post-health.

COMMAND -> Run the standalone executable with an optional JSONL receipt path.
Coverage is21 cases: rows1/2/4/8/9/16/17 crossed with tiny, mixed and saturated
injection profiles. The row counts9/16/17 explicitly cross the shared helper's
8-row chunk boundaries. Device queues are in-order. CPU_REFERENCE_ONLY mode
compiles with -DHC_FIXTURE_CPU_ONLY and does not discover or execute SYCL.

RESULT -> The preregistered CPU FP64 reference is:

R_out[t,c,d] = original_F32_R[t,c,d]
             + original_F32_bo[t,d] * 2 * sigmoid(original_F32_inject[t,c]/4).

The independent reference uses original F32 values promoted to FP64 and FP64
exp/sigmoid arithmetic. It does not reuse either GPU write implementation.
Every case must be finite, with NMSE <=1e-6 and normalized maximum error <=1e-4.
Normalized maximum error is max_abs_error / max(1e-6,max_abs_reference).
These retain the previously registered projection/composition gates without an
observed-result adjustment. Profiles use injection magnitudes up to3e-6, about4,
and +/-96, covering the linear neighborhood and both finite saturation tails.
Residual and block-output values vary by row and element, making incorrect chunk
or token offsets observable.

Require exact output bytes for manual <=8-row chunks versus concatenated single
writes, versus the actual native_hc_write_rows shared wrapper, and versus the
pending R_out of actual hc_native_read_f32 calls. Pending composition is chunked
with the same contiguous row offsets and receives original Q8 down/up, F32 norm,
and no new injection projection. Its remaining workspaces are guarded.
Identical-input repeat bytes must match after another independent buffer set
runs. Original bo/injection bytes and pending-read original R/weight bytes must
remain unchanged. All allocations have128-byte leading/trailing guards.
Standalone R is the explicitly mutable output; each rerun restores its original
source image before computing it again.

The public standalone launcher must reject15 combined cases before submission:
three null source/output pointers, three insufficient capacities, row counts0,
-1 and9, an out-of-order queue, plus five wrapper null/zero/negative-row cases.
The wrapper has no capacity argument; its model callers own the contiguous array
bounds, while the public primitive checks each explicit capacity. Rejected calls
must leave all bytes unchanged. A corrupted host guard image must fail the guard
checker. Each numeric oracle must reject a deliberate1-percent-scale output
perturbation. No rejection case uses inaccessible memory as an accepted input.

VERDICT -> A GPU pass requires21 unique case rows and GPU_WRITE_NUMERICAL summary,
zero failures,15 rejection cases,21 numerical negative controls and a successful
guard negative control. Each case must meet its numerical gates and exact repeat,
single, wrapper, pending-write, guard and unchanged-input gates. CPU mode is
fixture-reference validation only; GPU behavior and model route coverage remain
unproved until leased execution and independent receipt validation complete.
This fixture does not establish complete prefill/verifier integration, model
coherence, serving speed, request history isolation, or two-device qualification.
