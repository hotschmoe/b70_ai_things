# Lower-overhead profiling after S142: one conditional next discriminator

SOURCE / PRIMARY-DOCUMENT REVIEW ONLY. No build, download/install, profiler
execution, GPU/metrics enumeration, model/binary hash or rawtrace read.

Installed/pinned scope -> Local PTI tree HEAD6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033;
scoped gitstatus clean. tools/unitrace/src/version.h says2.4.0. The already
recorded S136 tool identities are reused (unitrace00fc84e5..., injected
library66ec9ee2...), not rehashed or newly executed. oneprof/build,
samples/ze_metric_streamer/build and sdk/build are absent at inspected roots;
no ready alternative binary is claimed. PTI README explicitly retires oneprof
and auxiliarysamples; unitrace remains maintained. SDK docs test2026.0, not
our exact2026.1; ArcB-series/B580 support does not independently certify every
B70 metric group, permissions, firmware or26.22 backend capability.

Ranking (capability and overhead are different questions):

| Route | Evidence it can provide | Main limitation | Rank |
| --- | --- | --- | --- |
| Native sparse timestamp tags on one in-order queue | Device-clock envelope of one existing private operation/graph bundle; no host collector on every node | Native aspect unqueried; not individual kernel service/occupancy or causal readiness | FIRST conditional discriminator |
| Existing unitrace -d -v with exact kernelinclude; optionally device-only timeline | Actual per-kernel timestamps, names/shapes and static resource fields | Still ZeCollector/event instrumentation; lower trace output is not proved low overhead | Later targeted private measurement |
| Device-level raw hardware metricstream | Time-sampled activity/stalls/throughput IF supported metric exists | New client/build here; sampling resolution/permissions/metric semantics unknown; no automatic kernel identity | PARK pending need |
| unitrace -k/-q or per-kernel PTI metrics scope | Kernel-correlated counters | Kernel attribution re-enables device timing/hooks; not an escape from per-node instrumentation | PARK |
| oneprof/old samples | Legacy rawstream/query examples | Unbuilt here, no longer maintained; no reason to add another tooling stack | PARK |

What source actually establishes

1. Pinned tracer.hL101-112 andL135-173 use the SAME ZeCollector for -d,
   --chrome-device-logging and --chrome-kernel-logging. Device-only output
   removes Chrome host interaction callbacks, but retains kernel collection.
   ze_collector.hL3525-3535 skips selected excluded kernels BEFORE event
   instrumentation; explicit includes can reduce that work. Barriers/copies,
   discovery and other tracing overhead are not thereby guaranteed absent.
   unitrace.ccL519-535 makes -k/-q attribution implicitly enable device timing
   when no device trace option is present. Therefore neither metrics flag nor
   deleting -h/Chromehost is a demonstrated zero-overhead substitute.

2. ze_collector.hL1360-1365 defaults ResetEventOnDevice true. Its event-reset
   hookL5419-5490 can query/append timestamp commands, take locks, and in one
   immediate-list case synchronize a matched event. This is a concrete source
   mechanism worth knowing, NOT proof it caused a particular S136 gap or all
   S142 changes. Do not blindly flip its internal env knob: timestamp validity
   across reused graph events must survive; no run or alternateflag is proposed.

3. unitrace SLM/private/spill/register fields are static kernel properties,
   not achieved occupancy. Hardware utilization/counter samples could address
   occupancy/stalls ONLY with actual metric definitions/units/support. Metric
   streams lack kernel granularity by themselves. Event-based query and
   instructionstall collection may perturb scheduling; their overhead has not
   been measured here. The SDK knownissues explicitly warns of significant CPU
   overhead for frequent short-kernel submissions. Sampling shorter than the
   operation makes attribution coarse; raising sample rate trades more overhead.

4. Core SYCL event profiling of a selected direct kernel can provide actual
   command_start/end, if a private submission returns its event and appropriate
   profiling is enabled. Existing public void wrappers hide that event; attaching
   a collector to every main node is a different cost. Graph event profiling is
   an executable-finalize property plus a profiling queue. Official graphspec
   says it can disable graph optimizations and yields a first-start/last-end
   envelope, not each node or achieved occupancy. Queueprofiling alone does not
   change graphrecording. Do not alter main graph properties speculatively.

5. Better sparse route exists in ACTUAL installed e41 SDK2026.1:
   /opt/intel/oneapi/compiler/2026.1/include/sycl/ext/oneapi/experimental/profiling_tag.hpp
   SHA8bf94c6e3c5006bdacd042c78293894823170cbd7db3930c7448c2124a9a8bd3.
   Actual API is submit_profiling_tag(queue&,...), not a guessed future signature.
   If aspect::ext_oneapi_queue_profiling_tag is true it records an internal tag;
   otherwise it falls back to a barrier ONLY if queueprofiling was enabled, or
   throws. The proposed test MUST guard the native aspect and not silently take
   that fallback. The official extension describes native in-order tags as
   lightweight and permits timing a commandsequence without wholequeueprofiling;
   device support and actual overhead remain unmeasured on these B70s.

ONE next low-cost discriminator, IF root selects it

Add TWO native profiling tags OUTSIDE capture around the NEXT already-required
single-in-order-queue private operation bundle, not a separate fullmodel tracing
campaign. Reuse the existing strict uintfixture first if API integration is
needed, or add the same sparse measurement to the already-needed private GDN
oracle. No new arithmetic/kernelclasses for profiling. Explicit unchanged queue,
context and physicalcard; aspectfalse => recordUNSUPPORTED, drain/normalcleanup,
park WITHOUT globalprofiling fallback, retries or adapter/driverchanges.

In one source-reviewed private build, runtime tagOFF/ON with all existing
strict output/guard/read-only/A-B-A checks and a finite balanced comparison.
Record host submission and completed bundle wall for BOTH arms. In taggedarm,
query timestamps ONLY after the existing completion wait:
 end.command_start minus start.command_end (same devicequeueclock).
Do not use tag.command_submit as a reliable host/device synchronization point.
If multiplequeues are used, require the benchmark's EXISTING explicit join into
this queue before endtag; do not insert new crossqueue ordering just to time it.
Native tags inside graphcapture are not proposed. Only two timestamp events per
measured range, without pernode hostcollector or executablegraphprofiling.

Accept as a profiling METHOD only if strict outputs/normalhealth pass, timestamps
are finite/ordered/nonzero with documented units and the completedwall overhead
is reported rather than assumed negligible. Keep all finite samples; no noise
threshold/significance or speedclaim fromoneABBA. The device interval is an
ENVELOPE including dependency/queue gaps and tag boundaries, not exclusive
kernel service, data starvation, host cause, memory traffic or occupancy.
A successful range probe can guide sparse private stage measurements; it does
not solve all-op model attribution. No main rebuild or framework follows until
this small discriminator actually works. CurrentS144/GDN work takes priority.

Primary references (current docs are interpretation guides, not exactSDK proof):
- [PTI unitrace](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/README.md)
- [PTI SDK metrics](https://intel.github.io/pti-gpu/metrics.html)
- [SYCL sparse profilingtag specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_profiling_tag.asciidoc)
- [SYCL graph properties](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)

VERDICT -> Native sparse tags are the ONE source-supported next discriminator
for private bundle DEVICE envelopes; B70 nativeaspect stillUNKNOWN. No existing
route is established to supply individual service, achieved occupancy AND causal
data readiness at negligible overhead. Keep fullmodel unitrace snapshots short,
separate from clean qualification, and do not treat S136 instrumentation timing
as a production budget. Hardwaremetrics wait for a specific kernel question and
actual supported counters, not another broad tracing/stack campaign.
