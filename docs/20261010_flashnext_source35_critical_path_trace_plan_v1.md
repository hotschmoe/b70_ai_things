# Source35 critical-path trace proposal V1

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Source-only proposal from consumed SDK
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/source,
35-plan SHA82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7.
Preserve current static/segmented mirror/device-plan/nohost normal graphs,
source35/native HC/PLE math, source/registry/weights and all actual evidence.
Current CPU/one/pair four tiny functional outputs match; fullmath, broad
quality and concurrent cache4/6 remain open. Tracing cannot promote these gates.

COMMAND: Read pinned local x2 reference e0756d2115dd22af0b7399515de41d9390004422,
its coordination/benchmark/runtime and Hexagon profiling findings, source35
trace-coverage audit and actual one-card functional timing audit. No SDK,
controller, registry, bin or model mutation; no patch/build/Docker/GPU/runtime
experiment. This new proposal is the only output.

RESULT: Recommend two separately declared stages. H is a bounded host trace
plus passive runtime API observation, with normal queue/graph properties.
D is an OPTIONAL later device-envelope capability experiment. Do not silently
turn D on while interpreting H as unchanged normal-graph execution.

H implementation contract, for ROOT review before a new source generation:

- STRATA_CRITICAL_PATH_TRACE unset/0 is OFF. Parse once before readiness;
  OFF retains the original submit/capture/sampling/control flow, with no trace
  allocation, events, native-handle queries, disk writes or extra device work.
  ON writes integer CLOCK_MONOTONIC ns records to preallocated per-thread
  rings. Initial bounded proposal:32768 records/thread, at most4 native
  writer threads (overflow/thread-limit failure); frontend quotas are frozen
  separately for the declared API2 or serial lane. No per-token heap,
  formatting, locks on shared hot paths, waits, barriers, host tasks or stamp
  kernels. Overflow marks invalid coverage rather than silently dropping work.
- Each record carries schema, producer/process/engine generation, accepted
  request identity, existing rid/slotgen where available, request-local ordinal,
  stage/card binding, absolute input/output position, route/T/accepted rows,
  scope begin/end or instant, sequence, queue/context/graph generation and
  graph family/first-use status. Legacy GEN needs a proven ordered pipe/engine
  request join; don't assume unrelated counters equal. No captured values are
  fed into original reference or native arithmetic.
- Save raw records after the measured completion boundary, before ring reuse,
  or at owned shutdown for the initial bounded four-request experiment.
  Trace export is separately marked and excluded from its request interval;
  it can perturb later requests and must not masquerade as steady serving.
  Retain error/cancel/partial coverage and exceptional exits. One namespace
  per producer; correlate frontend/engine/client only through explicit IDs.
  Before requests, snapshot actual backend, context/queue IDs and in_order/
  profiling properties, device UUID/PCI identity where exposed and driver IDs;
  map logical ordinals ONLY through ROOT's leased physical-card/health receipt.
  Identical B70 names are not physical identity. Own graphs/queues/contexts until
  every referenced record/event is copied; no handle reuse across generations.
  Local host producers use the same named monotonic clock; remote client
  clocks require separate lanes/calibration. UTC and device cycles are not ns.

Exact hooks (G=program/generate.cpp, V=core/verify.cpp, S=serve/server.py;
all under SDK source; file hashes in the earlier coverage audit):

| Hooks | Required records and limits |
| --- | --- |
| S:4680,3243,3291,3550-3581 | HTTP/body, render/tokenize, FIFO admission/engine selection; include queue wait, no inference from rounded queue_s. |
| S:1556-1557,1956-1964,5072-5073 | GEN write/flush, T receive, SSE encode/write/flush; client receive is a separate producer. |
| G:9771,10111-10117,10597-10661 | Request start/reset/checkpoint/refill and each actual short-read/batched span. Prompt timer covers preceding n-1 rows, not final prompt row. |
| V:1791-1796,1858-1887,2072-2078 | Capture cache-hit versus actual record/finalize/queue wait; graph family/T generation. Logged upload uses ue=0 migration stub: NO upload event. |
| V:2078-2120; src/kernels/ngram.cpp:446-465,507-549 | Input staging, row-index calculation and PLE gather/fences; wall interval and optional thread fault counters, not GPU residency attribution. |
| V:2126-2157,2280-2288 | Existing optional device drain, one graph submit entry/return, existing completion wait entry/return. Leave selected drain policy unchanged. |
| V:2595-2632,2688-2711 | Self-commit versus separate commit graph, existing pending-commit wait; record no separate graph for T1 self-commit. |
| V:2381-2404,3611-3620; G:11524-11551 | Greedy in-graph sample versus actual extra sampler; LP full-head memcpy/wait, CPU normalization/top20, T printf/flush. Keep baseline LP20. |
| G:11590-11595,11775-11784 | Decode-clock endpoint, final commit wait, finish/checkpoint bookkeeping, DONE; engine counter excludes some finish work. |

Cache/device compute has an explicit coverage limit: V:1394-1410,1535-1577 and
kernels/cuda/verify_kernels.dp.cpp:2074-2207 build device residency/grouped expert
work. With nohost the CPU pool loop is skipped; its timers are not this path's
cache costs. Host-USM mirrors are populated before READY; mapped expert reads
happen inside kernels and are not necessarily memcpy operations. Logical bytes
are not measured PCIe traffic. Pair handoff V:855-864,1637-1643,2281,2387-2389
uses mapped host residual/bo/inject buffers and completion edges; do not label
this direct device P2P. Initial H covers host preparation/submit/wait edges,
not individual replayed kernels, memory stalls or complete device cache costs.

External passive UR/Level Zero trace, separately named/configured/pinned:
observe program/module build and kernel creation, command-buffer/list finalize,
queue submission, explicit memory copy and queue/event/fence waits. ROOT must
first verify the installed tracing tool/ABI and archive its version/config/log
and actual loaded loader/adapter/UMD/library IDs. Capture-time nodes are not
per-replay execution events. Host API spans can test first-use JIT/finalization
hypotheses; they cannot supply missing kernel start/end. Match graph/queue/
context generations to H; preserve unknown mappings and missing coverage.
Never infer invisible copies or GPU page faults from an absent API call.

D event/clock contract, only after an isolated ROOT-leased capability oracle:
retain the sycl::event returned by each actual replay/explicit transfer/commit
ON, without executing the call twice. Keep an owned reference until the
existing corresponding completion boundary, extract/copy before event reset,
queue/context destruction or replay/ring reuse, then release the reference.
Never destroy a runtime-owned native handle. Capture events aren't replay
samples. Ordinary events can expose submit/start/end if supported; graph events
are envelopes, not per-node timelines. The current dpct queue constructor
makes profiling conditional (device.hpp:678-689), and V:1858 finalization adds
no profiling property. Record effective properties/capabilities and any query
failure. Adding profiling properties may disable graph optimizations: D is a
new configuration, not a transparent performance result. Query only after
existing waits; no additional per-token synchronization to obtain timestamps.

If Level Zero timestamp events are needed, prove pool flags, timer units,
global/kernel valid bits and wrap handling on each physical card. Pair host
bounds around device-clock calibration and retain uncertainty; do not subtract
cross-card or host/device timestamps until clock domains match. Zero gpu_stamp
in verify_kernels.dp.cpp:2536-2540 is unsupported, not a measured zero. Missing
requested device records fail D coverage; H-only exports say not_requested,
never complete_device_profile. Existing VERIFY_PROFILE changes overlap and
EAGER adds waits; neither is an accepted normal-path substitute.

OFF/ON admission and next bounded experiment:

1. Freeze exact new source/patch/headers, compiler/image, rebuilt ABI/ELF and
   Python/library identities; default OFF normal routing/geometry unchanged.
   Old source35/C113 proof does not qualify a patched tracer. A passive-only
   run keeps old binaries but separately binds tracer/layer configuration.
2. CPU controls reject lost/duplicate/out-of-order scopes, wrong request/card/
   graph generation, overflow, absent completion, stale/reused events and
   invalid timestamp/wrap data; an intentional dropped T must fail coverage.
3. ROOT runs matched OFF/ON with the accepted two prompts/repeats, LP20,
   fresh=1, exact IDs and initial-state/cache policy. Require complete emitted
   IDs/natural finish plus full-vocabulary head-byte equality at declared
   witnesses, using an existing correctness capture lane in BOTH arms.
   Top20 text alone is insufficient. This preserves relative native outputs,
   not independent full48 mathematical correctness.
4. Separate process startup/first use from warm semantic-fresh requests;
   balanced same-build rounds and distributions, not one old/new mean. Repeat
   traced/untraced controls and retain raw runs, health, known pages/full4,
   identity and owned teardown. Keep H coverage/perturbation conclusions
   distinct from any future serving latency qualification.

VERDICT: H is the minimal useful next tracer: actual functional data already
shows lazy first-request T2/T1 capture and repeat timing differences, but no
causal JIT/residency attribution. D remains unqualified until actual capability
and clock/ownership evidence. LP0 is a separately preregistered future control,
never a silent reinterpretation of established GEN20 data. No performance,
stability, fullmath/broad quality or concurrency4/6 PASS follows this plan.

Pinned local evidence SHA256:

```
AGENT_COORDINATION.md e2fb88499536ec7ae581eef07275c8823aa6d509aebcb63bc3d5bb30df520819
BENCHMARKS.md 1f6f52925e264268d5f6a59d383ea6f7311e6798c72841ea6a409bcfed489c46
native_nvfp4/runtime/README.md 55a3c4db80987e05663c0cbdbb70c9829ac797bc51d73989990de28744793554
npu_campaign/OPS650_LOG.md ed5dbb225f632c70309113fc6778e25b53bd5242e1b773dcef842116008bdd71
npu_campaign/PREFILL2_LOG.md 5059e6fdf92bf2769752ee810d86f9f991be68a616f1d5c5119492373e09fa99
npu_campaign/STATUS.md c085279fdaa33864ac9066c86b8ede04de3a4d509ae16a4ef2f51e769c4d9ab9
SDK/sycl/include/dpct/graph.hpp 9933c09306a7a91bb242fa913694541fc4166bcf6d5a1a2a81ee9dac91b05f31
SDK/sycl/include/dpct/device.hpp 66776dce10405ad3826e6f19edf23a9811113fa6c62832de56d93d907b827154
SDK/serve/server.py 3166f70fa013b905adf713dba404a2eb91a4f3381842dafb542af98cf978d241
```

B70 source35 coverage audit SHA31afbff0fdb3739df05908201b3484478e8794148a55e3be7be5afa9700f35de;
functional timing audit SHA39ac97842d54779966a13806e7e49ab867932afaa1018602889a10af2f1258ac.
Transfer x2 BENCHMARKS:777-785 barrier-free/three-replay/median/distribution
method and runtime README:28-34 record-before-next-sync ownership. Coordination
requires same-build repeats and interleaved comparisons; Hexagon attribution
and model-specific thresholds are not B70 acceptance values.
