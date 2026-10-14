# Source35 request/token trace coverage audit

CONFIG -> Read-only consumed source35 SDK
`/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/source`.
Exact35 plan SHA82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7;
C113/P30V4 static/segmented1024 host mirrors, native HC/PLE, MTP0,
VERIFY_NO_HOST=1, VERIFY_DEVICE_PLAN=1, normal graph/EAGER absent.
COMMAND -> Read/hash the source below, local pinned x2 reference, and official
SYCL/Level Zero specifications. No model payload, GPU, Docker, runtime, SDK or
reference mutation; only this new document was written.
RESULT -> Concrete instrumentation seams found; no timing samples collected.
VERDICT -> Audit/proposal only. No profile, speed, quality or fullmath PASS.
Root must choose a tracer strategy before any patch/build/runtime experiment.
Any source change needs a new immutable generation, freshly rebuilt ABI targets,
source390/uploadV2 and baseline/source4/health/teardown admission; old source35
proofs cannot silently qualify a patched tracer.

Tags below resolve file/line references in the coverage table. V/G/K/M/S/O hashes
match the reviewed35 patched ledger; P/Q are independently hashed pristine SDK
copy files, not additional members of the63-file patched ledger.

| Tag | SDK-relative file | SHA256 |
| --- | --- | --- |
| V | `sycl/src/core/verify.cpp` | `cdd837fd3dd31873094587e549623bae887765c48d8c4de7f542185f52c4d84f` |
| G | `sycl/src/program/generate.cpp` | `27e52a23debf493b57654a7ca871859b2655c5d8bb11c51be9e473d5ba557a39` |
| K | `sycl/src/kernels/cuda/verify_kernels.dp.cpp` | `1617c1322a1fe7aced3d37cab01a7c46b8f68f8ed07cefc3ab8dd0257bf64a66` |
| M | `sycl/src/core/gguf_expert_source.cpp` | `8aa515675de91cee119631dd5c0ed8ed5b5904e4cdd031b10bbf82c07b5be191` |
| P | `src/kernels/ngram.cpp` | `e609caf1bb0812f366d96a5304c0f98e2864c1c0d22a0bef7b35d818e09debd5` |
| Q | `sycl/include/dpct/device.hpp` | `66776dce10405ad3826e6f19edf23a9811113fa6c62832de56d93d907b827154` |
| S | `serve/server.py` | `3166f70fa013b905adf713dba404a2eb91a4f3381842dafb542af98cf978d241` |
| O | `sycl/include/strata/core/stage_expert_mirror.hpp` | `1cdd4626c6d6602aa5e8d00f62acd6d423363126589b84c6707f46cabe367f8d` |

## Existing critical-path seams

| Source lines | Actual boundary / what a tracer can establish |
| --- | --- |
| S:4680,3243,3291,3550-3581 | HTTP arrival/body, prompt rendering/tokenization, FIFO admission, load and engine invocation. Existing queue_s is rounded; it is not a complete correlated ns timeline. |
| S:1556-1557,1956-1964,5072-5073 | GEN pipe write/flush, pipe receive, SSE serialization/write/flush. Engine token production, API receipt and client receipt are different events. |
| G:10112,10237-10280,11479-11524,11555 | Prompt read_windows versus batched path; T2/T1 run and commit; selected token publication and stdout flush. Correlate accepted/committed positions separately from speculative offered rows. |
| V:2069-2119,1989-2010; P:446-465,507-549 | Capture/first-use versus steady input staging, PLE row lookup/decode and host fences. Mmap row accesses may fault inside decode; Linux prefetch_rows has no mmap-prefetch branch at P:398-420. |
| V:2130-2157,2280-2288 | Optional existing window drain, actual graph submission entry/return and existing completion wait. Submit-return is not device start; wait duration includes outstanding work. Retain the returned replay event instead of discarding it. |
| V:1394-1410,1535-1577; K:2074-2207 | Device residency plan and VRAM/host-mirror grouped work. With nohost, host pool loop V:2182 is skipped. Cache/pool callback timings cannot represent this path. |
| M:214-258; G:4960-4995 | Mirror segments are host USM, populated at startup, with a device pointer table. Device reads of mapped expert bytes occur inside compute; they need not appear as separate memcpy events. Logical expert bytes are not measured PCIe traffic. |
| V:855-864,1637-1643,2281,2387-2389; G:7085-7103 | Intercard handoff uses mapped host residual/bo/inject buffers: first-stage copy kernels, host completion, next stage reads. Attribute these edges separately; source does not establish direct device P2P traffic. |
| V:1727-1738,2391-2402,2592-2626,3611-3620 | Greedy sampling is in the graph. Non-greedy/penalty parameters launch another device sampler then wait; this is not CPU-only sampling. Commit may be separate; LP full-head readback adds memcpy+wait. |

The current SYCL gpu_stamp writes0 (K:2536-2540). VERIFY_PROFILE also disables
shared-expert overlap (V:1347,1577); EAGER timestamps add waits (V:824-831).
Existing trace/profile modes therefore cannot establish the normal graph's
per-stage device costs. Missing stamps are skipped/converted to zero in
V:2022-2052; a new collector must reject missing coverage instead. Cumulative
ms_host is not additive: precollected PLE is charged at V:2102 and again inside
ms_since(t0) at V:2120. Neither those counters nor launch-to-wait wall time is
pure kernel execution.

## Event and clock limits

SYCL graph profiling requires both an appropriate queue and graph finalization
property. The current graph finalization V:1858-1860 passes no profiling property;
Q:678-689 enables queue profiling only conditionally. Official graph profiling
exposes submit, first-node start and last-node end for a replay envelope, not a
complete per-node timeline. It disables optimizations and can lengthen execution.
Capture-time launcher calls/labels run once and cannot timestamp each replay.
A passive UR/Level Zero API trace may identify capture nodes and replay command
list submissions; absent timestamp-enabled replay events it supplies host API
intervals, not per-kernel execution. These conclusions follow the
[graph specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc).

Ordinary SYCL event profiling provides submit/start/end, not OpenCL's complete
queued/submit/start/end ABI. Query only after the existing completion boundary;
profiling queries can block, and capture-event handles are not repeated-replay
samples. Record backend/property/support failures explicitly.
[SYCL event contract](https://github.khronos.org/SYCL_Reference/iface/event.html).

Level Zero kernel timestamps require a timestamp event pool. Record device/context,
API version, property structure type, timer units/resolution and timestamp valid
bits; detect wrap. Global timestamps and kernel timestamps must not be assumed
interchangeable. Use paired host/device calibration per card, bounded by host
clock calls, before merging lanes; synchronized timestamp extensions require
capability verification. Until domains/units/offset uncertainty are proven,
keep host and each GPU on separate lanes and do not subtract across cards.
[Level Zero timestamp contract](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#kernel-timestamp-events).
Online specifications are not proof that this loaded SDK/UMD supports the
proposed mode. No event pool or clock query was executed for this audit.

## Minimal default-off proposal and unresolved coverage

Propose STRATA_TOKEN_TRACE=0/unset as the initial host-only lane. OFF performs
no allocation, event retention, native-handle query, waits or extra device work.
ON records integer monotonic-ns begin/end in a bounded per-thread ring for the
seams above, with request/engine generation, slot generation, stage/card,
window T/group, token position/ID, queue/graph generation and first-use marker.
Export only at existing completion boundaries; record overflow, cancellation,
errors and unmatched intervals as invalid coverage, not zero time. Do not reuse
VERIFY_PROFILE/EAGER or add scope barriers, host tasks or GPU stamp kernels.
Record PLE thread fault-counter deltas around gather separately from its wall
interval; preserve cache warmth and startup/steady-state distinction.

A separate optional graph-envelope lane may retain replay events and query after
existing waits if the pinned runtime supports profiling. It needs a newly
qualified build and matched traceOFF/ON outputs plus clean interleaved timing;
it cannot promise unchanged optimizations or per-node breakdown. A passive
backend tracer is the alternative root should assess for replay node visibility.
Per-node timestamp support, event reuse and hidden runtime batching remain an
isolated leased-oracle question, not an inferred capability.

Own records/events and their queue/context/graph generations until completion.
Copy timestamps before event reset, graph replay or backing ring reuse; never
free native event handles borrowed from SYCL. Keep pointer/segment ownership
bound to the original stage owner (O:26-39,76-101). Capture descriptors identify
command names/shapes, not token-time execution. Calibration must identify the
host clock implementation and device timestamp domain; client-host clocks need
their own correspondence. Preserve dependency edges and compute busy-time unions,
not sums, before assigning a token critical path.

Still unresolved: graph internal scheduling/compute-transfer overlap, direct
mapped PCIe loads versus device-memory stalls, CPU versus GPU page faults and
physical residency, shared-stream completion edges, handoff bus cost, sampler
queueing, logprob readback and stdout/SSE buffering/backpressure. Fault counts
alone cannot identify the blocked interval. Request IDs/selected-expert routing
are not presently a complete timing stream; any future readback changes must be
separately budgeted. No layer, memory tier or queue is declared the bottleneck.

Transfer the local x2 methodology deliberately: barrier-free logical scopes,
owned event records copied after existing synchronize, complete output equality,
negative timing controls, interleaved clean runs and explicit unmeasured stages.
Pinned reference revision e0756d2115dd22af0b7399515de41d9390004422 is recorded in
[the campaign methodology](20261009_flashnext_latency_methodology.md).
Local native runtime lines637-646 retain events;1272-1327 queries after clFinish
and replaces completed records. This is OpenCL evidence, not SYCL support.
Reference file hashes: `AGENT_COORDINATION.md` `e2fb88499536ec7ae581eef07275c8823aa6d509aebcb63bc3d5bb30df520819`;
`native_nvfp4/runtime/nvfp4_runtime.cpp` `82a2b36004240bab371e50220d81e8fdef1e0a9a8e67946186de79abf61bb09b`;
`BENCHMARKS.md` `1f6f52925e264268d5f6a59d383ea6f7311e6798c72841ea6a409bcfed489c46` (unmeasured request-stage ledger at756-767).

Before any actual profile, root should freeze one diagnostic configuration and
output status, preregister the collector schema/budgets, then validate a bounded
CPU-delay negative and a known device-dependency fixture under the lease. Retain
raw records, matched byte/coherence outputs, source4, strict/compiled health and
owned teardown. Full model fidelity remains an independent pending gate.
