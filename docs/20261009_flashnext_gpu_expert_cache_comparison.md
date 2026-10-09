# Flash Next GPU expert cache comparison

This follows the [static screening baseline](20261009_flashnext_intake_and_primitive_qualification.md).
The target remains pinned Unsloth UD-Q4_K_XL. Raw F02 output is under
`/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f02-20261009/`.

## Preregistered candidate

CONFIG -> Same llama.cpp source de7fa0a, runtime39992d, compiler, model files,
FP16 KV, context8192, C1, temperature0/seed42, default warmup, graphs off and
persistent device-code cache off. Change expert placement to all host-resident
routed experts plus32768MiB total GPU cache, split48/52. PLE stays lazy mapped
in host memory. Primary model name remains hotschmoe-dd; secondary alias encodes
cache32768, quant, backend, context, precision and no-MTP method.

All expert weights must be on the host for this arm. Retaining the static GPU
expert placement and adding32GiB of cache would exceed combined VRAM. Do not
silently retain those static overrides.

Source-derived cache estimates, not measured allocations:

| Device | Cache MiB | Group slots |
| --- | ---: | --- |
| SYCL0 | 15,724.22 | Normal4826, layer2:218, layer4:218 |
| SYCL1 | 17,038.09 | Normal4984, higher-precision down group711 |

Banks pool slots across compatible layers. The cached tensor's expert dimension
is the pool's slot count, rather than512. A5120-slot synthetic test therefore
checks indices beyond the model's per-layer expert count without a full-sized
bank allocation.

The global cache batch limit is32. Singleton groups with218 slots conservatively
bypass slot lookup from batch22 because n_tokens * top10 can exceed capacity.
Batch33 bypasses slot lookup everywhere. Later qualification must exercise
transitions around21/22 and32/33, not only single-token hits.

## Source compatibility and remaining gaps

Cache banks have WEIGHTS usage. Current cached expert tensors are views whose
SYCL extra metadata remains null, so lazy persistent reorder returns without
repacking those views. This is not an explicit mutable-buffer safety contract.
Raw slot uploads do not transform layout or reset reorder metadata. Preserve
this source identity and test refills rather than assuming every future
backend view behaves identically.

SYCL counting-sort metadata grows with the expert/slot dimension; no fixed512
limit was found. The SYCL async tensor-copy hook is absent, so generic cache-hit
copies for larger batches use a synchronized fallback. This can affect prefill
performance without making the feature unavailable.

## Prerequisite oracle

COMMAND -> Apply test-only patches0001/0002/0003 to a controlled test source
overlay, leaving external source and server binary unchanged. Extend the existing
counted per-card CPU-reference comparison plan:

- Three I32 GET_ROWS slot maps: [1,512], selecting10/20/320 entries.
- Eight mutable cache-bank fixtures: four selected quant types, batches1/2,
 5120 view slots and one unused guard slot, deliberately small16x256 matrices.
- Five phases per fixture: A upload, A hit, B overwrite, B hit, A restore.
- Check raw target/control/guard bytes, output changes after overwrite, exact
  repeat/restore outputs, unchanged routes, null view extra metadata and WEIGHTS
  usage. Retain the existing5e-4 numerical comparison bound.

Expected full plan:47 fixtures/card, including40 mutable-bank comparison phases
per card. Every declared case and phase must run; unsupported, missing or
zero-case passes fail admission. Both-card and compiled P2P0 post-health required.

The oracle synchronizes and copies GPU inputs for its CPU reference. It checks
layout/refill correctness, not asynchronous callback event ordering. Production
512/top10 dimensions remain covered separately by the earlier tests. A complete
cache application still needs live hit/miss/refill and batch-transition evidence.

Initial F02 result -> Card0 reported39/39 upstream/slot-map cases, but no mutable
bank cases/phases. The strict aggregate check failed and card1 was not launched.
The eight constructor registrations were inside an upstream #if0 block.
The original patch/plan are archived in cache-prerequisites-v1. Registration
was moved outside that block; actual SYCL C++ preprocessing confirms all eight
fixtures survive. Fixture arithmetic/checks are unchanged. Rebuild/rerun pending.

## Candidate and diagnostics gates

After complete prerequisites pass, load the candidate through the owned
controller and run the same twelve repeated checks. Record output hashes
against the static authority; a mismatch cannot become a lossless speed claim.
Preserve failures and complete owned teardown/post-health.

Once the candidate screen passes, capture three useful code/prose/operations
requests capped at256 tokens, with natural completion. Record native token IDs,
top-five probabilities, raw SSE, client receive timestamps and cumulative engine
timings. The client makes no quality or speed verdict. Instrumented logprobs and
SSE gaps cannot by themselves identify a hardware bottleneck.

Cache hit/miss/upload counters are cumulative and printed only at destruction.
The controller captures final logs after stopping the server. Include warmup
and screening requests in the stated counter scope; do not label these totals
per-profile-request statistics. Large-batch copied-expert counts are projection
copies and do not include all uncached prompt-upload bytes.

For attribution, use a separate short traced run. Existing runtime includes
VTune2026.2 xpu-offload and software CPU hotspots collection. First inspect
ID-read/synchronization, CPU LRU work, raw upload and large-batch copy paths.
Keep traced timing separate from release-path performance. Source callbacks
read selected IDs to the host and synchronize each layer, even on hits.

No headline performance result or shelf promotion is authorized by a diagnostic
capture. Balanced cold-request comparisons and broader quality, concurrency,
long-context, teardown and post-health gates remain required.

## Complete prerequisite result

CONFIG -> Corrected registration, same kernel/server source and runtime,
default SYCL optimization enabled and persistent device-code cache disabled.

COMMAND -> cache-prerequisites-v2, full47 fixtures/card.

RESULT -> Card0 47/47, card1 47/47;40 cache-bank phases on each card, exact
counts, no unsupported/missing pass. Both processes exited0, containers removed,
both-card and compiled P2P0 collective post-health passed. Server binary
unchanged by the test-only overlay. No GPU fault signature observed.

VERDICT -> Declared slot-map/layout/refill numerical scope passes. The explicit
synchronization gap remains; this is not proof of all async cache callbacks.

## First 32 GiB cache candidate

CONFIG -> All CPU expert weights,32768MiB total GPU cache, same precision,
warmup and six-case repeated suite. Source/server/runtime remain pinned.

COMMAND -> cache32768-candidate-v1 through owned controller; diagnostic requests
are allowed only after its strict screen passes.

RESULT -> Model loads and all12 individual answer checks pass. Prose repeat
does not match: first response starts "A backup is only reliable...", later
response starts "Testing a backup by performing a restore...". Both have35
prompt tokens and cached_tokens0. The strict screen therefore fails and the
longer diagnostic suite does not start. Normal exit/removal and both-card plus
compiled P2P0 post-health pass; no GPU fault signature.

Actual cache allocations: SYCL0 15,724.22MiB; SYCL1 17,038.09MiB. Static GPU
model buffers shrink to1,994.55/2,611.89MiB because all experts are on the host.

Cumulative destruction counters include warmup and the twelve screening requests:

| Category | Hits | Misses | Reported bytes MiB |
| --- | ---: | ---: | ---: |
| Ubatch <=8 | 73,743 | 12,619 | 38,034.38 uploaded |
| Ubatch >8 | 16,238 | 12,127 | 35,772.46 uploaded |
| Large-batch reuse | 40,464 projection copies | not reported | 40,175.00 copied from GPU cache |

The small-batch hit rate is85.39%; large-batch lookup rate57.25%. These do not
establish a speed gain or identify a stall. Large-batch reuse bytes are device
cache copies, not total host uploads, and omitted uncached transfers must not
be filled in by inference.

VERDICT -> Cache mechanism operates and returns coherent bounded answers, but
does not pass repeated-text determinism. No lossless or speed claim.

Source follow-up finds ID-read synchronization, fills and compute on the same
in-order device queue. Cached views have no reorder extra metadata. Grouped
prefill reorder is skipped under this build's default F32 dynamic precision,
so a proposed stale staging-layout explanation is not established for this arm.
Request state, small arithmetic changes, or cache-copy behavior remain hypotheses.

One isolated exclusion arm sets GGML_SYCL_ENABLE_OPT=0, preserving every other
model/request setting. Its complete primitive plan runs first. Do not combine
more toggles or discard prose if it still differs. The prepared history suite
is original prose, immediate prose repeat, original JSON, then prose again,
with native token IDs and probabilities for first-divergence analysis.

## Software profiler preflight

CPU-only probes in the exact image, without any GPU device mapping, found that
attach/stop targeting container PID1 killed both Python and native yes targets
(exit255, not OOM). The same native program as childPID7 survived collection and
explicit stop, with profiler exit0 and a valid hotspot report.

Future profiling must use Docker init/supervision and discover the actual server
child PID; never attach to PID1. Preserve that topology in matched controls.
Wait for Collection started, run the diagnostic requests, explicitly stop and
wait for detachment, then prove the server remains alive. No host sysctl or
sampling-driver changes were made. CPU profiles quantify on-CPU activity;
off-CPU waits and PCIe routing require separate evidence.

## Optimization-off history controls

CONFIG -> Same pinned runtime/model/server, warmup on, optimization off;
complete47/card primitives and40 mutation phases/card pass before serving.
Native P/P/JSON/P capture follows the preserved twelve-case strict screen.

COMMAND -> cache32768-opt0-history-v2; offline analyze_history.py over its
profile-requests directory. Static comparison: static-opt0-history-control.

RESULT -> Cache optimization-off still fails strict prose repetition. Its
later three diagnostic prose outputs share all35 native output token IDs,
but pre-sampling top-five probabilities differ from token0. Maximum shared
logprob deltas are0.571483 (first/repeat),0.436178 (first/after JSON), and
0.360892 (repeat/after JSON). Cache run stopped normally and both-card plus
compiled P2P0 post-health passed.

Static optimization-off also fails strict prose repetition. In its native
history capture, first and after-JSON share35 IDs; immediate repeat diverges
at token0. Maximum matched-prefix shared logprob deltas are0.153818
(first/repeat),0.653837 (first/after JSON), and0.137632 (repeat/after JSON).
Static lifecycle/post-health is pending at this entry's creation.

All comparisons use exactly equal formatted string prompts, parameters, and
reported prompt progress[0,31,35]. This does not independently prove input
token IDs or every internal kernel shape. Top-five logprobs cannot reconstruct
full logits. The static optimization-off arm differs from the previously
qualified optimization-on static authority and must not replace it.

VERDICT -> Disabling optimization does not restore cache repeatability.
History-sensitive numerical changes also exist in this static control;
cache-specific corruption is not established. Next isolate shared state and
compare warmed optimization-on static history before attributing a cache bug.

A default-off cache byte verifier is prepared as patch004, CPU syntax checked
only. It compares selected bank/slotmap/staging bytes with immutable host
weights. Review requires independent stage coverage and verifier self-tests.
Its synchronization can mask timing races, so even a pass cannot qualify the
uninstrumented cache candidate. No GPU execution of patch004 yet.

Static optimization-off lifecycle follow-up: normal stop/removal, diagnostic
capture completion, both-card and compiled P2P0 post-health all pass. No GPU
fault signature. The failed screen remains failed. An optimization-on static
history comparison is now running under the same owned lifecycle.

## Cache byte-verifier preparation and reset source audit

CONFIG -> Default-off patch004, independent source-copy builder; no baseline
server or external checkout changes. Five stages have separate coverage
counters; actual request deltas are required, warmup-only and zero-hit stages
remain unexercised.

COMMAND -> test_cache_byte_invariants.py, CPU-only fake readback backend under
ASan/UBSan; both changed translation units checked with pinned-image g++
-fsyntax-only. No GPU devices exposed.

RESULT -> Positive/chunked copies, first/last corruption, bounds rejection,
type/dimension/stride failures, default-off path and cross-translation-unit
counter merge pass. Builder prepared but not executed. Instrumented SYCL
verification still untested.

VERDICT -> Diagnostic verifier self-tests pass within CPU scope. Additive
synchronization remains unsuitable as evidence that an async race is absent.

Source audit confirms native diagnostic cache_prompt=false already bypasses
prompt-prefix reuse/restoration. Checkpoint creation remains enabled;
--ctx-checkpoints 0 is a future single-change checkpoint/batch-shape control.
Recurrent removal invalidates metadata rather than clearing all physical
bytes. The next sequence maps to a zero state that the graph SCALE(0) node
initializes before gathering on the same in-order SYCL queue. SCALE uses
ordinary multiplication, which would not sanitize earlier nonfinite values;
initial allocation is explicitly zeroed. No nonfinite value, missing barrier,
or stale mapping is established. These are diagnostic hypotheses only.
