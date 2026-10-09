# Actual serving stage mirror ownership trace, source0031

CONFIG -> pristine Strata fb58 plus frozen integrated29 recipe and NEW patch31.
Image39992d70, GGML3cf03257, exact60 final sources/24 header payloads/six Python
files/eight fresh ABI targets. STRATA_MIRROR_OWNER_TRACE is absent/0 by default.
Enabled1 requires serving with stage-owned mirrors; malformed flags and missing stage/device
identity fail explicitly. Source31 does not modify inference equations, source
weight bytes, cache selection, batch scheduling or existing wait/copy/free order.

COMMAND -> python3 strata/flash-next/test_stage_mirror_owner_cpu_v1.py --output
 <NEW-external-CPU-receipt.json>

RESULT -> CPU source-body PASS with ASAN+UBSAN. The test clones pristine pinned
source, verifies every patch and exact reconstructed ledger, compiles the real
StageExpertMirror header plus extracted actual mirror_stage and close bodies,
and supplies bounded mock queues/allocators/byte source only. Compilation uses
host g++ in the pinned image, with networknone, memory512MiB and NO GPU devices
or SDK compiler. OFF emitted0 markers/0 native queries; ON emitted10 native
queries. Both modes made8 actual mock allocations and8 matching frees. Partial
allocation/gather failures, descriptor preflight and concurrent consumer binding
passed;20 chronological trace negatives reject. No real source weight payloads,
SDK build, GPU allocation, native context, graph or serving result was tested.

VERDICT -> READY CPU source; actual serving mirror lifecycle UNQUALIFIED. New
full SDK and all linked oracle/ABI objects must rebuild. Fresh source390 upload
is a separate prerequisite and never proves this process's mirror teardown.
Root will combine independent source30+31 only after source compatibility review.
No consumed29/V9 plan, controller, patch or evidence is changed by this increment.

Each MIRROR_USM record binds PID, stage, logical DPCT device, owner generation,
layer range, full table geometry, native queue context/device, pointer, exact
allocation bytes, allocation generation, role and segment index. Native bridges
are queried only when tracing is enabled, before mirror allocations. The real
factory records owned device_table, host segment or contiguous_host bases only.
Interior expert aliases are never registered as allocation owners. Host USM
ownership is context ownership with a declared native queue device; exclusive
physical device placement of host RAM is not inferred. Device-table allocation
UR device is matched to the recorded native-device bridge.

Source aliases bind after actual publication and retire after stage_alias_ clears
inside actual source close(). The actual caller first declares verifier_expected(main,&ver) or
verifier_expected(main,&stage.ver) BEFORE init. Verifier bindings begin after
actual hits_ mirror association, and retire after the existing verifier executable/batch/observer
graph destruction, immediately before dropping hits_.stage_mirror. The global
serving retirement marker follows actual token/session and every stage verifier
retirement. Source aliases retire separately before the final owning release.

After the existing owning queue wait, release_begin precedes per-pointer
free_begin, the ORIGINAL sycl::free(pointer,q), then free_returned. Pointer fields
are never dereferenced after free. release_returned requires all recorded frees.
Missing graph-retirement observations emit rejecting diagnostics while retaining
normal owning frees, so a missing trace marker does not silently suppress original
cleanup. Native allocator slab retention and stale get_pointer_type metadata are
not treated as leaks or logical-free evidence.

Trace bookkeeping uses fixed ordinary CPU storage:256 allocation descriptors and
16 consumer records per owner. It adds no USM, copies, waits, native-handle
queries or metadata locks when OFF. Enabled segmentation preflights the descriptor
quota BEFORE malloc; at most255 host segments plus the table qualify. No unowned
failed allocation can be hidden by overflowing trace storage. Enabled metadata
locks protect shared owner/consumer markers; timing/race effects still require
matched actual flagOFF/ON qualification and prohibit clean latency claims.

The parser requires complete current-process stage/device/range expectations,
actual context/device bridges, one full48x512 table per serving owner, exact host
payload-byte accounting, contiguous segment indices and both source/verifier
binding histories and exact caller-declared main verifier pointer per stage.
Pipeline/B verifiers and same-device split are excluded by the existing mirror
startup guards; their ownership is not inferred. The CPU negative removes ONE
of the two stages' binding/retirement pairs while keeping its expected declaration
and the other stage's valid pair. It must reject. Every owned allocation must have one successful context-
matched UR free between its begin/returned markers, after complete retirement.
Missing, duplicate, failed, wrong-context, wrong-generation, wrong-stage/device,
wrong-byte/segment/role, empty and missing-verifier traces fail. Ordinary other
engine allocations remain outside this parser's claimed retirement scope.

Example parser command AFTER actual root-owned runtime execution:

```sh
python3 strata/flash-next/audit_stage_mirror_owner_trace_v1.py --log <actual-native-combined.log> --expected-owners <same-process-stage-owner-roster.json> --output <NEW-mirror-owner-audit.json>
```

The roster is [[stage, logical_device, layer_begin, layer_end], ...], for example
[[0,0,0,48]] one-card or [[0,0,0,32],[1,1,32,48]] static two-card. Physical-card
selection is independently bound by the prepared affinity/topology contract.
The lifecycle harness must additionally bind actual new engine/source/parser,
owned terminal container/supervisor status, pre/post strict+compiled health and
current original source identity. This source/parser does not supply those gates.

The initial reviewed31 seven-file CPU snapshot is preserved byte-for-byte in
serving-mirror-owner-v1-review/. The corrected roster freeze supersedes only that
unconsumed CPU source proposal; no actual runtime evidence was altered.

CPU output interface correction before first commit: --output is mandatory and
uses exclusive creation. Missing --output or an existing path fails before
reconstruction/compilation. Default execution never rewrites repository evidence.
Root independently regenerated the corrected CPU receipt; its exact bytes are
retained in stage-mirror-owner-cpu-receipt-v1.json and a separate external copy.
The earlier corrected receipt has no byte-exact saved snapshot, so the source
plan explicitly rebinds to the actual root-produced receipt. This is CPU receipt
regeneration; no runtime proof changed. Patch31, parser and SDK recipe are unchanged.
