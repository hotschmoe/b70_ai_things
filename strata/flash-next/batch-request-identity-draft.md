# Source-only draft0022: strict capacity and batch request identity

CONFIG -> Separate overlay on fully compiled20 source
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T112318Z-4uag6j5x/source.
COMMAND -> Reconstruct tracked0022 against verified base hashes; Python syntax and
actual extracted API methods/TLS/identity guards under CPU threaded controls.
RESULT -> PASS source application/hashes, actual TLS last/progress/reuse/segments,
RID/engine/slot tags, unsupported/stale negatives and explicit parallel conflict.
VERDICT -> CPU review draft only. No complete C++ compilation, GPU execution or
2/4/6 serving/cache/logit/fairness qualification.

The source already had thread-local engine.last for batching. This patch extends
that path, rather than replacing it, and makes progress/reuse/first-prefill state
thread-local too. Each actual request resets its own result and segment list,
gets a process-local monotonic RID, and binds engine-generation/slot-generation
metadata. Native strict batch advertises protocol2. DONE/BADM/BT/BDONE tags are
validated before request attribution; detached slot draining rejects stale tags.
BSTOP commands carry RID/slotgen and the native direct/chunk-reader paths share
the same validation. Stale/unidentified cancellation does not stop another slot.

Strict native HC requests2/4/6 slots exactly, group1, no MTP/pipeline. If a stage
fits fewer than requested, startup fails before READY instead of silently serving
fewer streams. Initial configuration rejects all other requested capacities.
Python parallel=N cannot override or coexist with conflicting explicit batch0;
actual announced capacity must match the explicit request. The ordinary legacy
non-strict path retains existing capacity behavior.

The initial API/native request scope is greedy text, neutral penalties and no
control-vector changes or per-token logprobs. Unsupported requests fail before
model state changes. Public defaults/temperature0 and no vector change retain
existing math. Source pin/fresh guards still reject batch, and existing checkpoint
retention/copy, graph cache, stage math, slot sampling and interleaving policy are
not extended here. Request metadata is not evidence of full state correctness.

The added serve/batch_request_identity.py joins artifact_identity PYTHON_SOURCES.
A future concurrent controller must fingerprint that helper in the actual API
manifest. Frozen C1/V6 controllers/recipes are untouched; an old manifest missing
the new dependency must not quietly qualify the new API code.

0021 numerical observer is separately owned. This draft touches batch capacity,
BSlot/protocol and server bookkeeping regions, not record_window/observer math.
Independent combined source reconstruction must apply/check both patches before
any complete new engine build. No old20 source gate can qualify a newly linked
22 generation by reference only.

Remaining increments are mandatory before goal claims:

- Per-slot raw full heads/48-layer samples and evaluated logical spans with
  RID/engine/slot generation labels. First admission alone is insufficient:
  sample a real later batch step and reject absent/partial coverage.
- Automatic repeated/shared-prefix continuity on every stage, with bounded
  root+pin+leaf chain preservation, real live continuation, and pin/fresh guards
  removed only after implementation and full-state qualification. Public pin is
  optional; automatic cache continuity is required without a pin key.
- If penalties are later supported, store each slot's committed history and
  feed per-row penalty/Philox state. If control-vector variants are needed,
  isolate per-row mode or reject mixed active modes. Neither feature is required
  for the declared first greedy-text campaign scope; images/disk persistence
  remain optional and may explicitly reject.
- New exact source/library390/C1 gates, then actual1/2/4/6 simultaneous HTTP/SSE
  output/isolation/cache/cancel/restart/teardown qualification with stable primary
  hotschmoe-dd, actual INFO capacity and measured overlap. Metadata or parallel
  configuration alone never proves concurrent serving.
- Only after correctness, diagnostic-off matched P50/P95 per-stream TTFT/token
  gaps/fairness with an ongoing decode and a useful long prefill, and production
 8192 capacity. Profile host prep, control wait, state copies, lazy graph capture,
  queued/device work and stream send separately; retain safety waits until their
  measured cost and correct replacement are established.

Known draft review limits: full native parser/socket transport tests, restart/
BYIELD/solo migration and detached drain races still require a coherent fixture
before compilation/runtime. No numerical tolerances have been changed.
