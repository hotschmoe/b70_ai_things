# 0027 public batch fresh and exact pin source

CONFIG -> frozen25 full-chain source plus separate27; source inputs include
frozen20/21/22-v2/24-v2. Observer26 and package-import28 are independent increments
and must compose before their serving/observation gates. No model payload reads,
SDK compilation or device execution occurred in this work.

COMMAND -> python3 strata/flash-next/test_batch_public_prefix_cpu_v1.py
reconstructs every tracked patch and verifies consumed source hashes. It compiles
actual session_zero, qsa_state_zero, public_reset_main, checkpoint_at and native
request guards against explicit host state/queue mocks with ASan/UBSan in
image39992. Actual API sampling_keys/projection_key/resolve_prefix AST methods are
executed against those reconstructed bytes. No CUDA/SYCL SDK or runtime links.

RESULT -> full main-stage reset and active-peer preservation controls pass.
Old session_zero clears recurrence/conv and PLE/indexer payload, but leaves
indexer block-position and staging metadata. The public wrapper clears those
owned fields and restores resident identity page tables on every stage. Wrong
queue/context, absent metadata and partial reset fail before a successful reset
is reported. Actual cache matching prefers deepest11 normally, chooses5 with
pin5, chooses valid earlier3 with missingpin4, and returns no match for ceiling0.
Actual checkpoint_at captures state evaluated through5 on both stages, with
exact ids1..5 and pin flag, before later rows; the prompt-1 boundary12 also saves.
Native guards and API fields accept fresh and pin0/prompt-1 only in supported
scope, rejecting old unsupported batch mode, invalid ranges/types and capacity.

VERDICT -> CPU source READY for full compiler review, not an actual model
fresh/cache equivalence, concurrency, teardown or speed qualification.

STRATA_BATCH_PUBLIC_PREFIX=1 opts in and requires STRATA_BATCH_FULL_STATE_CHAIN=1
with strict native2/4/6/group1/no-MTP/non-pipelined static/no-borrow scope.
The first public profile requires resident fixed FP16 KV with complete owned
metadata. Streaming/ring/quantized KV requests fail explicitly; they are not
silently treated as fully reset. Fixed resident FP16 is the declared pilot and
8K goal profile; no context or slot count is reduced.

The wire keys are fresh=0|1 and pin=N. API requests use strata_fresh:true|false and
strata_shared_prefix:{"tokens":N}, or the existing resolved strata_prefix field.
Conflicting aliases and malformed fields fail explicitly. N is exact0..prompt-1;
no prompt tokens or templates are rewritten. Pin0 means the actual all-stage
initial state, with zero cached tokens; no nonexistent empty-prefix checkpoint
is invented. Positive pins require cache capacity>=3 and ckpt1 for root/pin/leaf.

Fresh uses an authoritative selection ceiling0. It skips every main/slot/parked
prefix lookup and incoming take, does not mount any root/leaf checkpoint, and
skips outgoing park selection. It resets only the main workspace sessions at all
stages after owning queues/metadata are prevalidated and commits are drained.
Every owned GDN recurrence/conv row, PLE history/token window, QSA KV payload,
indexer tail/dead/pooled/block position, step/status/positions and host staging are
reset. Identity page tables are rebuilt at their existing addresses. Immutable
weights/RoPE tables and private active-slot states are not reset. Queue sources
used for page-table upload survive until its wait. Failure is fatal to this
request/engine path; partial reset cannot produce a fresh success marker.

Fresh may create NEW checkpoints from the ensuing evaluated prompt. An idle
slot's old branch may be parked during later admission as storage preservation,
but it cannot supply fresh input state. Other active conversations remain owned
by their private slots. ckpt0 remains a checkpoint-policy key and is never
substituted for fresh. Each qualifier must assert actual read_from0/per-stage
full evaluation spans and numerical outputs, not just the fresh request flag.

For pinN, the same ceiling bounds all main, slot and parked matching to N.
A deeper hit is ineligible. If an exactN checkpoint exists, its complete staged
state is restored and retained. Otherwise the longest valid earlier state is
mounted, [resume,N) is evaluated and a real all-stage checkpoint is taken at N
before any later prompt rows. A selected live state already at N can be saved
there directly because it is the actual current state. Exact ids/stage parts,
full-chain byte/count bounds, pinned retention and source RID/generation transfer
checks remain from25. Pin cannot silently accept deeper state as if it were N.

The22 native/API RID/engine/slot generation and async STOP rules remain intact.
No new request can reset another active slot or issue its STOP. Observer26 must
supply honest span/request and post-migration raw proofs; frozen24 stale admitting
cannot stand in for that. Package import28 must pass actual native/API startup.

Required next gates: fresh27/26/28 composition, full ABI compiler/link, source390
and complete model identity, then same binary/geometry/warmed-weight same-process
fresh-vs-hit output IDs, full248320 head, all48 residuals, actual state/evaluation
counters. Qualify root/pin/leaf, repeated/shared/divergent suffixes, live assistant
turns, independent sessions, eviction, cancellation and ongoing decode beside
long prefill at1/2/4/6 slots. CPU mock acceptance or cache metadata cannot replace
those model and lifecycle gates.
