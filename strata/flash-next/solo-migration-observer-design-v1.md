# Observer0026 bounded solo-migration producer draft

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Frozen combined24 source, then separate0026. Independent source apply
with25/27/28 is recorded separately. Earlier24 patch, headers, collectors and
receipts remain unchanged. Strict native batch2/4/6/group1/text/greedy/no-MTP,
non-pipelined captured graphs and distinct devices for split stages remain the
existing source contract. Source identity, full SDK and runtime qualification
must be repeated for the new generation.

COMMAND: prepare_solo_migration_observer_overlay_v1.py reconstructs exactly five
source files from checked base hashes and the checked patch. Then run
 test_solo_migration_observer_cpu_v1.py,
 test_solo_migration_collector_cpu_v1.py and
 test_batch_graph_selection_cpu_v1.py.

RESULT: Actual new diagnostic headers execute against ordinary hostg++
ASan/UBSan graph mocks in image39992, with no devices, model files, SYCL flags,
SDK compilation or runtime libraries exposed. Two synthetic RIDs across two
stages publish294 raw vectors (three full48+head samples per RID). Current
identity clearing, stale prior-RID span suppression, source-slot close guards,
changed source-slot marker, poisoned ARM-loss replay, complete row/layer/head
epochs and phase2 main-session markers are checked. Collector controls reject
missing/duplicate/wrong-slot/wrong-generation/wrong-context/missing-body and
incomplete evaluated-prefix evidence. Actual capture-selection prefixes and
actual bkey code run in a separate host cache mock; normal warmed graphs never
satisfy the distinct armed observer cache.

VERDICT: CPU source/helper evidence only. No full SYCL translation unit, actual
GPU source/allocator/graph experiment, native migration math, cache correctness,
concurrent serving or performance qualification is claimed.

## Current identity and source lifecycle

Every begin clears admitting before any early return, including disabled,
unarmed, unknown solo RID or unsupported admission. Unarmed spans emit nothing.
Non-batch prefill/target spans must match the current admitting RID/generation/
slot generation; a copied prior identity cannot label a new unobserved request.
Unarmed selection clears every linked stage's host selector. An ARM loss at dump
poisons the frame; restoring ARM cannot publish its old marker. Explicit discard
requires the owning queue drain before a fresh arm. Producer graph retirement
still precedes snapshot buffers, slot arenas and mirror owners.

Default disabled and enabled-but-unarmed source paths allocate/copy/wait zero
additional diagnostic resources. Existing normal execution/scheduling, RNG,
sampling, cache policy and model tensors are unchanged. New source observers are
opt-in through the existing batch fidelity settings.

## Three sample roles and raw limits

Phase0 is an actual later private-slot batch window with at least two rows.
Phase1 is the existing BGEN prompt-1 admission. Phase2 is the first real main
session T1 window after the same RID's slot was actually completed with a cancel
(the native BSTOP-to-solo path). Phase2 requires completed phase1 and phase0,
matching native engine/request/slot generation and an extended submitted prefix.
Untracked initial solo GEN clears current identity and does not consume quota.
Later phase2 requests after that RID's third sample clear current identity and
remain unobserved; the quota does not add compute/sampling/cache behavior.

The source slot and its generation are carried into the phase2 marker's unused
frame words60/61. Actual main session is slot sentinel6, row0, active mask64,
selected mask1, rows1. This is a diagnostic main-session bit, not a seventh
allocated batch slot. It reaches the existing ordinary main verifier body, not
run_slot_rows with index6. Dynamic token/absolute position and all required
stage/layer/head epochs remain GPU marker checks.

A distinct migration graph array and phase2 immutable roster prevent using a
normal warm graph or phase1 admission graph without actual producer nodes.
Layout descriptors remain80B, frame descriptors512B and per-stage metadata4736B;
raw device snapshot allocation remains17,756,160B across48 layers/head plus
metadata per stage. Graph roster limit16 remains failclosed. Six RIDs with three
samples each write53,268,480B, below the64MiB process output cap.

SBF migration_begin binds the prior native slot-close identity. SBF replay/row
carry historical admission code2 for this new phase; vector phases are
solo_migration_residual and solo_migration_logits_before_sampler. SBF solo_event
is emitted only after actual verifier body return and successful linked-stage
raw publication. It is not a predicted/metadata-only completion. Legacy collector
v2 remains strict and cannot qualify phase2; collector v3 separately requires
all48/head, owning stages, one main row, source close and completed body event,
actual RESUME/read_from/reread_to and full contiguous per-stage evaluated spans.

## Actual qualification remains required

The new native/API harness must exercise same-process unarmed real multi-row
warmup, drain warm requests, then ARM and replay existing row layouts. It must
bind exact incoming native IDs, generated/consumed prefix, PID/incarnation/RID/
slot generations, current source/runtime/placement and private-stage ownership.
Compare each actual selected prefix against matched serial full248320 heads and
all48 residuals with preregistered byte equality; report first divergence and
numeric metrics without widening acceptance after observation.

Same-process graph reuse is required; recreating processes after warmup does not
qualify this transition. Cancellation, stale stop/slot reuse, ARM loss and real
solo migration require actual producer/queue/lifetime evidence before passing.
No output-only migration math classification is permitted. Public fresh/pin and
complete automatic cache continuity need independently qualified25/27 and their
actual ownership/state/spans; this observer does not remove their guards.

Reverse/extra migrations after three samples, duplicate BGEN histories and
additional cache turns are not covered by the initial strict collector. Bounded
later cases must explicitly select their target windows and prove their complete
history rather than infer cache correctness from flags or token equality.
Full original GGUF own-state mathematics and actual2/4/6 serving/fairness/clean
latency remain separate campaign requirements, not synthetic test conclusions.
