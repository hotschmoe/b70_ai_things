# Paired prefill cooperation candidate V1

CONFIG -> Source-only candidate against actual source52 SDK
strata-native-hc-engine-20261012T012115Z-_kwur0ww. Original UD weights,
chunk512 arithmetic, DQ/GEMM/router/expert/scatter operations and checkpoint
cuts are unchanged. STRATA_BATCH_PREFILL_LAYER_COOPERATE=0 is the default.
Initial runtime probe scope is strict native2 with both full-chain/public cache
gates, two stages, noBorrow, static placement, noMTP, no pipeline, no peer,
no helper, no CPU expert sharing and no fused prompt layout. Native2 is the
first probe scope, not an identified arithmetic dependency on two rows.

COMMAND -> Apply candidate-paired-prefill-cooperative-core-v1.patch and
candidate-paired-prefill-cooperative-scheduler-v1.patch to a NEW source52
snapshot. Copy candidate_cooperative_header_v1.hpp to
include/strata/prefill/cooperative.hpp. The core patch includes cancellation
V2 and owner-extents V1: do not apply those prerequisite patches again.
Rebuild every affected header consumer and all eight engine ELFs from this
new snapshot. The main-only source57 incremental recipe is insufficient.
No compiler, GPU or model execution has been performed by the author.

RESULT -> Both stages use one CPU ticket. At the ordinary host-router path's
existing compute queue wait, a stage may cooperate only after ple_next has
been consumed. Stage0 calls the original owner's batch_step directly.
Stage1 parks its worker; owner drain_pipeline pumps that same batch_step and
ACKs the same worker stack. There is no added GPU barrier, chunk split,
workspace or new decode arithmetic. Issuer prefetch uses its private ring
while compute is parked. Shared PleTable state is never serviced while a
prefill PLE gather remains outstanding; the one-chunk gate prevents a future
for the next chunk from being launched. Helper/peer/CPU paths are rejected.

Exact retained Prefill allocation lengths cover Alloc's owned takes and direct
token/identity device allocations; incomplete/borrowed descriptors reject.
The owner compares them with main-session memory, Verifier normal/batch
arenas and every SlotSessionArena allocation before installing a ticket.
This virtual-range check complements source ownership; it does not replace
physical USM/page guards or numerical runtime qualification.

The callback credits the original decode-share fraction of prefill wall time,
excluding ticket pause time, and subtracts actual prior decode service.
Post-chunk service uses the remaining allowance and skips nonpositive
allowance. A window may overshoot the allowance, as in the existing scheduler;
that actual service is debited before the next opportunity. Pending
STOP/BSTOP/BYIELD is observed without dequeuing or publishing partial state.
The original complete-chunk paths retain control and checkpoint publication.

Source read/copy failure publishes failed readiness, never zero-filled ready
or synthetic a_issued progress. Issuer exceptions publish a real error before
join. Host-side body failure closes the ticket before paired drain, so a
parked worker resumes, runs its issuer/stager joins and returns failure.
Existing queue waits and fail-hard SYCL paths remain intact; CPU cancellation
does not declare DMA complete or free queue/ring owners early.

VERDICT -> Concrete full paired source candidate, not a runtime or latency
qualification. The sealed V1 ticket's real-thread CPU harness passed in root's
isolated container; that does not compile or qualify these SYCL changes.
Before runtime: independent source review, exact patch application, fresh
header-dependent engine build, source/ELF binding and startup-option admission.
The smallest model proof remains matched candidate OFF/ON A512 plus incoming
C64 after actual256 tokens, default share0.5, full histories against serial,
real owner events and host-monotonic arrival/token timings, owned normal
retirement, identity/page/source/kernel/health gates. Also exercise a real
failure/cancellation while stage1 is parked before broader promotion.
No new speed, stability, all-population or full-cache qualification is claimed.
