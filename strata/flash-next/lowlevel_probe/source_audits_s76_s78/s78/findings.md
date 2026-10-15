S78: nine replay boundaries and one bounded source lead (2026-10-15)

CONFIG -> Existing S31/S32 evidence plus current pinned SYCL source only.
No source change, compile, GPU, model consultation, weight read or new profiler.
Root stopped a duplicate timeline campaign: the old audit already contains the
useful uncovered-interval result. The 270MB raw trace was not reread/rehashed.

COMMAND -> Read retained reviewed-events.json and existing cycle/host analyses;
inspect verify.cpp, mtp.cpp, generate.cpp and state ownership. Small JSON reads
only; no new union analysis, additive service budget or cross-clock alignment.
Source/evidence SHA256 and exact line regions are in source-pins.json.

RESULT: what the existing trace already establishes

- S31 raw identity eb7ac30d... / 270756565B / 1722051 events; native152055a6...
  Full raw hashes are historical retained identities, not hashes recomputed now.
  Reviewed five cycles5..9 contain29205 extracted records:25755 primary gpu_op,
  1685 peer gpu_op,1759 cpu_op and6 metadata records. The independent host-start
  extraction has1760 calls because its boundary rule differs. No new event was
  manufactured to reconcile them. Two primary tids are engine/thread labels,
  not proven application queue handles; API args lack graph/queue/timeout/result.
- Existing ALL-command unions in the287.335ms reviewed numeric selection are
  primary233.9595ms/peer184.5895ms. Raw-primary calculation233.95775ms differs
  by1.75us from shifted rounding; it leaves53.37725ms uncovered. These are
  recorded command coverage, not productive hardware occupancy. EP wait/spin
  kernels and reset/runtime commands count as busy. Peer EP-wait service sum
 27.100ms/cycle is NOT useful compute or an additive component of the union.
  Cross-card clocks are not joined; do not align peer idle to primary cycles.
- Existing primary top gaps: one per cycle3.300-3.941ms after final MTP select
  before next ep_bump; one per cycle1.5545-1.64825ms from instrumentation-sensitive
  barrier to copy_i32_from_mapped. Temporal host context is not a dependency or
  cause. No extra gap ranking/profiler run is justified to repeat this fact.
- S32 is a SEPARATE CPU-only marker run:1684 records,48 complete windows,
  40 conservative replay windows. Median commit region1.5907ms; pair submit
 0.0450ms; next gather0.8292ms; pair-return to next verifier-submit0.8653ms.
  These are overlapping CPU regions, not GPU time or removable latency budgets.
  They cannot be joined to S31's gaps. Later hot-cache S49 PLE evidence also
  prevents treating the old0.8292ms gather as a universal current opportunity.

RESULT: nine source calls, not nine inherently required host decisions

Source reconciliation at depth2 is 2 verifier replays +1 commit +6 pair replays:

1. verify.cpp3327: peer expert graph is queued first. Primary graph at3329
   consumes its results through the existing EP protocol. Different devices
   and cross-card data dependencies prevent a simple same-queue graph merge.
2. generate.cpp9383..9407 waits for target results, computes per-slot longest
   accepted prefixes, then calls commit_slot_prefixes. Acceptance must precede
   commit: rejected rows must not update recurrent/indexer/history state.
3. verify.cpp3515 submits batch commit; its3533 queue wait is unconditional.
4. mtp.cpp2221..2224 queues roundA,roundB,tail0,stepA1,stepB1,tail1 on the existing
   pair queue. T0/T1 depend on accepted prefixes; steps_of caps context/ring end.
   tail0 must produce each slot's token/residual before its depth1 step; tail1
   follows both depth1 projections. These are real DEVICE dependencies.
5. All six submits finish BEFORE polling/mapped reads/PLE at2230 onward. There
   is no host acceptance or new input between those six graph calls.

Thus nine observed AppendCommandListsExp/cycle is consistent with the source,
not proof of one-to-one native graph/handle mapping (trace arguments omit it).
The current pinned source includes later markers/PLE changes; it is not claimed
byte-identical to the S31 executable. Original S26 pair patch preserves the
same launch ordering and the measured S31 paired motif supports reconciliation.

HOLD pair-graph coalescing: one directly recorded ordered pair graph could
preserve these dependencies without S22 independent-slot submission. However,
this targets the separately observed small45us pair-submit region, not the
post-pair readiness gap. Backend native-list reduction is not established from
source counts. Do not spend a GPU window merely to reduce six source calls.

ONE source GO sketch: overlap batch commit with draft, join inside batch_tick

- Existing scalar commit already records commit_done_/commit_pending_ without
  host waiting (verify.cpp2827..2876); batch commit always waits instead. This
  is a distinct missing batch-path behavior, not the rejected S22 experiment
  that launched independent slot drafters and moved residual-copy queues.
- Batch commit3069..3148 updates MAIN per-slot GDN state/conv, QSA indexer tails,
  and PLE history, reading verifier private snapshots and mapped keep records.
  It does not write final_R or any private MTP operands. Drafts read copied
  final_R and their own allocated K/V/attention/scratch; qsa_state_init shares
  only read-only RoPE cos/sin. The queue choices already exist: dedicated
  verifier queue, default residual-copy queue, slot0 pair queue. No new queue.
- Prepare only a bounded fork/join: submit the SAME batch commit, record its
  existing completion event, advance host ple_prev before draft prefetch as
  today, allow current residual copies/draft, then join BEFORE batch_tick
  returns. Join/drain on every error as well. This retains the outer completed
  state contract before admission, park/restore, prefill, next stage or teardown.
  Keep mutable commit records/snapshots alive and untouched until that join.
- This is NOT permission to delete the wait. A future exact patch must audit
  BT/BDONE publication and error order (currently emitted after completed
  commit), partially active/terminal slots, solo fallback, graph cache lifetime,
  PLE metadata/history, exceptions, next_ restrictions and same-context bindings.
  The existing scalar event machinery is a reference, not automatic coverage.
- No hardware overlap or gain is established; both workloads use primary GPU.
  S32 commit1.5907ms only motivates this ownership check. Its CPU span cannot
  be subtracted from decode time, nor can scalar async behavior establish
  batch safety. Hold runtime until an exact small patch passes root/model
  source review and the usual build/numeric/work/lifecycle gates.

VERDICT -> GO for root to consider ONE batch-commit source preparation only;
HOLD new GPU/profiler/full-model runs and pair-graph coalescing. No causal idle,
PCIe, productive-busy, speed or universal-next-mechanism claim. Preserve rejected
S22 async slot submission, shared-stream toggles and global QFUSE. No new flag,
cache framework, source edit or shared documentation update was made.
