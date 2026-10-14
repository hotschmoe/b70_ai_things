# Prefill subchunk cooperation: source-only ownership audit

CONFIG -> Current52/57 paired native2/default-share0.5/no-MTP/no-borrow,
original weights/math/chunk512/checkpoint cuts. Source57 stays frozen.
Actual57 reduced maximum long gap11.983->8.516s but incoming TTFT31.151->36.624s;
P50/P95 did not improve. Coarse service redistributes work; it does not remove it.
No further chunk/share sweep is proposed.

Smallest existing subchunk seam:

- prefill.cpp3406: ordinary host-grouped FP16 MoE copies routing IDs then waits
  m.cs before CPU grouping. This already finishes the preceding FFN/residual
  work and current attention/router on that compute queue. A hook immediately
  afterward needs no added GPU barrier and preserves T512/all GEMM dimensions.
- This seam is conditional: fused/native no-host grouping can bypass it.
  Initial scope must require this exact path, no peer/helper/CPU-share and no
  borrowed buffers. Do not add STRATA_PF_STEP_SYNC: that debug option adds waits.
- Prefill m.R/activations/GU/dq/ring are carved from its owned allocation;
  Verifier R_/staging_/hit scratch have separate owner allocations. Main
  GDN/PLE/QSA state is partial at this point, but native batch graph uses each
  private slots_[brow[t]] bank. Therefore the hook must service only private
  decode, never checkpoint/publish partially read main state or resume main GEN.

Thread/stage limit of a direct callback:

read_part calls sp.run for one512 chunk and public run drains the successor
pipeline before returning. Stage0 runs on the controller thread. At its router
wait, next-stage work for this chunk has not started; initial narrow guard must
also prove next_run_ invalid/no older stage worker. Stage0-only service can run
existing batch_step on that owner thread, then resume the exact same loop/local
stack. It cannot cover stage1's several-second portion of the chunk.

Stage1 is launched by std::async at prefill.cpp4434. Controller waits through
Prefill::drain_pipeline's future.get. Stage1's on_chunk/on_stage_chunk/router hook
runs on its own thread/context. Calling batch_step there would move controller
Drive/verifier work to that thread and can race queue submissions or recording,
scratch, CPU pool and context state. Existing callbacks are not a ready paired
subchunk scheduler. Do not call decode from them directly.

Smallest complete paired implementation seam:

1. Add a nullable per-Prefill host safe-point callback at the existing router
   wait, after required current compute work is complete. Initial route rejects
   helper/peer/CPU-share/fused bypass/borrowing and multi-chunk pipeline overlap.
2. Stage0 owner invokes controller directly only when no other prefill worker
   runs. Stage1 publishes one CPU yield ticket (request/chunk/layer/device/
   owner nonce), parks before any further compute submissions and awaits ACK.
3. Replace the controller's blocking successor future.get seam with a small
   owner pump that services that ticket, or joins completed future. Only the
   original controller thread runs batch_step. No new GPU event/barrier/copy,
   activation checkpoint or chunk split is needed. Resume stage1 on its original
   scoped device with its unchanged stack, pointers and normed/half/layer values.
4. Spend at most one available existing private decode window per due safe point.
   Charge its actual elapsed service against the same decode-share allowance;
   remove that spent amount from postchunk/final-chunk allowance. Prefill elapsed
   must exclude service time, or the old 0.5*elapsed formula recursively credits
   decode and double-spends. This is a budget ledger, not a share sweep.

Required ownership proof before that is a runnable candidate:

- Record/assert per-device disjoint ranges: Prefill owned region/R/GU/IDs/decoded
  slots/ring, Verifier residual/hit/plan/staging scratch, each private slot's
  complete GDN/PLE/QSA banks, incoming main banks and handoff buffers. Immutable
  dense weights/owned stage-mirror tables may be shared. No owner rebinding.
- At yield, one stage's compute queue is completed at its existing wait and every
  other prefill compute producer is idle/parked. No graph begin_recording while
  another producer can submit to that queue. OnDevice scopes and queue/context
  identities must restore across private stage0->stage1 decode and resume.
- Stager and issuer can still hold future copies: issuer lives until chunk end
  (joins at4384), and StagerDone drains on exit. Either prove their copy queues/
  ring slots/descriptors are private and immutable to decode, or park issuance
  with ownership ACK. Do not insert a global queue wait to mask the problem.
- Async PLE read futures similarly must own independent host buffers; shared
  table/cache locking and File/GGUF source leases must survive private decode.
  CPU expert pool cannot simultaneously serve prefill CPU-share and decode.
- Pending STOP/BSTOP/BYIELD must be processed on the controller's existing path,
  not on a stage worker. Any failure/cancel/signal must release ticket, join
  stage/issuer workers and retain descriptor lifetimes before owned retirement.
  No exception path may leave a worker parked while the owner waits on get.
- Preserve all512-row layer math order and exact original checkpoint cuts.
  Matched first/later48+head/cache tests remain necessary before generalizing;
  latency alone cannot prove partially suspended main state is preserved.

Minimum useful proof after implementation: same authentic A512/C64 arrivalafter
actual256 as accepted fairness/57, matched defaultOFF/newON/source/config,
complete histories, exact safe-point ticket/ACK ownership, actual native2 windows
and same-host TTFT/P50/P95/max gaps. Include an incoming cancellation/owner-stop
close for the new parked-thread lifetime, then bounded cache-state comparison.
No large UR/clock trace or synthetic timing claim is needed. A small linked
state/descriptor fixture can reject scratch alias/failed-ACK bugs before model
run. Adding Prefill callbacks changes its class/header ABI: rebuild every actual
header dependant and all8 ELFs; main-only57 object reuse is not sufficient.

RESULT -> Source/metadata audit only. No flags-only safe paired inner-layer
callback exists today. No GPU, compiler, payload or model process; no57 edits.

VERDICT -> Real existing per-layer router wait is the narrow physical seam.
Stage0-only owner callback is the smallest partial prototype; whole paired
latency requires controller-mediated CPU yield/ACK and explicit ownership proof.
There is no supported one-line callback change that safely covers both stages.
