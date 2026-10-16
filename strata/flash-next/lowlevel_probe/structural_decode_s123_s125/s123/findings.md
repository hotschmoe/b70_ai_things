# S123 source lead: split two MTP forwards, retain primary head

CONFIG -> ONE conditional structural direction: keep slotA MTP forward on
card0, slotB forward on card1, join their small final hidden rows for the
existing card0 paired head. No head-only offload, TP/backend campaign or
implementation/prototype. Existing EP149/363, C2/depth2/fullheadQ5_K context.

COMMAND -> Current SYCL source ownership/callers, retained S114 effective
argv/load logs and small S95 operation CSV only. Exact pins/excerpts and
bounded census derivation accompany this note. No trace/model/binary scan,
compiler, decoder, GPU or new model consultation.

RESULT -> Placement hypothesis is source-supported. generate4159..4173
loads global and both slot drafters in the same primary device scope, sharing
weights. Slots bind primary WeightTable/native_head and primary residual
buffers7520..7567. MtpDrafter::load records current device and rejects shared
weights across devices278..303. Verifier's primary graph does dense/shared/
head work; peer ep_q_ only runs plan/inbox, routed experts and return-row
operations (verify1421..1443/1816..1825). Both verifier graphs are submitted,
peer first, at3327..3329. S95 whole-collection primary has9682 MMVQ events,
peer0; both have1392 GU and1392 down events. This corroborates ownership,
NOT peer idle, overlap, traffic or a critical-path saving. Clock domains and
spinning EP kernels prevent those inferences from service imbalance.

MtpPair explicitly requires same device/head2058..2073, uses a->cs_ and
primary scratch2100..2109. Its head-free pre_round/pre_step seam2134..2185
already calls record_rest/record_forward(...,false); however BOTH are recorded
and launched serially on cs_2355..2360, followed by the shared head. This is
a real opportunity to overlap A/B WITHIN each draft depth. It does not make
MTP independent of verification: generate9448..9458 first copies accepted
verifier residuals, then drafts, and depthj+1 needs the prior head/select.
Current record_tail2198..2205 quantizes both samples, runs C2 head and BOTH
post-head/select consumers on the same primary queue.

Source-derived payloads (N2560, HC4), not measured physical transfers:

- PeerB initial residual: G*4*2560*4 =40960G bytes; usual G3 is122880B,
  source maximum G4 is163840B. Plus G token IDs/step/position records.
- PeerB final sample to primary:2560FP32 =10240B EACH draft depth. Keep
  quantization on primary initially; no cross-context Q8/FP equivalence claim.
- Primary head result back to peer select: one tokenID4B and probability4B
  per depth, plus an explicit ready/error generation protocol. B's HC
  residual/select state remains peer-local; no need to send vocabulary logits
  (248320FP32 =993280B) or the full residual back to primary every depth.
- These minima exclude mapping alignment, control records, staging copies,
  prompt/KV initialization and fallback traffic. Gen3x16 across different
  Threadripper root domains is retained topology, not a measured new link cost.

Five exact touchpoints must close before calling this a small patch:

1. Load/bind B on device1 with its OWN MTP dense/512-expert/state allocation;
   shared-load/device checks cannot be bypassed. Source snapshot retention and
   replay/destruction must drain BOTH queues before any arena/bridge is freed.
2. NativeEmbed is process-global, one dev_ alias from primary-context pinned
   host memory (native_head151..161/205..266); record_front842 gathers through
   it. Peer accessibility is UNPROVEN. Provide a valid per-device mapping or
   explicit embedding-row bridge; selecting device1 alone does not prove it.
3. Add a two-device-only MtpPair path with per-owner capture/launch queues,
   same existing arithmetic, distinct graph/cache/owner lifetime. q_of and
   OnDevice do not migrate USM. ep_mem.hpp documents separate contexts and
   dedicated IPC allocations: do not hand primary arena pointers to card1.
4. Add explicit residual/sample/token/prob bridges and completion edges,
   including primary-side B head-selection scratch distinct from peer state.
   Existing EP inbox/flags have verifier layer-ring ownership; do not reuse
   them for MTP or infer peer graph completion from service/last return flag.
   No blind wait removal, shared-stream disabling or commit overlap.
5. Preserve one-active-slot/unequal-depth/error fallback. Current solo calls
   B.draft() use its local head, but proposed B cannot read primary head USM.
   Either coordinate B's SOLO head on primary too, or replicate its head on
   peer. This is an unresolved design choice, not permission to drop fallback.

Memory is plausible, not proved. Retained log reports one full MTP owner
1027MiB (675experts+111dense+241private); current slot clones each241MiB
because they share primary weights. Moving B needs roughly1027MiB peer
allocation, not just241. A peer head for solo fallback adds437043200B
(416.797MiB). Log's2010MiB peer-free figure is BEFORE final verifier exchange,
not current headroom or a budget guarantee. Main peer experts already28.34GiB;
no eviction/recompute assumption or expert rebalance is authorized. This does
not move the verifier's two200k-context slot sessions (logged5.63GiB on primary).

VERDICT -> Worth independent source design review, not implementation GO.
It can remove a genuine A-then-B dependency within a depth; attainable saving
is UNKNOWN because per-owner forward service is not isolated and peer is not
proved idle. Five ownership/bridge/fallback touchpoints make it more than a
device toggle, but do not by themselves prove a major rewrite. Minimum strict
oracle must exercise actual head-free forwards/paired head and bridge copies:
depth2 and fresh A/B/A replays, full consumed intermediates/KV/residual/sample and
head/token/prob outputs against single-device control, B-only/unequal-step
fallback, guards/read-only/normal drain and both-card health. A transport-only
stub can test freshness but cannot qualify MTP numerics. No oracle prepared.
Existing layer split moves MTP with the last verifier stage but is explicitly
incompatible with EP; pipeline-windows also rejected with EP. S15 streamOFF,
S79 commit-overlap and S106 graph submission changes are not this mechanism.
