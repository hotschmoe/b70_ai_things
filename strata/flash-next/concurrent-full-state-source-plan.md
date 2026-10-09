# Corrected Strata concurrent full-state serving: source audit and next plan

CONFIG -> CPU-only consumed-source audit of corrected0018 engine
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp/source,
base fb58e0 plus0001..0018, original UD-Q4_K_XL and segmented host expert mirrors.
COMMAND -> Inspect SYCL generate/session/verify and API server/artifact identity;
compare actual slot, state-transfer, scheduler and teardown call paths.
RESULT -> Ordinary batch slots2/4/6 have real two-stage implementations and the
strict native HC/PLE route reaches them. Current serving/profile/diagnostic
configuration is serial, and complete concurrent cache/telemetry/lifecycle gates
are missing. No GPU activity accompanies this audit.
VERDICT -> Retain the single engine with private per-stage slot sessions, first
qualify ordinary batch-groups1, then diagnose measured interference. Do not
advertise requested parallelism or full cache support until actual INFO/state/
output evidence passes. Existing consumed source/plans remain unchanged.

## Actual support, distinguished from the current profile

- generate.cpp:3818 allocates a separate SessionState for every slot on every
  owned stage. init_slots validates matching layer/context/QSA/GDN ownership.
  Batch1 is disabled: the one-stream baseline is batch0, not batch1. Eight rows
  is the verifier cap;2/4/6 plain slots fit that logical cap, subject to memory.
- generate.cpp:6979 builds linked stage verifiers with max(spec,batch) rows.
  Each gets its stage's private slot sessions at7207. verify.cpp:2940 feeds active
  row-to-slot mappings, then recursively executes later stages at3085; commit
  recurses too. The stage handoff carries residual/pending/inject row data.
- verify.cpp:807 maps row t to its private slot session;927 uses private PLE
  history;1014/1036 use private GDN recurrence/conv;1100/1225 use private QSA
  tails/pools/KV. Native HC dispatch946 explicitly invokes source-native one-row
  operations per row. Strict source PLE disables legacy rounded batching paths.
  This is source route coverage, not actual concurrent numerical qualification.
- generate.cpp:8710 and8770 copy admission or returning slot state on EVERY
  stage using checkpoint save/restore plus full KV/indexer images. They quiesce
  devices and transfer through host vectors. Shared weights/expert caches/mirrors
  belong to the process/stage; sequence recurrence/attention/PLE belongs to slots.
- Ordinary batches run serial stage passes, group1, with all currently active
  slots in one window. Batch-MTP and pipeline-windows remain prohibited. Multiple
  pipelined batch groups are a separate unqualified scheduling path; use group1.

Current C1 args explicitly contain --batch0. server.parallel_args:2204 returns
without adding batch whenever --batch/--slots already exists. Changing config
parallel alone therefore does nothing. A new concurrency launch/controller must
replace the explicit batch arg and require actual INFO batch_slots=N, not accept
fallback0 or silently reduced capacity. The API gets actual capacity from INFO,
but reporting available slots does not establish simultaneous decoding.

## Concrete gaps and unsafe paths

1. Correctness controls and captures are serial. GEN pin/fresh rejects batch>0
   at generate.cpp:9454;0014 rejects active batch diagnostics;0013 supports only
   serial admissions and stores one mutable current request. Batch BT has no
   per-slot first/full vocabulary capture or complete evaluated-token ledger.
   Serial full-logit/48-layer proofs cannot be reused as concurrent proofs.
2. Slot admission at11587 preserves only the deepest matching checkpoint.
   Root/pin/leaf retention metadata is discarded; explicit pin is currently
   rejected, so API shared-prefix requests error under batch. BYIELD preserves
   earlier checkpoints, but that is a different route. Extending pin requires
   retaining a bounded complete-stage chain, not removing the guard alone.
3. Slot arenas are raw malloc_device allocations, carried only through derived
   SessionState pointers. SessionState has no owning destructor. Allocations
   dropped by vector.resize/clear after another stage fits fewer slots are not
   explicitly freed; partial init can also leave side allocations. The malloc
   branch tests DPCT_CHECK_ERROR but not buf==nullptr before session_init. Fix
   ownership/null/partial-init rollback before any capacity stress experiment.
4. Plain batch layout graphs are captured lazily and cached by row layout.
   Limit defaults0; set_batch_graph_limit64 is called only for batch-MTP. Plain
   six-slot subsets can create63 compute+commit graph pairs per stage. This is
   finite at fixed slots but costly, with first-layout capture on critical path.
   Bound and account graph residency, and separately report cold-layout stalls.
5. Global control-vector enable state changes during admissions, while private
   slots retain only a matching boolean. Active mixed control-vector modes can
   therefore select the wrong math for another slot. Initial strict scope must
   reject mixed/global mode changes or implement true per-row math. Slot sampler
   also sets penalty_last_n=0: later BT tokens do not implement the serial penalty
   semantics. First phase is greedy, no penalties/control vectors, text only;
   unsupported requests need an explicit error, not partial semantics.
6. Server uses shared engine.last/reused/progress; BDONE merges a slot's finish
   into that global last dict. Per-request UI status exists, but admission/last
   timing fields can belong to another request. Native BT/BDONE identify only
   slot index, not request/generation; restarts use an engine generation counter.
   Add stable request/slot generation identities before trusting API telemetry.
7. _take_control uses a contended lock with prompt-length yield hints and at most
   two yields, not a measured fair FIFO. First decode may be STOPped and migrated
   into a slot; a slot left alone may migrate back to solo up to two times. These
   transitions involve full-state copying and can change row geometry. They
   require correctness and accounting gates; disabling transitions is a new
   declared profile, not a diagnosis of their cost or correctness.
8. Serial process QUIT drains stages, explicitly retires mirror-dependent graphs
   and mirrors, then _Exit0 instead of unwinding every GPU/session object. That
   proves only process/context reclamation after terminal exit/post-health.
   Slot raw allocations/fallback cleanup are not proven by that endpoint. Need
   explicit slot/graph owners and logical-free trace controls before claiming
   leak-free in-process resize/reconfiguration or successful discarded slots.

## Cache identity and budget contract

Same-process cache matching uses exact tokens/images and control-vector mode;
per-process model/method/precision are fixed. Artifact identity007 binds source,
tokenizer/template/runtime and rejects stale host session contracts. Disk
SAVE/RESTORE explicitly rejects layer split and parallel API; persistence is
optional for this goal. Native cache has no dedicated tenant identifier: identical
prefix sharing is math reuse, not an authorization policy. Test independent API
sessions, external request IDs and disconnects, not just named native branches.

Per-slot checkpoints contain GDN/conv, PLE history, QSA tail/dead/block_pos, with
completed KV/pooled rows retained in that slot. Shared parking needs full images
for every stage. Restoring only stage0 or token IDs cannot qualify reuse.
Per-slot idle-session/checkpoint bytes sit outside the current parked snapshot
budget: retaining three checkpoints across six slots could add about2GiB of host
GDN payload alone. A new whole-state cache budget must account main/slot chains,
parked and retained images, device slot sessions and graph metadata separately.
Pins may cause a bounded miss/skip rather than unlimited retention.

Source arithmetic: each sequence has36*(128*48*128+10240*3)*4 =117,669,888 bytes
of GDN persistent state (112.22MiB), independent of context. FP16 KV total is
48MiB at2048 and192MiB at8192. Six slots add six private sequences beside the
main admission sequence, before scratch/RoPE/indexer and per-stage duplication.
Allocating slots precedes expert-cache sizing, so additional slots can reduce
VRAM experts and increase required host mirrors. Record actual complete mirror
coverage/admission bytes, not only free VRAM after READY.

## Separate implementation increments after serial0019

Coordinator assigns patch numbers; do not modify frozen0013..0018/V5.

A. Add a stage-owned SlotSessionArena RAII owner: allocation base/bytes/context,
   SessionState side allocations, explicit null check, partial-init cleanup,
   discard on fit shrink, teardown order graphs then sessions then weights.
   Require exact requested slots for strict qualification, otherwise fail before
   READY. Exercise owner allocation/free with bounded per-stage source oracles.
B. Add request+slot generation protocol metadata and separate per-request sampler/
   cache/trace bookkeeping. Reject unsupported penalties/images/control-vector
   changes in strict first scope. Bound plain graph map using existing LRU hook;
   drain owner queue before graph retirement and preserve key/layout identity.
C. Extend complete-stage pin/fresh and full cache ownership into slot admission:
   fresh reconstructs only that request's staged sequence; active peers remain
   intact. Preserve root+one pin+rotating leaf at admission/BYIELD/solo migration;
   use global bounded accounting for all host checkpoint/snapshot allocations.
D. Add default-off per-slot raw observation and host work counters. Correlate
   request/slot/generation, original input token, absolute position, stage bounds,
   cache admission/victim and logical prompt spans. Capture one admitted T1 head
   plus one later actual batch step, mandatory full248320 logits and48 residuals
   per observed request; never use the last global request label for all rows.
   Keep bounded<=6 observed requests/process and an explicit total byte cap.
E. Build a new API concurrent controller/parser. Keep hotschmoe-dd first, distinct
   model/method/scheme+slots/topology/geometry aliases, exact /v1/models and eval
   registry checks, compiled source/image/model/PLE/template identity. Use the
   existing pair-lease parent lifecycle and fresh complete post-run hashes.

These are real source/API changes, not extra runtime flags to hide unresolved
state drift. First execute existing ordinary batch path in a bounded ownership/
math oracle after required ownership guards, then improve measured scheduling.

## Numerical and serving qualification sequence

1. Finish serial full-logit/cache gates and matched one/pair2048/64 geometry.
   Preserve production8192/128 as a separate required capacity/long-context gate.
2. Slots2, then4, then6 bounded stress, group1/no MTP/no borrowing/adaptation.
   Compare identical source token streams, exact generated IDs/natural stop,
   actual per-slot work and full raw heads/residuals. Batch row geometry can
   change reductions; preregister reference geometry and numerical contract
   before launch. Initial strict equality failures localize first divergence,
   never widen tolerances after seeing results. Matched batch-row references
   may supplement solo references, but cannot replace the solo fidelity question.
3. Independently test every slot/generation, shared prefix with divergent suffix,
   real generated-reply/new-user continuation, live/slot/parked restore, cache
   admission/victim identity, budget eviction, prefill/decode cancel and replay,
   queued cancellation, disconnect, slot reuse and restart/stale identity. Require
   nonfresh same/unrelated replay as well as authoritative fresh reconstruction.
4. HTTP simultaneous SSE requests at1/2/4/6, with explicit request IDs and exact
   served aliases. Require measured overlap of actual decode windows, capacity
   exactly requested, correct session association and no lost/duplicated tokens.
   As soon as one request is decoding, introduce a long useful prefill in another.
   Repeat short-prefill/long-prefill and cache-hit/miss controls in balanced order.
5. After correctness, diagnostic-off matched ABBA runs. Record client send,
   server arrival/control wait/admission, every native span/BT, sampler/stream send
   and client receive timestamps. Report per-stream P50/P95 TTFT/token gaps,
   max stall/starvation, equal-work completion ratio and overall throughput, with
   enough independent repetitions for tails. Aggregate tok/s and token arrival
   alone cannot assign a stall to host prep, capture, transfer or device work.
6. Correlate model-call critical path with queue submit/start/end and waits, whole
   state copy/cache bookkeeping, lazy batch capture and sampler/stream work.
   Prefill read_part interleaves only after completed chunks; default decode time
   budget is half the preceding chunk duration, with no interleave after the last
   chunk. BYIELD copies complete state. Measure those limits before changing
   scheduling; do not remove safety waits to make a profile look fast.
7. Every measured run owns both GPU leases through all child/container terminal
   states, per-card+compiled-pair post-health and fresh all4 model hashes. Preserve
   failures. No shelf promotion until concurrent correctness and matched clean
   latency pass across required production capacity and all tiers.

Goal scope clarification: disk persistence is optional; images and experimental
control vectors are not required campaign features. Explicit unsupported errors
are acceptable for those requests in the declared text scope. Automatic shared/
repeated prefix continuity is required. Explicit pin is one implementation tool,
not a substitute for automatic cache reuse when the caller supplies no pin key.
