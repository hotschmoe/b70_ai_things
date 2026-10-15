# S82 prefill critical-path source triage

CONFIG -> Current SYCL native EP IPC/PUSH, scalar per-expert FP16 oneMKL,
149/363 expert ownership, C12288 and all experts resident. Keep S38 alias,
S7 async, grouped GEMM, attention batch/lane and global QFUSE OFF. This audit
uses retained small summaries and current source, without executing anything.
The 2000 tok/s target is not met by the last retained clean matched prefill
screen: S38 OFF 1303.0 at 4K and 1546.1 at 32K (medians of three). These are
historical serving measurements, not measurements of the current source.

COMMAND -> Read current prefill.cpp, gemm.dp.cpp and consumers/allocation;
reuse S38 comparison/CPU spans, S33 sequential phase summary, S34 historical
trace summary and S57 header-only subset. No full trace parse, payload read,
GPU/build/model call or production edit. Exact source/evidence hashes are in
source-pins.json.

RESULT -> Select one small host-ordering lead for source preparation only:
DEFER THE UNCHANGED SHARED EXPERT UNTIL AFTER ROUTING READBACK.

Current source chain:
- prefill.cpp:3352..3361 queues router/route, then shared gate/up DQ+GEMM,
  swiglu_pair, shared down DQ+GEMM and BF16 scalar shared gate.
- :3464..3486 then queues routed-ID readback, IPC input copy and reduce-weight
  readback, and drains primary cs. Consequently the primary shared work must
  complete before host grouping and the peer submission burst can start.
- :3490..3504 folds optional peer timers, restores the primary device, then
  begins CPU counting/grouping. The earliest safe narrow insertion is here,
  before std::fill(m.cnt...), after both existing drains. Queue the EXACT
  shared block once here, before CPU grouping/peer submission; do not insert
  a new drain/event or move the existing routing/IPC readiness drain.
- :3697..3753 submits peer scalar gather/DQ/GU/SiLU/down, followed later by
  primary expert dispatch. :4457..4479 joins peer completion and performs
  the existing finish/combine, the first consumers of m.shared/m.sg.

The proposal changes when independent primary work is enqueued; it does not
change expert routing, arithmetic, ownership, GEMM shapes or peer protocol.
One host thread, original queues, no new allocations, launches or copies.
Do NOT defer until after peer submission: that would forfeit the intended
primary work during the host submission span.

Eligibility and shapes:
- Initial guard should be the actual eager native IPC/reduce_rows/PUSH,
  async-OFF/all-resident path, no MMQ/fused expert path, known Flash-Next
  geometry N2560/sharedFF640. Unsupported paths retain original location.
- Shared gate/up: T x 2560 activation times 640 x 2560 native weight to
  T x 640 FP32 each. Down: T x 640 FP16 to T x 2560 FP32. BF16 scalar gate
  is T x 2560 by 1 x 2560. T is actual chunk width, including short tail.
- S57 cached generic-WideK metadata independently lists 52 shared gate/up
  matrices of shape K2560/M640: IQ4_XS31, Q4_K19, Q5_K2. This is a FILTERED
  subset, not all 144 shared matrices or an exhaustive format inventory.
  Current source fixes all three shapes; no new header/payload read is needed
  to propose this format-agnostic movement. Any eventual patch must guard
  actual WeightRef dimensions and retain every native type unchanged.
- Expected block call counts remain three native_proj calls (each current
  dequant_f16+f16), one swiglu_pair and one BF16 gate, plus its existing
  optional BF16-low correction. No fusion or GEMM accumulation change.

Lifetime/alias audit:
- :1195..1204 carves mixed/mixed_bf/mixed_h and GEMM scratch separately from
  the shared MoE scratch region. They remain immutable across this half.
- :1233..1243 uses disjoint sequential allocations for ids/weights/slot/src,
  routed X/GU/H/Dm and shared sgate/sup/sh_h/shared/sg. Reuse with attention
  scratch occurs across halves only, after same-queue ordering, not here.
- All source references of m.shared and m.sg are their original producers
  and final combine/finish. No routed-ID or peer kernel consumes them.
- Shared native_proj uses m.gemm's scratch_; primary routed experts use
  distinct m.dq_gu/m.dq_d and peer P.gemm uses peer allocations/queue.
  The shared block retains its own within-block queue order and remains
  before final consumers. Do not move its writes across the next half.
- Gemm::f16 (:835..875) submits dpct::blas::gemm without an explicit host
  wait on this SYCL path; staged_gemm bypasses staging on BMG by default.
  Vendor submission may still block internally: source cannot promise the
  enqueue is cheap or prove actual overlap. m.gemm is initialized against
  the same caller stream (:1153); no new handle/queue is needed.
- The existing primary wait still completes route, IPC input and peer
  weights before the CPU accesses mapped tables or starts the peer.
- Refactor shared generation into one local callable only if preparing code;
  call at the old site for all fallback paths and new site exactly once for
  eligible paths. Preserve early-error behavior and do not introduce double
  execution or skip the block when no peer routes occur.

Cost scope and why this is not a forecast:
- S38 first fresh 4049-token OFF diagnostic reports 48 peer-loop CPU spans,
  median15.826ms/sum859.107ms; primary gather-return to first-primary-DQ
  entry median15.947ms/sum867.196ms. These are cold CPU ordering spans,
  not GPU service/idle or removable budgets. They make a distinct source
  overlap seam plausible; they do not measure this candidate's overlap.
- S34 historical 4044-token trace reports 48 primary gather-to-DQ uncovered
  trace intervals median52.987ms/sum2388.975ms, associated with peer host
  submissions. Ownership155/357, old source and profiler overhead differ;
  not current primary idle, speed prediction or an additive CPU/GPU budget.
- S33 Q0 long-call region labels: router+shared261ms, HC1480ms, GDN-in907ms,
  GDN-out631ms, QSA-proj638ms, expert-DQ472ms, GU671ms/down414ms. Gather2348ms
  includes peer host enqueue; QSA attention6442ms is not a dense operation.
  The 261ms bucket includes router/route as well as shared work, so even its
  whole region is NOT a movable-work estimate. Primary region total20963ms
  belongs to mixed 32K/chunks12288, not the short 4K clean screen.
- There is no retained clean current 4K per-operation critical path that
  proves shared movement is material. This is a low-cost scheduling lead,
  NOT a credible source-only solution to the full 1300->2000 gap. Dense
  dequant caching/packing is not selected: it adds residency/lifetime cost
  with no relevant isolated measured critical path. HIP-only PF_PAD/PF_GEMM
  branches are not a ready SYCL BMG knob. No inference from micro bandwidth.

Prior rejection scopes:
- S7 async spawned per-layer peer CPU work with join: slower 4K/32K. This
  lead adds no thread/queue/join and moves only independent primary work.
- S38 reverse alias retained same per-expert math but gave only1.78%/0.65%
  clean prefill differences; no repeat or alias rescue is proposed.
- S42 strict cross-algorithm mismatch was preserved. S80 new full-GU target
  passed, but grouped complete wave was4.19x scalar median; do not repeat it.
- S11 batch128/256 and S12 lane attention are rejected; global QFUSE producer
  correctness is not solved by this proposal. No math-policy change.

VERDICT -> CONDITIONAL GO FOR ONE SMALL SOURCE CANDIDATE REVIEW; HOLD BUILD/GPU.
Root may select this movement if a cheap source review confirms all fallback
paths. No standalone GEMM timing framework is warranted. Cheapest meaningful
future discriminator is one same-binary default-OFF narrow ordering guard
in an already planned serving build: same clean 4K and32K requests, native
work/routing/counts and exact output/usage gates, residency/functionals and
normal teardown/health. Existing S36 single first-request buffered markers
can establish unchanged shapes and ordering but cannot prove GPU overlap.
A bounded production actual-buffer strict comparison of shared/sg and final
MoE outputs is preferable if root has an existing hook; do not invent a new
readback campaign. Changed accumulation is not intended; bit mismatch would
stop the strict ordering candidate rather than invite a fitted tolerance.
No performance, default/promotion or full-model accuracy claim is made.
