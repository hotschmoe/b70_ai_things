# S75: canonical-Q8 IQ3_S C6 DPAS private prototype (2026-10-15)

CONFIG

- V2 SOURCE PREP ONLY: no build/frontend, GPU, model consultation, payload read,
  production edit or timing. Parent owns exact reviews and any later execution.
- Fixed type21/C6 shapes and actual slices are unchanged from S65: gate
  N2560/M6144, blk.0.attn_gate.weight at734367616,6758400B; out
  N6144/M2560, blk.1.ssm_out.weight at2545748608,6758400B. weight-identity.json
  retains exact SHA/FNV and NVMe path; no slice was reread during preparation.
- Seed7 finite normal input and the SAME changed input transformation as S65.
  Public native_mmvq anchor, original ControlSplit private control, one DPAS
  candidate. No FP16 activation substitution, SWAR candidate, permanent planes,
  format/shape sweep or full-engine integration. S74 runtime check is omitted;
  its separate executable is not queued. Full public byte checks are the gate.

COMMAND (prepared, unexecuted)

- After root source/model GO only: bash build.sh. Offline no-DRI/network image
  e41d59c0; cpu-run; source/work/common/frozen mounts readonly, fresh disk output.
  Exact command/flags are in build-container.sh; one adapter compile/prelink,
  ordinary host-only final link. No HC/main kernel TU or common AOT rebuild.
- Frozen public common wrapper8f5a7efe... plus kernel archivef2b53eb4... and
  core34e91c40... are reused readonly from S59, never restored into live build.
  source-link-bindings.json/expected-container.sha256 pin full identities.
  Small imported headers were hashed now; large frozen hashes are retained
  S65 records, to be rechecked under the actual build/runtime CPU lease.
- Any later GPU GO: bash ROOT_RUN.sh, quiet CPU outside card0 lease, physical
  ZE_AFFINITY_MASK=0/BDF0000:0b:00.0, one process, no profiler/retry. Existing
  corrected S65 CID-owned cleanup holds leases until removal is confirmed,
  before post-health. This is a recipe, not a queued workload.

RESULT: source mechanics and ownership

- private_iq3s_dpas_c6.hpp retains exact S65 ControlSplit body (:24-67) and
  scalar control. source-extraction.json hashes the unchanged decoder body;
  vs-s65-source.diff shows every source change. Original byte anchor must pass
  before the candidate for each initial/changed input. No posthoc tolerance.
- Candidate (:218 onward): one workgroup=one SG16, outputtile16, token tile8
  (six active, two zero), K32. Each WI lid decodes original row r0+lid via
  ControlSplit::load; it unpacks the already signed packed bytes into B.
  A takes unchanged canonical Q8 qs, with rows6/7 synthesized zero.
- B is K32x16 column-major stride32; A is8x32 row-major stride32. INT32 matrix
  accumulator is reset for EVERY p=K32 segment. Intel coordinate-aware apply
  scales only its own INT32 fragment element by ls[logical_out]. No shared
  side effects or assumed lane/ei++ coordinate mapping occurs in the callback.
- Uniform matrix_store writes an8x16 INT32 local tile. WI lid then owns that
  output column and performs exactly six FP32 ledger updates. Different WIs
  never update the same ledger cell. Scaled integer bounds are unchanged:
  <=1889760 for canonical [-127,127], so INT32 and integer->FP32 are exact.
  FP source statements are d=dw*dx; v=d*sumi; partial+=v.
- Shared payload: A256+B512+INT32tile512+ledger12288+dw64+ls64+dx12=13708B
  before allocator alignment/compiler overhead. Private tree[16] is reused
  per token per WI; no resource/occupancy claim before actual compilation.
- Bounds: M is divisible16 and host dispatcher rejects other shapes. Every
  work-item remains active through all collectives/barriers. Output writes
  c<6 only; xq/y retain real512B poisoned inactive tails plus4096B outer guards.

RESULT: barrier and reduction proof

- Ledger initialization has unique strided writers, followed by local barrier.
- Each p stages A/B/dw/ls/dx; pre-load local barrier publishes them to matrix
  loads and the callback. Full tile store then local barrier publishes INT32
  results before per-WI ledger/scales reads. Bottom local barrier finishes all
  tile/dw/dx reads before next p overwrites shared storage. It also orders
  partial writes. No collective call occurs inside a thread-dependent branch.
  Final iteration's bottom barrier is the final ledger publication boundary.
- Ledger slot p%32 receives p=lane+32*j in increasing j, exactly native partial
  order. Each final WI owns one output column and reduces six token outputs.
  Native mmvq stores lane=c (:201). Candidate first computes
  tree[i]=L[i^c]+L[(i+16)^c], then folds first8/4/2/1. Induction: a node after
  each fold equals the native XOR16/8/... subtree rooted at its (i^c) lane;
  final tree[0] is native stored lane=c with identical operand parenthesization.
- reduction-source-proof.json verifies that symbolic expression for all6 stored
  lanes and both80/192 K32 index sequences. It computes no FP values and is
  NOT a compiler/bit-identity proof. Optimizer contraction/inlining/storage can
  still differ; actual full strict public/control/candidate outputs gate it.
- Retained conservative barriers are intentional. This one design does not
  overlap staging with previous epilogues. Rejecting its timings would reject
  this exact implementation, not every INT8/DPAS mapping.

RESULT: predeclared gates and timing limits

- Both shapes: initial+changed eager public/original/candidate/repeat, own
  single and20-call graphs, full outputs, finiteness, guards, meaningful tails,
  input/weight/Q8 immutability. Changed Q8 AND outputs must differ from initial.
  All shapes' correctness completes before warm/timing; first mismatch exits5
  with actual input/Q8/output evidence. Runtime/IO failure exits2; no retries.
- Unchanged S65 schedule:3 warm20-call graphs/arm;8 nested OCCO/COOC rounds,
  four20-call samples/round in ONE process. Every warm/timed/final snapshot
  still checks full bytes/guards/readonly. Projection only, Q8 already ready.
- Expected successful artifacts:118 checked snapshots,10 DOTs/86 kernel nodes/
  76 dependency edges,92 saved binaries. Analyzer verifies actual public IQ3S
  C6/private-original/DpasC6 names, chains and the same three non-null global
  resources per node; implicit local-accessor metadata is not a fourth global
  resource. Final saved outputs/Q8 are checked against initial public/canonical.
- Exact DOT inventory and closure must pass before reporting timing claims.
  Timings are descriptive nested batches, not independent process samples,
  significance, ISA causality, physical bandwidth or a serving gain forecast.
- Local matrix loads and coordinate-aware apply exist in the pinned headers;
  source template accepts local pointers. S74 proved only global-pointer signed
  DPAS generation. This new local/scale/ledger AOT and runtime are UNMEASURED.

VERDICT

- READY_FOR_ROOT_SOURCE_REVIEW, not build/GPU authorization. One bounded
  canonical-Q8 prototype, no changed numerical policy or permanent integration.
  Compile/strict mismatch is preserved as evidence; no fallback flavor or
  relaxed acceptance is silently selected. No speed/cycle/roofline prediction.

V2 SOURCE REVIEW CORRECTION

- Parent read actual Opus source HOLD on possible contraction differences. V2
  names the product before the ledger addition, matching native apply-return
  then caller-add statement separation. This is a source-level precaution, not
  proof that either version contracts differently or that v2 forbids contraction.
- Only that statement split changes the kernel. Decoder/tree/barriers/scales,
  API, payload, cleanup and timing remain unchanged. No contract(off), compiler
  policy, tolerance, build, GPU or new model call. Future paths point to v2;
  v1 and its manifest are preserved. readiness-v2.json records exact identities.
