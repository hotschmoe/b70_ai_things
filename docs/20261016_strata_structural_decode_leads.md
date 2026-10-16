# Structural decode source leads (2026-10-16)

CONFIG -> S123 examines splitting the two MTP owner forwards across cards,
retaining the shared primary head. S125 examines Q6_K pre-expansion followed
by exact INT8 half-dots through DPAS. Both are source-only designs for current
Strata C2 serving; neither implemented a kernel, allocated memory or ran GPUs.

COMMAND -> Read pinned source ownership/callers, retained configuration and
small operation/tensor metadata. Actual responding models claude-opus-5-5
(50.373389s) and claude-fable-5-1 (67.309814s) both exited 0. Root read both
answers and recorded the authoritative [decision and corrections](../strata/flash-next/lowlevel_probe/structural_decode_s123_s125/reviews/root-decision.json).
Exact prompts, answers, commands and source payloads are in the
[evidence bundle](../strata/flash-next/lowlevel_probe/structural_decode_s123_s125/README.md).
Reported consultation durations are current wrapper elapsed times; cumulative
session usage is excluded from the curated metadata.

RESULT -> S123 identifies a real source dependency: within each draft depth,
MtpPair launches A's and B's head-free graphs serially on the same primary
queue, then runs the paired head. Both slot drafters share primary weights;
the peer's expert work does not establish spare time for another forward.
Whole-collection service imbalance cannot show peer idle time, overlap or a
removable wall-time budget. Depths still depend on prior head/select results,
and drafting still consumes accepted verifier residuals.

A cross-card split needs independent peer MTP weight/state ownership, a valid
per-device NativeEmbed mapping, owner-specific graphs/queues and lifetimes,
explicit residual/sample/token/probability bridges, and correct solo/unequal
fallback. Primary USM pointers cannot simply be passed across separate device
contexts. The proposal's source-derived minimum B sample is 10240B per depth;
initial residuals are 40960*G bytes. These exclude protocol/staging costs and
are not measured transfers. Rough retained allocation figures and historical
free memory do not qualify peer residency or the two 200k-context sessions.

RESULT -> S125 is algebraically feasible but differs from a plain K32 matmul.
Each native lane needs two independently scaled dot16 outputs, the original
per-lane kbx accumulation order and the SG32 XOR reduction tree. Compact
pre-expansion preserves signed codes, int8 scales and raw half-d fields;
baking scales into int8 or substituting FP16 activation/weight arithmetic does
not preserve this contract. Candidate FP contractions, dataflow and materialized
values still require compiled inspection and strict tests.

The proposed C6/6144-to-2560 layout uses SG16 matrix operations, private FP
partials and a second kernel for the native reduction tree. Its 37.5% useful
MAC fraction is padding geometry, not measured utilization. The one-matrix
partial scratch calculation is 1966080B; startup pack costs and actual resource
behavior are unknown. Original weights remain necessary for prefill/fallback.

| Scope | Logical extra compact bytes, originals retained |
| --- | ---: |
| 29 output matrices | 488202240 (about 465.586MiB) |
| S41 large 52 matrices | 1099857920 |
| All 71 Q6 matrices | 1127915520 |

The 29-output subset differs from the larger frozen source table. Subtracting
its extra bytes from historical 1778MiB gives about 1312.414MiB; this is
hypothetical arithmetic, not current headroom or allocation/residency proof.

VERDICT -> Root selected the following order:

1. S126 private Q5/C2 runtime-uniform ones source first. Actual payload,
   correction-DP4A lowering, FP ledger and control identity precede any strict
   private GPU oracle/timing. No general helper change, Q4 build or shape sweep.
2. S127 source-only same-GPU owner-queue overlap is the cheaper structural
   discriminator: audit state/capture/fallback/error drain and explicit per-depth
   joins before the paired head and B's next depth. No new IPC or arithmetic.
3. Opus's same-device C2 projection-batched forward is a distinct alternative,
   retained behind the simpler owner-queue audit.
4. Park cross-card placement and Q6 pre-expansion/DPAS for their ownership,
   scratch, padding, implementation and memory costs. Neither is categorically
   closed by the earlier A4 or S75 results.

Root rejects the reviewers' unsupported approximately 6ms paired-draft /
2.4ms head decomposition and saving bound. No calibrated noise band or numeric
promotion threshold follows from S106-S114. The uninspected Q4/C6 image may
share a source helper without sharing its compiled lowering. Instruction
counts, encoded bytes/service and A4 deltas establish neither bandwidth/ALU
dominance nor a serving forecast. FP proof needs operand/dataflow/order and
materialization, not register/PC identity. No constant-buffer alternative or
multiformat expansion is selected; if the chosen spelling fails its compiled
mechanism gate, close that spelling.
