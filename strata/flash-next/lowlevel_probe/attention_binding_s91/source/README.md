# S91 private score-only reduce12 source preparation

CONFIG -> Current public qsa_decode_attn INT8, SG32/256threads/CHUNK64/G12,
24queryheads/2KVheads/HD256. Preserve eager query batch/NC1, all K loads and
8-term partial expression, SLM arrays, probability/softmax, ascending value
FMA sequence, scratch layout, merge and public queue launch properties.
STRATA_ATTN_LANECELL must be ABSENT, asserted in leaf. No production patch,
source toggle, graph or format expansion. Source preparation is not GPU proof.

COMMAND -> Read current source and S13 fixture; mechanically generate private
score-only loop/helper delta. Remove the added helper and restore the old score
loop: ENTIRE current source reconstructs exactly (source-change.json). Current
public source SHA00bf1c45... is BYTE-EQUAL retained S13 baseline. Freeze the
CURRENT251888-byte qsa_decode_attn fatobject; ar extraction from the actual
production kernel archive equals it exactly. OldS13standalone251784-byte
object is NOT substituted. No payload/rawtrace, compiler/GPU/model execution.

ROOT BUILD RECIPE (only after actual source review):
  cd /mnt/vm_8tb/github/b70_ai_things
  p=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/attn-score-reduce12-s91-prep
  bash "$p/build.sh"
Pinned e41 image/networknone/NO DRI/CPU4/UID1000/diskTMP under cpu-run.
One supported public-attention-only prelink from existing fatobject, private
candidate TU plus fixture compile, private prelink and ordinary host link.
NO public TU rebuild, common8f rebuild, engine or library mutation. There is
no existing attention-only standalone wrapper among retained S84/S88 outputs
(their public wrapper contains QSA selection, not attention). Reusing IR still
creates NEW AOT images; it is NOT reuse of an old GPU binary. Root must inspect
actual map/wrapper retention/selected symbols and device images after build.
Host names qsa_decode_attn_{batch,step,scratch_floats} are macro-renamed to
reduce12_{batch,step,scratch} in ONLY private TU; function-local kernel IDs
therefore have separate parent-function scope. Device/helper ODR equivalence
is not proved by this source prep or an nm listing. No stubs or dummy APIs.
No build has run; baseline/source/object identities are exact, not inferred.

Precise changed computation and pair/tree proof:
Let P[h,l] be the original eight-term lane partial for one cell/head/lane.
Candidate computes EXACTLY the same P expression and loads for h0..11;
heads12..15 are initialized0 and never published. All32 lanes participate.
At XOR16, r8[i] at lane l selects head i+8*bit4(l), and adds that head's
P[l] and P[l xor16]. The shuffle SOURCE chooses the complementary array
index: at l xor16 its high bit flips, so both operands belong to SAME head.
At XOR8, r4[i] similarly selects i+4*bit3(l)+8*bit4(l) and combines the two
corresponding r8 subtrees. XOR4 selects i+2*bit2(l)+4*bit3(l)+8*bit4(l).
XOR2 selects bit1(l)+2*bit2(l)+4*bit3(l)+8*bit4(l)=floor(l/2).
Final XOR1 combines lanes whose selected head is equal, yielding that head's
full original XOR16/8/4/2/1 pair tree. Lane2h publishes head h, only if h<12;
odd lanes and heads12..15 do not write. Each original sp[h][cell] has exactly
ONE writer. Differences from baseline lane0 are operand swaps within the
same binary add tree, not reassociation across tree levels. Addition of the
same finite operands with the same rounding/FTZ behavior is commutative;
NaN payload semantics/generated nonfinites are outside this equality proof.
Expression identity is not proof that compiler scheduling/contraction is
unchanged: runtime full scratch/output bit checks remain mandatory.
Logical shuffle expressions: old12*5=60; new8+4+2+1+1=16 per lane/cell.
This is NOT an instruction census, bottleneck proof or predicted speedup.

Mask/ownership/barrier invariants:
The existing subgroup-uniform c>=n_here or negative-page branch writes12
-FLT_MAX sentinels and continues, unchanged. n_here<=0 publishes only empty
m/l as before, with initialized unused scratch. Existing workgroup barriers
before/after score/softmax are unchanged; no new event/wait/barrier/submit.
No root-group feature is added/removed. Same per-query selection/page table,
slot and scratch indexing. Only INT8 is executed by fixture; private source
retains generic unused modes but they gain NO new qualification.

Fixture scope and strict gates:
attn_score_probe.cpp adapts exact tracked S13fixture: same RNG order/seeds
7/31/109,12cases =>36. Queries1/4/12/32, width0/1/11/63/64/65/1999/2051,
cap64/128/2112, page4/512, shuffled page tables and negative pages. Preserve
original small one-query FP64 softmax oracle unchanged, including its ORIGINAL
2e-5 sanity threshold; it is NOT a tolerance for candidate acceptance.
For each case: full output AND entire initialized scratch bitwise equality,
finite outputs/scratch, own initial repeat, exact query sign-bit flip with
changed-state observability (exceptwidth0), own changed repeat and restored
initial full bytes. Guards and full readonly inputs checked after each
correctness snapshot and after timing. Input sign-bit write is explicit and
its readonly witness follows the actual current state; original bits restored.
All allocations have4096-byte prefix/suffix guards; prefix is64-byte multiple
and actual data alignment64 is asserted, preserving float4/uint2 alignment.
Both outputs are poisoned and scratch zeroed before each correctness phase.
Same production merge+full scratch serve as score/probability ownership gate.

Timing retains S13's4warm+20recorded repetitions/arm per case, alternating
order. Every recorded whole chunk+merge enqueue/wait observation is printed
raw, not best-only. Per-case correctness precedes that case's timing; no timing
is accepted if any later case fails. Full final bytes/guards/readonly checked
again. No per-stage waits, extra trace or graph, different formats, new epsilon,
cache flush or serving request. Host completed spans include enqueue+wait and
are not isolated device service; repeated resident pools differ from serving.
Actual future run must use quietCPU OUTSIDE gpu-run --card0 with physicalmask0
and selectorlevel_zero:*, owned-container cleanup and health as prior oracles.
No run wrapper/server/health/GPU touch is part of this preparation.

Prior rejection boundaries and result:
S11 changed outerquerybatch32->128/256. S12 changed lane/cell score ownership
with large local arrays. S13 tested probability transpose/four-value-loads.
ALL remain unchanged here; do not bundle pre75 NC2/query-SLM reuse or its
value loop. This is only the existing reduce-scatter communication algorithm.
Extra live part[16]/r8/r4/r2 can raise register pressure or spill; no resource
or speed conclusion from fewer source shuffles. Existing pre75/V100 result
is not BMG evidence. Current S90 attention pool motivates the target but is
instrumented whole-request service, not a removable critical-path budget.

RESULT -> SOURCE-ONLY, scripts bash syntax/ASCII/byte reconstruction checked.
No compilation, GPU/runtime correctness, timing or model consultation yet.
VERDICT -> ROOT FULL READ + ACTUAL MODEL REVIEW BEFORE ANY BUILD/GPU. Main
source/libraries and all old rejected sources/evidence remain untouched.
