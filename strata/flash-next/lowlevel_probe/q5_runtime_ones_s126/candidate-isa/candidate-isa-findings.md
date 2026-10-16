# S126 actual candidate ISA: packed correction mechanism passes

CONFIG -> Actual V3 executable66fae0f1 (root-built0/23.1255s), selected
RuntimeOnesQ5C2 image6069468cf1b16aa42e7a868b0e928b8c512784485f3b7c553bf4df78dd6612a7,
27816B. Root's ONE offline decoder completed0/10.153557s, input unchanged.
This independent review reads COMPLETE510-line actual candidate ASM and exact
metadata/options against old S122632-line operand ledger. No new extraction,
decoder, build, GPU or binary/archive/model scan. References below are actual
1-based ASM lines; labels are original byte anchors. Small exact pins retained.

RESULT -> PASS_LOCAL_COMPILED_MECHANISM, NOT overall audit/GPU GO. Metadata
arg5 is signed int4B at offset32; first three pointers/N/M remain0/8/16/24/28.
Candidate has an extra cross-thread SEND12 into r4, following inline32B copied
at8. The live first dword r4.0 is the trailing ones payload. All16 correction
DP4As use uniform r4.0<0;0>:d. No intervening write to r4.0 through the K
loop: line77 writes only r4.1 (next dword), then r4.0 is reused after loop408.
Thus payload is USED, not merely present in metadata. Exact01010101 value is
host argv/launch proof's separate obligation; device does not equality-fold it.

Actual32 static DP4As split16 weighted and16 correction. The FOUR expanded
40-instruction signed-byte pack/extend/ADD/ADD3 chains are gone. No ADD3
occurs anywhere in candidate computation. This is a compiler mechanism result,
not160 cycles saved, dynamic traffic, bandwidth evidence or a speed forecast.
The candidate also reschedules scale/min branches and loads, so no all-body
identity or zero-overhead-extra-payload claim is warranted.

Packed operand mapping:

| Semantic value | Candidate exact lines/result | Old S122 semantic match |
|---|---|---|
| Weight lo | Q5 decode149..183 ->r64/r66/r68/r72 | old r72/r70/r68/r66, same four words |
| Weight hi | same decode ->r62/r60/r58/r56 | old r80/r78/r76/r74 |
| Col0 ua | load315 ->r20/r22/r24/r26 | old load286 ->r12/14/16/18 |
| Col0 uc | load318 ->r12/r14/r16/r18 | old load291 ->r4/6/8/10 |
| Col0 rawA | DP4A324,325,331,335 ->r8 | old expanded sum ->r18 |
| Col0 rawC | 326,329,333,337 ->r22 | old expanded sum ->r14 |
| Col0 dotA | 328,332,336,340 ->r10 | old weighted sum ->r60 |
| Col0 dotC | 327,330,334,338 ->r20 | old weighted sum ->r14 before conversion |
| Col1 ua | load368 ->r20/22/24/26 | old load409 |
| Col1 uc | load367 ->r12/14/16/18 | old load414 |
| Col1 rawA | 377,381,384,388 ->r16 | old expanded sum ->r30 |
| Col1 rawC | 375,379,383,387 ->r10 | old expanded sum ->r12 |
| Col1 dotA | 378,382,385,391 ->r8 | old weighted sum ->r62 |
| Col1 dotC | 376,380,386,389 ->r20 | old weighted sum ->r16 |

All packed inputs/results are signed d operands, zero then ordered four-dot
int32 accumulation. Q5 byte values0..31 remain signed-positive; Q8 bytes use
the same signed interpretation. Exact signed integer sums for ones1 are within
int32 range. This alone does not prove universal FP output identity.

FP/dataflow ledger matches original semantics despite register renaming and
interleaving. Weight dm load148: low half d ->r2 then float293/294; high half
min ->r52 then float290/291. Two source scale/min integer branches produce
r14=s0,r16=m0,r12=s1,r20=m1 (before reuse). MUL299=r52=ml;
300=r50=dh;301=r54=dl;306=r48=mh. Original scalar mappings were
ml=r44,dh=r58,dl=r46,mh=r56. Branches use sb=lane&6 and sh=sb|1;
these source-enumerated values are0/2/4/6 and1/3/5/7, respectively.

Col0: rawA conversion339; MUL344(-ml*rawA); dotA conversion346 and
MAD351(dl*dotA + that product)=termA. RawC conversion341; MUL345(-mh*rawC);
dotC conversion343 and MAD348(dh*dotC + product)=termC. ds[0] loads319
(dc) and320(da), half conversions347/353; MUL352(dc*termC), then
MAD354(da*termA + product). Separate acc ADD369. Col1 follows rawC390,
rawA392,dotC393,dotA397; MUL394/MAD396 forms termC; MUL395/MAD400 forms
termA; ds[0] loads363/364 and conversions372/399; MUL398 then MAD401;
separate accumulator ADD402. This is the old MUL/MAD contraction boundary,
not a new multiplication reassociation or accumulator FMA. Half values remain
halves until explicit FP32 conversion; ds[1] is not substituted.

K loopL1600 starts lane>>3, increments4 at321, compares same n_in/256
and backedges403; g=lane&7 and weight row stride176*n_in/256 unchanged.
Two separate accumulators r74/r78, both initial+0, feed XOR16/8/4/2/1 FP
ADD treesL5168/L5776. Indirect register bases0x1280/0x1380 select those
accumulators. Lane0 stores452/492 use original column-major j*n_out+row.
No cross-column sum, altered reduction tree or extra output store is observed.
EOT497, nextL6272 is trailer; computation prefix6272B vs old8128B.
CR0 line15 ORs SAME raw0x400004C0. Option is SAME
-cl-fp32-correctly-rounded-divide-sqrt. No interpreted denorm/NaN/signed-zero
or general binary-equivalence claim. SG32/GRF128/EU8 match old metadata;
SLM/barrier/spill/scratch absent UNKNOWN, not0. Payload load/branch scheduling,
scoreboard waits and code size differ; only actual strict/timing can assess cost.

VERDICT -> Candidate realizes the desired packed correction lowering and
preserves the tracked weighted/FP/K/reduction/store ledger. Proceed only once
root closes public/private-original prefix/options/metadata comparisons and
auxiliary SDK-wrapper ABI bookkeeping (current original audit HOLD retained).
This review does not waive those gates or qualify runtime correctness/speed.
No automatic new experiment/prototype; strict finite oracle remains separate.
