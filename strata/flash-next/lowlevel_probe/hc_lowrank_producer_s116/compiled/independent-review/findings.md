# S116 independent selected host and candidate ISA review

CONFIG -> Six private HC classes, normal C16, sliced down, MORE=0, ILP=0,
LO_ONCE=true, default GRF, SG32, nsplit=1; T=1/2/6 fixtures only. Source packet
and production remain unchanged. This is offline compiled evidence, not a run.

COMMAND -> Read the actual four host disassemblies and complete candidate
reduce/UP instruction streams through EOT; derive selected byte anchors and
register-erased ordered FP signatures with derive.py. Read selected metadata,
actual source and old S115 control ASM. Read only the two small selected images
for literal instruction words. No decoder, build, GPU, model or main/archive
scan was added. Parent performed the one selected audit and two candidate
decodes; both decoder.exit files are 0. Status records decodes complete with
manual FP/order/ownership/resource review required, not numerical PASS.

RESULT -> Actual six distinct Info/name/image bindings and 12 distinct cache /
guard cells are retained. The original control computational prefixes match
S115 exactly: UP 82848 bytes, reduce 16272 bytes. Normalized execution/payload
metadata, options, constants and raw plus decoded relocations also match.
Raw complete text/image equality is FALSE: first text differences 82860/16284
are after EOT. This does not establish full-image byte identity. Norm/down have
retained exact source projections, not a newly established old binary match.

## Actual host selection and ownership

Object-relative reduce_only host-0: candidate flag arrives in esi and is copied
to eax at 0xe; test al at 0x51, false ->0x245. Candidate Info relocation 0x4a6,
cache read 0x148, guard read 0x139; control Info 0x4d8, cache 0x334, guard 0x325.
Direct submit relocations are 0x217/0x403. Both paths copy the SAME 0x4e8-byte
HCMultiS116 value (size immediate 0x49), preserving owner pointers and T.
Geometry computes round_up(T*324,128) in 0x13..0x3d; local128 at 0xd1/0x2bd.
T=1/2/6 therefore gives global384/768/2048 and 3/6/16 work-groups. These are
source-conditioned host arguments, not an observed dispatch or memory traffic.

UP host-1: flag test0x2b, false ->0x223; candidate/control Info0x488/0x4ba,
cache0x126/0x316, guard0x117/0x307; direct submit0x1f5/0x3e5. Both copy0x4e8
bytes and use global0xa000/local0x100 at 0x8b/0xaf and 0x27b/0x29f: 160 groups,
256 threads. Source instantiations are <16,false,true> and preserve root-sync
properties on both routes. Prefix host-2 copies the same struct at0x2e/0x39,
loads T at0x1e and scales by1024 at0x7f for norm global1024*T/local256.
Its final reduce_only call relocation0x3b1 passes the retained candidate flag.
Down invoker host-3 copies0x4e8 bytes, global0x5000 at0x214/local256 at0x24c;
Info relocation0x533, cache0x3e9, guard0x3da. Static binding is not GPU execution.
The exact relocation records and final linked pointers are in evidence.json.

The oracle uses one in-order queue (oracle.cpp158), separate route allocations
with checked spans, disjoint owner/token output pointers, nsplit=1 (line100),
and strict prefix/final snapshots. Thus the intended consumer begins after the
producer kernel completes, in eager and recorded whole-route order. This is a
source contract; no runtime ordering result has yet been observed.

## Producer raw operand, branch and FP ledger

Both reduce bodies retain 80 ordered :f additions, first using acc2:f and the
last materializing a :f GRF. Old first/last:1760/15152 ->r108:f; new:1952/15224
->r30:f. All80 register-erased addition signatures match in order. This check
retains acc2, operand regions/types/order/width and literals, but erases GRF
numbers; it is not an independent proof of every slice address. Source still
indexes (((sl*8+k)*324)+r), sl0..79. Raw part2 is read only.

New row predicate r<320 is cmp14544; branch15232 selects activation path for
these rows. Injection rows select inject_out, store the raw r30:2 at16176,
then goto16192 bypasses activation to17664. The null-w_inject skip is explicit
464/544/576/592; index bound384..416 prevents inactive rows. Source and fixture
checks retain actual token/row ownership and the untaken planes.

For low-rank rows, r30/r31 is the materialized raw FP32 payload of store16384.
Extra +0 at16240 consumes EXACTLY r30:f into acc0:f, followed by mul0.25 at16256
into r36:f; mul(-1.442695)16296 ->r20:f; math.exp16320 ->r14:f; +1 at16360
->r38:f. The instruction's raw scale word is 0xbfb8aa3b in both old UP1728 and
candidate reduce16296. Final division: invm16432/16624 with numeratorr36/37,
denominatorr38/39;14 madm refinement sites, seven per half, retaining the old
numerator/denominator/zero/one/refinement expression roles. Result r30/r31
is copied to r20/r21 at16800 and owner-lo store17648 publishes r20:2.

IMPORTANT actual schedule: raw part store16384 occurs AFTER extra+0, scale,
exp and denominator, despite source store-before-expression. It still consumes
the same raw r30:2; invm16432 reuses r30 only with {$1.src}. There is no reload
from m.part. Source disjointness and no external observer until completed
kernel are the ordering justification to be tested; do not describe a physical
store-before-compute sequence. Neither part store nor lo store is atomic.

Old UP raw d32 load1056 ->r38:f, extra+0 at1072 ->r28:f; with nsplit=1 the other
split additions are bypassed. Old scale1720 ->r42:f, exp-scale1728 ->r28:f,
exp1744 ->r30:f, +1 at1760 ->r44:f, invm1792/1968 and14 madm sites yield the
activation. New extra+0 destination acc0:f differs from old GRF r28:f; both
are encoded :f, but accumulator width/denorm/universal store-forward equivalence
is NOT inferred. The emitted mul0.25 matches the observed old lowering of
source /HC; this is not a universal source divide-vs-multiply assertion.
Signed zero, subnormal and nonfinite outcomes remain strict runtime obligations.

## Candidate UP and retained downstream arithmetic

Owner-lo selection uses k=i/320 and row=i%320, in contrast to producer raw rows
indexed /324. Candidate d32 owner-lo UGM load1568 ->r46/r47, SLM store1616 uses
the SAME payload and address2048+k*1280+r*4. Loop advances i by256. SLM fence
1728, UGM fence1760, barrier signal1824, sync.bar1840 retain the staging boundary.
There is NO FP ARITHMETIC before this barrier; :f movs are not arithmetic.
The former activation/global-lo store prologue is absent. Global lo now has one
producer placement per owner row; this is static source/ISA placement, not an
observed execution count or performance claim.

First w_up d32x4 load5072 reads from pointer r3.6:q plus row*640+lane*16; second
weight load5248 is guarded for lane<8. First SLM lo block load6000 feeds the
BF16 unpack/first eight FMA chain6048..6288, starting from0 via r4:f and ending
in r17:f. A separate zero-start second eight-FMA chain6384..6568 combines via
add6576. Old counterparts7448..7704 /7792..7976 /add7984. Lower/upper BF16 order
is preserved by shift16 / mask0xffff0000 and the corresponding lo lanes.

First XOR16/8/4/2/1 butterfly additions are6752/6808/6912/6992/7016; old
8160/8216/8320/8400/8424. Lane exchange uses indirect GRF access then 8-,4-,
2-,1-lane moves; the first address has source lane xor16 (xor setup retained).
Selected lane result uses predicate at7024; following token blocks preserve
same dot-then-butterfly ordering. Both source projections retain non-ILP order.

First retained pending-write gate: new12976 ->mul(-0.25),12992 ->mul(log2e),
exp13008/+1 13032, invm13104/13280 plus14 madm, multiply2 at13440, residual FMA
13472 ->r98:f, R_out store13504. Old14384/14400/14416/14432,
invm14480/14656,mul14816,FMA14848,store14880. The second retained sigmoid uses
new mine r32:f ->scale13536,exp13552,+1 13592,divide13680/13856; meanwhile
rv*wn at13576 then *rsc13616 ->r42:f; final multiply14064 and SLM store14080.
The source rv*wn*rsc association is retained. Later repetitions, including
last two exp79704/80224, remain; no old prologue-removal count is applied to them.

After second sync.bar80880 (old81552), SLM c0/c1/c2/c3 loads81160/81184/81216/
81240 feed +0, then c1,c2,c3 ordered additions81256/81264/81272/81296 ->acc0:f;
mul0.25 at81304 ->r12:f, mixed d32 store82080. Old mean81920/81928/81936/81960,
scale81968,store82752. The four-stream mean still includes the extra +0.

Whole downstream corroboration: after old/new first barriers, all1765 ordered
FP arithmetic signatures match after removing GRF and flag numbers and brace
scheduling annotations. Operand positions, subregisters, regions, signs,
immediates, :f type, accumulator names, width and mask remain. Counts BOTH:
mad1032,add404,mul57,exp16,invm32,madm224. This is stronger than count equality
but is NOT a cross-instruction GRF/SSA mapping proof, whole-UP byte equality,
or a universal numerical proof. Scheduling/register allocation has changed.
Old total UP exp17 becomes16; new reduce has1. These are static instruction
sites across different images, not dynamic SFU traffic, critical-path budget,
or a forecast of whole-model gain.

## Modes, resources, remaining gates

Both old/control and new kernels OR raw CR0 mask0x400004C0 (reduce280;UP328).
No denorm/rounding-bit interpretation is established. Selected FP arithmetic
is :f, no :df/:hf/:bf arithmetic observed. All six images carry the exact option
-cl-fp32-correctly-rounded-divide-sqrt. All six GRF128, SIMD32, eu_thread_count8.
Control/candidate UP SLM12288, metadata barrier_count1 but actual2 sync.bar;
reduce SLM/barrier fields absent UNKNOWN. Norm SLM64; down SLM absent UNKNOWN.
Spill/scratch absent UNKNOWN for all. Candidate reduce1253 static instructions,
EOT17712; candidate UP6376,EOT82176. No occupancy/dynamic-cycle inference.
Candidate data sections empty under the audited section definition; both have
two raw relocation type2 records at76/156 targeting
__INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; raw type semantics not invented.

VERDICT -> PASS_STATIC_CANDIDATE_REVIEW_FOR_BOUNDED_ORACLE_CONSIDERATION.
No new compiled mechanism blocker identified within this narrow source/ISA
contract. This grants no GPU authorization and proves no numerical result.
The existing12 ordinary base cases,36 historical anchors,A-B-A eager/captured
routes,raw parts/owner-lo/finals/guards/read-only checks remain mandatory.
Synthetic part2->actual reduce tests must print actual reduced raw bits; direct
control-UP raw special diagnostics cannot prove candidate cross-context cases.
Failure is drain/report/stop, without retry or counter-reset recovery. Timing
only after strict PASS and only completed whole-read scope; production unchanged.
