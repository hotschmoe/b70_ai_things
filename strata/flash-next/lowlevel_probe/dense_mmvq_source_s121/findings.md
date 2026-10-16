# S121 primary dense MMVQ attribution and one integer-preparation lead

CONFIG -> S95 canonical859-row operation CSV, primary device #0, operation
contains mmvq, excluding WideQ5K grid62080 vocabulary head. This is the SAME
extended predicate as S95. BF16 MMVF/GEMM, HC and routed GU/down are separate.
Source/retained descriptors only; no model/trace payload, binary hash,
compiler/decoder/GPU or source edit. Opus depth3 rerun
is NOT selected: S21 already rejected it;64/24 aggregate two streams, not a
per-stream3-token window ceiling.

COMMAND -> derive.py reads the small CSV and retained S41/S57/S63 header JSON,
joins format/output-row sets to current dispatch/callers, and pins exact source
excerpts. ranking.json retains exact operation strings, counts, scope and
source-conditioned role sets; no per-event tensor/layer/phase join.

RESULT -> Whole-collection quantized nonvocabulary MMVQ:102 rows,9600 calls,
361.772043ms recorded service. Family ranking (ms): Q6_K111.049292,
Q4_K64.939024, IQ4_XS60.430027, IQ3_S56.396303, Q5_K28.147440,
Q8_017.596080, IQ4_NL16.711370, Q2_06.502507. Whole collection includes retained
capture/cold/other phases. Do NOT divide by a cycle count, add these sums to
CPU/peer time, infer physical traffic/idle or call them removable wall budget.

| Largest individual C6 operation row | Calls | Recorded ms | Retained metadata role set |
| --- | ---: | ---: | --- |
| Q6_K grid640 |696|43.823245|6144->2560;10 attn_output +19 ssm_out tensors |
| Q6_K grid2560 |336|28.765068|2560->10240;14 attn_qkv tensors |
| IQ3_S grid1536 |240|21.133954|2560->6144;10 attn_gate tensors |
| Q4_K grid2560 |192|20.115752|2560->10240;8 attn_qkv tensors |
| IQ4_XS grid2560 |264|17.762699|2560->10240;11 attn_qkv tensors |

Four subgroups/one row each bind C6 grid to M=4*grid. Header supplies K and
role set; grid alone supplies neither. In particular Q6 M2560 is NOT a shared-
down K640 ambiguity here: retained Q6 descriptors contain only the29 K6144
matrices for that row count. Source calls verify.cpp1070/1283 name their
ssm_out/attn_output consumers. Counts reconcile sets, not individual events.

Both largest Q6 shapes were A4-tested in S104/S107; matched serving
observation+0.427673%, defaultOFF/no repeat. This closes an A4 repeat, not all possible mechanisms on those shapes. Source1552..1640 already
hoists ql/qh reconstruction, signed6-bit centering and d*scale outside NCOLS;
u0/u1/ds are prepared once per column/kbx and shared across RPW rows. Each
column's distinct dot remains. There is no new weight-decode-across-columns
hoist. Source hoisting is not emitted-load/CSE/traffic proof. CENTER, global
Q6 RPW2 and packed Q6 are distinct already-reviewed mechanisms, not next tests.

Prior levers stay closed: IQ4_XS C6 RPW2 S59 was tiny/inconclusive;
Q4/Q5 C6 RPW2 metadata already declares spill/scratch; IQ3_S SWAR S65 was mixed,
and S75 SG16/DPAS was7.7-7.9x slower. Q8_0's17.596ms pool mixes columns1/2/3/4/6/8
and MTP projections; names/rows alone do not identify each input/role. Its
Wide32Q8 already hoists weight load across columns. No quant-format sweep.

## Distinct lead: exact input-only minimum-correction sums

WideQ4K::apply at1954..1960 computes two integer sum16(q8) values per lane/column/
kbx as dp4a4(ones,ua/uc,0), solely for the affine minimum correction. Each
sum uses four logical DP4A calls: eight total, independent of the weight ROW.
Weights/scales are already hoisted outside j at2095..2098. WideQ5K::apply1994
forwards to this SAME helper. FP expression must stay literally ordered:

    da*(dl*float(dot_lo)-ml*float(sum16_a))
      + dc*(dh*float(dot_hi)-mh*float(sum16_c))

A possible narrow producer adjunct stores each canonical Q8 block's TWO raw
signed integer16-byte sums AFTER rounding/clamping/int8 encoding, from the same stored q bytes. Keep all
36-byte Q8_1 bytes unchanged; no new weight format, requantization or altered
FP scale/subtract/add/butterfly. For arbitrary signed int8, each exact sum is
[-2048,2032], fitting int16; for native clamped[-127,127] it is[-2032,2032].
strata::dp4a uses signed int32 operands; integer conversion to FP32 is exact. Native ds[1] stores the FP16 FULL32 raw-activation float sum, NOT either
raw16 encoded-q integer sum and NOT a substitute.
No universal compiled equivalence follows; FP/store/load modes still need proof.

For N2560, adjunct size is4*(N/32)*C:1920B at C6 or640B at C2. These are logical
storage sizes, not physical traffic. Added loads, producer integer reduction,
scratch ownership/stride/lifetime/capture and readiness could outweigh saved
integer work. Compiler may already CSE/simplify the ones-dot; actual lowering
is UNKNOWN. Do not build a candidate before resolving that cheaper question.

The SAME helper serves Q5_K fullvocabulary C2: current
MtpPair tail quantizes two separate owner rows2198..2199, calls
native_mmvq_head2200 -> defaultOFF head-RPW2 fallback3959 ->native_mmvq type13
->native_q5_k_mmvq3290 ->try_wide<WideQ5K>3298 ->native_mmvq_wide_kernel ->apply.
WideK/defaultRPW1 and N2560/C2 are explicit conditions; RPW2/head/subset routes
must not be silently substituted. Head M248320 encoded bytes437043200
(416.796875MiB), source-conditioned; no payload read/physical traffic claim.
Its EXCLUDED S95 C2 row is42 calls/50.470608ms, median1201.562us; this head
context does not inflate the nonvocabulary ranking above.

## Cheapest discriminator and stop condition

Select ONE existing Q5_K C2/RPW1 image, not a format sweep: bind actual public
host branch, Info/cache/guard/name/image and constants/options before inspecting
whether raw-q8 sum16 sites survive, how they lower, and what predicates/data
loads feed each min term. The source-only audit-plan.md specifies this target.
Reuse retained selected ISA only after exact body/metadata agreement; no new
decoder is authorized by this note. If no exact retained decode exists, report
ISA_UNKNOWN to root before considering one bounded offline decode.

VERDICT -> Concrete new integer-preparation hypothesis; NO candidate/build/GPU
GO or timing forecast. Stop if compiler already folds the correction cheaply,
actual role/image binding is ambiguous, or auxiliary ownership/canonical-Q8/
FP-expression preservation is unsupported. Repeated quantizers in MTP embedding
versus hidden consume different normalized inputs; shared GU and GDN QKV/Z
already share one Q8 image. Global QFUSE remains blocked by batch-rec output
omission and noncanonical finite/clamp helpers; do not enable it as a shortcut.
A strict private oracle would still be needed after source/compiled approval.
