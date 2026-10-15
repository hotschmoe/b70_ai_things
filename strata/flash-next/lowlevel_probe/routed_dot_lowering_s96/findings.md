# S96 routed expert reuse audit, 2026-10-15

CONFIG -> Current audited d94e96ca, S95 primary routed port; no new arm.
S=/mnt/vm_8tb/b70/build/strata-ab/src-ep; source-pins.json has exact
paths/SHA256/observed QT16/17 names.

COMMAND -> Read current source/S95 derived summaries/S66 selection metadata;
no rawtrace/model read, ISA, compile, GPU, model calls or main/docs edit.

RESULT -> Routed GU/DOWN7.504ms and nonvocabulary MMVQ12.680ms are DISJOINT
name predicates, not nested service pools or additive removable wall budgets.
QT42 DOWN is Q2_0 (64-value blocks), about1.96ms/30calls per reviewed cycle;
GU includes QT16 IQ2_XXS,17 IQ2_XS,18 IQ3_XXS,21 IQ3_S,22 IQ2_S. Observed
port templates have LN8/KM1/SG32/local256; GU grid40x60, DOWN80x60 imply
1280 GU rows (640 gate+640 up) and2560 down rows. Grid60 is capacity, not
observed active group/duplicate-expert count. QT17 GU covers10calls/~1.01ms
and QT16 covers9/~0.90ms in retained12-cycle interior summary; diagnostic
service only, no projected savings.

Call binding: S/sycl/src/core/verify.cpp1434..1439 constructs layout from
per-layer expert_layout().fmt and calls native_expert_grouped on peer;
1474..1480 does the primary resident/staged groups. iq_kernels.dp.cpp4703
sets GU/down row bytes and gate/up/down offsets. Entry point5282..5465
switches L.gu_type/L.d_type -> launch_gu/down -> port launchers4315/4351.
grp_ptr[g] is ONE expert blob; grp_start[g:g+1] its contiguous entries.
ent_tok indexes Q8 input; ent_dst preserves output; gate/up offsets differ.
E4/E2 grouping never crosses g or a weight row, so immutable weight prep can
serve several activations of that SAME expert; actual reuse frequency is
UNMEASURED; no reuse across experts/mutable activations.

Already hoisted: row_entries1969..1997 and down2188..2210 dispatch E4/E2
then singleton. Multi18/21/22 GU and20/42 DOWN have prep outside the E loop
(row_dot_multi_iq1934..1960); codebook/sign/weight scales are reused per
E2/E4 chunk, activations/dot/finish remain per entry. Singleton Fmt::dot
still decodes. Thus QT42 is not a missing source-hoist case.

Distinct gap: Multi1587 defaults has=false; only18/22/21/20/42 specialize.
KqSplit1760 defaults false and has no16/17 specialization. Consequently
kEntryMulti<TY,KM>1912 is FALSE for16/17 regardless KM1. native_gu_kernel
2002..2040 calls row_entries, which goes straight to per-entry
row_dot_lanes810..817/Fmt16/17(735..738). Weight loads, IQ2 codebook/sign
unpack and scales recur inside each entry's k loop. Split16/17 is NOT used here. Compiler motion remains unknown.

Integer/FP contract: Fmt17(171..211) uses two ordered four-DP4A chains,
ls0/ls1 in0..15, then (s0*ls0+s1*ls1+(s0+s1)/2)/4; Fmt16(144..169)
uses one eight-DP4A chain then sumi*(aux32>>27|1)/8. Conservative signed
byte operands give |half-sum|<=262144, QT17 numerator<=8126464;
QT16 sum<=524288, odd scale<=31, product<=16252928: INT32 safe. Preserve
signed extraction, integer truncation, half-to-float conversion, d=dw*dx,
return d*sumi, per-lane k stride8 and XOR4/2/1; algebra alone is not a
compiled/runtime bitwise proof. Existing source already uses SWAR helpers.

DP4A chain: iq_kernels94 -> strata::dp4a -> dpct::dp4a(math.hpp833).
Intel branch explicitly sign-extends four int8 values then four mul/adds;
hardware dot folding is an IGC opportunity, not source proof. S66 selected
a DENSE mmvq_multi_kernel<21,C6> image, not either routed port. Attention's
13-tag binding is unrelated. No exhaustive main ISA exists here.

VERDICT -> GO source eligibility for ONE later selected existing-image audit:
d94e native_gu_port<17,8,true>, exact observed name in pins. Bind unique
registered name/image; inspect actual signed dot and decode/weight-load
placement across dynamic entry loop, resources/spill fields. No build or
candidate yet. Only if repeated decode survives consider QT17 E2/E4 reuse;
unchanged ledger/full original-candidate bytes/routing outputs required. No format/lane sweep, FP16 activation substitute,
permanent planes, bandwidth/occupancy/cause/serving-gain claim.
