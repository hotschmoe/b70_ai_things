# Correction to frozen GU lead and scope ranking

The earlier frozen `routed-gu-paired-tile-next-source/findings.md` incorrectly
called TG17 IQ2_XXS and cited the XXS helper/formula. Preserve that packet as
history; this correction supersedes those statements. No code/test relied on
them. Current explicit C++ type-name switch gguf_reader.hpp:86..89 maps
16->IQ2_XXS and17->IQ2_XS. `Fmt<16>` calls vec_dot_iq2_xxs_q8_1; `Fmt<17>`
calls vec_dot_iq2_xs_q8_1. Both qk256/ipb8/step2, but different formats and
integer arithmetic. A shared loop geometry does not establish equivalence.

Actual TG17 helper iq_kernels.dp.cpp:175..210 uses four16-bit codes with
9-bit IQ2_XS grid indices and7-bit sign indices; two nibble scales ls0/ls1;
eight signed DP4As in TWO integer accumulators. Its integer finish is
`(sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4`, with the original signed division
order/truncation. Then half weight conversion times half activation scale,
then float integer conversion/product, lane FP accumulation and LN8 XOR4/2/1.
This is not XXS `sumi*ls/8`. Activation eight int32 words and ds[0] are a
possible shared operand set for two weight rows; actual sends/extracts and
lowering must be read from the bound existing native image before a claim.

S95 PRIMARY device, WHOLE canonical collection, disjoint GU-port event totals:

| TG/type | Events | Sum service ms | Share of this GU-only collection |
| --- | ---: | ---: | ---: |
| 21/IQ3_S | 377 | 30.452649 | 26.39% |
| 17/IQ2_XS | 290 | 26.745296 | 23.18% |
| 16/IQ2_XXS | 261 | 24.550934 | 21.28% |
| 22/IQ2_S | 290 | 20.838634 | 18.06% |
| 18/IQ3_XXS | 174 | 12.790453 | 11.09% |

All LN8/local256, row-axis40, group axes60/30/10. These are collection totals
including non-steady work, not reviewed-cycle format attribution, a wall-time
budget, sum of medians or forecast. Exact rows and arithmetic in adjacentJSON.
TG17 is neither a majority nor the largest GU format. S100's private40.6%
TG17 improvement yielded only descriptive+.3078% serving; that warns against
promoting another narrow format candidate. It does not predict this lever.

Priority -> ONE existing TG17 audit only as cheap compiled calibration/reuse;
no TG17-only kernel/fixture/build/GPU selected by this packet. The larger
paired-GU idea would need to cover dominant TG21 without losing its CURRENT
Multi<21>::has=true E4/E2/tail weight reuse. A single-entry tile must NOT be
ported blindly; pair two independent weight preparations while keeping every
entry's original two-integer-chain/finish/FP order, with resource pressure
explicitly checked. TG16/18/22 require their own format-ledger/selection proof.

Shared expert is a separate path: shared_expert.dp.cpp:313..349 already has
fused SwiGLU/Q8, and native_mmvq_pair currently supports only type8. The S95
broad-MMVQ12.680488ms and routedGU/down7.504495ms are disjoint family service
summaries, not additive wall budgets. No whole-family/dense/shared benefit
follows from this narrow GU audit. If actual TG17 ISA leaves only the tiny
consumer-launch removal or poor occupancy, close the idea here. S116 HC has
execution priority. No new numerical fixture or format sweep is prepared.
