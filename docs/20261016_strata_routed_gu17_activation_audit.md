# S120: routed GU17 activation-load calibration

S120 closed a cheap existing-image audit. The compiler already coalesces the
eight source activation words into two vector-load sites; there is no missing
scalar-load fusion to fix. A possible paired gate/up reuse mechanism remains
unmeasured, and no new kernel, fixture or GPU run was selected.

CONFIG -> Frozen S107 main
`064b63ce75d7568db541af5cf24eee07f41d0e54ea49813af9958a816d75d5ca`,
IQ object09447b7e and kernels archivecf710683, also unchanged in S114.
TG17/LN8/KMtrue means IQ2_XS, local256/SG32 and32 output rows per WG.
Source geometry N2560/FF640 and S95 row-grid40 are recorded; this is not a
per-replay argument trace or weight-pointer attribution.

COMMAND -> Root's first CPU audit saved one43272-byte selected image and
4754-byte metadata, then failed at V1 line59: literal single-space `name:`
parsing did not accept actual padded whitespace. Root reported exit1/1s and
authorized ONE correction/continuation. V2 used an exact registered-name
whitespace regex, reused the saved image and retained strict source/build/
object/section-comparison gates. Actual final Info/cache/guard/name pointer
was rebound through ELF metadata. No new main image scan, archive walk,
whole-main/archive/model hash, decoder, build, GPU or model consultation.

RESULT -> The continuation returned0 in0.244829s (CPU wrapper rounded1s),
with the small IQ object hash matching the retained build identity. Current
selected image SHAe5836dfc and7488-byte text SHAaecbb2eb match the retained
S100 OFF identities. The7296-byte native prefix through EOT/alignment and
metadata/options/codebook/data/relocation sections match S97. Whole-image and
raw-text equality remain false; post-EOT immediate differences and embedded
SPIR-V remain separately classified. This is bounded native reuse, not
SPIR-V or universal whole-program equivalence. The original executable image
offset was not saved before failure: V2 records null, with initial uniqueness
validation supported by V1's executed prefix, not a fabricated fresh offset.

The original frozen source proposal incorrectly called TG17 IQ2_XXS and
cited its helper/formula. It is retained unchanged with an explicit addendum:
the current C++ type-name switch maps16->IQ2_XXS and17->IQ2_XS; Fmt17 calls
`vec_dot_iq2_xs_q8_1`, while Fmt16 calls `vec_dot_iq2_xxs_q8_1`. Actual TG17
uses two integer chains, eight signed DP4As and
`(sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4`, preserving signed truncation/order.
It does not use the XXS `sumi*ls/8` finish. Half conversions, weight-times-
activation scale, integer conversion/product, FP accumulation and LN8 XOR
reduction remain part of its arithmetic contract.

The reused actual native K loop is L2304..L6632, k increment8. ASMline223
loads d32x4 from A+4 and line336 loads d32x4 from the same r18 address payload
at A+20: the32 activation bytes after block_q8_1's4 metadata bytes. The
36-byte activation and74-byte weight block strides are separate. Line457
loads the activation half scale,458 the weight half scale;464/465 convert,
466 multiplies scales,467 multiplies the converted integer finish,468 adds
to the FP accumulator. Weight-code loads, grid/sign decode and eight DP4As
are separate weight work. These are static instruction sites, not measured
transactions, cache misses, dynamic send counts or latency. Metadata reports
GRF128/SG32/eu_threads8; absent SLM/barrier/spill/scratch fields are UNKNOWN.

Scope ranking uses only primary-device GU-port rows from the S95 canonical
whole collection, including non-steady work. Totals below sum disjoint event
service, not medians, reviewed-cycle format attribution or a critical-path
budget. The routed family and broad dense MMVQ are distinct predicates.

| Format | Whole-collection GU service ms | Share of this GU-only collection |
| --- | ---: | ---: |
| TG21/IQ3_S | 30.452649 | 26.39% |
| TG17/IQ2_XS | 26.745296 | 23.18% |
| TG16/IQ2_XXS | 24.550934 | 21.28% |
| TG22/IQ2_S | 20.838634 | 18.06% |
| TG18/IQ3_XXS | 12.790453 | 11.09% |

VERDICT -> Close this calibration without another narrow optimization run.
Pairing gate/up might share two activation vector loads and one scale
conversion, while adding live accumulators, barriers and fewer WGs; net cost
is unknown. TG17 is not representative of the largest format. A broader
proposal must preserve TG21's existing E4/E2/tail weight reuse rather than
replace it with a single-entry tile. S100's large private TG17 result produced
only a small descriptive serving delta; it does not forecast this mechanism.
No gain, timing, stability, default or promotion claim follows from S120.

The [lean evidence bundle](../strata/flash-next/lowlevel_probe/routed_gu17_activation_s120/README.md)
preserves the original attribution error, parser failure context/partial,
versioned correction/diff, exact actual comparisons and command/exit. An
optional post-audit offset annotation failed at Python syntax before execution
and was not rerun; no derived byte addresses or extra audit pass are claimed.
Full binaries and ASM remain external with known identities; only a bounded
operand excerpt is curated. Primary dense MMVQ work proceeds separately.
