# S120 GU17 existing-image calibration, 2026-10-16

CONFIG -> Frozen S107 main064b63ce, unchanged IQ object09447b7e/archivecf710;
TG17 is IQ2_XS, Fmt17->vec_dot_iq2_xs_q8_1, LN8/local256/SG32. No new kernel.

COMMAND -> Root's first CPU audit stopped at v1 line59: a literal single
space after `name:` did not match actual padded ze_info. Its selected43272B
image/4754B metadata and frozen prep remain untouched. Root explicitly
authorized one routine parser correction/continuation. V2 uses an exact-name
whitespace regex and reuses that saved image, avoiding another main image
scan. Original strict comparator and source/receipt pins unchanged; small IQ
object rehashed and actual final Info/cache/guard/name pointer rebound from
ELF metadata. One bin/cpu-run command, exit0/0.244829s (wrapper rounds to1s).

RESULT -> PASS existing native reuse, zero decoder/build/GPU/model. Image
SHAe5836dfc..., text7488B SHAaecbb2eb...; native prefix7296 through EOT and
alignment, full metadata/options/codebook/data/relocs match retained S97.
Whole image/text remain raw-unequal; embedded SPIR-V/trailer differences stay
separate. Current GRF128/SG32/eu_threads8; SLM/barrier/spill/scratch fields
absent UNKNOWN. New selection records actual final host identities. Original
selected image offset was not saved before v1 failed, so remains null; the
v1 uniqueness assertion preceded its saved outputs. V2 does not invent that
offset or make a fresh whole-main hash/image-uniqueness claim.

Retained S97 native K loop is L2304..L6632, k increment8, exact current
computational body. Source has eight activation int32 words per weight part,
but compiler already coalesces them: ASMline223 is d32x4 load from A+0x4,
line336 is d32x4 from the SAME r18 address payload at A+0x14. Those offsets
cover the32 q8 bytes after block_q8_1's4 metadata bytes. Address arithmetic
lines147..152/171..190 carries36-byte Q8 block stride, separately from74-byte
IQ2_XS weight stride; no scalar-eight-load codegen defect is established.
Line457 loads activation half scale from the r18 block base; line458 loads
weight half scale through the separate weight-row payload. Lines464/465
convert halves,466 multiplies weight/activation scales,467 multiplies the
converted integer finish,468 adds to the lane FP accumulator. Weight codes,
four grid gathers, sign expansion and all8 DP4As remain separate weight work.
Signed integer /2 and /4 correction sequences remain at449..456. All anchors
refer to actual retained ASM lines and K-loop labels, not inferred timing.

VERDICT -> Cheap binding/calibration CLOSED, no new TG17 candidate selected.
Pairing gate/up could theoretically share those two activation vector sends
and one activation scale conversion, not weight decode/bytes/DP4As. There is
no measured dynamic send cost, cache traffic, occupancy or readiness saving.
Extra accumulators/SLM/barriers and fewer WGs can outweigh it. TG17 is23.18%
of whole-collection primary GU service, not a representative dominant path;
TG21/IQ3_S26.39% already has E4/E2/tail weight reuse that a larger proposal
must preserve. S100's large private TG17 result translated to only a small
serving observation. Do not build another narrow knob from this audit.
Primary dense MMVQ investigation belongs to the independently assigned agent.

An optional post-audit byte-offset annotation command had a Python syntax
error (`else16`) before execution and produced no file. It was not rerun;
no derived instruction byte addresses are claimed. Actual audit PASS and its
recorded raw/strict comparisons are unaffected and were not repeated.
