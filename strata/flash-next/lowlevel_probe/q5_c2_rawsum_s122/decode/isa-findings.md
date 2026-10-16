# S122 selected Q5_K C2 ISA: raw sums survive as expanded ALU

CONFIG -> ONE actual WideQ5K C2/RPW1 image selected from retained S107 main
064b63ce, kernel archive cf710683, object e4dafa84. Source-conditioned
N2560/M248320, generic RPW1; static binding does not prove runtime dispatch.
Image aa4d0202dd32cca82dad2cce6497c74125c640a0006f3f8dbe30ae04fc6bd859;
text 7e4b99f349256a49c64719845ae90e8a051b68c7d832907116c88bf348a67bff.
The decoder alias changes only text section naming, not code/data/relocations.
Root's separate decoder exited0 in9.858s; this review executes no decoder.

COMMAND -> Read COMPLETE selected ASM (632 lines), facts, exact metadata and
native_mmvq source1930..1997/2081..2106. Pin file records exact small inputs.
No new GPU/build, model/trace/binary/archive scan or candidate implementation.
ASM references below are original 1-based lines, labels are original byte
anchors. This is actual lowering evidence, not inferred source instruction cost.

RESULT -> Source dp4a4(ones, ua/uc,0) is strength-reduced, NOT eliminated.
All16 static DP4A sites are WEIGHTED Q5*q8 dots, four for each ua and uc in
each of the two unrolled columns. No ones-operand DP4A survives. The four raw16
signed-q sums each lower to16 byte-pack MOVs,16 signed-byte-to-dword MOVs,
one ADD and seven ADD3 instructions (40 vector instructions per sum;160 across
four sums in the static loop body). These counts exclude int-to-float conversion
and FP correction. They are not elapsed cycles or a throughput forecast.

Operand provenance, retained exactly by original lines:

- Weight Q5 high bits:139 load r4..r11 from qh at encoded+0x10;
  low nibbles142 load r12..r19 at qs+0x30. Shifts/masks/BFN147..185
  assemble lo weights r72/r70/r68/r66 and hi r80/r78/r76/r74, shared
  by both columns. Scale/min branches188..246 are lane-g dependent.
- Column0 activation ua:286 loads d32x4 into r12/r14/r16/r18;
  uc:291 loads into r4/r6/r8/r10. Addresses derive from x payload r3.1,
  kbx*8 Q8 blocks, sb*36 and half*16, plus qs+4 (268..290).
  ds[0] loads292/293 are d16 at base and base+36, not ds[1].
- Column0 weighted ua DP4As297/300/313/321 produce r60; raw ua bytes
  are packed298..320 and signed-extended/added307..340, yielding r18.
  Weighted uc341..354 produces r14; raw uc pack/sign/add344..386 yields
  a DIFFERENT sum r14 after weighted conversion to r4 at363.
- Column1 has column-stride address advance401..408 (n_in/32*36),
  distinct ua/uc loads409/414 and ds415/416. Weighted ua419..436 gives
  r62; its raw sum pack/sign/add421..462 yields r30. Weighted uc468..481
  gives r16; its raw sum471..511 yields r12. Columns cannot share sums.

This is not duplicate work within one ua/uc or accidental repeated same-input
column. It is row-independent setup repeated in separate row-owning subgroups.
RPW1 has no cross-row subgroup reuse, and compiler CSE cannot span workgroups.
The K loop is L1520 through backedge525, kbx increment4 at294; source lane
sub=lane>>3/g=lane&7. Short/zero-K and output-row guards remain in the ASM.

Actual FP ledger: dm half conversions250..258; scale/min integer conversion
and multiplication253/260/263/267 produce r46=dl, r44=ml, r58=dh, r56=mh.
For column0,385/387 convert raw ua / weighted ua to FP32;388 computes
-ml*rawA;390 MAD combines dl*dotA with that rounded product. uc conversion
is363/389;391/394 do -mh*rawC then MAD dh*dotC. ds[0] half conversions
395..399 feed398 dc*termC then400 MAD da*termA plus that product.
Column0 accumulation is417 (after next-column loads), not part of the MAD.
Column1 repeats465..466,512..523 and separate add524 into r90.
The actual MAD contractions/order must be preserved and rechecked in a future
candidate; matching the parenthesized source expression alone is insufficient.
No universal source FP equivalence, signed-zero/denorm or NaN claim is made.

Two independent subgroup reductions follow L7024/L7632: XOR16 via indirect
register read then permutes8/4/2/1, each followed by FP ADD; lane0 stores574/614.
No altered tree, reassociation, output ownership or accumulator order is licensed.
CR0 line14 ORs raw0x400004C0; exact denorm/rounding interpretation is UNKNOWN
here. Build option is -cl-fp32-correctly-rounded-divide-sqrt. Metadata gives
SG32/GRF128/EU threads8; SLM/barrier/spill/scratch fields absent = UNKNOWN,
not zero. No barrier opcode appears in the selected computational body. EOT
is619; following MOV-null/illegal bytes are outside computation.

VERDICT -> ONE narrow first candidate: runtime-uniform, kernel-captured
int ones_word=0x01010101, forwarded to the existing strata::dp4a calls in
place of the compile-time literal. This aims to retain native packed DP4A
lowering, not change the weighted-dot helper or introduce a sidecar, scratch,
new producer or global layout. Current source with constant ones already
expands, so rebuilding the unchanged literal is not a test. Root assigned
source-only private Q5C2 preparation separately; no prototype/build runs here.

Cheapest falsifiable gate: candidate actual selected-image/host-ABI audit must
show the new4B scalar kernel payload actually forwarded/captured with exact
value0x01010101, native correction DP4As using that operand, removal of the
four expanded chains, and unchanged weighted operands/FP ledger/tree. Check
resources and any inserted loads/register moves; don't assume the backend
will leave a runtime-uniform operand opaque. Failure to get the intended
lowering closes THIS spelling before GPU. Signed int8 dots/sums are exact
integer identities within int32 range; that is NOT universal compiled FP
output equivalence. Full strict output/intermediate dot+rawsum tests, fresh
A/B/A replay and byte extremes still precede timing. A raw16 signed-q sum is
[-2048,2032]; existing FP16 ds[1] is a full32 raw-activation float sum and
cannot substitute. Reusing exact encoded-q sidecar sums is a larger alternative,
parked behind the payload-only discriminator. Static40 instructions !=cycles;
no speed, bandwidth, traffic or critical-path attribution follows from counts
or encoded matrix size. No automatic prototype or numerical qualification.
