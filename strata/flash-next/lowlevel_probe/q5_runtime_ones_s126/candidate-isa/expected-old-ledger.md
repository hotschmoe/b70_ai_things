# S126 independent selected-ISA comparison obligations

CONFIG -> Actual V3 EXE66fae0f1087f04f5443d8276edf1722588ed155eeac728251c1933817122ab2b
is root-reported built0/23.1255s; no independent binary hash/read here. Selected
images/metadata/one candidate decoder still pending. This is a precomputed old
operand ledger, not candidate acceptance. Full S122632-line original ASM was
already independently read; no duplicate ASM copy or new decode is needed.

COMMAND -> Reuse S122 exact operand windows/findings and read V3 README40..53.
Public and private-original must first match S122 through8128B (EOT included),
with metadata/options/data/relocs; raw fulltext8320B/image may differ only if
root explicitly accepts proved post-EOT differences. Meaningful drift HOLD.

RESULT -> Old semantic table (original ASM lines):

| Value | Column0 origin | Column1 origin | Required candidate invariant |
|---|---|---|---|
| lo Q5 weights | r72/r70/r68/r66 from147..185 | Same four regs reused | Same packed signed weight bytes/order |
| hi Q5 weights | r80/r78/r76/r74 from147..185 | Same four regs reused | Same packed signed weight bytes/order |
| ua Q8 words | d32x4 load286, r12/14/16/18 | d32x4 load409, corresponding r12/14/16/18 | Same signed bytes from qs+4+16*half |
| uc Q8 words | d32x4 load291, r4/6/8/10 | d32x4 load414, corresponding r4/6/8/10 | DISTINCT adjacent Q8 block, not ua reuse |
| dotA | DP4A297/300/313/321 ->r60 | 419/420/423/436 ->r62 | Four weighted dots, ordered int32 accumulation |
| rawA | expanded298..340 ->r18 | 421..462 ->r30 | Four new dots with captured ones_word, same exact signed sum |
| dotC | DP4A341/342/343/354 ->r14 | 468/469/470/481 ->r16 | Four weighted dots, ordered int32 accumulation |
| rawC | expanded344..386 ->r14 after dotC conversion363 | 471..511 ->r12 | Four new dots with same payload, DISTINCT signed sum |
| dl/ml/dh/mh | 253/260/263/267 ->r46/r44/r58/r56 | Shared weight scalars | Same half/int conversions and FP multiplies |
| da/dc | d16 loads292/293; half conversions399/395 | d16 loads415/416; conversions522/518 | ds[0], not ds[1] or FP16 raw sum |
| acc | r88 += term at417 | r90 += term at524 | Separate ordered FP ADD each kbx iteration |

FP expression in ACTUAL lowered order: rawA->float, MUL(-ml,rawA),
MAD(dl,float(dotA),that product)=termA; rawC->float, MUL(-mh,rawC),
MAD(dh,float(dotC),that product)=termC; MUL(dc,termC),
MAD(da,termA,that product); then separate ADD to column accumulator.
Column0 conversion/MUL/MAD385..400, column1 conversion/MUL/MAD463..523.
Compiler register renaming/interleaving okay only with explicit operand proof;
new contraction/reassociation/changed half conversion/tree is HOLD, not source
algebra equality. CR0 raw OR0x400004C0; do not infer denorm semantics.

Kbx starts lane>>3, increments4 at294, backedge525 toL1520; source g=lane&7.
Both independent XOR16/8/4/2/1 FP ADD trees atL7024/L7632 and lane0 stores574/614
must match. Old resources SG32/GRF128/EU8; missing SLM/barrier/spill/scratch
fields UNKNOWN. No static count/sends proves cycles, bandwidth or traffic.

VERDICT pending -> Need actual sixth signed4B scalar metadata and host argv
forwarding, then device payload read traced to correction dot operand; metadata
existence or expected32 DP4As alone cannot establish this. Trace ALL16 weighted
and all correction chains; reject remaining expanded rawsum chains or payload
folding. New scalar offset/register must be READ, not predicted from source.
Record any resources/waits/moves/loads changes without latency attribution.
No prototype/GPU/decoder/build invoked by this review. Candidate strict oracle
and full output equivalence remain separate, even if compiled ledger matches.
