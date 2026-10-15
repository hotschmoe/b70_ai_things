# S83 Q6 native centering ISA audit (2026-10-15)

CONFIG -> CPU-only EXISTING S55 native C6/RPW1/A2 image; no candidate compile,
GPU, serving, weights, source edit, full fatbinary scan or common-archive hash.
COMMAND -> bin/cpu-run bash decode-v2.sh; pinned e41 image, networknone,
noDRI,4CPU limit/OMP/MKL threads4, diskTMP. Direct seek image771 at39553776,
43432B. Validate topELF bundle bounds, selected ELF section table, exact kernel,
ze_info offset/size/SHA, build options, old frozen S55 SHA/stat identity.
RESULT -> Decoder exit0, CPU lease10s. Selected image
f72e974271d2dac1e8704e45c3f6b6e33d021d5e98db0c5d51d653f71a15be27;
16576B target text SHA
ce58ba51362f5619ddea5f45ec2d6d3ac1fa943eb53325fef04619061a99b309.
VERDICT -> Compiled saturation cost survives. GO private SOURCE preparation
for the exact two-expression packed recenter change; build/GPU requires root GO.
No latency, dynamic instruction cost, occupancy or physical traffic claim.

## Actual lowering and source binding

The complete raw target asm and metadata are in original-disasm outside git.
centering-excerpts.json gives numbered exact slices; report.json records all
static sites. At target ASM272..281 one byte is extracted into signed byte,
widened to word, added with-32, checked for signed overflow by XOR/bfn.0x40
with-128, truncated/reextended and compared. At286 bfn.0xD8 selects saturated
or ordinary result; repeated for the other bytes. At748..761 word results are
packed back into dwords consumed by DP4A. Lines625..644 show another byte
sequence;708..763 show the last packed word/repack/DP4A path.

Across this one static loop body there are exactly32 add-word-minus32,
32 overflow-mask bfn.0x40(-128),32 word cmp-ne-zero and32 bfn.0xD8 select
sites, consistent with8 packed calls*4 bytes. The48 DP4A sites match6tokens*
8 integer-dot calls in current source. These are decoded static sites, NOT
iteration counts, dynamic issue counts, profiler timings or a savings model.
The unwanted bytewise overflow/select/repack operations were NOT folded to
root's packed add/XOR expression by the existing compiler.

S83's masked bytes are0..63, subtract32 stays[-32,31]. Root's unsigned packed
  (u+0x60606060u)^0x80808080u
is exact: each byte+96<=159 causes no carry between bytes; XOR128 gives the
same signed-byte result. The full source-only function/proof already frozen
in decode-criticalpath-s83-loop-supplement changes only these two expressions;
all original A2 loads/Q8/scales/DP4As/FP expression/laneKorder/tree stay intact.
Candidate compilation could still reschedule or introduce costs; none measured.

## Identity, helper sections and limitations

Only51151B were read by extractor, no other image reparsed. Retained frozen
S55 executable bb066900... size92071496/mtime1792065644153600176 match index;
whole executable SHA is inherited from its existing root record, not rehashed.
Selected ze_info SHAd73b7b95... exactly matches frozen S56. Actual metadata:
GRF128/SIMD32; spill/scratch fields absent, not proof of zero or occupancy.
Selected image hash is checked before/after offline decode, pinned image ID
readbacke41... . source/build options remain -cl-fp32-correctly-rounded-divide-sqrt.

The extra.text.Intel_Symbol_Table_Void_Program is320B, SHAa40507c5..., and is
named separately in ze_info. Its decoded body is preserved, including padding/
illegal decoder markers after EOT. Target contains no call/calla instruction.
We do not erase other sections, assume they are irrelevant, claim no helper
registration risk, or perform a whole-program ODR/function-equivalence proof.

First setup attempt exited1/CPUlease0s because the extractor incorrectly
required exactly one.text section, before export/container/decoder. Original
scripts and failure remain. Root authorized the versioned exact-target-section
filter and unexecuted --cpus4 typo repair to --cpus4's correct form --cpus 4;
corrected single decode passed. No GPU experiment was retried. Future extractors
must allow explicitly inventoried helper sections and select target by name.

## Minimum next discriminator, not a new campaign

Prepare one private C6/RPW1/A2 recenter-only adapter with the ORIGINAL Q6
control and public native anchor, reusing S44 real fixture geometry. Actual
compile/ISA must show the intended packed expression, unchanged dot/FP tree
and no new spill liability. Full strict original/public/direct/own initial+
changed graphs/repeats, all outputs/Q8 readonly/guards precede timing. No new
tolerance, packing, format/knob sweep, head/MTP replacement or serving change.
Historical S31 Q6 coverage71calls/3.8256372ms ranks this lead; it predicts no
whole-model effect and does not claim current dispatch. Root owns any next GO.
