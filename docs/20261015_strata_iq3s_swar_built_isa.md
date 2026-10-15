# S68: built IQ3_S C6 SWAR instruction audit

CONFIG -> Offline audit by /root/ple_upstream (native Codex) of the existing
S65 executable, SHA256
`4a81f24dcbb2dee280cfe009035e58f8c605a8f47cf970c5d080e8242ca30902`.
Select only `PrivateC6<false>`/`PrivateC6<true>` and verify the retained public IQ3_S C6
image. These are compiled generic-K classes for the two S65 projection shapes,
K2560/M6144 and K6144/M2560, C6. This document records static artifacts.

COMMAND -> ELF metadata seeks indexed1088 images and read5988737 bytes,
including the three selected images. Extraction took0.045694118 seconds.
Actual S65 build records supply executable identity; size/mtime stayed fixed,
and selected image/text hashes were computed directly. No redundant complete
executable hash was spent.

One `bin/cpu-run` hold ran extraction and TWO offline `ocloc disasm` commands
in11 seconds, exit0, pinned image
`sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`,
network none/no DRI, target `bmg-g31`. No compilation, GPU, model payload,
source or archive changes. The initial selector asserted before any decoder:
an auxiliary Intel_Symbol_Table_Void_Program also has a text section. Exact
kernel-name selection fixed this; the failed attempt is preserved. Both private
names are short, so no section-name alias or ELF modification was needed.

RESULT -> The public image is byte-identical to S66's canonical image:
SHA256 `bad2ccded9f4136e6809825844815652eba954f059b807774c45d7916d92beb4`,
49384 bytes. Its index moved84 ->86 because new adapter images precede it.
The retained common-wrapper image was not regenerated.

| Main kernel artifact | Private original | Private SWAR |
| --- | ---: | ---: |
| Image index |9|8|
| Main text bytes |16128|12352|
| Main ASM lines |1152|931|
| Static byte nonzero predicate instructions |32|0|
| Static DP4A instructions |48|48|
| Compiled GRFs / SIMD |128 /32|128 /32|

Both ze_info records specify required subgroup32, eu_thread_count8 and
workgroup walk-order[0,1,2]. Required workgroup size, spill, scratch and SLM-size
fields are absent. Missing fields do not prove zero runtime usage. Source
local geometry is(1,4,32); walk order is not geometry or occupancy evidence.

The original's qs/sign reconstruction at ASM164/167/168 and candidate's
167/170/171 retain THREE d32 A64 messages. Source-bound addresses are
row+110*kbx+4*iqs+2/+6 for qs, row+110*kbx+2*iqs+74 for signs. The110-byte
block stride remains. Separate qh/scales, half scale, codebook and activation
loads must not be counted as extra reconstruction messages. The aligned-load
draft was not compiled; S66 already disproved its source load-count premise.

Original224..258/302..338/390..411 contains byte mov(ne) predicates, selects,
extract/repack and subtraction chains. Candidate225..250/285..305/340..362
instead shows packed shifts, bit-function operations, mul255, masks, add and
XOR. All32 static byte nonzero predicate tests disappear. Other legitimate
byte operations remain. Both kernels retain48 static DP4A instructions;
representative chains are original652..663 and candidate429..440.

The private original's1152-line ASM differs from S66 public ASM on only two
lines,1140/1141: null-destination move immediates after EOT at1138. Their actual
text hashes differ. This is not identical GPU code and does not replace the
public/private full-output-byte anchor.

VERDICT -> The intended SWAR substitution survives compilation: smaller main
text, removal of byte predicate emulation, unchanged reconstruction load form
and static DP4A count. These facts do not establish dynamic instruction counts,
latency, traffic, bandwidth, occupancy, numerical behavior or the cause of a
later timing delta. No additional GPU run follows from this audit.

The separate S65 run subsequently recorded PASS_PRIMITIVE_ONLY: leaf, analysis,
pre/post-health exits0 and container removal, one process/no retry. Its closure
sets speed_qualified=false and serving_qualified=false. S65 runtime/timing
interpretation is separate; no inference speed claim is made here.

Exact [audit helpers](../strata/flash-next/lowlevel_probe/iq3s_swar_built_isa_s68/README.md)
preserve extraction, decoder and summarizer commands. Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
S68 evidence is `iq3s-swar-built-isa-s68-prep/`: selection.json, three canonical
zebins, ze_info, complete original/candidate ASM, assembly-excerpts.json,
s68-report.json and failed-selector record. Manifest SHA256
`6f5f63136e6cdbbbe662af8fda74e04d9b29f38d6e0ca3a13f999d5c8ea97de7`
binds50 files/479829 bytes; report SHA256
`641b5bcf22836bc902a375119fb32a99fe6288d2b441e6ef413bedf14a88fdb5`.
The report pins exact source/header/build identities. Separate runtime evidence:
`iq3s-dense-oracle-s65-prep/run-clean/closure.json`; baseline ISA:
`iq3s-existing-isa-s66-prep/`. Frozen S68 consumes no runtime result.
