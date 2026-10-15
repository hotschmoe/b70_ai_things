# S85v2 corrected build and private ISA audit (2026-10-15)

CONFIG -> Only include-repaired private Q6 C6/RPW1/A2 original/candidate.
Original S85v1 build failure and frozen packet/header remain unchanged.
COMMAND -> Versioned v2 adds ONLY strata/sycl_math.hpp defining strata::dp4a;
CPU-only build.sh, imagee41/noDRI/networknone/CPU4/OMP4/MKL4/diskTMP/cpu-run.
Then cpu-run decode.sh indexes ONLY the137120B new adapter bundle, selects
its two exact private target.text sections, binds BOTH wholeimages bytewise
into final executable using ELF bundle offsets and actual link-map leaf-first
contribution. Decode only those two images in pinned CPU-only ocloc container.
RESULT -> Build exit0/CPUlease24s; audit exit0/CPUlease11s. All source/common8f/
archive hashes passed pre/post. Final binarya0a1756b2670422caab13da80cc7e45a7beabd1fc694d684b0e34789b80112e8.
VERDICT -> Actual compiled mechanism gate PASS; root owns GPU GO. No GPU run,
no timing, dynamic cost, bandwidth, occupancy, serving or speed claim.

Both private targets actualSIMD32/GRF128. Spill/scratch fields absent for both,
not measuredzero. Original has32 word-add-minus32,32 overflowmask-bfn40 and
32 compare-ne-zero sites; candidate all0 and exactly8 packed add0x60606060 +
8 XOR0x80808080 sites. Both have48 staticDP4A and no call/calla sites.
Original targettext16576B SHA4a3dd695...; candidate10880B SHA2c20513b... .
Full exact hashes/numbered sites in report.json and excerpts.json, raw target
asm/resources/extra sections in original-disasm and candidate-disasm.
Original kernel_ZTSN3s859PrivateC6ILb0EEE; candidate_ZTSN3s859PrivateC6ILb1EEE.
Each is unique among10newadapterimages; onlyduplicateIntelSymbolTable2.
Targetwholeimagesmatch final executable bytes, so this is not merely unlinked
compiler output. No common images reindexed/reparsed or wholearchive rehash.

Original private control preserves SOURCE FP ledger/load/dot/tree; machine
rescheduling and other generated code differences are expected. Full strict
public/S44/private direct/graphs/changed/restore/read-only/guard runtime gates
remain pending. FP32inputbufferis a readonly witness; kernels read Q8+weights.
Ordinary final host link had1ld/no compiler/AOT/crtfastmath. Actual map resolves
native Q6 host API from frozen native_mmvq archive and links unchanged common8f.
Public image not separately decoded in this audit; actual public DOT binding
and full runtime anchors are still required. Extra symbol-helper kernels and
DPCT helper images inventoried, no whole-program ODR equivalence claim.

Setup cost: v1 failed16s due missing defining include, caught before GPU.
V2 include-removal reconstructs v1 header exactly; source leaf unchanged.
No other math/protocol or main-source change. Source-prep manifest is external
q6-center-s85-prep-v2-manifest.json; build artifacts live only in v2/build.
