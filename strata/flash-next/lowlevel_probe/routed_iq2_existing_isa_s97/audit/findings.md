# S97 routed IQ2_XS existing-image audit, 2026-10-15

CONFIG -> Frozen S94 d94e96ca, actual S95-observed routed GU QT17/LN8/KM1.
No candidate kernel, main edit, compile, GPU, model payload or rawtrace read.
Source/registered-name/image/object bindings: selection.json and report.json.

COMMAND -> Existing S66 ELF/alias pattern and S92/S93 bounded mmap parser;
ONE selected43,272B image, not an all-image walk. CPU-run + pinned e41,
no /dev/dri/network,4CPU, diskTMP. First actual ocloc handle73450 exited0;
CPU lease11s. Exact frozen executable SHA d94e96ca checked, inode/size/mtime
unchanged. Relevant current IQ object/archive member byte match. Decoder
input SHA unchanged; alias affects ONLY section-name-table bytes on COPY,
original selected7,488B text unchanged. Source acaa7da6 is pinned full inJSON.

Host metadata is a namespace-global C++ class+scalars17/8/1, while its actual
registered device string is function-scoped launch_gu_port<17,8,true>.
Named-only relocation matching was insufficient: the live launch_gu<17>
object uses STT_SECTION+addend. Host disassembly0x158ee loads .rodata+0x12538
(the selected CompileTimeKernelInfo),0x158f3 calls getDeviceKernelInfo,
then stores to selected cache .bss+0x838, guard .bss+0x840. Exact linked name
pointer, selected unique full image and S95 runtime demangled name agree.
The standard c++filt _ZTS prefix is explicitly normalized, not discarded
kernel differences. All binding details/addresses are retained inJSON.

RESULT -> Reuse gap SURVIVES compilation. Actual target ASM is
`disasm/.text.s97_qt17.asm`; annotated line excerpts in assembly-excerpts.txt.
Outer group loop L1616..7200 reads entry interval. Entry loop L1824..7152
increments r2.12 by1 and jumps L1824. Nested K loop L2304..6632 increments
k r38 by8 and jumps L2304; each entry resets its k/FP accumulator.
Inside that K loop: four16-bit weight-code sends(lines202/205/209/214),
four64-bit IQ2_XS codebook gathers(221/276/277/390), four cbit+SWAR sign
expansions, scales/half conversions and ALL8 signed SIMD32 DP4A instructions.
No decoded-row hoist across entries occurred. DP4A hardware folding DID
occur; there is no missing-dot-intrinsic reason to rewrite this path.

Metadata: GRF128/SIMD32/eu_thread_count8. SLM/spill/scratch fields ABSENT,
not measured zero. Source/S95 launch local256. Only extra kernel is
Intel_Symbol_Table_Void_Program,320B; selected target7,488B. Whole selected
image43,272B includes4KiB codebook globals. No helper body is treated as
another routed kernel; no unrelated image was disassembled.

VERDICT -> GO source basis for ONE future QT17 natural E4/E2/tail weight
prep reuse candidate, preserving signed dot/two integer chains/truncation,
FP conversion/product/k/XOR order and strict real-weight public/original/
candidate outputs/routing. This audit does not authorize that build/GPU or
predict savings. Repeated static loop instructions do not measure actual
routing reuse, cache misses, transactions, bandwidth, occupancy or cause
of the observed service. S66 dense IQ3_S never substituted for this proof.

Dev loop: six subsecond/1s parser preflights stopped before ANY decoder,
preserved separately. Reused helper assumptions (named relocs, single total
kernel, .shstrtab spelling, demangler prefix) were not universal. Actual
metadata enumeration/host callsite resolved them; only ONE ocloc ran. Future
leaf binding should retain section+addend support and enumerate target
sections before applying inherited assertions. No GPU experiment repeated.
