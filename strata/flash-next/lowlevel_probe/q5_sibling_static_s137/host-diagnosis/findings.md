S134 marker failure: bounded actual-host diagnosis + fresh CPU reproduction

CONFIG -> Actual root-bound S133 EXEfe57222e/nativeOBJ9dc9cccc, native source
e25708bf. S134 warm stopped before timed requests when OLD C2 marker appeared
under effective C2=0/C6=01010101; C6 marker absent. This failure remains intact.
No main source edit, full build, GPU replay, device decoder or retry here.

COMMAND -> GNU nm/objdump of ONLY actual native_mmvq_head and two parser lambda
functions, selected host symbols, ELF headers + small .rodata.str1.1 section.
Raw three function bodies retained, with critical windows and exact mangled
symbols. Root actual build receipt supplies large artifact hashes; no repeat
main/object/archive hash. Separate explicitly authorized standalone CPU fixture
under bin/cpu-run/e41/noDRI: ONE frozen command, two exact source variants,
-O3/-DNDEBUG/C++20/-fp-model=precise/-fsycl. Native runs0, total12.733256s.

RESULT -> Actual object has distinct C2/C6 parser word+guard pairs atbss360/368
and380/388; parsers72660/727c0. getenv arguments bind distinct strings at
rodata.str1.1+29eb and+2b3d (correct C2/C6 names). Both parse base16, return0
for exact0/null, validate full01010101. Logged values60/70 and guards68/78 are
also distinct; no simple shared-guard alias is established.

BUT C2 block68a44..68a82 and C6 block68b47..68b85 BOTH issue fwrite of
rodata.str1.1+2a42, length0x63. It is the exact OLD C2 marker. C6 marker text
is absent from that host string section. C6's own guard controls its logging
block, which returns to6844a; its word is loaded from380 (PC32addend37c).
C6 metadata init68b9c uses .rodata+11020, the actual distinct
CompileTimeKernelInfo<native_mmvq_q5k_runtime_ones_c6>; C2 uses10e08.
Thus OLD text does not by itself prove C2 dispatch/environment was ON.
This is compiled wrong marker text, not an observation of a GPU kernel launch.

Fresh tiny CPU reproduction: original two sibling static const bool logged
initializers print C2,C2 once per distinct guard. Changing ONLY the C6 static
identifier to logged_c6 prints the expected C2,C6, each once. Same native head
signature/namespace/branch names/marker literals; validators/device launches
replaced by CPU sink, parsers reduced. Both source variants compiled freshly
without project CMake/AOT cache/device kernels. Raw outputs/commands/compiler
and comparison.json live in q5-sibling-marker-host-repro-{prep,run}.
The project offline AOT cache is NOT required to reproduce this symptom.

VERDICT -> Strong isolated host compiler/source-name-context finding; tiny
unique-identifier spelling works in this reduction. Exact compiler pass/root
cause, behavior without -fsycl, full-main fix behavior and actual S134 GPU
selection remain unproved. No universal lambda-name collision theorem, GPU
correctness/speed claim, live fix or new run is authorized by this note.
A future minimal marker-identifier fix can use this tiny regression before
an otherwise-required main rebuild; ROOT/model review owns that decision.
Source search found only getenv accesses for the two flags in live native TU;
program/core setenv sites inspected target other keys, not these two. This
bounded source search is not a theorem against all external/runtime mutation.
