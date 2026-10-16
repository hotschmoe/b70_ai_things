# S137: sibling static marker reduction; S138 source proposal

CONFIG -> S134 stopped at its ON warm marker gate. The frozen S133 executable
was `fe57222eea163dd7396f2bdabb95d2c18326a729c4d7919d400a9614f42d47ee`;
native object was `9dc9cccc29eb293188e2d893705c9ee87475727b7fa43a9ccb66f58020533017`
and source was `e25708bf5fcd6c4be0e189db415ac6031ba51d443f51723a39c5d9304ceb82ed`.
Effective C2 word was0 and C6 word01010101. The old C2 marker appeared and the
C6 marker did not. S134 remains a failed marker/null comparison, with no ON
timed result and no claim about actual GPU dispatch.

COMMAND -> S137 inspected only the actual native head host function, its two
parser lambdas, selected symbols and selected string offsets. Narrow host
function bodies and critical instruction/relocation windows are retained. Large
artifact identities come from the existing root build receipt, not a new hash.
A separate single authorized CPU command compiled and ran two tiny source
variants in pinned e41 with no DRI or network, two CPUs, UID1000 and disk TMP.
Both used icpx2026.1.0, -O3 -DNDEBUG -std=c++20 -fp-model=precise -fsycl.
The exact command JSON, shell, source, compiler identity and raw output remain
in the payload. No project CMake, offline AOT cache, device kernels or GPU
queues were involved in this reduction.

RESULT -> The actual object has distinct C2/C6 word and parser guards, with
getenv references to the correct distinct names. Both parsers recognize exact0
as OFF and validate01010101. The logging guards are also distinct. However,
both logging blocks issue fwrite to the same old C2 string at
`.rodata.str1.1+0x2a42`, length0x63. The C6 literal is absent from that string
section. C6's block separately references its C6 kernel metadata. This proves
the compiled host marker text is wrong; it does not prove C2 dispatch was ON.

| Actual narrow host evidence | C2 | C6 |
| --- | --- | --- |
| Word / guard BSS offsets | 0x360 / 0x368 | 0x380 / 0x388 |
| Parser symbol address | 0x72660 | 0x727c0 |
| getenv string offset | 0x29eb | 0x2b3d |
| Logging value / guard | 0x60 / 0x68 | 0x70 / 0x78 |
| Logging block | 0x68a44..0x68a82 | 0x68b47..0x68b85 |
| Referenced kernel-info rodata | 0x10e08 | 0x11020 |
| Referenced marker string | old C2 at0x2a42 | old C2 at0x2a42 |

The fresh CPU reduction calls the C2/C6/C2/C6 branches. The original spelling
uses block-local `static const bool logged` in both sibling branches and prints
C2 then C2, once per distinct guard. The second variant changes only the C6
identifier to `logged_c6` and prints C2 then C6. Both native exits and the one
wrapper exit were0; total execution was12.733256427978631s. This reproduces the
marker symptom without the project's device-link/cache path. It does not
identify a compiler pass, demonstrate behavior without -fsycl or establish a
universal sibling-lambda collision rule.

VERDICT -> Keep S134's failed evidence intact. S137 supplies a bounded actual
host finding plus a small fresh CPU reproducer. S138 is a separate UNAPPLIED
source proposal: change only the C6 declaration and corresponding void-use
identifier. Its projected source SHA is
`dd08daf548eab9ab69375ed1702dfa85214bd1431e9957126d12eea51ed569d1`.
The inverse reconstructs the complete base TU; device helpers, FP/integer math,
parsers, marker literals, guards, launch arguments and C2/OFF paths are source
byte-exact. That is a source projection, not future compiled identity or GPU
proof. No main apply/build, decoder, GPU run or automatic retry occurred here.
Actual S138 source consultations were pending at this curation cutoff.

A future otherwise-required build can check the corrected C6 host reference
before any finite same-new-binary serving comparison. That build and its runtime comparison retain separate evidence; no repeated
CPU flag sweep is selected. The tiny reproduction avoids a
full-main rebuild merely to diagnose a host marker. Full small evidence and
absolute source/SHA provenance are in the accompanying indexed payload; old
executables, full rodata, device images and caches remain external.

Additive actual source-review disposition -> Root subsequently fully read
both own-session replies: claude-opus-5-5,45.326264946s, and
claude-fable-5-1,30.869940494s; both exit0 with packet SHA
`cb0d3adf85a32a53ce6a2520121d233ea6c8d614108b6d2d56023793769da4f7`.
Both accepted the S138 two-identifier source change and exact cached
three-action main build without configure. Root requires the narrow rebuilt
host check before GPU reload: C2 references C2 text, C6 references C6 text,
with distinct words/guards/Info. This addresses the demonstrated defect; it
is not optional broad image certification. Any later serving comparison uses
a fresh control and ON process on the same NEW binary and the existing strict
output/work/identity/normal-health gates. S134 remains failed, without an ON
timing or retroactive qualification.

Full replies and filtered identities are retained under actual-source-reviews/.
Compiler-stage explanations in the replies remain hypotheses. Both treat
S135 as API-ordering evidence only; it neither explains S131 nor approves MTP
integration. Root ranks current S136 head-free evidence before paired-HC or
projection design, preserving per-depth tail/h_out/PLE ordering. The original
frozen manifest is retained separately before this additive review update.
Payload: strata/flash-next/lowlevel_probe/q5_sibling_static_s137/.
