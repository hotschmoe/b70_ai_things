S102 source-only existing-main Q6 C6 binding and A4 feasibility audit

CONFIG -> No main/source changes, build, GPU, decoder, trace or model read.
The selected main is frozen S100 7ee675fff8c342772e4f6eae402d40c655a29cf7dcc119d504cdd8d1386b7bbc.
Its unchanged native_mmvq source c1cdeabe... and object b9f931cb... are
checked against actual S100 build inputs. Source preparation was requested
after actual Opus5.5 104.737s and Fable5.1 111.676s design consultations.
The Opus A4 hypothesis is not measured or accepted for a build yet.

COMMAND -> ROOT ONLY, after reading source and after quiet serving windows:
  bin/cpu-run python3 R/q6-a4-s102-audit-prep/audit.py
Expand R to the existing results root. The script creates NEW
R/q6-a4-s102-audit and refuses an existing directory. Inspect the same
handle after a timeout; do not restart. Preparation does not execute it.

RESULT -> audit.py is source only. It hashes one frozen main under CPU
lease, reuses ONLY the S92 ELF class with S97 bounded-string/mmap fixes,
resolves actual C6/A2 host Info/cache/guard via named or section+addend
relocations, writes exactly one host-function disassembly, cross-checks
registered name against exact S95 CSV name, and finds the unique selected
AOT image via its exact registered name in .ze_info. No all-image walk.
It inventories all selected sections/resources/options, preserves original
image bytes and creates a section-name-only alias for optional decoding.

It compares the main's computational target prefix against S85 ORIGINAL
private C6 image, whose existing ASM/hash is pinned. S85 ends thread at
16384B; all branch targets stay before that point. Raw full-image/text
differences and all differing text offsets remain in facts.json. Exact
computational prefix, normalized pre-misc execution/payload metadata,
options, constant data/BSS and raw relocation-section equality are required
for existing-target-ASM reuse. Private/public argument type names and SPV
remain separate raw inventory differences. Extra helper equivalence is not
claimed; S85 target has no call/calla sites. Missing/new significant sections
or any meaningful difference require root interpretation, not a blind retry.

VERDICT -> A passing identity comparison permits ROOT to reuse existing
target instructions without another decoder or primitive run. It does not
prove that the A4 proposal wins or that private binding covers main by
assumption: this script establishes the main host/name/image itself.
If comparison returns HOLD, read facts/raw differences before choosing any
new decoder. decode-selected.sh is SOURCE ONLY, never called by audit.py.
Root can later authorize ONE existing-image decode, passing the actual
alias SHA from facts, under bin/cpu-run. It uses pinned e41 image, UID1000,
no DRI/network, four CPUs/OMP/MKL, disk TMP and one selected image only.
Do not decode again if the existing S85 computational body is exact.

Specific instruction question, using old S85 ORIGINAL as evidence only:
- Source load16_a2 reconstructs two aligned int4 loads with a k=(a>>2)&3
  compare/select chain plus optional half-word funnel. Weight ql/qh decode
  is ALREADY hoisted outside NCOLS; proposing that hoist repeats source.
- Existing S85 ASM L1648..L13208 is the K loop (backedge line955).
  Lines169/180 show d32x4 weight loads; lines184..218 include explicit
  k compares, selects and half-word shifts. A second corresponding qh
  load/reassembly branch follows. This is a concrete compiled ALU surface
  in the private original, conditional on actual-main body identity.
- S85 original report records48 static DP4A sites,128GRF/SIMD32 and no
  call/calla. Missing spill/scratch fields are not measured zero.
- FP ledger is NOT inferred from C++ alone: for example old ASM lines944..
  954 contain half->float/int->float, MUL, MAD and ADD. Preserve the exact
  actual compiled contraction/order if a candidate is later built.
- A4 candidate would use dword-aligned int scalar loads (four plus a fifth),
  never a misaligned sycl::int4 dereference. Any compiler merging/splitting,
  removed selectors, retained funnel, register pressure and FP contraction
  need a separate selected candidate image check. No source syntax proves
  a wider send, lower latency or no extra spill.
- Existing S85 source/ASM recenter optimization is separate (S87). Keep
  CENTER0. Do not repeat packed Q6, GRF256, SWAR, SG16/DPAS or PF sweeps.

Root next decision: are the actual main's A2 reconstruction instructions
the same concrete compare/select sequence, rather than already optimized
loads? If yes, prepare one default-off A4 private leaf for the two existing
C6 shapes only, retaining signed-byte/int32 and FP/reduction ledgers. This
audit grants no automatic candidate build or GPU authorization.
