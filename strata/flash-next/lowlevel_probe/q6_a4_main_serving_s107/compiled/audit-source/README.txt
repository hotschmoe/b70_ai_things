S107 actual-main selected AOT audit PREPARATION, 2026-10-15
CONFIG -> FROZEN SOURCE ONLY; no artifact audit/build/decoder/GPU/model calls.
New executable/object/archive identities are mandatory root-provided arguments,
not anticipated hashes. Source applied identity must be ae47a632... . Existing
root build record must bind exact /out/strata-q6-a4-s107, selected native object,
and libstrata_kernels.a hashes. Root must first full-read actual build proof.

COMMAND -> Future explicit ROOT GO, under bin/cpu-run, e.g.:
python3 audit.py --exe ACTUAL_FROZEN_MAIN --exe-sha256 ACTUAL_FULL_SHA \
 --object ACTUAL_NATIVE_OBJECT --object-sha256 ACTUAL_FULL_SHA \
 --archive ACTUAL_KERNEL_ARCHIVE --archive-sha256 ACTUAL_FULL_SHA \
 --build-record ACTUAL_BUILD_DIR/build-sha256.txt \
 --build-proof ACTUAL_BUILD_DIR/root-build-audit.json \
 --out NEW_EMPTY_DISK_RESULT_DIR
No automatic execution, retries, compiler, decoder, DRI, network or full-model
reads. Root owns cpu-run serialization and follow-on selected decoder GO.

RESULT planned -> Audit once hashes actual new executable; bounded mmap ELF
class sets the already-verified exe hash rather than hashing it again. Object
is checked and its selected member in actual kernels archive must be byte exact;
archive full SHA comes from the authoritative build record, not another full
archive hash. Source/object/exe/archive stat continuity is checked at completion.
Root build proof is copied/hashed, not treated as semantically proved by merely
being readable: root must accept it separately. No common AOT image reparse.

Bind THREE distinct host Info/cache/guard/name pointers to actual object refs,
including STT_SECTION+addend, and final linked pointers. Preserve containing
launch<6> disassembly for exact guard and n_in/n_out argument-flow root readout.
Automated metadata binding is not a proof of every host instruction/argument.
Original/CENTER registered main names must match actual retained S102/S87 names.
New candidate class must resolve independently to native_mmvq_q6k_a4_c6.
Unique exact registered-name metadata seek selects actual images/target text;
no all-image extraction/disassembly. Full ze_info/resources/options and raw
section inventories, actual constants/relocations and all raw differences saved.
Absent scratch/spill fields remain unknown, not zero.

Reference mapping (each exact image/ASM pin in references.json):
- ORIGINAL C6/RPW1/A2: S85 private original, already actual-main bound at S102;
  computational prefix through actual decoded EOT end16384.
- CENTER6: actual MAIN S87 candidate.zebin, NOT S104 original and NOT A4;
  EOT end10688, all old goto/join targets below boundary.
- A4: actual S104 candidate.zebin, EOT end15344 (not guessed text length15552),
  all old goto/join targets below boundary.

For each arm gate exact computational prefix, normalized execution/payload
metadata, options, constant/bss sections and relocation sections. Full-image,
whole-text, embedded SPV, symbol/section table and post-EOT differences remain
raw and explicit. No opaque byte normalization or explanation of undocumented
post-EOT trailer values. Prefix equality plus unchanged relevant data may permit
root to reuse existing ISA; it is not dynamic/numerical/wholeprogram proof.
Other helper sections remain inventoried; no wholeprogram equivalence claim.

VERDICT -> Future PASS is pending root manual host argument/build/resource
readout, then serving correctness/health gates. Any computational/meta/data/reloc
drift returns exit5 HOLD with saved selection/facts, no automatic decoder or
build retry. Root may authorize ONE new candidate-only decode. A short selected
section alias is prepared on a COPY only; actual image/text unchanged and retained.
Source AST only performed now; existing helper AST/class behavior is retained.
