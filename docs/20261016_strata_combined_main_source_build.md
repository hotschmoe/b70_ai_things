# Combined Q5/owner-queue source and build setup: S129 (2026-10-16)

This checkpoint records applied source, two stopped setup attempts and one completed
CPU-only compile continuation. The selected audit stopped on a new candidate
body difference; optional further CPU audits are deferred by root/user
direction. Serving results are PENDING. Exact source/build evidence is retained under
strata/flash-next/lowlevel_probe/combined_main_s129/.

CONFIG -> One required incremental main build combines three independent
changes: default-OFF Q5 C2 head runtime ones, default-OFF MTP owner queues,
and a link-only offline AOT cache option. Later comparisons must select one
algorithm flag at a time on the same frozen binary. No production default
or speed follows from source review or S128 private timing.

The Q5 V2 seam is explicit native_mmvq_head: type13/N2560/M248320/C2,
wide-K ON/RPW1, after the unchanged RPW2 override. Word unset or literal0
retains the original route; host-parsed01010101 enables the unique new class.
The original wide/Q4/Q5 code and generic APIs remain source-exact. V2 restores
five original shape/pointer/stream validations before the new marker/launch.
The private S128 apply/kernel are projected by function renames only, not a
waiver for future compiler or floating-point differences.

This is C2 explicit-head scope, NOT MTP-only: a verifier window with T2 also
reaches native_mmvq_head and may select it. Actual incidence remains unknown
until observed; C1 and C6 do not select this new kernel. Markers establish
HOST selection, not execution counts, all eligible shapes or GPU duration.

The already-curated S127 V2 owner-queue patch is referenced, not duplicated.
It records head-free work on each owner's queue, joins both before the
unchanged paired head/tail, and preserves kernel arguments. ON readiness
waits both queues with throwing waits; errors drain writers before allocating
the error string, while OFF std::exception behavior rethrows. Queue-keyed HC
maps may lazily allocate850KiB per unseen queue with process-lifetime/raw-key
caveats. Source audit found no unkeyed mutable device-scratch blocker in its
bounded scope; it proved neither actual alias safety nor hardware overlap.

The empty-default CMake cache option adds only allow_caching/cache_dir to the
existing single backend argument. Frontend flags, precision, SG32/per-kernel/
BMG-G31 and jobs4 remain fixed. A stable UID1000-owned directory, matching
NEO_CACHE_DIR/CLI path and persistent env are required before ocloc;
SYCL_CACHE_PERSISTENT remains0. Prior S119 tiny exact-input hits do not predict
whole-main link savings. No standalone warm-up or extra main link is selected.

COMMAND -> Root applied the reviewed four-file source set at03:35:02UTC.
Exact full before/after pins and patch checks are preserved in apply.json:
CMakeLists66726a2a, mtp.cpp452a051e, mtp.hppe6314c05,
native_mmvq88a4e5ad. Existing unrelated dirty changes were preserved.

Actual responding claude-opus-5-5 (43.736249s) and claude-fable-5-1
(63.623681s), both exit0 and own-session resumes, independently returned
CPU_BUILD_GO_WITH_CONDITIONS. Root read both full texts. Their reasons differ:

- Opus accepted the exact source bodies/dispatch and owner error/readiness
  fixes, highlighted verifier T2 eligibility, and required actual payload,
  selected default-image and lazy-scratch/capture checks before runtime.
- Fable accepted the same source changes, noted repeated invalid static-init
  calls still throw rather than silently falling back, and required the
  post-configure plan, source/host binding and strict compiled gates.

Both proposed independent same-binary flag comparisons after separate root
GO; neither approved GPU work, inferred serving gain or waived control
identity. Their expectation of eight actions was a scope guard, not a result.
Observed extra actions correctly stopped the continuation for diagnosis.

RESULT -> Closed setup and CPU-build evidence:

| Step | Actual result |
| --- | --- |
| First wrapper,12.212106s | Exit1 before configuration/compilation: ownership/writability preflight stopped at CMakeCache.txt |
| Explicit three-file repair | Root atomically replaced CMakeCache.txt/build.ninja/rules.ninja with UID1000-owned byte-identical files; modes/mtimes retained; no privileged broad chown |
| Explicit continuation,11.877536s | Configure completed;168 existing compile edges, cache values and rules matched, with only the new cache PATH |
| Continuation dry plan | Ten scheduled actions rather than expected eight; guard encountered an icx ggml.c compile and stopped before scheduled compilation |
| Bounded dirty-dependency diagnosis,9.889389s | Read-only Ninja explain identified generated ggml-version.h newer than ggml.c.o |
| Explicit compile-only continuation,217.870345s | Exit0/tee0; exact ten-action plan passed; Ninja205.843s; no repeat configure |

The actual extra input was the generated77B ggml-version.h. Ninja reported
its mtime newer than ggml.c.o; ggml.c includes it and exposes its macros via
version/commit string functions. Current values are0.24.0/3cf03257f; previous
header bytes were not retained, so an old/new literal change is UNKNOWN.
After reading the diagnosis, root explicitly accepted only that extra icx
CPU compile and ggml-base archive. The resulting guard required five exact
compiles/four archives/one link and preserved all existing FP/SDK/device flags.
This was not an automatic retry or a blanket whitelist broadening. The ten-
action plan bytes equal those from the earlier stopped attempt, kept once.

Two additional planned root-owned CPU outputs, ggml.c.o and libggml-base.a,
received byte-identical atomic UID1000 replacements before compilation.
Together with the earlier three generated files, this repairs five named
files only; modes/mtimes/bytes were preserved, with no broad chown. Root reused
the successful configured-cache/168-edge/rules proof and prior backups rather
than reconfiguring or copying another main backup. Exact root script deltas,
decision and source/pre-post checks accompany the unchanged initial wrapper.

The new frozen main SHA is
ee8854f8313d9610f7e79197fc4b9451e79b2a1a447a062bbf50c3a50a211702.
Actual build-sha256.txt records new objects/archives, generated files and the
same four reviewed source hashes. Source checks passed before and after the
build; unchanged retained artifacts have stat-continuity evidence. This is
a successful compilation/link result, not a selected-kernel identity or
runtime proof. No additional model payload hash was required.

Persistent offline-cache adoption reported after counters hits0/misses1608;
before counters were UNKNOWN/fresh. This is an initial-adoption observation,
not measured reuse of this main's inputs or a whole-build speed comparison.
Statistics collection adds overhead. Wrapper217.870345s and Ninja205.843s
are distinct elapsed scopes; the two earlier setup stops remain separate.
No extra cache warm-up/link was performed.
The independent build-loop analysis records native-TU58.612s and final-main
link146.710s. The link includes driver/AOT/backend/host work; these subphases
are not separated. Concurrent compile durations must not be summed as wall.

Metadata projection is separately constrained source evidence. Q5KBlock and
Q81Block declarations match modulo whitespace and retain known size/offset
assertions. Existing selected metadata differ in TWO exact complete pointer
debug type_name strings: private s126 namespace versus native anonymous
namespace, one corresponding occurrence each. Only those strings and the
explicit registered class rename may be projected for the new candidate.
No broad namespace stripping/args_info deletion is allowed. Actual payload,
resources, options, constants, relocations and computation remain independent
strict gates; this source layout proof supplies no FP equivalence waiver.

One actual selected audit,2.478586s/exit1, then preserved this result:

| Selected image | Actual comparison |
| --- | --- |
| Default Q5 C2 |8128B prefix, metadata/options/constants/relocations PASS; raw whole image/text not exact, first text difference8140 |
| Default Q6 C6 |16384B prefix and same strict companion gates PASS; raw whole image/text not exact, first text difference16396 |
| New main Q5 candidate |Computation comparison HOLD: text6208B versus6464B private reference, first difference281; normalized metadata, ABI, options, constants and relocations PASS |

The new main candidate cannot reuse the S128 ISA as a byte-identical body.
Its actual EOT/arithmetic/scheduling are not decoded here. Root preserved the
HOLD and, following user direction to prioritize the finite GPU comparison,
deferred optional main ISA/independent-host/four-MTP certification. This is an
explicit scope decision based on reviewed source, metadata/ABI/default-control
evidence and immediate finite serving gates; it neither fixes the body
difference nor turns the audit into PASS. No new Claude call followed it.

VERDICT -> Source applied; setup/configuration inspected; CPU build/link PASS.
Default selected checks passed; candidate body HOLD remains, optional CPU
certification is deferred, and runtime results are PENDING. Keep both features defaultOFF and
preserve historical S126 HOLD/S128 private-context limits. No GPU, serving,
cache gain, stability or target-throughput claim is recorded here. Root will
append finite serving outcomes separately; this checkpoint contains no runtime qualification.
