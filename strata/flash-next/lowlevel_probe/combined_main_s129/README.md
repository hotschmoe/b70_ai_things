S129 source/build - CPU build PASS, selected candidate body HOLD
========================================================================

CONFIG -> Q5 C2 explicit-head V2 + S127 owner-queue V2 + link-only AOT cache.
Both algorithm changes stay defaultOFF. Q5 can cover verifier T2 as well as
paired MTP head; not an MTP-only scope. CPU build passed; default selected
images passed, new candidate body differs. All runtime results remain pending.

COMMAND -> Root-applied reviewed four sources. Two CPU wrapper attempts,
12.212106s/exit1 then explicit11.877536s/exit1. Exact receipts and commands are
under setup/. Build wrapper source is preserved once under build-wrapper/;
initial and continuation wrapper bodies are equal. These copies are provenance
for root readback, not approval to execute in a new location. A third explicit
compile-only continuation completed217.870345s/exit0; its narrow root deltas
are preserved instead of another complete wrapper copy. No repeat configure.

RESULT -> Ownership preflight stopped attempt1 before configuration. Root's
three-file atomic generated-file repair preserved bytes/modes/mtimes. The
continuation's168 compile edges/cache/rules guard passed after configuration,
then the ten-action plan (including an icx ggml.c compile) violated the eight-
action expectation and stopped before scheduled compilation. Read-only Ninja
explain identified generated77B ggml-version.h newer than ggml.c.o; prior
header bytes are UNKNOWN. Root explicitly accepted only that additional CPU
compile/archive and repaired those two planned outputs byte-identically.
The same ten-action plan then passed and the CPU build/link exited0/tee0.
Source checks passed pre/post; new main SHAee8854f8 and all new artifacts are
recorded in build-sha256.txt. Candidate computational identity remains HOLD.

VERDICT -> CPU build/link PASS; keep both setup stops, no GPU/serving PASS. Source
projection/debug-type-name allowances are narrow and do not waive FP/compiled
identity. Model answers approve a conditioned CPU build only. No new test,
hash of large inputs, compiler or GPU workload ran during this curation.

Index
-----

- q5-source/: exact V2 patch/inverse, validation-only delta/inverse, source
  projection, source seams and pins. Full candidate TU remains external.
- metadata-projection/: TWO exact lexical pointer-type-name substitutions,
  matched source layout evidence, no general normalization.
- cache-source/: optional CMake suffix patch, source findings/projection/pins.
- source-application/: root's four-file applied full hashes/checks.
- reviews/: full actual source responses and filtered own-session metadata;
  usage/events/prompt duplicates omitted.
- setup/: two failed attempts, explicit continuation/ownership repair,
  configuration guard and ten-action dry plan. Initial stops are retained.
- build-wrapper/: exact code/pins/baseline stats, historical reused identities.
  No full Ninja/CMake dumps or main/object/archive binaries duplicated.
- compile-continuation/: bounded generated-header diagnosis, two additional
  ownership repairs (five named files total), explicit root ten-action
  disposition, script deltas, execution/plan/policy/source+artifact hashes,
  pre/post identities and cache counters. The identical plan is linked once.
  Cache-before UNKNOWN/after hits0 misses1608 is initial adoption, not main
  link reuse or a speed comparison. No timing-attribution work was repeated.
- selected-audit/: actual exit1/HOLD and exact comparisons. Q5 C2/default Q6
  C6 prefix+companion gates pass; candidate text6208 versus6464 differs at281,
  despite matching normalized metadata/ABI/options/constants/relocations.
  Optional main ISA/host/four-MTP checks are explicitly deferred, not passed,
  under root/user direction toward finite serving; actual results pending.
- build-timing/: existing independent bounded analysis/findings/pins. No new
  parse, timings, Ninja log duplicate or build was performed by this curator.

Already committed owner-queue evidence (link when this subtree is published):
../mtp_owner_queues_source_s127/README.md, with its v2/owner-queues.patch and
static-audit/findings.md. The S127 patch is not recopied here.

The top-level source-copy-index.json OUTSIDE this payload records exact ASCII
copy paths/hashes and explicit filtered/lossless projections. Finite serving outcomes are recorded separately. The root manifest pins
this curated checkpoint; the draft-source manifest preserves original inputs.
