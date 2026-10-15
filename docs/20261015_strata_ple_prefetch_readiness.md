# S47: PLE prefetch readiness and actual CPU mechanism (2026-10-15)

CONFIG -> The key-matching helper passes88 actual CPU reader cases. It
eliminates second reader tickets for prefetched keys requested in a different
order, with byte-identical decoded output. This is a validated CPU mechanism,
not a measured serving improvement. Neither patch is applied to the runtime.
The qualified152055a6/headpairON/depth2 serving candidate stays unchanged.

The source audit also finds a more immediate obstacle: batch sessions do
not currently receive the table/hash metadata required for CPU prefetch.
Changing key matching alone therefore cannot be credited with recovering
S32's0.829ms gather interval in the current C2 path.

COMMAND -> Read current source and prior S31/S32 traces, refresh upstream
and contribution status, and consult actual Claude Opus5.5/Fable5.1 at
three decisions: key matching, corrected reachability, and exact CPU fixture.
Both exact-fixture reviews return READY. Root checks their provenance items:
reconstruct the candidate from the patch, match the baseline source, verify
the SYCL host reader/DirectFile paths in production Ninja, inspect the public
request counter and confirm close drains outstanding prefetches.

One CPU-only container run compiles both host binaries and executes both
arms in22s. It has no GPU device mount, network or GPU runtime linkage.
The earlier host launch exits127 before compilation because g++ is absent;
that setup failure is retained. Root also corrects an absolute git-apply
scratch path and the host-versus-container Ninja path assumption before
recording the successful provenance check. These change no production files.

Actual raw root P:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/ple-prefetch-permutation-s47/`.
Run recipe is `P/v2/build-and-run-container.sh`, executed under cpu-run in
`strata-ab-build:ocloc`, image
`sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`.
The compiler is g++13.3.0, C++17/O2/pthreads. Production archives are unused;
the host source files are compiled directly. Input hashes are rechecked
before and after the run. The compiled SYCL-tree reader and DirectFile are
host code; no SYCL compiler or device library is used for this test.

RESULT -> Existing `PleTable::gather_batch` tests prefetched row-vector
keys positionally. Pair drafting emits depth-major slot0/slot1 order;
verification groups all rows of one slot, often slot1 before slot0. Available
keys can therefore miss the positional fast path. The candidate searches
at most16 prefetched slots for each requested key and decodes from the
matched immutable slot into the original output position. Duplicate requests
may reuse a slot. It keeps the16-token cap and existing fallback/byte counts.
All prefetched tickets are drained before slot reuse, including after a
failure; the first error is retained. Runtime error injection is untested.

The fixture opens a tiny synthetic IQ4_NL GGUF through the actual PleTable,
PleReader and DirectFile. Existing GGUF writer/format helpers are reused;
half scales are made finite and the six permutation tokens must have distinct
outputs. A Mmap-mode oracle closes before Direct-mode runs. Every arm covers
11 cases at cache0/256 rows and io_thread0/1, giving44 cases per arm.

| Case | Baseline additional requested rows | Candidate additional requested rows |
| --- | ---: | ---: |
| Ordered | 0 | 0 |
| Permuted six-token group | 96 | 0 |
| All-present subset | 32 | 0 |
| Three duplicate requests | 48 | 0 |
| Duplicate prefetch suppression | 0 | 0 |
| Missing key | 32 | 32 |
| Extra prefetched slots | 16 | 0 |
| Zero tokens | 0 | 0 |
| Gather larger than16 | 272 | 272 |
| Prefetch overflow then latest key | 0 | 0 |
| Pending-token guard | 0 | 0 |

These exact counts hold in all four cache/thread configurations. The public
`io_report` first number is ReaderStats.requests, counting rows asked for,
including cache hits/deduplication. A zero gather delta proves no second
ticket issues requested rows; it is not a physical-I/O bandwidth result.
All successful outputs match the mapped oracle byte-for-byte. Buffer guards,
logical bytes_read, finite references, duplicate-prefetch counts and untouched
pending-error output also pass. Wall times are diagnostic observations only.
No out-of-range or I/O-failure runtime coverage is claimed.

| Exact artifact | SHA256 |
| --- | --- |
| Generic helper patch | 217550f5994f9159c9a443a81bc71429f481fd823493b44f189f11fc241199b5 |
| Executed fixture | 8415719e1b808a27fe9af5ca572b5d502f04c56b2255215ea424b581f058fb99 |
| Baseline executable | c1bf9a3b32701eda1f9e9588122ddcbe20a299d3d488f970d9b05203b399de87 |
| Candidate executable | 5ed248b74ce783d9940276eb90ad8b7ff3af3d4dc10c4107f9769d0dd965bb46 |

The separate source audit follows construction, every table/ready consumer
and pointer alias, draft/verify slot identity, reset/destruction, and host
worker boundaries. It finds that `session_init` carves history without wiring
PleRun; checkpoint restore updates history/ple_prev only. Batch drafters point
to those fresh slots, so the current ready() prefetch predicate is false.
This is source evidence, not newly observed live pointer/counter evidence.

A source-eligible proposal borrows only table and hash constants from the
validated main session into primary batch slots. Hist/token/prev/emb/scratch/
weight pointers stay unbound, keeping slot PleRun.ready() false. CPU-prefetch
predicates can check an open table independently. A full PleRun copy would
be inappropriate because it could share mutable model state. The current
source-only proposal has not been built or tested through serving.

The audit supports the required consumer guards and table lifetime. It also
qualifies serialization: prefill uses std::async for a table gather, but its
local future joins before the current mixed no-layer-split batch path drafts
or verifies. This is not a global single-thread assertion. C2/depth2 needs at
most6 prefetched vectors and C4/depth2 at most12, within the16-slot cap;
future deeper/admission schedules must not assume the same bound.

VERDICT -> Retain the generic helper as CPU-tested source, and advance a
separate bounded live prefetch candidate only after exact final wiring review.
Use actual startup/path counters and matched deterministic C2 work to prove
prefetch activation and reuse; enabling unmatched prefetches can add work.
Preserve token/coherence/tool/state/residency/teardown checks before any
serving claim. Do not infer saved time from S32, request counts or the tiny
fixture. No new GPU window or full200k campaign was spent on this stage.

The development loop reuses the existing real reader fixture helpers and
public counters, excludes the broad inherited selftest and its keep-alive
sleeps, and uses one small CPU lease for both arms. Source audits prevent
spending a serve window on a helper that is currently unreachable. The next
HC candidate is independently prepared for a one-predicate numerical/warm
probe; it has no measured result yet.

Raw evidence: `cpu-report.json`, `root-cpu-decision.json`,
`v2/inputs-sha256.json`, `cpu-container-build-run.{log,exit}`,
`v2/run-container-{baseline,candidate}/results.tsv`, actual-model event streams
and `metadata-wiring-audit.md`. Exact inspection copies in this repository
are `patches/ple-prefetch-key-match.patch` and
`lowlevel_probe/ple_prefetch_cpu.cpp`; the latter requires its pinned external
production test helper/source tree. Serving-source changes remain unapplied.

Frozen raw manifest `ple-prefetch-permutation-s47-frozen-manifest.json`:
62files,1,642,098bytes, SHA256
`97a0dfbd20afa3cad7769f764c0bf36a936a5a4920af88471c0751da577d622b`.
