# S141: existing QSA three-step commit oracle

CONFIG -> Compare three existing native_qsa_indexer_append calls with one
existing native_qsa_indexer_append_steps call, stride1/n_steps3, position-base0.
No main source change or new device arithmetic. Twenty-four disjoint states
represent12 QSA layers by two owners, separately allocated per arm. Analytic
text RoPE only, geometry128/4/64, capacity32, epsilon1e-6 and frequency base1e7.
The same existing public indexer and rope object inputs were pinned before/after.
This is a private synthetic-state oracle, not a model/weights/cache qualification.

COMMAND -> One CPU-leased e41/noDRI/network-none/UID1000 four-CPU build compiled
only the host fixture, reused two unchanged public objects and prelinked their
device images. Build and tee exited0, wrapper rounded21s. Exact component
receipts: host fixture9.987s, existing-object AOT1.428s, host link0.199s. No main
or kernel-source compile, image/ISA certification or repeated build occurred.
The frozen executable SHA is
`635f49f2b9bcc73c96fc30245b760f932b5c982b7e49b1142eebe6cdadf42065`.

V1 was unused. V2 added finite-float validation of complete returned state and
held the SAME leases for uncertain create/inspect/wait/removal states instead
of escaping to blind health. Exact delta/inverse, V1 manifest and complete V2
source/scripts are retained. These were preexecution corrections, not a failed
GPU attempt. Root read the original full source packet and narrow V2 delta.

One card0 CPU-quiet window then ran strict cases before conditional timings,
with fresh pinned pre/post card0 health, exact device/PCI pin, non-auto-removed
owned container and normal wait/removal. No retry, force stop, reset or extra
probe. Runtime prospective ldd is scoped as such, not actual mapped-library
identity. No model file or main executable was read for this oracle.

RESULT -> Native, analyzer, create/wait/rm and both health exits were individually
recorded0. Root's whole leased window was rounded41s; the attached container
start interval receipt was5.573246s, a different scope. Identity matched, final
OOMKilledfalse/noError, full CID and NAME absent, closure PASS_PRIVATE_STRICT_AND_NORMAL_HEALTH.

All12 base cases times initial/changed/restored A/B/A passed:36 strict phases,
eager and captured, all96 complete state fields per phase, finite floats,
guards and read-only operands. Starts0/2/3/7 and complementary accepted prefixes
exercise position0, block crossings and rejected -1 suffixes. Each state first
builds a chronological prefix, saves its device tail, applies all speculative
mutations, then commits from identical speculative state after restoring the
saved tail. Full tail/dead/pooled/block_pos comparisons retain stale speculative
pooled bytes; no invented clearing or failed-state reset masks a mismatch.

The completed analyzer indexed36 full159840B snapshots,5754240B total, left
external. An additional cheap saved-SHA check verified every case has A0=A2
and A0!=B1. No raw snapshot was re-read or rehashed for curation. This freshness
check supplements the original complete-byte and immutable-operand gates.

| Completed HOST captured-chain timing | Control three appends | Grouped steps |
| --- | ---: | ---: |
| Median of4 samples, us | 151.7345 | 87.9080 |
| Source tail D2D restores per sample | 24 | 24 |
| Source append calls per sample | 72 | 24 |

Change of medians is-42.064593089%. All eight ABBA/BAAB samples remain retained:
control151.579,155.547,150.888,151.890us; grouped88.048,87.768,87.527,88.057us,
in their original interleaved order in comparison.json and leaf.stdout.
Timing includes restores, graph submission and completed whole-chain work.
It is not isolated kernel latency. There was NO actual DOT export or measured
submission-count proof; the72/24 call figures are source counts. One process
with nested samples does not establish population speed or serving savings.

VERDICT -> Private finite complete-state contract passed and the tested grouped
restore+append chain was descriptively lower in this window. A later default-OFF
main selector and matched serving require their own source/build/runtime gates;
no main change, promotion or serving forecast follows here. Preserve per-owner
grouping, position-base0, tail snapshots, rejected-prefix semantics, PLE/history
completion and unrelated GDN/conv/KV behavior. This tests a small staged-state
contract, not full200k/four-conversation RAM eviction/recovery. Source and result
indexes preserve exact provenance while leaving binaries and larger buffers
external. Curation performed no compiler, GPU or model invocation.

[Complete source, build and runtime receipts](../strata/flash-next/lowlevel_probe/qsa_commit_group_s141/README.md).
