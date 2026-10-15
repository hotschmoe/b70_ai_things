# S92 corrected-binding attention value pipeline (2026-10-15)

CONFIG -> A private INT8 attention probe revisits the retained S13 value
pipeline with separate host metadata and all 11 private chunk/merge tags.
The current public source/object and completed public AOT wrapper are reused.
Main serving source, archives and frozen S89 executable are unchanged.
This is a primitive result, with no serving throughput or promotion claim.

The candidate reuses retired query SLM for cell-major probabilities after
all query reads finish. Existing workgroup barriers protect publication.
It preloads four values, then folds cells in the original increasing-cell
FMA order. Score/K loads, exp/max/sum, chunk statistics and merge are unchanged.
Masked groups retain a scalar fallback. It changes value-load scheduling and
probability layout, not the arithmetic policy or query-batch size.

COMMAND -> One CPU-only private build took 30s; a private-only selected ISA
decode took 10s. The exact public object/AOT and public S91 ISA were reused.
The linked binary is
`dfee38a6386ba74b1342969e6ae1e9795b5670c309ca82353d9bee47273a4c72`.
Actual object relocations, final kernel-info/cache/guard addresses and
registered-name pointers are separate for public and private host paths.
All 11 materialized attention tags are private; both selected device images
and complete bundles occur once. This resolves the demonstrated S13/S91
binding defect for this new binary. Distinct device names alone were insufficient.

Both selected M1/NC1 images report SIMD32, GRF128, SLM15872 and barrier_count1.
Private/scratch/spill fields are absent; that does not prove runtime zero.
The private instruction text is 21120B versus public 19328B. Static resources
are not occupancy, physical traffic or latency measurements.

One physical-card0 lease with quiet CPU took 43s, including a 15s leaf.
The fixture requested card0 through device pinning and recorded the SYCL
B70 name; it did not independently read PCI identity. All 36 cases passed
strict full scratch/output bytes, finite values, guards, readonly inputs,
initial/change/repeat/restore checks. The existing FP64 baseline sanity
limit 2e-5 is separate from strict public/private equality. All 1440 raw
host-completed timing records passed the parser before result acceptance.
Normal full CID/name removal, identities, driver/tee exits and pre/post-card0
health passed. No model, graph/DOT or full-cache campaign was needed.

RESULT -> Primary statistic is the median of three seed-level medians of
20 alternating paired rounds. The three seeds are nested in ONE process;
60 rounds are not independent process replications. Times include chunk plus
merge, enqueue and completion wait, with repeated synthetic pools.

| Queries / width | Capacity / page / mask | Paired change | Round directions |
| --- | --- | ---: | --- |
| 32 / 2051 | 2112 / 4 / none | -8.469% | 60/60 faster |
| 32 / 2051 | 2112 / 512 / none | -9.182% | 60/60 faster |
| 32 / 1999 | 2112 / 4 / masked | +2.392% | 60/60 slower |
| 1 / 2051 | 2112 / 4 / none | -13.803% | 60/60 faster |

All other shapes and raw rounds remain in comparison.json and leaf.log;
no best-only acceptance or tolerance relaxation was used. This measures no
physical memory traffic, cache residency, isolated GPU service, causal
mechanism, population significance or clean serving gain.

VERDICT -> Retain the PRIVATE full32 prefill lead. Preserve the masked
regression and original short/decode paths; the short primitive gain does
not justify broad integration. Capacity2112 differs from production2051,
including the ID-row pitch. The subsequent S93 numeric-only gate passes
32queries/width2051/cap2051/page4 on seeds7/31/109, closing that pitch gap
without repeating all36 cases or collecting timing. Full output/scratch,
guards/read-only, own repeat/changed/restored bytes, before/after identity,
cleanup and card0 health pass. The existing nq1 FP64 sanity is inactive for
this nq32 case; no new independent ideal-attention accuracy claim follows.
A default-OFF explicit prefill-only entry point is prepared separately.
Its proposed domain is staged INT8, full32 queries and width2051, with the
caller ensuring nonnegative staged identity-table rows. Query count alone
does not exclude decode. The CAP result reuses the S92 compiled candidate;
it does not validate newly compiled main code or qualify serving. Main
application/build/binding and matched serving remain separate gates; this
report contains no S93 main-serving result.

The actual S92 source consultations remain retained with S91's combined
post/source review. Completed S92 post/S93 source reviews responded as
`claude-opus-5-5` (40.775s, session
`b2074faf-37fb-4499-8dfe-5331418a878f`) and `claude-fable-5-1` (81.541s,
session `8b24cc8b-510b-4be8-b0d3-1fe758a1a074`), both exit0 and
ACCEPT_S92_AND_GO_S93_CAP_BUILD_MAIN_AFTER_CAP_PASS. Root read both full
answers and recorded its resolution. The subsequent three-seed CAP pass
fulfills the numeric pitch condition only; actual main host metadata/device
binding and same-new-binary serving gates still apply.

Root corrects Opus's broad narrow-shape neutral/noisy description: unmasked
63/64/65 and1/2051 improve about14-16%,4/2051 about7%,12/11 about3.6%; width1
and zero controls are null. Keep all12 shape results and the masked regression.
The caller checks the first query of EACH32-query batch, not the entire
chunk; existing monotone host steps and staged identity publication provide
that scoped contract. No general cache-validity promise is added. Combined
review texts retain their original wording; root resolution is authoritative.
The2000+ tok/s prefill and150+ tok/s per-stream targets remain unmet. S30 RAM
recovery and the separate S87/S89 serving evidence are unchanged.

Exact small records: [attention_value_s92](../strata/flash-next/lowlevel_probe/attention_value_s92/README.md).
Raw root is `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`:
`attn-value-binding-s92-prep`, `attn-value-binding-s92-build-audit`,
`attn-value-binding-s92-runner-prep/run-clean`, and
`attn-value-binding-s92-root-runtime-decision.json`.
The completed manifest covers 106 files / 8383798B, SHA256
`e4e691e0559e2bf849a55021b5fcff57ca114c5d223be022cee05c1e9d2f215d`.
Large binary/ISA artifacts stay external; no duplicate public ISA is copied.
