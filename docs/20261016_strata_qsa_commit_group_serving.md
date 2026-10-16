# S144: three-row QSA commit grouping serving discriminator

CONFIG -> Fresh OFF and ON processes on the SAME S143 executable
`b5104c309877cf242b153075609c3386a1683c9cc388dea414fdd6522192f779`.
Only STRATA_QSA_COMMIT_GROUP_STEPS0/1 differs. Eligibility is groupLen==3 only;
every other length retains the old append loop. C2/C6 runtime words, owner
queues, parallel argmax and all other candidate flags stay OFF. Fixed suffix0,
lookup0,spec4,MTPmax4,min-p0,batch2/depth2/pairedhead,EP149/363/all24576resident,
two200k context allocations,32GiB parking and12288prefill remain common.
S143 source/build is committed796faaaa; actual model source decision5c9aa822
is referenced, not duplicated. Private S141 tested analytic RoPE only.

COMMAND -> Root materialized the frozen V1 S139-derived controllers ONCE using
actual successful S143 build/source receipt and root-attested S142 full-health
context. BEFORE either arm started, root added ONLY common STRATA_ROPE_TABLE0
to both configs, recording before/after hashes in root-common-variant.json and
serving-inputs.root_common_variant. Both actual native environments independently
record RoPE0, groupflag0/1 and both Q5 words0. All frozen V1 source hashes match;
no V2 was created/executed. Comparator normalization removes only groupflag and
arm log names, so common RoPE0 remains part of matched config identity.

One107-prompt/256-output C2 warm, then three fresh107/1500 C2 pairs per arm,
bounded arithmetic/thinking/tool functionals, normal QUIT and full post-health.
OFF completed before ON. Every ON pair immediately matched its fresh control
before another request. Timed clients ran under quiet CPU conditions while the
server retained its pair GPU lease; startup/functionals used ordinary CPU lease.
No oldbinary control reuse, automatic retry/reset/retime, extra baseline, new
CPU ring, profiler, full200k campaign or daily restore.

RESULT -> Root sequence exited0 and comparator PASS_MATCHED_DISCRIMINATOR with
all checks true. All four warm/timed pairs matched complete request/text/SSE
content/usage, native per-request cache/prompt/output/offered/accepted work,
windows/pair/solo counts, rounded row reports and observed capture layouts.
Admission overlapped. Functionals passed both arms. Normal process/wrapper exits,
final diagnostics, full CID+NAME removal, per-card and compiled collective
post-health all passed. Root build/source binding and explicit RoPE0 were verified.
Raw token IDs were not observed; no unprinted per-window row equality is claimed.

| Timed client metric | OFF | Group3 ON |
| --- | ---: | ---: |
| Aggregate output tok/s, median | 103.088499 | 102.983199 |
| Pair wall seconds, median | 29.101209 | 29.130965 |
| Stream0 decode tok/s, median | 53.516687 | 53.456732 |
| Stream1 decode tok/s, median | 55.860143 | 55.804132 |
| Stream0 TTFT seconds, median | 0.911069 | 0.904519 |
| Stream1 TTFT seconds, median | 2.081352 | 2.083826 |

Change of aggregate medians:-0.1021454683%. Individual repeat throughput changes
are-0.1021454683%,-0.0968869848%,+0.05992118695%; corresponding pair-wall changes
are+0.1022499120%,+0.0969809467%,-0.0598853030%. All samples remain retained.
These are one fixed OFF-then-ON sequence with three repeats nested within each
distinct arm process. They do not quantify a noise band, population effect or
regression. SSE content-event p50/p95/max in comparison.json are client event
gaps and can cover multiple tokens, not native token ITL or device service.

The ON marker appeared exactly once by warm and final closure; OFF emitted
none. It proves first eligible HOST selection, not all groups/calls, GPU
submission counts or independent device dispatch. Logged verifier captures
matched. Actual argv/env/library/map-hash and container state projections are
explicitly derived; raw proc maps/full SSE remain external with source identities.

VERDICT -> Matched finite discriminator passed, with no observed serving gain.
Keep grouping default OFF and park this exact candidate; no retest or promotion
is justified by this record. S141's private completed restore+append-chain
reduction of about42% did not forecast whole-serving improvement; the stages
are different scopes and do not establish a hardware/driver cause. No main
ISA or compiled-image identity was added as a claim. RAM-restore/fullcontext
qualification remains separate. Source-copy index and compact raw work/closure
receipts preserve provenance without binary/model/output rehashes. Curation
performed no GPU, build, model CLI, config generation or live source change.

[Compact exact serving and normal-closure evidence](../strata/flash-next/lowlevel_probe/qsa_commit_group_serving_s144/README.md).
