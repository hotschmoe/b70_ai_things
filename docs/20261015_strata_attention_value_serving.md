# S93 staged-prefill attention value pipeline

Date: 2026-10-15. S93 closed as FAILED_ACTIVATION. CAP, source, main build
and binding checks passed, but the serving caller did not select the new
kernel. There is no candidate serving speed or neutrality result and no
default, daily-driver or shelf qualification.

CONFIG -> S92's private value-pipeline lead is scoped to the existing eager
staged INT8 prefill fallback: 32 queries, all widths2051, cap2051, page4,
24 query heads/2 KV heads/HD256, identity-mapped nonnegative staged pages.
Exact `STRATA_ATTN_VALUE_PIPELINE=1` selects the new wrapper. Absent/other
values, lane-cell requests, short/ragged batches, decode and verify/MTP keep
the original entrypoints. The existing prompt-attention path remains first.
Default stays OFF. S92's masked-page regression remains relevant outside
this bounded caller contract; it is not hidden or declared invalid math.

The template's default branch reconstructs the whole original attention
source; its value branch reconstructs the retained S92 arithmetic source.
Probability storage reuses query SLM after its existing barrier, and the
value loop preserves each head's FMA order. The new chunk and merge class
tags belong only to `qsa_prefill_attn_value_batch`; the old eleven remain.
A once-only eligible host marker establishes host selection, not an
independent device-dispatch count. The first eligible request pays that
logging cost. No new queue, event, copy or per-operation counter was added.

COMMAND -> Root ran one numerical-only production-pitch CAP case in three
seeds, then applied the exact source and built two TUs/two archives/one main
link. Main binary SHA256:
`551cca4140b3f014bfb800e2f6d2c5d9a46733da02b835d0b2ead9fa5cf05db3`.
The retained recipes and root closures are indexed in
[the small-record bundle](../strata/flash-next/lowlevel_probe/attention_value_s93/README.md).

RESULT -> The CAP32/width2051/cap2051/page4 case passed full scratch/output
bytes, guards, read-only inputs, initial/repeat, changed-query/repeat and
restore in seeds7/31/109. Identity, health and teardown closed with exit0
in25s. It recorded no timings and makes no independent ideal-attention
accuracy claim. It resolves the packed ID pitch difference from S92's
cap2112 fixture; it does not qualify the later main code generation.

The main build completed in174s (Ninja162.813s), pinned to e41/DPC++2026.1.
Root verified the unchanged native/verify/generate/cache and other build
inputs. Actual object relocations and final Info/cache/guard/name pointers
separate original/new chunk and merge ownership; each selected registered
name resolves to one validated device image. All13 attention tags are
materialized. Four selected images were decoded; none was assumed identical
to the private fixture. Chunk metadata reports SIMD32/GRF128/SLM15872/
barrier1/EU threads8 for both variants. Scratch/private/spill fields are
absent, which does not prove zero runtime traffic or occupancy.

The first read-only parser was stopped after113s because it split the whole
remaining symbol string table per symbol. A bounded first-NUL lookup made
host parsing complete in0.54s; correcting YAML whitespace/repeated metadata
names then completed the four-image audit in14s. Failed attempts and their
sources remain retained; no rebuild or GPU retry was needed for this repair.
Future audits should use this linear parser and selected-image lookup.

Actual consultations were `claude-opus-5-5` and `claude-fable-5-1`:
S92 post/S93 source40.775s/81.541s; main/runner48.918s/109.727s. Root read
the full responses and closed concrete ownership, cleanup, capacity and
barrier conditions before serving. These source consultations do not
establish runtime correctness or performance.

Same-new-binary flag0/1 serving completed six OFF prefill requests and OFF
functionals. ON completed its two warm streams, three4K requests and the
first32K request. All six completed cross-arm responses matched full
requests, text, SSE, usage and native work. Both final logs contained zero
candidate host markers. The client stopped with exit1 at the mandatory ON
32K activation gate; the remaining two32K requests and ON functionals were
not attempted. `timed_summary` is null. No retry or aggregate speed claim
was made. Each arm exited normally through QUIT, with full CID and name
absent, wrapper exit0 and per-card plus compiled collective post-health0.

Source diagnosis: the current launch omits `--kv-resident`, leaving
`kv_resident=0`. Its fixed resident `KvPlan` uses mode0 with slots=pages,
so `st.kv_mode==0`. The new caller requires staged mode1 and never reaches
the candidate wrapper. Correct static kernel bindings cannot establish
eligibility of this caller. This is failed activation, not a candidate
numerical failure, regression, neutrality result or hardware fault.

VERDICT -> Retain the private S92/CAP evidence and default-OFF source.
Serving effect remains unqualified. S94 is a caller-only source proposal
to prove the actual resident identity mapping before any new guard change,
build or test; it is not applied, built or measured here. Future reviews
must trace actual launch flags through the KV plan and caller before using
a server lease. Goals remain unchanged. Keep S30's RAM-recovery evidence
and S92's private primitive results separate.

Raw root: `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.
Relevant closures: `attn-value-cap2051-s93-root-runtime-decision.json`,
`attn-value-main-s93-build/build/root-build-closure.json`,
`attn-value-s92-post-s93-source-reviews-20261015/root-post-source-resolution.json`,
and `attn-value-s93-main-runner-reviews-20261015/root-source-resolution.json`.
Final serving disposition:
`attn-value-main-s93-serving/{root-runtime-decision,comparison}.json`.
