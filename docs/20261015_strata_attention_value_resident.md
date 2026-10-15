# S94 complete-resident attention value caller

Date: 2026-10-15. S94 closed FAILED_NATIVE_WORK_MATCH. The candidate
activated, but the first ON4K request failed native-work parity. Both arms
completed normal teardown and full post-health. No candidate speed,
regression, default, daily-driver or shelf qualification is established.

CONFIG -> S93's value kernel did not activate because its caller required
staged KV mode1, while the measured launch used fixed resident mode0. S94
changes only prefill caller eligibility. The complete identity-page domain
is staged OR `(st.kv_mode == 0 && st.kv_elastic < 0 && st.n_pages > 0 &&
st.n_slots >= st.n_pages)`. Existing INT8/full32/cap2051/first-width2051
guards remain. The wrapper, exact flag1/defaultOFF behavior, math, kernel
tags, launch geometry, pools and publication order stay unchanged. Mode2
rings and partial/elastic resident pools remain excluded. Short and decode
paths stay original.

The source proof follows actual launch flags through each SessionState's
KvPlan and prefill caller. Mode0 creates identity table[i]=i and waits for
its upload; mode0 resolution does not remap it. Zero/reset and snapshot
restore mapping changes belong to other modes. Nonelastic slots>=pages
bind allocated extent, not virtual capacity. Existing INT8 append publishes
the current chunk on the same in-order queue before attention. Complete
past history remains the old caller's contract. This is source/config
reasoning, not an independent table/pool readback or general validity claim.

COMMAND -> Root applied caller-only patch67e608b6; new prefill source2a1d8faa.
One prefill TU, one archive and one main link completed in175s
(Ninja164.079s), e41/DPC++2026.1. New main SHA256:
`d94e96ca399d8ecf1f15403db754b645697e82eaa72c052a2d18f4b29defc247`.
The whole attention object06eb664c and kernel archive5d465c2d remain exact
S93. No old36-case or CAP numerical rerun was needed.

RESULT -> A3s CPU-only audit rechecked the new final four selected
Info/cache/guard/name-pointer bindings and all13 materialized attention
tags. Original/new chunk and merge registered names each resolve to one
validated full device image. All four full-image hashes match S93 exactly;
existing ISA and resource evidence was reused only after this check and
its retained ASM hashes were verified. No new ISA decode or GPU operation
was required for the audit. Scratch/private/spill field absence remains
field absence, not a zero-traffic or occupancy result.

Actual consultations `claude-opus-5-5`56.118s and
`claude-fable-5-1`75.571s accepted conditional source progress. Root read
both full answers and resolved actual batch-state initialization, width
saturation, prompt-XMX fallback, resident layout/extent, queue order and
identity-table publication. S93 remains failed activation, not a candidate
numerical failure, neutrality result or regression. Its numerical/private
evidence is retained. The unchanged header's staged-only comment is
historically overnarrow; the caller proof explains the same operand contract
without rebuilding the unchanged attention TU for a comment.

The finite serving protocol uses the same new executable in both arms,
explicit prompt-XMX0/KV-ROT0 and otherwise matched controls. A107-token warm
does not reach full width; first4K ON must emit the single eligible HOST
marker before remaining requests. That marker proves host selection, not
independent GPU dispatch/counts. Full output/SSE/usage/native-work,
functionals, normal QUIT, full CID+NAME removal and health gates remain.

Observed gate: OFF completed six prefill requests and functionals. Both
warm streams matched output/work/capture. The first ON4K request emitted
the candidate HOST marker exactly once and matched request/text/usage and
output SHA4cc1afba. Cache0/prompt4059/predicted256/accepted136 matched, but
offered draft_n was207 OFF versus205 ON in both API and native serve
summaries. The client stopped with exit1 after7s. No other ON prefill or
functionals, retry or speed summary was attempted. Activation is now
observed at the host, but completed native work is unmatched. This does
not establish a measured regression, numerical failure or benign
nondeterminism; no diagnosis is claimed. Equal output is not a complete
internal numerical/state equivalence proof. The frozen timing summary is
null. Each arm exited normally through QUIT, with full CID+NAME absent,
wrapper0 and per-card plus compiled collective post-health0; ON closure
took59s. No forced stop or experiment retry was used.

Existing raw context: S93's inactive flag0/1 baseline paths each offered205,
while S94 OFF offered207 and ON205. All four accepted136 and produced the
same full output SHA. These are different binaries/process contexts, not
a matched same-binary OFF/OFF repeatability control. They neither identify
the cause nor clear the candidate. The S94 work gate remains unchanged.

VERDICT -> Park the prefill candidate with defaultOFF/unqualified. Pivot
to the separately prepared currentC2 operation/readiness profile with
candidates OFF. No repeated numerical/CAP/cache/200k campaign or offered-work
diagnosis follows from this result. Actual postconsultations
`claude-opus-5-5`34.340s and `claude-fable-5-1`20.127s, each resumed from
its own source session, accept keepOFF/park/pivot. Root read both full
answers and corrected the cross-binary counter interpretation. A proposed
production-pool operator check is deferred; it would touch GPU and require
the applicable GPU/CPU leases and health, not just a CPU wrapper. No such
oracle was run or is mandated as the only possible causal investigation.
Targets and S30 RAM recovery remain unchanged. Raw
source/build/audit/actual consultation records are indexed in
[the small-record bundle](../strata/flash-next/lowlevel_probe/attention_value_resident_s94/README.md).

Raw root: `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.
Completed closures: `attn-value-resident-s94-build/build/root-build-closure.json`
and `attn-value-s93-post-s94-source-reviews-20261015/root-post-source-resolution.json`.
Final runtime:
`attn-value-resident-s94-serving/{root-runtime-decision,comparison}.json`.
Final postdisposition:
`attn-value-resident-s94-post-reviews-20261015/root-post-resolution.json`.


## S105: retained-log work forensics

Both S94 first4K request logs contain120 verify windows. Their first width
difference is position4078: OFF4 versusON3; the second is4106, again4
versus3. Summing T-1 reproduces207/205 exactly. The position sequences
rejoin at4121. Suffix offers12/10 account for the aggregate difference,
but the logs do not identify individual suffix windows, MTP probabilities
or policy decision inputs. Causality remains unresolved and S94 stays
FAILED_NATIVE_WORK_MATCH/defaultOFF.

The exact S94-pinned SYCL caller permits a timing-trained suffix policy
after confidence gating. Current policy source illustrates that mechanism
but is not independently bound to the retained S94 object. Existing flags
can remove both selection paths for a NEW bounded comparison: BOTH arms
--suffix-draft 0 --lookup-chain 0 --spec 4 --mtp-max-t 4 --spec-min-p 0,
including any request-level spec_min_p override. This requests T4 except
the initial T1. Preserve exact work/output/identity/teardown/health gates.
A pass would qualify that fixed-work configuration, not explain or clear
S94's adaptive run. No serving test was run by this forensic stage.

[Exact excerpts, pinned sources and findings](../strata/flash-next/lowlevel_probe/attention_work_forensics_s105/README.txt).
