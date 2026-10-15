S94 retained-work mismatch forensics (2026-10-15)

CONFIG -> Read-only inspection of retained S94 first 4K OFF/ON request,
not a new serving experiment. Same S94 executable in both recorded arms;
request prompt4059/cached0/completion256, spec4, spec-min-p0.5, default
suffix-draft3, lookup-chain0. Requested complete-resident value candidate
OFF versus ON; source/config identity is retained in the original stage.
No GPU, build, model payload read, new trace or repository edit was used.

COMMAND -> python3 <this directory>/analyze.py
Small native/API/config files and selected source ranges only. SHA256,
byte counts and absolute input paths are in findings.json. The first
preparation invocation failed an assertion because its log slice included
later OFF requests; narrowing it to the first request's native summary
and immediately following suffix summary fixed the scope before freezing.
No serving/artifact experiment was retried or gate changed.

RESULT -> Both request logs retain 120 solo verify windows, not only the
207/205 summary totals. Summing T-1 reproduces the API offered totals
exactly. OFF T histogram: T1=30,T2=24,T3=15,T4=51. ON: T1=30,T2=24,
T3=17,T4=49. The first 11 (position,T) records match. Earliest observed
geometry difference is window12 (zero-based11), position4078: OFF T4,
ON T3. Each then advances to4079, so both committed one token there
(accepted0, derived from next position under the pinned source loop).
The second width difference is window29, position4106: OFF T4 next4110
(accepted3), ON T3 next4109 (accepted2). The next three positions remain
shifted by one (OFF4110/4114/4118 vs ON4109/4113/4117), then both rejoin
at4121, window33. All subsequent position/T pairs match. The next-position
inference yields134 accepted drafts across the first119 windows in each
arm; subtracting from native136 gives final-window accepted2 in each.
The final-window count is inferred from the summary, not independently
logged. No draft token values or per-window confidence/state are retained.

Suffix summaries: four suffix windows in each arm; OFF accepted8/offered12,
ON accepted7/offered10. Thus suffix offered totals differ by the entire
2-draft aggregate delta. By subtraction non-suffix offers are195/195,
non-suffix accepted128/129. This does NOT identify which exact windows were
suffix windows, prove equal underlying MTP tokens or probabilities, or
prove a benign cause. Native window lines omit from_sfx, lookup proposal,
policy choice inputs and MTP probabilities. They also omit per-window
accepted count; the position-derived values use source loop semantics.

Prompt/cache evidence before decode matches bounds: cache0, request4059,
prompt start4058, one batched interval0..4052 plus6 window-read tokens.
Both first-request capture sequences are 3-token,2-token,4-token with
successful upload/sync. ON alone emits the eligible value HOST marker.
Read times differ, as expected from separate measured executions; none
of these facts proves complete internal state equivalence. API text and
usage match; output SHA256 is
4cc1afbabebee14a0f00b57b64d9ad10b331db0ad4c508fb39894cbfe3b1bca0.

Source clue, not causal diagnosis: current SYCL generate.cpp SHA256
6433d750e414aeb9c71135257ae40a8f9e38ea661537c251b50cabdbc2217d83
EXACTLY matches the retained S94 source pin. Lines11773..11807 first
select MTP T using dprob >= req_spec_min_p, then potentially replace T
with DraftPolicy::choose(T, lookup_k, match), then optional chain work.
The same source lines11935..11940 pass measured elapsed milliseconds
from round0 (after selection, before verify) through verify/commit/emit/
draft/adapt completion to policy.observe. The policy object is process
lifetime at8943, not freshly reset per request. Current draft_policy.cpp
and header excerpts show size-cost EMA alpha0.1, MTP accepted-token EMA
alpha0.05, lookup acceptance priors/decay, and expected tokens per measured
cost comparisons with default margin0.03. Current policy source SHA is
04eba40c79c272043de4bf9d86823e9050ed72f5563f3ca1361faffd66d2992d;
this task has NOT established a retained S94 policy-object/source pin.
The exact S94-pinned caller nonetheless establishes timing-derived inputs
are supplied and a policy chooses the suffix window. Retained logs lack
those per-window elapsed costs, policy tables, lookup candidates, source
selection and dprob. SSE arrival gaps cannot reconstruct them. Therefore
original S94 causality remains unresolved: timing-policy choice and
numerical/confidence/state differences cannot be distinguished here.
S93's different binary/process context is not a repeatability control.

VERDICT -> Keep S94 FAILED_NATIVE_WORK_MATCH, defaultOFF and unqualified.
The existing evidence narrows investigation to an actual first observed
width divergence at4078 and two offered-width changes; it does not clear
numerical correctness or establish a regression, nondeterminism or win.
Stop offline investigation here: the required decision inputs are absent.

One minimal next discriminator IF prefill is resumed: use the same finite
first4K OFF/ON protocol with BOTH arms explicitly configured
--suffix-draft 0 --lookup-chain 0 --spec 4 --mtp-max-t 4 --spec-min-p 0.
Existing flags suffice; no new diagnostic flag is needed. The pinned
caller then requests T4 except its initial T1, disabling suffix-policy,
lookup-chain and probability-dependent width selection. Context is far
from max200000. Preserve all exact output/usage/native offered/accepted,
identity, teardown and health gates; stop on the first mismatch. No broad
4K/32K/cache/full-context rerun is needed before that discriminator.
This is a NEW fixed-work configuration; passing would qualify only that
configuration and would not retrospectively explain or clear S94's
adaptive run. Disabling only --suffix-draft 0 leaves probability-gated
widths adaptive; STRATA_LOOKUP_CHAIN_FIXED only affects chained lookup
and does not freeze the suffix policy. A diagnostic log/replay would be
needed to prove the original policy's cause, but no such change or
experiment is prepared, executed or required by this report.
