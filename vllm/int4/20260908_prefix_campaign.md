# INT4 GPU prefix caching qualification

### 2026-09-08 - GPU prefix caching is required for the INT4 daily driver

CONFIG -> User explicitly requests enabling conversational prefix reuse.
Start with exact qualified R276 image/INT4 weights/MTP4/FP16 KV/200K/c4.
Only configuration delta: --no-enable-prefix-caching becomes
--enable-prefix-caching. Environment, native stack and CPU tier0 unchanged.
COMMAND -> run_prefix_campaign.py, raw root
/mnt/vm_8tb/b70/results/int4_prefix_20260908/stock-mtp4. The idle serving
instance stopped cleanly through its owned lease and post-health. Candidate
starts only after prior exit.rc0. Register descriptive experiment IDs.
RESULT -> Candidate starting; no prefix qualification claim yet. Planned:
C4 answers, repeated guides,150K A/A/B/C/D/A/A (four distinct histories
exceed451K prior pool), four88K growing tool histories, cancellation,xhigh,
four110K+8K growth pressure, postpressure answers, four132K tool histories,
all164 coding problems after stress, final canaries and199K recall.
Inspect actual pool/admission/preemptions; do not infer eviction from request
count. Require matching recall, warm cached tokens >=90% of prompt and
warm TTFT <=25% of its cold request. Inspect growing tool-history hits,
all raw corruption/new coding failures and clean lifecycle health. Existing
cache-off INT4 controls are the paired correctness reference. No new profile
unless a changed native/runtime path justifies one; the matched R276 daily
collective profile is already qualified.
VERDICT -> Prefix-off serving is not the requested conversational outcome.
Promote GPU prefix caching only after measured reuse and correctness; if
stock MTP4 fails, diagnose the concrete failure and test a bounded fix or
MTP control. RAM offload and FP8 KV remain outside this campaign. Update
startup qualification and restore hotschmoe-dd, commit/push at milestones.
Upstream leads checked today (not applied or proven local fixes):
https://github.com/vllm-project/vllm/issues/53912
https://github.com/vllm-project/vllm/pull/48375
https://github.com/vllm-project/vllm/pull/43650


### 2026-09-08 - Review first prefix-on guide divergence

CONFIG -> Stock R276 MTP4, GPU prefix caching on, FP16 KV, no CPU tier.
COMMAND -> stock-mtp4 C4 and three2048-token guides, then fail-fast teardown.
RESULT -> C4 passes. Guides all coherent but byte-repeat flag false:
first two hashes3c52c745..., third99452102... exactly matches all three
cache-off controls. All text through the last closed code block is identical
across all six responses; only the trailing prose changes (syntax-note text
versus complexity heading). All three prompt-cache-hit counts are0, so no
cached prompt was exercised. Preserve raw job rc1. Lifecycle exit.rc0 and
post-health pass. Prefix-enabled pool446332tokens versus451562cache-off.
VERDICT -> This is not evidence of cache-state corruption or a prefix hit.
Allow a narrowly reviewed noncritical guide-tail variation only when every
coherence row passes and the complete text through the final closed code
block matches every cache-off guide. Changed code/earlier prose/loops still
fail. Keep byte-repeat claim false. Fresh stock-mtp4-b repeats and continues
the cache tests; retain all raw flags plus guide-review.json for any accepted
exception. No image or native-code change. Add one final cached-request
profile on this prefix-enabled graph to measure its actual collective shape
and paired rank entry/return counts; profiler inactive during timing tests.


### 2026-09-08 - Stock INT4 GPU prefix reuse works on long histories

CONFIG -> stock-mtp4-b, same R276 image/weights/MTP4/FP16 KV, prefix on,
446332-token pool, CPU tier0. Profiler configured but inactive during tests.
COMMAND -> 150K A/A/B/C/D/A/A and four8000-record growing tool histories.
RESULT -> All seven recall responses exact/correct. First cold A TTFT94.203s;
immediate repeat1.648s with148928/150045 cached prompt tokens. After four
distinct histories, A reports0 cached and93.756s (evicted); its next repeat
again hits148928 tokens and takes1.635s. No CPU reload involved.
Four growing tool histories pass32/32 checks in218.022s versus1521.243s
for cache-off INT4. All28 followups reuse98.53-98.85% of prompt tokens;
followup wall latency1.200-19.825s (initial concurrent prefills overlap).
VERDICT -> Actual GPU prefix reuse and post-eviction repopulation measured,
with correct answers/tool state so far. This addresses full recomputation
between turns while history remains cached. Cancellation, tighter growth
pressure, larger eviction tool histories, poststress coding/profile and
post-health are still running; no serving promotion yet. The reviewed
zero-hit guide-tail divergence remains recorded separately.


### 2026-09-08 - Reject stock INT4 prefix under oversized tool-history churn

CONFIG -> stock-mtp4-b; exact R276 image, MTP4, FP16 KV, prefix enabled,
446332 nominal GPU KV tokens, four admitted sequences, no RAM offload.
COMMAND -> run_prefix_campaign.py followed by manually queued postfailure
coding, coherence, 199K recall and warm collective-profile diagnostics.
RESULT -> Normal four88K growing histories passed32/32; an additional unsalted
(shared-cache) four2000-record control passed32/32 in32.292s. Four110071-token
prompts plus8192 output tokens each finished in519.040s but scheduler
preemption delta was0: this did not force active-sequence preemption.
The larger four132K-history run failed:23 checks in1378.876s, one session's
first answer timed out after900.009s (no response captured), and another
session's third tool request returned512 exclamation marks, finish=length,
prompt132502, cached_tokens0, created_cache_tokens131456. Two sessions
completed8/8. This is actual corruption under cache churn, not merely slow
queueing, and not proof that the bad request itself reused a corrupt hit.
Subsequent coding completed without symbol loops (157 base/149 plus of164),
short C4 coherence and199K recall passed. Lifecycle/profile evidence pending
when this entry was written. Raw evidence: results/int4_prefix_20260908/
stock-mtp4-b under the runtime root; manual-diagnostics.json records a paused
coordinator so the leased runner could finish postfailure diagnostics.
VERDICT -> Reject stock MTP4+prefix for daily promotion. Investigate the
installed Mamba manager ignoring drop_eagle_block using a Python-only
backport of upstream PR48375, with CPU regression and exact native-hash
comparison before GPU retest. This is a candidate, not a proven root cause.
Public serving has not been promoted; restoration remains required.


### 2026-09-08 - Native-preserving Mamba boundary candidate starts

CONFIG -> PR48375 aligned Mamba lookup backport into the installed R276
Python module only. Base image521eb277..., derived image43e77a22...;
source hash1a0dedb7 -> a3ab3679. No PYTHONPATH or entrypoint override.
COMMAND -> build_mamba_eagle_image.py; test_mamba_eagle_drop.py in CPU-only
base and candidate containers; run_prefix_campaign.py --churn-first --profile.
RESULT -> Base regression confirms Mamba retains80 tokens where full
attention drops to64. Candidate matches64, including zero/one-block and
no-drop controls. Six native/core routing file hashes match exactly.
Stock-b final warm profile has948 completed c10d calls per rank plus948
wrappers (not1896 independent collectives), matched shapes/counts; cached
prefill1137 rows, decode1/5 rows. Both stock-b teardown health gates pass.
VERDICT -> CPU bug is confirmed and patched; GPU root cause and serving
correctness remain unproven. Candidate must pass large-history churn on a
fresh process and again after pressure, plus existing reuse/quality/lifecycle
gates. Do not promote based solely on the CPU regression.
