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
