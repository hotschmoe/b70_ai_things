# Source52 native two-stream cache co-admission V2

CONFIG -> Exact V1 source52/two-stage/native2/shared1536/pref512/GEN8 inputs,
raw all49 gates and original observer quotas. Only scheduling difference is
existing STRATA_BATCH_DECODE_SHARE=0. Engine/model/math are unchanged.
No source rebuild is needed. Both BGEN commands precede any drain as before.

COMMAND -> Exact root argv is in
/tmp/b70-source52-native-batch2-cache-v2-launch-agent.json.
Producer/reader source plans are source52-batch2-cache-recipe-v2.json and
source52-batch2-cache-reader-source-plan-v2.json under strata/flash-next.
Independent public read after actual successful producer terminal:

  PYTHONPATH=strata/flash-next python3 strata/flash-next/validate_source52_batch2_cache_v2.py --root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source52-native-batch2-cache-v2-run-v1 --output NEW_EXTERNAL_OUTPUT.json

Actual SDK52 generate.cpp10445 documents the default0.5 decode share between
incoming prompt chunks. At10526, share<=0 skips these batch steps. Thus the
first admitted cold slot can remain at its first token while the second cold
prompt prefills. After both admissions, ordinary decode still runs native
rows2. This is a declared co-admission diagnostic policy; it does not qualify
production scheduling/fairness, simultaneous prefill or clean latency.

All V1 gates remain. Both cold and cached cohorts must provide actual native
rows2/mask3 and all48 residuals/full head at their identical first later input
positions, finite/bitwise equal with complete output histories. Missing cold
vectors cannot be replaced with cached vectors or inferred from output IDs.
Source/header/binary/model identity, page/stat, health/kernel, quota/EOF and
owned retirement boundaries remain. No larger output cap or capture budget.

RESULT -> Actual V1 root25323 terminal FAILED541s. Native process exited0 with
all6 requests complete and clean post-health/kernel/identity/removal. Raw roster
has392 vectors: RID1/2 direct49 each, RID3/4 admission49 each, RID5/6 admission49
plus later49 each. Cold RID3 emitted7 rows1/mask1 steps and BDONE8 before RID4
admission completed; RID4 then emitted7 rows1/mask2 steps. Cold cohort did not
execute a two-row window. Cached5/6 did execute rows2/mask3 with both-stage
later49. There is no reader label/schema defect or recoverable cold vector.
All6 saved8-token histories matched their A/B roles. First49 serial/cold/hit
arrays were bitwise equal for each role. These observations do not close the
missing cold later49 or change the original failed report.

V2 genuine actual52 CPU command construction passed with one phase/six inputs,
exact decode_share0 and all existing quotas/source/math settings. No GPU,
compiler, model payload read or native CPU model process was run by author.

VERDICT -> Minimal flags-only V2 candidate for the previously unobserved cold
co-admission window. Actual runtime and public reading still pending; full
cache family, native4/6, API/shelf/fairness/clean latency remain unqualified.
