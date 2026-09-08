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

### 2026-09-08 - Mamba drop backport does not clear oversized-history gate

CONFIG -> Derived image43e77a22 from exact R276 image521eb277, MTP4,
FP16 KV, prefix on,200K/c4, no CPU tier; unchanged native hashes.
COMMAND -> mamba-eagle-mtp4, --churn-first. Coordinator CPU checks also
exercise the installed HybridKVCacheCoordinator: eagle groups0/1, aligned
832-token blocks, partial-hash path false. Base hit3328, patched2496.
RESULT -> First four132K histories captured14 checks:13 correct, session3's
first answer timed out at900.017s with responseNone. No partial output was
captured for that request, so do not label this timeout a punctuation loop.
The original stock-b run remains the actual512-exclamation corruption proof.
SIGTERM cancelled the client after this decisive failed gate; raw remaining
sessions are incomplete, job rc=-15, manual-failure.json records the action.
Normal server teardown/post-health is underway, followed only on exit.rc0
by stock MTP0+prefix control. No serving promotion and no root-cause claim.
VERDICT -> Reject this backport as a sufficient daily fix. Retain the proven
CPU regression separately from unsuccessful GPU qualification. Next test
changes only speculative decoding from MTP4 to0 on the original image.
Add optional SSE capture to the tool probe to preserve partial timeout
output; fragmented tool arguments, final usage and incomplete-stream
retention passed CPU checks. The first MTP0 churn diagnostic uses streaming;
its final postpressure churn remains the original nonstreaming API path.
Raw roots: results/int4_prefix_20260908/mamba-eagle-mtp4 and stock-mtp0.

### 2026-09-08 - Diagnose mixed-prefill blocking; prepare 4K-batch candidate

CONFIG -> Stock R276 MTP0 with prefix caching, FP16 KV,200K/c4,32768
prefill budget. Nominal pool535222 tokens, versus446332 with MTP4.
COMMAND -> Streaming four132K tool histories, then profile_mixed_prefill.py
with one cold132K recall and three short128-token guides, under the same
owned lease. Pause coordinator only while the diagnostic queue completes.
RESULT -> Captured10/10 completed tool/answer checks before deliberately
cancelling this incomplete campaign. One otherwise correct tool-call stream
had about17-second fragment gaps. No MTP0 corruption was observed, but this
is NOT a completed correctness qualification. Planned154K oversubscription
extra was cancelled, not run. DRM fdinfo snapshot did not establish a
residency shortfall. All four mixed-profile responses pass: long recall
TTFT76.62s, subsequent max gap0.04s; short-guide TTFT0.32-0.65s and max
chunk gap23.17s. These are profiled diagnostic timings, not a speed result.
Both ranks match2324 completed CPU collective events (1162 c10d calls plus
wrappers); actual allreduce rows include32451 four times per129-layer-call
set, plus1667/30/594/61. The large prefill chunks support head-of-line
blocking as a cause of long gaps; they do not explain the original stock
punctuation corruption on their own. Coordinator resumed for normal health.
VERDICT -> Do not promote or call MTP0 failed for corruption. Next candidate
retains the scoped Mamba backoff image43e77a22 and MTP4, changes only
--max-num-batched-tokens32768 ->4096 in Config.json (Env identical), and
keeps200K/c4/FP16 KV/no CPU tier. Normalize extra churn workloads from the
measured new pool, since lower prefill budget may grow KV capacity. Require
both original132K case and capacity-normalized churn, before and after
pressure. Prepared authenticated150K cold/warm public-frontdoor smoke for
final serving validation; it is not yet run. Daily qualification unchanged.

### 2026-09-08 - MTP4 still corrupts with the backoff fix and 4K prefill

CONFIG -> mamba-eagle-mtp4-b4096, image43e77a22, MTP4, FP16 KV,
200K/c4,4096 prefill budget. Nominal GPU pool524324. Prepared normalized
four151839-token histories (13775 records, estimated83032-token excess).
COMMAND -> First streaming four132K tool histories, before any broader
campaign. Inspect per-session SSE rather than waiting for a900s timeout.
RESULT -> Four complete checks correct, then session3's initial tool request
streamed396 exclamation marks before cancellation. This is actual corruption,
not just slow queueing. It was the first request for that session's unique
cache salt; final usage was not received, so do not invent a cached-token
counter for the cancelled response. Preserve full SSE and manual-failure.json.
Client SIGTERM and runner STOP ended the failed workload; normal teardown,
per-card and compiled two-rank post-health passed, exit.rc0. Normalized
extra was not reached. Small-batch MTP4 plus the backoff patch is rejected.
VERDICT -> Prefill blocking explains long gaps but not this correctness
failure. Next is original-image MTP0 with4096 prefill and prefix enabled;
no patch-image promotion. Dynamically grow both tool-churn sizes and the
number of distinct150K reuse histories from the measured pool, and allow
appropriate total job duration without weakening the900s request gate.
The launcher now pins the qualified prefill budget as well as MTP depth;
its existing qualification is unchanged. Also clarify the preceding profile
entry:129 is a collective-call count per forward-shape set, not129 layers.

### 2026-09-08 - Retarget prefix serving to MTP2/MTP3

CONFIG -> User prioritizes MTP2/MTP3 toward >80 tok/s and permits rare
punctuation loops with bang-guard recovery. FP16 KV and GPU prefix caching
remain required; CPU offload stays disabled. Keep4096 prefill budget fixed
for the first depth comparison.
COMMAND -> End stock-mtp0-b4096 control through STOP and client SIGTERM;
run_prefix_campaign.py --mtp3 --screen on the original pinned R276 image.
Inspect installed engine/arg_utils.py and hotschmoe/bang-guard source.
RESULT -> Incomplete MTP0 control recorded12/12 correct completed checks;
normal teardown and post-health exit0. No full qualification claimed.
Installed R276 API-server batch default for32GiB cards is2048; the prior
FP8 and INT4 daily32768 setting was explicit. Chunked prefill permits
prompts larger than the scheduler token budget. Bang-guard detects32
consecutive exclamation marks, excludes the failed attempt, rotates cache
salt and permits up to3 consecutive retries by default. Its own README
explicitly leaves real vLLM recovery unverified. MTP3 screen is running.
VERDICT -> Replace zero-loop promotion preference with measured raw loop
frequency and recovery evidence per user instruction. Do not infer rarity
from a small sample or silently count recovered attempts as clean. A screen
is only a bounded comparison (coherence, decode, growing tools, shared cache,
thinking and150K warm reuse), not a shelf qualification. Preserve all earlier
failures. Source/default evidence and incomplete-control record are under
/mnt/vm_8tb/b70/results/int4_prefix_20260908/.

### 2026-09-08 - Measure rare loops and recovery separately

CONFIG -> User target is under1 percent raw bang-loop requests. User reports
old FP8 incidents clustered into about30 minutes between long clean periods;
this is an anecdote to guide testing, not proof of poisoned cache state.
COMMAND -> Inspect bang-guard d341fe6c9a3b53061e9936b0cb78b6f08e6f6438;
run its real Pi CLI offline integration with installed Pi0.84.3. Add bounded
live-model recovery probe and optional diagnostic tool-probe cancellation,
salt rotation, timestamped attempts and up to3 retries.
RESULT -> Offline actual Pi test passes: stream cancelled, corrupt context
excluded, salt rotated, recovered answer returned. CPU diagnostic checks
pass across stream chunks, reasoning, escaped tool JSON, and four concurrent
sessions with injected failures; raw failures remain separate from successful
checks. Live Pi probe is queued under the MTP3 server lease: first response is
explicitly injected; subsequent response goes to the actual88K-history model.
VERDICT -> Neither an injected recovery nor a small clean sample establishes
natural fault recovery or an under1 percent long-run rate. Measure natural
incidents, retry outcomes and clusters separately. Guard emulation in the
Python history probe is explicitly not execution of the Pi/OMP extension.
The new live Pi probe executes the actual pinned extension. No guard source
or client installation was changed.

### 2026-09-08 - MTP3 speed screen and functional guide review

CONFIG -> Original R276 image, MTP3, FP16 KV, GPU prefix on,200K/c4,
batch4096,util0.96,cgroup64GiB,CPU tier0. Pool530468 tokens.
COMMAND -> stock-mtp3-b4096-screen; author's fixed12-prompt benchmark,
actual Pi injected-failure/live-recovery probe, concurrent canaries and
three2048-token LRU guides. Review guide code in the pinned CPU grader
container with no network, devices, writable root or extra capabilities.
RESULT -> Author metric95.363772 tok/s (class-balanced first99 intervals),
zero prompt-cache hits and all canaries pass. Complete token arrays12/12
match the earlier strict MTP0 control; this cross-shape comparison alone is
not a fresh matched restart qualification. Actual Pi recovery into the live
88K-history model passes after the explicitly injected initial failure.
All3 guides coherent, but raw exact-repeat gate fails: generated custom LRU
variants differ in implementation, not only prose. All6 extracted complete
Python blocks pass18000 operations against an OrderedDict reference. Raw
failure remains preserved. Runner stopped before growing-history screens;
normal teardown and post-health exit0. MTP2 matched screen is running.
VERDICT -> Functional guide variation is reviewed, with no byte-exact claim.
MTP3 cache qualification remains incomplete. User-scoped final campaign now
keeps prefix reuse/eviction, bounded132K concurrent histories with separately
counted bang retries, coding, cancellation and health; it omits the previous
forced32K-output offload-pressure workload. Launcher accepts qualified MTP2
or MTP3 manifests, but the promotion manifest remains unchanged.

### 2026-09-08 - MTP2 meets speed target;4K history reuse is incomplete

CONFIG -> Original R276 MTP2, FP16 KV, GPU prefix on,200K/c4,batch4096,
util0.96,cgroup64GiB,CPU tier0. Nominal pool536758 tokens.
COMMAND -> stock-mtp2-b4096-screen fixed author benchmark, concurrent
canaries, four88K tool histories, isolated150K A/A/B/A reuse, xhigh canaries,
guide functional review and CPU /tokenize prefix comparison.
RESULT -> Author metric84.557267 tok/s; all12 complete token arrays match
strict MTP0 control. Initial and xhigh concurrent canaries pass. Incomplete
history control6/6 completed checks correct, but both completed followups
report zero cached tokens. Cancelled the remaining history requests and
pending4K promotion; no corruption claim. CPU tokenizer counts match actual
usage88314 ->88380, with all88314 initial tokens preserved as a prefix.
Isolated150K A/A/B/A answers4/4 correct: TTFT95.260,1.671,95.176,1.687s;
cache hits0,148928,0,148928. Thus GPU caching is functional for these repeated
prompts, but normal growing-history reuse was not established. All3 guides
coherent but non-identical; all6 Python blocks pass18000 reference operations.
Normal teardown and post-health exit0. MTP2/batch32768 direct comparison is
running, with an early8-check88K history arm before full qualification.
VERDICT -> Do not promote4K batching based on nominal capacity or isolated
reuse alone. The explicit old32K budget remains a candidate; do not infer
that smaller is always better. Add an early warm-hit/latency gate to stop
qualification if correct responses are recomputing. A new raw bang-rate
audit separates workload families, retries and natural incidents; CPU tests
verify the denominator and32-bang detection. It makes no long-run rarity or
independence claim and excludes injected Pi failures and intentional cancels.

### 2026-09-08 -32K batching restores the MTP2 tool-history reuse check

CONFIG -> Original R276 MTP2, FP16 KV, GPU prefix on,200K/c4,batch32768,
util0.96,cgroup64GiB,CPU tier0; profiler configured but inactive during tests.
Nominal KV pool456916 tokens (15.62GiB per rank), versus536758 at4K.
COMMAND -> stock-mtp2-b32768-daily/00b-strict and00a2-tool-boundary:
four concurrent88314-token histories, one tool call and answer each.
RESULT -> Author metric83.150939 tok/s with all canaries and zero cache-hit
checks passing; complete outputs12/12 token-exact against the4K MTP2 screen.
Boundary test8/8 correct,8 attempts,0 bang incidents,194.231s total. Cold tool
calls189.405-192.061s; all four followup answers reused87360/88380 tokens and
finished1.602-2.473s. The4K control's two completed followups had zero cached
tokens despite a full88314-token common prefix. Three long guides remain
coherent but not byte-identical; six Python blocks pass18000 reference ops.
Initial00a attempt failed in argparse before inference because a single-dash
salt looked like an option. Raw rc2 preserved;00a2 uses explicit --salt=value.
Queued legacy salt arguments now parse literally; CPU parser check passes.
VERDICT ->32K is the selected batching candidate on direct measured reuse
benefit, with slower cold concurrent first responses as the known tradeoff.
Do not claim the internal cause or that eight clean requests establish the
under1 percent long-run target. Full quality, eviction, recovery, teardown
and fresh public-serving qualification remain in progress.
