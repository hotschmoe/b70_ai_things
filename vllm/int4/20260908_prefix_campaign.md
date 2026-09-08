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

### 2026-09-08 - MTP2 prefix reuse survives eviction and growing tools

CONFIG -> stock-mtp2-b32768-daily, original R276, FP16 KV, GPU prefix on,
MTP2,200K/c4,batch32768,CPU tier0; same candidate as preceding entry.
COMMAND ->03-reuse:150K A/A/B/C/D/A/A;04-tools:four88K histories with four
tool/answer turns each, streaming evidence and optional bang retry detection.
RESULT -> Reuse7/7 correct and exact. Cold TTFT93.648s, warm1.668s with
148928/150045 cached tokens. Four distinct histories exceed the456916-token
pool; revisiting evicted A cached0 and took93.881s, then next A hit148928 and
TTFT1.675s. Growing tool histories32/32 checks,32 attempts,0 bang incidents,
218.761s total; all28 followups reused98.5315-98.8459 percent of prompt tokens.
VERDICT -> GPU prefix reuse and correct recomputation after eviction are
measured on this candidate. Larger-context churn, shared-prefix clients,
cancellation, coding quality, final health and fresh public deployment remain
in progress. No long-run under1 percent incident-rate claim is made.

### 2026-09-08 - One unsalted shared-prefix bang; continue with measured recovery

CONFIG -> Same MTP2/batch32768 candidate;04b-shared-tools omits cache_salt,
four22K histories and32 tool/answer checks. This arm did not enable retries.
COMMAND -> Inspect raw session0 turn3 answer and audit_bang_rate.py. Continue
on a fresh matched lifecycle with a fixed128-check shared-prefix recovery
soak, then the still-outstanding cancellation/churn/coding/profile checks.
RESULT ->31/32 shared checks correct; final session0 answer is512 bangs,
finish=length,21632/22662 prompt tokens cached. No timeout or hardware fault.
Normal teardown and post-health exit0. Independent raw audit counts1/106
across completed test mix (0.9434 percent), but1/32 shared (3.125 percent).
Old summary bang_attempts=0 reflected disabled live detection, not absence
of raw corruption; new summaries also scan completed unguarded responses.
CPU tests pass for32-bang detection and four concurrent shared requests that
switch to private salts on an injected failure, preserve clean histories and
count raw retries. No natural recovery was observed in the stopped process.
VERDICT -> Per user's rare-loop tolerance, one incident is not automatic
rejection. Do not use the aggregate to establish a long-run under1 percent
rate or hide the shared-workload result. Fresh continuation fixes the soak
size at128 checks in advance; natural failures and successful recovery remain
separate. It resumes after the prior measured prefix/eviction/growing-tool
checks, retains all old failures, and has its own continuation marker rather
than falsely marking the prior campaign fully passed. Actual Pi recovery is
also queued for MTP2; its initial failure is explicitly injected.

### 2026-09-08 - Compare MTP decode and accept user under5 percent tolerance

CONFIG -> User accepts an observed bang fraction below5 percent and asks
for reproducible diagnosis. This supersedes the earlier under1 percent target
for current acceptance, without rewriting earlier failed gates. FP16 KV and
GPU prefix caching remain required; CPU offload remains disabled.
COMMAND -> Review fixed12-prompt strict results and matched2048-token guide
workload, audit fixed128-check MTP2 shared recovery soak, and continue MTP3
qualification under --bang-fraction .05. Reattach only the coordinator to the
same leased server PID272967; no GPU process restart or runtime change.
RESULT -> INT4 strict MTP2=83.15094tok/s at32K prefill and84.55727 at4K;
MTP3=95.36377 at4K. Prefix-off MTP4 two-fresh mean100.21428 has a different
profile and is not a fully controlled MTP comparison. Clean-package FP8 MTP3
guide result a0clean/03-decode=6144/75.59694=81.273tok/s, exact repeats.
INT4 guide MTP2/4K=98.24 and MTP3/4K=123.75tok/s, coherent but variable
outputs; do not compare these full-output rates to strict first100 timing.
MTP2/32K has456916 shared KV tokens; current MTP3/32K reports451562,
versus old FP8 daily292968. All use FP16 KV. Completed INT4 prefix-off
HumanEval base/plus158/150 versus clean FP8 157/152; MTP2/3 quality pending.
MTP2 fixed soak:128 correct checks after134 raw attempts,6 bangs (4.4776
percent), all recovered. Every incident is logical session0; two occur after
private-salt rotation. MTP3 already reproduces session0 bangs. Add full raw
request capture and logical session-order controls to distinguish content,
submission order and concurrency. Thread submission order is not proof of
actual backend batch-row order. Queue reverse/shift/singleton/isolated arms.
VERDICT -> MTP2 observed rate is within the newly accepted tolerance. This
small correlated sample is not proof of a long-run below5 percent rate.
A shared-cache-only explanation is insufficient; root cause is unproven.
Finish matched MTP3 speed, coding and lifecycle qualification before public
promotion. Retain all prior failures and report workload-specific fractions.

### 2026-09-08 - MTP3 preferred; historical mixed-batch NaN lead recovered

CONFIG -> User leans toward INT4/MTP3/GPU prefix;32K prefill and FP16 KV.
Current stock-mtp3-b32768-daily retains base R276 identity and451562 KV tokens.
COMMAND -> Fixed128 shared checks; strict suite; reversed/shifted logical
submission IDs, singleton, private salts, and logprobs; inspect archived
June26-27 journal read-only and git commit ca762fa. No old runtime restored.
RESULT -> MTP3 fixed soak128/128 correct after132 attempts,4 bangs=3.0303
percent, recovered. Strict94.82656tok/s and12/12 complete output token arrays
exact against MTP3/4K. Three2048-token guides byte-exact,6144/49.98355=
122.9204tok/s full wall. Current150K cold prefill93.6631s (~1602tok/s) versus
valid installed FP8 a0b/04-reuse93.2081s (~1610tok/s), both32K budgets.
This isolated difference is about0.5 percent; not a meaningful prefill win.
Reverse3,2,1,0:32/33 attempts, one bang on session3 answer0. Shift1,2,3,4:
32/33, one bang on session1 answer0. Singleton0:8/8, no bang. Four private
salts:32/33, one bang on session0 answer2. Each bad answer's first bang
preceded the other three answers' first deltas while their requests were
outstanding (bang-timeline.json). This supports order/overlap sensitivity,
not a unique session0 prompt or a requirement for cross-session cache hits.
Logprobs arm32/32, no bangs,2112 finite logprob values;113.386s including
initial latency. Its different timing/path means no current NaN confirmation.
Historical June26-27 Qwen3.6/vLLM0.23 evidence (ca762fa) reported clean isolated
replay and homogeneous prefill, reproducible mixed-load degeneration, NaN
JSON errors with logprobs, and eventual persistent failure cleared by restart.
That older finding is a lead, not proof that R276 has the same defect.
VERDICT -> Continue full MTP3 qualification. Direct cold-prefill-over-active-
decode probes queued with and without logprobs; client overlap is measurable
but does not prove backend co-batching. Broader coding and lifecycle gates
remain required before promotion. Cache-salt recovery is mitigation, not a
root-cause fix; no new FP8 KV work or native/driver changes.

### 2026-09-08 - Bound the current overlap trigger without reviving old fixes

CONFIG -> Same R276 MTP3/32K prefix candidate. Installed _xpu_ops.py already
has VLLM_XPU_GDN_SPLIT_MIXED=1; server log confirms split-mixed execution.
This is not an absent-old-flag diagnosis. No runtime changes in these arms.
COMMAND -> probe_mixed_decode.py with and without --logprobs: two waves per
arm, each one2048-token anchor plus three cold32K recalls, prefills submitted
after an actual anchor text delta. Also32 shared tool checks with
--serial-requests and four logical histories; preserve SSE and raw requests.
RESULT -> Both direct mixed arms8/8 correct,0 bangs,6/6 client-observed
overlaps; logprob arm25116 finite values,0 nonfinite. The serialized four-
history control32/32 correct,0 bangs; timestamp audit confirms maximum one
observed active stream. Earlier concurrent counterpart was32/33 with one
bang. These are bounded controls, not representative long-run rate tests.
Current03-reuse completes7/7 correct; warm fraction0.9925556 and TTFT ratio
0.0173275. Existing prefix cache works while the growing-tool trigger remains.
VERDICT -> Generic cold-prefill/long-decode overlap alone is insufficient to
reproduce current bangs. Concurrency plus cached growing tool conversations
is the stronger reproducer. Next instrumentation should correlate actual
scheduler request/row order, cache hit length, accepted speculative tokens,
and recurrent state indices at the first bad step, on both ranks. Finite-
value observation must avoid changing the timing enough to hide the bug.
Do not claim NaNs, a particular kernel, or global persistent poisoning on
R276 from the older stack's evidence. Continue coding/churn/health and fresh
public MTP3 deployment under the user's under5 percent observed tolerance.

### 2026-09-08 - Promote user-accepted INT4 MTP3 prefix candidate

CONFIG -> R276 base image521eb277/source54aefaf0, AutoRound W4A16 g128,
MTP3, FP16 KV, GPU prefix on, CPU tier0,200K/c4,batch32768. User explicitly
accepts current INT4-vs-FP8 coding results and directs moving forward.
COMMAND -> Complete stock-mtp3-b32768-daily; analyze_prefix_campaign.py,
compare_code.py against both baselines, audit_bang_rate.py, paired profile,
normal teardown and per-card/two-rank post-health. Prepare qualification,
registry and systemd unit; start fresh leased public lifecycle20260908T220558Z.
RESULT -> Candidate WORKLOADS_PASSED and exit0; both GPU and collective
post-health pass. Pool451562 tokens. Four88K histories32/32 correct in217.876s,
all28 followups98.53-98.85 percent cached. Cancellation112.712s with exact
repeats and warm TTFT1.62-1.65s. xhigh24/24 and199K recall pass. Actual Pi
injected-failure recovery passes on MTP3/32K; no user client config changed.
Oversubscribed four132K histories:8/8 correct after9 attempts,1 bang recovered,
645.822s; all four final answers cached0, no active preemptions, capacity queue
observed. Its1/9 raw fraction exceeds5 percent in this small stress sample;
do not dilute it with easy requests. Fixed shared soak remains4/132=3.0303
percent. Known concurrency-sensitive tool-history bug is NOT fixed.
Coding164 raw outputs contain no bangs. Base/plus157/149 versus FP8 157/152,
previous INT4 158/150. Only additional failed task versus previous INT4 is132,
truncated at2048 tokens in comments. The raw outputs for33,91,97,125,151,154
are byte-identical to previous INT4. Original <=2-net-loss score gate stays
false; user's explicit acceptance is recorded separately, not a rescoring.
Extra task132/4096 diagnostic generated output but its one-task grading hit
missing-problems assertion; excluded and inconclusive, no additional GPU work.
Warm profile paired1666 completed CPU collective events each (833 c10d plus
833 wrappers), equal shapes/counts:525 all-reduces,308 all-gathers per rank,
1137-token cached prefill and1/4 decode rows; host event waits477/433. Captured
internals may be opaque; no one-fence/per-token claim.
VERDICT -> Promote the user-preferred guarded conversational recipe with
explicit quality/rate/oversubscription limitations. Public alias hotschmoe-dd
unchanged. Fresh startup/authentication, concurrent canaries,150K warm reuse
and strict restart parity queued under the serving lease. Startup pending;
boot unit prepared and statically verified, host systemd installation still
requires the documented user sudo command. Do not claim all clients have
bang-guard or that the server itself retries natural loops.

### 2026-09-08 - hotschmoe-dd restored on INT4 MTP3 with GPU prefix caching

CONFIG -> Fresh public lifecycle20260908T220558Z, same qualified R276 image,
AutoRound INT4/MTP3/FP16 KV/GPU prefix on/CPU tier0/200K/c4/batch32768.
Stable public alias hotschmoe-dd, authenticated port18080, backend18124.
COMMAND -> Leased launcher start; ready helper; smoke_daily_frontdoor.py;
smoke_prefix_frontdoor.py through the authenticated public API; author's
fixed12-prompt strict suite and compare-strict-attempt-outputs.py against the
qualified MTP3 candidate. No driver/image/native changes. Leave serve running.
RESULT -> All four leased validation jobs exit0. Missing/wrong keys return401;
authenticated identity and C4 arithmetic pass. Fresh pool451562 tokens.
150K public recall correct twice: cold TTFT94.10277s, warm1.622735s,
99.2542 percent cached. Fresh strict94.43162tok/s versus candidate94.82656;
all12 complete output token arrays exact. Candidate had clean teardown and
per-card/compiled two-rank post-health; fresh startup pre-health passes.
Public evidence: runtime root/public-validation.json. No STOP or exit marker
in the active lifecycle. Qualification and registry now reference live proof.
VERDICT -> Requested conversational prefix-serving outcome is live. Bang
bug remains unresolved: client recovery is validated; shared-soak4/132 and
oversubscribed1/9 raw incidents remain explicit, with no long-run below5
percent guarantee. User accepts coding157/149 versus FP8 157/152; primary
score and failed conservative score gate remain unchanged. CPU RAM offload
and FP8 KV remain disabled. This is not a no-loss or global-stability claim.

Boot configuration is prepared and statically verified; host systemd unit
installation was not performed. User command (does not interrupt this serve):

```bash
sudo bash /mnt/vm_8tb/github/b70_ai_things/vllm/fp8/install-hotschmoe-dd-systemd.sh
```

The installer backs up units, installs/enables the new launcher and disables
the retired boot recipe. Current manual serving continues; subsequent
service ownership changes require draining/stopping it and waiting for the
owned teardown, as the installer explains. User worktree changes preserved.

### 2026-09-08 - Manual serving lost on user-session shutdown; system start required

CONFIG -> Previously validated INT4/MTP3/FP16 KV/prefix recipe unchanged.
COMMAND -> Check public health, backend models/metrics, Docker, installed
system unit, user journal and previous lifecycle. Run bin/xe-reset through
its GPU lease; try systemctl --no-ask-password --no-block start hotschmoe-dd.
RESULT -> Ports18080/18124 refuse connections; no serving container. Grafana
and Prometheus containers remain up; Prometheus b70_daily reports connection
refused at18080/metrics. Manual launcher exit.rc=-1 denotes runner SIGHUP;
its server/exit.rc is absent. At22:23:55 UTC the user manager stopped the
tmux scope; the API subsequently performed SIGTERM shutdown. This supports
session-lifetime termination, not a demonstrated bang/GPU crash. The correct
MTP3 system unit is installed/enabled but inactive, with no service start log.
Recovery rebind completed, both card probes and compiled two-rank collective
health passed, exit0; no reboot. System service start was denied because
interactive authentication is required. No privilege bypass attempted.
Add SIGHUP to the runner's existing SIGTERM/SIGINT graceful-stop handler;
Python compilation passes. This repairs cleanup but is not a substitute for
system-service ownership. Historical performance/quality proof is retained;
deployment state now explicitly records the outage instead of claiming live.
VERDICT -> Endpoint is currently down. The earlier detached manual process
was still tied to the user-session scope; enabling the boot unit did not
transfer ownership of that process. User must start the installed service:

```bash
sudo systemctl start hotschmoe-dd.service
```

The service owns both GPU leases and repeats matched startup health before
serving. It is already enabled for boot. Verify API and Prometheus scrape
recovery after it becomes ready. Recovery evidence:
/mnt/vm_8tb/b70/results/int4_prefix_20260908/20260908-hangup-recovery.log.
