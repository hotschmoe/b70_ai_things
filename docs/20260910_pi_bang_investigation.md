# Pi bang investigation, 2026-09-10

Status: testing paused at user request. The authorized200K day trial is LIVE
as hotschmoe-dd on the existing authenticated endpoint. Systemd is active/running;
startup tiny24 and one authenticated streaming smoke pass. No shelf promotion,
completed200K qualification or production stability claim is made.

Two reproduced mechanisms are now separated: the GDN phase guard fixes a
fresh singleton initialization defect; a second accepted-count ordering patch
passes concurrent cache/MTP fixtures that still bang with the phase guard alone.
The second split occurs with FP16 and fresh calibrated FP8 KV at TP1. The FP8
card crossover repeats the split; repaired TP2 qualification is in progress.

## Answer to the GDN and TP questions

Steve's GDN phase finding explains a reproduced local bang mechanism. The
R276 builder treated a fresh one-token prefill as recurrent decode and could
read stale/uninitialized state. A one-call-site phase guard fixes that case.
It is not inherently TP2: stock fails on each physical card at TP1, while
phase-only candidates pass on both cards in a crossover. Native libraries
are unchanged. This does not establish that the original long Pi requests
hit the same internal phase; their exact upstream bodies are absent.

CPU execution of the actual scheduler further narrows that connection: a
cached or chunked one-token suffix normally has positive computed context
and existing recurrent state. In aligned mode, a fresh long prompt clipped
to a one-token first chunk is deferred at the reviewed block/budget settings.
Long history plus a singleton suffix therefore does not prove the reproduced
zero-state initialization defect occurred. Request-level trace correlation
or independent invalid-cache evidence is still needed.

CONFIG -> R276 stock versus phase-only, TP1 on both cards,8K/c4/batch512,
eager, MTP0, prefix off, FP16 KV. COMMAND ->24-case pinned tiny-prefill screen,
then swap the images between cards. RESULT -> Each stock run2 bangs / 24;
each patched run0 / 24, all 24 patched token arrays match between cards, and
all 22 nonfailed stock arrays match patched. VERDICT -> Causal bounded
reproduction, with normal teardown and the then-current selected-card health
checks. Later strict health is stronger than that old sentinel gate.

CONFIG -> Current official vLLM0.29 XPU,8K/c1/batch8192, other isolated
settings as above. COMMAND -> Stock and phase-only24-case screens on card 0.
RESULT -> Stock has 6 singleton loops, patched passes 24/24; patched arrays match both
R276 patched runs. Normal teardown and strict selected-card health pass.
VERDICT -> Updating alone does not fix this defect. This is a bounded
screen, not concurrent-batch or long-session qualification.

## Full-feature and harness evidence

R276 phase-only TP1 with MTP3, calibrated FP8 KV, prefix caching and decode
graphs completed the nine reconstructed Pi requests without bangs. Two
agent2 responses still fail broad quality checks; that supplied history
already contains corruption. Previously looping agent1 target requests now
finish without repetition flags. This temperature 0.7 replay is not a causal
proof or a semantic pass. Normal teardown and strict post-health passed.
The phase-patched TP2 full-feature run also passes all four stages: nine Pi
reconstructions, 24 tiny-prefill cases, three earlier clean-history requests,
and 32 concurrent tool checks. Tiny-prefill sequential and mixed token arrays
match exactly; server logs confirm four simultaneous requests. Teardown,
strict per-card health and compiled pair post-health pass. This is a bounded
100K-context qualification, not production promotion or a long soak.

The later matched FP16-KV arm passes nine Pi reconstructions, 24 tiny cases
and three clean-history requests, but fails one of 26 completed concurrent
tool checks. Session0's first answer emits 512 bangs from its first output
token instead of the known stock count730. Both its tool-call request and
answer request match the earlier passing FP8 arm after normalizing generated
tool-call IDs. The FP16 answer reports 832 cached tokens; the FP8 counterpart
reports zero. Teardown and strict per-card/compiled pair post-health pass.

This difference activates a cache path rather than isolating storage dtype:
resolved block sizes are FP16=832 and FP8=1600. For the 2360-token answer,
the full-attention hit finder can retain two 832-token blocks then drop one
for MTP, leaving832; one1600-token block is dropped to zero. Cache reuse is
therefore a concrete lead. The subsequent prefix-on/off and accepted-count
controls below isolate that path further.

The first TP1 pair reproduces a second full-feature bang with prefix caching
on (26/27 completed checks pass) while prefix off passes32/32. The failed
request is session2 turn1 tool selection, with2388 prompt tokens and832 cached;
it emits512 bangs immediately. The cache-off counterpart is identical after
tool-call ID normalization and returns the expected lookup_stock call. This
confirms TP2 is not required for the concurrent failure. The crossover repeats
the split: prefix on again fails1/27, prefix off again passes32/32. Cache-off
normalized request/choice arrays match exactly across cards. The cache-on
failure moves from session2 to session0, both at turn1 tool selection with832
cached tokens. All selected-card post-health checks pass. Requested teardown
can force-kill a remaining engine process; EngineDead appears only after
shutdown starts in the reviewed cache-off arm. This is not a quiet-clean
teardown or production stability claim.

MTP0 with caching on and an isolated MRV1 accepted-count repair with MTP3 each
pass32/32, with actual cache hits in every session. The repaired MTP3 arm
retains832-token hits and correctly answers both previously failing payloads;
MTP0 reuses1664 tokens initially. All32 normalized choices match between those
controls. Both lifecycles and strict selected-card post-health pass. The repair
moves the existing D2H wait before request-row movement and removes a redundant
second permutation. CPU source/AST and new-image per-card/compiled pair health
also pass. This supports the accepted-count explanation within the bounded
fixture; full production qualification remains outstanding.

Fresh calibrated FP8 controls are now using360 records (4274 prompt tokens),
so the1600-token cache layout can retain a prefix after the MTP margin. Their
review requires positive cache hits in every session. The earlier180-record
FP8 fixture did not exercise this reuse path despite caching being enabled.


CONFIG -> Phase-only versus phase plus MRV1 accepted-count fix, TP1/MTP3,
FULL decode graphs, prefix on, fresh calibrated FP8 KV, context/batch8192,
four concurrent sessions, records360. COMMAND ->32-check streamed tool fixture
and per-session positive-cache gate, first pair on cards0 and1 respectively.
RESULT -> Phase-only passes26/27 completed checks and fails session2 turn1
TOOL with512 bangs, prompt4368/cached1600. The fixed build passes32/32 and
returns lookup_stock(part-2-1) for the identical normalized failed payload,
with the same1600-token cache hit. Every session has positive reuse in BOTH
arms. Both lifecycles exit0 and strict selected-card post-health passes; both
require force-killing a remaining EngineCore after requested shutdown.
VERDICT -> The earlier no-hit FP8 pass was insufficient. This bounded result
supports accepted-count ordering as a second corruption mechanism with actual
FP8 reuse. The card crossover repeats the split: phase-only fails the same request,
while the repaired build again passes32/32. All27 phase-only normalized
request/choice arrays match across cards, including the failure; all32 repaired
arrays also match exactly. All four arms have positive1600-token reuse in
every session, lifecycle exit0 and strict selected-card post-health passes.
Raw: tp1-fp8-reuse-controls/{initial-pair,crossover}-outcome-review.json.

The first repaired TP2/100K replay arm has zero bangs in all9 reconstructed
requests, but fails its strict quality gate: both agent2 target continuations
produce malformed second tool arguments, including a gibberish suffix. These
inputs already contain corrupted history, but that does not prove the history
alone caused this output. The run stops before tiny/clean/concurrent stages;
it is not a passed qualification. A separate clean-workload plan retains the
failed diagnostics and tests tiny cases, earlier clean history and six
concurrent rounds. Original Pi semantic correctness remains unresolved.

An independent comparison finds the exact agent2 target request dictionaries
also produced valid tool JSON in four earlier configurations: phase/old-FP8,
stock/old-FP8 with MTP3, stock/old-FP8 with MTP0, and phase/FP16. The new
"b:,.: the/python:" suffix is absent from the request and present in raw SSE.
Thus history contamination is not an established explanation. Fresh scales,
runner repair, TP/configuration and runtime state still need matched controls;
a fresh-scale TP1 phase-versus-MRV1 pair is prepared. The known row-permutation
mechanism alone does not predict corruption for an unchanged serial row0.

Its startup trace records matching per-rank profile boundaries:138 all-reduces
on FP16[32768,5120] and3 all-gathers on FP16[32768,2560], with paired returns.
These are actual Python host boundaries, not independent device completion
or graph-replay operation counts. Strict per-card and compiled-pair post-health
pass. The scoped30second internal shutdown grace allows workers and engine
to exit without the earlier forced kill; this single TP2 lifecycle does not
establish the same result for TP1 or all shutdown paths.

The selected affected bundle contains 41 bangs among 317 non-429 messages; this
is not an unbiased production rate. All 36 recorded bang-recovery messages
immediately hit the harness's per-agent429 admission guard, 0-3 ms after the
aborted message was persisted. That is a separate recovery/admission race.
Gateway source and exact failed wire bodies are still needed to fix and
correlate that layer. No external tool calls from replayed histories execute.

## SGLang is viable, but not yet an equivalent replacement

The latest published Intel baseline, with a deliberate source-only XPU
pointer/device port, passes 26/26 bounded text/tiny/mixed requests. Explicit
Mamba extra_buffer with radix caching passes 32/32 tool checks across four
growing histories, with four requests running and measured cache reuse.
Both lifecycles exit0 and strict selected-card post-health passes.
These results use TP1, FP16 KV, eager mode and MTP0.

Current pinned SGLang source has been rebuilt from source with GPTQ and GDN,
explicit BMG, and Triton attention. Unused native FMHA/MoE/MLA kernels are
excluded; this is not a native vision-FMHA recipe. No quarantined binary is
copied into the stack. The CPU installed/API gate passes: all 63 installed
native artifacts match the rebuilt wheels and the held PyTorch/UMD/oneCCL
identities are unchanged. The current-source image passes strict per-card and
compiled pair health. Its first serve failed when automatic image warmup
called the deliberately excluded native vision FMHA. A text-only retry using
the supported skip-server-warmup flag passes all 26 real text checks,
including four-request mixed batches, with normal teardown and strict card1
post-health. This is TP1/FP16 KV/eager/MTP0/prefix-off only. Image requests are
not supported by this reduced recipe's default vision backend.

Two material parity gaps need qualification: native Intel attention prefill
omits FP8 descales, and Qwen lacks the required calibrated-scale loader. A
strict source-only loader and mapping are prepared for Triton. In addition,
current XPU MTP verification selects greedy tokens even for temperature 0.7.
A bounded unseeded sampling candidate passes CPU checks but is not activated.
Seeded sampling needs a shared ordinary/speculative mapping; the optional
FP32 prototype changes the legacy seed mapping and is not a drop-in parity
fix. Graph/MTP/FP8/TP2 parity
has not been demonstrated by the simpler passing screens.

## Calibration identity finding

All eight current safetensors match the publisher's pinned revision. Seventeen
model files were rehashed; only shard 2 differs from the old calibration
provenance. Ordinary and no-fallback direct disk reads both match the publisher.
The old freeze utility copied an existing manifest without rehashing weights.
This does not establish why its recorded shard 2 hash differs, or which bytes
the old calibration actually loaded. The old artifact is preserved unchanged.

The new loader refuses the mismatch. Fresh calibration completed against a
verified complete manifest: 256 short cases, four long prompts through 96K
tokens and two 4096-token continuations. All 17 model-file hashes and stat
identities match before and after collection. Teardown, strict per-card and
compiled pair post-health pass. The fresh 17-layer artifact is frozen with
SHA256 be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062.
The recorder requires eager/prefixoff; graph and prefix-on FP16-versus-FP8
qualification must be separate. Current collection is limited to a 100K
configuration and does not qualify the previous 200K configuration.

The 256 shared short cases have 247 exact text-plus-reasoning matches with the
old collection; nine differences are ordinary wording/formatting changes.
Twenty-five of 34 K/V scales are exactly equal; remaining fresh/old ratios
range from 0.95085 to 1.01898. Calibration configuration and long prompts
differ, so these comparisons do not resolve the historical weight identity.

CONFIG -> Phase-only vLLM R276, synthetic distinct K/V scales, 67 tokens,
permuted 64-token cache blocks, each physical card. COMMAND -> Actual cache
write and attention-read oracle, including exact representable bytes and
untouched slots. RESULT -> Both cards pass; attention relative L2 is below
0.000227 against the dequantized CPU reference. Strict per-card and compiled
pair pre/post-health pass, with owned containers removed. VERDICT -> This
bounded numeric test validates scale consumption, not full-model FP8 quality.
The corresponding SGLang oracle initially fails its exact-byte comparison on
card0. A diagnostic isolates all differences to XPU's scalar divisor being
rounded to FP16 before division: 156 K bytes differ from the original CPU
reference; identical-input casts and scatter agree exactly. Independent
power-of-two stores pass. A revised independent reference preserves this
distinction, matches all bytes, then reaches a failing attention comparison
(48/6144 elements outside the unchanged tolerance). Grouped decode source
also quantizes Q and softmax P to FP8, beyond cache quantization. That compute
policy was isolated with an independent CPU prediction and a source-only
read patch that retains FP8 cache storage while using FP16 attention operands.
The paired actual-Triton oracle now retains all9 checks: upstream fails9/9,
patched passes9/9 with zero out-of-tolerance elements. Worst maximum absolute
error is0.137542 upstream versus0.000921488 patched. Both exact byte/untouched
slot checks pass. The patched image passes strict per-card/compiled pair
preflight; both oracles pass selected-card post-health and verified cleanup.
This is a numeric fixture, not full-model FP8/graph/MTP/TP2 qualification.
Different cards were used without a numeric crossover; the result agrees
with the prior CPU policy prediction. A full-model FP8 cache test is prepared.

## Evidence and reproducible sources

- ../vllm/int4/20260910_bang_isolation_campaign.md
- ../vllm/int4/20260910_runtime_freshness.md
- ../vllm/int4/20260910_cache_correctness_review.md
- ../vllm/int4/diagnostics/README.md
- ../sglang/cache800k/20260910_feature_parity_review.md
- ../sglang/cache800k/calibrated_kv/README.md

Raw immutable plans, requests, responses, logs, identities, comparisons and
health results are under /mnt/vm_8tb/b70/results/bang_isolation_20260910.
The extracted redacted bundle is under results/bang_investigation_20260910.
Host kernel/drivers are unchanged. Experiments use per-card or paired gpu-run
leases with matching pins and P2P0. Failed lifecycles retain their evidence.

## User pause and next downtime

For the standalone ordered backlog, exact launch commands and prerequisites,
read [the campaign resume handoff](20260910_bang_campaign_resume.md).

The user requested pausing tests and serving the patched vLLM endpoint for the
day, explicitly retaining200K context. The clean100K arm had passed tiny24,
earlier clean history3, three deterministic32-check rounds, and one sampled
32-check round (128 concurrent checks total with strict semantics and positive
reuse). The three deterministic response-choice arrays match exactly after
normalizing only generated tool-call IDs. The arm was stopped at user request;
its in-flight sampled round was allowed to drain. Do not mark the whole arm
qualified or substitute it for the unrun200K long-context plan.

The day trial is a separate user-authorized scope, retaining MTP3, full decode
graphs, prefix cache, fresh calibrated FP8 KV and P2P0. The fresh calibration
still covers <=96K;200K is an explicitly accepted trial limit. The regular full
qualification gate remains separate and unchanged. Preserve authenticated
frontdoor18080, loopback backend18124 and primary model name hotschmoe-dd.

Prepared work for the next downtime, not launched:

- pi-agent2-freshscale-paired/{phase,phase-mrv1fix}/plan.json: matched diagnostic
  for the new garbled tool suffix, with identical fresh scales and requests.
- mrv1-tp2-fp8-long200k/plan.json:185K concurrent retrieval, reuse, generation,
  cancellation and recovery. Its full clean100K prerequisite remains unmet.
- sglang14ee-fp8-cache-plan/plan.json: full-model FP8 cache qualification after
  the passing9-case numeric patch; graph/MTP/sampling parity remains incomplete.
- vllm-fp8-geometry-plans/{q1,q4}/plan.json: actual23240-token/1600-block oracle.
- mamba-copy-lifecycle-plans/card{0,1}/plan.json: actual padded state-copy checks.
- Optional non-bang extreme-repetition guard artifact remains uninstalled;
  it does not classify arbitrary gibberish or fix the separate gateway429 race.

The original Pi wire payloads and LAN-installed guard/gateway source identity
remain unavailable. All replay conclusions must retain that limitation.

## Live day-trial checkpoint, 07:32 UTC

CONFIG -> User-authorized200K day trial, image7b107d0e phase+MRV1 fix, fresh
be02 FP8 scales, TP2/P2P0, MTP3, prefix on and FULL_DECODE_ONLY graphs. The
first served model name is hotschmoe-dd; the detailed secondary alias is
qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276phase-mrv1fix-tp2-mtp3-fp8kv-freshcal-daytrial-ctx200k.

COMMAND -> User installed the reviewed root-owned day-trial systemd override,
reloaded systemd and started hotschmoe-dd.service. The wrapper owns both GPU
leases, runs strict per-card/compiled-pair pre-health, verifies34fresh scale
receipts and runs only tiny24 before opening the authenticated frontdoor.
One public authenticated streaming arithmetic request verified client access.

RESULT -> Service active/running, stable primary name first in /v1/models,
max_model_len200000, unauthenticated /v1/models returns401, authenticated stream
returns42 with finish_reason stop. Tiny24 passes with no sequential drift;
strict pre-health passes. Public port18080 and loopback backend18124 and the
existing credential file remain unchanged. No full qualification marker was
created. Runtime root:
/mnt/vm_8tb/b70/results/qwen38_mrv1fix_day_trial/20260910T072531Z.
Its public-client-smoke.json and trial.json retain the exact outcome/scope.

VERDICT -> Serving for the day as explicitly requested. Research is paused;
remaining tests and the unresolved diagnostic Pi tool suffix resume at next
downtime. Do not stop the live service or launch competing GPU work without
new downtime direction. The prior test arm ended with lifecycle0 and strict
per-card/compiled-pair post-health; it remains an interrupted qualification.
