# Bang isolation campaign, 2026-09-10

Goal: identify and repair corruption while retaining prefix caching, graphs,
MTP and calibrated FP8 KV. User authorizes endpoint downtime, custom kernels,
custom ops, transformations and alternative vLLM images. TP1 versus TP2 is
now the priority contrast based on the user's unconfirmed TP1 observation.
SGLang with the same features is a parallel research direction, not an
assumed fix. No model output is allowed to execute tools in these probes.

Raw root: /mnt/vm_8tb/b70/results/bang_isolation_20260910/.
Earlier incident analysis: 20260910_live_bang_investigation.md.

## Baseline and shutdown

CONFIG -> R276 INT4, MTP3, calibrated FP8 E4M3 KV, prefix on, full decode
graphs, TP2, 200K/c4. Exact image521eb277..., lifecycle20260909T224703Z.
COMMAND -> Snapshot container, unit, host and endpoint identity/metrics.
SIGTERM verified owning trial launcher10892; leave the parent lease held
through backend teardown and post-health. No system configuration change.
RESULT -> Frontdoor and backend stopped. Lifecycle exit0; both per-card
checks and compiled P2P-off collective post-health pass. Unit Restart=no.
VERDICT -> Endpoint offline for authorized investigation. Original startup
recipe remains recoverable; no replacement promoted.

## SGLang C1: topology recognition alone

CONFIG -> Same adc915d266 image, synthetic2048x5120 FP16 subgroup reductions,
P2P0; change CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK from0 to1 versus C0.
COMMAND -> Existing collective_control.py under collective_lifecycle.py and
run_arm.py; named unit b70-bang-sglang-c1,240-second bound, both GPU leases.
RESULT -> Small4x5120 reduction passes both ranks. Rank0 enters first20MiB
reduction without returning. Rank1 completes that reduction numerically,
then enters the next and stalls. Timeout124. Container removal verified,
both devices rebound, per-card and compiled P2P-off post-health pass.
VERDICT -> Topology recognition alone is insufficient. Same-root-cause as
loaded SGLang is not proved. No full-model unchanged retry. Next candidate
for this collective branch is CCL_ENABLE_SYCL_KERNELS=0, holding C1's other
environment fixed; do not introduce it into the vLLM TP comparison silently.

## CPU/source preparation

CONFIG -> Three affected Pi histories, first corrupted response excluded.
No exact Pi system prompt/tool schemas/upstream payload in original bundle.
COMMAND -> replay_pi_history.py prepare; preserve payload hashes. Validate
recorded tool-result pairings, first-corruption cutoff, prior-thinking
omission and absence of tool execution. Seven lifecycle CPU tests include
single-card lease/device pin, health scope and deferred pair-wide recovery.
RESULT -> All six reconstructed histories have no orphan or pending tool
calls. They use a clearly reconstructed system prompt/minimal tool schemas,
manifest seeds,temperature0.7,medium thinking,max output8192. Successful prior
transcript/tool results are used as context without executing tools.
VERDICT -> Controlled reconstruction, not exact historical wire replay.
Requested actual upstream payload and gateway source from the other team.
Comparison retains actual source/payload hashes and SSE/partial failures.

## TP1 initial capacity control

CONFIG -> Exact same image, weights and scale artifact, MTP3/prefix/graphs/
FP8 KV on,200K/c4/batch32768. Single-card lease0 and matching device pin.
COMMAND -> vllm-tp1-fp8kv-mtp3-prefix-graph plan, fresh compiler cache.
RESULT -> Weight load17.48GiB. Startup rejects200K: needs7.26GiB KV but
4.25GiB available, estimated maximum105600 tokens. No inference occurred.
Per-card post-health passes. Pair-wide recovery was correctly deferred
until the single-card lease ended; separate two-card rebind and per-card/
compiled collective health passed before retry.
VERDICT -> Capacity rejection, not evidence about bangs. New matched TP1
and TP2 plans use100K context, keeping other features and batch budget.
This covers the selected first-failure histories.200K remains a later
qualification gate; a lower prefill batch budget is a possible memory
tradeoff to measure, not a demonstrated repair.

## Reviewed source candidates

Fetched source refs without moving either checkout:
- Steve: steveseguin/b70-optimization-lab origin/main ac2f723e.
- Sergio: SergiioB/intel-arc-pro-b70-inference-cookbook origin/master966c593.

Steve's134322d9 validates a GDN short-prefill call-site correction on his
stack: treat short target prefills as prefills when is_prefilling metadata
exists, preserving legacy behavior for metadata-less capture/draft callers.
Our exact installed R276 call also omits this flag. His actual-source CPU
regression passes8 stock cases and8 in-memory candidate cases in our exact
image, without GPU devices/network. This confirms the classification defect,
not its causal role in Pi. A one-call-site Python-only candidate is prepared
under gdn-phase-candidate-v2; original preparation's ASCII encoding error
is retained under gdn-phase-candidate and made no runtime change. Native
libraries must remain byte-identical before any candidate GPU test.

Sergio'sfb860bb ports three separate hybrid-cache defects: accepted-count
D2H/double permutation, ignored eagle boundary, and backward state copies.
Our installed V1 runner retains the late wait plus prev_positions gather;
input-batch swaps/condense mutate those same counters. Installed Mamba copy
sites also lack a backward-copy guard. These are source leads requiring
actual-path and causal tests. Do not apply all changes at once and claim
which one fixed the symptom.

Sergio'sd9c95e5 disables his draft INT4 patch for TP>1. Our R276 path is a
different implementation: a shallow draft-only ParallelLMHead copy with
private packed buffers and unchanged verifier head. TP>1 drift is still a
reason to test draft-head INT4 off, but his guard is not proof of the same
bug in this implementation.

No kernel, driver, PyTorch or native library has been updated. SGLang source
inspection still shows incomplete FP8 extend support in its intel_xpu
attention backend; Triton is a candidate route, not feature qualification.

CONFIG -> Python-only GDN phase candidate built atop exact R276 base.
COMMAND -> Build one COPY layer; compare five native/routing file hashes;
run eight actual-source cases in a CPU-only network-disabled2CPU/2GiB
container. Verify tracked patches/r276-gdn-prefill-phase.patch reproduces
exact candidate source bytes.
RESULT -> Image0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077;
source a646a6750f16b961d0fa8eb6d7e00b46b4ea4e6992b1cf620cb5eb01b02d6b38.
Native hashes equal, eight CPU cases pass. A separate controlled evaluation
of the actual installed accepted-count gather reproduces double permutation:
already-reordered counts[4,1] become[1,4]; proposed direct copy keeps[4,1].
VERDICT -> Candidate prepared, no GPU repair claim. Counter test is not an
actual asynchronous D2H race reproduction. The phase patch is a deliberate
port of Steve's134322d9 finding; no upstream checkout was moved.

## TP1/100K replay: no bangs is not coherence

CONFIG -> Matched100K TP1 control described above, same nine reconstructed
requests at temperature0.7, MTP3/FP8/prefix/graphs on, P2P0.
COMMAND -> Leased replay_pi_history.py run; normal STOP; offline
 audit_replay_quality.py against preserved responses and raw SSE.
RESULT -> Nine completed HTTP requests, zero32-bang trips; lifecycle exit0,
card0 post-health passed. Three visibly degenerate responses: agent2 warm
has repeated tool XML; agent1 target0 repeats literal escaped-newline/dash
404 times and has invalid JSON arguments; target1 repeats escaped-newline/
semicolon225 times consecutively, consumes8192 output tokens, and has a
95.166-second SSE gap before termination. No generated tool was executed.
VERDICT -> Coherence failure on TP1. The raw OUTCOME/WORKLOADS_PASSED markers
only reflect the original bang/transport gate and remain preserved; they
are explicitly superseded for quality assessment by replay-quality-review-v2.
The review is heuristic, not proof that unflagged responses are correct.
This does not prove that the original bang mode itself is TP-independent.
Matched TP2 is running; MTP0 is prepared as the next single-feature ablation.

CONFIG -> Offline audit of input histories before attributing output failure.
COMMAND -> Scan all assistant inputs for repetition; manually inspect matches.
RESULT -> Agent2 already contains1509 zeroes at message18 and long corrupted
\nIn/\n2 tool arguments at55, before its first recorded bang. Agent1's
matched input repetitions are ordinary single-line comment separators;
agent3 has no matches. Prepare a separate agent2 history truncated BEFORE
message18 under early-clean-history-payloads, with original source hash.
VERDICT -> Agent2's new XML failure is contaminated-context evidence, not a
clean-input runtime reproduction. Agent1's two new repeated-line failures
remain useful. Original nine-request TP pair remains unchanged. A separate
early-history case will check degeneration without known contaminated input.

## TP2/100K matched replay

CONFIG -> Same100K feature/sampling/payload/scale/image control as TP1,
change topology to TP2, retaining P2P0 and both leases.
COMMAND -> Nine reconstructed requests, same order; normal STOP and both
per-card plus compiled collective post-health; offline quality review.
RESULT -> Zero bangs/HTTP errors;9 completed. Agent1 target repeats each
use178 tokens and produce structurally valid short tool calls, without
TP1's new loops. Agent2 warm reasoning ends in malformed prose after1093
tokens, but this input already contains known corruption. Heuristic review
flags0 requests and misses that semantic failure, confirming its limitation.
Both per-card and compiled P2P0 post-health pass; lifecycle exit0.
VERDICT -> This bounded pair does not support TP2-alone causality. It also
does not establish stability or prove P2P0 repairs the original endpoint:
production used P2P1,200K, a different pool capacity and actual Pi payloads.
No unsafe P2P1 serve attempted. TP2/MTP0 with other features retained is
running, followed by the isolated GDN candidate if health permits.

## MTP0 local one-token bang reproduction

CONFIG -> R276 TP2/P2P0,100K context, calibrated FP8 KV/prefix/graphs on,
MTP disabled. Same nine Pi reconstruction requests, then pinned Steve
134322d9 tiny-prefill24-case source screen, temperature0/seed42/output64.
COMMAND -> Leased run; retain all completion text/token IDs. The screen
uses one-token[17],two-token[17,18],arithmetic and operations prompts:
two serial rounds per prompt plus four c4 mixed rounds. No tools executed.
RESULT -> Pi replay9 completes without bangs or heuristic flags. Tiny
screen FAILS: five of six one-token prompts return exactly64 exclamation
marks and token ID0 throughout; the first one-token response is also
visibly garbled. Other18 prompts avoid the extreme-repetition gate.
Early-history job is NOT RUN because the tiny gate fails. Normal teardown
completes; per-card and compiled P2P0 collective post-health pass; lifecycle0.
VERDICT -> Local bang reproduction without MTP, matching Steve's short-
prefill symptom. This does not establish that long Pi histories reached
the same trigger. The probe's false sequential_repeat_drift field excludes
failed response pairs and is NOT an exact-output match assertion. A GDN-
only patch comparison is now required. Independent TP1 stock/card0 and
phase-patched/card1 tiny screens are launched with eager/MTP0/prefixoff/
FP16KV/8K/c4/batch512; cross-card findings require same-card crossover.

CPU-only image discovery/extraction overlapped part of this arm at the
user's request. Docker operations were delayed briefly during simultaneous
layer extraction; both pulls completed before post-health. No speed claim
or matched host-load claim is made. No host driver/kernel changes occurred.

## GDN phase-only TP1 crossover confirms the local mechanism

CONFIG -> Exact R276 versus one-call-site phase candidate0328900c, native
libraries unchanged. TP1,8K/c4/batch512,FP16 KV,eager,MTP0,prefixoff,P2P0.
Run stock/card0 and patched/card1 concurrently, then swap images between
cards after both first-pair lifecycles and per-card post-health complete.
COMMAND -> Same pinned24-case tiny-prefill screen; full text/token arrays;
independent card leases/device pins and private localhost ports.
RESULT -> Each stock run has2/24 bang failures (one-token cases), on each
physical card. Each patched run passes24/24 with zero bangs. All24 patched
complete token arrays match between cards. On each card, all22 nonfailed
stock arrays match the corresponding patched arrays. All four lifecycles
exit0 and their selected-card post-health completes. Raw comparison:
gdn-tiny-crossover-comparison.json.
VERDICT -> The phase-only change fixes this bounded local bang mechanism
on both cards. TP2,FP8 KV,MTP,prefix caching and graphs are NOT prerequisites.
This is not a full semantic suite, long soak, or proof of the original Pi
trigger. The corresponding source fault also exists in current official
vLLM0.29; its separately pinned candidate is CPU-checked, not GPU-tested yet.

Host-only request/phase tracing is prepared separately under diagnostics/.
It links effective one-token target prefills to request IDs without prompt
logging or additional GPU synchronization. No trace image is live yet.

## Current official vLLM and strict experimental health

CONFIG -> Official0.29 XPU stock1db27a8b versus phase-only29f480e9,
TP1/card0,8K/c1/batch8192,eager,MTP0,prefixoff,FP16KV.
COMMAND -> Same pinned24-case tiny-prefill screen, selected-card lease,
strict experimental pre/post-health with finite outputs and exact sentinel.
RESULT -> Stock fails all six one-token cases: one repeats token17 and
five repeat token1023,64 times each. Candidate passes24/24; every complete
candidate token array matches both R276 patched card runs. Both lifecycles
exit0 and strict selected-card post-health passes.
VERDICT -> Updating to current official0.29 alone does not remove this
mechanism; the isolated phase guard does. Max-num-seqs1 does not qualify
concurrent batch execution. Full-feature and original-Pi causality remain
open. Detailed immutable identities are in20260910_runtime_freshness.md.

CONFIG -> Experimental strict health candidate, shared bin unchanged.
COMMAND -> diagnostics/check_xpu_health_strict.py; lifecycle preflight tests.
RESULT ->12 CPU finite/sentinel/cleanup checks and7 lifecycle tests pass.
Both new images additionally passed per-card and compiled two-rank P2P0
preflight under both leases. Earlier reported shared-health passes used
its old permissive sentinel; they are not equivalent to this stricter gate.
VERDICT -> Use recorded strict probe override for new experiments; no
unqualified change to shared shelf lifecycle or promotion claim.

CONFIG -> R276 phase-only TP1 all-feature100K control, card0, in parallel
with SGLang FP16-KV/eager/MTP0 viability on card1; CPU current-main build.
COMMAND -> Frozen vllm-tp1-fp8kv-mtp3-prefix-graph-ctx100k-gdnphase plan.
RESULT -> Launched04:16UTC; results pending. SGLang retry preserves compiled
cache bytes with corrected experiment-cache ownership and300s HTTP bound.
Unsupported optional language-model-only flag rejected Qwen during retry-v2
argument resolution, before serving; strict post-health passed. Retry-v3
had an invalid health path before GPU touch; corrected retry-v4 is running.
VERDICT -> These are unqualified candidates; production remains offline.

## Phase-patched TP1 with full serving features

CONFIG -> R276 phase-only0328900c, TP1/card0,P2P0,100K/c4/batch32768,
MTP3, calibrated FP8 KV, prefix cache and FULL_DECODE_ONLY graphs enabled.
COMMAND -> Frozen vllm-tp1-fp8kv-mtp3-prefix-graph-ctx100k-gdnphase plan,
same nine reconstructed Pi requests; broad replay-quality audit afterward.
RESULT -> Identity matches stable and detailed registry aliases. All17 KV
scale loads match d53fb656 artifact;103846 KV tokens, graph capture complete.
Nine requests finish without bangs; two agent2 responses are flagged: warm
repeats the two-character unit backtick+B584times and target0 has invalid
JSON tool arguments. This input
already contains pre-bang corruption. Agent1 target outputs417/407tokens
avoid the earlier stock TP1 loop flags; largest observed SSE gap0.215s.
Normal teardown and strict selected-card post-health pass, lifecycle0.
VERDICT -> Phase patch works through this full-feature TP1 workload but
quality is not fully clean. Nine temperature0.7 reconstructions do not prove
causality or stability; exact upstream Pi bodies remain unavailable. Do not
attribute all improvement to one patch without tracing/matched repetition.
TP2 phase comparison and clean concurrent qualification remain pending.

## Phase-patched TP2 full-feature bounded qualification

CONFIG -> Same R276 phase-only image, TP2/P2P0,100K/c4/batch32768,
MTP3, FP8 KV with preserved scales, prefix cache and FULL_DECODE_ONLY.
COMMAND -> Frozen TP2 phase plan:9 Pi reconstructions,24 tiny-prefill
requests,3 early clean agent2 requests,32 concurrent tool-history checks.
RESULT -> All four workload stages pass. Pi9 and early3 have no bang or
broad repetition/JSON flags; this is not a full semantic certificate.
Tiny24 passes; all4 sequential repeat pairs and16 mixed-versus-sequential
comparisons are exact. Concurrent tools32/32,32attempts,0bangs/retries;
server logs confirm up to4running requests. KV capacity773076 exactly
matches the stock100K TP2 control; graph capture completes. Normal teardown,
strict per-card post-health and compiled P2P0 pair post-health pass; lifecycle0.
VERDICT -> The phase patch operates through this bounded full-feature TP2
suite. No production/200K/long-soak promotion follows from it. Exact original
Pi trigger remains uncorrelated. A separate historical calibration manifest
has one incorrect/unexplained shard hash; current cached/direct reads match
the pinned publisher, but the old calibration identity is not established.
Fresh numerical oracle/calibration work follows without rewriting that artifact.

## Fresh numerical FP8 cache oracle

CONFIG -> Phase0328900c, both visible cards under both leases, length67,
block64, permuted block table[3,2], hybrid strided cache, query lengths1/4.
COMMAND -> Fresh self-leased scale-oracle runner with tracked --exact-write;
8GiB host container bound; strict per-card and compiled P2P0 pre/post-health.
RESULT -> Both query shapes pass on both cards. Representable K/V bytes
match exactly with distinct scales; untouched slots remain exact. Actual
attention consumes independent K/V read scales; relative L2 versus dequantized
reference is0.0002264(query1) and0.0002230(query4). Random cache-write error
is about0.0265-0.0267, within the existing oracle bound. Owned containers
removed, strict/pair post-health passes, lifecycle0. Raw fresh-scale-oracle/.
VERDICT -> Bounded numerical write/read path passes; this does not validate
historical calibration identity, all model activations, or SGLang Triton.
Fresh262-case eager/prefixoff calibration is now running with verified hashes;
subsequent serving qualification restores graphs and prefix caching.

## Completed fresh calibration with current model provenance

CONFIG -> Phase-fixed R276 image
`sha256:0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077`,
TP2/P2P0, MTP3, FP16 KV, eager, prefix cache off, 100K context. Bind the
current publisher-matching model files rather than reuse the old preservation
manifest with its unexplained shard2 hash.

COMMAND -> Run the 262-case collection, complete owned shutdown and strict
post-health, then `fresh-calibration-plan/verify_and_freeze.py freeze` and the
independent `sglang/cache800k/loader_overlay/review_fresh.py` acceptance gate.
Raw root: `/mnt/vm_8tb/b70/results/bang_isolation_20260910/fresh-calibration-plan/`.

RESULT -> All 262 requests completed: 256 core cases, four long contexts
(32K/64K/80K/96K), and two 4096-token continuations. Exact corpus/response
IDs, prompt and text hashes, usage, finish reasons and non-cancellation gates
pass. Both continuations reached 4096 tokens. All 17 model files match their
reviewed SHA256 values and retain identical pre/post device/inode/size/mtime/
ctime identity. Complete finite positive target/MTP observations cover all
17 attention layers on both ranks. Normal teardown, strict per-card and
compiled P2P0 pair post-health pass; lifecycle exit is 0.

The NEW frozen artifact is `fresh-scales.json`, SHA256
`be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062`.
Of its 34 merged K/V scalar values, 25 are exactly equal to the old artifact;
fresh/old ratios range from 0.9508474576271186 to 1.018976897689769. These
ratios describe scalar differences, not model-quality or runtime stability.

VERDICT -> Fresh calibration provenance and bounded collection gates pass.
Use this new artifact for subsequent numerical/serving qualification. The old
artifact remains unchanged; numerical similarity does not authenticate the
old calibration's model bytes or resolve the historical shard2 discrepancy.
Periodic activation snapshots still do not provide exhaustive final-step
coverage. No calibrated FP8 serving promotion follows from collection alone.
