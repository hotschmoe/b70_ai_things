# A1 - independent resumed-campaign evidence audit

Owner: Codex A1 / gpt-6.1-sol, 2026-10-21. Initial CPU-only audit through
16:49 UTC; completed r2 closure addendum below supersedes pending status.
No GPU access, build, lifecycle action, shared source edit,
commit, push or submission. Raw paths below are relative to
`/mnt/vm_8tb/b70/build/vllm-flashnext/`.

CONFIG: root's resumed Flash-Next campaign; O4+F1 dense-FA sidecar;
Strata remains the qualified daily driver. Read AGENTS, HANDOFF, BRIEF,
3R1/3D1/3N1/3Q1/3P1 and resumed JOURNAL milestones. D1 plan16-r2 is
active during this audit; its partial results are not qualification.

COMMAND: read completed raw JSON/logs and harness sources; independently
recompute canonical rates from SSE event timestamps and usage, repeat
token/text hashes, N1 case counts, q2 stage statistics, and identity
manifest deltas. Hash the small N1/P1 artifacts. No benchmark rerun.

RESULT: completed evidence supports diagnostic control rates and bounded
component gates. Strict serving repeat fails. One concrete prefill timing
label limitation was found and sent immediately to root; details below.
No promotion, stability, target attainment or submission is established.

VERDICT: continue coordinated correctness/decode work. Preserve the
invalid r1 arm and pending r2 closure distinctly from measured control.

## Initial recovery and matched control

- `restart-20261021/health-before.log`: leased native probes pass on
  both cards. `collective-before.log`: compiled two-rank 4x5120 BF16,
  ten iterations, P2P=0, COLLECTIVE_HEALTH_OK. This verifies the resumed
  host at that point; no stack layer was updated.
- `f1/merged-restart-r1/{models,bench,probe,gate}.json`: hotschmoe-dd
  first; research alias qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4;
  alias matches evals/configs/models.yaml. Arithmetic 323/667, Paris
  coherence and 8K/32K needles pass. C1 64-token ITL p50 is
  15.166917 ms; C2 19.559572/19.610850 ms per stream. Exact-repeat
  fails. All 16 stored repeat token/text hashes recompute exactly.
- Restart window records SERVE_RC0, HEALTH_AFTER_RC0, WINDOW_DONE;
  both-card native post-health passes. The earlier shell success despite
  GATE_RC3 is a recovered harness defect already corrected by root;
  the successful teardown does not make the strict gate pass.
- Matched control uses maxlen131072, chunk12288, maxseq4, BF16 dense KV,
  TP1 single-process EP sidecar, GGUF IQ3_XXS experts, N1=0, D1plan=0,
  prefix cache off. Actual serve argv and loader log identify both GGUF
  shards and the expected 1224 tensors; stable alias and registry agree.
  The effective EP override is f1c, not the earlier o4c entry in the
  concatenated recipe list. The load log confirms f1c.
- `f1/merged-d1-control-r1/bench.json`: C1 p50 15.168827 ms;
  C2 19.582852/19.604523 ms. Probe cold/warm 8K rates are
  1947.930587/2378.689205 tok/s. Both are observed content arrivals.
- Matched control arithmetic, Paris coherence and 8K/32K needles pass;
  promotion exact-repeat fails. Independent comparison of all 16 O5
  128-token stream hashes passes the stored hash integrity check;
  same-load C1 repeat streams are [false,false,true,true], C2
  [false,false,true,false]. These are actual same-load failures, not
  merely C1-versus-C2 differences.
- Control status is RC1, SERVE_RC0, both-card post-health HEALTHY,
  SO checks pass. Stored 408-file verification has bad=[]. Fork HEAD is
  0a6170da8ca9991972322fa1ca1d27674f0e75cd plus preserved dirty delta.

## Canonical 512-token metric and no-spec proof

For every control warmup and measured request, independently select the
nonempty content/reasoning SSE events and compute
`(usage.completion_tokens - 1) * 1e9 / (last_offset_ns - first_offset_ns)`.
All eight requests have 512 content events, authoritative completion
usage512, matching reconstructed output, +1 successful request,
+512 generation tokens, matching prompt-token deltas and zero prefix
hits. Three measured requests remain after excluding each warmup.

| Control suite | Recomputed median tok/s | Reciprocal ms/token |
| --- | ---: | ---: |
| reasoning-v1 | 61.64281699462504 | 16.22249028767123 |
| code-v1 | 61.80083059675335 | 16.18101229941292 |

The raw server-config has speculative_config=null, corroborated by the
engine initialization log and argv without speculative-config. d1_api
installs x5_server_info, which serializes the live app-state VllmConfig;
it does not construct a claimed config from recipe text. No draft metric
series is present in these samples, so their absence is not an observed
zero-draft counter measurement. Configuration/runtime initialization are
the positive no-spec evidence. B70_MTP_RT remains set for PLE data and
alone does not enable speculation.

Local validation says valid=true, submitted=false, ranking_eligible=false,
diagnostic_only=true. This audit confirms local metric arithmetic and
isolation; it provides no external leaderboard validation. The dirty
engine and strict-repeat failures remain visible and disqualify promotion.

## Concrete timing clarification and invalid arm

Restart-r1's 12K row has prompt_tokens12283, completion_tokens1,
n_chunks0, first_token empty, and ttft_ms exactly equal to
wall_ms5047.607103. f1_probe.py explicitly substitutes request wall time
when no nonempty piece arrives. Its 2433.430287 tok/s arithmetic is
correct, but this is a request-completion proxy, not measured first-content
TTFT. No few-ms error bound is established by the saved row. Root was
notified for a curated footnote; raw evidence stays unchanged.

All two D1 control, two discarded plan16-r1 and two active plan16-r2
8K prefill rows have n_chunks1 and nonempty first content. Thus the
empty-content fallback applies only to the restart 12K row among these
files. Active r2's cold8186-token request is 35520.949920 ms and its
warm8187-token request3558.202419 ms; this audit assigns no causal
explanation to that cold delay and does not qualify active-window data.

Plan16-r1 was aborted because the private guard could bypass real BF16
scale-view arguments. Its populated gate/bench/probe files are invalid
for treatment or speed attribution and remain unmeasured/discarded.
Stored closure has SERVE_RC0, post-health HEALTHY and unchanged408-file
identity; interrupted callback failure remains visible. Gate presence or
normal shutdown cannot rehabilitate a bypassed treatment.

## Corrected wrapper and pending r2

Independent comparison of `d1/matched-r1/identity.json` to
`d1/plan16-r2/identity-before.json` finds exactly four changed entries:
private helper symlink and its harness target, private FA interface,
and its builder. Two new entries are wrapper-gate probe/launcher. No
entries removed; fork HEAD, tracked-diff hash and untracked list equal.
This agrees with treatment-only-delta.json. Original c7b control sources
and effective SO hashes are unchanged in the recorded manifests, so
the earlier control remains attributable to its original executed path.

`d1/fa-wrapper-gate-2`: ACTIVE16, batch2, paired expanded FP32 scale
views; spy sees the same scale objects at native calls plus plan/work
tensors and split16. Three changed length/page cases pass FP32 tolerance
and exact replay; worst maxabs0.001304716. Native pre/post card1 health
passes; gateRC0/healthRC0, normal exit. Probe1's import-order failure is
preserved separately and contributes no numerical pass.

The real r2 serve emits D1_FA_PLAN_ACTIVE count16 batch4, proving the
corrected treatment was reached during graph capture. r2 remains active
at this report cutoff: no final rate comparison, qualification, teardown,
post-health or after-source claim is made here. Required closure is
unchanged410-file verification, effective SO checks, completed diagnostics,
normal teardown and native post-health, with strict-repeat outcome kept.

## N1, Q1 and P1 scope

N1 `run-p2/primitive.json` has 72 shape cases, three real GDN layers and
two seeds, M1/2/3/4/8/9/20/40/60/128/257/12288. All pass stored
O4-row parity, whole-output repeat, shifted/reversed tile checks. Actual
O4 reference coverage is990 rows total: every row through M60 and six
sampled rows above60. It is not exhaustive O4 parity at M12288.
primitive.log independently contains six graph shapes x8 changed-input
replays=48 passes. OneDNN M20 has83/100 matching identical-input repeats.
M20 FP32 maxabs0.003889322 is reported for both paths. Script time1.675s,
normal body exit, leased native pre/post health pass; r2 library hash
c90919594b0e2b355073e2b398ac76755a3ace5eb0094d014a72a05a411246d0
recomputes. This is component evidence, not full-model determinism,
batch invariance, MTP parity or serving speed. The M12288 N1/oneDNN
3848.2083/140.84895 us hot loop supports retaining a prefill-cost concern.

Q1 `tiny-r1` passes 12 real pooled uint8 QSA aliases during minimal
two-page profiling, final109-page allocation and FULL captures1/2;
page bytes3258368, block2752. Stored summary and log agree; C1/C2
generate8tokens per request, source-after checks and leased pre/post
health pass. These are random dummy weights and constant generated IDs4;
this demonstrates engine integration, not Flash-Next quality or latency.
The prepared Strata reference window has no completed runtime-reference
evidence at this cutoff.

P1 isolated build log is exit0/38s; binary/source hashes recompute to
the values in3P1. Static metadata report has105 unique candidate entries,
zero shared with129 control entries. No selected-DQ GPU result is present
or claimed. Its dequant source/build is isolated from shared runtime.
Independently parsing original q2 T12283/E374 rows yields48 layers,
mean total47.271458ms (includes group/plan), dequant median13.35,
combine9.80, gather4.415; median16chunks and combine/chunks Pearson
0.979969. These match recovered reports and exclude synthetic T12288.
Stage synchronization prevents treating these sums as clean E2E speed.

## Finite audit limits

This is raw-artifact/harness review, not GPU reproduction. Large model
weights were not rehashed, and the410-file serving manifest is a scoped
source/SO ledger rather than a whole-environment or weight-content digest.
Paths, loader metadata, live model identity and registry agree. No
200K/concurrency2, Flash-Next RAM eviction/recovery, full-model QSA,
N1-enabled serving, shelf promotion or external submission is established
by the resumed evidence audited here.

## Completed r2 closure audit - failed engine, healthy cards

CONFIG: corrected active plan16, same control settings, N1=0; parent
requested final raw closure review after the engine died. No new GPU
work, source change or diagnostic rerun by A1.

COMMAND: inspect completed r2 serve/window/repeat/callback logs, gate/
bench/probe/repeat JSON, lmx directory, source and SO checks and native
post-health. Independently compare completed C1 repeat token IDs.

RESULT: the first native fatal is timestamped16:48:53:
`RuntimeError: level_zero backend failed with error: 40
(UR_RESULT_ERROR_OUT_OF_RESOURCES)`. It surfaces at gguf/mtp_ple.py:175,
`self.ring[slots, past.remainder(ring_size)]`, called by PLE conv. The
scheduler dump has two running requests: a new20-token prefill alongside
one decode token,21tokens total, zero scheduled speculation. This is
the observed surfacing location; the root cause is not established by
the asynchronous-backend traceback. In particular, it does not prove
the ring index or the FA plan caused the resource failure.

The N1 repeat callback receives HTTP500 and saves c1_1, c1_2, c2_1.
The missing c2_2 then triggers a separate harness KeyError in
n1_repeat_http.py:41. Primary native failure precedes this secondary
summary bug. Complete C1 128-token repeated streams already differ:
[true,true,false,false]. No same-load C2 repeat can be computed without
the missing round. The earlier promotion gate completed and failed
exact-repeat while passing identity/arithmetic/8K/32K needles.

The combined callback records D1_REPEAT_RC1, D1_SOURCE_RC0, D1_LMX_RC1.
The lmx directory contains only fork-head.txt, fork.patch and
fork-untracked.txt: no canonical request samples, server-config snapshot,
validation or payload exists. Canonical measurement never started;
there is no plan16-r2 canonical512 rate or suite comparison.

Partial short bench observations remain in bench.json: C1p50
15.138275ms, C2p50 20.072390/20.102467ms. These are diagnostic values
from a failed window and do not qualify an end-to-end improvement.
Cold 8K first-content35.520950s and warm3.558202s are real observed
content timings. The saved evidence does not explain the cold spike;
do not automatically call it JIT, plan overhead or a prefill regression.

The API/worker cleanup finishes, and the outer serve command returns0;
SERVE_RC0 therefore cannot be interpreted as a failure-free engine.
Overall D1_PLAN16_R2_RC1 preserves failure. Post-window SO checks pass,
after-source verification has bad=[] for410files, and leased native
post-health reports both cards HEALTHY. These establish intact recorded
source/binary identities and recovered card health; they do not undo the
fatal engine failure or demonstrate candidate stability.

VERDICT: plan16-r2 failed full-model qualification. Preserve its partial
diagnostics, do not use it for promotion or canonical speed attribution.
Control remains diagnostic61.642817/61.800831tok/s, strict-repeat-failing;
no measured candidate suite replaces it. Root owns recovery and the next
bounded discriminator. r1 remains separately discarded for guard bypass.

## Final Q1 full-model audit - integration closes, parity gate fails

CONFIG: CPU-only audit after the authorized Q1 reference/full-model
windows. Evidence: q1/strata-references.json, q1/strata-window.log,
q1/strata-reference-summary.json and x6/flashnext-ista-iq3-xxs-ep-qsa-q1-r1.
Q1 uses maxlen65536/chunk12288, N1=0, D1plan=0, QSA pooled int8 pages;
Strata reference is canonical no-draft C1. Report additions occur after
the Q1 window and its source-after manifest complete. No GPU rerun.

COMMAND: recompute fixture/ref prompt SHA256 and text hashes, check
counts/zero drafts, independently compare each of the four vLLM output
texts to its single Strata reference, recompute all probe booleans,
inspect engine/teardown/source/health closure and latency summaries.

RESULT: all three fixture prompt SHA256s agree with their references;
fixture token-ID counts, target counts, Strata counts and each vLLM
usage are exactly4096/16384/32768. Strata text SHA256s recompute. Every
reference has64raw/outputIDs, generated64, no EOS IDs248044/248046 and
runtime drafts_offered=0/drafts_accepted=0. The CPU tokenizer log passes
all fixture IDs; the actual vLLM probe asserts identical fixture IDs
before requests. Reference process/validation/source/native post-health
all return0, with the same executable SHA25652d1778c909b90123a8947a3a2a9700370b76a2f64d293765f0e2510526fdad2.

The precise x6_probe.py contract is decoded TEXT equality. It saves no
vLLM output token IDs. strata_exact aggregates two C1 solos plus two C2
outputs against one Strata C1 output. Independently recomputed matches:

| Fixture | C1 first | C1 repeat | C2 first | C2 second |
| --- | --- | --- | --- | --- |
| 4K | match | match | differ | match |
| 16K | differ | differ | differ | differ |
| 32K | differ | differ | differ | differ |

All12vLLM responses have64content chunks and authoritative64completion
tokens. The 4K aggregate failure therefore does not mean that C1 fails
the reference: both C1s and one C2 match it exactly. At16K the C1 repeat
itself differs. At32K all four vLLM texts match each other but differ
from Strata, first differing at zero-based character123. All recorded
per-row and final probe booleans recompute exactly; final repeat_gate
and strata_gate are both false, X6_PROBE_RC1.

This is a reliable failure of the declared exact-text parity gate on
matched input fixtures. It does not isolate QSA as the cause: N1 remains
off, same-load/batch variance is already measured in dense control, and
there is no matched dense run of these long fixtures or repeated Strata
reference sample. The unanimous32K result is stronger within-window
repeat evidence than16K, while cross-backend attribution remains open.
All outputs stop at64tokens inside reasoning. These prefixes do not
establish a task-answer quality failure or a failure to return the code
and Paris facts later. No first mismatching token ID is available.

Latency summaries contain a concrete tail:32K C2 first stream p50
77.101816ms, p99 8972.907448ms, mean386.779298ms and decode2.585454tok/s;
the second stream mean76.488886ms/decode13.073795tok/s. The corresponding
decode windows are24.3671s/4.8188s. A median-only report would hide this
stall. Individual arrival timestamps were not retained by this client,
so this audit does not assign its cause. These are diagnostic latency
observations, not a qualified comparison to Strata.

Full-model integration itself closes cleanly:12expected QSA layers bound
without rejected binding lines, arithmetic323/667 and Paris coherence
pass, SERVE_RC0, ENGINE_ERROR_RC0 with empty engine-errors.txt, both-card
native post-health HEALTHY, WINDOW_DONE. All83before/after source/SO
manifest entries are byte-identical; the pinned reference artifact hash
is e91528a578214cff7e7a0852ca9c7f7bd5e9f160ef418e18a4eb44c8359e2b30.
Current small source/config hashes also match the launch ledger.

VERDICT: real Flash-Next QSA integration and teardown/health/source
closure pass. Exact-text reference and repeat qualification fail, with
the causal and quality limits above. No promotion or speed/stability
qualification follows from the clean engine closure.

## Final P1 matched serving pair audit - first-content TTFT improves

CONFIG: CPU-only independent audit after both P1 arms close. Raw:
p1/run-bucket-r1, p1/full-p1-{control,selected}-r1,
f1/merged-p1-{control,selected}-r1 and p1/matched-r1/summary.json.
Control first, selected second; one fresh serving process per arm.
Both use the same P1 bucket/selected libraries and overlay, original FA,
QSA/N1/oneDNNdet off, maxlen65536/chunk12288/maxseq4, no speculation,
prefix cache off. The selected-DQ flag0->1 is the treatment change.
No GPU/build/new test, shared harness/plugin edit, commit or submission.

COMMAND: independently recompute all24request content-arrival/usage
checks, warmup exclusion, medians, input rates, absolute/percent latency
deltas, fixture hashes and output-text variation; compare complete
source/P1 hash manifests, fork snapshots and live server-info configs;
review component and full-engine source/activation/health/teardown.

RESULT: bucket.json contains44outputs in11layer/T groups, each with
original reference, original repeat, overlay control and selected arm;
all are exact_original=true/maxabs0. This is33nontrivial equality checks
plus11reference outputs. Real weights/types and synthetic routes cover
T33/129 at five layers and T12283 at layer8. Its leased native pre/post
card1 health, identity checks and normal component exit pass. Isolated
single-call wall numbers are not used to qualify serving latency.

Both full-arm431-file source-identity manifests are byte-identical,
as are explicit14-entry P1 SO/source hash manifests, fork HEAD and full
binary tracked patches. After each arm, source verify has bad=[];
P1/native SO checks pass. Treatment fixture-before/after checks pass,
as does cross-arm fixture identity. The frozen fixture SHA256 recomputes
as d3811e41588b9f28ae94c0a5fbf0967876172921e966abbabb71680b72cb349a;
all three prompt SHA256s recompute, and live tokenization assertions
and usage agree at8189/12285/32765 actual prompt tokens. Requested
targets8192/12288/32768 are not the rate numerators.

Actual server-info confirms speculative_config=null,
enable_prefix_caching=false,max_model_len65536,max_batched12288,
maxseq4,TP1 in both arms. Comparing the full serialized live configs
finds only different instance IDs and the quant-config object's memory
address. Both actual card1 activation markers show tile32/maxpad32768,
selected0 and selected1 respectively; this is not merely flag intent.

Each fixture has one warmup followed by three measured requests per arm.
All24requests have16nonempty content arrivals,16completion usage tokens,
15saved ITLs, monotonically increasing arrivals and nonempty text.
TTFT equals the first saved content timestamp exactly, and separate wall
time follows the final content event. No wall fallback or early output
termination appears. All saved ITL samples recompute from the arrivals.

| Actual prompt tokens | Control median ms | Selected median ms | Saved ms | TTFT reduction | Selected input tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8189 | 3341.403082 | 3198.826496 | 142.576586 | 4.266968% | 2560.001304 |
| 12285 | 4876.349748 | 4678.923307 | 197.426441 | 4.048652% | 2625.604053 |
| 32765 | 13949.494369 | 13509.474830 | 440.019539 | 3.154376% | 2425.334842 |

Control rates are2450.766878/2519.302477/2348.830655tok/s. Throughput
increases are4.457153/4.219484/3.257118percent, which differ from the
latency reductions above. All three measured sample ranges are separated:
control3340.898..3346.080 versus selected3198.272..3207.971ms at8K;
4870.249..4884.626 versus4677.460..4693.876ms at12K;
13941.318..13957.275 versus13508.582..13520.149ms at32K. These
independent calculations agree with matched-r1/summary.json.

Short identity/arithmetic323/667 and Paris coherence pass in each arm;
both separate32K-context math requests also return323/667, using32791
chat prompt tokens. Short64-token decode C1p50 changes15.214564 to
15.158917ms; C2 streams19.556688/19.514938 to19.552129/19.562719ms.
These small decode differences do not establish a decode improvement.
The strict promotion repeat suite was deliberately not run; GATE_RC2
means disabled here, not a failed execution.

Both P1_ARM_PASSED markers exist, each underlying window returns0,
callbacks return0, SERVE_RC0 and ENGINE_ERROR_RC0. Engine-error files
are empty. Both logs show XPUWorker shutdown done and completed engine
manager/API shutdown without force killing. Both-card native pre/post
health is HEALTHY per arm; no monitor-stop artifact exists. Thus the
matched timing comparison has actual configuration/coherence/identity/
source/activation/engine/health closure, unlike failed earlier arms.

Limits: one fixed-order process pair and three samples per size do not
separate treatment from all order or session effects. The measured
endpoint is HTTP first-content TTFT, including normal serving overhead,
not isolated prefill kernel time. Whole-model output equivalence remains
unqualified: control has2distinct texts at each size; selected has1/2/3
distinct texts at8K/12K/32K across its four requests. Cross-arm hashes
are not all equal. Component arithmetic parity and sampled model
coherence do not resolve the known baseline nondeterminism. No matched
C2-prefill, canonical512 decode,200K, eviction or quality result follows.

VERDICT: the completed matched experiment supports an observed
3.15..4.27percent first-content TTFT reduction from selected-DQ on these
three fixtures, with all required experiment closure passing. Retain
the default-off treatment as a measured prefill candidate. No strict
determinism, shelf promotion, external submission or broader stability
claim is established by this pair.

## N1 instrumented original-FA diagnostic audit - early engine failure

CONFIG: CPU-only audit of f1/merged-n1-liveness-r1 after its finite
closure. N1 r2 enabled, original c7b FA, plan0, no P1 overlay, no QSA,
maxlen131072/chunk12288/capture1,2,3,4,8, TP1 sidecar. All445source
entries remain frozen during the serve. No GPU action, build, additional
HTTP request, test, source edit or commit by A1.

COMMAND: inspect d1_n1_diagnostic.sh, actual private lifecycle snapshot,
builder/classifier/monitor and callback; compare private prefix/suffix/
client body to its captured merged source. Audit actual live server-info,
EngineCore phase/stack/proc evidence, request status and closure checks.

RESULT: the private lifecycle prefix and teardown/health suffix are
byte-identical to captured merged-source.sh. The client body is identical
after removing the added detail-arm marker. The hyperparameters and
greedy -> promotion -> bench -> F1 8K twice+long math sequence match the
historical N1 request sequence. The diagnostic limits the callback to
identity plus its first Gate.greedy(17,19), rather than the full repeat
and canonical suites. A120s timeout bounds the client body only; teardown
and native health remain outside. Diagnostic hooks change execution
overhead, so this is not a performance measurement or an exact runtime
reproduction of the uninstrumented historical serve.

Actual worker integration succeeds: EngineCore107 emits monitor_ready,
installs16hooks and writes armed-107.json with coarse mode. There are
zero metadata_error events, including no recurrence of the prior
kwargs/cross-thread bookkeeping defects. This confirms supported worker
extension initialization/arming in a real model process; detailed-layer
execution was not reached before failure.

Actual server-info proves speculative_config=null,prefix cache=false,
maxlen131072,chunk12288,TP1 and capture sizes1/2/3/4/8. N1 library SHA
recomputes c90919594b0e2b355073e2b398ac76755a3ace5eb0094d014a72a05a411246d0.
Initial process maps contain that N1 library, installed native GDN and
original c7b FA libraries. The private D1 plan and P1 libraries are not
the configured paths. No D1-plan activation marker appears.

The engine exits unexpectedly at17:52:11 during the promotion96-token
repeat, before the later benchmark/F1/historical first-callback arithmetic
request can run successfully. Initial identity and323/667 pass. Promotion
repeat receives HTTP500; later needle/identity calls reset connections.
Bench fails at identity, F1probe fails, and the truncated callback reaches
only its failing identity call. Therefore this run never issues the
historical stalled callback's17*19 request. It does not reproduce that
particular no-return boundary, nor measure N1 whole-model determinism.

The last observed inference phase is execute step810, mixedT6 with
scheduled[1,5], entered without a logged return. Its gap to client-end
is4.457990s, below the8s watchdog threshold. Last10s contain3545phase
events,305execute returns and297sample returns. One actual graph
instantiate call returns in3.759213ms. There is no mixedT21 event,
long_active_phase witness or host-error-bank read. No zero-error-bank
claim is available.

Periodic faulthandler samples include a main-thread stack in ep_submit.
Nearby phase counters continue advancing, so that sampled common path
does not establish a blocked EP submission. The next stack dump ends
partway through the watcher thread. The external proc observer has22
samples over20.990s; final thread/FD counts collapse as the process exits.
Denied syscall fields are saved as PermissionError. These CPU samples
do not provide device completion or a causal native failure location.
The serve log has EngineDeadError but no originating UR/native exception
trace; the cause of the unexpected engine exit remains unlocalized here.

Actual engine termination signal is recoverable from the final saved
proc sample, independently of the outer container's143status. In
d1-liveness/proc/proc.jsonl at wall_ns1792605131298940821, host PID/TID
68426 maps to container NSpid107. Its saved task stat field52
(exit_code, Linux wait status) is11, after prior samples had0; FDcount0,
one remaining thread and zero address space accompany the exit transition.
Wait status11 means signal11/SIGSEGV, not a normal exit status11. Stat
field38=17 is the parent-notification SIGCHLD field and must not be
mistaken for the termination signal. Thus a real EngineCore SIGSEGV is
evidenced; its originating native operation/program counter is not.

Concrete reporting defect: d1-client-end.json records client_rc0 because
the clients shell completes status assignments despite individual command
failures. The classifier consequently prints "finite request sequence
completed; no stability qualification". That is shell-body completion,
not successful request completion. The actual status file is GREEDY0,
GATE3,BENCH1,PROBE1,CALLBACK1; SERVE_RC143 and ENGINE_ERROR_RC1. This
classifier label must not override the failed engine and HTTP evidence.
Nor did the120s deadline expire: the body ended after the early failure.

Closure: overall D1_WINDOW_RC1; native post-health both cards HEALTHY,
source445verify bad=[], N1/source hash check0 and private manifest check0.
These protect evidence identity and establish recovered card health;
normal engine teardown did not pass. No reset or speed/stability claim
follows from this audit.

VERDICT: real worker instrumentation/arming passes; the serving run
fails early engine qualification before its intended historical stall
point. Neither timeout nor >=8s no-return evidence supports a deadlock
claim here. Preserve original classifier output with the correction
above; root owns cause localization and any next bounded experiment.

## QSA top-k component profile audit - reduction benefit is small

CONFIG: CPU-only audit of q1/topk-r1/profile-r2 after closure17:57:20.
Physicalcard1/logical0, seeded populated paged QSA fixtures, capacity4,
active1/2 at4096/16384/32768. Native library plus isolated copied manual
tree and group-reduction variants load together. No actual model load,
loaded-model VRAM pressure or serving test. No GPU/new test/source edit
or commit by A1. The earlier profile-r1 is preserved separately.

COMMAND: review oracle/probe/C++ and build/window source; verify saved
component result, source/SO/image/native health and teardown; independently
parse raw unitrace GPU events into complete replay chains and recompute
every span/kernel/top-k/gap mean in trace-summary.json.

RESULT: profile-r1 rejects its hyphenated unitrace session before Python
probe execution. Body1, no numerical/performance output, native pre/post
health and source checks pass. It is a launcher failure, not a GPU fault.
profile-r2 uses an alphanumeric PID session, same compiled candidate;
body0, no leftover container, native pre/post HEALTHY,30source/SO checks
all OK, closed17:57:20. Candidate SHA256 recomputes
eb83f79dbc6256e9a99d9b8871b6a174a06209e13885336773df57d3f81f84e9;
ported C++ is byte-identical to built scratch source. Runtime is
Torch2.14.0+xpu, B70, imageea234fbf7123cba58b387f4046a0c32192dd5b6febead3684dbd8690882bf572.

There are63oracle invocations for each of three variants,189comparisons.
The list has61unique names because n1/width1 is deliberately emitted
twice for both input families. Independent unsigned64key full-sort
selection checks packed float-bit ordering, lower-ID ties and ascending
emission; cases include NaN payloads, signed zero, infinities, extremes,
zero/negative width and count clamps. Full4x2051ID arrays are compared,
including inactive/unused canary slots. These are within-tensor canaries,
not external allocation guards or a general memory-safety proof.

All six actual-shape components pass exact selected IDs against the
integer-key oracle, gated output parity versus native, changed-score
top-k graph replay and whole-chain replay after query/indexer/K/V/raw
changes plus a new append position. The full QSA chain comparison is
native parity, not an independent model-quality or FP32 attention oracle.
Fixture geometry24/2/256,indexer4x128,compress4,budget2048,cap2051,
block2752 matches the intended component. Pages contain seeded signed
int8KV, positive fp16 scales and finite f32 pooled state; they are not
captured real-model activations.

At32K/M1, saved event averages over12graph replays per arm independently
give these milliseconds:

| Measurement | Native | Copied tree | Group reduction |
| --- | ---: | ---: | ---: |
| Top-k only | 4.024640 | 3.276181 | 3.271923 |
| Whole one-layer chain | 4.639184 | 3.892956 | 3.888264 |

The controlled copied-tree to group difference is4.257833us top-k and
4.691750us chain here. Across the six shapes, chain differences are
4.36..5.54us/layer; top-k differences2.66..4.50us. The tracked native
build recipe uses O2/intel_gpu_bmg_g31; the isolated build uses O3,
spir64_gen/-device bmg and a smaller compilation unit. Therefore the
rough0.75ms native-to-copy32K difference is confounded by build context
and cannot be assigned to group reduction. Both copied variants share
one binary/build; their small difference is the relevant reduction
comparison. Single-run, fixed variant ordering and short averaging do
not establish a robust production gain at this microsecond scale.

The raw trace contains594GPU operation events. Independently partitioning
them by complete11-operation structure yields54replays, three for each
of18shape/variant phases; each top-k symbol confirms native/tree/group.
All reconstructed summary means match. Ordered trace structure, rather
than naive host-walltime timestamps, is needed because the host/GPU
clocks have an offset.

Native M1 device chain spans at4K/16K/32K are1.166333/2.585583/4.634833ms;
top-k contributes0.550520/1.969479/4.016631ms, approximately47/76/87percent.
Attention remains0.488368/0.487569/0.488437ms, gather approximately0.075ms,
index score approximately0.032ms, within-chain gaps approximately9us.
At32K/M1 trace top-k tree/group is3.269514/3.265104ms and chain
3.888500/3.883000ms. This independently corroborates top-k as the rising
cost on these fixtures and the small group-reduction improvement.

Auxiliary uncaptured per-operation event boundaries are substantially
larger and include host submission/readiness effects. They cannot replace
the uninstrumented captured trace as kernel-time attribution. Event
averages and trace spans are distinct: for example4K/M2 native event
chain1.557539ms versus trace1.191250ms; do not silently merge these
measurement methods. A twelve-layer sum is a projection, not measured
model critical-path latency or proof of the full serving overhead cause.

VERDICT: numerical/capture/lifecycle component gates pass. The reduction
substitution provides only a small same-build component improvement;
native-to-copy timing is build-confounded. Preserve the profile and
prioritize a selector algorithm change before a costly model load.
No serving speed, fidelity, loaded-pressure liveness, fault clearance,
promotion or external claim follows from this component window. The
next upstream Torch selector is CPU preparation only at this cutoff.

## Future QSA selector full-serve CPU review - two pre-load fixes

CONFIG: read-only review of q1_selector_hook.py, q1_selector_api.py,
optional x6_serve.sh API/PYTHONPATH/alias support and scratch
q1/selector-serve-prepared.sh. No GPU load, new test, shared source edit,
registry change or commit. The proposed native/fused-r2 pair is not run.

COMMAND: trace import/patch/cache behavior through QsaBatch decode and
prefill callers; review native/torch/fused factories, source/SO ledgers,
graph warmups, live-config checks, client-progress watchdog and teardown.
Compare the watchdog budget to already saved full-QSA request durations.

RESULT: default mode native leaves ops.topk_batch unchanged. The hook
itself imports only stdlib modules; opt-in factory import occurs lazily
inside the actual selector call after base QSA loading/readiness. The
top-level API hook is designed to run again under multiprocessing spawn.
Each cache key includes all three input/output buffer pointers and
shapes. With persistent decode buffers and configured graph warmups2,
workspace construction should precede capture; actual worker integration
and loaded-model allocation behavior remain untested.

Two concrete issues were sent to root before any full load:

1. The scratch watchdog watches x6-probe.json, which is written only
   after two solo requests and one concurrent pair complete per fixture.
   The preserved native-QSA32K phase takes approximately139s in total
   (35.70s+35.77s+67.66s). Its proposed120s no-file-progress rule can
   therefore stop the native control during normal active request
   progress. Use per-request progress or a phase budget consistent with
   measured durations before interpreting that watchdog as liveness.
2. The hook replaces all topk_batch calls, including prefill. QsaBatch
   prefill allocates fresh scores/ids/step per query tile. Every cache
   miss permanently retains its factory workspace; current fused-r1
   stores an int64 keys tensor shaped like scores. At configured
   max_cells147456 and prefill tile256, each such workspace is
   301989888bytes/288MiB. Allocation-address reuse may reduce misses,
   but source has no cache eviction or lifetime bound. Fixed-buffer CPU
   checks do not establish bounded memory for this transient prefill
   path. Bound its ownership or keep the first selector integration
   confined to persistent decode buffers; actual unique-buffer count
   and memory use are unmeasured, not a demonstrated OOM.

The prepared wrapper requires an explicit fused SO path and hashes it
before/after; broad source identity covers repository hook/API/selector
source, while x6's ledger includes native QSA. The scratch wrapper itself
is outside that broad manifest until ported or explicitly hashed. Native
pre/post health, EngineCore error scan and normal teardown are retained;
live config checks require no speculation, prefix cache off,
maxlen65536/chunk12288/maxseq4/TP1. Selector activation is checked from
actual serve logs for non-native modes. These checks are prepared code,
not actual full-model outcomes.

Current fused factory still resolves q1_fused.selector, the serial-r1
implementation. Root explicitly plans to select q1_fused_r2.selector
only after its independent bitonic component pass and port. The planned
research aliases are not yet registered. Both are known preparation
items, so this review does not label the wrapper executable as fused-r2.
The wrapper also propagates known baseline exact-reference/repeat failure
as overall arm failure; if a diagnostic comparison is intended, preserve
that failure separately from operational health/source/teardown closure.

VERDICT: import/default-off structure and scoped decode cache design are
reasonable for an opt-in integration candidate. Resolve progress-budget
and transient-prefill ownership before a loaded serve, then require
actual worker/activation, source/SO/config, coherence and lifecycle
evidence. No GPU permission, numerical pass or serving readiness is
implied by this CPU-only review.

## QSA selector preload fixes prepared and CPU-checked in scratch

CONFIG: parent explicitly authorized scratch replacements under
q1/selector-review-r2 while the D1 serving sources remained frozen.
No GPU work, build, actual selector execution or campaign source edit
by A1. Root later ported the reviewed production files after its freeze
release. This stage supersedes the two unresolved preload issues above.

COMMAND: implement q1_selector_hook.py, q1_selector_outcome.py and
q1_selector_serve.sh in scratch; run nine focused stdlib fake CPU checks,
real multiprocessing-spawn/API-import checks, Bash syntax and an offline
historical-outcome classification. Independently compare root's three
ported production files byte-for-byte with the checked scratch copies.

RESULT: scope now follows QsaRunner._decode via thread-local batch owner,
and selection requires exact identity of owner.scores/ids/step. The
_prefill wrapper clears that scope and restores it in finally, even
when called inside decode. This is necessary because _prefill's unpooled
fallback itself calls QsaBatch.run_decode; wrapping that batch method or
checking batch-size<=4 alone would not reliably preserve native prefill.
Direct topk calls, transient tiny prefill tails and foreign buffers stay
native. The scope follows the existing runner dispatch; it does not add
host reads of device request metadata.

Each persistent QsaBatch owns one selector/workspace plus its exact buffer
references. No global pointer dictionary retains dead batches. Replaced
buffers/shapes fail closed instead of evicting or rebinding a workspace
that a captured graph may own. Same-owner repeated warmups reuse one
workspace; a fresh batch gets separate state. Keys remain alive with the
owner, and graph memory ownership still requires actual worker validation.
No workspace cache is created by the prefill path.

Fused mode lazily imports the ported q1_fused_r2.selector, preserving its
measured loader and namespace. Default native mode does not patch methods;
the hook/API pre-worker path imports only stdlib. No FakeTensor kernels
are added for compilation0/FULL capture; other compilation modes remain
outside this candidate's qualification.

The wrapper now uses a180s whole-fixture output-progress budget and
rejects configured values below180. The historical native32K phase is
approximately139s. A budget stop is explicitly an operational failure,
not a deadlock diagnosis. The prepared native/fused aliases are q1-native
and q1-fused-r2; root registered them. Fused mode requires the r2 loader
port, explicit SO path, and exact match to the measured-library digest
in q1_fused_r2/measured-source-sha256.json:
23e456afdc5ab7bff44b40a45fe51744773d19fe2191eb6f3963f57871d69325.
The real measured SO independently hashes to that value. Selector/API/
wrapper/fixture/reference/manifest sources are explicitly hashed, in
addition to broad campaign identity and candidate SO before/after checks.

Operational outcome is separated from strict repeat/reference status.
The classifier requires all native lifecycle/source/health/error markers,
identity/arithmetic, the complete fixture sequence and all12content
streams with authoritative usage. Prompt counts come from frozen fixture
token-ID lengths, not the target label. Requested output counts come
from each fixture's max_tokens, and content chunks must be positive,
without assuming one chunk per token. Missing/partial data fails closed.
An expected completed strict rejection is retained as false parity flags
and qualification_pass=false; it does not become a healthy-fidelity claim.
Outer source/SO/config/actual activation and watchdog checks remain
mandatory for operational success.

Nine CPU checks pass: persistent workspace reuse, tiny prefill native,
nested prefill native, exception owner restoration, replaced-buffer
failure, thread-local owners, owner/workspace garbage collection, spawn
native/no-Torch and spawn fused/no-Torch. The spawn checks import the
actual API shim as __mp_main__ and exercise fresh fake target-module
imports; no Torch module or device operation is loaded. Historical
x6/flashnext-ista-iq3-xxs-ep-qsa-q1-r1 classifies operational_complete=true,
strict_repeat=false, strict_strata_reference=false, qualification_pass=false,
with no issues. This preserves the earlier independently audited outcome.

Root ported production files as harness/q1_selector_hook.py,
q1_selector_outcome.py and q1_selector_window.sh. All three are
independently byte-identical to the checked scratch replacements.
Root separately adapted the CPU check to write fake-runtime artifacts
outside the repository and reran its nine checks successfully. Scratch
cpu-result.json, spawn-{native,fused}.json and historical-outcome.json
preserve A1's original CPU evidence. ASCII and shell syntax checks pass.

VERDICT: the concrete preload issues are addressed in reviewable code
and targeted CPU regressions. Root owns the new matched full-model GPU
grant, actual spawn/graph activation, memory, source/config and teardown/
health evidence. No serving benefit or actual worker integration pass
is inferred from these CPU-only checks.

## QSA selector pair native-arm audit - healthy diagnostic baseline

CONFIG: CPU-only review after native arm closure; fused-r2 treatment
is starting and sources remain frozen. Raw q1/full-q1-selector-native-r1
and x6/flashnext-ista-iq3-xxs-ep-qsa-q1-selector-native-r1. No new test,
GPU call, heavy CPU work, source change or commit.

COMMAND: independently recompute saved content-arrival/usage/ITL summaries
and per-output reference equality; inspect actual server-info and
activation intent, source/SO/fixture/config/health/kernel/lifecycle
evidence. Compare the native manifest with treatment's initial manifest.

RESULT: operational_complete=true, strict_repeat=false,
strict_strata_reference=false and qualification_pass=false agree with
the raw completed client and lifecycle. Stable hotschmoe-dd is first;
registered research alias is qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-q1-native.
Actual config has speculation null,prefix cache false,maxlen65536,
chunk12288,maxseq4,TP1. Native mode is configured and produces no fused
selector activation marker. Explicit candidate-SO hash23e456... is
recorded/unchanged for pair identity; native mode does not execute it.

All12requests have64content arrivals,64authoritative completion tokens,
matching4096/16384/32768fixture usage/hash and63individual ITLs. Every
TTFT, p50, p99, mean and saved ITL recomputes from the arrays. The native
second-solo C1 p50 is30.173296/47.574053/70.832722ms at4K/16K/32K;
C2p50 is35.537847/35.530343,53.157973/53.162010 and
76.937344/76.892548ms per stream. Short64-token benchmark C1p50 is
18.150536ms, C2 22.591668/22.594013ms. These are diagnostic baseline
observations; no treatment gain is available at this cutoff.

At32K, C2stream0 has mean387.346937ms,p99 8990.108941ms and
decode2.581665tok/s, despite p50 76.937344ms. Its first content is at
43.005214s, then gaps13.597015s and6.166521s. The other stream's
first content is62.803493s and its mean is76.427946ms. Those gaps
overlap the other request awaiting its first token. This is consistent
with a scheduling/interference hypothesis; it does not identify GPU,
engine, host or client causation. Report p99/mean and full decode rate
alongside median; a median-only claim would hide the dominant tail.
The complete32Kfixture phase takes138.896s, below the corrected180s
phase-output budget. No watchdog stop occurs.

Reference text match vectors [C1first,C1repeat,C2first,C2second] are
4K[T,T,F,F],16K[F,F,F,F],32K[F,F,F,F]. C1 repeats match at4K/32K but
not16K; C2-versus-first-solo vectors are [F,F],[T,F],[T,T]. This is
known baseline repeat/reference failure, not successful fidelity from
operational completion. All outputs are truncated64-token reasoning
prefixes, not answer-quality verdicts.

Actual serving score width follows QsaRunner._limits and QsaBatch:
ceil(65536/2752)=24blocks, already aligned for the default128token
block-table alignment, giving66048score columns. The147456environment
value is the fallback when runtime maxlen is absent. Component fixtures
used147456columns; the loaded-serving shape is therefore different.
Native has no selector marker reporting columns, so66048 is source plus
live-config/block-size derivation here; treatment activation must confirm
its actual shape before using component timing as a serving predictor.

All486source entries verify bad=[]; treatment's starting486-file ledger
is byte-identical. The native QSA before/after ledger and seven explicit
selector/API/wrapper/fixture/reference/manifest entries pass. Twelve QSA
layers bind, identity/arithmetic/Paris coherence pass, SERVE_RC0 and
ENGINE_ERROR_RC0 with empty error log. Actual XPUWorker cleanup, stopped
engine manager and completed FastAPI shutdown appear; native pre/post
health is HEALTHY on both cards. Kernel summary captures journalctl rc0,
66lines and no matching fault/reset entries during this arm. Metrics
connection refusal observed after client completion coincides with
normal shutdown, not a stalled request.

VERDICT: native arm closes as a healthy finite diagnostic baseline,
while strict repeat/reference qualification remains false. Preserve
large32KC2tails and the66048-versus147456shape distinction in the later
pair assessment. No full-model stability, fidelity or speed promotion
is established by this single arm.

## Final QSA native/fused-r2 serving-pair audit - context tradeoff

CONFIG: CPU-only audit after both selector arms close. Raw directories:
q1/full-q1-selector-{native-r1,fused-r2-r1} and corresponding
x6/flashnext-ista-iq3-xxs-ep-qsa-q1-selector-* directories. One fresh
process per arm, native first. Same original QSA and native prefill;
fused-r2 replaces only persistent runner-decode selection. No GPU call,
new test, heavy CPU work, production source change or commit by A1.

COMMAND: independently recompute all24stream arrivals/usage/ITLs,
percentiles/means/rates and text-reference flags; compare all saved
source/SO/fixture ledgers and serialized live configs; inspect actual
activation order, lifecycle/native health and captured kernel journals.

RESULT: both wrappers report operational_complete=true, strict_repeat=false,
strict_strata_reference=false,qualification_pass=false. Their underlying
X6 armRC1 is the completed strict-parity reject; it does not conceal an
engine error. All486source entries and seven explicit extra identity
entries match byte-for-byte across arms and verify unchanged afterward.
Native QSA ledgers match within each arm. Both pin unchanged fused-r2 SO
23e456afdc5ab7bff44b40a45fe51744773d19fe2191eb6f3963f57871d69325.
Frozen fixture text hashes, token-ID/usage counts and reference artifact
identity agree across the pair. Stable hotschmoe-dd is first; the two
registered research aliases distinguish native and fused-r2.

Actual treatment activation has12unique workspace owners, all
scope=decode,batch4,columns66048,cap2051, before the first graph-capture
completion line; no prefill-sized selector workspace appears. This
confirms real worker import/patch/warmup integration, rather than flag
intent or component-only activation. The persistent score width66048
matches the live65536context/block2752 geometry and differs from the
147456columns used in the prior isolated component profile.

Configured runtime parameters match: speculation null,prefix cache false,
maxlen65536,chunk12288,maxseq4,TP1. The full serialized live configs are
not identical: realized GPU blocks234->233, planned KV tokens
567978->565551 (2427fewer) and computed maximum concurrency
8.666667->8.629630. First graph reservation is0.30->0.35GiB. These
are observed treatment-associated footprint/capacity changes, not a
measured200K capability. Other serialized differences are instance IDs
and the quant-config object's repr address. Do not describe all actual
configuration as identical or ignore this capacity cost.

All24responses have64observed content arrivals,64authoritative completion
tokens, matching4096/16384/32768prompt usage and63saved ITLs. First
arrival equals TTFT; every saved interval, p50/p90/p99, mean and full
decode rate recomputes from the arrays. No empty-content wall fallback
or early completion occurs. The observed ITL tradeoff is:

| Fixture | Native C1 p50 ms | Fused C1 p50 ms | C1 latency change | Native C2 p50 ms | Fused C2 p50 ms |
| --- | ---: | ---: | ---: | --- | --- |
| Short64-token bench | 18.150536 | 22.748984 | +25.335% | 22.591668/22.594013 | 27.253158/27.253047 |
| 4K, second solo | 30.173296 | 28.779576 | -4.619% | 35.537847/35.530343 | 34.378046/34.348109 |
| 16K, second solo | 47.574053 | 28.827638 | -39.405% | 53.157973/53.162010 | 34.567697/34.568429 |
| 32K, second solo | 70.832722 | 28.917990 | -59.174% | 76.937344/76.892548 | 34.637872/34.609147 |

Short C2 latency regresses20.634/20.621percent. Long C2 medians improve
approximately3.26/3.33percent at4K,34.97/34.98percent at16K and
54.98/54.99percent at32K. These are matched finite diagnostic observations,
not canonical512-token benchmark or global serving qualification.

The32KC2tail persists. First-stream native/fused mean is
387.346937/346.677597ms, p99 8990.108941/8981.121534ms and full decode
rate2.581665/2.884524tok/s. Its median improvement therefore does not
remove the approximately9s p99 or establish uniformly low latency.
The second stream mean76.427946->34.054738ms and p99
77.945226->35.350924ms improves much more consistently. Native first
stream's two huge initial gaps overlap the other stream awaiting first
content; the candidate retains a similar pattern. This supports an
interference/scheduling investigation, not a causal GPU diagnosis.

Observed first-content TTFT remains nearly unchanged in the warmed
second-solo and C2 requests, consistent with native prefill remaining
in use. C1 TTFT native->fused is2.806044->2.822570s at4K,
12.548501->12.531559s at16K and31.213743->31.255721s at32K.
C2 pairs are5.630264/5.650698->5.649624/5.670908s,
24.563141/24.592380->24.581406/24.609363s and
43.005214/62.803493->43.037261/62.817860s. The first16Ksolo is an
exception:13.057397->15.189530s (+2.132133s), with no proven cause in
this audit. Do not claim that every prefill timing is unchanged.

Whole long-client logs are216s native and207s fused. Summing observed
request phases (two solo walls plus maximum C2 wall per fixture) gives
215.879313->206.473348s,9.405964s saved. Long-context decode improves
substantially while prefill/TTFT and C2tail occupy much of this short
64-output workload. Do not substitute a large token-median reduction
for its much smaller end-to-end client-duration change.

Strict reference/repeat evidence remains mixed. At4K native Strata-text
matches [T,T,F,F], fused [T,T,F,T]; at16K/32K none of the four outputs
in either arm matches its single Strata reference. Native C1 repeats are
[T,F,T] across context
sizes; fused [T,F,F]. These are baseline and candidate failures, not a
fidelity fix. Without full-model selected-ID/intermediate comparisons,
the text differences do not isolate selector causation. All saved
outputs are64-token reasoning prefixes and do not qualify answer quality.

Both arms bind all12QSA layers, pass identity/arithmetic/Paris coherence,
complete engine-manager/API shutdown without a forced kill, record
SERVE_RC0/ENGINE_ERROR_RC0 with empty engine-error logs, and pass native
both-card pre/post health. Neither phase watchdog fires. Captured kernel
journals have readRC0 and no matching fault/reset entries in the arm
windows. These establish finite operational closure and intact recorded
identity; they do not establish global stability or fault clearance.

VERDICT: fused-r2 is a measured context-dependent decode candidate:
short decode regresses,16K/32K medians improve markedly, C2tails and
strict fidelity/repeat failures persist, planned KV capacity drops one
block. Keep this tradeoff explicit. One fixed-order process pair with
two C1 streams and one C2 pair per context is not a repeated statistical
campaign, promotion, quality or external-submission qualification.
