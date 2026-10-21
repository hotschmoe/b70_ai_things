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
