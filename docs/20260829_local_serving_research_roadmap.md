# Local B70 serving research roadmap

Date: 2026-08-29

Status: execution underway. Phase 0 H01-H07 and Phase 1 M01-M02 passed on
2026-08-29; the isolated M03 completion-ownership A/B is next. This roadmap
supersedes no historical evidence. It incorporates the 2026-08-29
Neural.Download and XeCores audit recorded in
`docs/20260829_neural_xecores_deep_dive_and_campaign_state.md` and the current
Terminal-Bench relaunch contract in `evals/terminalbench/CAMPAIGN_RELAUNCH.md`.

The Phase 0 preflight now proves the exact Pi 0.84.3 off/xhigh payloads,
policy-dependent launcher state, observed model and dtype validation,
trajectory classification, full lifecycle clock, and deterministic local-70
manifest. The M01 line-level ledger is
`docs/20260829_steve_completion_state_port_ledger.md`. It found the eager
async/wait concept present, but compiled completion ownership, compiler-visible
GDN state, current-API cache binding, fixed-256 B/A, exact two-row RMSNorm, and
the deterministic Inductor launch contract remain unported. M02 then proved
the required BF16 P2P-off collectives across three fresh lifetimes. Its exact
route boundary is recorded in
`docs/20260829_m02_p2p_off_compiled_collective_oracle.md`: all-reduce supports
functional-wait graph replay, while all-gather graph replay requires an opaque
direct custom op because a functional event wait is illegal in XPUGraph.

## Goal

Find the highest-scoring Terminal-Bench 3.0.0 recipe for one Pi agent decode
stream while also requiring enough speed and robustness to finish the task.
The recipe may use TP1 or TP2, any retained or deliberately introduced
backend, any qualified quantization, graph or compiled mode, cache policy,
custom operation, and target-exact speculative method.

Terminal-Bench score and normal task completion are the primary outcome.
Total task time and total machine occupation distinguish the speed of viable
high-scoring recipes. Server TTFT, prefill, decode, cache reuse, acceptance,
errors, and restarts are explanatory metrics rather than substitutes for the
task result.

TP1 and TP2 have equal standing in this campaign. A TP1 winner has a possible
later DP2 deployment benefit because each B70 could host an independent
replica, but DP2 concurrency is not part of the next recipe-selection campaign.

## Campaign topology definitions

| Label | Physical shape | Primary question |
| --- | --- | --- |
| TP1 | One model replica on one B70 | What is the best latency and simplest stable control? |
| TP2 | One model replica sharded across both B70s | Does sharding improve one-request completion or enable otherwise impossible capacity? |
| DP2, deferred | Two independent TP1 replicas, one per B70 | Possible local-user concurrency benefit after the single-stream winner is selected; not tested in this campaign |

## Non-negotiable experiment contract

Every GPU transaction uses `bin/gpu-run`. Every result is recorded in
`JOURNAL.md` as CONFIG -> COMMAND -> RESULT -> VERDICT. No result is trusted
without all applicable gates below.

### Identity and configuration

- Record host kernel, UMD, Level Zero, oneCCL, PyTorch, backend, image digest,
  source revision, model revision, and patch hashes.
- Query `/v1/models` and require the exact served ID from
  `evals/configs/models.yaml` or the experiment-specific extension.
- Record target dtype and observed KV dtype from the runtime.
- BF16 KV is the default. FP16 or FP8 KV is a separately labeled control.
- Record TP, PP, replica count, card affinity, P2P state, graph mode and sizes,
  MTP method/depth, cache policy, context, batch-token limit, maximum running
  requests, seed, power cap, and prompt corpus.

### Correctness and stability

- Target-only reference is captured on the same stack before MTP or draft
  quantization.
- Greedy fixed corpus must be repeat-exact within a fresh server and across two
  fresh servers before promotion.
- MTP and draft candidates must match the same-stack target token arrays. A
  high acceptance rate does not excuse divergence.
- Require concurrent coherence at the intended maximum request count.
- Scan server and kernel logs for engine death, Level Zero abort, GPU VM fault,
  OOM, NaN, garbage, assertion, and incomplete response markers.
- Require graceful teardown, endpoint down, per-card health, and compiled
  P2P-off collective health.

### Performance and task metrics

Measure separately:

- startup and graph-capture time;
- TTFT and input/prefill tok/s;
- post-first/decode tok/s;
- MTP proposed tokens, accepted tokens, and acceptance by depth;
- graph pieces, fences, host waits, submissions, and collective count/shapes;
- cache hit/reuse rate and KV occupancy;
- per-request latency and output rate at C1; concurrency results are recorded
  only when needed for runtime robustness or a later shelf qualification;
- server-start-to-task-finish and full pre-health-to-post-health machine time;
- Terminal-Bench reward, normal completions, timeouts, length stops,
  infrastructure failures, tool calls, edit occurrence, and post-edit tests.

### Matched-comparison rule

Do not call a topology or backend faster unless model revision, quantization,
target/KV dtype, context, prompt corpus, output cap, cache state, MTP state,
concurrency, and timing definition are matched. Cross-quant and cross-backend
results are product comparisons, not mechanism attribution.

## Evidence already closed or rejected

These controls should not be repeated without a new mechanism hypothesis.

| Finding | State |
| --- | --- |
| Arbitrary full-model vLLM P2P-on TP2 | Closed by the local queue-handoff/device-loss boundary; use P2P-off unless an isolated oracle first changes the boundary |
| Qwen W8A8 SGLang FULL for 65K agent work | Rejected after the approximately 17K hardware fault |
| Qwen NVFP4 SGLang FULL for 65K agent work | Rejected after the approximately 19K Level Zero graph abort |
| Qwen GPTQ vLLM PIECEWISE for 65K agent work | Rejected after the long-agent Level Zero abort |
| Ornith memory fraction 0.90 | Rejected after global host OOM and tmux/user-systemd loss |
| Ornith MTP1 on the current SGLang speculative route | Rejected after changing 7/8 target completions |
| Qwen GPTQ TP2 eager on the vLLM 0.28 control | Negative control: 4.4903 versus 7.9304 tok/s TP1 |
| Qwen GPTQ five MTP linears INT4 on the local vLLM 0.28 overlay | Rejected after changing 3/8 target completions |
| FP8 KV for the GPTQ campaign | Rejected after repeat nondeterminism on both tested vLLM stacks |
| Four-thousand-token alleged thinking-off arm | Invalid as true-off evidence because Pi still enabled thinking; the cap also ended before an edit |
| CPU affinity as the Qwen TP2 gap | Closed locally at approximately +0.07 percent |

## Phase 0: repair the evaluation contract

No official GPU pilot starts until this matrix passes.

| ID | Priority | Test | Configuration or comparison | Pass gate |
| --- | --- | --- | --- | --- |
| H01 | P0 | Pi true-off payload oracle | Pi 0.84.3, `thinking=off`; inspect provider request before network send | `enable_thinking=false`, `preserve_thinking=true`, no `reasoning_effort`, and off is not clamped upward |
| H02 | P0 | Pi xhigh payload oracle | Same adapter with `thinking=xhigh` | `enable_thinking=true`, intended strict-thinking policy present, no unsupported `reasoning_effort` |
| H03 | P0 | Policy-dependent cap oracle | Compare off and xhigh launcher environments without GPU | True off has no strict-thinking grammar/cap; xhigh has the recorded cap only |
| H04 | P0 | Observed dtype and identity capture | Feed fixture server logs and `/v1/models` responses into lifecycle parser | Result records observed target/KV dtype and exact served ID; mislabeled values fail closed |
| H05 | P0 | Stop-reason and activity capture | Replay preserved Pi trajectories | Final reason, length stop, timeout, tool count, edit occurrence, and post-edit test are all recorded |
| H06 | P0 | Full lifecycle clock | Dry-run lifecycle with mock endpoint | Clock begins before pre-health and ends after teardown/post-health; startup, Harbor, and engineering time remain separable |
| H07 | P0 | Local-70 manifest lock | Hash Terminal-Bench 3.0.0 manifests and exclude the four H100 tasks | Exactly 70 local tasks, exact four-task exclusion list, stable shard manifest |

## Phase 1: source and mechanism closure

This phase is mostly read-only or micro-oracle work. It prevents a long serve
from being the first test of a risky source transfer.

| ID | Priority | Test | Source hypothesis | Pass gate and disposition |
| --- | --- | --- | --- | --- |
| M01 | P0 | Steve completion/state source diff | Audit current official-FP8 `async_op=True` plus `Work.wait()`, GDN state binding, cache mutation, and RMSNorm replay against retained vLLM source | Produce a line-level port ledger; no patch is accepted merely because it exists upstream |
| M02 | P0 | P2P-off compiled collective oracle | BF16 shapes `[1,5120]`, `[4,5120]`, and all-gather `[4,2560]`; direct, compiled, and replayed | Numerical equality, matched rank entry/return, repeated teardown and collective health |
| M03 | P0 | Explicit completion A/B oracle | Source-default c10d versus explicit asynchronous work plus wait, P2P off | Same values and no consumer race; endpoint performance is tested only if the oracle changes completion ownership safely |
| M04 | P0 | Graph boundary census tooling | Count graph pieces, fences, waits, submissions, collectives, and tensor shapes per target token | Census agrees across ranks and adds bounded overhead; required for topology matrix |
| M05 | P1 | XeCores full draft S+M1 source audit | Compare cookbook draft-head and five-linear patches with local vLLM 0.27.2 and 0.28 APIs | Exact target/draft ownership map, fail-closed guards, no target-head mutation, tracked source only |
| M06 | P1 | Prefix-cache state audit | Trace SGLang radix/vLLM prefix cache interaction with Qwen/Ornith GDN recurrent state and graph reclaim | Define an exact cache key, state-buffer lifecycle, invalidation rule, and correctness oracle before enabling cache |
| M07 | P1 | llama.cpp transfer ledger | Identify communication-consumer fusions useful to SGLang/vLLM without copying ABI binaries | Rank source-port candidates by compatible tensor shapes and expected removed boundary count |
| M08 | P2 | Fresh llama.cpp research lane decision | Evaluate a new tracked source checkout and fresh weight fetch, not archive restoration | Open only if retained backends cannot answer the fusion/topology question and the maintenance cost is accepted |

## Phase 2: establish corrected per-recipe baselines

### Qwen3.8 W8A8 compressed-tensors, SGLang

| ID | Priority | Topology | Change under test | Required comparison and gate |
| --- | --- | --- | --- | --- |
| W01 | P0 | TP2 | Reconfirm target-only breakable size 1 plus reclaim500, BF16 KV, memory fraction 0.70, max request 1 | Same-stack exact corpus, 50K forced-output replay, flat late throughput, clean teardown |
| W02 | P0 | TP2 | Eager versus breakable versus breakable+reclaim500 | Match prompt/output and cache-off state; attribute speed to graph and stability to reclaim separately |
| W03 | P1 | TP2 | Prefix/radix cache off versus on | Growing 8K -> 16K -> 32K tool-history trace; exact output/state, reported reuse, no replay fault |
| W04 | P1 | TP2 | MBT 8192 versus 16384 | Same growing trace; compare TTFT, total trace time, KV pressure, and decode |
| W05 | P1 | TP2 | Existing source-default c10d versus any M03-qualified completion route | Require coherence and at least 3 percent matched endpoint gain or a measured stability improvement |
| W06 | P1 | TP2 | Target-only versus supported MTP1 | Run only after W01; exact target arrays and real Pi-shaped acceptance required |
| W07 | P2 | TP1 | Capacity/fit screen at 32K then 65K BF16 KV | If it fits with safe headroom, qualify it as a single-stream recipe; otherwise record capacity failure without tuning around OOM |

### Qwen3.8 RadixArk NVFP4, SGLang

| ID | Priority | Topology | Change under test | Required comparison and gate |
| --- | --- | --- | --- | --- |
| N01 | P0 | TP2 | Eager target-only BF16-KV long control | Repeat exact, 50K forced output, 65K fit, teardown health; provides score-completing fallback |
| N02 | P0 | TP2 | Port quant-neutral breakable graph support from W8A8 | Exact equality to N01 before performance; collect boundary census |
| N03 | P0 | TP2 | Breakable plus reclaim500 | Cross 50K output and prior 19K failure boundary without Level Zero abort |
| N04 | P1 | TP2 | FP8 W8A16 projection cutoff M<=1 on winning graph mode | Match target arrays; confirm endpoint gain survives long context |
| N05 | P1 | TP2 | Prefix/radix cache on | Same growing-history gate as W03 |
| N06 | P1 | TP1 | Current vLLM or SGLang one-card candidate at 32K and 65K | Establish fit, exactness, graph coverage, and long stability independently of TP2 |

### Qwen3.8 GPTQ INT4 G128 plus BF16 MTP checkpoint, vLLM

All earlier speed claims are controls until the observed BF16 target and KV
dtype gate passes again.

| ID | Priority | Topology | Change under test | Required comparison and gate |
| --- | --- | --- | --- | --- |
| G01 | P0 | TP1 | vLLM 0.27.2 eager, target-only, `--dtype bfloat16`, observed BF16 KV, MTP off | Fresh two-server target corpus, 65K capacity, 48K prefill canary, long forced-decode canary |
| G02 | P0 | TP1 | vLLM 0.27.2 target-only eager versus PIECEWISE/breakable | Only after G01 score-completing path; exact arrays and graph census required |
| G03 | P0 | TP1 | Add reclaim/re-instantiation to the winning graph mode | Must cross the prior long-agent abort boundary and 50K forced output |
| G04 | P1 | TP1 | MTP depths 1, 2, and 4 on corrected target | Screen in order; stop at first target divergence; record acceptance by depth on Pi-shaped prompts |
| G05 | P1 | TP1 | BF16 draft head versus draft LM-head INT4 | Same accepted MTP depth, exact target, matched cache and graph mode |
| G06 | P1 | TP1 | Complete draft S+M1 INT4 | Apply both cookbook overlays only after M05; fail closed on any target difference; compare acceptance and total latency |
| G07 | P1 | TP1 | Prefix caching off versus on | Growing 8K -> 16K -> 32K trace with exact cache/state evidence |
| G08 | P1 | TP1 | vLLM 0.27.2 versus 0.28 regression bisection | One stack layer at a time; same corrected BF16 target configuration; rebuild ABI extensions |
| G09 | P1 | TP2 | Critical Steve replication matrix: eager and PIECEWISE size 1, MTP0, BF16 KV, P2P off | Compare matched TP1/TP2; TP2 must be at least 1.25x TP1 with target equality or remain a negative control |
| G10 | P2 | TP2 | Add the G04 winning MTP depth | Only after G09 passes topology and stability gates |

### Ornith-1.5-35B-A3B

Ornith is evaluated as separate capacity and TP1 product lanes. Dense-Qwen TP
expectations do not transfer to its sparse active expert set.

| ID | Priority | Topology/backend | Change under test | Required comparison and gate |
| --- | --- | --- | --- | --- |
| O01 | P0 | TP2 SGLang W8A8 | Reconfirm target-only breakable size 1 plus reclaim500, BF16 KV, memory fraction 0.70 | Exact corpus, 50K output, long-agent stability, no host OOM |
| O02 | P0 | TP2 SGLang W8A8 | True thinking-off policy after H01-H03 | Must edit, run a post-edit test, reach verifier, and avoid length/timeout/infrastructure stop |
| O03 | P1 | TP2 SGLang W8A8 | Re-test the new official/Shisa head at MTP1 only after source-path repair | Target equality is mandatory; previous 7/8 divergence remains the negative control |
| O04 | P1 | TP1 vLLM MixedCal-v2 GPTQ INT4 | Reproduce target-only BF16-KV one-card recipe from tracked weights/source | Exact two-server corpus, 65K fit, long soak, full identity ledger |
| O05 | P1 | TP1 vLLM MixedCal-v2 | MTP1 BF16 draft, then draft INT4 | Do not test MTP2/4 until MTP1 is exact and beneficial; compare against O04 |
| O06 | P1 | TP1/TP2 | Matched topology control where the same quant fits both | Expect TP1 to win unless evidence changes; record active-expert and collective census |
| O08 | P2 | Winning stable route | 150 W versus 230 W | Same growing agent trace; compare TTFT, total task time, decode, energy, and temperatures |
| O09 | P2 | TP1/TP2 | Prefix caching off versus on | Exact recurrent-state and growing-history gate from M06 |

## Phase 3: matched single-stream topology matrix

This phase pits qualified recipes against each other only after their internal
correctness and long-stability gates pass.

### Single-request topology matrix

For each model/quant that supports matched TP1 and TP2, run:

| Cell | Topology | Graph | MTP | Cache | Purpose |
| --- | --- | --- | --- | --- | --- |
| T1 | TP1 | eager | 0 | off | Base arithmetic and launch control |
| T2 | TP2 | eager | 0 | off | Reproduce or reject eager scaling |
| T3 | TP1 | winning graph/compiled mode | 0 | off | Isolate single-card graph benefit |
| T4 | TP2 | same graph/compiled intent | 0 | off | Critical graph-enabled TP scaling cell |
| T5 | TP1 | winning mode | winning MTP | off | Draft benefit without collectives |
| T6 | TP2 | winning mode | same MTP | off | Distributed draft/collective cost |
| T7 | TP1 | winning mode | winning MTP | on | Single-user prefix-reuse product route |
| T8 | TP2 | winning mode | winning MTP | on | Sharded long-session product route |

Run short-prefill/long-decode and long-prefill/128-output shapes separately.
Repeat the winning cells at 32K and 65K. A TP2 cell below 1.25x TP1 at C1 or
with TTFT above 2x TP1 is not extended with deeper MTP until its synchronized
forward and boundary census explain the result.

### Deferred DP2 deployment note

DP2 is intentionally absent from the experiment matrix above. If a TP1 recipe
wins the single-stream campaign, a later deployment qualification can run two
independent replicas for local-user concurrency. That follow-up must not delay
or influence selection of the best one-stream Terminal-Bench recipe.

## Phase 4: Terminal-Bench qualification ladder

### Policy calibration

| ID | Arm | Test | Pass gate |
| --- | --- | --- | --- |
| TB01 | Qwen W8A8 stable reclaim500 | True thinking off, 8,192 max output, `bun-sourcemap-leak` | Edit within about ten minutes, post-edit test, normal verifier, no length/timeout/endpoint loss, clean lifecycle |
| TB02A | Qwen W8A8 | One true-off 12,288-token retry only if TB01 ends on length before edit | Do not keep raising the cap against one task |
| TB02B | Qwen W8A8 | One native-thinking comparator with 16,384 max output and an 8,192 private-thinking cap after TB01 completes normally | Compare edit latency, verifier result, and full machine time; do not infer benefit from a longer trace alone |
| TB03 | Ornith W8A8 stable reclaim500 | Transfer the exact TB01/TB02 winning policy | Normal verifier and clean lifecycle; no model-specific policy drift |
| TB04 | NVFP4 winning long route | Same policy and Bun task | No graph failure and normal verifier |
| TB05 | GPTQ winning corrected BF16 route | Same policy and Bun task | No PIECEWISE failure and normal verifier |

A normal zero reward is model-quality evidence. Zero caused by timeout, length,
or infrastructure loss is not a successful task-time result.

Qwen's official Qwen3.8-27B model card at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` makes thinking the default, names
xhigh as its deepest reasoning policy, and reports `max_tokens=32768` for its
QwenSWEBench evaluation. That supports testing above the inherited 4,096
private-thinking cap, but does not establish that a higher cap improves this
1,800-second Terminal-Bench harness. The local Pi endpoint cannot safely send
Qwen's official `reasoning_effort` field, so TB02B is labeled native thinking
plus an explicit server cap, not an exact official-xhigh reproduction.

### Three-task expansion gate

Every arm then runs, one task per fresh server lifecycle:

1. `bun-sourcemap-leak` for long tool use and replay pressure.
2. `production-planning` for operations reasoning.
3. `sglang-qwen-burst` for ML/code tool use.

All three must produce verifier results without infrastructure loss, and at
least one must score nonzero before sharding begins.

### Recipe tournament

Qualified recipes enter one matched single-stream tournament:

| Tournament | Shape | Ranking |
| --- | --- | --- |
| R1 single stream | One task at a time, C1, cold and warm-prefix labels separate | Reward first, then normal completion rate, total task time, and machine time |

Use deterministic resumable shards of four to six tasks. Preserve the same Pi
version, prompt, thinking policy, output cap, context, task order, and verifier
environment across recipes. Aggregate by unique task, never by unweighted
shard mean.

The local tournament covers `TB3-local-70`. The four H100 tasks remain a
separate remote lane and cannot be relabeled as local official results.

## Phase 5: optional deeper source work

These are opened only after the roadmap above reveals a measured bottleneck.

| ID | Trigger | Candidate work | Required proof |
| --- | --- | --- | --- |
| X01 | TP2 graph works but remains boundary-bound | Port a communication-consumer fusion inspired by Steve's llama.cpp root reduction and residual/RMS/Q8 handoff | Remove a counted boundary or wait; numerical oracle first, then matched serve |
| X02 | Recurrent-state breaks broad capture | Port compiler-visible GDN/cache mutation concepts from Steve's current vLLM route | Target equality, long-context paged attention, no forced-chunk regression |
| X03 | Prefill dominates total task time | Optimize or gate push all-reduce for large eager prefill while leaving captured decode on the qualified route | Matched TTFT and total trace improvement; no decode regression or queue handoff failure |
| X04 | vLLM 0.27.2 remains much faster than 0.28 | Layer-by-layer runtime/compiler bisect | One changed layer at a time with rebuilt ABI extensions and full health |
| X05 | Retained backends cannot reach useful Q4 TP2 scaling | Open a fresh tracked llama.cpp SYCL research lane | New source/weights, full identity, no archive dependency, output and health gates equal to retained backends |
| X06 | Neither TP1 nor TP2 has an acceptable single-stream result | Investigate PP=2 or a hybrid topology only with a current correctness oracle | No captured-PP empty output, no poisoned collectives, clear single-stream completion hypothesis |

## Stop rules

- Stop an arm immediately on target divergence, repeated nondeterminism,
  endpoint death, Level Zero abort, GPU VM fault, host OOM, or failed post-health.
- After one TP>1 crash, recover and re-run health before another risky arm.
- Do not chain repeated worker-init failures.
- Do not add MTP to an unexplained target-only topology failure.
- Do not add cache to an unexplained graph/state failure.
- Do not optimize a graph mode that cannot complete the long canary; retain an
  eager score-completing fallback.
- Do not promote a less than 3 percent speed delta without position-balanced
  repeats and low enough variance to support it.
- Do not rank timeout or crash wall times as successful completion speed.
- Do not call a cross-quant product comparison a topology mechanism
  attribution.

## Promotion outcomes

Each qualified recipe receives one of four labels:

| Label | Meaning |
| --- | --- |
| Mechanism-only | Oracle or profile changed as predicted; no serving claim |
| Research-qualified | Identity, exactness, short performance, teardown, and health pass |
| Agent-qualified | Long growing trace and three-task Terminal-Bench gate pass |
| Single-stream winner | Tournament evidence supports a TP1 or TP2 recipe as the highest-scoring robust C1 route |

Shelf promotion still requires concurrent coherence, measured speed, long
stability, graceful teardown, post-health, and an unambiguous served ID. A
single fast canary, imported recipe, or external headline is insufficient.

## Recommended execution order

1. H01-H07: repair the evaluation contract.
2. M01-M06: close source, completion, boundary, and cache oracles.
3. W01 and O01: reconfirm the two existing stable long-agent controls.
4. N01-N03 and G01-G03: create score-completing NVFP4 and corrected BF16 GPTQ
   long routes.
5. G09 and the T1-T4 matrix: answer graph-enabled TP1 versus TP2 cleanly on a
   dense model that fits one card.
6. W03, N05, G07, and O09: qualify C1 prefix reuse.
7. G04-G06 and O03-O05: screen model-specific MTP and draft quantization.
8. N06 and O04-O06: ensure all viable one-card recipes receive a fair TP1
   qualification.
9. TB01-TB05 and the three-task gate.
10. R1 single-stream Terminal-Bench tournament in resumable shards.
11. Open X01-X06 only in response to measured remaining bottlenecks.

This ordering deliberately secures a robust score-completing route for every
arm before pursuing record speed. It gives TP1 and TP2 equal campaign standing:
the winner is the topology and recipe that scores highest and completes one Pi
agent stream robustly, not whichever topology looks best in a decode-only
microbenchmark. DP2 remains a possible post-selection deployment bonus for a
TP1 winner.
