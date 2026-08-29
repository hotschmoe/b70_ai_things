# Neural.Download and XeCores deep dive plus campaign state

Date: 2026-08-29

Status: evidence ledger and handoff document. No result in this document is a
shelf promotion unless the referenced local record already says so.

## Purpose

This document records the state of the local Intel Arc Pro B70 serving and
Terminal-Bench campaign after a deep audit of Neural.Download, XeCores, their
linked repositories, recipe books, raw result records, repro packets, and the
local retained evidence. It is intended to survive a closed terminal or a new
Codex session without requiring the literature audit to be repeated.

The companion execution plan is
`docs/20260829_local_serving_research_roadmap.md`.

## Audit method and source boundary

Three independent read-only audits were used:

1. Neural.Download was traversed through its site index, `llms.txt`, result
   pages, experiment notes, repro packets, raw JSON matrices, and the linked
   `steveseguin/b70-optimization-lab` source tree.
2. XeCores was traversed through its static pages, Cookbook, browser document
   corpus, vendored Intel GPU skills, and the linked
   `SergiioB/intel-arc-pro-b70-inference-cookbook` repository.
3. A cross-audit normalized TP=1, TP=2, graph, eager, KV dtype, MTP depth,
   prompt shape, concurrency, and evidence quality, then compared them with
   the local graph, collective, topology, recovery, and Terminal-Bench logs.

No GPU was touched and no external binary was imported during the audit.
Published values below remain the authors' measurements unless a paragraph
explicitly identifies a local reproduction.

Primary external indexes:

- https://neural.download/
- https://neural.download/llms.txt
- https://github.com/steveseguin/b70-optimization-lab
- https://xecores.com/cookbook.html
- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook

## Current local boundary

### Hardware and runtime discipline

- Host baseline: kernel 7.1.0-070100.
- Current host Compute Runtime: 26.22.
- GPUs: two Intel Arc Pro B70 cards on a Threadripper 1950X-era, split-root,
  PCIe Gen3 topology.
- Every GPU action must run through `bin/gpu-run`.
- TP>1 experiments require per-card and compiled two-rank collective health
  before and after the transaction.
- Production-safe TP=2 remains `CCL_TOPO_P2P_ACCESS=0`. The separate vLLM
  multiprocess/oneCCL queue-handoff failure is not cured by the newer kernel.
- A failed or crashed TP>1 attempt is followed by `bin/xe-reset` before another
  risky attempt. Repeated blind retries can poison later collectives.
- BF16 KV cache is the campaign default and must be verified from runtime
  evidence, not inferred from a served ID or launcher label.

The current backend boundary is SGLang for primary serving and new work, with
vLLM retained as a controlled baseline and transfer workspace. llama.cpp was
removed from the live tree on 2026-08-26. Useful llama.cpp mechanisms may be
ported deliberately from tracked source; archived ABI-specific binaries must
not be restored into the refreshed stack.

### Current four-arm Terminal-Bench campaign

| Arm | Current useful state | Rejected or unresolved state |
| --- | --- | --- |
| Qwen3.8 W8A8 compressed-tensors/GPTQ | SGLang TP2, P2P off, BF16 KV, target-only, breakable graph size 1, reclaim every 500 replays, memory fraction 0.70 survived the long Bun pilot through 26K live tokens | FULL graph faulted around 17K. The stable reclaim500 pilot timed out under the verbose Pi/xhigh policy and scored zero |
| Qwen3.8 RadixArk NVFP4 | SGLang TP2 short-context FULL plus FP8 W8A16 decode projections reached 32.6206 tok/s for p879/o512 | FULL graph aborted around 19K live tokens in Terminal-Bench. Breakable plus reclaim500 is not yet ported and qualified |
| Qwen3.8 GPTQ INT4 G128 with BF16 MTP checkpoint | vLLM TP1 has a strong short/C1 control and a draft LM-head INT4 overlay | Historical BF16 claims are invalid until the launcher is requalified with observed BF16 target and KV dtype. PIECEWISE aborted during the long agent run. FP8 KV failed repeat exactness. TP2 eager regressed in the previous stack |
| Ornith-1.5-35B-A3B W8A8 RTN plus Shisa/official MTP material | SGLang TP2, P2P off, target-only, BF16 KV, breakable size 1 plus reclaim500, memory fraction 0.70 survived the long pilot | MTP1 changed 7/8 target completions on the current SGLang path. The long pilot timed out under Pi/xhigh. Memory fraction 0.90 caused a host OOM that killed user systemd and the tmux session |

The 0.90 Ornith graph-capture attempt allocated about 58 GiB of shared GPU
memory and triggered the host OOM. This explains the closed tmux session; it
was not an unexplained Codex crash. Memory fraction 0.70 left approximately
9.67 GiB per card after graph capture and is the retained long-agent control.

### Terminal-Bench validity state

The campaign is currently blocked on harness validity and two graph-safe
runtime recipes. The authoritative relaunch contract is
`evals/terminalbench/CAMPAIGN_RELAUNCH.md`.

Two historical labels were invalidated on 2026-08-29:

1. `thinkingLevelMap` mapped `off` to JSON null. Pi 0.84.3 treated it as
   unsupported, clamped upward, and sent
   `chat_template_kwargs.enable_thinking=true`. The alleged true-off 4K result
   was native thinking with a hard 4,096-token response cap.
2. The retained GPTQ launcher hard-coded `--dtype float16` and left KV dtype
   on auto. Its runtime log reported FP16. Served IDs and lifecycle metadata
   that called it BF16 do not override observed runtime identity.

The H01-H07 harness gate passed on 2026-08-29. The exact Pi payload oracle,
observed model/dtype assertion, final stop-reason and activity capture,
endpoint-before-teardown health, complete pre-health-through-post-health
clock, and deterministic local-70 lock are automated. This closes harness
validity; it does not qualify the rejected NVFP4 FULL or GPTQ PIECEWISE routes.

Terminal-Bench 3.0.0 contains 74 tasks, but four require an H100 task
environment. Local B70 work is therefore a labeled `TB3-local-70` campaign;
an official 74-task result requires suitable remote workers.

## Neural.Download findings

### The central TP scaling result

Steve's cleanest topology experiment used Qwen3.8-27B AutoRound INT4 W4A16,
MTP0, one sequence, cache disabled, FP16 target/KV, 32K maximum length, and a
matched prompt suite in one vLLM nightly container.

| Execution | TP1 decode | TP2 decode | TP4 decode | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Eager | 23.7-24.2 | 16.77 | 17.38 | TP2 and TP4 are flat or worse |
| XPU graph | 30.2-30.3 | 48.8-49.0 | 71.6-71.9 | TP2 is about 1.62x TP1; TP4 about 2.36x TP1 |

Prefill scaled even in eager mode, approximately 281 -> 500 -> 860 input
tok/s from TP1 to TP2 to TP4. Large prefill GEMMs amortize launch and
collective overhead. Autoregressive decode repeatedly pays small-operation,
small-collective, fence, and host-submission latency. Graph capture removes
enough of that critical path for split weight bandwidth to become visible.

Sources:

- https://neural.download/experiments/qwen38-27b-b70/notes/2026-08-23-qwen38-tpscale-nightly-finding.html
- https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/data/2026-08-23-qwen38-tpscale-nightly-matrix.json

This is strong mechanism and speed evidence, not shelf evidence. Multi-GPU
XPU graph was experimental or unsupported in the tested stack. TP2 had only
19/25 cross-boot complete-output agreement and TP4 had 21/25. Cache sealing
did not remove the cross-boot difference.

### Current strict official-FP8 TP2 route

Steve's 2026-08-28 Qwen3.8 official-FP8 repro uses the same pinned `f01e24f6`
vLLM image family already present in the local campaign. It deliberately runs
XPU graph off and instead makes compiled state and collective completion
explicit.

| Mode | TP2 result | Qualification |
| --- | ---: | --- |
| Target-only MTP0 | 34.0316 tok/s | Strict cache-zero 12-prompt, two-fresh-server route |
| Publisher MTP1 | 51.9188 tok/s | +52.6 percent over MTP0, exact target agreement 12/12 |

At exact 32K, the MTP1 route reported 46.636 tok/s, 10.487 second TTFT, and
78.32 percent acceptance. Its key repairs include:

- asynchronous `dist.all_reduce` followed by explicit `Work.wait()`;
- compiler-visible recurrent convolution and SSM cache mutations;
- binding compiler-visible state to allocated cache tensors;
- deterministic small GDN prefill reduction handling;
- a fixed two-row RMSNorm replay path for MTP1.

Source: https://neural.download/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/

The lesson is not that graph should always be on. The general requirement is
to remove CPU-visible boundaries while preserving explicit collective
completion and recurrent-state ownership. Graph capture and disciplined
compiled execution are two possible implementations.

The strict route uses FP16 KV and direct P2P settings. Its speed is a control,
not a BF16 or local-topology qualification. The local queue-handoff failure
also prohibits copying its P2P setting into a full-model serve.

### llama.cpp TP scaling and fusion

Steve's optimized llama.cpp routes show that B70 hardware can scale strongly
when communication boundaries and their immediate consumers are fused.

| Quant | TP1 | TP2 | Scaling |
| --- | ---: | ---: | ---: |
| Qwen3.8 Q4_K_M | 27.826 | 49.718 | 1.79x |
| Qwen3.8 Q8_0 | 19.619 | 36.726 | 1.87x |

Q4_K_M retained about 1.81x scaling at 32K. The implementation uses
asynchronous copies, an exact root-kernel reduction, peer-visible
residual/RMS/Q8 handoffs, Q8 producer fusion, direct KV writes, and attention,
GDN, and recurrent fusion. These are backend-specific source mechanisms, not
evidence that stock oneCCL should scale the same way.

Sources:

- https://neural.download/repro/qwen38-27b-q4km-tp1-b70/
- https://neural.download/repro/qwen38-27b-q4km-tp2-asrock-b70/
- https://neural.download/repro/qwen36-27b-q8-tp2-asrock-b70/

The TP1 and TP2 patch sets and some complete output arrays differ. Treat the
numbers as strong directional evidence. Any transfer into this repository
must be a deliberate source port into the refreshed backend or a newly tracked
research tree, never restoration of quarantined binaries.

### P2P and PCIe conclusions

P2P is conditional rather than magical. In one official-FP8 graph experiment,
enabling P2P changed throughput by approximately zero. Steve's current fast
route uses P2P, but it also carries explicit completion and state repairs, so
the P2P effect is not isolated.

Steve's controlled PCIe downgrade reduced large-message collective bandwidth
without materially moving decode intervals. Qwen decode uses many small
reductions, so latency, synchronization, root selection, and submission count
matter more than peak bulk GB/s. Source:
https://neural.download/docs/pcie-topology-and-llm-inference.html

This matches the local record. The local Qwen3.6 PIECEWISE path executed 41
graph pieces, 41 fence resets, 41 host synchronizations, 82 submissions, and
81 all-reduces per token. FULL decode reduced that to one fence, two waits,
and two submissions and improved 50.3706 -> 61.5536 tok/s. Triton MoE reached
64.9843, and source-default c10d reached 66.3438. Capture coverage and host
boundaries were first-order; a slightly different collective was secondary.

### Dense versus sparse topology

Dense Qwen benefits from halving weight reads when orchestration is controlled.
Ornith's sparse active expert set already fits within one B70's useful working
set. Steve measured Ornith Q4_K_M at approximately 104.82 tok/s TP1 and 102.10
TP2. TP2 is therefore a capacity mechanism for Ornith, not a default decode
optimization. Source:
https://neural.download/repro/ornith-15-35b-a3b-q4km-b70/

### Evidence-quality corrections

Neural.Download's 2026-08-27 integrity audit warns that several older headline
rates used selected prompts, short caps, one restart, or older timing formulas.
The newer six-class cache-zero and fresh-server packets supersede those claims.
The historical Qwen3.6 AutoRound 95.385 tok/s transaction remains valuable as
a mechanism record, but later cross-restart equality did not meet the newer
standard. Source:
https://neural.download/docs/benchmark-integrity-audit-20260827.html

## XeCores findings

### Important correction: no measured tensor-parallel TP2

Every measured XeCores LLM recipe is TP1 on one B70. `MTP1`, `MTP2`, and
`MTP4` mean speculative depth, not tensor parallel size. The site's generic
Intel multi-GPU skills describe launch setup but provide no XeCores TP scaling,
P2P, queue-handoff, or long-run stability evidence.

The useful role of XeCores is as a set of TP1 recipe, quantization, MTP, graph,
cache, scheduler, and power controls.

### Normalized measured recipes

| Family | Principal recipe | Main result and lesson |
| --- | --- | --- |
| Qwen3.6-35B-A3B | GPTQ INT4, FP16 target/KV, graphs, prefix cache, MTP0/1/2/4 | p512/g128 rises 96.79 -> 170.91 tok/s through MTP4; exact 128K is won by MTP2 because deep acceptance falls |
| Qwen3.6-27B | GPTQ INT4, FP16 target, FP8 KV, graphs | p512/g128 target 32.85, MTP4 69.30; FP8 KV does not meet the local BF16 policy |
| Qwen3.8-27B | GPTQ INT4 target, FP16 target, FP8 KV, BF16 MTP tensors, graphs, MTP4 | complete draft INT4 S+M1 changes 81.20 -> 112.65 tok/s at p512/g128 |
| Ornith-1.5-35B-A3B | experts-only MixedCal-v2 GPTQ INT4, FP16 target/KV, graph, MTP1 | target about 70.74, BF16 draft MTP1 96.43, draft INT4 MTP1 106.27; MTP4 66.27 and loses to target-only |
| Nemotron-3.5-Lightning | GPTQ INT4 target plus BF16 DFlash draft | graph changes eager 21.8 -> about 90 tok/s; DFlash wins while native MTP does not |
| Muse-Glimmer | llama.cpp Q4_K_XL, mixed KV quantization, DFlash2 | demonstrates model-specific draft depth and context-dependent gain |

Primary recipes:

- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/QWEN38-VLLM-XPU.md
- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md
- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen36-27/QWEN36-DENSE-VLLM-XPU.md
- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/ornith15-35a3/ORNITH-VLLM-XPU.md

### Complete draft INT4 S+M1

XeCores' Qwen3.8 gain combines two independent overlays:

1. `B70_DRAFT_LMHEAD_INT4=1` quantizes the draft LM head.
2. `B70_DRAFT_MTP_INT4=1` quantizes five MTP linears.

The matched p512/g128 result improved 81.20 -> 112.65 tok/s, +38.7 percent,
while reported acceptance fell about 1.4 percentage points. At p8192/g128 the
result improved 77.52 -> 103.63. The published quality A/B covered only 15
tasks, so it is a source hypothesis rather than a local promotion.

Source:
https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/DRAFT-INT4-S-M1.md

The local vLLM 0.28 attempt to quantize the five MTP linears changed three of
eight target completions and was correctly rejected. That does not close the
idea permanently: the published recipe uses a different exact stack and both
S and M1 overlays. A new test must begin after the corrected BF16 target route
is established and must fail closed on target divergence.

### Prefix reuse and total task time

XeCores' single-user Pi trace measured a 32,640-token cold read at 38.2 seconds
TTFT, followed by growing turns around 2.6-4.9 seconds with approximately
90-95 percent prefix reuse. At concurrency five, reuse collapsed to 0-38
percent. Terminal-Bench repeatedly extends one tool conversation, so C1 prefix
reuse may improve total completion time more than a small decode kernel gain.

Source:
https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/REAL-WORLD-PI-BENCHMARKS.md

The local SGLang refresh launchers disable radix cache and the GPTQ launcher
disables vLLM prefix caching. Cache qualification is therefore a major open
lane, but it must cover hybrid GDN state, graph reclaim, exact output, and
long replay rather than being enabled blindly.

### True Pi thinking-off mapping

XeCores documents a concrete Pi payload route:

```text
thinkingLevelMap: off -> none
chat_template_kwargs.enable_thinking = false
chat_template_kwargs.preserve_thinking = true
delete reasoning_effort
```

The exact representation used to make `off` supported must agree with Pi
0.84.3's metadata rules; the decisive gate is the emitted provider payload,
not the label. Source:
https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/PI-AGENT-BACKEND.md

### Model-specific speculative depth

There is no universal MTP depth:

- Qwen3.8 dense often favors MTP4.
- Qwen3.6 MoE favors MTP4 short and MTP2 at exact 128K.
- Ornith favors MTP1; MTP4 can be slower than target-only.
- Nemotron's native MTP is ineffective while DFlash works.
- Prompt content can move draft acceptance from about 94 percent to the
  mid-40s.

Every local MTP decision must therefore use target-exact verification and
Terminal-Bench/Pi-shaped prompts with proposed/accepted counters. Synthetic
repetition is not sufficient.

### Graphs, concurrency, scheduler size, and power

- XeCores' Nemotron graph result, about 21.8 -> 90 tok/s, reinforces that CPU
  enqueue and launch count can dominate decode.
- Its Qwen concurrency results show aggregate throughput rising while MTP
  acceptance falls. Aggregate tok/s is not a topology metric unless prompt,
  output, concurrency, and acceptance are matched.
- `max_num_batched_tokens=8192` and 16384 trade wins across traces. The actual
  growing agent trace must decide.
- Ornith's decode moved little between 150 W and 230 W, while cold prefill rose
  from about 7.1K to 9.7K input tok/s. A prefill-heavy agent may therefore
  favor a different power cap than a decode-only benchmark.

## Combined causal model

The literature and local evidence support this priority order:

1. Graph or compiled coverage and the number of CPU-visible boundaries.
2. Explicit collective completion and recurrent-state ownership.
3. Transport latency and topology after capture overhead is controlled.
4. Backend-specific fusion of each small communication boundary and consumer.
5. Dense versus sparse model structure and quantized weight traffic.
6. MTP draft communication, acceptance, and model-specific depth.
7. Prefix reuse and exact graph/cache shape for long agent sessions.
8. Scheduler and power policy under the actual prompt trace.
9. oneCCL algorithm micro-tuning and CPU affinity.

This explains the apparent contradiction. Steve's eager TP2 is flat just like
ours. His graph or heavily fused routes expose scaling that eager orchestration
hides. Our older topology and P2P-off safety boundary add residual cost, but
the local 41-piece profile proves that host/runtime boundaries are still the
first problem to remove.

## What transfers and what does not

Transfer candidates:

- explicit asynchronous collective completion oracles;
- compiler-visible recurrent-state mutation and cache binding;
- minimum graph-safe compiled regions rather than one assumed graph mode;
- consumer fusion around small reductions;
- complete draft S+M1 as a fail-closed experiment;
- C1 prefix reuse qualification;
- model-specific MTP depth screening;
- true thinking-off payload construction;
- honest total wall-time and concurrency accounting.

Do not transfer without a new local qualification:

- FP16 or FP8 KV performance claims into the BF16 campaign;
- `CCL_TOPO_P2P_ACCESS=1` full-model launch settings;
- multi-GPU graph stability claims;
- old selected-prompt headline rates;
- llama.cpp binaries or ABI-specific packages from the archive;
- FULL or forced-chunk attention recipes into 32K-65K agent service;
- Ornith MTP4 or a dense-Qwen TP scaling expectation;
- aggregate concurrency throughput as single-user completion speed.

## Current verdict

The campaign should not resume as four long Terminal-Bench jobs yet. The
highest-value next work is to repair the harness and observed-dtype evidence,
then qualify each runtime through short target-only, graph/cache, long-context,
and teardown gates.

The best current working hypothesis for local serving is:

```text
BF16 KV
+ verified thinking-off
+ C1 prefix reuse
+ target-exact model-specific MTP
+ complete draft quantization only when exact
+ graph or compiled execution with the fewest safe host-visible boundaries
+ P2P-off production safety
```

Terminal-Bench should rank final candidates by score and total machine time to
completion. Server decode, prefill, TTFT, acceptance, cache reuse, errors,
restarts, and unfinished tasks remain separate explanatory measurements.

### Single-stream objective and deferred DP2 benefit

Both TP1 and TP2 are active candidates for one objective: find the recipe that
produces the highest Terminal-Bench 3.0.0 score on a single Pi agent stream
while remaining fast enough and robust enough to finish its tasks. TP2 is not
assumed to win because it uses both cards, and TP1 is not merely a diagnostic
baseline. Model, quant, backend, graph/compiled mode, cache policy, MTP, draft
quantization, and topology all remain open recipe variables.

The campaign ranks successful single-stream recipes primarily by
Terminal-Bench score and normal completion, with total task and machine time
used to distinguish the speed of viable high-scoring recipes. TTFT, prefill,
decode, cache reuse, acceptance, errors, and restarts explain the result but do
not replace the task score.

If the eventual winning recipe fits on one B70, two independent TP1 replicas
could later provide a DP2 concurrency benefit for local users. That is a
deployment follow-up, not an objective or selection criterion for the next
testing campaign.
