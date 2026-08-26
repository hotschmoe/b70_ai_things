# Steve Qwen3.6 W8A8 Forensics And Reproduction Plan

Date: 2026-08-25

## Question

Why does Steve Seguin's Qwen3.6-35B-A3B Quark W8A8 TP=2 smoke report
85.87 output tok/s while the current local paths are much slower, and which
parts can be transferred to Ornith-1.5-35B-A3B W8A8?

## Bottom Line

Steve's result is not a stock-vLLM result plus one P2P environment variable.
It is a coordinated source, kernel, graph, collective, GDN, MoE, sampler, and
benchmark packet.

The first preserved custom-vLLM checkpoint after the record changes 84 files
and adds about 17,148 lines relative to its upstream base. The record was made
on 2026-06-15; the first source checkpoint is 2026-06-16. The durable result
packet also says that the live source tree was dirty and must not be assumed to
be a promoted patch set. Therefore the exact record source and binary identity
are not fully reconstructible from the public packet alone.

The accepted dirty binary is unavailable, but native source recovery is now
stronger than the original audit stated. Steve's live object database contains
exact June-16 checkpoint `122b698b`, parented directly on the public base, and
June-19 child `3ed399a`. The former precedes the accepted synchronized timing
packet and adds quantization output-buffer and scratch-aware native operators missing
from the local June-9 reconstruction. It is a controlled source target, not a
claim of byte identity with Steve's accepted binary.

The current pinned S2B image is reproducible, but it is a later snapshot. Three
source file hashes in the image exactly match the local detached vLLM source at
commit `44fc8fde0`, dated 2026-08-18. That snapshot includes later speculative
decode and exactness work. It is not the 2026-06-15 record environment.

Direct P2P is not a sufficient explanation for the gap. Our custom push
all-reduce measured 34.5-40 us for decode-sized messages, versus about 85-90 us
for oneCCL with direct P2P, and reaches comparable large-message bandwidth.
The remaining leverage is keeping the complete decode step, including
collective handoffs, inside a low-overhead graph/runtime path.

## Compared Identities

| Lane | Model/runtime | Parallel path | Result | Validity |
| --- | --- | --- | ---: | --- |
| Steve TP=2 smoke | Qwen3.6 35B-A3B Quark W8A8, custom vLLM/XPU | TP=2, PIECEWISE forced-comm graph, P2P=1 | 85.87 tok/s | JSON 16/16, color 16/16, quality skipped |
| Local true-June source control | Exact Qwen checkpoint, June `e190923b` source, rebuilt June package, recovered scratch ABI | TP=2, 9/9 PIECEWISE graphs, P2P=1 | 48.53 tok/s | native dense/MoE, exact metric, both canaries 16/16, both post-health layers green |
| Local August-adapter native control | Exact Qwen checkpoint, rebuilt June package plus narrow August-source repairs | TP=2, 9/9 PIECEWISE graphs, P2P=1 | 45.36 tok/s | native dense/MoE, coherent, both canaries 16/16, both post-health layers green |
| Local exact-package control | Exact Qwen checkpoint, rebuilt June package plus narrow August-source repairs | TP=2, 9/9 PIECEWISE graphs, P2P=1 | 47.54 tok/s | coherent and both canaries 16/16; rejected because August Quark still dispatched routed MoE through Triton |
| Local matched S2B control | Exact Qwen checkpoint, later S2B image plus narrow Qwen repairs | TP=2, PIECEWISE with split collectives, P2P=0 | 17.06 tok/s | coherent semantic canaries; exact Steve-shaped p498/o512 request |
| Local v0.24 control | Same HF model family and Quark format, vLLM 0.24 shelf | TP=2, PIECEWISE, P2P=0 | 22.96 tok/s | coherent eight-request random sweep; not Steve's one-request metric |
| Local S2B/Qwen trial | Later Steve S2B image plus the Ornith compatibility shim | TP=2, split collectives, P2P=0 | 12.72 tok/s | invalid: corrupted output after the first word |
| Local Ornith PP baseline | Ornith W8A8 RTN, later Steve S2B image | PP=2, graph, no MTP | 29.78 c1; 41.15 c2 aggregate | coherent |
| Local Ornith PP+MTP | Same target, experimental PP/MTP compatibility patches | PP=2, eager, MTP1 | not benchmarked | invalid: mechanically nondegenerate but semantically corrupted |

The first matched local control used Steve's exact metric shape: a natural-chat
prompt requested at 512 tokens and tokenized to 498, one 64-token warmup, then
one streaming 512-token measured request with EOS ignored. It reported 624.29
ms client TTFT, 30.010 s corrected decode time, and 17.06 corrected output
tok/s. Steve's 85.87 result decoded in 5.963 s. This isolates a 5.0x
decode-step gap after model identity, native INT8 selection, graph startup,
and basic coherence are controlled.

The local v0.24 sweep sent eight sequential 512-output requests and is retained
as a historical coherent reference, not a promotable matched A/B ratio.

The first full exact-package endpoint then captured all nine graphs and reached
47.5448 corrected tok/s with 336.710 ms TTFT and 10.7657 seconds of server
decode. This is 2.788x the 17.06 control and 55.37% of Steve's result, but it
is not a native-MoE result. The pinned image's Quark method unconditionally
called Triton `fused_experts`; mounting the rebuilt June package registered the
grouped operator without making that dispatcher select it.

The repaired August adapter then selected native routed MoE and closed the
compiled collective boundary at 45.3649 tok/s. The closest surviving June
source subsequently ran as a full overlay and reached 48.5315 tok/s with all
the same correctness and health gates. Its 12-component contract pins graph,
collective, GDN, MoE, scheduler, sampler, runner, and kernel-interface source
origins. The June source requires a scratch-aware fused-MoE interface absent
from the June-9 minimal package; the recovered interface at kernel commit
`2dd55f38` closes that ABI seam. Full June source is therefore a measured
6.98 percent gain, not the remaining 1.7693x explanation.

## Steve's Accepted Lever Ladder

The following rows come from Steve's tracked 2026-06-15 ablation summaries.
They are most useful as directional attribution because some adjacent rows
change more than one setting.

| Configuration | Corrected output tok/s | Reading |
| --- | ---: | --- |
| Conservative baseline | 12.83 | No usable XPU graph stack; native GDN prefill+decode fallback |
| Fast conservative | 76.48 | PIECEWISE graph, forced graph with comm, prefill replay disabled, greedy top-k fallback |
| Async scheduling | 76.64 | Async alone was noise-level here |
| Fast conservative plus INT8 mixed workspace | 74.82 | Mixed workspace alone was not a decode win in this row |
| Custom collectives disabled | 70.79 | About 7.4% below the 76.48 fast control |
| Prefill-safe plus mixed workspace | 93.56 | Decode stays on the graph-compatible GDN path; native fallback is prefill-only |
| Same plus async | 95.02 smoke; 93.55 deep gate | Async contributes at most a small final increment |
| Same TP=2 | 85.87 | The target control for this host |

The largest demonstrated lever was not raw MoE kernel speed. It was making the
decode execution graph usable and correct. The next largest visible change was
avoiding native GDN fallback during decode. Custom collectives mattered, but
were a single-digit percentage lever in this accepted ladder.

## Component Census

### 1. PIECEWISE graph and communication

Active record settings:

```text
XPU_GRAPH=1
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
COMPILATION_CONFIG={"cudagraph_mode":"PIECEWISE"}
```

The 2026-06 source used `GRAPH_NOOP_COMM_CAPTURE` in the distributed path. The
current 2026-08 S2B source no longer contains that knob. Copying the old
environment variable into the current image does not recreate the old graph.

The current P2P-off Ornith launcher explicitly splits at all-reduce and
all-gather. This is coherent, but returns through many graph boundaries. The
Ornith TP trace observed 84 all-reduces per verify step; earlier dense paths
observed up to 128 collectives per token. Even a 35 us collective kernel is not
an 80 tok/s solution if every boundary also incurs host/runtime scheduling.

### 2. Custom collective wrappers

Record-era settings included custom-op collectives and graph-safe input clones.
The later current source still implements:

- XPU custom-op collective selection;
- a compile-time `vllm::all_reduce` custom op;
- a graph-owned cloned handoff buffer;
- a graph-safe in-place all-reduce option added later.

Steve's tracked no-custom-collective ablation lost about 7.4%. Our custom push
all-reduce already beats oneCCL for small messages and matches its useful bulk
bandwidth. The transfer target is therefore not a new peer-copy primitive. It
is a push collective integrated into the same graph-owned handoff contract,
without splitting the decode graph at every collective.

### 3. Quark W8A8 INT8 linear path

Steve's later source has a real Quark INT8 scheme that calls the XPU INT8
linear kernel registry. The accepted checkpoint keeps weights INT8 and applies
dynamic activation quantization. The local v0.24 Qwen shelf instead dequantizes
minority linears to BF16 and keeps routed experts on true INT8. That shelf is a
coherent compatibility path, not a faithful copy of Steve's math path.

The Ornith compatibility shim installs its own native INT8 registry adapter.
Applying that shim to Qwen produced corrupted output. A clean Qwen control must
not import any Ornith monkeypatch.

### 4. INT8 fused MoE and mixed workspace

Steve's source calls the XPU fused-MoE kernel for INT8 weights and dynamic INT8
activations. `VLLM_XPU_INT8_MOE_MIXED_WORKSPACE=1` changes workspace handling,
but the tracked matched row did not show a standalone decode win. Keep it in
the reproduction identity because it was part of the accepted packet; do not
claim it is the headline speed source.

The first local exact-package endpoint exposed a second routing regression in
the later image. Its Quark source SHA256 `7e4c13d2...` contains the XPU INT8
oracle and experts implementation elsewhere in the tree, but
`QuarkW8A8Int8MoEMethod.apply` always calls generic Triton `fused_experts`.
The grouped schema in the startup log therefore established registration, not
execution. The local adapter now restores XPU backend selection, weight/scale
layout conversion, modular-kernel construction, and native apply. A no-device
contract proves the repaired August ABI before the next guarded model run.

Ornith currently reports missing tuned B70 `E=256,N=256` MoE configurations
and falls back to generic Triton tuning. This is a real third-order lever after
graph/runtime correctness, not before it.

### 5. GDN path

The accepted packet uses native recurrent GDN only for prefill and disables
prefill graph replay. Decode stays on the graph-compatible GDN path. Moving
from native-decode fallback into this configuration is associated with the
76 tok/s to 93-95 tok/s class jump in Steve's ladder.

This is highly transferable to Ornith because both resolve to the Qwen3.5 MoE
hybrid architecture. It must be validated with exact repeated outputs because
GDN state and graph replay errors frequently preserve speed while corrupting
the first token or later state.

### 6. Greedy sampling path

The record uses a top-k fallback for greedy sampling. Later source replaces
expensive full-vocabulary operations with bounded/masked max variants. These
are real sub-millisecond improvements, but the accepted June ladder does not
support treating sampling as the 3-4x gap.

The fast path is greedy-specific. Pi/Terminal-Bench product evaluation may use
sampling, tool calling, or logprobs, so a greedy-only benchmark record cannot
be substituted for the final serving configuration without an explicit policy
decision.

### 7. Async scheduling

Steve's matched rows show 76.48 versus 76.64 tok/s before the final lane and
about 93.56 versus 95.02 in the later smoke. It is useful but small. It is not
the missing 60 tok/s.

### 8. P2P and PCIe topology

Steve enables `CCL_TOPO_P2P_ACCESS=1`. His current reference host is PCIe 4.0
x16 class, while this Threadripper 1950X host exposes PCIe 3.0. However Steve's
own controlled Qwen TP=2 Gen4/Gen3 experiment kept decode inside the Gen4
control envelope while bulk all-reduce bandwidth changed. This rules out CPU
generation or Gen3 x16 as the primary multi-x performance explanation.

Kernel 7.1 cured this host's GuC/BCS firmware-skew hardware wedge. It did not
prove that the separate oneCCL/vLLM multiprocess P2P initialization wedge was
cured. A one-shot 7.1 retest harness already exists at
`vllm/w8a8/sweep_p2p_71_retest.sh`. It requires an operator-approved risk
window and a reboot contingency. It should be run only after the P2P-off source
control is exact and coherent.

### 9. MTP

Steve's valid 85.87 TP=2 result is target-only. His Qwen3.6 speculative work
did not produce a valid greater-than-150 tok/s result; fast MTP rows failed
JSON/color gates or graph-state correctness. Therefore MTP is not a prerequisite
for reproducing 80+ tok/s.

The current Ornith PP+MTP issue is separate: initialization and GPU kernels now
work, but speculative token/hidden-state propagation across PP corrupts the
sequence. Do not mix that correctness problem into the target-only Steve
baseline transaction.

## Reproduction Transactions

Run these in order. Do not combine transactions until each has a coherent
endpoint and an exact identity artifact.

### R0. Clean Qwen S2B control

Status: complete for the coherent P2P-off split-collective graph arm. Artifact:
`/mnt/vm_8tb/b70/results/logs/qwen36_s2b_p2p0_steve_metric_20260825T030225Z.json`.

- Add a Qwen-only launcher for image digest
  `f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94`.
- Do not mount the Ornith compatibility shim.
- Record hashes for vLLM source files and every loaded XPU kernel SO.
- Use the exact HF revision `cced56592e8c8935f8220836b4baa04dfd389118`.
- Start with P2P=0 and no MTP.
- Use Steve's metric helper: natural-chat prompt, 512 prompt tokens, 64-token
  warmup, one 512-token measured request, ignore EOS.
- Run repeated JSON and color canaries before interpreting speed.

### R1. Graph boundary attribution

Compare coherent R0 variants with device and host step traces:

1. PIECEWISE with all-reduce/all-gather split boundaries.
2. Graph-owned custom-op all-reduce handoff using stable cloned inputs.
3. Capturable push all-reduce with the collective inside replay and no
   per-layer split.

Count graph pieces, graph replays, eager boundaries, collective calls, host
wall per step, and device time. Kernel-only microbench latency is insufficient.

### R2. Accepted math-path attribution

Starting from the fastest coherent R1 arm, test one factor at a time:

1. true INT8 XPU linears versus dequantized BF16 compatibility linears;
2. graph-compatible GDN decode versus native decode fallback;
3. mixed MoE workspace off/on;
4. tuned B70 MoE config versus generic config;
5. greedy sampler fallback variants;
6. async scheduling off/on.

Use position-balanced A-B-B-A for any shelf or campaign claim.

### R3. Kernel 7.1 P2P retest

Only after R0-R2 are coherent:

- one P2P=1 vLLM start, not a chain of retries;
- health before, after startup, after generation, and after teardown;
- immediate stop on worker-init failure;
- reboot if post-health fails;
- compare graph structure and end-to-end latency, not only all-reduce GB/s.

If P2P=1 works but capturable push is equal or faster with the same graph
structure, keep push as the safer production path.

### R4. Transfer to Ornith

Port only the R1/R2 pieces that won on the clean Qwen control. Keep Ornith's
model-specific differences explicit:

- on-box RTN W8A8 rather than Steve's Quark checkpoint calibration;
- trained Shisa MTP sidecar;
- `E=256,N=256` MoE tuning requirement;
- Qwen/Ornith output-equivalence cannot be assumed;
- target-only 80+ comes before MTP and PP+MTP work.

## Decision

The highest-value effort is graph/runtime integration, followed by the GDN
decode path and complete native INT8 route. Raw P2P bandwidth, async scheduling,
and isolated MoE workspace tuning are lower-order. PP=2 remains a useful safe
baseline and concurrency lever, but it is not the direct route to Steve's
target-only 85.87 TP=2 result.

The direct-P2P oneCCL, clone-correct custom op, and graph-capture repairs now
pass in the full model. The first coherent exact-package endpoint loaded both
ranks, captured all nine general graphs, passed the exact 498/512 metric and
both 16/16 canaries, and tore down healthy at 47.5448 tok/s. Its strict route
gate rejected the run because request-time Triton `fused_moe_kernel` proved
that the later Quark dispatcher still bypassed the registered June grouped
operator.

The native Quark repair, closest surviving June vLLM source transaction, and
instrumented decode census are now complete. Replay tracing reports the same
41 graph pieces as Steve. Under his synchronized pure-decode protocol, local
rank-0 model-forward is 22.6748 ms versus 5.6946 ms, a 3.9818x execution gap.
The exact native sibling swap from checkpoint `122b698b` is now measured at
50.3706 tok/s versus the matched June-9 endpoint at 48.5315 tok/s (+3.79
percent). Both 16/16 canaries, graph capture, per-card health, and compiled
collective post-health passed. Its output-buffer operators are consumed
conditionally by the already active scratch-aware dispatcher and are absent
from the June-9 package. The gain is real but does not explain the missing
1.7x; synchronized native-checkpoint timing and integrated collective cost are
next. Do not compare this endpoint with the deliberately synchronized 35.4699
tok/s diagnostic.
The accepted path performs two scratch-targeted quantizations in each of 40
MoE layers, or 80 calls/step; its fused SiLU+quant switch is explicitly unset.
Transfer proven graph/runtime changes to dense 27B one factor at a time;
census that model's graph pieces and profile collective shapes, derive its own
clone-fence threshold, and exclude MoE-only layerlet/sidecar conclusions.
