# Qwen3.6 Exact-Control Graph Runtime Profile

Date: 2026-08-26

## Purpose

Localize the residual gap between the coherent exact Qwen3.6 35B-A3B Quark
W8A8 TP=2 control and Steve Seguin's accepted TP=2 result after model identity,
June vLLM source, June-16 native operators, graph-piece count, and collective
correctness were closed.

The clean local endpoint is 50.370643 corrected output tok/s. Steve's accepted
endpoint is 85.869114 tok/s. Synchronized rank-0 model-forward is 21.994441 ms
locally and 5.694625 ms in Steve's packet. Both systems report 41 PIECEWISE
graph pieces and 81 compiled TP all-reduces per decode step.

## Bounded XPU Profile

CONFIG -> exact model revision `cced5659`, image digest `f2e5a94e`, June vLLM
source `e190923b`, June-16 native checkpoint `122b698b`, TP=2, direct P2P,
PIECEWISE graph, no MTP, no prefix cache, eight profiled decode iterations per
rank after a two-iteration delay. The profile ran as a separate request before
the ordinary metric and both repeat canaries.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint XPU_PROFILE=1 P2P_ACCESS=1 \
  I_KNOW_P2P_WEDGES=1 ALLOW_EXISTING_CACHE=1 \
  CACHE_DIR=/mnt/vm_8tb/b70/vllm_cache_qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_20260826T013300Z_june122_synctiming \
  STAMP=20260826T020000Z_june122_xpu_profile \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> both ranks recorded eight scheduler iterations. Each rank showed the
same per-token graph replay signature:

| Driver operation | Median calls per token |
| --- | ---: |
| `zeFenceReset` | 41 |
| `zeEventHostSynchronize` | 41 |
| `zeCommandQueueExecuteCommandLists` | 82 |
| `zeCommandListAppendBarrier` | 123 |
| `zeCommandListAppendLaunchKernel` | 105 |
| `urEnqueueKernelLaunch` | 105 |

This is direct structural evidence that the local runtime crosses all 41
piece boundaries every token. It is not evidence that the graph is absent.

Kineto exposed only 1.671066 ms/rank0 and 2.170872 ms/rank1 of device work per
iteration. The visible rank-0 buckets were 0.960325 ms GEMM, 0.367745 ms GDN,
0.222418 ms full attention, and 0.026757 ms collective. Rank 1 was similar
except for one visible final all-gather at 0.525390 ms. The routed MoE kernels
and 81 compiled all-reduces inside XPUGraph replay were not emitted as
individual device events. Visible device time is therefore incomplete and
must not be subtracted from full synchronized model-forward time.

The profiler perturbed the runtime. The profiled request reached 50.628099
tok/s, but the ordinary request after profiling fell to 38.389977 tok/s and
profiled scheduler iteration CPU ranges were about 33.4 ms. Profiler-enabled
endpoint speed and CPU spans are diagnostic only. The clean 50.370643 tok/s
endpoint remains the comparison anchor.

VERDICT -> graph-piece topology is closed, but captured device ownership is
still opaque. The leading residual is integrated XPUGraph, collective, and
host coordination across 41 replay boundaries, not any visible attention or
GDN kernel family. A next intervention should change one boundary mechanism
at a time and use the clean endpoint plus canaries and post-health as its gate.

## Threadripper Split-Die Affinity A/B

CONFIG -> the same clean exact control, with rank 0 restricted to logical CPUs
`0-7,16-23` and rank 1 to `8-15,24-31`. Both cards remain in their current
slots. The 1950X exposes one NUMA memory node, so both workers retain memory
node 0.

The June `--numa-bind` implementation initially bound EngineCore to rank 0's
CPU mask before it spawned both TP workers. Linux then prevented rank 1 from
expanding outside the parent mask, and `numactl` rejected the second CPU list.
The exact-control adapter now leaves EngineCore unbound and applies the two
lists only to worker subprocesses.

RESULT -> the corrected run loaded and captured in 101 seconds, bound worker 0
and worker 1 to the intended lists, passed the semantic probe and both 16/16
canaries, and left per-card plus compiled-collective health green. It measured
50.406626 tok/s, 306.627 ms client TTFT, and 10.155217 seconds server decode.
The clean unbound endpoint measured 50.370643 tok/s, 307.853 ms, and 10.163436
seconds.

VERDICT -> split-die worker affinity changes throughput by only +0.07 percent.
It is not the missing 1.7x lever. Keep the worker-only adapter available for
controlled topology work, but do not make affinity a production default and
do not move a card merely to repair CPU scheduling. A physical slot move is a
separate topology A/B and should be justified by a specific collective or
bandwidth hypothesis.

## Full-Decode Boundary Intervention

CONFIG -> the same exact June source and June-16 native control, with
`cudagraph_mode=FULL_DECODE_ONLY`, `TRITON_ATTN`, no MTP, no prefix cache, and
a fresh compile cache. Mixed prefill remains outside full capture. Six decode
FULL graphs were captured in 26 seconds after model compilation and used only
0.11 GiB of graph memory.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint CGMODE=FULL_DECODE_ONLY \
  ATTN=TRITON_ATTN P2P_ACCESS=1 I_KNOW_P2P_WEDGES=1 \
  STALL_TIMEOUT=600 STAMP=20260826T024000Z_full_decode_triton \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> server health completed in 259 seconds. The p498/o512 metric measured
61.553562 corrected output tok/s, 332.269 ms client TTFT, and 8.317629 seconds
server decode. The exact PIECEWISE control measured 50.370643 tok/s, 307.853
ms, and 10.163436 seconds. Full decode therefore gains 11.182920 tok/s, or
22.20 percent, while adding 24.416 ms TTFT in this one run. The semantic probe,
JSON 16/16, color 16/16, graceful teardown, both card probes, and compiled
two-rank collective health all passed.

VERDICT -> reducing the decode replay boundary is the largest measured local
lever after graph capture itself. It recovers 22.20 percent but reaches only
71.68 percent of Steve's 85.869114 tok/s, leaving 24.315552 tok/s. The old
blanket statement that FULL capture is blocked on B70 is incorrect for this
exact no-MTP June stack. Stock v0.23 MTP full capture can still be blocked by
the GDN speculative shape or SYCL scratch paths; do not generalize this result
to MTP or another source/runtime without a fresh gate.

This arm is a local-speed intervention, not a reproduction of Steve's accepted
PIECEWISE command. PIECEWISE remains the provenance control. Next, profile the
full-decode arm to verify the expected replay/host-wait collapse, then isolate
the remaining 81-collective and native-MoE cost inside the single replay.

The bounded full-decode profile confirms that collapse. Per-token medians fall
from 41 to 1 fence reset and from 82 to 2 queue submissions. Host event waits
fall from 41 to 2. GDN, full attention, routed MoE, and the 81 all-reduces move
inside the opaque full graph; the remaining visible device work is only about
1.08 ms/rank and is not a full graph ledger.

Across six steady-state profiled iterations after skipping the first two,
rank 0 begins its longest wait 9.215851 ms into the iteration, waits 2.731938
ms for the preceding asynchronous graph tail, and has 2.163153 ms of host work
after the next graph submissions. Rank 1 values are 8.938722, 3.300870, and
1.962969 ms. Mean iteration ranges are 14.429410 and 14.517247 ms. The wait is
only the exposed tail after host input preparation overlaps the preceding
graph; it is not the full graph device duration.

The profile arm itself measured 61.543223 tok/s, and the ordinary request after
profiling measured 61.559842 tok/s. Unlike the PIECEWISE profiler process, this
full-decode process did not show a large endpoint slowdown. Endpoint timing is
still treated as diagnostic because profiling was enabled, but the repeat
agrees with the clean 61.553562 result to 0.01 percent. The remaining target is
now explicitly in-graph MoE and collective execution, not replay-piece count.

## MoE Backend Attribution

SOURCE -> June e190 exposes `--moe-backend triton`, but
`TritonExperts._supports_quant_scheme` admits INT8 only when the platform is
CUDA. The underlying Triton expert implementation accepts W8A8 arguments and
had already run on XPU through the older generic route. The adapter therefore
relaxes only the exact `(kInt8StaticChannelSym, kInt8DynamicTokenSym)` pair
when `B70_QWEN36_INT8_MOE_TRITON_INTERVENTION=1`. It prints an intervention
marker, changes no other support decision, and leaves the default native route
intact.

CONFIG -> exact June source, June122 native package, TP=2 direct P2P,
`FULL_DECODE_ONLY`, `TRITON_ATTN`, no MTP or prefix cache, explicit
`--moe-backend triton`, and a fresh isolated compile cache. The native package
remains mounted for dense W8A8, GDN, and collective identity; routed experts
use Triton W8A8.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint CGMODE=FULL_DECODE_ONLY \
  ATTN=TRITON_ATTN MOE_BACKEND=triton P2P_ACCESS=1 \
  I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=900 \
  STAMP=20260826T031500Z_full_decode_triton_moe_retry \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> both workers logged the labeled intervention and selected `Using
TRITON Int8 MoE backend`. Model loading, all 81 profile clone fences per rank,
and all six FULL decode captures completed. Capture took 38 seconds and 0.11
GiB. The p498/o512 metric measured 64.984330 corrected output tok/s, 363.490 ms
client TTFT, and 7.878107 seconds server decode. The matched native-MoE full
control measured 61.553562 tok/s, 332.269 ms, and 8.317629 seconds. Triton gains
3.430768 tok/s, or 5.57 percent, and saves 0.439522 seconds of decode while
adding 31.221 ms TTFT in this one comparison. Semantic output, JSON 16/16,
color 16/16, graceful teardown, both card probes, and compiled two-rank
collective health all passed.

The first launch did not enable the intervention because an over-escaped shell
test emitted `0`. Both workers rejected Triton during model construction,
before weight loading, profile execution, or graph capture. Both post-health
layers passed. The scoped unbind/rebind reset then recovered under the same boot
ID before the corrected retry. This is launcher-debug evidence, not a backend
performance sample.

VERDICT -> the June122 native grouped-MoE route is 5.57 percent slower than
Triton inside the already collapsed full graph. Native dispatch is therefore
not Steve's missing mechanism. The best local exact arm now reaches 75.68
percent of Steve's 85.869114 tok/s and leaves 20.884784 tok/s. The next
attribution target is the 81 in-graph TP collectives or another accepted
runtime/kernel family, not more graph-boundary work or native-MoE restoration.

## Dense 27B Transfer Contract

Dense 27B must reuse the method, not Qwen3.6-specific constants:

1. Pin model, quant method, image, vLLM source, native SOs, graph mode, and
   cache identity before timing.
2. Count its own graph pieces, compiled TP collectives, shapes, and per-token
   fence, host-wait, and queue-submit calls.
3. Test no-MTP `FULL_DECODE_ONLY` plus a capturable attention backend as a
   separate arm. Do not inherit an old blanket FULL-capture prohibition from
   the stock v0.23 MTP path.
4. Derive any clone-completion fence threshold from the dense model's actual
   profile tensors. Do not copy Qwen3.6's 8192-row threshold.
5. Run synchronized timing only as a diagnostic and keep its endpoint separate
   from the clean unsynchronized metric.
6. Treat Kineto-visible device kernels as a lower bound when XPUGraph replay is
   opaque.
7. Transfer dense quantization output buffers, oneDNN scratch reuse, graph
   runtime changes, and collective adapters one factor at a time. Do not carry
   the Triton support intervention, routed-MoE workspace, expert layouts,
   grouped GEMM, layerlet, sidecar, or expert-dispatch conclusions.
8. Require fixed semantic output, JSON 16/16, color 16/16, graceful teardown,
   per-card health, and compiled two-rank collective health for every TP=2 arm.

Evidence and tools:

- `vllm/w8a8/analyze_qwen36_xpu_trace.py`
- `results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_xpu_profile_20260826T020000Z_june122_xpu_profile/`
- `results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_cpu_split_die_20260826T022000Z_cpu_split_die_fixed/`
- `results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_cg_full_decode_only_attn_triton_attn_moe_triton_intervention_20260826T031500Z_full_decode_triton_moe_retry/`
