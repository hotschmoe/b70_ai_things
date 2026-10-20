# D1 decode host overhead and XPU graph capture

Agent D1. Card 1 only. Card 0 was left to the other agents.
Stack: vLLM `0.32.1rc1.dev94+g5fd62688f` in
`/mnt/vm_8tb/b70/build/vllm-flashnext/venv`, torch 2.14.0+xpu,
image `vllm-flashnext-xpu:py312`. Model Ornith-1.5-35B-A3B GPTQ int4
mixedcal-v2, TP=1, dtype float16, language-model-only, max-model-len
33792, gpu-memory-utilization 0.90, max-num-seqs 4, port 18081,
container `vllm-fn-ornith-c1`. Served name `hotschmoe-dev`, alias
`ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2`.
Latency harness: `vllm/flashnext/harness/bench_latency.py`, decode 64,
temperature 0. ITL is the SSE inter-chunk gap. Logs and JSON live under
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/`. ASCII only.

Claude consult at the close: `claude --bare -p --model fable-5.1` and
`--model claude-5.5` both exited `unrecognized_model` and
"Not logged in". Same result A3 recorded. No second-model review.

## 1. Eager step, ranked

CONFIG: enforce-eager, FlashAttention, SYCL GDN, no MTP, no graph.
In-process timer on `GPUModelRunner.execute_model` and `sample_tokens`
(`harness/patches/d1_step_timer.py`). cProfile on step 58. A leaf
dispatch timer on step 50. Kineto XPU profiler on step 70. py-spy is
not installed and `ptrace_scope=1`, so it was not used.

COMMAND: `B70_AGENT=d1-eager-profile B70_STEP_TIMER=1 CARD=1 PORT=18081
GRAPH=0 MTP=0 bash vllm/flashnext/harness/serve_ornith_gptq.sh`
Bench `--tag eager-profile --skip-prefill`.
Logs: `d1-serve-eager-profile.log`, `d1-step-timer.log`,
`d1-bench-eager-profile.json`.

RESULT: identity ok, coherence ok (Paris, both texts identical).
C1 ITL p50/p90/p99 71.85/73.46/350.30 ms (p99 is the instrumented
step), 12.17 tok/s, TTFT 153.14 ms, n=64. C2 ITL p50 70.94 ms,
14.08 and 14.31 tok/s. A3's clean eager row remains the baseline:
C1 74.78/75.36/76.40 ms, 13.57 tok/s.

Steady `execute_model` for tokens=1, excluding the three instrumented
steps: n=98, p50 68.86 ms, p90 70.81, min 65.43, max 76.29.
`sample_tokens` p50 1.37 ms. tokens=2 steps p50 68.82 ms (n=62).
A 24-token step was 75.88 ms and a 30-token step was 85.32 ms.

cProfile of step 58 (tokens=1): 25017 calls in 0.085 s, matching that
step's 83 ms wall. Cumulative inside the forward: model 80 ms, MoE
runner 44 ms across 40 layers, `moe_forward_shared` 41 ms, linear
22 ms across 270 calls, GDN `forward_xpu` 21 ms across 30 layers.
Largest exclusive bucket: `Parameter.__torch_function__` 20 ms across
360 calls. Kernel-launch exclusive time is small: grouped GEMM 4 ms
/ 80 calls, `torch.empty` 3 ms / 424, SYCL `gdn_attention` 2 ms / 30,
Triton launch 2 ms / 52, FlashAttention `varlen_fwd` 1 ms / 10.
No `synchronize` sits in the top 30 exclusive.

The dispatch timer is not a device budget. It stretched step 50 to
execute 421 ms + sample 316 ms and reported `device_ms` 539, including
`aten.to.dtype_layout` 306 ms for 5 calls. Leaf counts from that step
are usable: 1000 leaves, 40 `moe_forward_shared`, 110 linear, 80 fused
RMSNorm, 30 GDN, 10 full attention, 10 KV updates, 4 `aten.item`.
Kineto failed: `PTI_ERROR_INTERNAL`, failed to enable the XPU profiler.
The `EngineCore.step` hook logged nothing on the async path; the 70 ms
is inside `execute_model`.

VERDICT: the eager decode step is a Python walk of about 1000 leaf
launches. A 24-token step costs about the same as one token. The
sampler is 1.4 ms. Graph replay is the lever that removes the walk.

## 2. Graph capture

Overlays, all under `vllm/flashnext/harness/`:

- `patches/d1_graph.py`: `B70_XPU_CG_RECLAIM` subclasses
  `torch.xpu.XPUGraph`, forces `keep_graph=True`, and calls
  `instantiate()` every N replays (the B70_XPU_CG_RECLAIM workaround
  for command-list accumulation). `B70_REPLAY_LOG=1` records host time
  around `replay()` and one synchronized XPU-event sample.
- `patches/d1_step_timer.py`: execute, sample, and `EngineCore.step`
  wall times, plus the one-step cProfile / dispatch / profiler arms.
- `patches/sitecustomize.py` loads both when the env vars are set.
- `serve_ornith_gptq.sh`: `GRAPH_MODE` FULL, FULL_DECODE_ONLY, or
  PIECEWISE; `ATTN`; `GDN_BACKEND`; `ASYNC=0` passes
  `--no-async-scheduling`. PIECEWISE with
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1` sends compilation mode NONE so
  inductor does not run.

`vllm/patches/xpugraph_submit_without_event.patch` was not applied.
These benches are shorter than 500 replays per graph, and none hit a
NEO command-list abort. No new `register_fake` was required: FULL,
FULL_DECODE_ONLY, and breakable PIECEWISE all captured.

FlashAttention v2 and the default SYCL GDN op both captured into FULL
and FULL_DECODE_ONLY on this torch 2.14. Capture at sizes 1,2,3,4,8
is 2 profile graphs plus 4 runtime graphs for decode query length 1
(size 8 is above `max_num_seqs`), and 2 plus 2 when MTP sets the
decode query length to 3. PIECEWISE captured 5 plus 5. Each pass took
1 to 2 s once the Triton cache was warm. The first Triton
FULL_DECODE_ONLY startup spent 484 s in the profile forward while
kernels compiled. Later startups were about a minute.

## 3. Latency matrix

All graph rows used `B70_XPU_CG_RECLAIM=500` and async scheduling
unless the row says otherwise. Identity and coherence passed on every
row (same Paris text). MTP acceptance is accepted/drafted during the
bench.

| config | C1 ITL p50/p90/p99 ms | C1 tok/s | C2 ITL p50 ms | C2 tok/s | prefill tok/s 1k/8k/32k |
| --- | --- | --- | --- | --- | --- |
| eager, FA, no MTP (A3) | 74.78/75.36/76.40 | 13.57 | ~73.9 | ~13.6 | 6001/7888/6398 |
| eager, FA, MTP 2 (A3 serve) | 82.60/85.09/95.52 | 20.99 | 82.0 | 22.0/22.6 | 5996/7774/6114 |
| FDO, Triton attn+GDN | 13.20/14.82/15.93 | 76.02 | 14.4 | 70.3/69.8 | 3477/3327/1663 |
| FDO, Triton, MTP 2 | 20.40/20.73/20.88 | 84.31 | 22.9 | 82.9/79.8 | 3716/3038/1148 |
| FDO, Triton, no-async | 15.79/16.14/16.92 | 63.14 | 17.1 | 52.3/58.3 | prefill skipped |
| FDO, FlashAttention, SYCL GDN | 12.81/12.99/13.12 | 78.23 | 13.75 | 73.4/73.0 | 6030/7888/6414 |
| FDO, FlashAttention, MTP 2 | 18.81/19.17/19.37 | 93.88 | 20.9 | 84.9/87.4 | 6141/7795/6088 |
| PIECEWISE, breakable, FA | 13.23/13.36/13.49 | 76.38 | 14.10 | 70.2/72.0 | 6035/7882/6418 |
| FULL, FlashAttention | 12.84/13.05/13.25 | 77.93 | 13.81 | 73.2/72.8 | 6051/7899/6420 |

MTP acceptance stayed 0.36 to 0.40 (eager 0.399, Triton graph 0.395,
FlashAttention graph 0.383).

JSON: `d1-bench-mtp.json`, `d1-bench-eager-profile.json`,
`d1-bench-graph-fdo.json`, `d1-bench-graph-mtp.json`,
`d1-bench-graph-sync.json`, `d1-bench-graph-fa.json`,
`d1-bench-graph-fa-mtp.json`, `d1-bench-graph-piecewise.json`,
`d1-bench-graph-full.json`.

## 4. Where the 13 ms goes

On the no-async Triton FULL_DECODE_ONLY server, `EngineCore.step` p50
was 15.98 ms, matching client ITL 15.79 ms. `replay()` host time p50
was 0.05 ms (it returns before the device finishes). One XPU event
around a single replay, synchronized before and after, was 11.436 ms.
`execute_model` submit p50 was 2.48 ms and sample p50 was 1.11 ms.

Async scheduling is vLLM's default. Turning it off made C1 ITL worse
(15.79 ms versus 13.20 ms on the same Triton graph). Async hides about
2 ms of host beside the device replay. The eager 70 ms Python walk is
gone once the graph replays.

The 11.4 ms figure is one replay of the Triton graph. The
FlashAttention graph's C1 ITL is 12.8 ms, so that graph sits in the
same range. C3's standalone MoE microbench at T=1 was about 0.11 ms;
40 layers of that is about 4 ms, which leaves the rest of the replay
to GDN, attention, norms, and the captured command list. This session
did not split the replay into those kernels. Kineto could not arm.

## Verdict

C1 ITL on the 35B int4 is 12.81 ms p50 (p99 13.12 ms) with
FULL_DECODE_ONLY, FlashAttention, SYCL GDN, and async scheduling.
FULL is the same number (12.84 ms). Both are under the 15 ms
milestone. PIECEWISE is 13.23 ms. Eager was ~75 ms. The best
throughput row is FlashAttention FULL_DECODE_ONLY plus MTP at
93.9 tok/s, with chunk ITL 18.8 ms.

No fork commit is requested from C4. The overlays were enough.
Further ITL cuts are inside the captured graph: a per-kernel split of
the ~11 ms replay, then faster GDN or attention kernels on that
critical path. A long serve that passes 500 replays per graph should
keep `B70_XPU_CG_RECLAIM`, and if NEO still aborts, port
`vllm/patches/xpugraph_submit_without_event.patch` onto torch 2.14.

Teardown: `docker stop vllm-fn-ornith-c1` after the FULL bench. Card 1
lock released. No watcher left running.
