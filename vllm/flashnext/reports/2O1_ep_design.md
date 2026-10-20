# 2O1 -- Strata 2-card expert parallelism, and a vLLM EP=2 design for XPU

Agent: Claude Opus 5.5 (claude-opus-5-5), 2026-10-20. CPU-only source study.
No GPU, no build, no service touched, nothing committed.

Sources read:
- Strata `/mnt/vm_8tb/b70/build/strata-ab/src-ep`, worktree HEAD `e753e9cb`.
  The DD pin `1c360526` is an ancestor. Between the two, only
  `sycl/src/core/verify.cpp` differs among the EP files (+22/-4, the default-off
  GDN2 commit). The line numbers below are from HEAD.
- DD config `strata/flash-next/dd/hotschmoe-dd.json`, and the live DD engine log
  `/mnt/vm_8tb/b70/results/hotschmoe-dd/engine.log` (2026-10-16 start).
- Night-lab agent reports `strata/flash-next/night/agent-reports/{moe_report,trace_final_analysis}.md`,
  the docs named in the task, `docs/P2P_GPU.md`, and
  `vllm/contrib/vllm_push_allreduce/README.md`.
- vLLM clone `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`, branch `b70/flashnext`,
  HEAD `4d73572e14`. vLLM paths below are relative to `vllm/` in that clone.

## 0. Verdict

Stock vLLM EP does not do what Strata does. Strata runs one process with two
SYCL queues. Card 0 runs the whole model. Card 1 is an expert engine with a
P2P inbox, and it never runs attention. Every cross-card transfer is a posted
peer write issued by a kernel. Every sync is a device-memory epoch flag that
an EU spins on with uncached loads. All of it is recorded in two SYCL graphs,
one per card. A decode window therefore has zero host syncs and zero oneCCL
calls inside it: 2 graph submits and 96 device handshakes for 48 layers.

On XPU, stock vLLM EP means TP=2+EP (two oneCCL all-reduces per layer) or
DP=2+EP. DP=2+EP uses the AgRs manager: three all-gathers and one
reduce-scatter per MoE layer, plus a DP all-reduce per step and dummy forwards
on the idle rank. Repo evidence puts a small oneCCL collective at about 85 us.
That makes stock EP 8-16 ms of collectives per decode step, against Strata's
roughly 1.3 ms of total EP overhead per window.

Recommended design: an "EP sidecar" plugin. vLLM stays a single rank (TP=1,
DP=1, world size 1) on xpu:0. A new out-of-tree package, `b70_ep_xpu`, owns
xpu:1, the remote expert weights and the exchange regions. It ports Strata's
transport kernels unchanged in protocol. It records card 1's half of each
captured forward into its own SYCL command_graph and submits that graph just
before vLLM replays the card-0 graph. It needs no oneCCL, no DP lockstep and
no all2all manager.

The vLLM-side changes are:
- a MoE quant method or runner for EP layers;
- a loader split by placement;
- graph capture/replay hooks: a 20-line monkeypatch of
  `CUDAGraphWrapper.__call__` first, upstreamable later.

A symmetric DP=2 doorbell all2all is kept as phase 3, if card balance at C2
needs it.

## 1. How Strata does EP=2 (night-final-v3)

### 1.1 Process and device model

- One engine process and one host thread drive both cards. Card 0 is the
  verifier's main queue `cs_`. Card 1 is a second in-order queue `ep_q_`,
  created in `Verifier::ep_init` (`sycl/src/core/verify.cpp:2081-2082`).
- dpct gives each device its own `sycl::context`. Shared regions are therefore
  raw Level Zero allocations: `zeMemAllocDevice` on the owner, then
  `zeMemGetIpcHandle` and `zeMemOpenIpcHandle` on the other card's context
  (`sycl/include/strata/core/ep_mem.hpp:31-53`). Each region must be its own
  allocation. A small pooled `sycl::malloc_device` block page-faulted or
  disturbed its neighbours (`ep_mem.hpp:3-4`).
- EP requires `--stream-experts`, every expert resident on one of the two
  cards, a native pack and no pipelined windows (`verify.cpp:2025-2033`). CLI
  parsing is at `generate.cpp:552` and `:1973`.

### 1.2 What lives where (DD, from the engine log)

| Item | Card 0 | Card 1 |
| --- | --- | --- |
| Routed experts (512 per layer x 48 layers) | 138 per layer (6624, 10.77 GiB) | 374 per layer (17952, 29.19 GiB) |
| Attention / QSA, GDN, HC, dense GEMVs, router | all | none |
| Shared expert | yes. With SH_LATE it runs after card 0's own experts and before the peer wait (`verify.cpp:888-900`, `:1691-1709`) | no |
| Token embedding, vocab head, draft-vocab head | yes | no |
| PLE | 26.8 GiB table read from the GGUF (`--ple-io direct`), consumed on card 0 | no |
| MTP draft layer, including its 512 experts (1027 MiB: experts 675, dense 111) | yes, single-card | no |
| KV / GDN state for 2 slots at 200K (5.63 GiB) | yes | no |
| Decode inbox, rows and scratch; prefill exchange (1260 MiB) | return regions, plan buffers | inbox regions, `ep_rows1_`, scratch |

The placement is a traffic snake, not a quota split (`generate.cpp:4795-4842`).
Per layer, experts are ranked by `--expert-profile` frequency and dealt to
cards 0,1,1,0,0,1,1,... until card 0's VRAM quota `q0` is full. The coldest
remaining experts go to card 1. The log line reads "each layer's traffic dealt
evenly between them". The quota split is 138/374 experts, but traffic is about
50/50. `STRATA_NIGHT_EP_SHARE0` skews the deal and measured no gain (N19).

### 1.3 Decode (verify window), per MoE layer

Recorded inside `record_window`'s `pre(l)` and `post(l)` lambdas
(`verify.cpp:1616-1661`, `:1684-1789`). The layer loop is at `:1816-1825`.
In the DD the group count is G=1: batch windows are one group (`:869`).

Card 0 (queue `cs`):
1. Router GEMV and top-10. With PLAN1, `resident_plan_ep_pair` builds both
   cards' expert plans in one launch (`verify.cpp:1526-1533`; kernel at
   `verify_kernels.dp.cpp:2793`).
2. `quantize_q8_1_rows` of the post-attention hidden state, n x 2560 to
   2880 B per row (`:1659`).
3. `ep_push_w` (`ep_kernels.dp.cpp:294-314`, call at `verify.cpp:1626-1632`).
   One work-group stores ids, routing weights, card 1's plan (about 1.6 KB) and
   the q8_1 rows straight into card 1's inbox. These are posted PCIe writes.
   It then does a system-scope release fence and a system-scope group barrier
   (the S28 fix, `docs/20261015_strata_ep_publication_scope.md`). Finally it
   stores `inflag[ring] = *ctr` with an uncached L1+L3 write hint
   (`sycl_doorbell.hpp`, `sys_store`).
4. Own routed experts: `native_expert_grouped`, GU + SwiGLU + down
   (`verify.cpp:1690`). Then the shared expert (SH_LATE).
5. `ep_wait(ret[ring], ctr0)`: a single-task spin with uncached loads, bounded
   by 2,000,000 spins, writing an error word to host-mapped memory if the bound
   is hit (`ep_kernels.dp.cpp:250-263`, call at `verify.cpp:1727`).
6. `ep_combine_psum`. Own entries in entry order, plus card 1's per-token sum,
   plus the gated shared expert, into `bo_` (`ep_kernels.dp.cpp:349-411`, call
   at `verify.cpp:1779-1785`). It hard-codes k=10.

Card 1 (queue `ep_q_`), recorded in the same `pre(l)`:
1. `ep_wait(inflag[ring], ctr1)` (`verify.cpp:1636`).
2. `ep_copy_uncached`: inbox to a local copy, reading past L1/L3
   (`ep_kernels.dp.cpp:19-29`; call at `:1637`).
3. With PLAN1 there is no planning; it uses card 0's plan from the inbox
   (`:1640`).
4. `native_expert_grouped` on card 1's expert arena (`:1648-1650`).
5. `ep_send_psum`: one FP32 weighted row per token, entry order, into card 0's
   `psum` region. Every work-group fences at system scope. The last one, found
   with a system-scope done-counter, stores `ret[ring] = *ctr1`
   (`ep_kernels.dp.cpp:316-347`; call at `verify.cpp:1651-1654`).

The window prologue is `ep_bump(ctr0)` on card 0 and `ep_bump(ctr1)` on
card 1. These are the first node of each card's graph, so the epoch lives in
device memory and is not frozen at capture (`verify.cpp:905-909`;
`ep_kernels.hpp:1-5`).

Capture records card 0 into `cs_` and card 1 into `ep_q_` at the same time,
as two separate command_graphs (`verify.cpp:2166-2191`; batch keys at
`:3271-3295`). Replay submits card 1's graph first, because its first node
waits for layer 0 (`verify.cpp:2603-2607`, `:3710-3712`, `:3980-3982`). After
the window, `ep_check` does one `ep_q_->wait()` and reads three host-mapped
error words (`verify.cpp:2128-2152`).

Counts per decode window (48 MoE layers, G=1):
- 2 graph submissions.
- 1 host wait for card 1, folded into the window's end.
- 0 host syncs inside the window.
- 0 copy-engine operations.
- 0 oneCCL calls.
- 48 pushes and 48 returns, which is 96 flag handshakes.
- About 6 card-1 kernels per layer.

Bytes per layer per row:
- push: 40 B ids + 40 B weights + 2880 B q8_1, plus about 1.6 KB of plan per
  layer;
- return: 10240 B (FP32 psum).

Measured costs (trace_final_analysis.md, C1, profiled 29.3 ms window,
about 22.4 ms unprofiled):
- card 0: EP push 50 launches 0.18 ms, EP wait spin 0.57 ms, EP plan/group
  0.55 ms;
- card 1: 4.92 ms compute, 15.54 ms spinning.

From moe_report.md:15-40:
- push latency 3.0-4.75 us; return latency 1.4-2.9 us;
- card 1 has about 2.6 us of in-order gap between kernels;
- card 1's routed kernels run 20-30% slower than card 0's
  (`docs/20261016_next_investigations.md`).

EP versus layer split on the same binary: C1 91.1 vs 81.3 (night lab).

### 1.4 Batch MTP and two streams

- `--batch 2` plus `STRATA_BATCH_MTP=1 DEPTH=2`. Each slot contributes 1 + depth
  rows to one verify window. C2 is a 6-row window, which is 60 routed entries
  per layer (moe_report.md:14).
- Both streams share one push and one return per layer. Card 1 sees one inbox
  per layer, whatever the stream count. Graphs are keyed by rows/slots
  (`ep_exec_bm_[bkey(rows,S,base)]`, `verify.cpp:3293`).
- The MTP drafter and the commit graph run on card 0 only. Card 1 idles
  through them. The C1 card-0 host gaps (verifier to commit 3.17 ms, MTP to
  verifier 1.40 ms) are single-card time.

### 1.5 Prefill (eager, host-driven per layer)

Flags: `STRATA_EP_PREFILL_REDUCE`, `_PUSH`, `_GATHER_ALIAS`, and the night
flags EARLY, F16SUM, GROUPED and HALVES (`prefill.cpp:2031-2060`, `:310-340`).

1. Card 0 runs attention and the router.
2. The host waits once per layer (per half) for the routed ids
   (`prefill.cpp:3778`, `:3791`). The ids are needed to group experts on the
   host.
3. `ep_push_signal` posted-writes T x 2560 FP16 activations into card 1's
   region and raises a doorbell (`ep_kernels.dp.cpp:194-212`; calls at
   `prefill.cpp:3772`, `:3786`).
4. Card 1 waits on the doorbell (`ep_wait_value`), copies the activations
   uncached, and runs grouped `joint_matrix` experts.
5. `ep_prefill_reduce_h` (`ep_kernels.dp.cpp:101-160`; call at
   `prefill.cpp:4214`) pushes back one FP16 row plus one power-of-two scale
   per token, with a done-counter doorbell.
6. Card 0's `ep_prefill_finish_h` (`prefill.cpp:5029`) adds its own experts
   and the shared expert.
7. HALVES interleaves two half-chunks per layer so card 1's experts hide under
   card 0's attention (`prefill.cpp:5216-5227`).

Measured results:
- REDUCE+PUSH took 4K prefill from 570 to 1037 tok/s and 32K from 576 to
  1106 (S6).
- The night stack reaches about 2200 tok/s at 4K-32K
  (`docs/20261016_night_lab.md`).
- Historical S10 traces show about 1300 host appends per layer for card-1 work
  (`docs/20261015_strata_ep_prefill_dispatch.md`). The host-side grouping is
  the weak spot.

### 1.6 Why Strata's EU spin works when J.9-C hung

`docs/P2P_GPU.md` J.9-C (June, kernel 7.0, DPC++ 2025.3) found that a flag
written mid-kernel and spun on from the other card never converged. The sglang
pure-SYCL push all-reduce was abandoned for the same reason
(memory note capturable-ar-dead-end).

Strata (kernel 7.1, CR 26.22, DPC++ 2026.1) differs in four ways.
1. Flag stores and spin loads use explicit `cache_control` uncached L1+L3 read
   and write hints through `annotated_ptr`, not atomics or `volatile`.
   `sycl_doorbell.hpp` documents that a system-scope atomic store sat in the GPU
   cache until the kernel ended. That is exactly the J.9-C symptom.
2. Payload readers on the destination card use uncached loads
   (`ep_copy_uncached`, `load_uncached`).
3. The producer barrier has system scope (S28).
4. Transfers are push only. Nothing reads remote memory: posted writes run
   about 11 GB/s, while pull is 3.5x slower (P2P_GPU.md J.4, J.5).

This is the transport contract to port. The S28 poison/epoch oracle
(`strata/flash-next/ep_*_probe.cpp`) is the gate for it.

## 2. vLLM EP today (b70/flashnext tree)

### 2.1 How EP is configured

- `use_ep = dp*pcp*tp > 1 and enable_expert_parallel`. EP flattens TP into
  `ep_size = tp_size`, `ep_rank = tp_rank` and sets MoE TP to 1
  (`model_executor/layers/fused_moe/config.py:1117-1245`).
- all2all kernels are used only when `use_ep and (dp>1 or pcp>1 or
  sequence_parallel)` (`config.py:1033-1038`). TP=2+EP with DP=1 has no
  all2all.

### 2.2 Placement

- `determine_expert_map` gives `linear` (contiguous) or `round_robin`
  placement, as an int32 map with -1 for non-local experts
  (`fused_moe/expert_map_manager.py:22-150`).
- EPLB is allowed on XPU (`config/parallel.py:563-567`). It uses `torch_xccl`
  for rebalancing and needs `E % ep_size == 0` (`layer.py:269-282`).

### 2.3 Backends and managers

- `--all2all-backend` is a closed Literal (`config/parallel.py:45-60`). The
  default is `allgather_reducescatter`. `naive` and `pplx` are rewritten to it
  (`:537-543`).
- `XpuCommunicator` accepts only naive and AgRs. Every other backend falls
  back to `AgRsAll2AllManager` with a warning
  (`distributed/device_communicators/xpu_communicator.py:30-47`).
- `all_reduce` is `clone()` plus `dist.all_reduce` over xccl (`:62-70`).
- There is no custom all-reduce for XPU (`ca_comm=None`).
- The AgRs dispatch is `all_gatherv` of [hidden, topk_w, topk_ids]. On XPU
  that is one all_gather per tensor (`xpu_communicator.py:155-207`). The
  combine is `reduce_scatterv` (`all2all.py:67-173`).
- Prepare/finalize selection is a hard-coded if/elif chain
  (`fused_moe/all2all_utils.py:182-453`). DeepEP, MoRI, NIXL and MoonEP are
  imported only under `is_cuda_alike()` (`:103-125`).

### 2.4 Collectives per decode step

| Topology | Collectives per MoE layer | Per step | At ~85 us each, 48 layers |
| --- | --- | --- | --- |
| TP=2 | 1 all-reduce on o_proj/out_proj + 1 on the MoE output (`moe_runner.py:509-515`) | -- | 8.2 ms |
| TP=2+EP, DP=1 | same 2 (no all2all; `no_dp_ep`) | -- | 8.2 ms. GGUF packed TP sharding is also not implemented (`reports/1C4_e2e_gguf.md`) |
| DP=2+EP | 3 all_gather + 1 reduce_scatter | `coordinate_batch_across_dp` 4xdp int32 all_reduce (`v1/worker/dp_utils.py:57-76, 222-260`); gloo under async scheduling (`config/vllm.py:1717-1728`); dummy batch on the idle rank (`v1/engine/core.py:2454`); padding to the max tokens | 16 ms + per-step sync |
| Strata-style sidecar (this design) | 1 push + 1 return, device flags | 2 graph submits | about 0.3 ms transfer latency on the critical path (section 4) |

Notes on this table:
- The 85 us figure is the small oneCCL all-reduce from
  `vllm/contrib/vllm_push_allreduce/README.md` ("oneCCL ~85 us") and
  P2P_GPU.md H.10 (88 us at 10 KB).
- The captured push-AR graph is about 35 us per all-reduce.
- Memory note decode-is-allreduce-bound: TP=2 decode on this box is 43%
  collective time.

### 2.5 Graph capture

- `vllm::moe_forward` is not a split point, so its collectives are captured
  inside the graph (`moe_runner.py:200-213`).
- `CUDAGraphWrapper` (`compilation/cuda_graph.py:145-362`) captures with
  `torch.cuda.graph`, which is patched to `torch.xpu.graph`
  (`v1/worker/xpu_model_runner.py:70-73`).
- It already calls `get_offloader().sync_prev_onload()` before capture and
  before replay, and `join_after_forward()` inside capture
  (`cuda_graph.py:302`, `:322`, `:360`). Those are the natural hook points.

### 2.6 Shared expert and plugin seams

- Shared expert: the runner computes it before routing
  (`_maybe_apply_shared_experts(..., NO_OVERLAP)`, `moe_runner.py:616-618`).
- On XPU there is no aux-stream overlap: `aux_stream()` is None off CUDA.
- Plugin seams: `vllm.general_plugins` runs in every worker
  (`plugins/__init__.py:77-90`). `PluggableLayer.register_oot` can swap
  `MoERunnerInterface`, `RoutedExperts` or `GateLinear`
  (`model_executor/custom_op.py:44-88`). The platform's
  `get_device_communicator_cls` can be overridden to return an
  `XpuCommunicator` subclass with our own all2all manager
  (`platforms/xpu.py:541-548`, resolved at `parallel_state.py:548-557`).

### 2.7 Our fork already has

- `B70GGUFMoEMethod.apply` maps global ids through `expert_map` on device. An
  id of -1 contributes 0 (`models/qwen4_exp/gguf/moe.py:127-179`).
- The plugin's fused `moe_layer` decode op is graph-capturable
  (`plugins/b70_gguf_xpu`; JOURNAL C7 07:21, FULL_DECODE_ONLY C1 ITL 11.11 ms on
  one card for the 35B IQ3_XXS).

C7's planned DP=2 EP eager run goes through the naive path
(`moe_runner.py:800-803`, `do_naive_dispatch_combine`). That is a useful
correctness baseline. Its speed is not a Strata comparison.

## 3. Design: `b70_ep_xpu`, an EP sidecar for vLLM on 2 B70s

### 3.1 Topology

- vLLM runs with world size 1: `--tensor-parallel-size 1`, no
  `--data-parallel-size`, no `--enable-expert-parallel`.
- `harness/xpu-run.sh --card both` exposes both cards. vLLM's worker uses
  xpu:0.
- The plugin opens xpu:1 itself and is the only user of card 1.
- This mirrors Strata exactly. The single process and single host thread
  avoid oneCCL and its multiprocess queue-handoff failure (AGENTS.md
  "Multi-GPU safety"), DP lockstep, dummy forwards and per-step DP
  all-reduces.

### 3.2 Placement

Card 0 holds everything vLLM already places:
- embedding, PLE (host table), attention/QSA, GDN, HC, dense layers, router,
  shared expert, head;
- the MTP drafter, including all of its experts (single-card, as Strata);
- KV and GDN state;
- card 0's routed-expert set.

Card 1 holds the remaining routed experts and the exchange regions. The split
is the traffic snake ported from `generate.cpp:4795-4836`. Its inputs are a
per-layer frequency ranking and card 0's per-layer quota:
- Flash-Next: Strata's `data/expert-profile-sycl.bin`, which gives 138/374 at
  the DD's KV size.
- 35B dev model (40 layers, 256 experts, top-8, hidden 2048): a histogram
  collected by a counting hook in our MoE method. Start from a linear 128/128
  split.

The placement is written as two per-layer int32 tables, `res0[E]` and
`res1[E]`, holding the local slot or -1. A JSON placement file sits next to
the model lock.

### 3.3 Weights

The GGUF loader already keeps expert blobs on CPU (`requires_device_loading =
False`, `moe.py:45`). A new `B70EPMoEMethod` subclasses `B70GGUFMoEMethod`. In
its `process_weights_after_loading`, per layer:
1. Gather the set-0 experts into a contiguous packed tensor on xpu:0.
2. Gather the set-1 experts into a contiguous packed tensor on xpu:1.
3. Drop the CPU copy.

Peak card-0 usage never sees the 29 GiB of card-1 experts. vLLM's memory
profiler only measures xpu:0, so card 0's KV budget is computed correctly.
The plugin checks free memory on card 1 itself and fails at load if it does
not fit. That check mirrors `generate.cpp`'s "card 1 has X GiB free" line.

### 3.4 Transport

A port of `ep_kernels.dp.cpp`, generalized, built with the plugin's DPC++
2026.1 (the same compiler as Strata S28, so the `cache_control` properties are
available).
- Kernels: `ep_bump`, `ep_push_w`, `ep_wait`, `ep_copy_uncached`,
  `ep_send_psum`, `ep_combine_psum` (generic K and N; Strata hard-codes k=10).
- Prefill kernels: `ep_push_signal`, `ep_wait_value`, `ep_prefill_reduce_h`,
  `ep_prefill_finish_h`.
- The doorbell helpers from `sycl_doorbell.hpp`.
- Regions are dedicated `zeMemAllocDevice` allocations, one per region, as in
  `ep_mem.hpp`. They are mapped for the other card by one of two routes,
  chosen in M0:
  - IPC-open into the peer device (Strata's route). This assumes torch keeps
    separate contexts per device.
  - A direct peer USM pointer with `ext_oneapi_enable_peer_access` (P2P_GPU.md
    J.5, 11.2 GB/s). This applies if torch-xpu uses one platform context for
    both devices, which is believed but not verified here.
- Fallback transport if an EU spin ever misbehaves inside torch's graph: L0
  IPC events in a two-device pool with command-streamer waits, injected through
  `ext_codeplay_enqueue_native_command`. That is the proven push-AR graph
  route: `vllm/contrib/vllm_push_allreduce/118_xpu_push_ar_graph.cpp:162-227`,
  memory note b70-capturable-cross-device-sync. It costs about 35 us per
  handshake versus about 3-5 us, so it is a fallback only.

### 3.5 The MoE op, one per EP layer

`torch.ops.b70_ep.moe_ep(x, topk_w, topk_ids, layer_idx) -> y` is called from
`B70EPMoEMethod.apply` on card 0. It is registered with a fake implementation
for compile. It has no host syncs and uses Python ints only, as in
`moe.py:92-102`.

Card 0 work goes on torch's current stream:
1. Build the q8_1 image (decode) or FP16 rows (prefill).
2. `ep_push_w` ids, weights, the plan and the rows into inbox slot `l`. With
   PLAN1, card 1's plan is built on card 0.
3. Run the existing `moe_layer` on the set-0 tensor with the `res0` map.
4. `ep_wait(ret[l])`.
5. `ep_combine_psum` into the output: own entries, then the peer sum. The
   shared expert stays in the runner at first (section 3.7).

Card 1 work goes on the plugin's queue for xpu:1:
1. `ep_wait(inflag[l])`.
2. `ep_copy_uncached`.
3. Run `moe_layer` on the set-1 tensor with the `res1` map.
4. `ep_send_psum`, which raises `ret[l]`.

Epochs: the first EP op of each forward issues `ep_bump` on both cards. A
per-forward "EP session" object counts card-0 calls and card-1 submissions.
Any mismatch raises before the next forward. Spin bounds plus host-mapped
error words, as in `ep_check` at `verify.cpp:2128-2152`, are read once per
step without a sync. They turn a pairing bug into a clean error, not a hang.

### 3.6 Graph integration (FULL_DECODE_ONLY first, which D1/E1 found best on XPU)

During capture, the first `moe_ep` call sees
`torch.xpu.is_current_stream_capturing()` and calls `begin_recording` on the
plugin's card-1 queue. The card-1 graph is keyed by the
`batch_descriptor`.

A capture-end hook does `end_recording` and `finalize` and stores the
executable graph under that key. A replay hook submits the card-1 executable
for the same key just before `entry.cudagraph.replay()`. That gives Strata's
"peer's graph first" (`verify.cpp:2603-2604`).

Where the hooks live, in order of preference:
1. A plugin monkeypatch of `CUDAGraphWrapper.__call__` (`cuda_graph.py:235-362`).
   It is 3 calls wrapped around the existing code, which the brief allows as a
   runtime patch.
2. Later, an isolated fork commit "graph lifecycle hooks", worth upstreaming.

The card-1 graph needs the same periodic re-instantiation as card 0's
`B70_XPU_CG_RECLAIM` (memory note mtp-graph-neo-abort).

Eager forwards (prefill, mixed batches, uncaptured sizes) submit card-1
kernels directly on the plugin queue. Ordering is enforced only by device
flags.

PIECEWISE is not supported in v1. A forward would span several wrappers, and
the card-1 graph must cover exactly one forward.

### 3.7 Shared expert

v1 keeps the runner's order: the shared expert runs before routing on card 0.
Its cost is that card 1 starts about 24 us later per layer (1.14 ms per C1
window / 48).

v2 has two options:
- An OOT `MoERunner` (`register_oot`) for EP layers that pushes first and runs
  the shared expert while card 1 works. This is Strata's SH_LATE.
- Better: run the shared expert on card 1. Card 1 already receives the q8_1
  rows, the result rides back inside the psum row, and card 0 sheds 1.14 ms per
  C1 window. This is investigation 1a in
  `docs/20261016_next_investigations.md`.

### 3.8 MTP and concurrency

- The vLLM verify forward already batches all requests' 1+k rows into one
  forward. One push and one return per layer cover C1, C2 and C4. That is
  Strata's batch-MTP shape for free.
- The drafter's MoE is marked non-EP (layer prefix `mtp.`). It runs on card 0
  with all of its experts.

### 3.9 Prefill

v1 ports Strata's REDUCE + PUSH + F16SUM, with one change: no host wait per
layer. The card-1 grouped expert path must plan on the device from the
pushed ids (EARLY generalized). The plugin's `prefill_moe_layer` must be
checked for host-side counts.

v2 adds HALVES as two micro-batches per layer. vLLM's ubatching/DBO
infrastructure is the seam. It is CUDA-centric and needs an XPU audit.

### 3.10 Why not the alternatives first

| Option | For | Against |
| --- | --- | --- |
| Stock DP=2+EP (C7) | no code | about 16 ms of xccl per step; DP sync and dummy forwards; padding |
| Stock TP=2+EP | no all2all | 96 oneCCL all-reduces per step; GGUF TP sharding missing; queue-handoff risk |
| TP=2 + push-AR graph | proven 35 us/AR | still 96 handshakes x 35 us = 3.4 ms; attention sharded over PCIe |
| Phase 3: DP=2 + doorbell all2all (`get_device_communicator_cls` override + prepare/finalize) | each card runs its own stream's dense work at C2 (card balance) | DP lockstep, dummy forward at C1, dense weights + MTP replicated, KV split; decode GEMVs are weight-bound, so splitting rows across cards saves little |

## 4. Per-step latency budget versus Strata

The model below is per MoE layer at C1 with 5 verify rows, Flash-Next.

Transfer:
- push: 5 x 2960 B + 1.6 KB = about 16.4 KB, about 1.5 us of wire at 11 GB/s,
  plus 3.0-4.75 us flag latency;
- return: 5 x 10240 B = 51 KB, about 4.6 us, plus 1.4-2.9 us. FP16 halves the
  wire time.

Critical path:

```
max(card-0 own experts ~70-150 us + shared,
    push latency + card-1 chain ~135-220 us + return latency)
+ combine ~2 us
```

Card 1's chain is critical today: card 0 waits 0.57 ms per window. The
exchange itself adds about 10 us per layer on the critical path, about
0.5 ms per window.

Expected card-0 EP overhead per window equals Strata's: push 0.18 + plan 0.55
+ wait 0.3-0.6 ms, about 1.1-1.3 ms of a 22.4 ms window. The new vLLM-side
costs are:
- submitting one more graph per step: tens of us of host time;
- Python per-op overhead only in eager prefill.

The torch/vLLM graph runs one card-0 graph per step where Strata ran three
(MTP, verify, commit). Host gaps then come from the vLLM step loop, not from
EP.

Target for the 35B dev model: EP C1 ITL at most 1.05x the one-card 11.11 ms
FULL_DECODE_ONLY, and better once experts are the critical path. For
Flash-Next, which does not fit on one card, the reference is Strata's C1 141
and C2 92/stream.

## 5. Risks

1. Cross-card visibility. J.9-C hung and the sglang push-AR failed, yet Strata
   works. Port the cache-control hints, the system-scope barrier and push-only
   transfers verbatim. Run the S28 poison oracle before any serve.
2. torch-xpu context model. Whether one context covers both devices decides
   IPC-open versus a direct peer pointer. Allocate regions with raw
   `zeMemAllocDevice`, never from torch's pool (`ep_mem.hpp:3-4`).
3. Recording a plugin graph on xpu:1 while torch records on xpu:0. Strata does
   this with dpct queues. Under torch the capture stream is dedicated (memory
   note b70-capturable-cross-device-sync #3), and the plugin graph is
   independent of it. This needs an M0 probe.
4. Replay pairing. Every card-0 replay or eager EP forward needs exactly one
   card-1 submission, and async scheduling queues forwards back to back.
   Bounded spins and the session counter turn a mismatch into an error.
5. Graph-replay command-list growth (the NEO abort). Reclaim the card-1 graph
   on the same cadence as card 0's.
6. Numerics. PSUM reorders FP32 (Strata NLL gate: PSUM -0.048, within noise);
   F16SUM prefill returns relative error up to 2^-11. Gate with matched greedy
   text plus the NLL harness.
7. Card 1 runs 20-30% slower (unexplained in Strata). Expect the same here;
   it bounds the traffic split.
8. Rebase drift. Keep it to a monkeypatch plus one new method class. Make no
   edits to the all2all chain or the Literal.
9. oneCCL. The design never initializes xccl for EP. Still run `bin/xpu-health`
   on both cards around every two-card run. Use `bin/xpu-collective-health`
   only for C7-style stock arms.

## 6. Build and test plan

Every step uses `bin/gpu-run --card both` with the DD stopped by the
orchestrator, `bin/xpu-health` before and after, and `bin/xe-reset` only after
a crash.

| M | Work | Gate (CONFIG -> COMMAND -> RESULT -> VERDICT) | Size |
| --- | --- | --- | --- |
| M0 | `plugins/b70_ep_xpu`: transport kernels plus region setup in torch's context; a single-process two-device torch probe porting `strata/flash-next/ep_ipc_probe.cpp` and the S28 ready-probe arms (poison, epochs, 64 reps) eager, then card-0 `torch.xpu.graph` plus a plugin card-1 graph | 65/65 oracle passes in both modes; push and return latency at most 5 us each; health OK | M |
| M1 | `moe_ep` on one real dev-model layer (35B IQ3_XXS GGUF, top-8, 2048) with a 128/128 split; T=1..8 and T=4096 | output vs one-card `moe_layer`: own experts bitwise, total within FP32 reorder; EP layer time vs one-card | M |
| M2 | `B70EPMoEMethod` plus loader split; dev model served eager, sidecar EP | greedy text equals C4/C7 one-card ("Paris..."); `/v1/models` identity; health | M |
| M3 | Capture hooks (monkeypatch) and FULL_DECODE_ONLY EP | C1/C2 ITL vs one-card 11.11 / 16.5 ms and vs C7's stock DP=2 EP; text unchanged | M |
| M4 | MTP on, drafter MoE non-EP | acceptance equals one-card MTP; C1/C2 ITL | S |
| M5 | Prefill EP with device-side plan and F16 return | 4K/32K tok/s vs one-card; NLL gate | M/L |
| M6 | Flash-Next qwen4_exp: Strata profile placement (138/374), 29 GiB on card 1, PLE host table | eager greedy tokens vs Strata DD reference; then captured C1/C2 vs 141 / 92 | L |
| M7 | Card balance: shared expert on card 1, then more experts on card 1, then the MTP drafter or head on card 1 | C1 and C2 deltas vs M6 | M each |
| M8 (optional) | Phase 3: DP=2 doorbell all2all via a communicator-class override | C2/C4 vs M7 | L |

Repo work items (sizes):
- New package `vllm/flashnext/plugins/b70_ep_xpu/`: SYCL .so, ops.py and a
  hooks.py monkeypatch (M).
- `B70EPMoEMethod` in `models/qwen4_exp/gguf/` and the qwen3_5 GGUF path (S/M).
- Placement tool: profile to JSON (S).
- Probe script under `plugins/b70_ep_xpu/tests/` (S).

Requests to other agents:
- C7: record the stock DP=2 EP eager greedy text and ITL as the M2/M3 control.
- A2: confirm whether `prefill_moe_layer` needs host-side expert counts.

## 7. Transport: push-AR and topology

This section was added at the coordinator's request. It is CPU-only: it cites
existing measurements and does not run any GPU work.

### 7.1 Topology (what the fabric is)

The box is a Threadripper 1950X on an ASRock X399. The two cards hang off
different dies, each with its own root complex.

```
card 0  pci0000:00 -> 00:03.1 -> 09:00.0 (switch) -> 0a:01.0 -> 0b:00.0
card 1  pci0000:40 -> 40:03.1 -> 42:00.0 (switch) -> 43:01.0 -> 44:00.0
```

Sources: `docs/20260825_xe_nonreboot_recovery_and_pcie_topology.md:98-113`
and `docs/P2P_GPU.md` J.0 (lines 533-548).

Facts from those sources:
- Both uplinks run at Gen3 x16 (8 GT/s), the platform maximum. The Gen1 x1
  reading on the GPU function is an on-card bridge artifact. This was
  re-verified on kernel 7.1 (P2P_GPU.md K.10).
- The BIOS is in UMA mode, so Linux reports `numa_node=-1` and CPUs `0-31`
  for both cards. That hides the fact that every card-to-card transaction
  crosses Infinity Fabric between dies.
- IOMMU is off and ReBAR is 32G on both cards.
- `zeDeviceCanAccessPeer` is True in both directions on kernel 7.x. It was
  False on 6.18 (P2P_GPU.md H.11).
- Peer atomics are not supported ("peer ATOMICS=N", J.9). Any cross-card
  signal must therefore be a plain store plus a fence, or a Level Zero event.
  It cannot be a remote atomic.

The Infinity Fabric quirk is a direction asymmetry. Posted writes stream
well. Non-posted reads are throttled by read credits and cross-die latency.

Measured transport numbers, all on this box:

| Path | Large-message BW | Small-message latency | Source |
| --- | --- | --- | --- |
| PUSH, peer write executed on the source (EU or copy engine) | 11.2-11.3 GB/s (72% of the Gen3 wire) | -- | J.2; K.10 (kernel 7.1): EU 11.21, BCS 11.20 |
| PULL, peer read executed on the destination | 3.23-3.24 GB/s | -- | J.2, K.10 |
| Host bounce d0 -> host -> d1 | 3.5 GB/s | -- | J.2, K.10 |
| torch `.copy_()` across devices | 1.35 GB/s | 452 us per 8-element copy | H.8, P2P_GPU.md:357-358 |
| Raw L0 8-byte peer ping | -- | 8.6-9.3 us | J.2, K.10 |
| 2-process L0 IPC push | 11.06 GB/s | 13.1 us one-way (host-timed) | K.10 |
| L0 command-streamer event sync (push + signal + wait + read, closed lists replayed) | -- | 21.5 us at 10 KB | K.3 |
| Captured push-AR (`ar_allreduce_graph`) in torch XPUGraph, 2 processes | -- | 35.4 us per all-reduce at 10 KB | K.6 |
| Single-context push all-reduce, eager | 10.66 GB/s | 50.8-59.5 us at 10 KB | J.7, K.10 |
| oneCCL P2P=1 with SYCL kernels | 10.34-10.39 GB/s | 90 us at 4 KB | K.10 |
| oneCCL P2P=0 SYCL / eager | 1.1 / 1.0 GB/s | 90 / 286 us at 4 KB | K.10 |
| Strata in-graph kernel push, device timestamps | -- | 3.0-4.75 us push to wake; 1.4-2.9 us return | night `moe_report.md:40` |
| Strata S8 PUSH vs uncached remote read, GPU1 -> GPU0 | PUSH 11.22 vs read 3.32 GB/s (256 MB) | 16 KB host wall: PUSH 0.011 ms, read 0.017 ms | `docs/20261014_flashnext_strata_optimization.md:522-534` |
| Strata S10, prefill FP32 sum consumer: remote read vs local uncached read after PUSH | -- | 11078 us vs 307 us per layer | same doc `:644-646` |

What the table says:
- On this fabric, data always moves as a posted write from the producing
  card.
- The consumer reads its own VRAM, with uncached loads.
- Every pull, `.copy_()` or host-staged path costs 3-10x in bandwidth.
- In latency, a device-side flag (3-5 us) beats an L0 event (21-35 us), and
  both beat oneCCL (90 us).

### 7.2 Push-AR: source, evidence and status

Source and binaries live in `vllm/contrib/vllm_push_allreduce/`:
- `118_xpu_push_ar_graph.cpp` is the source of
  `prebuilt/libxpu_push_ar_graph.so`.
- `119_xpu_push_ar_eager_async.cpp` builds `libxpu_push_ar_eager.so`.
- `_push_ar_patch.py` monkeypatches `XpuCommunicator.all_reduce`.
- The prebuilt `.so` is listed as modified in the current git status. It is
  ABI-bound to an older torch/oneAPI. Rebuild it from source, never reuse the
  binary (`docs/20261014_flashnext_strata_optimization.md:499-500`).

The mechanism (`118_xpu_push_ar_graph.cpp`):
- Each rank owns one scratch buffer allocated with `zeMemAllocDevice`.
- `zeMemGetIpcHandle` produces a dma-buf fd, sent over a named Unix socket
  with SCM_RIGHTS (lines 76-88). The peer opens it with `zeMemOpenIpcHandle`
  (line 131).
- A shared-memory host barrier is used during setup.
- An IPC event pool spans both devices (`zeEventPoolCreate(...,2,both)`,
  line 162). With a one-device pool, the waiter segfaults (K.5).
- Each all-reduce is a push kernel (posted write into the peer's scratch),
  then `ext_codeplay_enqueue_native_command`, then a local reduce kernel. The
  native command does: signal mine, `zeCommandListAppendWaitOnEvents(peer)`,
  then reset (lines 212-227).
- The native command records into torch's XPUGraph only if it is issued on
  the current capture stream (K.6).
- Eager prefill uses the host-barrier `ar_allreduce_ptr_dt` instead.

Results:

| Result | Evidence |
| --- | --- |
| 27B W8A8 TP=2 live serve | +48-64% throughput and 3.35x TTFT vs oneCCL (J.14) |
| Captured decode push (GRAPH=1) | +8-10% per-stream decode over prefill-only push; promoted to that shelf default (K.8, K.9) |
| NVFP4 27B TP=2 prefill overlay | 3.3x cold prefill, decode-neutral (memory note nvfp4-tp2-pushar-prefill, commit 67750c9) |
| Eager per-call host-sync variants (MTP all_gather redirect) | 2.4-2.5x slower decode. Any per-call host sync loses to oneCCL at about 631 small collectives per step (memory note decode-is-allreduce-bound) |
| sglang pure-SYCL push AR with EU-spin, pull or flush sync | 10-30% coherent under real serve cadence. Bulk payload arrival was the unfixable part on DPC++ 2025.3. Dead end (memory note capturable-ar-dead-end; `sglang/patches/push_ar_xpu.py`) |
| Loaded June vLLM/XCCL stack | Preinit fixes asymmetric IPC import on both ranks. The native push graph call then stalls at the first submission after both ranks enter capture. Do not run a full-model push arm until that oracle captures and replays (`JOURNAL.md:3186-3230`; `docs/20260825_steve_stack_component_ledger.md:815-818`; AGENTS.md "Current transfer findings") |

The stall happened on the `ext_codeplay_enqueue_native_command` plus L0-event
path, inside a process that also holds an XCCL communicator. The standalone
torch harness (K.6) and the 27B shelf (K.7-K.9) pass. The loaded June stack
does not, and the root cause is open.

### 7.3 Strata shares the lineage

Strata's EP transport is the push-AR idea, re-derived for a one-way expert
exchange:
- S7/S8 reviewed the vLLM 118/119 source and P2P_GPU J/K and reused the
  posted-write idea, not the binary
  (`docs/20261014_flashnext_strata_optimization.md:499-537`).
- `STRATA_EP_PREFILL_REDUCE` came first (commit `0897c76c`). Card 1 reduces
  its expert rows to one FP32 row per token, so the return payload is k times
  smaller.
- `STRATA_EP_PREFILL_PUSH` came next (commit `78cea7a1`). The sum region moved
  to card 0's memory, so card 1 posted-writes it and card 0 reads it locally
  with uncached loads.
- Commit `68c5ad51`, the S28 system-scope barrier, fixed the publication bug.
- Night additions: EARLY (device doorbell), F16SUM (half the return bytes)
  and HALVES.
- Upstream: recutita/Strata PR1 (PUSH) and PR2 (barrier scope)
  (`docs/20261015_strata_ep_publication_scope.md`).

Decode EP used push from the start: `ep_push` into card 1's inbox, and
`ep_send_rows`/`ep_send_psum` into card 0's region.

Two differences from push-AR matter for vLLM:
1. Push-AR is a symmetric two-way all-reduce. Strata's exchange is
   asymmetric: card 0 pushes inputs, card 1 pushes results. Neither card ever
   reduces into remote memory, which would be the 2.36 GB/s addpush trap
   (J.5).
2. Push-AR syncs with L0 events through a native command. Strata syncs with
   kernel-only device flags, using uncached write/read cache-control hints and
   a system-scope barrier. That needs no `ext_codeplay` node, no event pool
   and no host involvement, and its handshake is 5-10x lower latency.

### 7.4 Decision per EP exchange

| Exchange | Size per layer | Transport | Sync | Graph | Why not the others |
| --- | --- | --- | --- | --- | --- |
| Decode dispatch, card 0 -> card 1 (ids, weights, plan, q8_1 rows) | about 3 KB per row + 1.6 KB; C2 6 rows = 19 KB | Kernel posted write into card 1's inbox (`ep_push_w`) | Device epoch flag with an uncached store; card-1 EU spin with uncached loads, bounded, error word in host memory | Captured as an ordinary kernel in both cards' graphs | L0 event = 21-35 us per handshake x 96 per window = 2-3.4 ms; oneCCL = 90 us x 96 = 8.6 ms; host-staged has a host sync per layer |
| Decode combine, card 1 -> card 0 (per-token psum) | 10 KB per row FP32 (5 KB FP16) | Kernel posted write into card 0's region (`ep_send_psum`) | Done-counter; the last work-group raises the flag; card 0 reads with uncached loads | Captured | Same as dispatch. Pull (card 0 reading card 1's rows) is 3.5x slower |
| Prefill dispatch (FP16 activations, T x 5 KB) | 4K tokens = 20 MB | Kernel posted write (`ep_push_signal`, 64 work-groups, vector4) at about 11 GB/s, about 1.8 ms per 4K chunk per layer, overlapped by HALVES | Device doorbell (EARLY) | Eager | oneCCL P2P1 SYCL matches the BW (10.3 GB/s) but needs XCCL in the process, the risk documented in 7.2; the copy engine is no faster (11.12) |
| Prefill combine (FP16 sum + scale per token) | T x 5 KB | Kernel posted write (`ep_prefill_reduce_h`, push) into card 0 | Done-counter doorbell | Eager | The remote-read consumer was 36x slower in S10 (11078 vs 307 us per layer) |
| Setup, IPC/peer mapping | -- | Single process: dedicated `zeMemAllocDevice` plus an IPC open into the other context (Strata) or a same-context peer pointer (J.1/J.5) | Host, once at load | -- | No SCM_RIGHTS sockets are needed with one process |
| Fallback sync only | -- | Push-AR's L0 IPC event (two-device pool, CS wait, native command) | -- | Capturable only outside a loaded XCCL context until the 7.2 stall is understood | Use only if the M0 oracle shows EU-spin staleness inside torch's context |

oneCCL is not used for any EP exchange. The sidecar design never creates an
XCCL communicator, so the known queue-handoff and first-submission failure
modes cannot occur.

Host-staged copies are used only for one-time weight loading and for KV/GDN
parking to RAM. Parking is a separate connector (G1).

### 7.5 Missing measurements (probes for later, M0)

1. Device-timestamped one-way flag latency and round trip for the doorbell
   inside torch's context, in both directions (card 0 -> 1 vs card 1 -> 0).
   Strata measured only inside its own runtime.
2. Small-message bandwidth curve from 2 KB to 64 KB for the kernel push, by
   work-group count. Decode messages fall in the latency regime; 16 KB took
   1.47 GB/s host-timed in S8.
3. Concurrent bidirectional push. The PCIe links are dual-simplex, but
   Infinity Fabric is shared, and prefill HALVES overlaps both directions.
4. The cost of uncached loads on the consumer per 10 KB, compared with plain
   loads. Strata's psum combine uses plain loads behind the flag
   (`STRATA_NIGHT_PSUM_UNCACHED=0`), which is a correctness risk to settle
   with the S28 oracle.
5. torch-xpu context model, and whether a same-context peer pointer works for
   kernel stores into the other card's memory under torch's allocator.
6. Doorbell transport inside a process that also has an XCCL communicator,
   only if phase 3 (DP=2) is ever built. This mirrors the 7.2 stall.
7. Same-die slot A/B per the topology doc's procedure. This is the only
   hardware lever. It is optional and needs a power-off with the user present.
8. Why card 1's routed kernels run 20-30% slower (clocks, the root complex,
   or memory). This bears on whether card 1 should get more traffic.
