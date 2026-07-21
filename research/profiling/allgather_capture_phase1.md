# MTP all_gather capture -- Phase 1 origin + capture scoping (2026-07-21)

Config: Qwen3.6-27B NVFP4 TP=2, vLLM 0.25.1, MTP spec=5, PIECEWISE (XPUGraph) capture.
No-GPU source read (container `b70_daily_0` vLLM install + repo). Builds on
`pushar_decode_gap.md`, `eager_async_ar_plan.md`, `torch213_2992_scoping.md`. ASCII only.
Container vLLM root below abbreviated `V=` `/opt/venv/lib/python3.12/site-packages/vllm`.

## TL;DR bottom line

1. ORIGIN: the ~631 `oneccl_allreduce_pcie` under `vllm::all_gather` are NOT per-layer and NOT
   oneCCL-forced. They are the MTP spec-decode tensor-parallel gathers -- dominated by the drafter's
   per-draft-step FULL-VOCAB logits gather (`compute_logits` -> `LogitsProcessor._gather_logits`,
   `V/model_executor/layers/logits_processor.py:83`) plus the target verify logits gather -- all
   realized as a zero-pad + `dist.all_reduce` by the DD's OWN shim, sitecustomize block (3)
   `_all_gather_via_allreduce` (`vllm/nvfp4/patches/sitecustomize.py:306-328`).
2. CAPTURED-OR-EAGER: EAGER, and structurally so. These gathers live in the model-runner /
   spec-decode proposer Python orchestration BETWEEN piecewise graph replays, not inside the
   captured model forward (which is split only at attention/GDN ops -- the per-layer RowParallel
   all_reduce is inside the graph and is ALREADY push-AR/do_ar). The shim's own note
   (`sitecustomize.py:316-320`, dated 2026-07-08) already states "this all_gather runs EAGER
   (capturing=False)" and that routing it through push-AR does not help.
3. BEST PHASE-2 LEVER: reduce the gather COUNT+SIZE, do not fight the transport. The drafter uses
   full-vocab `compute_logits` (~10.5M-elem gather, ~0.75 ms) every draft step because
   `use_local_argmax_reduction=False` (`V/config/speculative.py:139`) and the qwen3 MTP models do
   not implement `get_top_tokens`. Switching the drafter to the vocab-parallel argmax gather shrinks
   each drafter gather from O(batch*vocab/tp) to O(batch*2*tp) -- ~76000x fewer bytes -- and removes
   ~80% of the gather COUNT (the K-1 drafter gathers), capture-free and transport-free. Second lever:
   a concat-style L0-IPC "push all_gather" (copy, no reduce, half the bytes) replacing the padded
   all_reduce for whatever gathers remain.
4. CAPTURE go/no-go: capturing the gather so `ar_allreduce_graph`/`do_ar` records it is a **NO-GO**
   -- structural. The gathers are in eager, data-dependent proposer/runner control flow, not in the
   piecewise-captured forward; pulling them in needs full-graph spec-decode capture that vLLM v1 does
   not do on XPU and that the drafter's Python control flow forbids.
5. Decisive result: the last DECODE lever is NOT "capture the all_gather" (closed, no-go). It is
   COUNT/SIZE reduction of the MTP logits gathers (argmax-reduction), which is orthogonal to and
   stacks with push-AR, and is lower risk than any eager-transport substitution.

## Q1 -- exact origin of the ~631 `vllm::all_gather`

### The realization is a DD shim, not oneCCL and not the base communicator
- `XpuCommunicator` (`V/distributed/device_communicators/xpu_communicator.py`) has NO `all_gather`
  override (it defines `all_gatherv`, `reduce_scatter`, `gather`, ... but not `all_gather`).
- So stock `all_gather` would fall to `DeviceCommunicatorBase.all_gather`
  (`V/distributed/device_communicators/base_device_communicator.py:194`), which is CONCAT-style
  (`dist.all_gather_into_tensor`, NO all_reduce).
- BUT the DD REPLACES it: `vllm/nvfp4/patches/sitecustomize.py:328`
  `XpuCommunicator.all_gather = _all_gather_via_allreduce`. That function
  (`sitecustomize.py:306-326`) builds a `[world_size, *input]` zero buffer, writes its own shard,
  and calls `dist.all_reduce(buf)` (line 321) -> SUM fills every slot -> concat reshape. THIS is the
  `oneccl_allreduce_pcie<bfloat16>` the trace saw under `vllm::all_gather`. So `pushar_decode_gap.md`
  fact 2 is exactly right about the mechanism; the source is the DD's own capture-safety shim.

### Which `vllm::all_gather` call sites feed it (MTP spec-decode)
Dispatch chain: `tensor_model_parallel_all_gather` -> `get_tp_group().all_gather` ->
`torch.ops.vllm.all_gather` (`V/distributed/parallel_state.py:160,343`) -> `_all_gather_out_place`
-> `device_communicator.all_gather` -> the shim. The producing call sites active in this config:

1. **Drafter per-step logits gather (dominant COUNT + SIZE).** MTP proposer
   `V/v1/spec_decode/llm_base_proposer.py`: `_greedy_sample` (line 428) -> line 438
   `self.model.compute_logits(hidden_states).argmax(dim=-1)` -> `LogitsProcessor._get_logits` ->
   `_gather_logits` (`logits_processor.py:60,83`, `use_all_gather=True` on XPU,
   `V/platforms/interface.py:1080` returns True). This is a FULL-VOCAB gather
   ([batch, vocab_size/tp], ~10.5M elems bf16 ~21 MB after the 2x pad) and it runs once per EXTRA
   draft step -- the loop `for token_index in range(num_speculative_tokens - 1)`
   (`llm_base_proposer.py:682`), i.e. K-1 = 4 times for spec=5, plus the initial
   `_sample_draft_tokens`.
2. **Target verify logits gather.** The target model's `compute_logits`
   (`V/model_executor/models/qwen3_5.py:353`) -> same `_gather_logits` full-vocab all_gather, once
   per engine step (over the 1+K verify tokens).
3. **MTP head fc gather (smaller).** `ColumnParallelLinear(gather_output=True)`
   (`V/model_executor/models/qwen3_5_mtp.py:95-98`) -> `linear.py:562`
   `tensor_model_parallel_all_gather(output_parallel)`. This is HIDDEN-sized ([tokens, 5120]), far
   smaller than the vocab gathers, so it is not the cost driver.

Ruled out: the qwen3_next per-layer sequence-parallel-MoE all_gather
(`V/model_executor/models/qwen3_next.py:118`, gated on `use_sequence_parallel_moe`) is NOT the
source. If it were active it would sit INSIDE the captured forward (between attentions) and appear
captured, contradicting the trace's 630/631 parent=None. The eager gathers are the logits/argmax/fc
gathers in the runner/proposer, exactly as the shim note says.

### Count-per-step caveat (honest)
`pushar_decode_gap.md` cites ~126/step from "631 / 5 steps". A steady-state decode trace more
plausibly spans ~50-60 engine steps, giving ~10 gathers/step (K drafter full-vocab gathers + 1
target + fc), which matches the structure above and the shim's "~10.5M-elem" size note better than a
per-layer collective would. Either way the AGGREGATE (631 gathers, 472 ms, 39%) and the eager nature
are what drive the conclusion; the exact per-step divisor does not change any verdict. The per-call
0.75 ms is large precisely because each is a vocab-sized (2x-padded) all_reduce, NOT a small
hidden-sized one (contrast do_ar at 0.034 ms on hidden-sized tensors).

## Q2 -- why pad+all_reduce and not concat; is there a cheaper concat path

The pad+all_reduce is a DELIBERATE capture-safety trade, documented at
`vllm/nvfp4/patches/sitecustomize.py:291-297`:
- oneCCL 2021.17's ALLGATHER scheduler has NO SYCL-graph-recordable implementation. Left in a
  captured region it CRASHES capture; ejected to eager it breaks vLLM's captured-piece input-address
  contract (stale-read garbage at the piece boundary).
- oneCCL's ALLREDUCE **is** SYCL-graph-recordable (under `CCL_ENABLE_SYCL_KERNELS=1`), so they
  reimplement all_gather as zero-pad + `dist.all_reduce`. Accepted cost: world_size x bytes (2x at
  TP=2), "which MTP amortizes."

So the base's concat `all_gather_into_tensor` was intentionally abandoned for capturability, not
because oneCCL cannot gather. Cheaper concat options:

- **oneCCL P2P-copy concat allgather: BLOCKED.** P2P is disabled on this box
  (`CCL_TOPO_P2P_ACCESS=0`, the H.13 reboot-wedge). Without P2P, oneCCL realizes allgather over PCIe
  as the same allreduce-class collective; there is no cheap peer-copy path through oneCCL here.
- **push-transport concat all_gather (L0-IPC copy, NO reduce): feasible, half the bytes.** The
  push-AR .so already owns the cross-card L0-IPC posted-write + IPC-event machinery. A gather is
  strictly SIMPLER than do_ar: each rank writes its shard once to the peer's output-buffer offset
  (no zero-pad, no SUM) -> both ranks hold the concat. That halves bytes vs the padded all_reduce AND
  uses the fast transport. It is CORRECT (pure copy, no numeric reduction). BUT it does NOT change
  the capture story: these gathers are eager (Q1/Q3), so a captured push-gather cannot reach them; an
  EAGER push-gather inherits the one-host-sync floor of `eager_async_ar_plan.md` (the L0->torch-queue
  result-ready handoff cannot be recycled race-free without one host sync). So a push concat gather is
  a real but bounded win: half bytes + fast copy, still one host sync/call.

## Q3 -- capture strategy or no-go

**NO-GO to bring these gathers into a captured region.** Reasons, structural:

- The piecewise (XPUGraph) capture covers ONLY the model forward, split at attention/GDN ops
  (`splitting_ops` = `_attention_ops`, `V/config/compilation.py:1122`; the DD passes exactly the attn
  op list, `vllm/nvfp4/serve_nvfp4_27b.sh:86`). RowParallel all_reduce sits inside those subgraphs
  and is already captured/push-AR. `compute_logits` + sampling + the whole MTP drafter loop run in
  the model runner / `llm_base_proposer.propose` as EAGER Python AFTER the forward, between graph
  replays -- so their gathers are eager by construction.
- The drafter loop (`llm_base_proposer.py:682`) is data-dependent control flow (per-step attention
  metadata rebuild, rejection handling `seq_lens -= num_rejected_tokens_gpu` at line ~737, dynamic K,
  argmax feedback). vLLM v1 on XPU does not full-graph-capture this, and it cannot be piecewise-split
  to enclose the gather without enclosing that control flow. The 2026-07-08 shim note confirms the
  in-graph drafter collective is the row-parallel all_reduce (already push-AR); the logits/fc gathers
  are the eager remainder.
- Consequence: `torch213/#2992` (record oneCCL allgather as a graph node) would only help if these
  ran during capture -- they do not -- so it is doubly moot (also slower than push-AR per
  `torch213_2992_scoping.md`).

### Options ranked by tractability x wedge-risk x payoff

| # | Option | Payoff | Wedge risk | Tractability | Verdict |
|---|--------|--------|-----------|--------------|---------|
| A | Drafter vocab-parallel argmax gather (`use_local_argmax_reduction`) | HIGH: kills ~80% of gather COUNT + shrinks each drafter gather ~76000x in bytes | NONE (no new collective/transport) | needs a small model-side `get_top_tokens` (models lack it) + config flag | **TRY FIRST** |
| B | Concat-style L0-IPC push all_gather (eager, one host sync) replacing the padded all_reduce | MED: half bytes + fast copy, but one host sync/call | MED (cross-card IPC latch, same class as eager-async AR) | reuses push-AR machinery; new gather op + shim wrap | second |
| C | Route padded all_reduce through the built eager-async push-AR (119, `PUSH_AR_ALLGATHER_ASYNC`) | LOW-MED: still 2x bytes (strictly dominated by B) | MED (same latch) | already built | only if B is not built |
| D | Capture the gather so do_ar records it | would be HIGH | -- | structurally impossible here | **NO-GO** |
| E | torch 2.13 / #2992 capturable oneCCL allgather | LOW (slower than push-AR, and gather is eager anyway) | HIGH (rebuild-everything) | 4-7 sessions | NO |

A and B/C STACK: A cuts count/size, B/C accelerates the transport of whatever remains (chiefly the
one target-verify full-vocab gather, which A cannot remove -- rejection sampling needs full target
logits). Do A first; it is the largest, lowest-risk win.

## Q4 -- concrete Phase-2 plan for the coordinator

### Smallest first test: Option A (drafter argmax-reduction gather)
Rationale: capture-free, transport-free, no new .so, no wedge risk, attacks the dominant COUNT+SIZE
directly. The drafter is greedy on XPU (MTP is greedy-only), which is exactly `get_top_tokens`'
requirement.

Change surface (files):
1. `V/config/speculative.py:139` path -- set `use_local_argmax_reduction=True` for the mtp method.
   Prefer wiring it through the serve `--speculative-config` JSON (`serve_nvfp4_27b.sh:54`, add
   `"use_local_argmax_reduction":true`) rather than editing the wheel; verify the field is accepted
   for `method:"mtp"` (it is a generic SpeculativeConfig field; the greedy_sample use at
   `llm_base_proposer.py:430` is method-agnostic).
2. The qwen3 MTP models do NOT expose `get_top_tokens` (only `compute_logits`:
   `qwen3_5_mtp.py:275`, `qwen3_next_mtp.py:214`). Add a thin `get_top_tokens(self, hidden_states)`
   that delegates to `self.logits_processor.get_top_tokens(self.lm_head, hidden_states)` (the
   LogitsProcessor already implements it, `logits_processor.py:107-117`). Do this as a
   sitecustomize monkeypatch block (new block, default-gated) so no wheel edit / rebuild -- mirror the
   existing block (3) style.
   - If adding get_top_tokens is undesired, a lighter variant: keep compute_logits but set
     `use_all_gather=False` is NOT useful on XPU (its `gather` also all-gathers,
     `xpu_communicator.py:179`), so get_top_tokens is the real path.

Gates (in order):
- COHERENCE FIRST: `bin/serve-sweep --smoke` green, then a HumanEval+ spot-check on the NVFP4 27B DD
  -- this changes draft-token selection numerics slightly (argmax over gathered value/index pairs vs
  argmax over full gathered logits are byte-identical for greedy, so acceptance should be UNCHANGED;
  verify accept rate does not drop).
- TRACE: re-run `decode_optrace.sh` / `trace_driver.sh`; expect the `oneccl_allreduce_pcie` COUNT to
  drop by ~K-1/step (drafter gathers gone) and the remaining ones to be tiny (argmax pairs) except
  the single target-verify vocab gather. `parse_trace.py` collective share should fall from 43%.
- BENCH: `research/profiling/bench_decode_completions.py` decode t/s vs current DD; TTFT ~unchanged.
- Rollback: unset the flag / disable the get_top_tokens block -> byte-identical DD.

### Second test (stacks on A): Option B concat push all_gather
Only after A lands. Add an `ar_allgather_eager` (copy, no reduce, half bytes, one host sync) to the
push-AR .so and wrap `XpuCommunicator.all_gather` (behind a new env, e.g. `PUSH_AR_ALLGATHER_CONCAT=1`)
to use it for the remaining big gather (target verify). Reuse the eager_async wedge-guard recipe:
microbench-first (2-proc, byte-exact vs reference concat), `B70_AUTO_RESET=1`, DD down, do not chain
attempts. Same coherence -> trace -> bench gates. If the microbench hangs/wedges: NO-GO, stop, reset.

### What NOT to do
- Do not pursue capturing the gather (D) or torch 2.13/#2992 (E) -- closed above.
- Do not enable `CCL_TOPO_P2P_ACCESS=1` to get a concat oneCCL allgather -- H.13 reboot-wedge.

## Files
- This analysis: `research/profiling/allgather_capture_phase1.md`.
- Shim that realizes the gather: `vllm/nvfp4/patches/sitecustomize.py:291-332` (block 3).
- Origin call sites (container): `V/v1/spec_decode/llm_base_proposer.py:428-438,682`;
  `V/model_executor/layers/logits_processor.py:60-117`;
  `V/model_executor/models/qwen3_5_mtp.py:95-98,275`; `V/model_executor/layers/linear.py:560-562`.
- Serve config: `vllm/nvfp4/serve_nvfp4_27b.sh:77-86,106-114`.
