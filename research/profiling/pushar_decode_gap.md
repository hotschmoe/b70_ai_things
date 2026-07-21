# push-AR decode gap -- why ~39% of decode is still on slow oneCCL

Date: 2026-07-21. Config: Qwen3.6-27B NVFP4 TP=2, vLLM 0.25.1, MTP5 captured decode,
torch-profiler trace `nvfp4_decode_rank0.json.gz` (rank 0, steady-state decode).
No-GPU analysis (trace + source read only). ASCII only.

## TL;DR root cause (NOT contiguity)

The slow oneCCL all-reduce is emitted INSIDE vLLM's `all_gather` collective, not the
tensor-parallel `all_reduce`. On XPU the `all_gather` custom op realizes the gather as a
zero-pad + `torch.distributed.all_reduce(SUM)` over the TP device_group. That direct
`dist.all_reduce` bypasses `XpuCommunicator.all_reduce` -- the ONLY method the push-AR
patch overrides -- so it lands on oneCCL. The push-AR patch is working perfectly on the
path it can see; it simply never sees the all_gather path.

The task's contiguity hypothesis is REFUTED (evidence below): 100% of the all-reduces
that go through `XpuCommunicator.all_reduce` already hit push-AR; there is ZERO
contiguity fallback. `PUSH_AR_FORCE_CONTIG` would be a no-op.

## Evidence from the trace (decode, rank 0)

Category breakdown (device-kernel time, 1208.36 ms total):

| category            | ms     | share |
|---------------------|--------|-------|
| allreduce/collective| 519.82 | 43.0% |
| linear-gemm         | 473.92 | 39.2% (BW roofline, not improvable) |
| gdn/mamba-scan      | 111.25 |  9.2% |

Two DISJOINT all-reduce paths (counts are exact, 1:1 within each path):

| host op (cpu_op)     | count | device kernel                        | count | device ms |
|----------------------|-------|--------------------------------------|-------|-----------|
| `vllm::all_reduce`   |  760  | `do_ar<bfloat16>` (push-AR, 2 lambdas)| 760x2 | 27.15     |
| `vllm::all_gather`   |  631  | `oneccl_allreduce_pcie<bfloat16>`    |  631  | **472.20** (39.1%) |

Key facts that pin the cause:

1. `vllm::all_reduce` count (760) == `do_ar` count (760). EVERY tensor-parallel all-reduce
   is on push-AR. None fall back. So non-contiguity/dtype/size fallback is NOT happening.
2. The 631 `oneccl_allreduce_pcie` come from 631 `c10d::allreduce_` host ops, and ALL 631
   of those are lexically nested inside a `vllm::all_gather` scope (parent-op analysis:
   631/631 enclosed by `vllm::all_gather`). They do NOT pass through `vllm::all_reduce`
   at all (if the push-AR method were falling back, its cpu_op count would be 760+631; it
   is only 760). So this is a SEPARATE collective, not a fallback.
3. `vllm::all_gather` runs EAGER at top level: 630/631 have parent op = None (i.e. between
   the piecewise `CompiledFxGraph` subgraph calls, not inside a captured region). This is
   the key capture-safety fact (below).
4. The oneCCL kernel is a single SUM variant (`oneccl_allreduce_pcie<bfloat16, Rt64_128_PCIE, 16>`)
   -- a plain sum all-reduce, so a push-AR sum substitution is numerically correct.
5. Prefill has only 21 `vllm::all_gather` (vs 408 `vllm::all_reduce`), which is why the
   prefill trace is "almost all push-AR" -- the all_gather cost is DECODE-specific.
6. The decode gdn kernels are `gated_delta_rule_SPEC_kernel` / `causal_conv1d_SPEC_kernel`
   (x4992) -- MTP spec-decode is active. The 631 all_gathers scale with the MTP/spec path
   (30x the prefill count), consistent with a per-draft-step gather in the spec-decode
   verify. (Secondary TODO: confirm exactly which layer/step issues the gather -- reducing
   its COUNT would be an orthogonal win, but is not required for the fix here.)

## Why the existing patch misses it

`_push_ar_patch.py` overrides `XpuCommunicator.all_reduce` only (README: "Only all_reduce
is accelerated. reduce_scatter / all_gather ... fall back to oneCCL"). vLLM's
`XpuCommunicator.all_gather` (base `DeviceCommunicatorBase.all_gather`, and the 0.25.1
variant seen in the trace) issues its own `dist.all_reduce` / `dist.all_gather*` directly
on `self.device_group`; it never calls `self.all_reduce`, so the monkeypatch cannot reach
it. This was a KNOWN documented limitation -- the trace just shows it is the dominant
decode cost once the all_reduce path is already fast.

## The fix (env-gated, default OFF): route the all_gather-internal SUM all-reduce to push-AR

Implemented additively in `_push_ar_patch.py`, gated on `PUSH_AR_ALLGATHER=1`
(default off -> byte-for-byte the proven behavior). Mechanism:

- Install ONE permanent shim over `torch.distributed.all_reduce` that is INERT unless a
  thread-local flag `ag_self` is set (so it only ever acts on the all-reduce emitted
  inside an XpuCommunicator gather -- surgical blast radius; the startup oneCCL warmup and
  every other collective are untouched).
- Wrap `XpuCommunicator.all_gather` / `all_gatherv` to set `ag_self=self` for the dynamic
  extent of the call. Inside that extent, a SUM, non-async, device_group all-reduce on a
  contiguous bf16/fp16/fp32 tensor <= MAXB is done IN PLACE via the existing host-barrier
  `ar_allreduce_ptr_dt` (same kernel/path as the proven eager prefill all_reduce), else it
  falls through to the original oneCCL.

### Capture-safety argument (explicit)

- The all_gather (and its internal all_reduce) run EAGER at top level (evidence fact 3:
  630/631 parent=None, between compiled subgraphs, not inside a captured region).
  `torch.xpu.is_current_stream_capturing()` is therefore False there, so the host-barrier
  push-AR runs exactly like the proven eager prefill path. No uncapturable op is inserted
  into any captured region.
- The shim additionally guards on `not is_capturing()`: if an all_gather ever DID run
  during capture, we DECLINE (fall to oneCCL, which is graph-recordable) rather than inject
  the host barrier into the graph. So the change is strictly capture-safe by construction --
  it can only ever move an EAGER oneCCL call onto push-AR.
- The permanent `dist.all_reduce` shim never alters behavior during capture (ag_self unset,
  or is_capturing True -> original), and never touches the already-fast `vllm::all_reduce`
  path (that path does not call `dist.all_reduce`). It also leaves the push-AR method's own
  oneCCL fallback (non-contig/oversize) on oneCCL, unchanged.
- In-place semantics: `dist.all_reduce` mutates its tensor in place; `ar_allreduce_ptr_dt`
  already reads+writes the same buffer (the existing eager path clones first only because
  its API returns a new tensor). Calling it directly on the tensor's `data_ptr()` is the
  correct in-place equivalent.

### Correctness caveat to verify on GPU

The substitution assumes the all_gather-internal all-reduce is a plain SUM over
zero-padded per-rank shards (the standard "all_gather via all_reduce"). Trace fact 4 (single
SUM oneCCL kernel variant) supports this, but it MUST be coherence-gated (sweep) before
trust. If the gather is realized some other way in 0.25.1 and the shim does not catch it,
the re-trace simply shows unchanged oneCCL (no regression, no speedup) -- see below.

## Expected win

Moving 472.20 ms of oneCCL (39.1% of decode device time, ~0.75 ms/call) to push-AR
(~0.034 ms/call, ~22x) collapses it to ~21 ms. Decode device-kernel time 1208 -> ~757 ms
(collective share 43.0% -> ~7%). Real decode t/s gain will be smaller than 1.6x because
decode is partly launch-bound, but the collective is on the serial critical path so a
material gain is expected. This reframes the decode bottleneck: after push-AR on all_reduce,
the #1 remaining cost is the UN-accelerated all_gather, not any all_reduce fallback.

## Test recipe (for the coordinator)

Enable (add to the NVFP4 27B serve env, on top of the existing push-AR DD config):

    PUSH_AR_ALLGATHER=1

i.e. keep the DD's `PUSH_AR=1 PUSH_AR_GRAPH=1 PUSH_AR_SO=.../libxpu_push_ar_graph.so ...`
and ADD `PUSH_AR_ALLGATHER=1`. On rank 0 startup you should see:
`[push_ar] ALLGATHER redirect ENGAGED ...`.

Confirm the win:
1. Re-trace decode (same `decode_optrace.sh` / `trace_driver.sh` path that produced the
   current trace). In the new trace:
   - `oneccl_allreduce_pcie` ms should COLLAPSE (631 calls -> few or zero); the calls that
     remain should now be `do_ar` under `vllm::all_gather`.
   - `do_ar` count should rise by ~631/step-set; `parse_trace.py` collective share should
     drop from 43% toward <10%.
2. Coherence gate FIRST: `bin/serve-sweep --smoke` (or the shelf sweep) must stay green --
   this is a collective substitution, so verify coherence before trusting speed.
3. `bench_code` decode t/s vs the current DD (expect a decode t/s rise; TTFT/prefill
   unchanged since prefill has almost no all_gather).

If the re-trace shows oneCCL UNCHANGED (shim did not catch it): the 0.25.1 `vllm::all_gather`
custom op dispatches its `dist.all_reduce` through a path other than
`XpuCommunicator.all_gather`. In that case grep the running container for where
`vllm::all_gather` is registered and which method emits `dist.all_reduce`
(`grep -rn "def all_gather\|dist.all_reduce\|all_reduce(" .../vllm/distributed/`), and point
the wrap at that method. The shim itself is inert until `ag_self` is set, so a miss is a
no-op, never a regression.

Rollback: unset `PUSH_AR_ALLGATHER` (or set `PUSH_AR_ALLGATHER=0`). The whole block is
skipped and `torch.distributed.all_reduce` is left untouched -- byte-for-byte the current DD.

## Files

- Patch: `vllm/contrib/vllm_push_allreduce/_push_ar_patch.py` (additive ALLGATHER block).
- This analysis: `research/profiling/pushar_decode_gap.md`.
- Trace: scratchpad `nvfp4_decode_rank0.json.gz` (decode), `nvfp4_prefill_rank0.json.gz`.
