# sglang-XPU: two upstream blockers gating Qwen3.6-27B decode performance (2026-06-27)

Context: Qwen3.6-27B (`qwen3_5` hybrid Gated-DeltaNet) serves CORRECTLY on 2x Intel Arc Pro B70
(Battlemage/Xe2) via sglang-XPU (no GDN NaN, unlike vLLM-0.23). Stable single-stream decode is
~9.2-9.4 t/s -- the eager ceiling. Both paths that would break that ceiling are blocked by XPU-specific
bugs, NOT by the model or our patches. This documents them for upstream fixes. Repro context + the full
campaign data: `sglang/PERF.md`. Our wiring: `sglang/patches/woq_shim.py`, `sglang/patches/mtp_tree_xpu.py`.

## Blocker 1: torch-xpu CUDA-graph replay degrades (command-stream accumulation)

We enabled decode CUDA-graph capture on XPU (torch.xpu.XPUGraph) by adding "xpu" to the in-tree decode-graph
device list in `model_runner.init_cuda_graphs` and redirecting `torch.cuda.{CUDAGraph,graph,graph_pool_handle}`
-> `torch.xpu` (with a `graph(cuda_graph=...)` -> `graph(xpu_graph,...)` signature adapter). Capture SUCCEEDS
for the hybrid GDN model and decode logs `cuda graph: True` at ~23.6 t/s initially (2.5x the ~9.4 eager).

BUT the replayed-graph decode rate DEGRADES over a soak (~23 -> ~8 t/s on a single serve) with periodic
multi-second stalls, so the END-TO-END rate is ~7.6 t/s (worse than eager). This matches the known torch-xpu
/ Level-Zero graph-replay command-stream accumulation (we hit the identical pattern in vLLM PIECEWISE: 26->7
t/s over a soak). No env knob (UR_L0_USE_IMMEDIATE_COMMANDLISTS, etc.) fixed it; recapture crashes.
=> Needs an upstream torch-xpu / oneAPI L0 fix so XPUGraph replay does not accumulate command-stream state.
Until then sglang should keep XPU decode-cuda-graph OFF (we gate ours behind B70_XPU_CUDAGRAPH=1, off by default).

## Blocker 2: sglang-XPU spec-decode (EAGLE/NEXTN) verify forward-batch passes a full-pool out_cache_loc

We implemented the two CUDA-only EAGLE tree kernels (`build_tree_kernel_efficient`, `verify_tree_greedy`) as
pure-torch XPU fallbacks for the linear-chain case (topk=1), ported from `sgl-kernel/csrc/speculative/
eagle_utils.cu` and validated correct (`sglang/patches/mtp_tree_xpu.py`). They install and RUN (build_tree
ENTER/EXIT confirmed; mask + retrieve graph built).

The spec-decode VERIFY forward then crashes in the KV-cache write, identically for BOTH attention backends
(intel_xpu `xpu_backend.py:forward_extend` and triton `triton_backend.py:forward_extend`) and for both
`--page-size 64` and `--page-size 1`:

    _set_kv_buffer_impl: k_cache[indices] = k
    RuntimeError: expanded size (75904) must match existing size (2) at dim 1.
    Target [1, 75904, 4, 256], tensor [2, 4, 256]

i.e. the verify batch's `out_cache_loc` (`indices`) arrives as a 2D FULL-POOL index `[1, num_kv_slots]`
instead of the `num_draft_tokens` (=2) newly-allocated slots, so writing the 2 tree-token K/V broadcasts
against the whole pool and throws. Normal (non-spec) decode writes are fine (`cache_k=(1,4,256) loc=(1,)`).
=> The XPU spec-decode verify forward-batch metadata (out_cache_loc construction for the draft tree) is wrong;
this is upstream sglang-XPU spec-decode plumbing, independent of the tree kernels (which are solved here).
Fixing it would unlock EAGLE/NEXTN spec-decode (est. ~15 t/s stable for chain num-steps=1) on B70.

## Repro (single B70 card)

    # image sglang-xpu:woq (= sglang-xpu:bmg + auto-round-lib + woq_shim + mtp_tree_xpu)
    # Blocker 2 (spec-decode): grafted int4+vision+MTP ckpt, topk=1 chain, B70_XPU_MTP=1
    EXTRA="--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 2 --speculative-draft-attention-backend triton \
      --max-running-requests 4 --disable-cuda-graph --skip-server-warmup"
    DENV="B70_XPU_MTP=1 B70_MTP_DEBUG=1"  # B70_MTP_DEBUG traces build_tree + the failing KV write
    # -> serve loads healthy; first generation 500s at _set_kv_buffer_impl (see Blocker 2).
