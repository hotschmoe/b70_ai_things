# torch 2.13 / torch-xpu-ops #2992 scoping -- capturing the MTP all_gather -- 2026-07-21

Research-only (no GPU, no builds). Question: what does it take to make the MTP spec-decode
`vllm::all_gather` (an eager oneCCL SUM all-reduce over PCIe, ~43% of decode) GRAPH-CAPTURABLE
via torch-xpu-ops PR #2992 (adds `errorIfCapturingNonCapturableXCCL` + Recording mode so a oneCCL
allreduce records as a static SYCL-graph node)? Our torch==2.12.0+xpu predates #2992.

This builds on `backend_currency_2026-07-21.md` item 3, which flagged the route as NO-GO but did
not scope the mechanism. This doc scopes it concretely and compares to the push-AR alternative.

## Decision table

| # | Item | Finding | Source |
|---|------|---------|--------|
| 1a | Is torch 2.13+xpu released? | YES. v2.13.0 is tagged and has an XPU wheel on `download.pytorch.org/whl/xpu` (`pip install torch --index-url .../whl/xpu`). Released after 2026-06-09 (its torch-xpu-ops pin is a June-9 commit), i.e. ~July 2026. | https://github.com/pytorch/pytorch/releases/tag/v2.13.0 , https://docs.pytorch.org/docs/main/notes/get_start_xpu.html |
| 1b | Does torch 2.13.0 INCLUDE #2992 (no cherry-pick needed)? | YES. torch 2.13.0 pins torch-xpu-ops commit `bc294243` (`third_party/xpu.txt`), which is the merge of PR #3914 (merged to **main** 2026-06-09). #2992 merged to **main** 2026-05-07 (commit `1c395b7`). #3914 is a later main commit than #2992, so #2992 is an ancestor of the 2.13 pin -> **the XCCL SYCL-graph capture guard ships in the stock torch 2.13.0+xpu wheel.** No source build of torch-xpu-ops required for the guard itself. | https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/third_party/xpu.txt , https://github.com/intel/torch-xpu-ops/pull/2992 , https://github.com/intel/torch-xpu-ops/pull/3914 |
| 2 | Does vLLM (0.25.1 or newer) support torch 2.13+xpu? | NO. vLLM **main** still pins `torch==2.12.0` in `requirements/xpu.txt` (same as 0.25.1, our current). 0.25.1 is the newest release. Moving to torch 2.13 means patching vLLM's xpu requirement AND rebuilding the vLLM base image against 2.13 -- an unvalidated combo that risks new XPU regressions (we already had to fix 4 for 0.24->0.25.1: device-env, int8g anchor, oneCCL, hpc_rope). | https://raw.githubusercontent.com/vllm-project/vllm/main/requirements/xpu.txt , https://github.com/vllm-project/vllm/releases |
| 3a | oneCCL version #2992 needs | oneCCL **>= 2021.17.2 (maybe 2022.0)** for SYCL-graph allreduce record/replay; the guard `errorIfCapturingNonCapturableXCCL` hard-errors capture on older oneCCL. Supported collectives: allreduce / allgather / reduce_scatter. Broadcast + P2P explicitly NOT supported. | https://github.com/intel/torch-xpu-ops/pull/2992 |
| 3b | What oneCCL do we ship vs what 2.13 bundles? | We ship oneCCL **2021.17** (int8g image swaps 2021.17 over the upstream-bundled 2021.15 to fix the TP=2 `ze_handle_manager mem_to_ipc_handle` worker-init death). 2021.17 == oneAPI 2025.3; 2021.17.2 is a patch above it; 2022.0 == oneAPI 2026.0. **Our 2021.17 is BELOW the 2021.17.2 #2992 threshold.** Whether torch 2.13's bundled oneCCL clears 2021.17.2 is unconfirmed (Intel prereq page 403'd); it may still need a manual oneCCL swap, which re-opens the TP=2 mem_to_ipc_handle wedge we just fixed. | `vllm/images/int8g/bake_v0251.sh:108-123` ; oneCCL/oneAPI mapping: PyTorch/Intel XCCL docs https://pytorch.org/blog/pytorch-2-8-brings-native-xccl-support-to-intel-gpus-case-studies-from-argonne-national-laboratory/ |
| 4 | Rebuild surface (torch 2.13 ABI break) | ALL custom `.so` are ABI-locked to torch 2.12 (`torch::Library::_def` fails across torch versions). A torch 2.13 bump forces rebuild + revalidation of every one -- see enumeration below. | `kernels/README.md` , CLAUDE.md Repo Layout Contract |
| 5 | Cheaper alternative to a full 2.13 bump | Build torch **2.12.0 from source** with `third_party/xpu.txt` bumped to a torch-xpu-ops commit that contains #2992, keeping the ABI at 2.12.0 so NO custom `.so` rebuild and NO vLLM bump. Still needs oneCCL >= 2021.17.2 swapped in. Feasible but non-trivial (see below). LD_PRELOAD-oneCCL-alone does NOT work. | this analysis; #2992 code lives in libtorch_xpu.so, not liboneccl |

## Rebuild surface if we bump to torch 2.13 (item 4 detail)

Every artifact below is compiled against torch 2.12's C++ ABI and must be rebuilt against 2.13,
then re-ABI-verified (load without `torch::Library::_def` failure) and re-benched:

1. **int8g image kernels** -- `int8_gemm_w8a16`, `int8_gemm_w8a8`, `int8_gemm_w8a8_fusedq`,
   `dynamic_per_token_int8_quant` (baked into `vllm-xpu-env:int8g` via `vllm/images/int8g/build.sh`
   + `bake_v0251.sh`; source `kernels/*.h` + `int8_gemm_kernel.patch`).
2. **NVFP4 kernels** -- `nvfp4_gemm_w4a16`, `nvfp4_gemm_w4a8` -> `nvfp4_kernel/_xpu_C.abi3.so`
   (`vllm/nvfp4/build_nvfp4_kernel.sh`, `build_nvfp4_w4a8.sh`). This is the daily-driver's kernel.
3. **W4A8 / sglang W8A8** -- `w8a8_kernel/_xpu_C.abi3.so` (`research/w8a8/W8A8_BUILD.md`) and the
   w4a8 `.so` (`sglang/W4A8_BUILD.md`); built against **sglang's** torch, a separate ABI target.
4. **push-AR .so** -- `libxpu_push_ar_torch.so` (prefill) + `libxpu_push_ar_graph.so` (capturable
   decode) from `vllm/contrib/vllm_push_allreduce/`. These are C-ABI `icpx -fsycl` builds that grab
   `torch.xpu.current_stream().sycl_queue` and operate on `data_ptr()`s -- less TORCH_LIBRARY-fragile
   than the oneDNN ops, but they link torch's stream/queue accessors, so at minimum a rebuild-and-verify.
5. **vLLM base image v0251** -- rebuild from source against torch 2.13 (patch `requirements/xpu.txt`),
   plus re-fix any new XPU regressions the 2.12->2.13 core delta introduces.

Revalidation (each item): the serve gate (18/18), `bin/serve-sweep --smoke` green across shelf models,
coherence under concurrent prefill+decode load, HumanEval+ on the NVFP4 27B DD, and a DD soak.
The sitecustomize shim blocks (i64 two's-complement wrap, MTP reclaim, thinking-budget, register_fake)
are Python-level -- no ABI rebuild, but all need re-verification against 2.13 internals.

**Effort estimate (full 2.13 bump): ~4-7 focused sessions, HIGH risk.**
- torch 2.13 wheel + vLLM rebuild against 2.13 + new-regression triage: 1-2 sessions.
- rebuild all custom `.so` (int8g, nvfp4, w4a8, sglang _xpu_C, push-AR) + ABI verify: 1-2 sessions.
- oneCCL: verify/swap to >= 2021.17.2 (or 2022.0) + re-test TP=2 mem_to_ipc_handle wedge: 0.5-1 session.
- full revalidation (gate + shelf sweep + soak + HumanEval): 1-2 sessions.

## Cheaper path (item 5 detail): from-source torch 2.12.0 + #2992 cherry-pick

The only ABI-safe shortcut. #2992's guard code compiles INTO `libtorch_xpu.so` (torch-xpu-ops is a
submodule built into libtorch), NOT into `liboneccl` -- so an LD_PRELOAD of a newer oneCCL alone is
**insufficient**; without the #2992 code the capture path is never taken. But we can:

1. Build **torch 2.12.0 from source** (tag v2.12.0) with `third_party/xpu.txt` re-pointed to a
   torch-xpu-ops commit >= `1c395b7` (or cherry-pick #2992 + its intervening deps onto 2.12's pin).
   Because the torch tag stays 2.12.0 and the public C++ ABI is unchanged (#2992 only touches XCCL
   internals), our existing custom `.so` should keep loading -- **no kernel rebuild, no vLLM bump.**
2. Swap in oneCCL >= 2021.17.2 (or 2022.0) and re-test the TP=2 mem_to_ipc_handle path.

Feasibility: MEDIUM. Risks: (a) torch-from-source XPU build is a multi-hour heavy compile with its
own toolchain-match hazards; (b) #2992 may not cherry-pick cleanly onto 2.12's older torch-xpu-ops
pin (may drag intervening commits -> effectively a torch-xpu-ops main bump on a 2.12 base, which can
break other XPU ops); (c) the oneCCL >= 2021.17.2 swap still risks the TP=2 wedge. Estimate ~2-3
sessions, and it hinges on (b) being clean. If the cherry-pick is not clean, this collapses back to
"just take torch 2.13" and the full rebuild surface.

## Bottom line (5 lines)

1. torch 2.13.0+xpu is released AND already ships #2992 (its torch-xpu-ops pin `bc294243`/#3914 is a
   later main commit than #2992's `1c395b7`) -- so the guard needs NO cherry-pick, but everything
   around it does: vLLM still pins torch 2.12 (needs a bump), and our oneCCL 2021.17 is below #2992's
   2021.17.2 threshold (needs a swap that re-opens the TP=2 wedge we just closed).
2. The full 2.13 route is a ~4-7 session, HIGH-risk batched rebuild-everything campaign (int8g, nvfp4,
   w4a8, sglang _xpu_C, push-AR all ABI-locked to 2.12) + full re-gate/shelf-sweep/soak.
3. Smallest viable path if we ever need the capturable oneCCL collective: build torch **2.12.0 from
   source** with #2992 back-pointed into `third_party/xpu.txt` (ABI stays 2.12 -> no .so or vLLM
   rebuild) + oneCCL >= 2021.17.2 swap; ~2-3 sessions, contingent on a clean cherry-pick.
4. But #2992 only makes the all_gather record **as an oneCCL node (~85us)**, whereas our
   `libxpu_push_ar_graph.so` already records into XPUGraph at ~34-45us AND the goal is to record the
   all_gather as the fast **push-AR do_ar**, not as oneCCL. So even when #2992 works it is SLOWER than
   the transport we already own -- it is a capturability fallback, not a speed win.
5. VERDICT: NOT worth a dedicated campaign now. The eager-device-async-push-AR prototype (separate
   agent) is the right primary -- it extends the faster transport we already have to the MTP all_gather.
   Keep torch-2.13/#2992 as a documented fallback ONLY if push-AR extension fails; and if invoked,
   prefer the from-source-2.12 cherry-pick over the full 2.13 bump.
