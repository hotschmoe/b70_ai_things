# TensorFold review for the B70 Flash-Next team (Zig engine, XPU backend, concurrency lessons)

Responding model: Claude Fable 5.2 (model ID claude-fable-5-2). Date 2026-11-20.
Read-only static analysis. No GPU touch, no build, no source or config edit. Only this file was written.

## 0. Scope, sources, conventions

CONFIG -> TensorFold clone /mnt/cache/b70/refs/TensorFold, main at 9e2a9bd7 (2026-11-20 02:34 UTC), version
1.3.0 (build.zig.zon), Zig 0.17.0 (.zig-version). Shallow (50 commits); history was not needed.
One extra ref was fetched into the clone's .git (no worktree change): refs/pr/3603 = open PR #3603 head
207aaae1 (2026-11-19), the top of the XPU Flash-Next stack (#3585 merged -> #3586 -> #3587 -> #3603).
Files quoted from that ref are marked "PR3603:"; everything else is main.
GitHub issues/PRs were read with gh (read-only): #3583, #3358, #2081, #801, #1793, #2543, #2897, #1328 and
PRs #3585, #3586, #3587, #3603. Issue/PR text is contributor-reported data, not verified by us.
Our side: analyst/fable_decode_window_walk.md and analyst/opus_sched_prefill_restore_walk.md (this directory),
docs/P2P_GPU.md.

Tags: [R] read in a TensorFold file at the cited path. [R-gh] read in a GitHub issue/PR. [M-TF] a number
TensorFold docs or contributors report as measured (their hardware, not ours, not re-run). [I] my inference.
All paths without a prefix are relative to the TensorFold clone.

## 1. Headline findings

1. TensorFold ALREADY HAS a native Intel backend, called xpu, written for our exact card. [R]
   - Level Zero opened with dlopen and no C headers (zig/src/xpu/ze.zig, 549 lines), OpenCL C kernels compiled
     offline by ocloc for device bmg, plus prompt GEMMs emitted as ISA by oneDNN's nGEN at build time, all
     embedded in the binary. No SYCL, no PyTorch, no Python at run time (docs/xpu/README.md, NAVIGATION.md).
   - It opens the device by PCI id 0xe223 (Arc Pro B70). It serves Qwen3.8-27B (GPTQ int4 g128 or MLX 4-bit
     g64) with the DFlash2 tree drafter, plus Nemotron 3.5 Lightning, LFM3 and Gemma 4.
   - On ONE B70 at 230 W [M-TF, docs/xpu/README.md]: prefill 1,975 tok/s (GPTQ), 2,617 (W4A8 prompt GEMMs),
     2,830 (rotated W4A4); one stream 133 tok/s code / 90 chat; bench_openai aggregate at 1/2/4/8 requests
     165 / 300 / 337 / 439 tok/s. Concurrent replies are bit-equal to solo replies.
2. Flash-Next (qwen4_exp) on XPU is NOT on main yet, but a complete implementation is in open PRs that the
   maintainer has marked "ready to land" (#3587, #3603; gate issue #3583). [R-gh]
   - Level Zero P2P tensor parallel at TP = 1, 2, 4 (collectives merged in #3585: zig/src/xpu/tp.zig,
     zig/kernels/xpu/tp.cl), all 48 layers, 512 experts split per rank, host-RAM PLE table, MTP head with
     confidence-stopped chains, shared verify rounds up to 32 streams, prompt snapshots.
   - Reported on 4x B70 + EPYC Turin + PCIe Gen5, MLX 4-bit g32 checkpoint, TP=4 [M-TF, PR #3603]: single
     stream 158-176 tok/s greedy, cold prefill 1,245-1,640 tok/s (1,174 at the full 256K window), steady
     aggregate 196 / 223 / 237 / 215 tok/s at 2 / 4 / 8 / 32 streams, decode gap p95 about 200 ms while a
     16K-32K prompt fills, all concurrent/solo and drafted/plain token hashes equal.
   - It does not fit our box as is: the MLX 4-bit checkpoint puts about 79.2 GiB on the cards (19.8 GiB per
     card at TP=4), so TP=2 would need about 39.6 GiB per 32 GB card [I from R-gh]. No sub-4-bit Flash-Next
     format exists on XPU; no vision on XPU; no RAM KV tier or conversation parking on XPU.
3. Nobody gets good Flash-Next concurrency scaling, including TensorFold. Aggregate at 8 streams is 1.35-1.5x
   one stream on their 4-card XPU stack and about 1.0x on a GB10 (docs/recipes/flash-next-cuda.md: drafted
   64 / 45 / 66 tok/s at 1 / 4 / 8 streams) [M-TF]. The cause is the same as ours: every extra row reads
   about 10 more experts per layer, and a stream's draft rows are traded for other streams' rows. Our C1 134.6
   -> C4 141 aggregate (1.05x) is below their shape (about 1.3-1.4x at 4 streams), so there is roughly
   25-35% of C4 aggregate to recover by scheduling and launch structure before the bandwidth law binds [I].
4. Their scheduler is one shared core for every model and backend ("lanes", zig/src/core/lanes/, 13.1K lines):
   one engine thread, one forward per round for all streams' windows, draft rows allocated by expected tokens
   per millisecond from a startup-measured cost table, prompts filled a chunk per round between decode
   rounds. This is the main thing to copy into Strata (section 5).
5. For a Zig Intel engine: do not start greenfield and do not "add a Level Zero backend to TensorFold" - it
   exists. The realistic paths are (a) build the PR #3603 stack on our host and extend it (TP=2 fit via a
   sub-4-bit expert format, RAM KV tier, vision), or (b) keep Strata and port ideas. Section 6 has the
   costs. License is Apache-2.0.

## 2. Architecture (question 1)

### 2.1 What is Zig and what is not, with sizes [R, counted with wc -l over git ls-files]

Everything on the host side is Zig, including the HTTP server, tokenizer, chat templates, sampler, scheduler,
checkpoint readers and the GPU driver bindings. GPU kernels are in each vendor's kernel language.

| Part | Path | Files | Lines |
|---|---|---:|---:|
| All Zig under zig/ | zig/**/*.zig | 5,951 | 1,066,778 |
| Zig in zig/src, non-test / test | zig/src | - | 596,868 / 185,132 |
| Core (lanes, caches, KV pool, formats, contract) | zig/src/core | 934 | 168,561 |
| - of which the lane scheduler | zig/src/core/lanes (non-test) | 74 | 13,109 |
| HTTP server, API, templates | zig/src/server | 584 | 109,117 |
| Model families (37 families) | zig/src/families | 2,089 | 360,761 |
| - flashnext family (non-test / test) | zig/src/families/flashnext | 332 | 128,869 / 11,681 |
| - flashnext cuda_* files / hip/ | same | 145 / 22 | 21,773 / 5,383 |
| Per-backend server hosts | zig/src/native | 173 | 26,587 |
| CUDA runtime binding | zig/src/cuda | 149 | 26,318 |
| HIP (AMD) runtime binding | zig/src/hip | 157 | 26,114 |
| Metal runtime binding | zig/src/metal | 21 | 3,380 |
| XPU (Intel) runtime + 27B/Nemotron runners | zig/src/xpu | 37 | 7,070 |
| Vulkan (nascent) | zig/src/vulkan | 7 | 1,314 |
| Multi-node fabric / cluster | zig/src/fabric, cluster | 228 | 33,970 |
| Build graph | zig/build + build.zig | 198 | about 13,000 |
| Tests outside src | zig/tests | 1,699 | 239,141 |
| Metal kernels (.metal) | zig/kernels/metal | 376 | 60,992 |
| CUDA kernels (.cu, .cuh) | zig/kernels/cuda | 291 | 53,557 |
| HIP kernels (.hip, .hpp) | zig/kernels/hip | 66 | 11,061 |
| XPU kernels (OpenCL C .cl) | zig/kernels/xpu | 26 | 5,652 |
| Vulkan kernels (GLSL .comp) | zig/kernels/vulkan | 2 | 412 |
| C/C++ under zig/ (HIP headers, nGEN generator, xgrammar bridge, mocks) | - | 90 | 9,489 |
| nGEN prompt-GEMM generator (C++) | zig/tools/xpu/w4_ngen.cpp, w4_prompt.hpp, w4a8_prompt.hpp | 3 | 775 |
| Bench/conformance tools (Python, client side only) | tools | 408 | 53,221 |
| Desktop/mobile app, web UI | app, web | 2,963 | about 303,000 |

The XPU Flash-Next PR adds [R, PR3603 diff stat]: 8 Zig files in zig/src/families/flashnext/xpu_*.zig (3,934
lines), 4 OpenCL C files in zig/kernels/xpu/flashnext/ (1,634 lines), zig/src/native/xpu_flashnext.zig (233),
xpu_lane.zig (570), xpu_lane_prompt.zig (79). That is a whole Flash-Next engine for a new GPU vendor in about
6.5K lines, because server, tokenizer, scheduler, drafting policy, prompt cache and sampling are reused.

### 2.2 Layering [R]

    HTTP server (zig/src/server)  ->  engine_api.Engine (zig/src/core/engine_api.zig)
      -> LaneHost (zig/src/core/lane_host.zig): one engine thread, queue, admission, prompt cache, KV ledger
        -> lanes.Engine (zig/src/core/lanes/engine.zig): the round loop, depth rule, row allocation
          -> lanes.backend.Backend vtable (zig/src/core/lanes/backend.zig, 272 lines)
            -> a family on a backend (for example zig/src/native/xpu_lanes.zig + zig/src/xpu/qwen27.zig)
              -> the vendor binding (ze.zig / cuda/driver.zig / metal/objc.zig / hip/driver.zig) + kernels

The Backend vtable is the abstraction layer a new engine implements. Required calls: prefill, first, queue,
read, verify(windows), keep(windows, paths), draft(requests), release. Optional: prefill_step(max_rows),
cancel_prefill, prefill_many, begin/piece/finish/pieces (prompt in pieces inside rounds), probabilities, tree,
alternatives, features (drafter taps), first_masked (grammar), reports_fill. Flags: chunk_invariant,
penalties. "A new family does not add another scheduler beside the engine" (CONTRIBUTING.md).

Model code is NOT backend-neutral. A family is written once per backend: its own forward, loaders, state
layout and kernels (flashnext: Metal files unprefixed, cuda_*.zig, hip/, and xpu_*.zig in the PR). What is
shared is everything above the vtable plus format readers (core/safetensors, core/gguf.zig, core/quant),
conventions and FP64 references. docs/recipes/adding-a-zig-family.md is the map of the shared blocks.

### 2.3 How kernels are written, compiled and loaded, per backend [R]

- CUDA: .cu/.cuh compiled by nvcc into fatbins per SM (zig/build/cuda_fatbin.zig, cuda_kernels.zig), loaded
  with cuModuleLoadData through the driver API opened at run time from libcuda.so.1 (zig/src/cuda/driver.zig,
  abi.zig, module.zig). Launches are cuLaunchKernel / cuLaunchKernelEx with cluster and dependent-launch
  attributes (launch.zig). Decode and verify shapes are captured as CUDA graphs and replayed (graph.zig,
  graph_cache.zig). No cuBLAS in the text path (cuBLASLt only for the vision tower). NCCL for ranks, plus an
  own RoCE one-shot all-gather (docs/roce-gather.md, zig/src/cuda/roce.zig).
- Metal: .metal sources compiled with xcrun metal into a metallib that is embedded (zig/build/*metal*.zig);
  the Objective-C runtime is called from Zig through typed objc_msgSend (zig/src/metal/objc.zig, command.zig,
  icb.zig for indirect command buffers, residency.zig).
- HIP: hipcc code objects per gfx arch, runtime opened at run time (zig/src/hip/driver.zig, code_object.zig).
- XPU: OpenCL C compiled by "ocloc compile -device bmg -options -cl-std=CL3.0 -cl-intel-256-GRF-per-thread"
  (128-GRF for some kernels) into device binaries, and nine prompt GEMM binaries generated by a C++ nGEN
  program run at build time; all embedded with @embedFile (zig/build/xpu.zig). Loaded with zeModuleCreate in
  the NATIVE format (ze.zig ModuleDesc.format = 1), so there is no run-time JIT and no SPIR-V at run time.
  Kernels use sub-groups of 16, DPAS (XMX), 2D block loads, and document their bit-level reduction order in
  each file header (zig/kernels/xpu/qwen27/*.cl; NAVIGATION.md section 5).

### 2.4 XPU command submission and memory [R, zig/src/xpu/ze.zig, tp.zig; PR3603: xpu_forward.zig]

- One in-order immediate command list per device, asynchronous mode (ze.zig immediate(): flags in-order,
  mode 2). Kernels are appended with zeCommandListAppendLaunchKernel; the host blocks only at
  zeCommandListHostSynchronize.
- The Flash-Next forward queues ALL 48 layers on every rank without a host round trip and syncs once at the
  end (PR3603: zig/src/families/flashnext/xpu_forward.zig forwardPass; syncAll only before the head gather).
  Routing top-10 (fn_route_topk) and the expert plan (fn_moe_compact) run on the GPU, so no pick ever comes
  back to the host. The host PLE gather runs while layer 0 executes on the cards ("Queue Layer 0 ... so
  hardware executes Layer 0 in parallel with CPU PLE gather").
- No command-graph capture is used. ze.zig declares the record/replay graph extension table (GraphApi) but no
  file on main or in PR3603 calls it (grep). They reach 158-176 tok/s without graph replay [M-TF].
- Memory: zeMemAllocDevice / zeMemAllocHost USM, every handle tracked and released in order by Gpu.close()
  (drain all lists first). Free VRAM is read through Sysman and the number of stream slots is computed from
  it (slotsThatFit, zig/src/native/xpu.zig:160; docs/xpu/NAVIGATION.md section 6). Kernels that follow
  pointer tables set ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE (ze.zig indirect()).
- Multi-card: openPeers puts 1-8 devices in ONE Level Zero context after checking zeDeviceCanAccessPeer in
  both directions; ring buffers are made resident on every peer (zeContextMakeMemoryResident); a pool of
  device-scope events (signal and wait on the device, "without host round-trips") orders the ranks
  (ze.zig sharedEvents). A reduce kernel on rank r reads every rank's send buffer directly over PCIe and
  sums in rank order in fp32; above 512 KiB it switches to reduce-scatter plus copy-engine all-gather
  (tp.zig allReduceF16/F32, RS_THRESHOLD_BYTES). 4-slot ring, up to 256 pipelined steps between host syncs.
- Per-kernel GPU time comes from Level Zero kernel-timestamp events (ze.zig events(), seconds();
  XPU_PROFILE=1), no external tracer.

### 2.5 The exactness design that everything else leans on [R]

"The bits of a row do not depend on the number of rows, on its position, or on the tiling" (NAVIGATION.md
section 5; docs/kernel-contract.md). Each row has its own accumulator and a fixed summation order, so a
verify row, a multi-stream row, a prompt row and a plain decode step give the same bits. Consequences:
drafted == serial, concurrent == solo, resumed == fresh, any chunk cut are all testable by token hash
(tools/bench_concurrent.py --alone --serial), and the scheduler is free to choose any window shape. The
maintainer's rule since 2026-11-15 is that every change keeps every token, opt-in or not [R-gh #3358].
The price is visible: Nemotron's XPU prompt path fell from about 3,100 to 850 tok/s when it was made
chunk-invariant on the vector units (docs/recipes/xpu-arc.md "Known limits").

## 3. Serving and concurrency design (question 2)

### 3.1 Scheduler and continuous batching [R]

- One engine thread owns the model (lane_host.zig). HTTP threads only enqueue jobs. Streams join between
  rounds; a cancel releases the slot for the next job.
- A round = one forward over every live stream's window: the pending token plus that stream's drafts, as a
  chain or a tree (backend.zig Window: pending, held, tokens, parents, positions, masks). XPU 27B packs at
  most 16 rows per forward across streams (MAX_SLOTS 16); Flash-Next CUDA takes 2-32 streams of up to 16 rows;
  the XPU Flash-Next PR runs up to 32 streams in sub-batches.
- Draft rows are ALLOCATED, not fixed. zig/src/core/lanes/allocate.zig "Draft prefixes per stream with the
  most expected tokens a ms": it walks the likeliest next node across all streams (each stream's acceptance
  probability by depth, or the MTP head's own probabilities) and keeps adding rows while expected tokens per
  millisecond rises, using a forward-cost table by row count. depth.zig holds per-stream acceptance state,
  plain_guard (compare with the stream's own plain rounds), plain_streams (at N streams or more nobody
  drafts), confidence_floor. row_cost.zig and shared_price.zig learn what a row costs on this machine;
  lone_costs.zig keeps calibrations per context band (under 8,192, then one band per doubling) [R-gh #2897].
- The cost table is measured at server start: zig/src/native/xpu_lanes.zig:93 measure() times a verify plus
  keep at 1..16 rows and logs "window ms by rows"; cost_rule.zig keeps the fastest of 7 runs, retimes dips
  over 3%, and flags a table that drifted 15% from its stored reference. "Timings steer windows, never
  bits."
- Lone stream fast path: a single stream runs its own driver (XPU: DFlash2 trees of 15 nodes; CUDA: graphs)
  and is handed to shared rounds when a second stream arrives, then handed back (Lone.resumes,
  zig/src/core/engine_api.zig:546, lane_host_lone.zig; lanes/handover.zig). This is how they keep C1
  untouched while adding concurrency.
- Keep is lazy and cheap: a window kept whole is a state-buffer swap (rec / rec_save); a partial keep replays
  the kept rows from a stash; "settle" defers an all-kept window's keep to the start of the next call
  (NAVIGATION.md sections 3 and 6). Metal Flash-Next can fold the replay into the next window
  (FZ_GDN_REPLAY=1, docs/recipes/qwen3.8-flash-next.md last paragraph).
- Drafts of all streams are produced in one batched pass (XPU latticeMulti; Metal "Every stream's MTP drafts
  share one command buffer"). Tokens are drawn on the GPU and stay behind handles; the host reads values
  only when it must (backend.zig Feed, read).

### 3.2 Mixed prefill and decode [R]

- A prompt fills one chunk per round between the other streams' decode rounds (Backend.prefill_step;
  lane_host.filling). The next fill is chosen by lane_host_fill.zig: foreground before background, then least
  remaining work, and after 8 skipped rounds the oldest wait wins. TENSORFOLD_PREFILL_ORDER=srpt adds a
  shortest-remaining-time rule between a reply and a prompt (lane_host_srpt.zig).
- Chunk size is adaptive (lanes/solo_chunk.zig): a wide "solo" chunk while no other stream decodes, a small
  "busy" chunk otherwise. Flash-Next CUDA: 3,072 busy, 8,192 solo at --parallel 1 (flash-next-cuda.md
  "Prompt chunk size"); XPU Flash-Next PR: 2,048 solo, 64 busy. The trade is stated plainly: every pass
  re-reads every routed expert its rows touch (1.84 MB of gate+up per expert), so few large passes are faster,
  but a pass is not interruptible, so a decoding stream waits a whole pass.
- --decode-share gives decoders a time share against a filling prompt as a debt (lane_host_fill.zig Share).
  Note docs/api.md:621: the XPU server does not parse this flag yet.
- prefill_many packs several short prompts in one pass; the opt-in Rounds planner (lane_rounds.zig) runs
  prompt pieces and decode rows in one forward so each layer's experts are read once for both (the Python
  engine's --parallel behavior, docs/recipes/qwen3.8-flash-next.md "Serving": live replies then decode at
  about a tenth of their rate during a burst instead of stopping).
- Because prompt arithmetic is chunk-invariant, any chunking schedule and any cache cut gives the same
  tokens (Backend.chunk_invariant).
- Measured effect [M-TF, flash-next-cuda.md]: first token of a short request sent 2 s into a 31,736-token
  prompt 0.36 s with 1,536-row chunks vs 1.9 s with 4,096-8,192 solo chunks; longest decode gap during an 8K
  prompt 0.81 s. XPU PR: gap p95 197-200 ms during 16K-32K prefill.

### 3.3 Prefix/KV caching and memory tiers [R]

- Prompt cache (zig/src/core/prompt_cache.zig Store): whole-state snapshots at planned marks. Marks: end of
  the system prefix and 512 / 2,048 tokens before it, message starts, end of prompt minus a lookahead;
  optional power-of-two intermediates (TF_PROMPT_INTERMEDIATE), one interior state inside a long last user
  message, a repeat mark for raw completions (flash-next-cuda.md "What the core runs"). On XPU the snapshot
  lives in VRAM under --prompt-cache-gib (2 GiB default; 64 KiB per token + 187 MiB for the 27B), restored by
  device copy: 6.1 s -> 1.2 s on a 9.7K document [M-TF, docs/xpu/README.md].
- Paged KV (zig/src/core/kv_pool.zig): streams lease pages from one pool as they grow instead of reserving a
  whole window; kv_ledger.zig admits a request by prompt + reply + one window. Done on CUDA for most
  families, "experimental" for Flash-Next (docs/optimization-ledger.md rows 2, 6).
- Shared paged prompt store (prompt_radix.zig): a kept prompt stays as the pool's pages, a resume copies
  nothing (ledger row 3; not yet for Flash-Next).
- RAM tier: "7. Host KV tier" = KV pages in pinned host memory. Done only for GLM (TF_GLM_KV_HOST); a core
  host tier is an open PR (#1501) and Flash-Next is "open". Kept prompt states can live in host RAM (CUDA host
  store) and on disk (--spill-gib, --learn-conversations; zig/src/native/disk_spill.zig, cuda_disk.zig).
- Fluid Subgraph Residency (docs/design/fluid-subgraph-residency.md): routed experts streamed from SSD into a
  bounded GPU pool with a persistent expert-to-slot map, CLOCK eviction, load-before-execute and next-layer
  route forecast; Metal Flash-Next only (--ssd-experts, --ple-on-ssd). Reported 0.31-0.39x decode when
  experts stream [M-TF]. Not needed for us (all experts resident), but the design rules are sound.
- KV precision: XPU 27B has an int8 KV option (XPU_KV_INT8: half the memory, 12 streams instead of 7 at
  16K; answers not bit-equal to bf16 KV). Native Flash-Next KV is bf16 only: 28 KiB per token (CUDA) vs our
  int8 12.7 KB per token [I from fable_decode_window_walk.md section 2].
- Verdict for our goal "4 slots x 200K with RAM streaming and RAM parking": TensorFold has no equivalent on
  XPU today. Strata is ahead here. A contributor fork has "prompt state pages and a host park tier" for XPU
  27B (xpu/pages.zig, park_runs.zig, native/park_tier.zig) proposed but not upstream [R-gh #3358 F3].

### 3.4 Speculative decoding [R]

MTP heads (Flash-Next: up to 6 chained drafts, stop when the product of head probabilities falls under 0.4;
TF_FLASHNEXT_DRAFTS, TF_FLASHNEXT_CONFIDENCE), DFlash/DFlash2 block and tree drafters reading target taps
(zig/src/xpu/drafter.zig, drafter_tree.zig), DSpark, DSpine, and copy drafts from the context with no drafter
weights (zig/src/core/copy_index.zig, lanes/context_copy.zig; the XPU PR reports 214-253 tok/s with multi-turn
copy drafts vs 113-136 pure MTP [M-TF]). A draft is accepted only if it equals the engine's own plain token.
Long-context caveat they measured on our card [R-gh #2897, one B70, 27B]: a 15-node tree is slower than
plain decode at 121K context because every tree row re-reads the K/V; tree 3 beats tree 15 from 29K up. The
fix accepted upstream is to price windows per context band, not a fixed cut.

### 3.5 Multi-GPU and multi-node [R]

- docs/multi-rank-receipts.md: rank placement is family-specific; every rank count needs its own equalities;
  a one-host pair must be named as such. docs/roce-gather.md: small all-gathers as one-shot RDMA writes with
  a GPU-side wait, 10.4 us vs NCCL 12.3 us at 4 KiB on two GB10s. docs/n-link-speed-up.md: 1-8 bonded
  Thunderbolt RDMA links between Macs. docs/speed-up-mode.md: two Macs each holding the whole model and
  splitting prompt rows.
- Flash-Next TP on CUDA (flash-next-cuda.md): ranks partition expert intermediate channels and gather
  vocabulary ranges; prompt passes wider than 64 rows reduce-scatter expert slots by row halves (3.43 vs
  5.81 MiB per prompt token over the link, +26.5% cold prefill). "At --tp 2 each stream's window verifies
  alone" (no shared verify at TP=2 on CUDA yet).
- XPU TP (merged collectives + open PRs): shards dense columns AND experts (128 experts per rank at TP=4),
  two fp32 all-reduces per layer (after the mixer, after the MoE) = 96 collectives per forward, head logits
  gathered by peer copies. Collective cost [M-TF #3583]: TP=2 3.81 us per pipelined op at 1 row and
  34.9 GB/s at 4K rows on PCIe Gen5; TP=4 8.5 us (1.09 ms per token for 128 ops).
- Topology wall [R-gh #3585 comment]: on a consumer Intel board with the two cards on sibling CPU root ports
  zeDeviceCanAccessPeer returns 0 and openPeers refuses; a host-staged fallback is promised as a follow-up,
  not written. Our box: docs/P2P_GPU.md H.11-H.12 measured canAccessPeer = True on kernel 7.0 with IOMMU
  off and 9.7 GB/s P2P all-reduce through oneCCL, so the P2P path should open here on 7.1 [I, must be
  re-probed], at Gen3 bandwidth (their 34.9 GB/s number will not transfer).

### 3.6 Their measured concurrency numbers [M-TF]

| Stack | 1 | 2 | 4 | 8 | Source |
|---|---:|---:|---:|---:|---|
| 27B dense, 1 B70, GPTQ, bench_openai aggregate tok/s | 165 | 300 | 337 | 439 | docs/xpu/README.md |
| 27B dense, 1 B70, code streams with drafts, aggregate | 133 | 235 | 310 | 270 | docs/xpu/README.md |
| 27B, 4x B70 as 4 x TP=1 servers, 32 streams | - | - | - | 1,007 agg | #3583 |
| Flash-Next, 4x B70 TP=4, steady aggregate (32 streams: 215) | 158-176 | 196 | 223 | 237 | PR #3603 |
| Flash-Next, GB10 CUDA, ModelOpt, drafted aggregate | 64 | - | 45 | 66 | flash-next-cuda.md |
| Flash-Next, GB10 CUDA, MLX 4-bit, no drafts aggregate | 35 | 53 | 78 | - | flash-next-cuda.md |
| Nemotron MoE, 1 B70, 4 streams vs the same serially | - | - | 72 vs 84 | - | docs/recipes/xpu-arc.md |
| Ours (Strata, 2x B70 EP, IQ3) aggregate | 134.6 | 131 | 141-149 | - | task statement |

Reading: dense models scale 2.7x by 8 streams on one B70; MoE Flash-Next scales 1.0-1.5x everywhere.
Nemotron on XPU states the reason: "concurrent streams do not share weight reads".

### 3.7 MoE at high batch [R]

- Expert plan on the GPU: sort a step's (row, slot) pairs by expert into items, one grouped call runs every
  expert's rows (zig/src/cuda/grouped.zig plan kernels: plan, rank, offsets, scatter; experts.zig Layer and
  Format enum: affine4, affine3, fp8g, nvfp4, nvfp4_csf0, exl3, iq2xxs, bf16). A graph captured for one step
  replays any other step's routing because the plan is read on the device.
- Row tiles: experts_rows.cu runs 32 pairs per item (two row tiles of 16), "each row's bits are those it
  gets alone" (zig/src/cuda/experts_rows.zig). So an expert picked by many rows is read once per 32 pairs.
- XPU PR3603 zig/kernels/xpu/flashnext/fn_moe_head.cl: fn_moe_compact builds 32-pair tiles per expert in
  SLM with atomics and keeps ascending slot order per row; fn_moe_up_dpas / fn_moe_down_dpas run a
  [32 x 64] DPAS layout with active-subgroup masking because "86.6% of busy c=64 groups have m <= 8";
  decode/verify (R <= 16) uses per-pair kernels fn_moe_up_1 / fn_moe_down_1 with 4-way split-K.
  Expert weights are repacked at load into 16-column tiles so a 16-lane sub-group reads 4 contiguous 64-byte
  lines at any batch width (PR #3603 description).
- Metal groups a decode window's experts only from 5 rows up (TF_FLASHNEXT_GROUPED_ROWS); smaller windows
  keep per-row kernels (docs/recipes/qwen3.8-flash-next.md).
- A 16-bit "scale index" replaces per-group stored scale/bias pairs: 10% fewer expert bytes read, same
  tokens (ledger row 28).
- There is no cross-stream trick beyond reading each distinct expert once per window. Their own ledger marks
  "Shared drafts" as a measured null on Flash-Next CUDA (row 19, #1762).

## 4. Intel / XPU and Flash-Next specifics (question 3)

### 4.1 Status of the XPU backend [R unless tagged]

- History [R-gh]: proposed 2026-10-08 (#522 Nemotron) and 2026-10-13 (#801 27B on a B70), shipped as a
  build-from-source preview in 1.0.12 (#802). Kernel contract for XPU #2080/#2155/#2269. P2P collectives
  #3585 merged 2026-11-19. 27B TP #3586, Flash-Next #3587 and its speed PR #3603 are open and queued.
- It is contributor-driven. The maintainers state "XPU is hardware we do not have, so your own runs ... are
  the device evidence" [R-gh #3583]. docs/xpu/README.md calls itself an unofficial port. Release archives do
  not carry XPU. The repo description still says Metal, CUDA and Vulkan.
- Active B70 people upstream [R-gh]: Mirrdhyn (original author), SergiioB (B70 cookbook, GPTQ checkpoint),
  tolewis (2x B70 TP=2 fork with a 12-PR series), supernovae (4x B70, the Flash-Next stack), munikor (2x B70
  on Z890, P2P refused), jschmied (LFM3 on Iris Xe).
- Files: zig/src/xpu/ (ze.zig, qwen27.zig, forward.zig, load.zig, runner_multi.zig, slot_states.zig,
  drafter.zig, drafter_tree.zig, tp.zig, tp_bench.zig, rt.zig + context/memory/module/launch/pool for the
  Nemotron-style generic runtime), zig/src/native/xpu*.zig (server hosts), zig/kernels/xpu/ (qwen27/*.cl,
  moe.cl, mamba.cl, nem_*.cl, lfm3*.cl, gemma4/, tp.cl), zig/tools/xpu/ (nGEN generator, rotate_w4.py),
  zig/tests/xpu/ (52 files incl. contract_*.zig), zig/src/core/contract/xpu_records.zig, docs/xpu/*,
  docs/recipes/xpu-arc.md.

### 4.2 XPU facts worth knowing on our card [M-TF unless tagged]

- Prefill is power-bound: "at 230 W, the card runs at ~2,180 MHz during a prompt (PL2), not at the 2,600 of
  pure DPAS. The gate|up GEMM alone sustains 146 TFLOPS fp16 and ~209 TOPS in int8" (docs/xpu/README.md).
- Prompt GEMMs written in nGEN at ISA level: work-group = 2 x 8 sub-groups, 64 x 256 tile, int4 codes fed
  to DPAS as fp16 subnormals with a one-group lag fold, SwiGLU in the epilogue (NAVIGATION.md section 8).
- W4A8 prompt path (int8 activations per row and group of 128): +33% prefill, perplexity +0.04..+0.13%.
  Hadamard-rotated W4A4: +8..12% more. int8 KV with Q.K^T in int8 DPAS.
- Decode attention at long context re-reads K/V once per row; reordering the grid so rows of one slice run
  back to back took 35-37% off attn_decode [R-gh #2543].
- tolewis' fork series on 2x B70 TP=2 at 100K context [R-gh #3358]: weight reads past the cache with an IGC
  builtin (+4.7%), card-to-card exchange with a counter event (+3.7%), one commit launch for all 48 GDN
  layers (+1.0%), vocab head split over both ranks (+2.0%), drafter batching and fused norms (+6.6%).
- Runtime hazard [R-gh #3586 comment by SergiioB]: on compute-runtime 26.05, two devices + timestamp event
  pools + host-reset events on in-order immediate lists that wait on each other hang; fixed in 26.40, or
  NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0. We run 26.22: check before using
  cross-card events with kernel timestamps.
- Guard: every XPU program refuses to start while other processes hold more than 8 GB of device memory
  (docs/recipes/xpu-arc.md; tools/zig/xpu_guard.sh). It would refuse beside our running DD.

### 4.3 docs/flash-next-rotary-bounds.md [R]

It is a numerical-contract note, not a kernel or a performance document. It defines the absolute error
allowance for ten rotary (RoPE/YaRN) probe kernels of the Flash-Next QSA path against an FP64 reference:
scalar caps (rsqrt 4u, exp2 8u, sin/cos 8u absolute, u = 2^-24), exact-fixture arguments (inputs on a 1/8
grid so fp32 sums are exact in any order), phase error propagation with a Lipschitz/Taylor bound, and the
final comparison after BF16 rounding. It says explicitly that the device gate has not passed yet. Useful to
us only as a template for how to bound a rotary kernel; nothing to port.

### 4.4 Flash-Next / qwen4 / GDN elsewhere in the tree [R]

- docs/recipes/qwen3.8-flash-next.md (Python-era + native Metal/HIP), docs/recipes/flash-next-cuda.md,
  docs/recipes/flash-next-vision.md, docs/recipes/qwen4-dense.md, docs/receipts/flashnext-* (lanes, mtp
  chain, prompt chunk 3072, readout rows, sampled heads).
- Formats served somewhere: MLX affine 4-bit g32 and 6-bit (Metal, CUDA, HIP), NVFP4/MXFP8/block-FP8
  ModelOpt (CUDA), EXL3 3.05 bpw (Python CUDA engine; native has only a reference decoder), oQ mixed
  (HIP). CUDA has a ggml IQ2_XXS + Q8_0 grouped-expert path for another MoE (zig/src/cuda/ggml.zig, Format
  .iq2xxs) and a core GGUF reader (zig/src/core/gguf.zig, gguf_repack.zig for K-quants). No IQ3/IQ2
  Flash-Next path exists on any backend.
- HIP note relevant to quant choice (qwen3.8-flash-next.md "Native HIP"): uniform MLX 4-bit Flash-Next
  loses quality (MMLU-Pro 55.9 vs 69.1 for a mixed 8-bit-mixer checkpoint; KL 0.40 vs 0.06). Dense mixers
  dominate decoded bytes on a bandwidth-bound device and are the wrong place to save bits.
- PLE: tables stay memory-mapped on the host; NVFP4 gathers sort row ids, dequantize each distinct row once
  on a worker pool, positional reads by default, optional next-chunk read hints (flash-next-cuda.md).
- QSA: past 2,048 keys each attention layer reads 512 blocks of 4 keys chosen by the block indexer;
  fn_qsa_scores blocks 16 queries against the pooled keys in SLM (1.9x at 256K) [R-gh #3603].
- XPU Flash-Next memory [M-TF #3583]: 19.8 GiB per card at 16K, 23.8-24.4 GiB at the full 262,144 window
  (TP=4); PLE 29.8 GiB in host RAM, gather 0.06 ms per token.

## 5. Lessons to apply to Strata now (question 4)

Ranked by expected gain for C4-C16 without hurting C1. Gains are [I] estimates unless a TensorFold or analyst
measurement is cited; each needs our own A/B. "Analyst item N" refers to fable_decode_window_walk.md section 9.

1. Allocate draft rows by expected tokens per millisecond, per stream, from a measured window-cost table.
   - TensorFold: zig/src/core/lanes/allocate.zig, depth.zig, row_cost.zig, shared_price.zig, cost_rule.zig,
     lone_costs.zig; MTP confidence stop (TF_FLASHNEXT_CONFIDENCE -0.4).
   - Strata today: fixed depth 1 per slot in batch mode, so C2 uses 4 of 8 rows and a lone slot in batch mode
     runs at 88 instead of 134.6 tok/s (analyst item 1). Their rule subsumes "depth by active slot count":
     at C2 it fills the window (analyst: +8..25% per stream), at C4 it gives rows to the streams whose MTP
     head is confident and none to the others, at C8+ it decides plain rounds by price (plain_streams).
   - Add the context-band dimension: a row's cost at 200K context is not its cost at 20K (#2897).
   - C1 safety: keep the solo driver and hand over when a second stream arrives (Lone.resumes).
2. One launch per kernel for all slots, with a per-row table of each row's state/cache addresses.
   - TensorFold: TF_FLASHNEXT_ATTN_LANES (flash-next-cuda.md "One-launch attention"), XPU Mode.multi with the
     MT table (NAVIGATION.md section 3; zig/src/xpu/runner_multi.zig; zig/kernels/xpu/qwen27/decode.cl).
   - Strata: this is analyst item 2 (the lost STRATA_NIGHT_ATTN_BATCH under streamed KV): C4 +7..14%.
     TensorFold confirms the direction and also its limit: their CUDA one-launch path excludes page-table KV,
     which is our streamed-KV case, so the per-slot resolve step stays ours to write.
3. Make keep/commit nearly free and never block on it.
   - TensorFold: whole-window keep = buffer swap; partial keep = replay from stash; lazy "settle"; replay
     folded into the next window (FZ_GDN_REPLAY=1); "one commit launch for the 48 GDN layers" (+1.0%, #3358
     P7); a redundant host sync after keep removed in #3603.
   - Strata: commit + emit are 5.8 ms of the 44.1 ms C4 window and strictly per slot (analyst item 3):
     C4 +6..12%.
4. Draft all slots in one batched pass queued behind the verify without a host wait.
   - TensorFold: latticeMulti / absorbMulti (zig/src/xpu/drafter.zig), shared MTP command buffer on Metal;
     XPU Flash-Next draft step 6.45 -> 1.74 ms per round after batching and DPAS tiles (#3603).
   - Caveat: their ledger records shared draft rounds as a null on Flash-Next CUDA (#1762). Bound it first.
5. Interleave prompts with decode a chunk per round, with adaptive solo/busy chunk sizes and a fill order.
   - TensorFold: Backend.prefill_step, lanes/solo_chunk.zig, lane_host_fill.zig, lane_host_srpt.zig,
     prefill_many, cancel between chunks.
   - Strata: Python admits one request at a time, all prompts go through the one main session, and a
     12,288-token chunk runs with every slot stopped (opus_sched_prefill_restore_walk.md section 2). At about
     1,000 tok/s that is a 12 s stall per chunk for running streams; TensorFold holds p95 gaps near 0.2 s
     with a 64-row busy chunk and keeps prefill speed with a 2,048-8,192 solo chunk.
   - What to build: prompt fill in the request's own slot (no main-session read then copy_to_slot), a busy
     chunk of a few hundred rows when anyone decodes, round-robin between filling slots by least remaining
     work. Cold C4 TTFT 17/39/61/84 s becomes fair and short prompts stop waiting behind long ones; total
     prefill work does not shrink.
   - Later: coalesce prompt rows and decode rows in one pass so each layer's experts are read once for both
     (their Rounds planner; Strata has this only in the CUDA engine as BCO).
6. Prefill throughput: no host sync inside the layer loop.
   - TensorFold: routing and the expert plan run on the device, the whole forward is queued and synced once,
     the PLE gather overlaps layer 0 (PR3603 xpu_forward.zig).
   - Strata prefill copies T*K*4 bytes of expert ids to the host and waits, per layer (48+ host blocks per
     chunk; opus note 1b.4). Moving the grouping to a device plan kernel removes them. Expected +5..20% on
     prefill [I, unmeasured; bound it with a per-layer host-wait timer first].
   - Also check power: log card frequency and power during prefill. Their B70 sits at 2,180 MHz at a 230 W
     cap vs 2,600 MHz possible; if our cards are capped the same way, prefill has up to about 19% in the
     power limit alone [I].
   - Integer-activation prompt GEMMs (their W4A8: +33% on the 27B) are the analog of our XMX prefill; their
     nGEN kernels (zig/tools/xpu/w4_prompt.hpp, w4a8_prompt.hpp) are a reference for register-level DPAS
     scheduling under Apache-2.0.
7. Share pooled-key and K/V reads between the rows of one slot.
   - TensorFold: fn_qsa_scores blocks 16 queries per pooled-key pass (1.9x at 256K); attn grid order so rows
     share K/V in L2 (-35% attention time).
   - Strata at 200K: the indexer's pooled keys are 128 B per context token per layer, about 25.6 MB per
     layer per row and about 307 MB per row over 12 QSA layers [I from analyst section 2], the same order as
     a row's expert bytes. Rows of the same slot (2 at C4, 4 solo) should score in one pass.
8. Restore/TTFT: prompt-cache marks and prefix matching inside the engine.
   - TensorFold: marks at the system-prefix end and before it, power-of-two intermediates, radix prefix
     store (zig/src/core/prompt_radix.zig), no Python in the request path.
   - Strata: 0.53 s of the 2.08 s forced-restore TTFT is Python, including an O(n log n) list-slicing prefix
     search on 132K-token prompts (opus note 1a.4). Replace it with a radix/hash-of-prefix lookup.
9. Adopt their gates as cheap regression tripwires.
   - Log "window ms by rows" (R = 1..8, fastest of 7) at server start and compare with a stored table; fail
     loudly at 15% drift (cost_rule.zig drifted()). This would have flagged the lost batched-attention path
     (run 32.4 -> 38.4 ms at R=8) at first boot of the 0.2.16 port.
   - Token-hash equalities: concurrent == solo and drafted == plain at temperature 0, resumed == fresh
     (tools/bench_concurrent.py --alone --serial, 361 lines of stdlib Python against an OpenAI endpoint; it
     needs a token_sha256 per reply, which Strata's server would have to emit). Only meaningful if our batch
     kernels are row-invariant; worth testing once, because if they are, most of our coherence checks
     collapse into one hash compare, which is what AGENTS.md asks for (less test cost).
   - bench_fill-style metric: p95 inter-token gap of a running stream while another prompt fills.
10. Process: "bound before building" and a ledger with failed rows (docs/optimizing.md,
    docs/optimization-ledger.md). Write the ceiling and the bar before coding; record nulls. Nulls they
    already paid for on Flash-Next that we should not repeat without a new reason: L2 prefetch side stream
    (+0.45%, null), small-kernel fusion (bounded 2.4-4.0% of a round, not built), prefetched/double-buffered
    stream tiles (no gain), prompt grid sized by filled items (at most 1.2%), add+norm fusion (bound under
    1%). Wins they measured: grouped experts (big), solo chunks (TTFT -5..-10%, 15-23% at one stream), direct
    load past the page cache (cold load 74 -> 48 s), fused projections that share an input (+3..13%),
    scale index (10% fewer expert bytes), dependent launches (+4..7%, CUDA only).
11. Per-kernel audit table from one profile: time per window, launches, achieved GB/s against the 608 GB/s
    floor, grid vs EU count (docs/optimizing.md "The per-kernel audit"). Our window law is 13 ms fixed + 3.2
    ms per row; the byte floor is about 0.84 GB of routed-expert bytes per row, 0.7 ms on two cards [I], so
    the per-row term is about 4-5x above bandwidth. That says the rows are launch/latency-bound, not
    bandwidth-bound, which favors items 2-4 over kernel micro-tuning.
12. Level Zero mechanisms usable under SYCL or beside it: device-scope cross-card events and peer-resident
    buffers (ze.zig sharedEvents, makeResident; tp.zig) instead of host-mediated EP waits; kernel-timestamp
    event pools for per-kernel GPU time. Mind the 26.05 hang noted in 4.2.

Not a lesson: their bit-exact rule. We serve a lossy IQ3 pack and int8 KV by choice; adopting the rule
wholesale would cost prefill speed (see Nemotron XPU). Adopt the test method, not the constraint.

## 6. A Zig Intel engine: extend TensorFold or go greenfield (question 5)

### 6.1 The question as posed is already answered by upstream

A Level Zero backend with offline-compiled kernels (ocloc), zeModuleCreate, immediate command lists, peer
P2P collectives, a kernel contract for Xe2 launch limits and a Flash-Next family (open PRs) exist. Adding one
is not the work. The work is the delta between that stack and our serving requirements.

### 6.2 Delta between PR #3603 and our Flash-Next goals

| Need (AGENTS.md goals) | TensorFold XPU state | Work |
|---|---|---|
| Fit 2x B70 (64 GB) | MLX 4-bit: about 79 GiB on cards [M-TF]; no TP=2 fit [I] | sub-4-bit experts |
| ISTA GSQ-RCO IQ3 GGUF | GGUF reader + CUDA IQ2_XXS only; no IQ on XPU [R] | IQ kernels in OpenCL C (a) |
| No fast P2P (Gen3) | P2P required; no fallback yet [R-gh] | probe on 7.1 (b); host-staged path |
| 200K per stream, C2-C4 | bf16 KV, 5.3 GiB per 200K stream [I] | int8 KV for Flash-Next XPU |
| KV to RAM, parking | none on XPU; core host tier open (#1501) [R] | host KV tier + park tier |
| Image input | --vision refused on XPU [R-gh #3587] | CPU or XPU vision tower |
| 150+ C1, 2000+ prefill | 158-176, 1,245-1,640 on 4 cards Gen5 [M-TF] | measure on 2 cards Gen3 (c) |
| Batch MTP | confidence chains, shared rounds to 32 streams [R-gh] | none |
| OpenAI server, grammar | yes (xgrammar optional) [R] | none |

(a) IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL, Q2_0 experts and Q4_K/Q5_K/Q6_K/IQ4_XS dense, row-invariant, with FP64
    references if upstreamed. (b) canAccessPeer was True on kernel 7.0 with IOMMU off (docs/P2P_GPU.md H.11).
(c) Expect lower than their numbers on Gen3 + Zen1 [I].

Rough size [I]: the IQ expert/dense kernel family is the largest item (Strata's ISA IQ decode is the hard
part we already own); KV int8 + host tier + parking is a second project of similar size; P2P fallback and
vision are smaller. Toolchain: this host has zig 0.16.0 (0.17.0 required) and /usr/bin/ocloc; nGEN headers
are needed for the 27B prompt GEMMs.

### 6.3 Options

A. Evaluate first (days, not weeks). Build refs/pr/3603 with zig 0.17, run tf-xpu-tp-bench on our two cards
   (does openPeers pass on 7.1; latency and GB/s on Gen3), then xpu-qwen27 on one card with the SergiioB GPTQ
   checkpoint against our 27B numbers. This costs no engineering and tells us what Gen3 + Zen1 do to their
   numbers. Flash-Next itself cannot be loaded at TP=2 for the memory reason above. Needs the DD stopped
   and both leases (their 8 GB guard refuses otherwise).
B. Thin fork of TensorFold as the Zig Intel engine (recommended over greenfield if the user wants Zig).
   We inherit about 280K lines of server + core that a greenfield engine must rewrite (HTTP/OpenAI API,
   tokenizer, templates, sampling, grammar, lane scheduler, prompt cache, ledger) and a working Level Zero
   layer, and write only the section 6.2 deltas as a family variant. Risks: upstream moves very fast (PR
   numbers past 4,000 in five months; files change under open PRs; rebases are constant), the XPU Flash-Next
   code is days old and unreleased, and its row-invariance rule shapes every kernel.
C. Greenfield Zig engine. Only justified if we want none of their constraints. We would still copy the same
   structure (ze.zig-style dlopen binding, ocloc/nGEN build steps, lanes-style scheduler). I see no technical
   reason to pay for the server and scheduler twice [I].
D. Stay on Strata (SYCL) and port ideas (section 5). Lowest risk for the daily driver; keeps IQ3, int8 KV,
   RAM streaming, parking and vision that TensorFold lacks. Does not give the user a Zig engine.

My recommendation [I]: D now for the C4-C16 campaign (items 1-5 of section 5 are scheduler/launch-structure
changes that do not need a new engine), A in parallel as a cheap probe, and decide B vs C only after A,
because A's numbers on our Gen3/Zen1 box decide whether their TP design or Strata's EP design is the better
base here. Their TP does two all-reduces per layer on every card; our EP keeps the dense path on card 0.
On a host without fast P2P the EP shape is likely the better one for decode latency [I].

### 6.4 License and contribution implications [R]

- Apache-2.0 from 0.6.0 (LICENSE, NOTICE); code written before 0.6.0 keeps MIT (LICENSES/MIT.txt). Third
  parties in THIRD_PARTY_NOTICES.md. nGEN is oneDNN (Apache-2.0); xgrammar Apache-2.0; exl3 references MIT.
  No CLA appears in CONTRIBUTING.md: "TensorFold is Apache-2.0; a contribution follows that license."
  A private fork or copying kernels into our tree is allowed with the license text and notices kept.
  Whether Apache-2.0 code can go into Strata upstream depends on Strata's license (not checked here).
- Upstream process (CONTRIBUTING.md, docs/recipes/SKILL.md, docs/pr/SKILL.md): open an issue and agree a
  gate before a new family, backend, format or arithmetic change; claim the issue with their ATC tool
  (publishes an anonymous session to a public radar unless --local); one change per PR; source files at
  most 600 lines; lean checker must add zero findings; every new or changed kernel registers with the
  kernel contract and posts its SUMMARY line; receipts must hold no machine names, paths or addresses.
- Hard rules that affect us: every change keeps every token (bit-exact, even opt-in) [R-gh #3358];
  "what lands is written from our own code and the model's own definition, never read from or ported from
  another engine" (docs/optimizing.md), so Strata-derived kernels cannot be upstreamed, only clean
  implementations; "tool co-author trailers do not belong in the history" (CONTRIBUTING.md), which conflicts
  with our commit attribution lines for anything sent upstream.
- A new lossy checkpoint format (GGUF IQ3) is a new reader, not a token change to an existing path, and
  there is precedent (IQ2_XXS on CUDA), so it is plausibly acceptable with FP64 references and its own gate
  [I]. The maintainers cannot run Intel hardware; our B70 receipts would be the device evidence.
- Useful low-cost contribution if we engage: a 2x B70 cross-die Gen3 run of tf-xpu-tp-bench and the
  host-staged collective fallback that #3585's thread asks for; both match hardware only a few people have.

## 7. Limits of this report

- No TensorFold code was built or run. All their numbers are theirs (4x B70 on EPYC Turin with Gen5, one
  B70 at 230 W, GB10), not reproduced on our host.
- PR #3603 was read at head 207aaae1; it is being rebased by the maintainer and may change before landing.
- The lane scheduler (13K lines) was read at the level of module headers, the backend vtable, allocate.zig,
  depth.zig, fill/srpt/solo_chunk and the XPU docs, not line by line. docs/kernel-contract.md was read to
  the end of "Known findings" (the per-kernel coverage tables were skipped).
- The TP=2 misfit (39.6 GiB per card) is arithmetic from their per-card footprint, not a load attempt.
- P2P availability on our host is taken from docs/P2P_GPU.md (kernel 7.0, IOMMU off), not re-probed on 7.1.
- Gains in section 5 are estimates tied to the analyst notes' window law; none is measured.

VERDICT -> TensorFold is the closest existing thing to "a Zig inference engine for Arc B70": it has the
Level Zero layer, B70-specific kernels and, in open PRs, Flash-Next itself with batch MTP and 32-stream
shared rounds. It does not replace Strata today (4-bit only, 4 cards needed, no RAM KV tier, no vision,
P2P required). Its scheduler design (price-driven row allocation, one forward per round, chunk-per-round
prompts, lazy keep) is directly portable to Strata and targets exactly the C2-C16 losses our analysts
measured. If a Zig engine is built, fork TensorFold's XPU stack rather than starting from zero.
