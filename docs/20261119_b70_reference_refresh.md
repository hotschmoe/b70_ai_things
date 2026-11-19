# B70 reference refresh, 2026-11-19

Scope: bring Steve (steveseguin) and Sergio (SergiioB) reference trees up to date and digest B70/Xe2 relevant changes.
Method: `git fetch --all --prune --tags` only (no checkout/reset; pinned detached HEADs unchanged); cookbook cloned fresh.
Last sync points: Steve lab 54aefaf06 (2026-09-08 pin), steve-s2b vllm-xpu-kernels 2dd55f3 (2026-08-17), launcher 06bf7af (2026-10-16),
cookbook not previously cloned here (earlier audit: docs/flashnext_udq4xl_paused/20261011_flashnext_steve_sergio_strata_performance_audit.md).
Relevance key: (a) Strata Flash-Next concurrency, (b) vLLM/27B fallback, (c) not relevant.

## 1. New HEADs fetched

| Tree | Remote | Old | New HEAD | Note |
|---|---|---|---|---|
| steve-repro/qwen38-int4-neural-20260908 (lab) | steveseguin/b70-optimization-lab | 54aefaf06 (09-08) | origin/main c902eab53 (2026-11-17) | 11,300 new commits; tags ctxkeeper-*, sc117-repro-draft-20261028 |
| steve-s2b/vllm-xpu-kernels | origin steveseguin/... | 2dd55f3 | origin/main 0fd18a7 (fork frozen 2026-08-18) | Steve's fork has had NO pushes since 08-18; real news is upstream |
| steve-s2b/vllm-xpu-kernels | upstream vllm-project/vllm-xpu-kernels | - | upstream/main 40356aa (2026-11-18) | 104 non-merge commits since 08-18; tags v0.1.14..v0.1.17, nightly-v0.1.18-20261119 |
| steve-s2b/vllm | vllm-project/vllm | 44fc8fde09 (08-18) | origin/main 878ae13ff0 (v0.34.0rc0-971) | 8,969 commits; ~150 XPU-tagged since 08-18 |
| steve-s2b/oneccl-src | uxlfoundation/oneCCL | b52f40c | master 2f2a676 (2026-10-26) | docs-only (spec pages); tags 2022.1.1/1.2/2.0 |
| scratch/b70-launcher | SergiioB/b70-launcher | 06bf7af | main 7a5f446 (2026-11-13), v0.6.15..0.6.26 | worktree is on branch feat/strata-engine (clean); only local ref `main` advanced via fetch origin main:main, worktree untouched |
| refs/intel-arc-pro-b70-inference-cookbook | SergiioB/... | new clone | master d5a9b50 (2026-11-19) | /mnt/cache/b70/refs/intel-arc-pro-b70-inference-cookbook |

Other Steve repos (gh repo list steveseguin): only b70-optimization-lab is B70 related (pushed 2026-11-17); no new B70 repos.
vllm-e190 and the other steve-repro dirs share the same remotes; not fetched separately (pinned controls).

## 2. Steve

### 2.1 Big picture
Steve moved off vLLM entirely. Since Sep 2026 the lab is a home-built SYCL/C++ engine ("own runtime") for Qwen3.8 Flash-Next SC117 IQ3_S
(same model family as our Strata target) and, since Nov 9, DeepSeek V4.1 on SIX cards (mixed 275/230/550 W boards; not B70-only).
Engine is single-stream, bit-exact-vs-reference by policy; no batching/concurrency work. He is therefore BEHIND us on Flash-Next decode:

| Steve config (README 2026-10-26..31) | Result | Ours (daily driver) |
|---|---|---|
| 2 B70, chat service, 64K ctx, 12 cold prompts | 66.21 tok/s (72.45 after two-card flags, 12/12 prompts, A/B/A/B) | C1 115-141 |
| 4 B70, each stage split TP across 2 cards, 64K | 87.63-94.18 tok/s fresh servers, outputs identical | n/a (we are 2-card EP) |
| 4 B70, 121,854-token prompt @128K | first token 213.0 s (~570 tok/s prefill), then 84 tok/s | prefill ~2000 tok/s |
| 3 target + 1 draft B70, 262K | 58.57 tok/s; 262K prompt 250.8 s, 48.5 tok/s | |

Verdict: his absolute numbers are not a target; his methodology (profile per site, bitwise gates) and a few low-level findings are useful.

### 2.2 Findings worth porting/testing (B70/Xe2 level)
| # | Finding (commit) | Measured | Relevance |
|---|---|---|---|
| S1 | Hidden per-kernel private memory: kernels that index a launch-arg table by a runtime value spill the argument block to private memory; driver reserves ~35-70 MiB PER KERNEL (per hw thread x lanes), first run, never freed; 17 kernels = 0.94 GiB/card disappeared from "free after load" (7a0a9bdd3, round19-privmem/README.md). Detect with zeinfo per_thread_memory_buffers (privmem_check.py). Fix: tables in device memory or constant-index. | pr5->pr6: free mem flat (2.57 GiB vs 1.75), results bit-identical, speed unchanged; fixed 3/8 "exchange timed out" failures at 194K | (a) HIGH: 4 slots x 200K KV on 32 GB cards is memory-tight and our elastic-KV/RAM overflow plan depends on exact free VRAM; audit Strata SYCL kernels with zeinfo for per_thread_memory_buffers >= 30 MiB. CPU-only check. |
| S2 | Two-card default-off flag bundle on the 4-card stack gave +9.6% on 2 cards: deferred event retirement, paired compact expert launch, HC fusion, rank-0 snapshot/rollback trims (7aca12ec7, fixed-cost-2026-10-30: paired expert launch -0.34 ms/half, deferred event retirement -0.30 ms/half = 78.75->84.49 tok/s on 4 cards) | 66.09->72.45 tok/s (2 cards) | (a) MEDIUM: both ideas (retire events lazily; fuse the two expert launches) are generic fixed-cost levers on a launch-bound step; check whether Strata already defers L0 event retirement per token. |
| S3 | Decode is launch/latency bound: 863/938 kernels per rank per step, 84% of launches <20 us = half of GPU time; 0.12 ms gaps; experts only 26-42% of 590 GB/s roofline (decode-site-census-2026-10-30) | measured per site | (a) MEDIUM: same shape as our C1; tells where C4 batching headroom is. |
| S4 | Multi-row (verify/batch) decode cost: batched-prefill path costs 2.7-3.0x of a 1-row pass at 2-4 rows; choosing existing small-row tiles gives 1.55x @2 rows, 2.2x @4 rows; consecutive tokens share only ~4 of 10 experts so 2/3/4 rows touch 16/22/27 distinct; perfect weight-stationary kernel projects 1.2x@2, 1.5x@4 (same census; K1 small-row token tiles c00bacafe; K2 grouped expert kernels slower, closed) | exact, bitwise | (a) HIGH for C2/C4: expert union growth is the real limiter of multi-slot decode; shows grouped-by-expert kernel LOST to per-slot kernels (DO-NOT-REPEAT). Confirms our C4 aggregate (~188) is near what expert-streaming allows without new kernel tiles. |
| S5 | Transport: EU peer stores ~1 GB/s regardless of cache hint, several destinations at once are slower; ONE peer at a time 41 GB/s per card; copy engine (BCS) 27-30 GB/s beside compute; copy-engine prompt pushes -8% on 3K prompt (5.42->5.02 s) (44e081735, c017ecf0e, 459c9d8db); peer copy 18 us/10 KiB, 52 us per exchange round trip (SCOREBOARD) | measured | (a) MEDIUM: for prefill under load and EP exchanges, use copy engine (blitter queue) pushes instead of EU stores to overlap exchange with compute; relevant to our card-balance / exchange items. |
| S6 | Long context: index-score/top-k attention costs dominate (+65 of +89 ms/block at ~189K); dtopk (device top-k), idxrp, kvshard/kvdedup gave 1M prompt read 500->551 tok/s, writing 104.5->130.7 tok/s with guessing (cac7070fd, DeepSeek only) | DeepSeek V4.1, 6 cards | (c) mostly; method only. |
| S7 | Host stability (AMD host, 6 cards): repeated host resets traced to stuck PCI config reads / data-fabric sync flood; mitigations tried: stable mode (OWN_PX_SERIAL=1 serial exchange), Gen4 forced, iommu=pt, SMT off in BIOS, 200 W caps (-3% write, -10-13% prompt); GPU GuC crash/off-bus logged first on last reset (2026-11-15 16:56). Not declared fixed. | 54/54 starts stable-mode; 12/12 SMT-off | (a)(b) LOW-MED: keep kernel-log follower + serial exchange option in mind if our 1950X box ever shows resets; do not copy BIOS changes blind. |
| S8 | oneCCL: only docs (spec pages) since pin; no code change | - | (c) |

### 2.3 vllm-xpu-kernels upstream (Intel) since 2026-08-17 -- 104 commits (this is where Steve's fork pins would be updated)
Fork pin 1e90ffa/2dd55f3 is ~3 months stale. B70/Xe2-relevant items (SHAs are upstream/main):
| Commit | Date | What | Rel |
|---|---|---|---|
| 40356aa #671 | 11-18 | Xe3P chunk-prefill/paged-decode update (not B70) | c |
| 6a59a3d #670 | 11-03 | fused routed/shared MoE decode for Qwen3.6-35B-A3B | (b) |
| 8412164 #605 | 11-06 | Remove out-of-spec scale block-2D prefetch in Xe2 grouped GEMM (correctness on Xe2) | (b) |
| 470c19f #647 | 10-30 | FA refined for spec-decode adaptive verification | (b) MTP |
| 78799fe #618, 1d5b4f5 #537, da16a55 #600, 5802a41 #544, b73bdc3 #660, daf6240 #658 | 08-19..11-02 | GDN spec-decode: recoverssm kernel, ragged spec traversal, conv-state layouts, NULL_BLOCK_ID skip, 64-bit conv offsets | (b) MTP+GDN correctness |
| 17c93a1 #568 | 09-14 | BlockFP8 Xe2 perf: scale in accumulator | (b) |
| 6231eb3.. 2e113b6 #557, f1c4861 #519 (fork), b027ac8312 (vllm #51956) | 08-14..10-09 | KV-offload pointer overhead, xpu_host_register pinned mmap DMA, KV offload mmap as pinned host memory | (b) relevant to our "evict KV to RAM" goal on vLLM |
| 1d70984 #595, 9a0d51a #577, 3063346 #641, a397c58 #532 | 09-14..11-02 | wider act_and_mul vectorization, fused LayerNorm/GemmaRMSNorm, qk_norm_rope | (b) |
| 9b18db1 #561, 5d70547 #576, 4111ba2 #582 | 09-02..09-09 | faster topk/topp sampler, compact_sampling_mask | (b) |
| 6be8122 #621, 0bfb372 #525 | 09-16, 10-30 | torch 2.14 -> 2.15.0 XPU; oneDNN 3.13/3.14.2 | (b) stack bump (one layer at a time per AGENTS) |
| a05f550 #503, a96998c #502, 53301ca #488 | 08-10 | MoE: SiLU/TopK-16, native Xe2 MXFP8/block-FP8 | (b) |

vLLM main XPU highlights (since 08-18; ~150 commits): XPU graph enabled in tests (bfa7e5b7af), Triton W8A8 block-FP8 GEMM tuned for B70 (bdfb02674f #56063), Triton fused MoE tuned for XPU (f7f62d1916), Mamba SSU configs for B70 shapes (4cf1307e72), GDN metadata reuse across KV groups (d42c8a8d8f #64479), triton greedy_argmax for spec decode on XPU (7a10561277 #67196), spec-decode adaptive verification (35d01660ff, 06dbdda4b2), Qwen3.5 eager port with fused MoE decode (dd60cb8540 #64687), XPU all-to-all EP backend for batched MoE (0555390170), batch-invariant collectives with sharded reductions (2deea9af98), KV offload pinned mmap (b027ac8312), startup heap freeze in XPUWorker (f35d1e2150). All (b); the first four are plausible wins for the 27B fallback and Flash-Next vLLM campaign, none needs a fork.

## 3. Sergio (SergiioB)

### 3.1 b70-launcher (06bf7af..7a5f446, 10-16..11-13; v0.6.15-0.6.26)
- 10-18: xestrata recipe for Qwen3.8-Flash-Next on 2 B70 (claims 122.2 tok/s in commit 050afcc), IQ3_S recipe (d915e17), YaRN 350K ctx, vision flag, --effort. Later cookbook measurement is lower (below).
- 11-10..13: storage manager, speed/energy telemetry, portable recipes, "desktop apps no longer block a card", EXL3 honors --no-mtp, docs for running Strata natively on Windows with 2x B70 (39bd130).
- New branch wip/ornith-122b-recipe-20261020. Relevance: (c) except recipes as cross-check of Strata flags; our own launcher state (hotschmoe-dd) is ahead.

### 3.2 inference cookbook (421 commits; HEAD d5a9b50, 2026-11-19)
Authoritative numeric file: data/benchmarks.v1.json; ledgers DO-NOT-REPEAT.md (33 closed arms), WHAT-WORKED.md.
| Topic | Measured (config) | Rel |
|---|---|---|
| XeStrata (MistvVK/XeStrata xe0.1.45-1, pipelined MTP4, int8 attn cache, 262K ctx) Flash-Next IQ3_S vs IQ3_XXS, 2 B70 | C1 code decode 86.7 / 86.8 tok/s (thinking off, P34/G512, n=3, GPU0 180 W/GPU1 230 W); prefill 1879.5 / 1894.2 tok/s at 32,429-token prompt; no concurrency measured ("Not measured: concurrent requests") | (a) cross-check only: our C1 115-141 and prefill ~2000 are ahead; his doc confirms IQ3_S fits but GPU1 nearly full (24,488/24,576 resident expert slots) |
| llama.cpp SYCL Flash-Next AtomicChat M64 | 33.3 tok/s C1 2 cards; opt-in sparse attention +7-9% decode at 28K (PR #28796); Xe2 TILE quantized-KV decode dispatch +13% @32K (PR #26689) | (c) |
| llama.cpp -sm tensor with local peer-store allreduce patch (patches/llamacpp-sycl/sycl-tp-p2p-0002-peer-store-mode-2.patch) | Qwen3.8-27B Q4_K_M, tg128 32.18 tok/s vs 23.28 stock tensor-split, 28.28 one card; pp512 544 vs 558 | (b)(a) interesting: P2P peer-store allreduce is the trick that makes 2-card TP beat 1 card on llama.cpp; same idea as Steve S5 |
| TensorFold (ashhart/TensorFold, Zig/SYCL engine) + DFlash2 drafter, Qwen3.8-27B 4-bit | 1 card: 123-128 tok/s code at 68-token prompt (drafted text byte-identical to greedy), 90 @32K with prefix attention kernel (XPU_ATTN_PREF=256), 51 @131K; 2 cards --tp 2 (PR #3586): 173.3/151.4/123.8/81.6 tok/s at 68/8K/32K/131K (230/230 W); round 31.65 ms; 62.3 tok/s at 249,784 tokens | (b) HIGH for the 27B fallback single-stream: 128 tok/s lossless is ~2x our vLLM 27B INT4; also shows two cards TP gives only 1.3-1.5x |
| Concurrency datapoints (same host) | 4 streams: two one-card TensorFold servers 159-161 tok/s aggregate vs 129-130 for one two-card server; greedy 8-32 streams on one card: exl3xpu 342-375 vs TensorFold 271-298 tok/s; TensorFold 183/183 streams equal solo text, exl3xpu 49/183 | (b)(a) policy: for independent users run one server per card (DP) -- already our fallback stance; useful batching ceiling numbers for 27B |
| EXL3 (exl3xpu) 27B one card, trellis 4bpw + MTP3 + fp8 KV | 73.8 tok/s @32K, 48.2 @131K, prefill 2,455 tok/s @32K; MTP3 changes greedy output on 27% of a fixed 329-item set (rows labelled lossy 229-241/329) | (b) |
| Warm-turn prefix cache: TensorFold --prompt-cache-gib sized to one full state | follow-up turns 0.3-1.1 s up to 131K (22.5 s with 2 GiB default); 250K warm follow-ups 1.1-1.6 s vs 205 s (int8 K/V, fix branch xpu-tp2-fixes-20261119) | (a) concept: matches our RAM-backed KV restore-without-recompute goal; reference point for expected warm TTFT |
| Equal caps matter | both cards 230 W/2800 MHz raised two-card rates 5-6% vs 180/230 W | (a) check our card power caps / clocks (xe sysfs) match on both cards |
| Evidence discipline | spec-parity column (does drafter change greedy answer), DO-NOT-REPEAT ledger | (a) process only |

Other B70 repos seen recently on GitHub (not required; skim candidates): Red-Weasel/machx-inference-engine (C++/SYCL, Qwen3.8/DeepSeek-V4.1 on B70, multi-GPU, updated 11-19), NoobBandi/Qwen3.8-QSA-vLLM-XPU (QSA sparse prefill/decode/MTP on Xe2 vLLM, 11-12), MikkoP88/vllm-arc (vllm fork: XPU full-decode graphs + LMCache hybrid-KV + MTP k3 on 2x B70, 11-11), snailium/llama.cpp-sycl-intel-b70, newjordan/NexN2-B70-Turbo (reorder-on-MoE SYCL kernel), urakozz/intel-arc-inference-server, tennatoppa/TennaToppa (xe GuC stalls under sustained inference, kernel 7.0.0-42).

## 4. Ranked actions for concurrency (Strata Flash-Next)
1. Audit Strata/our patches for hidden per-kernel private memory (S1): run the zeinfo per_thread_memory_buffers check on the v0.2.16 binaries; any kernel >=30 MiB should have its launch-arg tables moved to device memory. Reclaims VRAM that 4x200K KV needs; CPU-only, no lease.
2. Quantify multi-slot expert union cost (S4): Steve shows 2/3/4 rows touch 16/22/27 distinct experts of 10 each; use that model to predict our C2/C4 aggregate ceiling and to decide whether expert-grouped kernels (he found them SLOWER) are worth trying -- likely not.
3. Copy-engine (BCS/blitter) pushes for inter-card exchange/prefill (S5): -8% prefill at 3K; a candidate for prefill-under-load and card-balance items.
4. Deferred event retirement + paired expert launch (S2): -0.3 ms/half each (~+6-7% on his 4-card, +9.6% 2-card flag bundle); check Strata for equivalents.
5. 27B fallback: evaluate TensorFold + DFlash2 (and equal power caps) as a faster 27B single-stream fallback (123-128 tok/s lossless one card) and vLLM main XPU commits (GDN spec-decode fixes, B70-tuned Triton MoE/FP8, KV offload pinned mmap) for the vLLM campaign.

Not recommended: porting Steve's DeepSeek long-context kernels; XeStrata numbers as targets (we exceed them); BIOS/SMT changes from his host-reset investigation without evidence of resets here.
