# Intel Arc Pro B70 — LLM Inference Findings (hobbyist field notes)

What we've actually measured running LLMs on a single **Intel Arc Pro B70** (32 GB GDDR6,
Battlemage/Xe2, ~$949), in Docker on an Unraid host. Goal: help fellow Team Blue tinkerers
skip the dead-ends. Living doc — see [archive/RESULTS.md](archive/RESULTS.md) for older raw number
tables (Qwen3-14B, superseded) and [JOURNAL.md](JOURNAL.md) for the blow-by-blow.

## TL;DR
- **The B70 is a solid single-card inference GPU for ~14B-class models.** Qwen3-14B at **FP8**
  does **~35 tok/s single-stream** and **~556 tok/s aggregate** at concurrency 64, near-lossless.
  (Default `--max-num-seqs 16` caps you at ~330 — raise it for throughput.)
- **Best backend: upstream vLLM-XPU built from source** (`Dockerfile.xpu`). It has the real XPU
  FP8 matmul kernel and native model support. (llama.cpp SYCL also works well for standard models.)
- **INT8 W8A8 now WORKS on Battlemage — we wrote the kernel.** Stock vLLM had no XPU int8 kernel (hard
  `KeyError: PlatformEnum.XPU` at load). We added a native oneDNN `int8_gemm_w8a8` (s8s8s32) + a fused
  per-token int8 quant + `XPUInt8ScaledMMLinearKernel` + the registry entry. W8A8 serves, beats FP8 ~1.6x in
  prefill, ~matches decode, and composes with FP8 KV cache (2x context). FP8 still wins pure single-stream
  decode; W8A8 wins prefill/large-batch — the long-context coding-server pick. docs/kernel/02 + `contrib/vllm_int8_xpu/`.
- **PIECEWISE XPU graph capture = free +16.7% decode (2026-06-19).** vLLM 0.23.0 already wires XPU graph
  capture (`VLLM_XPU_ENABLE_XPU_GRAPH=1`); adding `register_fake` meta kernels for our 2 custom int8 ops lets
  dynamo trace through them. PIECEWISE mode (attention eager, linear/MLP captured) lifts 14B W8A8 decode
  23.33 -> **27.23 t/s**. FULL capture is blocked by Intel's SYCL Graph ext (`work_group_scratch_memory`, via
  flash-attn). Use image `vllm-xpu-env:int8g` + `cudagraph_mode=PIECEWISE`. (This w8a8 number predates the
  compile-pass fix below and is being re-confirmed on the current image.)
- **[BREAKTHROUGH 2026-06-20] PIECEWISE graph capture lifts W4A8 decode 16.79 -> 48.18 t/s (+187%, 2.87x).**
  The biggest decode win on the B70 so far. W4A8 eager is severely dispatch-bound because its per-token
  activation quant is the UNFUSED pure-PyTorch `dynamic_per_token_int8_quant_ref` (hundreds of tiny ops/token);
  graph capture fuses the whole decode step, taking decode from ~26% to ~74% of the BW ceiling (9.3 GiB @
  608 GB/s ~= 65 t/s). That is why W4A8 gains 2.87x where W8A8 (which uses a fused quant op) gained only 1.17x.
  **The real headline (corrected after capturing the rivals too): graph capture roughly DOUBLES int4 decode on
  the B70.** Apples-to-apples, PIECEWISE: **w4a16 28.04 -> 54.57 (+95%)**, w4a8 16.79 -> 48.18 (+187%), w8a8
  23.62 -> 26.68 (+13%). So **w4a16 LEADS decode (54.57), w4a8 leads prefill/TTFT** (int8-XMX) -- both 9.3 GiB,
  both ~2x'd by capture; w4a16's int4xfp16 GEMM is ~13% more BW-efficient at m=1 (no act-quant). w8a8 gains
  little (already-fused quant). W4A8 is no longer "dominated" -- it co-leads with w4a16, split by decode-vs-
  prefill. (Earlier note "w4a8 beats everything" was only true while the others were eager; corrected here.)
  Two gotchas fixed (both in `w4a8/30_serve_w4a8_graph.sh`): (1) a vLLM XPU+compile crash
  `NameError: MLARoPEKVCacheCatFusionPass` -- vLLM auto-enables CUDA-only inductor fusion passes under
  torch.compile but XPU never imports them; disable the CUDA/ROCm-only fusion flags in the serve `pass_config`.
  (2) the int4 `register_fake` is REDUNDANT -- vLLM already ships a native fake for `int4_gemm_w4a8`. Serve:
  `w4a8/30_serve_w4a8_graph.sh GRAPH=1` (image `:int8g`, dtype float16); confirm via the
  `Capturing CUDA graphs (PIECEWISE)` + `Graph capturing finished` log lines.
- **Quant quality measured (2026-06-19): the int8 ACTIVATION quant is the quality cost, not int8 weights.**
  First eval campaign, Qwen3-14B × 6 quants (see [evals/](evals/) + [evals/results/SUMMARY.md](evals/results/SUMMARY.md)):
  ppl + top-1 token-agreement vs a CPU-scored bf16 anchor + gsm8k. **W8A16 (int8 w / fp16 a) is near-lossless**
  (ppl 12.76 vs bf16 12.70, **0.981** agreement) but **has NO XPU kernel** (`XPUwNa16` is int4-only). Quantizing
  activations to int8 (W8A16→**W8A8**) drops agreement 0.981→**0.881** — yet gsm8k barely moves (95.3% vs fp8 96.0%),
  because it flips low-confidence tokens, not answers. **Weight bits dominate:** W8A8 > W4A16 on every metric.
  **Kernel takeaway:** keep optimizing **W8A8** (only path that lights the INT8 systolic datapath; ≈fp8 task
  quality, 15 GB); a W8A16 int8-weight kernel is the one gap but it's a fidelity/memory play (fp16 acts, no INT8
  matmul), not a speed one. **W4A8** (0.822 agree, gsm8k 92.7%) only when memory-bound (9.3 GB).

## What works (single B70, 32 GB)
| Thing | Result |
|---|---|
| **Qwen3.6-27B (Gated-DeltaNet)** | **RUNS** via int4 AutoRound on vLLM **0.23.0** — 7.9 t/s, coherent. Only known single-card path. |
| **Qwen3.6-35B-A3B (256-expert MoE)** | **RUNS** via int4 AutoRound (W4A16) on **0.23.0** + a 16-line INC-XPU MoE routing patch — loads 19.6 GiB, coherent, ~6 t/s eager. `contrib/vllm_moe_xpu/`. |
| Qwen3-14B **FP8** (vLLM-XPU) | **35 t/s** single / **556 t/s** @ C64 (raise `--max-num-seqs`!), near-lossless |
| Qwen3-14B **F16/BF16** | 18.7 t/s, but ~28 GB barely fits (tiny KV). FP8 is ~1.9x faster — just use FP8 |
| Qwen2.5-7B **Q4_K_M** (llama.cpp SYCL) | ~90 t/s decode — llama.cpp SYCL is great for standard-attention models |
| **torch.compile (inductor)** | cuts single-stream **TTFT ~6x** (1032->176 ms) + ~11% decode at low concurrency |
| **Qwen3-14B W8A8 + PIECEWISE XPU graph** | **27.23 t/s** decode (+16.7% over eager 23.33) — image `:int8g`, `cudagraph_mode=PIECEWISE` |
| **Qwen3-14B W4A8 + PIECEWISE XPU graph** | **48.18 t/s** decode (**+187% / 2.87x** over eager 16.79) — `:int8g` + `30_serve_w4a8_graph.sh GRAPH=1`. Best prefill/TTFT (int8-XMX). |
| **Qwen3-14B W4A16 + PIECEWISE XPU graph** | **54.57 t/s** decode (**+95% / 1.95x** over eager 28.04) — fastest 14B decode; same 9.3 GiB; no act-quant tax. `GRAPH=1`. |
| **Qwen3.6-27B int4 + PIECEWISE XPU graph** | **30.84 t/s** decode (**+293% / 3.93x** over eager 7.84) — flagship; ~89% of BW ceiling; erases the density tax. Image `:v0230` (needs GDN). |
| **Qwen3.6-35B-A3B MoE int4 + PIECEWISE XPU graph** | **56.84 t/s** decode (**+617% / 7.17x** over eager 7.93) — **fastest config on the board**; A3B = ~3B active/tok; `:v0230moe`. |

## Single-card concurrency (captured) — Qwen3.6-27B int4 (w4a16 AutoRound) [2026-06-21]
Served `:v0230` GRAPH=1 PIECEWISE (capture sizes 1..64), NOMM=1, UTIL=0.92, MAXSEQS=64, **fp16 KV**;
`vllm bench serve` random 512/128 `--ignore-eos`. C1 bench (30.9) == perf_probe 30.84 -> capture validated.
Repro: `scripts/56_27b_conc_campaign.sh`. CSVs: `results/sweep_27b-w4a16-cap-*.csv`.

| C | aggregate out t/s | per-stream decode t/s | mean TTFT |
|--:|--:|--:|--:|
| 1  | 28.1  | 30.9 | 0.44 s |
| 2  | 52.0  | 29.3 | 0.60 s |
| 4  | 87.8  | 26.7 | 1.07 s |
| 8  | 134.3 | 21.7 | 1.77 s |
| 16 | 178.3 | 14.5 | 2.68 s |
| 32 | 216.7 | 8.4  | 3.67 s |
| 64 | **234.7** | 6.7  | 15.1 s |

- **Aggregate max ~235 t/s @ C64**, but TTFT balloons to 15 s and per-stream collapses to 6.7 -> the
  practical knee is **~217 t/s @ C32** (TTFT 3.7 s); for low latency stay at **C2-C4** (per-stream ~27-29,
  near single-stream). GDN/linear-attn batches sublinearly (C1->C8 = 4.8x for 8x conc; C8->C32 only +1.6x).
- **Context window is throughput-neutral.** Re-running the whole sweep at **128k ctx** (fp16 KV) gave
  near-identical numbers (C2 51.3, C4 87.6, C32 215.6, C64 232.9) — the KV pool is sized by gpu-util, not
  max-model-len; only a single very-long sequence would differ.
- **256k ctx does NOT fit at fp16 KV on one card:** a 262144-tok seq needs 16.2 GiB KV vs 8.31 GiB available
  (model 16.69 GiB @ UTIL 0.92) -> vLLM caps "estimated max model length 133120". **Max fp16-KV context ~133k;**
  use `MAXLEN<=131072`, or `KVDTYPE=fp8_e5m2` to roughly double it. (Campaign auto-fell-back 256k -> 128k.)

## Multi-GPU (dual B70, TP=2) [2026-06-21 -- 2nd card just installed]
Second B70 landed. Both cards are **compute-usable**, not just PCI-visible: inside `vllm-xpu-env:v0230`,
`sycl-ls` enumerates `[level_zero:0]` + `[level_zero:1]` (both [0xe223]) and two OpenCL GPUs. TP=2 serves:
0.6B sharded 0.57 GiB/card (half the single-card weight = real split), KV pool ~26.8 GiB, max concurrency
122x (2x single-card), coherent output. `system_fingerprint: vllm-0.23.0-tp2`. Serve via `30_serve_w4a8_graph.sh
TP=2` (captured) or `43_serve_multi.sh TP=2` (eager).

**[KEY WIN] For models that fit one card, DATA-PARALLEL (2 independent replicas) is the dual-GPU answer -- not
TP/PP.** Measured 2026-06-21 (`scripts/64_dataparallel_2rep.sh`): one captured 27B-int4 replica per card (card0
:18080, card1 :18081, zero inter-GPU traffic). Aggregate out tok/s = sum of both, vs a fresh single-card solo
baseline: **C1 56.3 (2.09x), C8 278.7 (2.11x), C32 456.9 (2.14x), C64 524.8 (2.03x)** -- ~linear, NO contention
(each replica under concurrent load ran at full solo speed; host CPU/PCIe is not a bottleneck). Single-stream
decode stays full single-card (~30.8 t/s/replica). So DP beats both model-parallel options on BOTH axes:
throughput (~2x vs TP/PP's sublinear) AND latency (30.8 vs TP=2 4.18 / PP=2 6.11). DP's only limit: it cannot
serve a model bigger than one card -- that's the ONLY case for PP=2 (or TP=2). Serve two replicas via the new
`PORT`/`DEVICE` knobs on `30_serve_w4a8_graph.sh` (`DEVICE=0 PORT=18080` / `DEVICE=1 PORT=18081`) behind a
round-robin proxy. Stack MTP per replica for the decode multiplier.

**The interconnect is the bottleneck FOR MODEL-PARALLEL (TP/PP). There is NO usable GPU P2P on B70** -- every TP all-reduce round-trips
`GPU -> host RAM -> GPU over PCIe` (Unified Shared Memory; set `CCL_TOPO_P2P_ACCESS=0`). No XeLink/NVLink.
PCIe topology: each card behind its own ON-CARD Intel switch (`08/42:00.0` upstream -> GPU `0a/44:00.0`);
switch-to-CPU trained at **Gen3 x16 -- the real, healthy link (1950X platform max).** The "Gen1 x1" that lspci
shows at the GPU endpoint is the documented Intel Arc reporting ARTIFACT (KB 000094587), NOT the real link --
read the upstream bridge. Both cards `numa_node=-1` (single-NUMA 1950X) so host-staging does not cross a NUMA hop.
So the comms cap is host-staging + no-P2P, **not** the PCIe gen.

**Stability (vLLM #41663 is our exact hardware):** dual-B70 TP=2 GP-faults + Xe BCS engine resets at
ProcessGroupXCCL init unless **`CCL_ENABLE_SYCL_KERNELS=0`** (the load-bearing fix; we set it). Full stable env:
`CCL_ENABLE_SYCL_KERNELS=0 CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 SYCL_UR_USE_LEVEL_ZERO_V2=0
CCL_ATL_TRANSPORT=ofi CCL_ZE_IPC_EXCHANGE=pidfd VLLM_WORKER_MULTIPROC_METHOD=spawn` + `--distributed-executor-backend mp`.
NEVER `CCL_ALLREDUCE=ring` (collapses to ~0.5 tok/s).

**Expectation (from community data -- TP on Arc is a CAPACITY play, not a SPEED play):** TP=2 *hurts* single-stream
decode (comms-bound: StorageReview B60 GPT-OSS-20B batch=1 went TP=1 49 -> TP=8 23 tok/s) and helps only
aggregate throughput at concurrency / models too big for one card. [CORRECTED 2026-06-21: the previously-cited
"dual-B70 TP=2 Qwen3.5-27B FP8 13.25/97.84" and "30B-A3B MoE FP8 912 t/s" are UNVERIFIED -- not in any public
source. The "13.25/97.84" is actually Puget's 4xB70 TP=4 27B-DENSE FP16 (13.1/95.9); no source has "912".]
**Verified public multi-Arc numbers:** Puget 4xB70 TP=4 FP16 35B-A3B MoE = 16.3 (C1) / 63.7 (C4) / 122 (C8);
Puget 4xB70 TP=4 27B-dense = 13.1 (1u) / 95.9 (8u); StorageReview B60 GPT-OSS-20B TP=1 49 -> TP=8 23 single-stream
(TP hurts single-stream, confirmed). No public clean TP=1-vs-TP=2 same-model decode comparison exists -> our
own `scripts/58_tp2_campaign.sh` + `scripts/64_dataparallel_2rep.sh` generate the real numbers.

**Our measured numbers (2026-06-21, dual B70):**
- **[SUPERSEDED 2026-06-24] "Captured PIECEWISE TP=2 is BLOCKED"** -- the original claim below was wrong/incomplete.
  Captured PIECEWISE TP=2 WORKS (W8A8 27B no-MTP captured = 18.10 t/s COHERENT; MTP spec5 captured = 26.10 t/s
  COHERENT, 26% accept). See the "[BUG B] captured TP=2 garbage" entry just below for the full root cause + fix.
  Original (kept for history): *oneCCL errors `sched algorithms do not support sycl_graph recording` -- stable init
  needs `CCL_ENABLE_SYCL_KERNELS=0` but graph-recording an all-reduce needs `=1`. Eager-only at TP=2.* (The real
  story: all_reduce/reduce_scatter DO record under `=1`; only oneCCL ALLGATHER can't, and that is now worked around
  by an all-reduce-of-padded all_gather shim so even all_gather records -- nothing needs ejecting.)
- **[BUG B -- captured TP=2 W8A8 garbage: ROOT CAUSE + FIX, 2026-06-24] (full log: JOURNAL 2026-06-23/24,
  scripts/106-111).** Symptom: 27B W8A8 (16 full-attn int8 + 48 BF16 GDN) at TP=2 with PIECEWISE capture emitted
  garbage (`!!!!`); eager was fine; single-card 14B/W4A8 captured fine. It was NOT a captured-int8-numerics bug.
  - ROOT CAUSE: vLLM's piecewise `CUDAGraphWrapper` does a pure `replay()` with NO input copy -- every captured
    piece's inputs MUST sit at the SAME device address on replay as at capture (it even asserts this in DEBUG). The
    XPU TP collectives are all OUT-OF-PLACE (`all_reduce` returns `input_.clone()`; all_gather/reduce_scatter return
    fresh `torch.empty()+.contiguous()`). Listing a collective in `splitting_ops` EJECTS it to eager at a piecewise
    boundary; its fresh output does not reproduce the capture-time address on XPU -> the next captured piece reads
    STALE data -> garbage. Eager has no graph (no contract); single-card has no collectives.
  - FIX: eject NOTHING. Keep all collectives CAPTURED (`splitting_ops` = attention/GDN ops only), use `IGP=false`
    (the inductor partitioner KeyErrors on the mixed W8A8+BF16-GDN region). all_reduce/reduce_scatter record fine.
    For MTP, the spec-verify `all_gather` (which oneCCL 2021.17 cannot graph-record) is replaced by an
    ALL-REDUCE-OF-PADDED all_gather (`rdy_to_serve/.../patches/sitecustomize.py`, = `scripts/110_csag_shim`) so it
    too records and stays captured. SECOND FACET: merely ejecting only all_gather left the body coherent but the
    captured spec-VERIFY gave ~0% accept (drafts rejected, MTP became pure overhead 9.63 t/s); capturing all_gather
    correctly (plan B) restored accept to 26% -> 26.10 t/s.
  - NUMBERS (coherence-gated, hard prompt, temp=0): eager-no-MTP ~4.1; captured-no-MTP 18.10; eager-MTP 10.43 @36%;
    captured-MTP-eject-all_gather 9.63 @~0%; **captured-MTP plan-B 26.10 t/s @26% accept** = 1.44x vs captured-no-MTP,
    2.5x vs eager-MTP, 6.4x vs eager-no-MTP. The old "63 t/s/3.4x" headline was benched on degenerate garbage.
- **Eager TP=2 27B int4 decode: C1 4.18 t/s, C2 4.02 t/s/stream** -- vs **eager TP=1 single-card 7.84 -> 0.53x**.
  TP=2 HALVES single-stream decode (collective-latency tax). Weights shard 8.42 GiB/card (half of 16.7 single).
- **Cross-card all-reduce microbench** (`scripts/60_allreduce_bench.sh`, xccl): **latency floor ~0.29 ms** (small
  msgs), **bandwidth ceiling ~0.70 GB/s busbw** (>=2 MB). That's ~17-70x below a healthy PCIe link (Gen3 x16
  ~12 GB/s, Gen5 x16 ~50 GB/s), ~850x below NVLink. **The multi-GPU bottleneck is NO GPU P2P -> host-staged
  all-reduce + oneCCL overhead (NOT the link, which is healthy Gen3 x16 -- see CORRECTED note just below).**
- **[CORRECTED 2026-06-21] The "Gen1 x1 link" was a MISDIAGNOSIS -- it is an Intel Arc lspci reporting artifact.**
  The real link is **Gen3 x16** (read at the on-card switch UPSTREAM bridge `08/42:00.0`; the GPU endpoint always
  fakes 2.5 GT/s x1 per Intel KB 000094587). The earlier "200/200 samples x1 under load" polled the artifact node;
  the "~220 MB/s weight-load" is a SATA-SSD/loader bottleneck (models on `/dev/sdd1`), not PCIe. Reseating/moving
  the cards to other slots gains NOTHING -- they are already at full Gen3 x16 (the 1950X is a Gen3 host). The
  ~0.7 GB/s all-reduce is REAL but caused by host-staged collectives (no GPU P2P) + oneCCL overhead, not the PCIe
  gen. **TP=2 stays a CAPACITY tool** (bigger models/KV), not a single-stream speed tool -- but because of no-P2P,
  not a fixable link. (Confirm with a clean H2D bw test: expect ~10-12 GB/s, not ~0.25 GB/s.)
- **[NOVEL] AutoRound quantization runs across BOTH XPUs.** `device_map="0,1"` -> `xpu:0`+`xpu:1`; a full 0.6B
  quant ran in 22 s with `peak_vram {'0':0.62GB,'1':0.51GB}` (both cards active), 196/197 layers quantized.
  No public multi-XPU AutoRound precedent existed -- this rig does it. (`scripts/59_autoround_2xpu.sh`.)
- **[KEY] Use PP=2, NOT TP=2, for dual-card serving on this rig.** Pipeline-parallel does ONE hidden-state
  handoff/token (vs TP's ~128 all-reduces), so it dodges the host-staged all-reduce tax. Eager 27B int4 single-stream:
  **PP=2 6.11 t/s (0.78x single-card) vs TP=2 4.18 (0.53x)** = PP +46%. PP=2 also gives a far bigger KV pool
  (19.44 GiB/stage vs TP's tight split). TP only wins if a single layer can't fit one card (not our case).
  Re-evaluate TP only if GPU P2P becomes achievable (the PCIe link is already full Gen3 x16 -- nothing to fix there).
  (`scripts/62_pp2_27b.sh`.)

## What does NOT work (yet) — save yourself the time
- **INT8 W8A8:** stock vLLM has no XPU kernel -> hard-crashes at load (`KeyError: PlatformEnum.XPU`).
  **We FIXED it (2026-06-18):** wrote a native `int8_gemm_w8a8` oneDNN kernel (s8xs8->s32) + the vLLM
  `XPUInt8ScaledMMLinearKernel` + registry entry. The W8A8 checkpoint now SERVES on the B70 and selects our
  kernel (`Selected XPUInt8ScaledMMLinearKernel`), coherent output. First INT8 W8A8 path on Battlemage. See
  docs/kernel/01 + contrib/vllm_int8_xpu/. (Perf vs FP8 in prefill/large-batch = TODO; FP8 still wins decode.)
- **W4A8-INT** (`XPUW4A8IntLinearKernel`, int4 weights + per-token int8 activations, oneDNN `int4_gemm_w4a8`):
  **[SUPERSEDED 2026-06-20 -> now in "What works"]** the "decode ~16.6 t/s, half of FP8, FP8 stays the pick"
  conclusion below was measured EAGER. With PIECEWISE graph capture W4A8 decode is **48.18 t/s** -- it now
  BEATS fp8 single-stream and is the fastest decode config measured (see the breakthrough bullet in the TL;DR).
  Original eager notes kept for the record: **Tested 2026-06-18 — it's the only path that lights the INT8-XMX
  datapath, and it serves (9.3 GB, smallest footprint, 12x concurrency). BUT single-stream decode is ~16.6 t/s,
  half of FP8** (the int4_gemm_w4a8 decode kernel is unoptimized). Head-to-head (identical harness): **FP8 wins
  per-stream at every matched concurrency (~1.7-1.8x)**; W4A8's only real edge is VRAM (9.3 vs 15.2 GB). FP8
  stays the pick. (Note from literature/06: FP8 is conversion-based on Xe2; INT is native systolic — so a real
  INT8 W8A8 kernel could beat FP8 in *prefill/large-batch*. That kernel doesn't exist upstream yet = our top
  contribution target.) [The eager-vs-capture gap was the real story: w4a8 was the most dispatch-bound config.]
- **Dense W8A8 TRUE int8 on v0230 (2026-06-23): WORKS but is a PERF DEAD-END; keep `:int8`.** We made a dense
  compressed-tensors W8A8 (14B) run true-int8 on `vllm-xpu-env:v0230` -- triton_scaled_mm (tl.dot int8->int32 -> Intel
  DPAS/XMX), int8 weights stay int8 (15.3 GiB), via a sitecustomize hook that registers the kernel for the
  `CompressedTensorsW8A8Int8` path + wraps it as an OPAQUE torch custom op so PIECEWISE graph capture succeeds
  (contrib/vllm_int8_xpu/v0230/dense_test). Coherent. BUT decode = **~1.7 t/s vs `:int8` oneDNN ~23.5 (~13x slower)**:
  the act-quant is un-fused plain torch on every one of ~280 dense linears (XPU has no `_C.scaled_int8_quant`) and the
  opaque op is capturable-but-not-fusable. **`:int8` (our oneDNN s8s8s32 + fused per-token quant) stays the W8A8 perf
  path; v0230-int8 is a correctness/portability fallback only.** Closing the gap needs a fused XPU int8 act-quant op +
  an inductor-fusable GEMM (Track 1).
- **Speculative decoding -- method/format-dependent (UPDATED 2026-06-23):** ngram (prompt-lookup) on 14B W8A8 is
  net-NEGATIVE (eager 23.33->21.51 -7.8%; PIECEWISE 27.23->25.28 -7%; ~16% accept -- low acceptance on diverse text).
  **BUT native MTP on the qwen3_5 27B is strongly POSITIVE on PIECEWISE: W4A16 + MTP spec=4 = 55.28 t/s vs 30.84
  MTP-off = 1.79x (+79%), accept_len 3.25, Half-KV FREE.** The warmup-spoof PIECEWISE fix (910182c) captures the
  spec-decode decode batch (1+spec), so the verify is no longer eager-attention-bound -- this REFUTES the old
  "net-negative even with capture / -19%" (a stale pre-fix measurement). FULL capture would lift it further but is
  **CONFIRMED KERNEL-GATED (2026-06-22):** porting vllm-ascend #7148's dispatcher fix (scripts/88) let capture proceed
  past the Python dispatcher, then it crashed in the BAKED kernel -- `_xpu_C.gdn_attention -> spec_query_start_loc must
  have size [num_spec_decodes + 1]`. So FULL needs an Intel vllm_xpu_kernels fix, not a vLLM-Python fix; TRITON_ATTN
  doesn't dodge it (GDN decode core always uses the baked op). **PIECEWISE 1.79x is the single-card ceiling on stock
  v0230** (issue draft docs/kernel/21). TP=2 MTP is DEAD (spec-allgather not graph-capturable); MoE MTP is FLAT (+3%,
  sparse 3B-active). **MTP is a DENSE-model lever.** Full campaign: MTP_TODO.md (M0-M5).
  **Ctx=2048 follow-ups (2026-06-23, `vllm bench serve`, random 2048/128, card1, golden
  `rdy_to_serve/qwen36-27b-int4`):** MTP is a C1 latency/decode lever, not a C4 throughput lever on this workload.
  `tg` = 1000/mean TPOT; `pp` ~= input_len * concurrency / TTFT. Full CSVs:
  `results/mtp_table_qwen36-27b-int4_ctx2048_20260623.csv` and
  `results/mtp_spec_sweep_qwen36-27b-int4_ctx2048_20260623.csv`.

  | config | C | pp tok/s | TTFT | tg tok/s | agg out tok/s | accept_len |
  |---|---:|---:|---:|---:|---:|---:|
  | no-MTP PIECEWISE | 1 | 1605.8 | 1.275 s | 29.78 | 23.10 | - |
  | MTP spec=4 PIECEWISE | 1 | 1453.0 | 1.410 s | **46.69** | **30.99** | 2.92 |
  | no-MTP PIECEWISE | 4 | **2410.9** | **3.398 s** | **19.54** | **51.69** | - |
  | MTP spec=4 PIECEWISE | 4 | 1843.5 | 4.444 s | 16.09 | 40.56 | 2.41 |

  Single-stream spec sweep, C1 only, ctx=2048:

  | config | KV | pp tok/s | TTFT | tg tok/s | agg out tok/s | accept_len |
  |---|---|---:|---:|---:|---:|---:|
  | no-MTP PIECEWISE | fp16 | **1608.2** | **1.273 s** | 30.48 | 23.53 | - |
  | MTP spec=3 PIECEWISE | fp16 | 1499.0 | 1.366 s | 57.24 | **35.70** | 3.15 |
  | MTP spec=4 PIECEWISE | fp16 | 1472.4 | 1.391 s | **57.64** | 35.60 | 3.53 |
  | MTP spec=5 PIECEWISE | fp16 | 1392.4 | 1.471 s | **57.64** | 34.83 | 3.86 |
  | MTP spec=3 PIECEWISE | fp8_e4m3 | 1483.6 | 1.380 s | 52.99 | 33.89 | 3.12 |
  | MTP spec=4 PIECEWISE | fp8_e4m3 | 1470.3 | 1.393 s | 55.74 | 34.87 | 3.60 |
  | MTP spec=5 PIECEWISE | fp8_e4m3 | 1456.6 | 1.406 s | 55.31 | 34.57 | 3.88 |

  Verdict: for C1 ctx=2048, spec=4 fp16 KV is the best pure `tg` row, while spec=3 fp16 KV is the best
  user-visible aggregate row and has lower TTFT. Half-KV is capacity/context headroom here, not a speed win:
  it loses 1.9-4.3 tok/s of `tg` at ctx=2048. Keep `DD_MTP=1` for single interactive coding-agent streams; leave it OFF for C4+ batch/fan-out at
  ctx=2048 unless a workload-specific bench proves otherwise. The older 55.28 t/s number remains the TTFT-cancelled
  single-stream decode probe, not a concurrent `vllm bench serve` row.
- **Qwen3.6 (Gated-DeltaNet) FP8/8-bit:** does NOT fit a single card (28.5 GiB, no KV room) and the FP8
  DeltaNet kernel only exists in vLLM **0.23.0** (older images: ESIMD `.weight` bug / `scaled_mm` `KeyError(XPU)`).
  BUT int4 (AutoRound) on 0.23.0 **does run** (see above) — just slow. 8-bit Qwen3.6 needs a 2nd card.
- **torch.compile on vLLM 0.23.0:** broke (torch 2.11 + inductor). Run `--enforce-eager` on 0.23.0
  (its eager TTFT is already ~7x better than 0.20.2 eager anyway).
- **Gemma 4 12B:** the `gemma4_unified` multimodal checkpoint routes through vLLM's generic Transformers
  fallback, which mis-reshapes its mixed head dims (256/512) and crashes. Needs a native-resolving checkpoint.

## Practical gotchas
- **Unraid `docker.img` is 50 GB by default** — multi-GB vLLM images overflow it. Grow it (we went to 200 GB
  on the NVMe cache, non-destructively) or relocate Docker storage.
- **Keep model weights + all caches on a fast SSD via bind-mounts**, never baked into image layers.
- **Build vLLM from source** at a known-good commit (we used `c51df4300` = 0.20.2rc1.dev2) — pre-built XPU
  images lag, and the Intel `llm-scaler` image wraps an *older* upstream for dense models.
- It's a **memory-bandwidth game** at batch 1: decode t/s ≈ 608 GB/s ÷ model-bytes. Smaller quant = faster
  decode; FP8 (~1 byte/wt) ≈ 2x BF16. Compute (XMX TOPS) only matters for prefill + big batches.

## Hardware
Intel Arc Pro B70: Xe2/Battlemage, 32 GB GDDR6, 608 GB/s, 367 INT8 TOPS, PCIe 5.0 x16 (card spec).
`xe` kernel driver. Pass into containers with `--device /dev/dri`.
- **[!] The "Gen1 x1" reading is an Intel Arc lspci ARTIFACT, NOT a real link (corrected 2026-06-21).** Arc/
  Battlemage cards put an on-card PCIe switch between the slot and the GPU die; the GPU-adjacent nodes (GPU
  endpoint `0a/44:00.0` + switch downstream `09/43:01.0`) ALWAYS falsely report 2.5 GT/s x1 by design (Intel
  KB 000094587). Read the REAL link at the card switch's UPSTREAM bridge (`08/42:00.0`): it shows **8 GT/s x16
  = Gen3 x16**, the platform max (1950X is a Gen3 host; card silicon is Gen5-capable, LnkCap 32 GT/s x16). Both
  cards confirmed D0/active under live load still read x1 at the endpoint = artifact, not power state. So the
  PCIe link is FINE; there is nothing to fix by reseating/moving slots.
- **The real multi-GPU bottleneck is NO GPU P2P** -> TP all-reduce round-trips GPU->host RAM->GPU (host-staged
  oneCCL/xccl, ~0.7 GB/s measured). That is a software/architecture limit (no P2P + collective overhead), NOT the
  PCIe gen -- a clean H2D copy should show ~10-12 GB/s. Two cards sit on SEPARATE CPU root complexes (`00:03.1`
  die0, `40:03.1` die1) of the 2-die 1950X; single NUMA node (UMA mode), but cross-die for any P2P attempt.

*Next up: prefer PP=2 for dual-card serving (no-P2P makes TP comms-bound; link is already full Gen3 x16, nothing
to fix); 27B W8A8 INT8 at TP=2/PP=2 (Phase C headline, needs the custom int8 kernel in a GDN-enabled image);
real 2-card AutoRound of a >32GB source (now proven viable); int8 MoE kernel for 35B-A3B (docs/kernel/18).*

================================================================================
INT8 FAST-PATH QUANT CAMPAIGN -- BOTTOM LINE (2026-06-21/22)
================================================================================
Goal: quantize the qwen3.6 family to W8A8 (int8 wt + int8 act) and W4A8 (int4 wt + int8 act) and measure how far
the B70's int8-XMX systolic path carries vs the int4-weight/fp16-act (W4A16) baselines. Full log: JOURNAL.md;
queue+results: QUANTS_TODO.md sec 6; method bible: docs/kernel/15; microbench: docs/kernel/19.

THE HEADLINE (answers "where can int8-act beat w4a16"):
- **W4A8 (int4 wt + int8 act) is the all-rounder and BEATS W4A16.** 14B ctx-2048 ladder (our int8 kernel, captured):
    scheme           dec t/s (c1/c8)   TTFT(c1)   c8 aggregate
    W4A16-gptq       52.5 / 22.2       571 ms     132.9 t/s
    W4A8-gptq        49.3 / 25.5       405 ms     161.7 t/s   <- -29% TTFT, +22% c8 throughput vs W4A16
    W8A8 (AR/gptq)   25.1 / 18.0       347 ms     125.0 t/s
  W4A8 ties W4A16 decode (same int4-weight BW) AND wins prefill (int8-XMX) -> lower TTFT + higher concurrent tput.
- **W8A8 is a prefill-latency / accuracy play, NOT a decode play.** Its decode is ~half the int4 schemes because
  decode is weight-bandwidth-bound and int8 weights are 2x the bytes of int4. Best single-stream TTFT though.
- **Decode ordering (bytes-bound):  W4A16 ~= W4A8  >  W8A8  >  bf16.   Prefill (compute-bound): int8-XMX ~1.6-2x bf16.**

MICROBENCH (docs/kernel/19, real int8_gemm_w8a8 op, 341 rows): int8-vs-bf16 GEMM median 1.68x (peak 250.9 INT8
TFLOP/s, grows with M); GEMV ~2x on large-N dense shapes (up to 433 GB/s) but only ~1.1x on small-N (35B MoE
experts, KV-proj) which are overhead-bound -> reinforces W4A16-int4 for the 35B MoE decode.

MTP RECEPTIVITY (docs/literature/09): with a BF16 draft head, body W8A8 vs W4A16 cause only SECOND-ORDER spec-
decode acceptance differences (arXiv:2505.22179). So W8A8 is NOT a meaningful MTP unlock over W4A16 on our stack;
our -19% MTP result is a graph-capture problem, not an acceptance problem. Ordering: BF16 >= W8A8 >= W4A16 >= W4A8.

CHECKPOINTS PRODUCED: 14B W8A8-autoround; 27B W8A8-sqgptq + W4A8-sqgptq(+prepacked); Qwable W4A8 (in progress).
(W8A8-on-VLM uses GPTQ-selective-SQ: AutoRound's data-driven calib auto-routes VLMs to an MLLM path that needs an
image processor -- breaks for text-only. Same int8 kernel/perf as AutoRound per the 14B.)

KNOWN SERVE LIMITS (B70, current vLLM-XPU build): single-card 27B int8 serve is fragile -- W8A8 27B (35GB) exceeds
one card; W4A8 27B serve hit a chain of 5 fixable-but-stacked bugs (fp8-KV reject, 4304 vision-fc2 odd-dim ->
graft-ignore fix, 28GB unpacked-int8 load transient -> prepack fix, prepack merged-column shape mismatch). 27B int8
perf inferred from the 14B ladder + the existing 27B-W4A8-q-prepacked (20.9 t/s captured). 35B int8 MoE: produce-only
(no XPU int8 fused-expert kernel; docs/kernel/18).

TOP NEXT LEVERS: (1) col-reorder + dp4a int8 GEMV (docs/08/19) to lift the small-N decode GEMVs toward BW
(W8A8 decode ~26->35-40 t/s); (2) align prepack tensor layout w/ vLLM merged-column loader to unblock 27B W4A8 serve;
(3) int8 MoE fused-expert kernel for the 35B (docs/kernel/18 Track A) -- prefill/throughput win only.

================================================================================
INT8 CAMPAIGN -- UPDATE 2 (2026-06-22): serving cracked, TP=2, P2P verdict, n-gram
================================================================================
Extends the bottom-line above. The "serve limits" caveat is now largely RESOLVED -- the 27B int8 models SERVE.

SERVED LADDERS (27B, ctx2048, our int8 kernel) -- the real numbers, not inferred:
  scheme/TP            c1 dec   c1 TTFT   c8 dec   c8 agg
  W4A8 TP=1 (1 card)   20.7     876ms     12.2     67.8     <- best for a fit-one-card model
  W4A8 TP=2 (graph)    22.1     2858ms    6.3      34.3     <- +6.5% c1 dec, worse TTFT/conc (allreduce tax)
  W8A8 TP=2 (graph)    17.5     2728ms    6.1      34.0     <- 35GB, TP=2-only; now servable
  + n-gram spec (c1)   37.8 t/s decode (~1.8x the 20.7, workload-dependent; concurrency path WIP)
The full 27B-W4A8 serve took fixing 6 stacked XPU-serve bugs (JOURNAL); the keystone was ignore-list 339->4-regex
(explicit linear_attn names missed the DeltaNet FUSED projections -> packed-int4-vs-bf16 shape assert).

TP=2 / multi-card (docs/P2P_GPU.md):
- [unlock] **CCL_ENABLE_SYCL_KERNELS=1** makes the oneCCL allreduce GRAPH-CAPTURABLE -> TP=2 + PIECEWISE graph works
  with NO vLLM source patch (Seguin needed a code patch). This is what made the 35GB W8A8 servable. Novel B70 result.
- TP=2 LOSES for fit-one-card models on our Gen3 cross-die box (allreduce tax: 3.3x TTFT, 2x worse concurrency, vs a
  +6.5% c1-decode edge). TP=2's job is fitting >32GB models, not speed.
- [P2P verdict] B70<->B70 P2P is UNAVAILABLE on kernel 6.18: a raw 12-variation Level-Zero ctypes probe
  (zeDeviceCanAccessPeer) returns False for EVERY env (debug keys, FLAT/COMPOSITE hierarchy, IPC drmfd/pidfd, ...);
  vLLM P2PACCESS=1 fails worker-init. NO userspace spoof. Gated on a host reboot: iommu=off pre-test, then kernel 7.0+
  (drm/xe P2P patch). Plan: P2P_GPU sec I. Host-staged (P2P off) is the only working TP=2 path today.

OPTIMIZATION LEVERS proven (no reboot):
- n-gram speculative decode: TESTED, NOT a general win (logged negative). c1=37.8 t/s at OUT=128/k=3 was a short-output
  artifact; at OUT=256/k=2 it is 17.9 < no-spec 20.7, and c>=2 fails (spec+concurrency broken on XPU). Niche-only
  (highly-repetitive output). See JOURNAL 2026-06-22 correction.
- (orthogonal, not yet done) Seguin's clone-safe + oproj-delay allreduce fusion patches to cut the host-staged
  allreduce tax further.

35B int8 MoE (Q6/Q7): [SOLVED 2026-06-22] **Quark W8A8 INT8 35B-A3B SERVES + GENERATES on 2x B70 TP=2.** Not on the
llm-scaler 0.14.1 image (no XPU MoE op suite -- `vllm._moe_C` unbuilt), but on **`vllm-xpu-env:v0230` (vLLM 0.23.0)**,
which already ships `QuarkW8A8Int8MoEMethod` + the dynamic-per-token int8 linear dispatch AND routes the 256 int8 experts
through the **Triton `fused_moe_kernel`** on XPU (same path as our int4 MoE). Only patch: one bind-mounted `quark.py`
rerouting the int8 LINEAR layers to a weight-only int8->bf16 dequant GEMM (XPU has no int8 scaled-mm kernel; experts stay
true int8). `scripts/76_quark35b_v0230.sh` (TP=2, #41663 env, enforce-eager). Verified: load 17.54 GiB/card, KV 10.2 GiB,
conc 89x@8192, backend=xccl, gen "...Paris, a city renowned for its rich history...". **EAGER perf** (random 2048/128
sweep): c1 agg 4.46 t/s (per-stream 4.80, TTFT 2233 ms), c2 agg 8.16 (per-stream 4.46), c4 agg 14.08 (per-stream 4.17,
TTFT 5878 ms) -- aggregate scales ~3.15x at c4, per-stream holds ~4.2-4.8.
**GRAPH CAPTURE perf** [2026-06-22] (PIECEWISE, SYCLKERNELS=1, CAPSIZES=1,2,4,8; rdy_to_serve serve.sh GRAPH=1):
c1 = **41.02 t/s per-stream decode** (TPOT 208->24.4 ms, e2e out 27.85 t/s, TTFT 2233->1499 ms) = **8.5x the eager
decode** -- same class of win as the int4 MoE's +617%. BUT c>=2 currently BREAKS: a new prefill batch shape at higher
concurrency triggers a mid-serve torch.compile and the engine hangs on shm_broadcast (c2/c4 rows timed out to 0/NaN).
The one-time captured-startup compile is ~6 min cold (267 s decode + 105 s prefill range), fast on a warm
/vllm_cache/torch_compile_cache. **[2026-06-22] FULL_DECODE_ONLY is BLOCKED on stock v0230**: it uses the SYCL Graph
extension for the decode capture and dies at KV-cache init with `RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory
feature is not yet available for use with the SYCL Graph extension` (confirms SERVING.md "FULL blocked by SYCL-Graph
scratch"). The community 102 t/s run used FULL_DECODE_ONLY via a PATCHED/custom image -> that is the route (build our own
image). **[FIXED 2026-06-22 -- warmup spoof] the PIECEWISE c>1 break was just the cold one-time torch.compile per batch
shape that the bench did not wait for** (the server finishes the compile even after the client times out). A warmup
sweep before the measured one (serve.sh WARMUP=1, default with GRAPH=1) warms /vllm_cache/torch_compile_cache -> the
measured sweep works at ALL concurrencies on STOCK v0230, no patched image: PIECEWISE+warmup agg/per-stream-decode
t/s = c1 20.04/25.85, c2 33.02/21.27, c4 45.73/17.46 -> **3.2-4.5x aggregate, ~4-5x decode vs eager**, ~46 agg @ c4
(single-stream decode varies ~25-41 run-to-run). So FULL_DECODE_ONLY (patched image) is NOT needed for usable captured
concurrency -- PIECEWISE+warmup suffices.
**[2026-06-22] TRUE int8 linear kernel BUILT + works (task c).** `contrib/llm_scaler_quark_int8_moe/v0230/quark.py`
registers `XPUInt8TritonScaledMMLinearKernel` into `_POSSIBLE_INT8_KERNELS[XPU]` (subclass of TritonInt8: cutlass
weight-transpose + `triton_scaled_mm` whose `tl.dot` int8->int32 lowers to Intel **DPAS/XMX int8**); activations are
quantized per-token in plain torch (XPU lacks `_C.scaled_int8_quant`). `B70_INT8_LINEAR=triton` (default) -> stock
QuarkW8A8Int8 picks it ("Selected XPUInt8TritonScaledMMLinearKernel for QuarkW8A8Int8"); `=dequant` falls back.
Result: serves + **generates coherently** (true W8A8 on the linear layers, experts already int8), and **load drops
17.54 -> 16.88 GiB/card** (int8 weights stay int8, no bf16 blow-up). BUT EAGER perf is ~25% SLOWER than dequant
(c1 3.36 vs 4.46 out t/s): the per-forward torch act-quant overhead on the MINORITY linear layers outweighs the int8
GEMM win, while dequant's pre-dequantized bf16 F.linear is cheap. Captured int8-triton (act-quant fuses into the graph)
measured: captured int8-triton DECODE c1 = 6.37 t/s (TPOT 157 ms) -- capture does NOT amortize the per-forward act-quant,
so it stays ~eager and LOSES badly to dequant+capture (25.85). PREFILL (8192-in, eager, c1): int8-triton TTFT 9627 ms
vs dequant 9626 ms -- **IDENTICAL** (~851 prefill tok/s both). **VERDICT: c = correctness + memory win (16.88 vs 17.54
GiB/card), NOT a speed win.** Why no pp gain despite the int8 XMX/DPAS fastpath being a compute lever: the 256 MoE
experts (the bulk of prefill compute) ALREADY run int8 via the Triton fused_moe_kernel in BOTH configs; c only swaps the
MINORITY linear layers (linear_attn.*, shared_expert), so it can't move total prefill, and on decode the per-token
act-quant overhead makes it slower. **Recommendation: keep the dequant linear path as the default** (simpler, faster
decode, more accurate W8A16); B70_INT8_LINEAR=triton stays available for the memory win / true-W8A8 faithfulness. Open:
verify whether triton_scaled_mm tl.dot actually emits DPAS (the identical pp suggests the linear fraction is just too
small to tell); a real accuracy eval; bake an image if true-W8A8 is wanted standing.
**[2026-06-22] Tuned MoE config = IMPRACTICAL via benchmark_moe.py on XPU.** Two blockers: (1) `--tune` does
`ray.init()` -> `available_resources()["GPU"]` KeyError on XPU (bypassed with a sitecustomize `ray.init(num_gpus=1)`);
(2) the benchmark worker is CUDA-centric -- `device="cuda"` + CUDA-graph timing (`torch.cuda.CUDAGraph()`,
`torch.cuda.graph`) -> `AssertionError: Torch not compiled with CUDA enabled`. And it estimated ~1.5 h PER batch
size (~6 h for 1,2,4,8). Not worth it for an fp16-PROXY config (`--dtype auto` -> the no-dtype filename int8 reads).
Future option (TODO **RESEARCH_TODO.md Track 9**): patch device->xpu + replace the CUDA-graph timing with a
synchronize loop (Ray bypass already proven via patches/sitecustomize.py), or hand-write a config.
**Open (the real lever):** true-int8 linear kernel that hits the Intel XMX/DPAS int8 fastpath (drop the dequant;
Full chain: kernel/20 sec 9, SERVING.md (WORKING recipe), contrib/llm_scaler_quark_int8_moe, rdy_to_serve/.
(Supersedes the earlier "deferred / Steve serves at 99 on TP4" note.)

MTP: ~~not viable on B70~~ **VIABLE + POSITIVE as of 2026-06-22 (this note STALE)** -- the earlier "not viable / Seguin
spec-decode VERIFIER bug" was a stack gap on the old image. On `vllm-xpu-env:v0230` (vLLM 0.23.0, #43565 native) MTP on
the qwen3_5 27B is +79% (1.79x, PIECEWISE spec=4); see the "Speculative decoding" bullet above + MTP_TODO.md M0-M5. The
remaining bug (the spec-op can't run in a FULL captured graph) only blocks the FULL-capture upside, not PIECEWISE MTP.
