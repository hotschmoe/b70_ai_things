Responding model: claude-opus-5-5 (Claude Opus 5.5), 2026-11-20.
CPU-only design study. No GPU touch, no build, no source or config change. Only this file was written.

# Own Zig inference engine for 2x Arc Pro B70 + Flash-Next: rooflines, design, and where the juice is

## 0. Sources and tags

Tags: [R] read in a source file, doc or log; [M] measured number from an existing log, trace or doc;
[M-partial] from the Stage B partial unitrace tables (strata/flash-next/concurrency-0216/ledger/b3, unclosed trace,
counts may be low; per-window values divide totals by 286 C4 windows = HC launches / 97 per window, and are profiled,
not unprofiled); [I] inferred by arithmetic or reasoning (not measured). Every ceiling below is [I].

Read for this study:
- Model geometry: /mnt/cache/b70/refs/QuixiInfer/docs/model-facts.md [R]; tensor types and byte sizes of the served
  ISTA GSQ-RCO IQ3_XXS pack from analyst/fable_decode_window_walk.md section 2 and 5 (GGUF header parse) [R].
- Decode window walk (Fable), scheduler/prefill/restore walk (Opus): analyst/*.md [R].
- Old C1 census: strata/flash-next/night/agent-reports/trace_final_analysis.md (TFA) [M].
- Stage B ledgers: concurrency-0216/LEDGER.md, ledger/b1/operations.csv (C1, complete), ledger/b3 (cached C4,
  partial) [M / M-partial]; JOURNAL.md Stage A numbers [M].
- P2P/PCIe measurements on this host: docs/P2P_GPU.md sections J.0-J.8 [M]; docs/20260825_xe_nonreboot_recovery_and_pcie_topology.md [R].
- Reference refresh (Steve, Sergio, TensorFold numbers): docs/20261119_b70_reference_refresh.md [R];
  QuixiCore/QuixiInfer review: docs/20261119_quixicore_xpu_kernel_reference.md [R].
- TensorFold source at /mnt/cache/b70/refs/TensorFold (HEAD 9e2a9bd): docs/xpu/README.md, docs/xpu/NAVIGATION.md,
  zig/src/xpu/{ze,tp}.zig, docs/recipes/{qwen3.8-flash-next,flash-next-cuda}.md, docs/environment.md [R].
- Memory notes: b70-int8-xmx-roofline, int4-dpas-esimd-xe2, b70-capturable-cross-device-sync, xpugraph-stable-on-b70,
  zml-* , int4-s4s4-dpas-showcase-not-serve, flashnext-decode-review-20261016 [R].
- Licenses: Strata LICENSE = MIT (Niko1221 and contributors) [R]; TensorFold = Apache-2.0 [R]; QuixiCore-XPU and
  QuixiInfer = MIT [R] (some QuixiCore kernels are translated from Apache-2.0 vllm-xpu-kernels, see the 20261119 doc).
- Host: lscpu/numactl show ONE NUMA node (BIOS in UMA/interleave mode; numa_node = -1 for the GPUs) [M]; GPU0 on
  die-0 root complex 00:03.1, GPU1 on die-1 root complex 40:03.1, both Gen3 x16 [R P2P_GPU.md J.0].

## 1. Headline answers

1. Bandwidth roofline vs today. With the served bytes per window, an engine whose kernels run at 70% of 608 GB/s
   plus a ~2.2-5.2 ms (2.0 ms + 0.05 ms per row) per-window latency floor would reach about: C1 ~250 tok/s (d=3) to ~290 (d=5), C2 ~210/stream,
   C4 ~120/stream (d=1) to ~170/stream (d=3), i.e. 480-670 aggregate, C8 ~700-870 aggregate, C16 ~880-1100 aggregate
   (4K context; 200K costs 5-20%). Strata measures C1 134.6, C2 65.4/stream, C4 37.3/stream (141 aggregate). Left
   on the table: ~1.9x at C1, 2.2-3.3x at C2, 3.2-4.8x at C4, and C8/C16 are not reachable in Strata at all
   (8-row window cap) [I, section 2].
2. Why. At C1 the window is latency-bound (1,400-1,575 small launches, HC at 38% of roofline, shared expert 18%,
   IQ3_S 16%, a 1.39 ms KV-append anomaly, host gaps). At C4 the partial census shows card 0 busy ~44 ms per window
   while it must move only ~8.3 GB (31% of peak), and card 1 busy ~8.5 ms of ~48 ms (it spins 40 ms in ep_wait):
   per-row QSA path ~8.3 ms, HC 4.9 ms, GDN 4.1 ms, PLE host wait 3.0 ms, commit 2.8 + emit 3.0 ms (serial per slot)
   [M-partial + M].
3. Prefill. Flash-Next needs ~14.2 GFLOP per prompt token. At the whole-model XMX efficiency TensorFold already
   demonstrates on a B70 for the 27B (~105 TFLOPS bf16, ~139 TOPS W4A8 per card [M, TF README]), two cards give
   ~15-20K tok/s before MoE losses; a realistic MoE target is 6-12K tok/s. Strata measures 889/1173 (4K/32K) on
   0.2.16 and ~2200 on the old night build: 3-10x on the table, the single biggest relative gap [I, section 3].
4. 27B. A dense 4-bit 27B reads ~14.5 GB per weight pass versus ~5-6 GB for a Flash-Next window, and its dense
   attention reads the whole KV every step (32 KB per token per slot, int8): at 200K that is 6.4 GB per slot per
   window. It wins only at short context and C>=16 (DP=2 or TP=2, ~1.1-1.3K aggregate ceiling), loses at C1-C4 and
   at every long-context point, cannot hold 16 x 200K, and lost the langs/sysadmin A/B 33/43 vs 41/43. Do not move
   to the 27B for concurrency [I, section 4; A/B M].
5. Build it? Yes, if "most juice" is the goal: the measured gaps are structural to Strata's architecture (card-0-owns-
   everything, one serial host thread with ~12 blocking waits per window, SYCL graphs per row layout, dpct per-row
   kernels with an 8-row cap, Python control lock). But do it in a way that is not wasted: microbench harness and
   a kernel library first (both feed Strata too), then a Level Zero host prototype that replays ONE Strata decode
   window as the go/no-go gate. TensorFold already proves the exact recipe on B70 (Zig, Level Zero via dlopen,
   OpenCL C via ocloc, nGEN GEMMs, graph record/replay, P2P TP collectives, lanes scheduler) for the 27B, so the
   host-runtime risk is low; the risk is in the Flash-Next kernels (IQ formats incl. ISTA Q2_0 type 42, QSA, HC,
   GDN) and in qualification (section 5, 6).
6. Cheapest juice first (inside Strata, weeks, not months): batched QSA under KV streaming (C4 +12-16%), fill the
   8-row window by slot count (C2 +15-30%/stream), collapse commit/emit and take the PLE wait off the critical path
   (C4 +12-18%), fix the C1 kv_append_q8_steps anomaly (C1 ~+5%), shared expert to card 1 + cold experts on card 0
   (C1/C4 +5-10%), restore/admission path (TTFT 2.08 s -> ~0.8 s). Combined, Strata could plausibly reach C1 ~155-165,
   C2 ~95-105/stream, C4 ~55-65/stream (220-260 aggregate) [I]. The own engine targets 1.4-1.8x C1, 2-4x C4,
   C16 that Strata cannot run, and 5-10x prefill [I].

## 2. Decode roofline, Flash-Next IQ3_XXS on 2x B70

### 2.1 Model terms (per decode window)

Geometry [R model-facts]: 48 layers = 36 GDN + 12 QSA (full attention, lightning indexer top-2048), hidden 2560,
4 hyper-connection streams, 512 routed experts top-10 (FFN 640) + 1 shared expert, PLE n-gram embedding (layer 1,
16 hash heads, 160 dims), vocab 248,320.

Fixed bytes per window, read once whatever the row count if every op is batched [R Fable 5.1]:

| Group | MiB | Note |
|---|---:|---|
| HC projections, Q8_0 copies (96 down/up reads + final mixer) | ~650 | BF16 originals 1,200 MiB |
| GDN dense: qkv 597 + gate 304 + ssm_out 376 | 1,277 | IQ4_XS/Q6_K/Q4_K/Q5_K/IQ3_S mix |
| QSA dense: attn_q 226 + attn_output 142 + k/v 22 | 389 | |
| BF16 GEMVs: router 120, indexer 35, ssm alpha/beta 17, PLE key/value 62.5 | 235 | router could be Q8 |
| Shared expert gate/up/down | 115 | |
| Vocab head Q5_K | 417 | |
| Total | 3,083 MiB = 3.23 GB | 5.3 ms at 608 GB/s, 7.6 ms at 70% |

Growing terms [I unless tagged]:
- Routed experts: bytes = D x 1.746 MB x 48, D = distinct experts per layer in the window (average expert gate+up+down
  is 1.746 MB [R Fable]). Within one stream, consecutive rows share experts: 1/2/3/4 rows touch 10/16/22/27 distinct
  (Steve's census [R refresh S4]; Strata's own C1 plan: 10 + 18 = 28 groups for 4 rows [R moe_report via Opus walk]).
  Extrapolated f(5)=31.5, f(6)=36. Across S independent slots: D = 512 x (1 - (1 - f/512)^S). This is uniform across
  experts; real routing is skewed, so D is an upper estimate at high C. Cross-check: the C4 partial census shows
  ~7.0 ms of routed kernels on each card [M-partial], i.e. ~2.8-3.0 GB per card at ~400 GB/s, against 5.1 GB total
  predicted for D=61: consistent.
- GDN state: 817,152 fp32 = 3.27 MB per slot per GDN layer, 117.7 MB per slot [R Fable]. Minimum traffic read once +
  write once = 235 MB per slot per window. Strata reads it in verify and again in commit and writes it in commit:
  ~353 MB. (Confirmed size: a D2M copy of exactly 117,669,888 B per slot appears in B3 [M-partial].)
- Attention per row per QSA layer: min(ctx, 2048) selected cells x 1056 B (int8 K/V incl. scales) = 2.16 MB, plus the
  slot's pooled indexer keys 128 B x ctx (fp32, 128 floats per 4-cell block) read once per slot if batched (per row
  in Strata's current per-row path). Window: 12 x (R x 2.16 MB + S x 128 B x ctx).
- MTP drafter per step, batched over all slots: dense Q8 111 MiB + draft-vocab head 68 MiB + union of MTP experts
  (1.36 MB each) [R Fable 5.2]. Strata pays one drafter pass per PAIR of slots.
- PLE: 16 x 90 B rows per row from a 26.8-28.8 GB table on NVMe/page cache, host side, ~10 KB H2D per row.
- Tokens per window: chain MTP with per-step acceptance a = 0.85 gives 1 + a + ... + a^d = 1.85 (d=1), 2.57 (d=2),
  3.19 (d=3), 4.15 (d=5). Calibrated on Strata: d=3 solo measured 3.12, d=1 batch measured 1.85 (C2) and 1.70 (C4)
  [M]. Code-heavy text accepts more, chat less.
- Compute: ~6.63 G MAC per row (2.08 G GDN dense + 0.62 G QSA dense + 0.63 G HC + 0.06 G router + 0.24 G shared +
  2.36 G routed + 0.64 G head) [I from geometry]. Even at 64 rows that is ~8 TOPS per card at the window times below:
  decode is bandwidth/latency bound at every C here. The hidden compute is IQ dequant ALU work per weight byte
  (~2.7 weights per byte, several ops each, not amortised for experts that see ~1-2 rows): estimated ~8 ms per card per
  window at C16 d=3, i.e. still under the bandwidth time but only ~6x margin; dense layers at >=16 rows should use
  DPAS tiles with in-register dequant [I].

### 2.2 Placement layouts compared

All at 70% of 608 GB/s = 425.6 GB/s per card, plus a latency floor of 2.0 ms + 0.05 ms per row (about 450 dependent
kernel boundaries per window at ~3-4 us, a purpose-built engine) [I]. The bytes column is the critical card.
- EP-now: Strata's split. Card 0 = all dense + GDN state + KV + drafter + 45% of routed; card 1 = 55% of routed.
- EP-best: shared expert moved to card 1, card 0 holds the COLDEST experts so its routed traffic share f0 is minimised
  (bound 0.2) and balanced against card 1. Same data flow as Strata (no new exchanges).
- TP2: Megatron-style split of attention/GDN heads and dense matrices plus expert split (EP or intermediate split as
  in TensorFold's CUDA 2-rank Flash-Next [R flash-next-cuda.md]); HC weights replicated; 96 all-reduce hops per window
  at 10 us + R x 5 KB / 11 GB/s each, on the critical path.
- PP2: card 0 layers 0-23, card 1 layers 24-47 + drafter, two micro-batches of S/2 slots in flight. Note: each card
  reads its half of the dense weights twice per period (once per micro-batch), so dense bytes per card per period
  equal the full dense set: PP does NOT halve the dense read; it only balances.
- Balanced-BW-only: (all unique bytes) / 2 cards, no latency floor. A hard upper bound.

Cells: "GB / ms / tok/s per stream / tok/s aggregate". Context 4,096:

| Config | Rows | D/layer | Routed GB | State GB | KV GB | Draft GB | Tok/window | EP-now | EP-best | TP2 | PP2 per/agg | BW-only per/agg |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| C1 d=3 | 4 | 27 | 2.26 | 0.24 | 0.11 | 0.6 | 3.19 | 5.2 / 14.4 / 221 / 221 | 4.51 / 12.8 / 249 / 249 (f0 0.2) | 3.56 / 11.7 / 272 / 272 | - | 421 / 421 |
| C1 d=5 | 6 | 36 | 3.02 | 0.24 | 0.16 | 1 | 4.15 | 5.99 / 16.4 / 254 / 254 | 5.12 / 14.3 / 290 / 290 (f0 0.2) | 4.17 / 13.3 / 312 / 312 | - | 462 / 462 |
| C2 d=1 | 4 | 32 | 2.64 | 0.47 | 0.12 | 0.21 | 3.7 | 5.22 / 14.5 / 128 / 256 | 4.44 / 12.6 / 146 / 293 (f0 0.2) | 3.68 / 12 / 154 / 309 | 127 / 254 | 236 / 472 |
| C2 d=3 | 8 | 53 | 4.41 | 0.47 | 0.22 | 0.64 | 6.37 | 6.55 / 17.8 / 179 / 358 | 5.33 / 14.9 / 214 / 427 (f0 0.2) | 4.83 / 15.1 / 212 / 423 | 168 / 336 | 302 / 605 |
| C4 d=1 | 8 | 61 | 5.12 | 0.94 | 0.23 | 0.24 | 7.4 | 6.95 / 18.7 / 99 / 395 | 5.55 / 15.4 / 120 / 479 (f0 0.2) | 5.22 / 16 / 116 / 463 | 100 / 398 | 161 / 645 |
| C4 d=3 | 16 | 100 | 8.36 | 0.94 | 0.44 | 0.72 | 12.75 | 9.1 / 24.2 / 132 / 527 | 6.89 / 19 / 168 / 672 (f0 0.2) | 7.19 / 21.4 / 149 / 597 | 126 / 502 | 198 / 792 |
| C8 d=1 | 16 | 115 | 9.62 | 1.88 | 0.47 | 0.29 | 14.8 | 10.2 / 26.8 / 69 / 553 | 7.77 / 21.1 / 88 / 703 (f0 0.21) | 8.09 / 23.5 / 79 / 630 | 70 / 563 | 102 / 813 |
| C8 d=3 | 32 | 180 | 15.09 | 1.88 | 0.88 | 0.87 | 25.49 | 13.65 / 35.7 / 89 / 714 | 10.99 / 29.4 / 108 / 867 (f0 0.28) | 11.32 / 32.6 / 98 / 782 | 85 / 680 | 124 / 988 |
| C16 d=1 | 32 | 204 | 17.09 | 3.77 | 0.93 | 0.38 | 29.6 | 16 / 41.2 / 45 / 719 | 12.77 / 33.6 / 55 / 881 (f0 0.26) | 13.04 / 36.6 / 51 / 808 | 45 / 728 | 62 / 992 |
| C16 d=3 | 64 | 297 | 24.88 | 3.77 | 1.76 | 1.13 | 50.99 | 21.08 / 54.7 / 58 / 932 | 17.48 / 46.3 / 69 / 1102 (f0 0.31) | 17.72 / 50.7 / 63 / 1006 | 54 / 867 | 78 / 1248 |

Context 200,000 (same model; KV term includes 307 MB of fp32 pooled keys per slot):

| Config | Rows | D/layer | Routed GB | State GB | KV GB | Draft GB | Tok/window | EP-now | EP-best | TP2 | PP2 per/agg | BW-only per/agg |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| C1 d=3 | 4 | 27 | 2.26 | 0.24 | 0.41 | 0.6 | 3.19 | 5.5 / 15.1 / 211 / 211 | 4.81 / 13.5 / 236 / 236 (f0 0.2) | 3.71 / 12.1 / 264 / 264 | - | 402 / 402 |
| C1 d=5 | 6 | 36 | 3.02 | 0.24 | 0.46 | 1 | 4.15 | 6.29 / 17.1 / 243 / 243 | 5.42 / 15 / 276 / 276 (f0 0.2) | 4.32 / 13.7 / 304 / 304 | - | 445 / 445 |
| C2 d=1 | 4 | 32 | 2.64 | 0.47 | 0.72 | 0.21 | 3.7 | 5.82 / 15.9 / 116 / 233 | 5.04 / 14 / 132 / 263 (f0 0.2) | 3.98 / 12.7 / 146 / 292 | 121 / 242 | 216 / 433 |
| C2 d=3 | 8 | 53 | 4.41 | 0.47 | 0.82 | 0.64 | 6.37 | 7.15 / 19.2 / 166 / 332 | 5.93 / 16.3 / 195 / 390 (f0 0.2) | 5.13 / 15.8 / 202 / 404 | 162 / 324 | 283 / 567 |
| C4 d=1 | 8 | 61 | 5.12 | 0.94 | 1.44 | 0.24 | 7.4 | 8.15 / 21.6 / 86 / 343 | 6.75 / 18.3 / 101 / 405 (f0 0.2) | 5.83 / 17.4 / 106 / 425 | 93 / 370 | 144 / 574 |
| C4 d=3 | 16 | 100 | 8.36 | 0.94 | 1.64 | 0.72 | 12.75 | 10.3 / 27 / 118 / 472 | 8.09 / 21.8 / 146 / 585 (f0 0.2) | 7.79 / 22.8 / 140 / 560 | 119 / 476 | 182 / 728 |
| C8 d=1 | 16 | 115 | 9.62 | 1.88 | 2.87 | 0.29 | 14.8 | 12.61 / 32.4 / 57 / 456 | 10.08 / 26.5 / 70 / 559 (f0 0.2) | 9.29 / 26.3 / 70 / 563 | 63 / 508 | 88 / 704 |
| C8 d=3 | 32 | 180 | 15.09 | 1.88 | 3.29 | 0.87 | 25.49 | 16.06 / 41.3 / 77 / 617 | 12.19 / 32.3 / 99 / 790 (f0 0.2) | 12.52 / 35.4 / 90 / 720 | 79 / 633 | 111 / 891 |
| C16 d=1 | 32 | 204 | 17.09 | 3.77 | 5.75 | 0.38 | 29.6 | 20.81 / 52.5 / 35 / 564 | 16.42 / 42.2 / 44 / 702 (f0 0.2) | 15.45 / 42.3 / 44 / 700 | 40 / 639 | 52 / 834 |
| C16 d=3 | 64 | 297 | 24.88 | 3.77 | 6.58 | 1.13 | 50.99 | 25.9 / 66 / 48 / 772 | 19.8 / 51.7 / 62 / 986 (f0 0.21) | 20.13 / 56.3 / 57 / 905 | 49 / 791 | 69 / 1097 |

Reading the table [I]:
- EP-best is as good as TP2 at every C on bandwidth: card 0's dense read is the floor, and TP2 pays for replicated HC
  (0.68 GB per card) and 96 synchronous hops. TP2's real advantages are capacity (KV, pooled keys and GDN state split
  across both cards) and robustness to expert skew. PP2 loses at every point because each card re-reads half the dense
  set per micro-batch.
- The routed expert union is the term that grows fastest: 2.3 GB at C1, 5.1 GB at C4 d=1, 17 GB at C16 d=1, 25 GB at
  C16 d=3 (40-58% of all experts read every window). This is why per-stream decode falls with C even in an ideal
  engine, and why "batched-everything" alone saturates near ~1 K aggregate on two B70s.
- GDN state traffic becomes the second-largest term at C>=8 (1.9-3.8 GB per window). Levers: read once/write once
  (fused verify and commit, section 5.5), and a BF16 state (halves it; needs a quality gate).
- Fixed floor matters most at C1: 2.3 ms of a 12.8 ms window.

### 2.3 PCIe traffic per window, EP vs alternatives

Host link facts (this box) [M P2P_GPU.md J.2-J.8]: posted peer WRITE (push) 11.3 GB/s from EU or copy engine, peer
READ (pull) 3.24 GB/s, 8-byte peer latency 9.3 us raw L0, 10 KB push 7.9 us, 16 MB push 1.5 ms; a two-rank push
all-reduce measured 59.5 us at 10 KB (4 launches + 2 syncs, unfused). Host-staged copies 1.16-1.35 GB/s.

| Flow (per row per layer) | Bytes | Per row per window (48 layers) |
|---|---:|---:|
| EP push card 0 -> card 1 (q8_1 row 2,880 B + ids/weights 80 B) | 2,960 | 142 KB |
| EP return card 1 -> card 0 (fp32 weighted sum) | 10,240 | 492 KB |
| TP2 all-reduce, 2 per layer, bf16 | 2 x 5,120 each way | 480 KB each way |
| PP2 hand-off (4 HC streams x 2560 fp32), once per micro-batch | 40,960 per row per window | 41 KB |

| Rows | EP bytes/window | EP time (return-bound) | TP2 bytes each way | TP2 time |
|---:|---:|---:|---:|---:|
| 4 | 2.5 MB | 0.17 ms | 1.9 MB | 0.17 ms + 96 x hop |
| 8 | 5.1 MB | 0.35 ms | 3.8 MB | 0.35 ms + 96 x hop |
| 16 | 10.1 MB | 0.70 ms | 7.7 MB | 0.70 ms + 96 x hop |
| 32 | 20.3 MB | 1.39 ms | 15.4 MB | 1.36 ms + 96 x hop |
| 64 | 40.6 MB | 2.79 ms | 30.7 MB | 2.73 ms + 96 x hop |

[I] Bandwidth is never the problem in decode; latency is: 96 dependent hand-offs x ~8-10 us = 0.8-1.0 ms per window,
which EP can hide behind card 0's own experts + shared expert (Strata already does: SH_LATE), and TP cannot.
Returning bf16 instead of fp32 halves the dominant flow (needs a parity/NLL gate). Prefill is different (section 3).

### 2.4 Strata measured vs ceiling, and why

| C | Strata measured | Window and rows | Ceiling (EP-best, 70% BW + floor) | Gap | Where the time goes (measured) |
|---|---|---|---|---:|---|
| C1 | 134.6 tok/s [M] | 23.17 ms, 4 rows, 3.12 tok | 249 (d=3), 290 (d=5) | 1.85x | Old census [M TFA]: 1,575 launches, 683 under 5 us; HC 3.28 ms at 38% of roofline; shared expert 18%; IQ3_S KREG2 16%; host gaps ~1-1.3 ms unprofiled; PLE ~0.7 ms. New B1 [M]: kv_append_q8_steps 1.39 ms/window at 108 us per launch (the per-row variant costs 7.9 us in B3), card 1 spins 17.7 ms of the window. |
| C2 | 65.4/stream (130.8 agg) [M]; 73.1 in Stage A [M] | 24.2 ms p50, 4 rows, 1.85 tok/stream | 146/stream (d=1), 214/stream (d=3) | 2.2-3.3x | Depth pinned at 1 (row budget 4 of 8 used), per-row QSA, per-slot commit/emit, admission stalls 1.7-4.0 s in the harness [M Fable 1.7]. |
| C4 | 37.3/stream, 141 agg [M]; 36.5/138 Stage A [M] | 44.1 ms, 8 rows, 1.70 tok/stream, 20K ctx | 120/stream 479 agg (d=1), 168/stream 672 agg (d=3) | 3.2-4.8x | Partial census per window [M-partial]: QSA per-row path ~8.3 ms (vector attention 3.1, top-k 1.9, resolve 1.1, scores 0.7, launch_normal 0.7, KV append 0.8, indexer append 0.5); HC 4.9 ms (0.68 GB = 140 GB/s); GDN 4.1 ms; PLE host-flag wait 3.0 ms; dense KREG ~8.6 ms; head 1.25 ms at 8 columns; card-0 routed 7.0 ms; card 1 busy 8.5 ms and spinning 40.0 ms; commit 2.76 + emit 3.01 ms [M LOG]. Card 0 moves ~8.3 GB in ~44 ms = 189 GB/s (31%). |
| C8 | not runnable | kVerifyMaxT = 8 rows, KREG 1-8 columns, --parallel 4; 12-row windows were slower [R/M] | 88/stream 703 agg (d=1), 108/867 (d=3) | - | needs new kernel families and a scheduler |
| C16 | not runnable | same | 55/881 (d=1), 69/1102 (d=3) | - | plus capacity: 16 x 200K needs the RAM tier (section 5.6) |

Host-side shape (Stage B) [M]: C1 B1 shows ~510 zeCommandListHostSynchronize calls per window (a polling loop: the
host thread is synchronous for the whole window); cached C4 shows ~14 blocking host syncs per window, mean 4.1 ms.
A D2M copy of one slot's 117.7 MB GDN state took 26.5 ms (4.4 GB/s) and 2.6 MB copies took 564 us (4.6 GB/s), well
below the 11 GB/s posted-write ceiling [M-partial].

## 3. Prefill roofline

Per prompt token [I from geometry]: weights 2 x 6.63 G = 13.3 GFLOP; sparse QSA attention 24 heads x 256 x 2048 x 4 =
50 MFLOP per QSA layer = 0.6 GFLOP; indexer scoring 3,072 x (average context) FLOP (0.006 GFLOP at a 4K prompt,
0.05 at 32K, 0.31 at 200K); chunked GDN ~0.3 GFLOP. Total ~14.2 GFLOP (4K-32K) to ~14.5 (200K).

| Basis (per card) | Two cards perfectly split | One card carrying dense + attention + GDN (Strata layout, 82% of FLOPs on card 0) |
|---|---:|---:|
| XMX bf16 peak 183 TFLOPS [R memory note] | 25.8K tok/s | ~15.7K |
| Whole-model bf16 efficiency TF demonstrates on B70 (27B, 1,975 tok/s x 53 GFLOP = ~105 TFLOPS) [M TF README] | 14.8K | ~9K |
| W4A8 int8 efficiency TF demonstrates (2,617 tok/s = ~139 TOPS) [M] | 19.6K | ~12K |
| Realistic MoE derate 0.5-0.6 of the above (per-expert M ~160 rows at 8K chunks, IQ dequant, routing/sort) | 7-12K | 4.5-7K |
| Strata measured | 889 (4K) / 1,173 (32K) on 0.2.16; 1,082/1,182 old night (BRIEF) and ~2,200 night-lab harness [M] | |

Other prefill limits [I]:
- Weight bytes: a chunk of >= ~600 tokens touches nearly all 512 experts per layer; reading all experts once per
  chunk is 42.9 GB split over two cards = ~50 ms at 70%; an 8K chunk costs ~0.8 s of compute at 10K tok/s, so
  prefill is compute-bound for chunks >= ~1K tokens.
- PCIe: EP prefill moves 634 KB per token (fp32 return) = 6.3 GB/s at 10K tok/s (return 4.9 GB/s one way, 43% of
  a direction); bf16 return gives 3.9 GB/s. TP2 needs 4.8 GB/s each way at 10K tok/s. Gen3 becomes the binding limit
  above ~20K tok/s; it must overlap with compute (fused push epilogues or the copy engine).
- PLE: 16 random 4 KiB reads per token = 160K IOPS and 655 MB/s at 10K tok/s from NVMe (fine for a modern NVMe at
  high queue depth with io_uring), or nothing if the 28.8 GB table is page-cache resident (RAM budget, section 5.8).
- Host: Strata blocks the host 48+ times per chunk for expert grouping (D2H 480 KiB of ids per layer at T=12,288)
  [R Opus walk 1b.4]. A device-side sort/segment removes that entirely.
- Cold C4 (four 20K prompts): 80K tokens at 8K tok/s = 10 s total vs 84 s today; FCFS TTFTs ~2.5/5/7.5/10 s [I].

## 4. Qwen3.8-27B dense (hybrid GDN, 4-bit) for comparison

Terms [I unless tagged]: weights read per pass ~14.5 GB (int4 g128 + fp16 gates + 4-bit head; int8 pass is ~27 GB,
22.5 tok/s ceiling [R memory note]); KV 16 attention layers x 4 KV heads x 256 x 2 = 32 KB per token per slot at
int8/fp8 (64 KB at bf16 [R TF README]), and dense attention reads ALL of it every step (rows of a slot share it);
GDN state ~151 MB per slot [R TF] -> 302 MB read+write per window; drafter ~0.5 GB per window; ~26.5 G MAC per row.
Same 70% BW + 2.0 ms + 0.05 ms/row floor; TP2 adds 128 all-reduce hops at 10 us + R x 10 KB / 11 GB/s.

| Config | Context | Layout | GB per card | Window ms | tok/s per stream | Aggregate | TOPS needed per card |
|---|---:|---|---:|---:|---:|---:|---:|
| C1 d=3 | 4096 | 1card | 15.44 | 38.5 | 83 | 83 | - |
| C1 d=3 | 4096 | TP2 | 7.72 | 22.1 | 144 | 144 | 4.8 |
| C2 d=3 | 4096 | DP2 | 15.44 | 38.5 | 83 | 166 | 5.5 |
| C2 d=3 | 4096 | TP2 | 7.94 | 23.3 | 137 | 274 | 9.1 |
| C4 d=1 | 4096 | DP2 | 15.87 | 39.5 | 47 | 187 | 5.4 |
| C4 d=1 | 4096 | TP2 | 8.37 | 24.3 | 76 | 304 | 8.7 |
| C4 d=3 | 4096 | DP2 | 15.87 | 39.7 | 80 | 321 | 10.7 |
| C4 d=3 | 4096 | TP2 | 8.37 | 25.7 | 124 | 497 | 16.5 |
| C16 d=1 | 4096 | DP2 | 18.49 | 46.2 | 40 | 640 | 18.3 |
| C16 d=1 | 4096 | TP2 | 10.99 | 34.5 | 54 | 858 | 24.6 |
| C16 d=3 | 4096 | DP2 | 18.49 | 47 | 68 | 1084 | 36.1 |
| C16 d=3 | 4096 | TP2 | 10.99 | 39.9 | 80 | 1277 | 42.5 |
| C1 d=3 | 32768 | 1card | 16.38 | 40.7 | 78 | 78 | - |
| C1 d=3 | 32768 | TP2 | 8.19 | 23.2 | 137 | 137 | 4.6 |
| C2 d=3 | 32768 | DP2 | 16.38 | 40.7 | 78 | 157 | 5.2 |
| C2 d=3 | 32768 | TP2 | 8.88 | 25.5 | 125 | 250 | 8.3 |
| C4 d=1 | 32768 | DP2 | 17.75 | 43.9 | 42 | 169 | 4.8 |
| C4 d=1 | 32768 | TP2 | 10.25 | 28.7 | 64 | 258 | 7.4 |
| C4 d=3 | 32768 | DP2 | 17.75 | 44.1 | 72 | 289 | 9.6 |
| C4 d=3 | 32768 | TP2 | 10.25 | 30.1 | 106 | 424 | 14.1 |
| C16 d=1 | 32768 | DP2 | 26.01 | 63.9 | 29 | 463 | 13.3 |
| C16 d=1 | 32768 | TP2 | 18.51 | 52.2 | 35 | 567 | 16.3 |
| C16 d=3 | 32768 | DP2 | 26.01 | 64.7 | 49 | 788 | 26.2 |
| C16 d=3 | 32768 | TP2 | 18.51 | 57.6 | 55 | 885 | 29.5 |
| C1 d=3 | 200000 | 1card | 21.86 | 53.6 | 60 | 60 | - |
| C1 d=3 | 200000 | TP2 | 10.93 | 29.6 | 108 | 108 | 3.6 |
| C2 d=3 | 200000 | DP2 | 21.86 | 53.6 | 60 | 119 | 4 |
| C2 d=3 | 200000 | TP2 | 14.36 | 38.4 | 83 | 166 | 5.5 |
| C4 d=1 | 200000 | DP2 | 28.71 | 69.7 | 27 | 106 | 3 |
| C4 d=1 | 200000 | TP2 | 21.21 | 54.5 | 34 | 136 | 3.9 |
| C4 d=3 | 200000 | DP2 | 28.71 | 69.9 | 46 | 182 | 6.1 |
| C4 d=3 | 200000 | TP2 | 21.21 | 55.8 | 57 | 228 | 7.6 |
| C16 d=1 | 200000 | DP2 | 69.84 | 166.9 | 11 | 177 | 5.1 |
| C16 d=1 | 200000 | TP2 | 62.34 | 155.2 | 12 | 191 | 5.5 |
| C16 d=3 | 200000 | DP2 | 69.84 | 167.7 | 19 | 304 | 10.1 |
| C16 d=3 | 200000 | TP2 | 62.34 | 160.6 | 20 | 317 | 10.6 |

Capacity [I]: one card holds ~15.5 GB of weights + ~1 GB drafter, leaving ~14 GB for KV/state = ~440K int8 tokens:
2 x 200K per card (DP=2 -> 4 x 200K total, as the vLLM fallback serves). C8/C16 at 200K do not fit (and a RAM tier
cannot help: every token reads the full KV, 6.4 GB per slot per window, ~0.6 s over PCIe).
Measured references [M, TF README and Sergio cookbook]: TensorFold XPU 27B, one B70: single stream 129-133 tok/s
(DFlash2 tree, ~5 tokens per round), 8 streams 439-463 aggregate (bench_openai), prefill 1,975-2,830 tok/s; two cards
TP=2: 173/151/124/82 tok/s at 68/8K/32K/131K context; two one-card servers 159-161 aggregate at 4 streams vs 129-130 for
one two-card server.

Verdict [I + M A/B]: the 27B's ceilings are roughly C1 83 (one card, chain d=3; ~135 with a tree drafter) to 144 (TP2),
C4 ~320-500 aggregate at 4K but ~180-230 at 200K, C16 ~1.1-1.3K aggregate at 4K (needs XMX-class decode kernels:
36-43 TOPS per card) and infeasible at 200K. Flash-Next's ceilings are higher at C1-C8 at any context and at C16 for
long context; the 27B only edges ahead at C16 short-context. Add the quality result (Flash-Next 41/43 vs 27B 33/43,
8 paired wins to 0, docs/20261013_flashnext_iq3_vs_27b_ab_brief.md) and the answer is: no model change for
concurrency. Keep the 27B as fallback; TensorFold XPU would roughly 1.5x the fallback's single-stream speed over the
current vLLM INT4 recipe and is worth one evaluation.

## 5. Engine design: a Zig engine on Level Zero for this host

### 5.1 What Strata structurally cannot do (vs what is fixable)

Structural (changing these is a rewrite of Strata's core, ~120K lines of engine/sycl/src) [R walks + I]:
1. Card roles are hard-wired: every session's KV, GDN state, pooled keys, MTP state and all dense work live on card 0
   (session_init gen:4406-4470); card 1 is a remote-expert server (6 launches per layer, 289 per window). No TP, no
   per-card state, no PP; balancing beyond moving expert traffic is not expressible.
2. One host command thread, serial windows, synchronous host: 2 graph launches then 4-12 blocking waits per window
   (verify, commit, per-slot residual copies, per-pair drafts), ~510 host polls per C1 window [M]. --pipeline-windows 2
   crashes on SYCL (#2800) and EP disables it.
3. SYCL command graphs: one captured graph per row layout (LRU 64; a capture costs ~42 ms [M TFA]), no mutable or
   indirect dispatch, event resets inside graphs, cooperative launches fail inside SYCL graphs [M night lab]. Mixed
   prefill+decode windows or variable depth per slot multiply layouts.
4. Kernel model: dpct-migrated CUDA kernels; per-row and per-slot launches baked into verify/commit (QSA 8 launches
   per row per layer, commit 110 launches per slot); decode kernels capped at 8 columns (kVerifyMaxT = 8, KREG 1-8),
   expert grids sized 10R regardless of the plan.
5. Prompt path: every prompt read through the one main session, one prompt at a time, host-side expert grouping with
   48+ host blocks per chunk; SYCL lacks the CUDA engine's BCO / SLOT_READ / PIPED_CONCURRENT.
6. Control plane: Python ctl lock held across an admission's park/copy/draft_first; tokenization and prefix search in
   Python (O(n log n) list slicing on 132K-token prompts).
7. Divergence tax: the DD needs ~65 env switches, several inert in this config (BATCH_MTP_PAIR_GRAPH, GDN_COMMIT_LAYERS,
   BATCH_ATTN_ROWS, GDN_GRF in batch windows [R Fable 8]); the 0.2.16 port silently dropped the night batched
   attention. Every upstream refresh re-ports.

Fixable inside Strata (cheap to medium; section 6 ranks them): batched QSA under streaming, depth by slot count,
generalized one-launch commit, non-blocking commit and concurrent pair drafts, PLE gather overlap, the C1 KV-append
fix, shared expert and cold-expert placement, restore into a free slot / async park / multithreaded host copies /
KV-map prefetch / ctl release, prefill device-side grouping, private-memory audit, lone-slot solo window.

### 5.2 Process, devices, memory

- One process, one ze_context spanning both devices; single-context peer access needs no IPC handles (J.1: canAccessPeer
  true both ways) [M]. Load libze_loader with dlopen and declare the entry points in Zig (TensorFold ze.zig, 549
  lines, does exactly this [R]), or @cImport ze_api.h.
- Weights: one device arena per card (zeMemAllocDevice, large pages), filled by mmap of the GGUF shards + copy-engine
  H2D at load; layout repacked at load into kernel-native tiles (the "KREG" layouts, VNNI for DPAS paths).
- Elastic KV and state pools: zeVirtualMemReserve + zePhysicalMemCreate + zeVirtualMemMap, so pools grow and shrink
  without moving addresses (captured command lists stay valid) - the native form of Strata's --kv-grow, for any
  slot count and both cards.
- Host tiers: pinned zeMemAllocHost (or imported mmap memory via the Intel import-external-pointer extension, [I] to
  verify on 26.x) for the authoritative KV store, park snapshots and PLE staging.
- Audit every kernel with zeinfo for per_thread_memory_buffers (Steve S1: 35-70 MiB hidden private memory per kernel
  that indexes argument tables dynamically) [R refresh].

### 5.3 Queues, events, graphs, host loop

- Per card: one in-order immediate compute list (ordinal 0) for the window chain; one copy list (ordinal 1, BCS) for
  park/restore, KV write-back and PLE rows; optionally a low-priority compute list for background work.
- Counter-based events (no reset) for all intra-device ordering; cross-device ordering by either (a) device-side flags
  in peer memory with system-scope loads (what Strata's ep_wait does; it works but burns an EU thread and hid 40 ms
  of card-1 idle per C4 window), or (b) a command-streamer wait on an event from a pool created on BOTH devices
  (memory note: CS waits are correct and replayable; a 1-device pool segfaults the peer). Measure both hop latencies
  in the Stage 1 harness.
- Record once per row bucket (4/8/16/32/64 rows), not per layout: all variable data (row count, positions, slot ids,
  page tables, expert plan counts) lives in a per-window descriptor in device memory, read by the kernels; grids that
  depend on the plan use zeCommandListAppendLaunchKernelIndirect (group counts from device memory), so no empty
  expert work-groups and no recapture. Replay via the record/replay graph extension (TensorFold binds
  zeCommandListBeginGraphCaptureExt / zeGraphInstantiateExt / zeCommandListAppendGraphExt [R ze.zig]) or a closed
  regular command list appended with zeCommandListImmediateAppendCommandListsExp (what Strata's SYCL graphs already
  lower to: 489 such calls in B1 [M]).
- Device-driven decode loop: the sampling kernel writes the next window's tokens and positions into the next
  descriptor and the mapped output ring; the host only appends the next replay and drains the ring for streaming.
  Commit, draft and next verify are queued back to back; the host waits on nothing inside the steady state. This is
  the single biggest structural win at C1 (removes the 4-12 blocking waits and most of the ~1-3 ms host gaps).
- Launch economy: target <= 10 dependent kernels per layer (~480 per window) by fusing the small ops: q8_1 quantize
  into the producing norm/HC kernel; router GEMV + top-10 + plan in one; K/V append + indexer append + RoPE in one;
  combine into the down-projection epilogue; GDN conv + alpha/beta + recurrence + norm in one per layer for all slots.
  Persistent/cooperative megakernels are a later experiment (they regressed in Strata [M night lab]); with Level
  Zero directly, zeCommandListAppendLaunchCooperativeKernel is available outside SYCL's graph limits [I].

### 5.4 Kernel strategy

- Primary language: OpenCL C compiled offline by ocloc for bmg (Xe2, 256-GRF mode), embedded zebins, loaded with
  zeModuleCreate (native binary format) - TensorFold's route, proven on B70 [R]. Intel extensions give what decode and
  prefill need: sub-group shuffles and block reads, 2D block IO on Xe2, DPAS (subgroup matrix multiply accumulate),
  required sub-group size 16. Arguments are plain kernel parameters (clean zeKernelSetArgumentValue ABI).
- Hot kernels needing explicit LSC control or s4/s2 DPAS: ESIMD compiled by icpx to a native image and loaded the same
  way, or Intel's C-for-Metal (cmc) to zebin [I: argument-ABI extraction must be validated]. nGEN (oneDNN's ISA
  generator) for prefill GEMMs, as TensorFold does (9 generated GEMM binaries) [R].
- Reuse: Strata is MIT, so its algorithms and code (KREG GEMV layouts, ISA-guided IQ decode incl. the ISTA Q2_0 type
  42 down projection, gr_q8 HC, GDN group recurrence, QSA scan top-k, transposed scores, EP push/psum shapes, XMX
  joint_matrix prefill experts) can be ported with the MIT notice kept; IQ grids come from ggml (MIT). TensorFold is
  Apache-2.0 (keep NOTICE, state changes). For A/B during the transition, Strata's SYCL kernels can be compiled to a
  native image and launched from the Zig host as parity oracles [I].
- Kernel families required (rows 1-64): dense GEMV/GEMM for Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3_S/Q8_0/BF16 (vector KREG form to
  8-16 rows, DPAS form above); routed expert gate/up/down for IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/Q2_0(42) grouped
  by expert with plan-sized grids and any-count row reuse; HC Q8 read/write; GDN conv+recurrence (verify and fused
  commit); QSA indexer scores/top-k/attention batched per slot with page-table resolve; router top-10; sampling;
  MTP layer; PLE projection. Prefill: DPAS GEMMs with IQ->bf16/int8 dequant in the loader stage, chunked GDN on DPAS,
  union-of-selection sparse attention, device-side expert sort/segment.
- Discipline: random-data benches (Xe2 memory compression fakes bandwidth on constant fills [M night lab]); report
  achieved GB/s and % of 608 for every kernel at 1/2/4/8/16/32/64 rows.

### 5.5 Window, MTP and state

- One verify forward per window for all slots, every op batched (weights once per window, pooled keys once per slot).
- Depth per slot chosen to fill a row bucket and by measured acceptance (QuixiInfer-style varlen verify): e.g. C1 1+5
  rows, C2 2 x 4, C4 4 x 2 or 4 x 4, C16 16 x 2. One batched drafter pass per draft step for all slots (Strata pays one
  per pair, serially).
- GDN state: verify carries the recurrence in registers across a slot's rows and speculatively writes the post-window
  state into a second buffer; on full acceptance (p ~0.85 at d=1) commit is a pointer swap; on partial acceptance
  replay the accepted prefix from stashed k/v/beta/g (TensorFold "stash" does this [R]). Expected traffic ~2.3x state
  instead of Strata's ~3x [I]. BF16 state as a gated option.
- KV append writes VRAM only; the RAM copy is written back by the copy engine after commit, off the critical path
  (Strata stores into pinned RAM from inside the append kernel; the C1 steps variant measures 108 us per launch [M]).

### 5.6 Scheduler, paging, park/restore

- Continuous batching with a per-window token budget. Decode rows first (bucketed). Prefill is chunked and batched
  across waiting prompts (one pass per chunk shares the expert reads, larger M per expert). Because any chunk over
  ~600 tokens reads nearly all experts, mixing prefill rows into decode windows is not free: a 2K-token chunk costs
  ~0.2 s at 10K tok/s. Policy knobs: (a) piggyback decode rows on prefill chunks (decode advances once per chunk) and
  (b) a decode share that alternates decode-only windows with chunks; prefix-aware admission (QuixiInfer) [I].
- Paged KV: int8 cells in blocks (e.g. 64 cells) with an authoritative pinned-RAM copy and a VRAM block cache keyed by
  recent top-k selections; pooled indexer keys stay VRAM-resident per active slot (307 MB fp32 at 200K, 154 MB bf16).
  Flash-Next's top-2048 sparse attention is what makes 16 x 200K plausible: only selected blocks need VRAM. Risk: miss
  traffic. After a cold admission Strata measured 69.5% hits and 0% right after a restore [M Opus walk]; at C16 d=1 a
  5% per-token selection churn would be ~41 MB per window (~3.7 ms on one link, ~1.9 ms split over two) [I].
- Park: GDN + conv state (118 MB) + pooled keys (up to 307 MB) + MTP state via BCS D2H; KV is already in RAM. At
  11 GB/s that is ~40 ms for a 200K slot (measured D2M today: 4.4-4.6 GB/s via the compute engine [M-partial]).
  Restore: the reverse plus a prefetch of the last selection's blocks (12 layers x 2048 x 1056 B = 26 MB). Target
  restore TTFT ~0.1-0.2 s vs 2.08 s measured [M].
- Keep Strata's engine line protocol (GEN/BGEN/BT/BDONE/park lines) in the first engine version so terminal/serve,
  the hotschmoe-dd launcher and every existing harness (nl.py, bench_concurrency, c4_compare, nl_ppl gates) run
  unchanged; replace the Python control plane later (or adopt TensorFold's Zig server).

### 5.7 Card balance plan

1. Start with EP-best (section 2.2): dense, attention, GDN, head and drafter on card 0; shared expert and the hot
   expert majority on card 1 (card 0 holds the coldest ~169/layer so its routed share is ~20-30%); EP push as posted
   writes fused into the producer kernel, return as bf16 sums. Same data flow as Strata, so window replays are
   directly comparable. Expected: card 1 busy 60-90% instead of 18% (C4).
2. At C>=8 or long context, move to head-split TP for attention/GDN/KV/pooled keys (capacity and state traffic split)
   while keeping experts EP or intermediate-split (TensorFold's 2-rank Flash-Next splits expert intermediate channels
   [R]); the extra 96 hops are worth it only when state/KV bytes dominate.
3. Not recommended: PP2 (re-reads dense per micro-batch), dense TP at C1 (hops on the chain), hot-expert share knobs
   (tried, no gain [M]).

### 5.8 CPU, NUMA, PLE, RAM

- The BIOS is in UMA (interleaved) mode: one NUMA node, GPUs report numa_node -1 [M]. Each GPU hangs off a different
  die. Option: switch to NUMA (die-local) mode, then pin card-0 service threads, its pinned buffers and its PLE I/O
  threads to die 0 and card 1's to die 1; vision encoder threads (strata-vision, 8 threads) away from the scheduler
  core. Expected gain small (0-3%), mostly park/restore throughput and jitter; needs a reboot window [I].
- Multithreaded host copies for snapshots (single memcpy is ~5-8 GB/s; the 1950X's 4 DDR4 channels support ~30-40 GB/s
  with 8-12 threads) [I].
- PLE: io_uring with deep queues (or page-cache residency of the 28.8 GB table), prefetch from draft tokens one window
  ahead, gather never on the critical path (it is today: 3.0 ms per C4 window [M-partial]).
- RAM budget [I]: PLE 28.8 GB + conversation cache 32 GiB + KV RAM tier (4 x 200K int8 = 10 GB; 16 x 200K = 40 GB) +
  OS/vision ~8 GB = ~79 GB at 4 slots (fits 121 GB), ~109 GB at 16 slots (tight).

### 5.9 Build on TensorFold or from scratch?

TensorFold (Apache-2.0) already has, on B70: the Level Zero binding, OpenCL C + nGEN build pipeline, graph record/
replay, P2P TP collectives over a shared context (tp.zig: 4-slot ring, cross-device events, push-shaped), the lanes
scheduler with shared verify rounds, prefix cache with bit-exact resume, OpenAI/Anthropic API, grammar; plus a Flash-Next
family on Metal/CUDA/HIP (sparse attention, HC, PLE-on-SSD, MTP, two-rank CUDA split) [R]. Missing for us: an XPU
Flash-Next family, GGUF IQ readers/kernels (TensorFold reads MLX/safetensors; it reads no GGUF on CUDA [R]) and the ISTA
GSQ-RCO formats. AGENTS.md prefers upstream features over a long-lived divergent fork: a Flash-Next XPU family inside
TensorFold (upstreamable) fits that rule better than a private engine, at the cost of TensorFold's heavy contract/receipt
process and churn (pinned Zig 0.17). Recommendation [I]: build the Stage 1-3 pieces (harness, kernels, window replay)
as a small standalone Zig repo that borrows TensorFold's ze.zig/tp.zig patterns with attribution; decide at the Stage 3
gate whether the full engine lands as a TensorFold family or stays standalone.

## 6. Where is the juice: ranked list

Gains are [I] estimates versus today's measured numbers (C1 134.6, C4 37.3/stream 141 agg, C16 not runnable).
Effort in agent-weeks of focused work (GPU-lease serialized).

| Rank | Item | Bucket | C1 | C4 | C16 | Effort | Evidence |
|---:|---|---|---|---|---|---|---|
| 1 | Batched QSA select/resolve/attend/append per slot under KV streaming (restore night ATTN_BATCH) | a | 0 | +12-16% (QSA ~8.3 -> ~2 ms) | n/a | 1 wk | M-partial 8.3 ms; R both walks |
| 2 | Fill the 8-row window by slot count (depth 3 at C2; solo window for a lone slot) | a | lone-slot 88 -> 134 | 0 (C2 +15-30%/stream) | n/a | 2-4 days | M Fable 1.1, 3.6 |
| 3 | Commit/emit collapse (one-launch commit any layout, no host wait, both pairs submitted, one residual copy) + PLE gather overlap | a | +3-5% | +12-18% (-5.8 ms) | n/a | 1-2 wk | M commit 2.76, emit 3.01, PLE wait 3.0 |
| 4 | C1 kv_append_q8_steps anomaly (108 us vs 7.9 us per-row) + KV RAM write-back off the append kernel | a | +5% | +2% | n/a | days | M B1/B3 |
| 5 | Shared expert to card 1 + cold experts on card 0 (EP-best placement) | a | +5-8% | +5-10% | n/a | 1 wk | I section 2.2; M card-1 spin |
| 6 | Restore/admission: restore into a free slot, async park, MT host copies, KV-map prefetch, release ctl before BADM copies | a | TTFT 2.08 -> ~0.8 s | burst TTFT -0.6..-3 s | - | 1-2 wk | M Opus walk 3 |
| 7 | Prefill: device-side expert grouping (no 48 host blocks per chunk), card 1 share, decode-share knob for cold bursts | a | prefill 1.2K -> 2-3K | cold-C4 mean TTFT 50 -> ~25-40 s | - | 2-3 wk | R Opus walk 2 |
| 8 | Own engine decode: roofline kernels at 1-64 rows, record-once-per-bucket graphs, device-driven loop, fused layers, EP-best | b | 134.6 -> 190-250 | 141 -> 400-600 agg | 700-1000 agg | 4-6 months total with 9-10 | I section 2 |
| 9 | Own engine prefill on XMX across both cards | b | prefill 1.2K -> 6-12K | cold-C4 84 s -> ~10 s | - | (in 8) ~2 months | I section 3; M TF 105-139 TFLOPS/card |
| 10 | Own engine RAM-tier paged KV + head-split TP option for 16 x 200K | b | 0 | 0 | capacity enabler (+0-10% speed) | +1-2 months | I 5.6 |
| 11 | Deeper/tree drafting at C1-C4 (d=5 chain or a trained tree drafter) | c/b | +15-40% | +10-25% | +0-5% | 2-8 wk (training is heavy) | I tables; TF DFlash2 ~5 tok/round on 27B |
| 12 | BF16 GDN state, bf16 EP return, bf16/int8 pooled keys | b (gated) | +2-4% | +5-8% | +10-15% | 1-2 wk each + NLL gates | I 2.2 |
| 13 | Move to 27B (DP2/TP2) | c | ~= today (TF 133-173 M) | short ctx ~= own-engine FN, 200K 2-3x worse | 4K ~1.1-1.3K agg; 200K infeasible | weeks | A/B loss M; I section 4 - reject |
| 14 | Equal power caps/clocks on both cards (230 W) | d | +0-5% | +0-5% | +0-5% | hours | M Sergio (his box) |
| 15 | BIOS NUMA mode + thread/buffer pinning | d | 0-2% | 0-3% | 0-3% | 1 reboot | I 5.8 |
| 16 | Both cards on one die/root complex | d | 0-2% (hop latency) | 0-2% | 0-3% | slot move + reboot | I; J.6 queued test |
| 17 | 256 GB RAM | d | 0 | 0 | capacity (16 x 200K + PLE cached + snapshots) | $ | I 5.8 |
| 18 | PCIe Gen5 platform (B70 links are Gen5 x16 capable; host is Gen3) | d | +3-8% | +5-10% | +5-15%; restore and prefill exchange 4x | $$$ + migration | R P2P_GPU.md line 32; I |
| 19 | Third/fourth B70 (X399: x16/x8/x16/x8, Gen3) | d | +10-20% | +30-60% | +50-90% | $$ + PSU + new split | I (more bandwidth; x8 links halve hop bandwidth) |

Expected end states [I]:
- Strata with items 1-7: C1 ~155-165, C2 ~95-105/stream, C4 ~55-65/stream (220-260 agg), prefill 2-3K, restore ~0.8 s.
- Own engine (8-10): C1 190-250 (up to ~300 with deeper drafting), C4 100-150/stream (400-600 agg), C16 45-65/stream
  (700-1000 agg), prefill 6-12K, restore 0.1-0.2 s, 4-16 slots x 200K with a RAM tier.

## 7. Staged plan (nothing wasted if we switch, or if we do not)

Stage 0 (now, 1-3 weeks, Strata): finish the B1-B6 census (in flight); land items 1-5 as default-off commits. Output:
the per-op ledger becomes the engine's target table; Strata dumps (hidden states, GDN state, KV, expert plans) for
one C1 and one C4 window become the engine's parity fixtures.

Stage 1 (2-3 weeks, reusable regardless): "b70bench", a small Zig program on raw Level Zero (dlopen, one context, both
cards) that loads native kernel binaries and measures the platform constants the design depends on: in-order immediate
list kernel-to-kernel gap; graph replay per-node overhead; counter-based vs regular events; indirect dispatch cost;
cross-card hop latency for EU-spin flags vs CS event waits; push bandwidth/latency at 1-64 KB; BCS H2D/D2H at
100 MB (why 4.4 GB/s today); host-mapped flag latency; cooperative launch. Answers feed Strata decisions too
(e.g. the EP hand-off shape).

Stage 2 (4-8 weeks, reusable): "b70k" kernel library with a C ABI: every Flash-Next op at 1/2/4/8/16/32/64 rows, a
Zig CPU reference decoder for every GGUF type in the pack (incl. Q2_0 type 42), parity against the CPU reference and
against Strata's kernels, and a GB/s table vs 608. Winners are back-ported to Strata as SYCL kernels with the same
algorithm (Strata gains even if the engine never ships).

Stage 3 (3-4 weeks): GO/NO-GO GATE. A Zig host that replays ONE Strata decode window on both cards with real weights
(mmap of the GGUF) and real state from the Stage 0 dumps: C1 (4 rows) and C4 (8 rows), EP exchange included, outputs
compared to Strata's (tolerance and argmax identity), unprofiled timing. Gate: C1 window <= 14 ms (vs 23.2) and C4 <=
20 ms (vs 44.1) at matched rows and context. Fail -> stop, port the findings into Strata.

Stage 4 (8-12 weeks): full engine: MTP drafter, sampling, scheduler, paged KV + RAM tier, park/restore, DPAS prefill,
Strata line protocol, then OpenAI server (TensorFold's or our own). Decide here whether it becomes a TensorFold XPU
family.

Stage 5 (2-3 weeks): AGENTS.md qualification: identity (served name hotschmoe-dd first), deterministic coherence,
NLL gates, concurrent serving, 2 x 200K needle and 4-conversation forced eviction without recompute, abfast A/B vs
Strata, teardown and xpu-health, before any DD change. Strata remains the DD until then.

## 8. Limits of this study

- All ceilings are arithmetic; the 70% bandwidth and the 2.2-5.2 ms (2.0 ms + 0.05 ms per row) latency floor are assumptions, to be replaced by the
  Stage 1 measurements. Real kernels at 1-2 rows per expert and IQ dequant may sit below 70%.
- The expert-union model assumes uniform routing across slots (upper estimate at high C) and Steve's within-stream
  overlap; the real per-layer group counts are on the device (plan counts[0]) and were never exported.
- Acceptance a = 0.85 is calibrated on Strata's coding prompts; chat and sampling accept less.
- B3/B2 tables are from an unclosed trace (watchdog abort): counts may be missing; profiled durations inflate host gaps.
- 27B byte counts (14.5 GB per pass, 151 MB state) are estimates from TensorFold docs and geometry, not a header parse.
- Prefill efficiency transfer from TensorFold's dense 27B to a 512-expert MoE is the largest uncertainty (derated 0.5-0.6).
- Not verified: argument-ABI extraction of SYCL/ESIMD kernels for raw Level Zero loading; Intel host-pointer import on
  compute runtime 26.x; CS cross-device wait latency; whether the X399 board can host both cards on one die at x16.

VERDICT -> The hardware allows roughly 2x Strata at C1, 3-5x at C4, a C16 regime Strata cannot reach, and 5-10x prefill.
Most of the C2/C4 gap is cheap to recover inside Strata (batched streamed-KV attention, window filling, commit/emit/PLE
off the critical path); the rest needs an engine whose host loop, graphs, kernels and card roles are designed for
batched windows on two B70s. Build the harness and kernel library first, gate the engine on a one-window replay, keep
Flash-Next (not the 27B), and leave hardware changes until the software uses the bandwidth it already has.
