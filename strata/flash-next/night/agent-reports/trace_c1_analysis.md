# C1 (and C2) decode trace analysis: strata-hcq8a, STRATA_HC_Q8=1

Analyst: Claude Opus 5.5 (claude-opus-5-5), 2026-10-16. CPU-only analysis of
an existing trace; no GPU was touched.

## Scope and method

CONFIG -> unitrace `-d -h -s` chrome trace of
`/mnt/vm_8tb/b70/results/night-20261016/builds/strata-hcq8a` with
STRATA_HC_Q8=1. Expert split 139/373 per layer (the engine log says "this card
holds 139 experts per layer, card 1 the other 373"). Session: load and capture,
a warm 64-token request, one C1 256-token request (spec4 MTP: 4-row verifier
plus 3 draft steps per window), then one C2 pair of 192-token requests. Raw
trace: `runs/trace-c1/trace/strata-hcq8a.467.json` (1,022,698,038 B).

COMMAND -> One parse under `bin/cpu-run` (38 s, 4.9 GB RSS) into
`/mnt/vm_8tb/b70/results/night-20261016/agents/trace_cache_c1/` (2,037,576
complete events, 4,169,447 flow events, int-ns timestamps). The analysis reads
only the cache (`trace_gap_scripts/analyze2.py`, about 15 s per pass). CSVs go
to `/mnt/vm_8tb/b70/results/night-20261016/agents/trace_c1_csv/` with prefixes
`hcq8_c1_*`, `hcq8_c2_*` and `s136_c2_*`. The S136 trace was re-run through
the same code so the C2 comparison is like-for-like.

Alignment. The trace clock is offset from the client `time.time()` marks by
about 13.2 s, as in S136. Mapped by activity, relative to the first GPU op:

| Stretch | Prefill (eager, ~10% busy) | Decode |
| --- | --- | --- |
| Warm request | 26.0-28.0 s | 28.5-30.5 s |
| C1 | 30.75-32.5 s | 32.77-36.16 s |
| C2 | 36.25-41.5 s (two prompts, serialized) | 42.55-46.6 s, then a solo tail |

Window definitions:
- C1 window = card0 4-column vocab head (`WideQ5K,4`, grid 62080) end ->
  next one's end. There are 78 C4 heads. Windows 1-76 are steady (41.9-44.5 ms,
  no create/close API calls); window 0 (102 ms) is the request start.
- C2 cycle = six-column head end -> next one's end (69 cycles). The 64 steady
  cycles are 3-14, 16-24 and 26-68; cycles 0-2, 15 and 25 contain graph
  capture (69-116 ms).

IMPORTANT, HC_Q8 is a no-op for the HC kernels in this binary. The build diff
(`builds/strata-hcq8a.diff`) only adds BF16 -> Q8_0 conversion at load. The
SYCL `fused_gr_read_multi` (`night-wt/hc/src-ep/sycl/src/kernels/cuda/fused_gr.dp.cpp`,
final line) says: `return false; // SYCL port: ... the Q8_0 reads (S23
STRATA_HC_Q8) are not built`. The verifier HC kernels still read the BF16
weights, so the 0.63 GiB of Q8 copies is dead VRAM.

The trace agrees. In C2, `hc_gr_up_cols` is 29.79 us median vs S136's 29.90,
and `hc_gr_down_sl` is 22.71 vs 22.71, with identical names and grids. All HC
bandwidth below therefore uses BF16 bytes (6.55 MB per 10240x320 matrix).

Profiler caveat (as in S136/S142): host gaps and the per-node eventReset cost
are inflated. Profiled C1 here is 3.31 tokens per 43.13 ms window, i.e. 76.8
tok/s. The common-context baseline is 91 tok/s on strata-s147 (a different
binary), roughly 36.4 ms per window, so about 6-7 ms per window is likely
profiler overhead.

## 0. Key numbers, C1 (76 steady windows, mean per window)

| | card0 | card1 |
| --- | ---: | ---: |
| Window wall | 43.13 ms (min 41.89, p10 42.68, median 43.05, p90 43.79, max 44.48, sd 0.49) | same |
| Compute union | 32.18 ms (74.6%) | 7.61 ms (17.6%) |
| EP-wait spin, not overlapped by own compute | 1.58 ms | 19.88 ms |
| eventReset busy | 1.78 ms (1,934 resets) | 0 |
| Idle (no GPU op) | 7.59 ms | 15.64 ms |
| Kernel launches | 1,988 + 48 ep_wait | 289 + 48 ep_wait |
| Barriers / memcpy | 9 / 0 | 0 / 0 |

Window time distribution (1 ms bins): 41 ms: 1, 42 ms: 32, 43 ms: 39, 44 ms: 4.

## 1. Steady C1 window structure

Card0, in order (mean ms; span p10-p90 in brackets; phases reconcile to 43.13):

| # | Phase | Start | Span | Busy | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Verifier tail: sample_tokens g{4} + 1,766 eventResets (previous verifier replay) | 0.00 | 4.29 [4.23-4.33] | 1.71 | Host sync waits through it |
| 2 | Host gap | - | 1.35 | 0 | |
| 3 | Commit graph (in-order; 98 kernels: 36 gdn_step_commit, 36 gdn_conv_commit, indexer append) | 5.64 | 0.98 [0.89-1.03] | 0.74 | |
| 4 | Host gap + 1 eager command | - | 0.16 | 0 | |
| 5 | MTP draft step 1 (63 kernels + 62 resets) | 6.77 | 1.91 [1.86-1.93] | 1.80 | Includes 1-col vocab head 0.996 ms |
| 6 | MTP draft step 2 | 8.69 | 1.78 | 1.69 | Head 0.995 ms |
| 7 | MTP draft step 3 | 10.47 | 1.78 | 1.69 | Head 0.995 ms |
| 8 | Host gap (PLE staging + card1/card0 verify submits) | - | 2.64 | 0 | Card1 already spinning |
| 9 | Verifier replay, 4 rows x 48 layers + C4 head (1,718 kernels + 48 ep_wait) | 14.90 | 28.23 [27.75-28.69] | 27.92 | Kernel union 26.34; head 1.72 |

Card1: idle 12.99 ms, then its verifier replay spans 28.35 ms (compute 7.61,
spin 19.88, in-graph gaps 0.86), then idle 1.79 ms.

Each MTP step (about 1.78 ms) is a 1-column full-vocab head (0.995 ms, 56%),
sampling/select (`sample_tokens` g{1} 89 us, `row_top_prob_split` 32 us,
`mtp_select` 16 us) and the draft-layer body (~0.65 ms: Q8 wide32 MMVQ, HC,
attention g512 chunk/merge, a small MoE).

Fraction of the window:

| Part | ms | Share |
| --- | ---: | ---: |
| MTP drafter, 3 steps | 5.47 | 12.7% |
| ...of which the 3 draft heads | 2.99 | 6.9% |
| ...of which body and sampling | 2.48 | 5.8% |
| Verifier replay | 28.23 | 65.5% |
| ...of which C4 head | 1.72 | 4.0% |
| Verifier reset tail | 4.29 | 9.9% |
| Commit | 0.98 | 2.3% |
| Host gaps | 4.16 | 9.6% |

- Drafter / verifier-replay ratio is 0.19.
- Counting the reset tail (part of the verifier graph), the verifier is 75.4%
  of the window.

Verifier per layer, C1 (`hcq8_c1_verifier_per_layer.csv`):

| Per layer | GDN | Attention |
| --- | ---: | ---: |
| Span | 543 us | 575 us |
| Card0 pre-push dense part | 350 us | 391 us |
| Card0 local routed work after push | 106 us | 112 us |
| Card1 chain (copy -> send) | 175 us (gaps 12.6, send 22.0) | 160 us |
| Card0 waits for card1 | 36 us | 23 us |
| Card1 waits for card0 | 418 us | 404 us |

Card0 waits on card1 in 84% of layers, 1.58 ms per window in total. Push
latency is 3.8 us. Attention now uses one batched launch per layer:
`qsa_attn_chunk` g{33,2,4}, 44 us, 12 launches per window. The S136 C2 path
used 6 per-row launches; C1 is not in batch-record mode.

## 2. Per-window family census, C1, with C2 and S136 C2

The C1 and C2 columns come from this trace. C2 values are per cycle (6
verifier rows plus the paired 2-depth MTP); C1 values are per window (4 rows
plus 3 MTP steps). n = launches.

| Card0 family | C1 n | C1 us | C2 n | C2 us | S136 C2 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| HC down/up projections (BF16 read) | 212 | 4,504 (verifier 4,179) | 218 | 5,538 | 5,526 |
| Vocab heads | 4 | 4,703 (3 x draft 2,986 + C4 1,717) | 3 | 4,677 | 4,676 |
| EP wait spin (exclusive) | 48 | 3,405 (1,581) | 48 | 4,069 (1,921) | 3,196 (1,254) |
| MMVQ Q6K | 71 | 3,261 | 71 | 3,845 | 3,827 |
| Routed GU (card0) | 48 | 2,927 | 48 | 4,046 | 4,365 |
| Routed down (card0) | 48 | 2,046 | 48 | 2,784 | 3,099 |
| eventReset | 1,934 | 1,776 | 2,434 | 2,231 | 2,242 |
| MMVQ IQ4XS | 67 | 1,736 | 67 | 2,078 | 2,066 |
| MMVQ Q4K | 45 | 1,733 | 45 | 2,320 | 2,311 |
| MMVQ IQ3S (type 21) | 55 | 1,639 | 55 | 1,941 | 1,929 |
| GDN (commit + verifier) | 180 | 1,603 (646 + 958) | 324 | 2,794 | 2,819 |
| HC norm/reduce | 212 | 1,133 | 218 | 1,192 | 1,193 |
| BF16 GEMV (router g512, indexer, gates) | 128 | 1,034 | 130 | 1,221 | 1,212 |
| Attention chunk/merge/select | 54 | 943 (MTP 278) | 296 | 3,445 | 3,417 |
| MMVQ Q5K (dense) | 14 | 784 | 14 | 1,005 | 1,005 |
| MMVQ IQ4NL (shared down) | 48 | 667 | 48 | 803 | 797 |
| Q8 activation quantize | 267 | 659 | 277 | 696 | 707 |
| EP plan/group | 108 | 576 | 112 | 674 | 677 |
| MMVQ Q8 (MTP layer) | 28 | 549 | 38 | 736 | 737 |
| Sampling/select | 13 | 502 | 17 | 639 | 640 |
| Router top-k | 51 | 388 | 52 | 398 | 400 |
| Attention prep (norm/rope/kv/indexer) | 105 | 343 | 293 | 685 | 695 |
| EP push | 50 | 178 | 50 | 185 | 188 |
| Barrier | 9 | 138 | 10 | 167 | 164 |
| MoE combine/shared gate | 102 | 132 | 104 | 158 | 160 |

| Card1 family | C1 n | C1 us | C2 us | S136 C2 us |
| --- | ---: | ---: | ---: | ---: |
| EP wait spin | 48 | 19,883 | 26,487 | 26,797 |
| Routed GU | 48 | 3,619 | 4,770 | 4,614 |
| Routed down | 48 | 2,532 | 3,326 | 3,263 |
| EP copy/send | 97 | 1,108 | 1,573 | 1,481 |
| EP plan/group | 96 | 351 | 397 | 397 |

Full per-phase rows: `hcq8_c1_family_census.csv`, plus
`hcq8_c1_kernel_census.csv` for each kernel and grid.

## 3. Achieved bandwidth (weight bytes inferred)

Shape inference:
- MMVQ grid maps to output rows as M = 4 x grid (checked against known shapes:
  2560 -> 10240 qkv, 1536 -> 6144 gate, 3072 -> 12288 attn_q, 640 -> 2560
  out-proj, 128 -> 512 k/v, 160 -> 640 shared gate/up, 62080 -> 248320 head).
  The exception is `mmvq_multi_n<Q20>`, where M = 2 x grid.
- K is 6144 for out-projections, 640 for shared down, 2560 otherwise.
- Bytes per element come from the engine's byte table:

| Format | Bytes per block | Format | Bytes per block |
| --- | --- | --- | --- |
| Q4_K | 144 B / 256 | IQ3_S (ggml 21) | 110 B / 256 |
| Q5_K | 176 B / 256 | IQ2_XXS (16) | 66 B / 256 |
| Q6_K | 210 B / 256 | IQ2_XS (17) | 74 B / 256 |
| IQ4_XS | 136 B / 256 | IQ3_XXS (18) | 98 B / 256 |
| IQ4_NL | 18 B / 32 | IQ2_S (22) | 82 B / 256 |
| Q8_0 | 34 B / 32 | Q2_0 (42) | 18 B / 64 |
| BF16 | 2 B | | |

Percentages are against 608 GB/s. Durations are medians.

Verifier, C1 (4 columns) vs C2 (6 columns), time-weighted by role (card0):

| Role (all formats) | Weight MB per window | C1 ms | C1 GB/s (%) | C2 ms | C2 GB/s (%) | ms at 85% roofline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All dense MMVQ (300 launches) | 1,871 | 9.84 | 190 (31%) | 12.01 | 156 (26%) | 3.62 |
| Out-proj 6144->2560 | 543 | 2.66 | 204 (34%) | 3.19 | 170 (28%) | 1.05 |
| qkv 2560->10240 | 626 | 2.43 | 258 (42%) | 2.98 | 210 (35%) | 1.21 |
| gate 2560->6144 | 318 | 1.82 | 175 (29%) | 2.21 | 144 (24%) | 0.62 |
| Shared expert gate/up/down (144 launches) | 125 | 1.71 | 73 (12%) | 2.12 | 59 (10%) | 0.24 |
| attn_q 2560->12288 | 237 | 0.99 | 239 (39%) | 1.22 | 194 (32%) | 0.46 |
| k/v 2560->512 | 23 | 0.23 | 97 (16%) | 0.29 | 79 (13%) | 0.04 |
| HC down+up, BF16 (194 launches) | 1,271 | 4.18 | 304 (50%) | 5.09 | 250 (41%) | 2.46 (1.31 if Q8 were actually read) |
| Router GEMV 2560->512 BF16 | 157 | 0.60 | 262 (43%) | 0.71 | 221 (36%) | 0.30 |
| Vocab head C4 / C6 | 437 | 1.72 | 255 (42%) | 2.26 | 193 (32%) | 0.85 |
| Draft head, 1 column (MTP) | 437 | 0.995 each | 439 (72%) | 1.21 (2-col) | 362 (60%) | 0.85 |

Notable individual kernels (C1 verifier, median):

| Kernel | Role | Time | GB/s |
| --- | --- | ---: | ---: |
| `mmvq_q6k_wide<4>` g640 | out-proj | 52.1 us | 248 |
| `mmvq_q6k_wide<4>` g2560 | qkv | 70.3 us | 306 |
| `mmvq_wide<WideQ4K,4>` g2560 | qkv | 74.5 us | 198 |
| `mmvq_wide<WideQ4K,4>` g640 | out-proj | 63.5 us | 139 |
| `mmvq_wide<WideQ5K,4>` g640 | out-proj | 68.3 us | 158 |
| `mmvq_multi_nc<21>` (IQ3_S) g1536 | gate | 71.7 us | 94 |
| `mmvq_multi_nc<21>` (IQ3_S) g640 | out-proj | 95.0 us | 71 |
| `mmvq_multi_nc<21>` (IQ3_S) g160 | shared gate/up | 12.4 us | 57 |
| `mmvq_wide32<IQ4NL,4>` g640 | shared down | 12.4 us | 74 |
| `mmvq_multi_n<Q20>` g1280 | shared down | 21.3 us | 43 |
| `hc_gr_up_cols` BF16 | HC up | 24.0 us | 274 |
| `hc_gr_down_sl` BF16 | HC down | 19.2 us | 342 |

For reference, the MTP-layer `wide32<Q8,1>` g3072 runs at 552 GB/s (91%):
already at roofline.

What the bandwidth numbers show:
1. The same weight bytes cost more with more columns: dense MMVQ is 9.84 ms
   at 4 columns and 12.01 ms at 6. HC goes 24.0 -> 29.8 us (up) and
   19.2 -> 22.7 us (down) from 4 to 6 rows; 1-row HC in the MTP layer is
   18.8/15.2 us (350/431 GB/s).
2. These kernels are therefore latency/issue-bound, not weight-bandwidth-bound,
   consistent with the native_mmvq source comment.
3. IQ3_S dense (ggml type 21) and every small shared-expert GEMV run at
   7-17% of roofline.

Routed experts (estimate, NOT measured bytes). This assumes each card owns
half of the rows x 10 entries ("traffic dealt evenly") and that each owned
entry reads its own expert (no de-duplication):

| Card | C1 GU | C1 down | C2 GU | C2 down |
| --- | --- | --- | --- | --- |
| card0 | 257-468 GB/s | 217 (Q2_0) / 436 (IQ4NL) | 266-516 | 235 / 488 |
| card1 | 235-359 | 183 / 339 | 253-412 | 205 / 385 |

Card1 runs the same formats 20-30% slower per launch than card0. Repeated
experts across rows would lower the true DRAM bytes, so read these as upper
bounds on the bytes actually streamed. Q2_0 down is the weakest at 183-235
GB/s.

Full table: `hcq8_c1_bandwidth.csv` / `hcq8_c2_bandwidth.csv`.

## 4. Idle / gap accounting, C1 (per window)

Card0 idle (7.59 ms):
- Host gaps between submissions: 4.16 ms (2.64 before the verifier, 1.35
  before commit, 0.16 around the eager command).
- Gaps inside the reset storm: 2.58 ms (1,761 gaps under 2 us).
- Commit in-graph gaps: 0.24 ms (97 gaps, median 2.48 us; in-order graph).
- MTP in-graph gaps: 0.30 ms (median 0.21 us).
- Verifier in-graph gaps: 0.31 ms (1,415 gaps, all under 2 us, median 0.21).

Card1 idle (15.64 ms): no work submitted for 14.78 ms (12.99 before its replay,
1.79 after it), plus 0.86 ms of in-graph gaps (336 gaps, median 2.52 us).

Card0 kernel durations: 1,988 launches per window. 912 launches run under
5 us (2.38 ms), 1,256 under 10 us (4.91 ms, 15% of card0 kernel time), and 4
run over 500 us (the heads, 4.70 ms).

Cross-card (whole window, ms):

| | card1 compute | card1 spin | card1 idle |
| --- | ---: | ---: | ---: |
| card0 compute | 6.14 | 17.71 | 8.32 |
| card0 spin | 1.44 | 0.00 | 0.14 |
| card0 reset-only | 0 | 0 | 1.78 |
| card0 idle | 0.04 | 2.17 | 5.39 |

| Who computes | ms | Share |
| --- | ---: | ---: |
| Both cards | 6.14 | 14.2% |
| card0 only | 26.04 | 60.4% |
| card1 only | 1.48 | 3.4% |
| Neither | 9.48 | 22.0% |

"Neither" is 5.39 ms with both idle (host), 1.78 ms of card0 resets with card1
idle, 2.17 ms of card0 idle while card1 spins (card1's replay is submitted
first), and 0.14 ms of card0 spinning with card1 idle.

## 5. Ten largest single costs per C1 window (card0 timeline)

| Rank | Cost | ms per window | Kind / note |
| ---: | --- | ---: | --- |
| 1 | HC down/up projections, still BF16 (194 verifier + 18 MTP launches) | 4.50 | Kernel; 50% of BF16 roofline; Q8 path unbuilt |
| 2 | Verifier-tail reset storm (1,766 resets + sample) | 4.29 span | Overhead: 1.62 ms busy + 2.58 ms gaps; profiler-inflated |
| 3 | MMVQ Q6K dense (71 launches) | 3.26 | Kernel; 248-314 GB/s |
| 4 | MTP draft heads (3 x 1-col full vocab) | 2.99 | Kernel; 72% of roofline already |
| 5 | Routed GU on card0 (48) | 2.93 | Kernel |
| 6 | Host gap before verifier (PLE + submits) | 2.64 | Idle; profiler-inflated |
| 7 | Routed down on card0 (48) | 2.05 | Kernel |
| 8 | MMVQ IQ4XS (67) | 1.74 | Kernel |
| 9 | MMVQ Q4K (45) | 1.73 | Kernel; 139-201 GB/s |
| 10 | C4 verifier head | 1.72 | Kernel; 42% of roofline |

Next in line: MMVQ IQ3S 1.64 (57-100 GB/s), GDN 1.60, EP wait exclusive 1.58,
host gap before commit 1.35, HC norm/reduce 1.13, BF16 GEMV 1.03.

All dense MMVQ together is the largest group: 9.84 ms per window, 37% of the
verifier's kernel union, at 31% of roofline. At 85% it would take 3.6 ms.

## 6. C2 pair windows in this trace vs S136 (same method)

| | HC_Q8 trace, 64 cycles | S136, 16 cycles | Delta |
| --- | ---: | ---: | ---: |
| Cycle mean / median | 56.67 / 56.58 ms (sd 0.86) | 57.14 / 57.21 (sd 0.26) | -0.48 ms (-0.8%) |
| Verifier replay span | 37.68 ms | 37.60 ms | +0.08 |
| Host gap before verifier | 3.06 ms | 3.55 ms | -0.49 (host side) |
| Card0 compute | 42.06 ms | 42.65 ms | -0.59 |
| Card0 EP wait-only | 1.92 ms | 1.25 ms | +0.67 |
| Card0 routed GU + down | 6.83 ms | 7.46 ms | -0.63 |
| Card1 routed GU + down + send | 9.67 ms | 9.36 ms | +0.31 |
| HC down/up | 5.54 ms | 5.53 ms | identical (Q8 unused) |
| All other card0 families | within +-30 us | | |
| Card0 waits on card1 | 82% of layers | 76% of layers | |
| Overlap: both / c0 only / c1 only / neither | 8.25 / 33.81 / 1.82 / 12.79 ms | 8.57 / 34.08 / 1.18 / 13.31 ms | |

The expert split is 139/373 here vs 149/363 in S136. That moves about 0.6 ms
of routed work from card0 to card1, and card0 waits correspondingly longer,
so the GPU-side cycle is unchanged. The -0.48 ms is entirely host side.

VERDICT -> On this binary, HC_Q8 changes nothing on the GPU. Every C2 GPU
family matches S136 within noise, apart from the split shift. The real HC_Q8
win still needs a SYCL Q8_0 read path in `fused_gr` (`gr_down`/`gr_up`).
Bounded estimate: HC is 4.18 ms per C1 window and 5.09 ms per C2 cycle. At
the current 304/250 GB/s effective rate, halving the bytes saves roughly
2.1/2.5 ms; at 85% of the Q8 bytes, roughly 2.9/3.8 ms.

## 7. Leads ranked by ms per C1 window (estimates; overlap noted)

1. Dense MMVQ efficiency. 9.84 ms at 31% of roofline; the 85% target is 3.6 ms,
   so up to about 6 ms. Cost grows with columns, so the work is issue/latency
   bound. The worst offenders are IQ3_S multi_nc (57-100 GB/s), the 144
   shared-expert GEMVs per window (73 GB/s, launch-floor-bound, fusable with
   gate/up pairs), and Q4K/Q5K out-proj (139-158 GB/s).
2. HC Q8 read path (not built in SYCL): about 2-3 ms. Overlaps nothing above.
3. Verifier reset storm (1,766 resets): 4.29 ms profiled; unprofiled cost
   unknown (S136 analysis inferred at most ~1 ms in C2).
4. C4 head at 42%: about 0.9 ms if it reached 85%.
5. EP: card0 waits 1.58 ms per window on card1, whose routed GU/down kernels
   run 20-30% slower than card0's and whose send is 22 us per layer. Moving
   about 10 experts per layer back to card0 (149/363 behaved better on card0
   wait) or pre-reducing the send could save up to ~1.5 ms.
6. Draft heads: 2.99 ms at 72% of roofline. Not a bandwidth target; only a
   reduced draft vocab would cut it.

## Limits

- Profiled run: host gaps and the reset storm are inflated.
- One C1 request (76 windows, one process).
- Routed bytes are assumption-based (50% ownership, no de-duplication).
- Shape inference comes from grid and kernel names, checked against known
  model shapes but not against tensor pointers.
- Clock alignment to the client marks is by activity pattern (offset about
  13.2 s).
