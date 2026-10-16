# S136 C2 decode trace: GPU idle, gaps, tiny kernels, cross-card overlap

Analyst: Claude Opus 5.5 (claude-opus-5-5), 2026-10-16. CPU-only analysis of an
existing trace; no GPU was touched.

## Scope, inputs, method

CONFIG -> Existing S136 unitrace (`-d -h -s`, chrome) of one instrumented C2
pair on the S133 executable: EP149/363, MTP depth2 + paired head, spec4,
all candidate flags 0. Raw trace
`.../current-mtp-profile-s136/run/trace/strata-s133.467.json` (279,154,635 B).

COMMAND -> One parse under `bin/cpu-run` (10 s, 1.4 GB RSS) into an int-ns cache
`/mnt/vm_8tb/b70/results/night-20261016/agents/trace_cache/{events.npz,names.json}`
(749,782 complete events + 971,073 flow events). The ns integers come straight
from the printed decimal strings, so no double rounding at epoch scale. All
later analysis reads only the cache (about 7 s per pass). Scripts:
`/mnt/vm_8tb/b70/results/night-20261016/agents/trace_gap_scripts/` (`parse_once.py`,
`tl.py`, `analyze.py`, `gap_catalog.py`, `gap_by_phase.py`, `layer_seq.py`).
CSVs: `/mnt/vm_8tb/b70/results/night-20261016/agents/trace_gap_csv/`.

Definitions:
- Cycle = card0 six-row vocab head (`WideQ5K,6`, grid 62080) end -> next one's end.
  There are 22 such heads, so 21 cycles.
- Steady set = cycles 3-14 and 17-20 (16 cycles, 56.54-57.60 ms, no
  create/close API calls). Cycles 0-2 and 15-16 contain command-list/graph
  creation: 15/16 take 69.4/69.9 ms, about 12 ms of capture inside the cycle.
- Host submit time and API for every GPU op come from the `Flow_H2D` flow
  events. All 243,982 GPU ops matched a host API call.
- Cross-card clock check (768 layer instances): card0 `ep_push` end -> card1
  first op is 3.0-4.75 us and never negative. card1 `ep_send_rows` end -> card0
  `ep_wait` end is 1.4-2.9 us when card0 was waiting. The two device timelines
  agree to within about 1.5 us.
- "compute" = kernels other than the `ep_wait` spin, plus memcpy/fill.
  `ep_wait` is a single-work-item spin kernel. eventReset and barrier are
  command-stream ops.

Profiler caveat (important). S142 ran the same config without unitrace. Its
whole window was 46.47 ms vs 57.18 ms here, for these medians:

| Stage | profiled | unprofiled |
| --- | ---: | ---: |
| verify_submit | 3.22 ms | 0.067 ms |
| pair_submit | 1.95 ms | 0.046 ms |
| commit | 2.93 ms | 1.59 ms |
| verify_queue_wait | 43.87 ms | 38.64 ms |

So the host-gap and reset-storm durations below are inflated. In-graph kernel
durations, op counts, order and the cross-card structure are the robust part.
Where it matters, estimates are given against the unprofiled ~46.5 ms cycle.

## 0. Key numbers (16 steady cycles, mean per cycle)

| | card0 (primary) | card1 (peer) |
| --- | ---: | ---: |
| Wall | 57.14 ms (56.54-57.60) | 57.14 ms |
| Busy union, all GPU ops | 46.15 ms | 36.55 ms |
| Compute union | 42.65 ms (74.6%) | 9.75 ms (17.1%) |
| ep_wait spin, not overlapped by own compute | 1.25 ms | 26.80 ms |
| eventReset (exclusive) | 2.24 ms | 0 |
| Idle (no GPU op at all) | 10.99 ms | 20.59 ms |
| Kernel launches | 2,654 + 48 ep_wait | 289 + 48 ep_wait |
| eventReset / barrier / memcpy / fill | 2,435 / 10 / 2 / 0 | 0 / 0 / 0 / 0 |

Card0 is the critical device. Inside its 37.6 ms verifier replay it does
compute 95.6% of the time. Graph launch gaps there are tiny (median 0.21 us,
0.40 ms per cycle in total). The idle time is outside the verifier:

- host-submission gaps, 6.25 ms profiled;
- a 2,226-reset storm at the verifier tail, 5.43 ms span profiled;
- in-graph gaps in the in-order commit graph (0.58 ms) and in MTP (0.46 ms);
- 1.25 ms of EP waiting on card1.

Card1 has no work queued for 19.7 ms per cycle and spins on `ep_wait` for
26.8 ms.

## 1. Per-device per-cycle accounting and phases

Card0 phase timeline (mean of 16 cycles; reconciles to 57.14 ms):

| # | Card0 phase (submission) | Span ms | GPU busy ms | Ops | Kernels | Resets | Gap before ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Verifier tail: sample + barrier + 2,226 eventResets (previous verifier replay) | 5.434 | 2.144 | 2,229 | 1 | 2,226 | 0 |
| 2 | Host gap (host waits through #1, then commit prep) | - | 0 | - | - | - | 1.772 |
| 3 | Commit graph (in-order; GDN step/conv commit, indexer append) | 2.092 | 1.511 | 244 | 244 | 0 | - |
| 4 | Host gap + 2 eager D2D slot copies | 0.004 | 0.004 | 2 | 0 | 0 | 0.775 |
| 5 | MTP bodies, 4 replays (2 depths x 2 owners) | 2.693 | 2.317 | 431 | 213 | 209 | 0.190 |
| 6 | MTP C2 heads + sample/select, 2 replays | 3.016 | 2.978 | 18 | 18 | 0 | 0.013 |
| 7 | Host gap (PLE stage + verify submit; card1 already spinning) | - | 0 | - | - | - | 3.550 |
| 8 | Verifier replay (48 layers + C6 head) | 37.603 | 37.198 | 2,226 | 2,178+48 | 0 | - |

Card1 has one submission per cycle: its verifier replay. It is idle 17.36 ms
before it, then the replay spans 37.43 ms (compute 9.75, spin 26.80, in-graph
gaps 0.88), then idle 2.36 ms to cycle end.

Class times per cycle (sum of durations / exclusive union). Card0: kernels
42.69 / 42.65, ep_wait 3.20 / 1.25, eventReset 2.24 / 2.24, barrier 0.16 /
0.005, memcpy 0.004. Card1: kernels 9.75, ep_wait 26.80. Per-cycle rows are in
`cycle_device_accounting.csv` and `cycle_phases.csv`, which also cover
capture cycles 15/16.

Host side (profiled, from L0 API events and S136 CPU markers):
- The host blocks in `zeCommandListHostSynchronize` until the reset storm ends.
- It then spends about 1.3 ms with no L0 calls before the commit submit.
- Between the second draft head and the verifier submit (3.55 ms) it does PLE
  staging (0.71 ms), the card1 replay append (44 us), about 2.1 ms with no L0
  calls (profiler perturbation per S142), and the card0 append (197 us).

## 2. Gap distribution between consecutive GPU ops

View A treats every GPU command (including resets and the spin) as busy, so a
gap means the engine had nothing. Per cycle: count / idle ms.

| Bucket | card0 count | card0 ms | card1 count | card1 ms |
| --- | ---: | ---: | ---: | ---: |
| 0-2 us | 4,503 | 4.043 | 18.8 | 0.035 |
| 2-5 us | 248.5 | 0.611 | 317.2 | 0.840 |
| 5-10 us | 4.0 | 0.025 | 0 | 0 |
| 10-20 us | 1.2 | 0.021 | 0.1 | 0.001 |
| 20-50 us | 1.4 | 0.037 | 0 | 0 |
| 50-100 us | 1.0 | 0.067 | 0 | 0 |
| 100-500 us | 1.0 | 0.155 | 0 | 0 |
| >500 us | 3.0 | 6.030 | 2.0 | 19.713 |
| Total | 4,763 | 10.99 | 338 | 20.59 |

The same view A idle, attributed by location (`gap_histogram_by_phase_view_A.csv`):

| Device, location | Idle ms per cycle | Character |
| --- | ---: | --- |
| card0 host gaps (before verifier 3.55, before commit 1.77, slot copies 0.93) | 6.25 | next op submitted after the gap began: host-bound; card1 is idle or spinning |
| card0 inside reset storm (2,225 gaps of ~1.46 us between resets) | 3.29 | command-stream only |
| card0 commit graph (in-order; 243 gaps, median 2.38 us) | 0.58 | dispatch/dependency |
| card0 MTP replays (incl. 209 resets, 8 barriers) | 0.46 | dispatch, reset, barrier |
| card0 verifier graph (1,862 gaps, every one < 2 us, median 0.21) | 0.40 | dispatch |
| card1 no work submitted (17.36 before replay + 2.36 after) | 19.71 | waiting on card0/host by design |
| card1 verifier graph (in-order; 336 gaps, median 2.58 us) | 0.88 | dispatch; on the EP critical path |

View B counts only compute as busy. Card0's non-compute time is 14.49 ms per
cycle: 11.38 ms in three gaps over 500 us, 0.91 ms at 100-500 us (EP-wait
windows), 0.64 ms at 50-100 us, and 1.13 ms under 5 us.

Top 30 gaps. Individually, the top 30 on each device are the same two or
three structural gaps repeating in each of the 16 cycles (full lists by
device and view in `top_gaps.csv`). Grouped by structural position (view A,
`gap_catalog_view_A.csv`, 40+ rows):

| Dev | Position (before -> after) | Per cycle | Mean us | Max us | Host-submitted after gap start | Other card during gap |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c1 | cycle start -> ep_bump (replay start) | 1 | 17,356 | 17,969 | 100% | card0 compute 39%, idle 48%, reset 13% |
| c0 | 2nd mtp_select -> ep_bump (verifier start) | 1 | 3,550 | 3,975 | 100% | card1 spin 61% (replay submitted first), idle 39%: host-bound |
| c1 | last ep_send_rows -> cycle end | 1 | 2,358 | 2,380 | n/a | card0 compute 100% (C6 head, epilogue) |
| c0 | reset-storm end -> commit first op (copy_i32_from_mapped) | 1 | 1,772 | 1,875 | 100% | card1 idle 100%: host-bound |
| c0 | commit end (copy_indexed) -> slot memcpy | 1 | 708 | 741 | 100% | card1 idle 100%: host-bound |
| c0 | slot memcpy -> MTP first op | 1 | 155 | 162 | 100% | card1 idle |
| c0 | commit: gdn_conv_commit -> gdn_step_commit | 59.6 | 2.47 | 3.2 | 0% | card1 idle |
| c1 | ep_send_rows -> ep_wait | 46.4 | 2.97 | 4.7 | 0% | card0 compute 52%, spin 45% |
| c1 | ep_copy_uncached -> resident_plan_ep | 47.9 | 2.70 | 3.5 | 0% | card0 compute |
| c0 | commit: gdn_step_commit -> gdn_conv_commit | 49.9 | 2.44 | 3.5 | 0% | card1 idle |
| c1 | ep_wait -> ep_copy_uncached | 42.2 | 2.55 | 3.4 | 0% | card0 compute |
| c0 | commit: indexer_append -> indexer_append | 40.6 | 2.43 | 3.2 | 0% | card1 idle |
| c1 | native_expert_grouped -> routed_down | 28.8+17.6 | 2.6 | 3.6 | 0% | card0 compute |
| c1 | routed_down -> ep_send_rows | 27.3+16.7 | 2.6 | 3.6 | 0% | card0 compute 60-74%, spin 26-40% |
| c0 | slot memcpy -> slot memcpy | 1 | 66.7 | 71.4 | 100% | card1 idle |
| c0 | MTP: eventReset -> eventReset | 20.1 | 2.15 | 2.5 | 0% | card1 idle |
| c1 | resident_plan_ep -> routed_gu (all GU variants) | 43.6 | 2.55 | 17.2 | 0% | card0 compute |
| c0 | MTP: barrier -> next | 8 | 2.9-6.3 | 6.9 | 0% | card1 idle |

The largest individual gaps under 500 us (`top30_gaps_below_500us_view_A.csv`)
are all the host-side slot-copy gaps (150-162 us and 63-71 us). Card0 has no
verifier gap of 2 us or more.

## 3. Kernel duration distribution and per-launch overhead

| Duration | card0 launches | card0 ms | % time | % launches | card1 launches | card1 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <2 us | 523.8 | 0.808 | 1.9 | 19.7 | 46.7 | 0.073 |
| 2-5 us | 904.3 | 2.779 | 6.5 | 34.1 | 50.3 | 0.134 |
| 5-10 us | 279.4 | 2.034 | 4.8 | 10.5 | 48.0 | 0.271 |
| 10-20 us | 402.1 | 5.645 | 13.2 | 15.1 | 2.7 | 0.046 |
| 20-50 us | 325.7 | 9.761 | 22.9 | 12.3 | 50.0 | 1.559 |
| 50-100 us | 186.8 | 13.640 | 32.0 | 7.0 | 71.8 | 5.479 |
| 100-500 us | 29.4 | 3.344 | 7.8 | 1.1 | 19.5 | 2.194 |
| >500 us | 3.0 | 4.676 | 11.0 | 0.1 | 0 | 0 |

`ep_wait` is excluded from this table. The >500 us bucket is the three
vocab heads: 2 x C2 at 1.21 ms and 1 x C6 at 2.26 ms.

On card0, 1,428 launches per cycle (54%) run under 5 us and take 3.59 ms;
1,708 run under 10 us and take 5.62 ms. Card0 p1 kernel duration is 0.94 us
(minimum 0.52); the smallest families have a median of about 0.94-1.15 us.

Back-to-back dependent launch gaps inside a replay (next start minus running
end, same submission):

| Graph | Pairs per cycle | min | p5 | median | p95 us | Sum ms per cycle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| card0 verifier (out-of-order graph, resets) | 1,914 | 0.00 | 0.04 | 0.21 | 0.40 | 0.404 |
| card0 MTP (out-of-order) | 213 | 0.00 | 0.04 | 0.23 | 2.11 | 0.080 |
| card0 commit (in-order, no resets) | 243 | 1.56 | 1.88 | 2.38 | 2.98 | 0.582 |
| card1 verifier (in-order, no resets) | 336 | 1.61 | 1.98 | 2.58 | 3.29 | 0.877 |

Estimated fixed cost per launch:
- Out-of-order graph: ~1.1-1.2 us, i.e. a 0.21 us gap plus a ~0.9 us floor
  for a trivial kernel. The graph pays a separate ~2.4 us per node later in
  the reset storm (profiled).
- In-order graph: ~3.7-4.0 us, i.e. a ~2.4-2.6 us gap plus a ~1.35 us floor.
- Both per-node costs (in-order gap, out-of-order reset) are plausibly
  profiler-inflated, because they track the S142 deltas. The relative cost of
  in-order vs out-of-order WITHOUT the profiler is unmeasured. Decide it with
  a small unprofiled microbench before changing graph style; see 6.D.

## 4. Per-forward / per-layer census and fusible chains

Verifier, card0, per layer type (mean us; 36 GDN + 12 attention layers; the
families are name-based roles):

| Per layer | GDN layer | Attention layer |
| --- | ---: | ---: |
| Span (combine to combine) | 675.6 | 912.8 |
| card0 kernels | 35.6 | 74.0 |
| Pre-push dense part (card1 spinning) | 428.0 | 673.8 |
| Card0 local routed MoE, push end -> ep_wait | 159.6 | 160.1 |
| Card0 waits for card1 (ep_wait-only) | 27.8 | 21.2 |
| Card0 in-graph idle | 6.5 | 14.0 |
| HC norm/down/reduce/up (8 launches) | 127.3 | 127.2 |
| Dense MMVQ (Q4K/Q5K/Q6K/IQ4XS/IQ4NL) | 203.6 | 204.8 |
| GDN (conv_l2 x2, ab, step_norm x2) | 42.3 | - |
| Attention select/chunk/merge (6 rows x 4 kernels) | - | 254.1 |
| Attention prep (norm/rope, kv_append x6, indexer x6) | - | 36.6 |
| Q8 activation quantize | 12.9 | 12.8 |
| Routed GU / down on card0 | 90.1 / 65.2 | 93.3 / 62.5 |
| Shared expert (mmvq multi etc, overlaps ep_wait) | 50.9 | 26.1 |
| Router logits bf16 GEMV / top10 / plan+group / push | 21.5 / 7.7 / 9.6 / 4.0 | 32.9 / 7.7 / 9.6 / 3.8 |
| card1 chain (copy start -> send end) | 220.2 | 203.2 |
| ...of which card1 in-graph gaps | 12.8 | 13.1 |
| ...of which ep_send_rows | 29.7 | 27.7 |

Verifier totals per cycle (card0):

| Family | ms |
| --- | ---: |
| Dense MMVQ | 9.79 |
| HC | 6.11 |
| Routed GU | 4.36 |
| Routed down | 3.10 |
| Attention select/chunk/merge | 3.05 |
| Shared-expert MMVQ multi | 2.15 |
| GDN | 1.52 |
| BF16 GEMV | 1.17 |
| Q8 quantize | 0.62 |
| Attention prep | 0.44 |
| Router top-k | 0.37 |
| EP plan | 0.46 |
| EP push | 0.19 |

The verifier also has an epilogue of 2.32 ms, mostly the C6 head at 2.26 ms.

Outside the verifier:
- MTP bodies: HC 0.55 ms, MMVQ Q8 0.74 ms, and MTP attention at 0.37 ms
  (chunk g512 at 47 us plus merge g24 at 45 us, x4).
- MTP heads: 2.42 ms of vocab GEMV plus 0.55 ms of sampling/select.
  `sample_tokens` runs on a single work-group (g{1}) at 89 us x4.
- Commit: gdn_step_commit 72 x 16.1 us = 1.16 ms, indexer append 72 x 2.3 us,
  conv commit 72 x 1.9 us.

Representative layer sequences are in `representative_layers_cycle5.txt`; full
per-layer rows are in `verifier_per_layer.csv`, and kernel-plus-grid rows by
phase are in `kernel_census_by_phase.csv`.

Notable per-layer patterns:
- Attention runs one row at a time in C2 batch mode. Each attention layer
  launches (scores, topk, chunk, merge) six times, once per verifier row:
  `verify.cpp` `batch_rec_` branch, `qsa_decode_attn_batch(..., 1, cs)`. Each
  `attn_chunk` launch is grid {33,2,1} x 256, i.e. 528 SIMD32 threads or
  about 26% of the 2,048 hardware thread slots. The row takes 42.7 us if it is
  the first row of a slot and about 30.6 us otherwise. The kernel already has
  a z-batched form ("query blockIdx.z, with its own q row, selection, step and
  scratch", used elsewhere with z = 2/4/32). The non-batch verify path already
  passes n rows sharing one slot's pools.
- Per layer, card0 does HC (4 kernels) -> attention/GDN -> HC (4 kernels) ->
  router chain (bf16 GEMV, top10, single-WG `resident_plan_ep` 6.3 us,
  quantize rows, single-WG `ep_push`) -> own routed GU/down -> `ep_wait`
  overlapped with the shared expert (7 kernels) -> combine.
- `ep_send_rows` (card1) copies one full fp32 hidden row per peer-owned
  (token, expert) entry; the grid is rows x top-10 = 60. Card0 combines
  afterwards. Send takes 12.5-46.7 us (median 29.5) and correlates with
  card0's wait.

Chains of 3 or more consecutive small kernels (card0; same submission;
internal gaps < 5 us):

| Threshold | Chains per cycle | Kernels | Kernel time | Internal gaps |
| --- | ---: | ---: | ---: | ---: |
| <5 us | 145.3 | 716 | 1.63 ms | 0.33 ms |
| <10 us | 216.8 | 1,049 | 3.37 ms | 0.39 ms |

Card1 has none. The top <10 us signatures (`small_kernel_chains.csv`):

| Chain | Per cycle | Kernel us | Gap us |
| --- | ---: | ---: | ---: |
| router_top10 > resident_plan_ep > quantize_q8_1_rows > ep_push | 48 | 991 | 28 |
| qsa_attn_merge > qsa_block_scores > qsa_block_topk (per-row loop) | 60 | 446 | 23 |
| qsa_rms_norm_rope > copy_from_mapped x2 > kv_append_q8_step x6 > indexer_append_steps x6 | 10-11.4 | 332-380 | 28-32 |
| bf16_gemv (shared gate) > sigmoid_scale > moe_combine > hc_norm | 20.9 | 298 | 18 |
| commit: copy_from_mapped > indexer_append x3 > copy > indexer_append x3 > gdn_conv_commit (in-order) | 11 | 204 | 216 |
| gdn_conv_l2 x2 > gdn_ab | 16.2 | 220 | 7 |
| qsa_attn_merge > gate_apply > quantize_q8_1 | 12 | 98 | 5 |

Merging every <10 us chain into one launch removes about 832 launches. At
~1.1 us each that is at most ~0.9 ms per cycle; much of it overlaps items
6.A and 6.C.

## 5. Cross-card overlap and EP exchange

Whole cycle, mean ms (W = compute, S = spin-only, R = reset/barrier-only,
I = idle):

| | card1 W | card1 S | card1 I |
| --- | ---: | ---: | ---: |
| card0 W | 8.571 | 24.254 | 9.825 |
| card0 S | 1.148 | 0.003 | 0.103 |
| card0 R | 0 | 0 | 2.248 |
| card0 I | 0.036 | 2.540 | 8.414 |

| Who computes | ms | Share of cycle |
| --- | ---: | ---: |
| Both cards | 8.57 | 15.0% |
| card0 only | 34.08 | 59.6% |
| card1 only | 1.18 | 2.1% |
| Neither | 13.31 | 23.3% |

"Neither" breaks down as:
- 8.41 ms both idle: host-bound;
- 2.25 ms card0 resetting while card1 is idle;
- 2.54 ms card0 idle while card1 spins: card1's replay is submitted 2.1 ms
  before card0's, and its first wait lasts 2.60 ms on average;
- 0.11 ms card0 spinning while card1 is idle.

Within the verifier window (37.60 ms): both 8.57, card0 only 27.37, card1 only
1.18, neither 0.48 (0.11 of it card0 in ep_wait).

EP exchange per layer:
- Push latency (card0 push end -> card1 start): 3.9 us median (3.0-4.75).
- Return latency (card1 send end -> card0 wake): 2.3 us median (1.4-2.9).
- Card1 chain: 216 us mean = compute 203 (GU ~96, down ~68, send 29, plan 5.6,
  grouped 2.6, copy 1.6) + in-graph gaps 12.8.
- Card0 local work after push: 160 us of routed work, plus a ~55 us shared
  expert that runs during ep_wait.
- Card0 waits on card1 in 76% of layers (36.4 of 48): mean 26 us, median 12,
  max 161; 1.25 ms per cycle. Card1's chain and card0's local work are close
  to balanced, so EP is not the dominant sink.
- Card1 waits for card0 515 us per layer (layers 1-47; layer 0 waits 2.60 ms
  because of the early submit), 26.8 ms per cycle in total. This is card0's
  dense pre-push work while card1 spins.

## 6. Where the idle time goes, and ranked opportunities

Idle sinks, ranked (card0 decides the cycle):

1. Host-submission gaps: 6.25 ms profiled. These are mostly profiler
   artifacts (S142). What remains real in production is PLE staging
   0.67-0.70 ms (S142 median; cold row cache, SSD p50 299 us), verify submit
   0.07 ms and some commit prep. Inferred real host-bound GPU idle is about
   1.0-1.5 ms per cycle.
2. Verifier-tail eventReset storm: 2,226 resets (one per out-of-order graph
   node) plus 210 in MTP. 5.43 ms span profiled, and the host sync waits
   through it. Inferred unprofiled bound: S142 verify_queue_wait is 38.64 ms
   against a 37.60 ms profiled verifier span, which leaves at most ~1 ms for
   the storm, assuming the verifier kernel span is unchanged.
3. EP wait (card0 spinning on card1): 1.25 ms.
4. In-graph dispatch gaps: verifier 0.40, commit 0.58, MTP 0.46 ms. Card1's
   0.88 ms sits inside the EP pool, not on top of it.
5. Launch-floor-bound tiny kernels: these are busy time, not idle. 1,428
   launches under 5 us take 3.59 ms.

Card1 is structurally underused: it computes for 9.75 ms of a 57 ms cycle.

Opportunities, with estimated savings per cycle against the unprofiled
~46.5 ms cycle. A, B and C are on different segments of the critical path and
are roughly additive. G overlaps A and C. B is capped by the 1.25 ms EP pool.

| # | Opportunity | Est. saving | Basis and assumptions | Risk / status |
| --- | --- | --- | --- | --- |
| A | Batch verifier attention per slot in C2 mode: 6 per-row calls -> 2 calls with n=3 for scores/topk/chunk/merge; 288 -> 96 launches per cycle | 1.2-1.7 ms | Measured 3.05 ms per cycle (254 us per attention layer). Assumes a z=3 chunk launch (198 WGs, one wave) takes 1.0-1.3x the cold single-row 42.7 us, and merge/scores/topk shrink similarly. On card0's critical path (card1 spins during attention). One 6-row launch with per-z pool pointers might add ~0.4 ms. | Small `verify.cpp` change using the existing batched kernel; per-z math looks independent, so outputs may be bit-identical, but this must be checked. Not previously tried (QSA group3 was the commit-side append). |
| B | Shorten card1's EP return: pre-reduce card1's expert rows per token before sending (6 rows instead of ~30-40 full fp32 rows; send 29 -> ~5 us) and cut card1's 12.8 us per layer of in-order gaps | 0.7-1.0 ms, cap 1.25 | Card0 waits a mean of 26 us per layer in 76% of layers. The saving is min(wait, chain cut) per layer. | Changes fp32 summation order, so it needs a parity/tolerance decision. The gap part depends on graph style (see D). |
| C | Fuse router top10 + resident_plan_ep (single WG, 6.3 us) + quantize rows + ep_push | 0.4-0.6 ms | 48 chains at 20.6 us of kernels + 0.6 us of gaps; assumes a fused kernel of ~9-12 us. Card0-critical, and it starts card1 earlier too. | Small. Card1 also runs `resident_plan_ep` (5.6 us), which could be precomputed and pushed. |
| D | Remove per-node event resets: in-order graph for linear spans, and an explicit cross-queue event only for the ep_wait/shared-expert overlap (S135 shows the API works) | 0-1.0 ms unprofiled | Profiled storm 5.43 ms, inferred unprofiled storm <= ~1 ms. CAUTION: in-order graphs show a 2.4-2.6 us per-kernel gap in this profile. If that is real unprofiled, 2,178 nodes would cost ~5 ms, which is far worse. | Needs an unprofiled microbench first: N=2,000 dependent tiny kernels, in-order vs out-of-order replay, wall time with no profiler. That one result also decides B's gap part and the commit graph (H). |
| E | Overlap or prefetch PLE staging for verifier tokens (existing `STRATA_BATCH_MTP_PLE_PREFETCH`, OFF here) | up to ~0.6 ms | Unprofiled stage_PLE is 0.67-0.70 ms (min 0.03 when cached), between the draft heads and verify submit while both GPUs idle. | Partly an artifact of a cold cache in a 64-token test; measure in a warm long run. |
| F | Draft-head cost: 2 x 1.21 ms full-vocab (248,320) Q5K C2 heads, plus 4 x 89 us single-WG `sample_tokens` | 0.3 ms from existing parallel argmax (S114 measured +0.58%); ~2 ms if a reduced draft vocab | Measured 2.42 ms + 0.36 ms per cycle. | A reduced draft vocab changes drafts and acceptance, so it is an algorithmic decision. |
| G | Fuse the remaining small-kernel chains (kv_append x6 + indexer x6 per attention layer, gdn_conv_l2 x2 + ab, combine chain, HC norm+down+reduce) | 0.3-0.5 ms | 217 chains / 1,049 launches per cycle at ~1.1 us fixed cost each; upper bound 0.9 ms. Excludes the chains counted in A and C. | Many small edits. HC is a known lead (S115/S116). |
| H | Commit graph: layer-batched GDN commit (S145, -26% private) plus fewer in-order gaps | 0.3-0.6 ms | 2.09 ms span: 1.51 ms of kernels, 0.58 ms of gaps. | S148 HOLD (native-work gate). |
| I | Structural: card1 computes only 17%. Card0's 23.5 ms per cycle of pre-push dense work (MMVQ 9.8, HC 6.1, attention/GDN) runs while card1 spins. TP-split dense projections across cards | ~2-4 ms (speculative) | Assumes ~50% of dense MMVQ/HC moves to card1, with 1-2 extra flag exchanges per layer at the measured ~6-10 us round trip plus transfer. | Large redesign. Not a small measured change. |

Bottom line. A + B + C + E is a realistic, non-overlapping first tranche of
roughly 2.3-3.9 ms per cycle, about 5-8% of the unprofiled 46.5 ms, before D
(which is uncertain), F (algorithmic) or I (structural). The single best
kernel-level lead is A: per-slot attention batching, using a kernel that
already supports it. The cheapest decisive measurement is D's unprofiled
graph-style microbench, because its answer gates D, part of B, and H.

## Limits

- One instrumented pair: 16 nested steady cycles from one process, not
  independent replicates.
- Profiler inflation is large for host brackets and plausibly for per-node
  command-stream costs. Unprofiled values are inferred from S142 CPU brackets
  only; there is no CPU/GPU clock join.
- Family labels come from kernel names and grids, not tensor identity.
- Savings are estimates with the assumptions stated above. None is a measured
  GPU result.
- Timestamp resolution is about 0.02-0.04 us. Cross-card clock agreement was
  checked only through the EP flag protocol (within ~1.5 us).
