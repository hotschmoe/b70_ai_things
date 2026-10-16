# Final stack: C1 window census, card 0

Analyst: Claude Opus 5.5 (claude-opus-5-5), 2026-10-16. CPU-only analysis of
an existing trace; no GPU was touched.

## Scope and method

CONFIG -> unitrace (same flags) of `builds/strata-fin3`. Stack = best3 plus:
- KREG BF16 GEMV.
- Fused shared expert.
- Router TOPK10 tree.
- GDN_AB per row and GDN_GRF256.
- GDN commit layers (C2).
- MMVQ2 IQ3_S.
- D2D admission and XMX prefill.

Raw trace: `runs/trace-final/trace/strata-fin3.467.json` (458 MB).

COMMAND -> One parse under `bin/cpu-run` (13 s) into
`agents/trace_cache_final/` (731,648 complete events). `analyze2.py` was
extended for `kreg2<IQ3S>`, `shexp_gu/down` and `bf16k`. CSVs:
`agents/trace_final_csv/final_c1_*`.

Windows and context:
- C1 window = KREG C4 head end -> next one's end.
- 83 windows; steady = 4-22 and 24-82 (78). Window 23 is a capture (42.1 ms);
  windows 2-3 are request admission.
- C1: 256 tokens in 80 windows (175 of 240 drafts accepted), 3.2 tokens per
  window. Profiled 29.26 ms per window = 107.8 tok/s.
- At the unprofiled 140 tok/s, a window is about 22.9 ms (inferred). Host and
  dispatch gaps are profiler-inflated (see the best3 report).
- Kernel durations are device timestamps.

## 1. C1 window (78 steady): phases and what shrank vs best3

Window 29.26 ms (min 28.25, median 29.24, max 30.25, sd 0.36), was 31.21:
-1.95 ms (-6.2%).

| Card0 phase | best3 ms | final ms |
| --- | ---: | ---: |
| Verifier tail (sample) | 0.09 | 0.09 |
| Host gap -> commit | 3.85 | 3.17 |
| Commit graph (98 kernels) | 0.96 | 0.96 |
| Host gap + eager | 0.16 | 0.16 |
| MTP steps 1 / 2 / 3 | 0.83 / 0.72 / 0.72 | 0.83 / 0.71 / 0.71 |
| Host gap -> verifier | 1.48 | 1.40 |
| Verifier replay span | 22.39 | 21.22 |
| ...kernel union | 18.41 | 17.29 |
| ...in-order dispatch gaps | 3.69 | 3.36 |
| Card0 compute / EP-wait only | 21.06 / 0.29 | 19.94 / 0.57 |
| Card0 kernel launches | 1,731 | 1,575 |

Card1: compute 4.92 ms (unchanged), spin 15.54 ms.

Shrank, card0 ms per window:

| Item | Saved |
| --- | ---: |
| BF16 GEMVs (KREG BF16; 128 -> 68 launches) | -0.37 |
| Dense + shared MMVQ (6.60 -> 6.22) | -0.37 |
| ...of which shared expert (144 -> 96 launches, 1.41 -> 1.13) | -0.28 |
| ...of which IQ3_S out-proj | -0.07 |
| GDN (step_norm GRF256, gdn_ab per row) | -0.20 |
| Router top-k | -0.08 |
| Combine | -0.07 |
| EP wait | +0.28 (card0 now waits on card1, whose chain is unchanged) |

Unchanged: HC, routed experts, heads, attention, MTP.

## 2. C1 family census, card 0 (per window; 29.26 ms)

| Family | n | ms | % window | Achieved GB/s (% of 608) | Status |
| --- | ---: | ---: | ---: | --- | --- |
| HC projections (verifier down2q/up2q Q8 2.95; MTP BF16 0.33) | 212 | 3.28 | 11.2% | 231 (38%) Q8 verifier; 344-409 MTP | LAT target |
| Routed GU, card0 | 48 | 2.26 | 7.7% | 345-500 (estimate, upper bound) | ROOF (estimate) |
| MMVQ Q6K (KREG) | 67 | 1.97 | 6.7% | 433-481 (71-79%) | Near ROOF |
| GDN (commit 0.63 + verifier 0.76) | 180 | 1.39 | 4.8% | n/a | LAT/ALU target |
| Routed down, card0 | 48 | 1.22 | 4.2% | 438-468 (estimate) | ROOF (estimate) |
| Shared expert, fused (gu 9-18 us, down 8.2 us) | 96 | 1.14 | 3.9% | 110 (18%) | LAT target |
| MMVQ IQ4XS (KREG) | 36 | 1.09 | 3.7% | 321-362 (53-60%) | Medium |
| MMVQ IQ3S (kreg2 at 4 cols) | 15 | 1.03 | 3.5% | 94-109 (16-18%) | LAT/ALU target |
| Vocab head C4 (KREG) | 1 | 0.83 | 2.8% | 525 (86%) | ROOF, not a target |
| MMVQ Q4K (KREG) | 26 | 0.67 | 2.3% | 399-453 (66-75%) | Near ROOF |
| BF16 GEMV (KREG; router+indexer, gates) | 68 | 0.64 | 2.2% | n/a (combined shapes) | Medium |
| Attention chunk/merge/select | 54 | 0.58 | 2.0% | n/a | LAT |
| EP wait spin | 48 | 0.57 | 2.0% | - | Wait on card1 |
| Q8 activation quantize | 219 | 0.55 | 1.9% | - | LAUNCH floor |
| EP plan/group (incl. MTP experts 0.16) | 108 | 0.55 | 1.9% | - | LAUNCH |
| MMVQ Q8 (MTP layer) | 28 | 0.54 | 1.9% | 203-557 (attn_q 92%) | ROOF for large shapes |
| Draft head subset (3 x 131 us) | 3 | 0.39 | 1.3% | 545 (90%) | ROOF, not a target |
| MMVQ Q5K (KREG) | 12 | 0.35 | 1.2% | 407-436 (67-72%) | Near ROOF |
| Attention prep | 105 | 0.34 | 1.2% | - | LAUNCH |
| Router top-k (tree) | 51 | 0.31 | 1.1% | - | LAUNCH |
| MoE combine | 54 | 0.28 | 1.0% | - | LAUNCH |
| Sampling/select | 16 | 0.20 | 0.7% | - | |
| EP push | 50 | 0.18 | 0.6% | - | |
| eventReset (MTP graphs only) | 153 | 0.14 | 0.5% | - | |

Dense MMVQ plus shared expert overall: 1,871 MB in 6.22 ms, 301 GB/s (49.5%).
By role:

| Role | ms | GB/s (%) |
| --- | ---: | --- |
| qkv | 1.47 | 426 (70%) |
| out-proj | 1.48 | 368 (61%) |
| gate | 1.34 | 237 (39%) |
| attn_q | 0.64 | 370 (61%) |
| shared | 1.13 | 110 (18%) |
| k/v | 0.16 | 142 (23%) |

Card0 idle: 8.62 ms per window (host 4.74, verifier in-order gaps 3.36, commit
0.24, MTP 0.27). Kernel durations: 1,575 launches; 683 under 5 us (1.89 ms) and
966 under 10 us (3.83 ms).

## 3. MTP drafter, 3 steps, by family (card0, per window)

Span 2.25 ms = 7.7% of the window; drafter / verifier ratio 0.11. It is
unchanged vs best3 (kernels 2.05 ms both).

| Family | n | us | Status |
| --- | ---: | ---: | --- |
| MMVQ Q8, MTP layer dense (attn_q 60 us at 557 GB/s; out-proj 16 us at 429; small shapes 200-330) | 27.7 | 543 | Large shapes ROOF; small LAT |
| Draft head 40,525 (3 x 131 us) | 3 | 393 | ROOF (90%) |
| HC down2/up2, BF16 (19.1 / 16.0 us, 344 / 409 GB/s) | 18 | 334 | Q8 HC would roughly halve the bytes: target, ~0.15 ms |
| MTP MoE experts (moe_grouped_s2 x6, 23.7 us) + group/plan | 12 | 161 | LAT |
| eventReset (MTP graphs still out-of-order) | 153 | 137 | Overhead |
| Sampling/select (sample, top-prob, select, map_ids) | 15 | 113 | |
| Attention chunk/merge (g512) | 6 | 99 | |
| Attention prep | 21 | 64 | |
| Q8 quantize | 26 | 58 | |
| Barrier | 6 | 49 | |
| BF16 GEMV | 6 | 32 | |
| Copies / router / PLE / misc | 23 | 69 | |
| In-graph gaps (median 0.23 us) | | 270 | |

## 4. Top 12 remaining card-0 costs per C1 window

Ranked by profiled ms. "% of 22.9" uses the inferred unprofiled window and
applies to kernel rows only.

| Rank | Cost | ms | % of 29.26 | % of 22.9 | Flag |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Verifier in-order dispatch gaps (1,367 x 2.46 us) | 3.36 | 11.5% | - | IDLE (launch); largely profiler-inflated, check unprofiled |
| 2 | HC projections, Q8 v2 (+ MTP BF16) | 3.28 | 11.2% | 14.3% | LATENCY bound, 38% of roofline: TARGET |
| 3 | Host gap verifier -> commit (sync wake + host) | 3.17 | 10.8% | - | IDLE; profiler-suspect |
| 4 | Routed GU, card0 | 2.26 | 7.7% | 9.9% | AT ROOFLINE (estimate): not a target |
| 5 | Dense out-proj 6144->2560 (all formats) | 1.48 | 5.0% | 6.5% | Mixed: Q6K 71%, Q5K 67%, Q4K 66%, IQ4XS 53% near; IQ3S 0.22 ms at 16% TARGET |
| 6 | Dense qkv | 1.47 | 5.0% | 6.4% | NEAR ROOFLINE (70%): not a target |
| 7 | Host gap MTP -> verifier (PLE + submits) | 1.40 | 4.8% | - | IDLE; partly real (PLE ~0.67 ms) |
| 8 | GDN (verifier step_norm 0.50 + ab 0.17 + conv 0.09; commit 0.63) | 1.39 | 4.8% | 6.1% | LATENCY/ALU: TARGET |
| 9 | Dense gate 2560->6144 | 1.34 | 4.6% | 5.9% | IQ3S kreg2 gate 0.67 ms at 100 GB/s (16%) TARGET; rest 53-73% |
| 10 | Routed down, card0 | 1.22 | 4.2% | 5.3% | AT ROOFLINE (estimate): not a target |
| 11 | Shared expert, fused (96 launches) | 1.14 | 3.9% | 5.0% | LATENCY bound, 18%: TARGET |
| 12 | Vocab head C4 | 0.83 | 2.8% | 3.6% | AT ROOFLINE (86%): not a target |

Next in line:
- BF16 GEMV 0.64 ms (medium).
- Attention 0.58 ms (latency).
- EP wait 0.57 ms: card1 routed chain now gates card0; card1 routed kernels
  run 20-30% slower than card0's.
- Q8 quantize 0.55 ms and EP plan 0.55 ms (launch floor).
- MTP Q8 MMVQ 0.54 ms (roofline).
- Draft head 0.39 ms (roofline).

VERDICT -> Remaining kernel targets, by size:
- HC Q8 v2: about 1.7 ms possible if it reached 85% of roofline.
- IQ3_S at 4 columns: the kreg2 path is still about 100 GB/s, 0.9 ms, against
  about 0.2 ms at 85%.
- GDN: 1.39 ms.
- Fused shared expert at 18%.

Roofline-bound, not targets: routed experts, Q6K/Q4K/Q5K dense, both heads,
MTP Q8.

## Limits

- Single profiled run.
- The unprofiled window is inferred from tok/s.
- Routed bytes assume 50% ownership with no de-duplication (upper bound).
- BF16 combined-GEMV and attention bytes are not inferred.
