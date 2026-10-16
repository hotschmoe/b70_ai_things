# best3 stack trace analysis: C1 window and C2 cycle, vs hcq8 C1 and S136 C2

Analyst: Claude Opus 5.5 (claude-opus-5-5), 2026-10-16. CPU-only analysis of
an existing trace; no GPU was touched.

## Scope and method

CONFIG -> unitrace (same flags as before) of
`/mnt/vm_8tb/b70/results/night-20261016/builds/strata-iq1` ("best3").
Session: warm 64 tokens, one C1 256-token request, one C2 pair of 192 tokens
each. Engine log flags:
- STRATA_NIGHT_ATTN_BATCH=1, SH_LATE=1 (linear verify graph), HC=q8
  (two-kernel read), KREG MMVQ (Q4_K/Q5_K/Q6_K/IQ4_XS incl. head).
- STRATA_NIGHT_IQ=1, EP_PSUM=1, EP_PLAN1=1, EP_ROWS=2.
- QSA_SCORES_T, TOPK_SCAN, ATTN_VPIPE, ATTN_MERGE_ACTIVE.
- 40,525-token draft head. Expert split 138/374.

Raw trace: `runs/trace-best3/trace/strata-iq1.467.json` (700,913,589 B).

COMMAND -> One parse under `bin/cpu-run` (26 s, 3.2 GB RSS) into
`agents/trace_cache_best3/` (1,508,266 complete events). Analysis uses the same
`trace_gap_scripts/analyze2.py`, extended for the new kernel names (KREG
`night_mmvq_kreg[_g]`, `night_hc` down2q/up2q, `gu/down_n_k<T>` where T+100 is
the ALU variant of type T, `ep_send_psum`, `ep_combine_psum`,
`attn_*_night`). The older traces were re-run through the same code for a
like-for-like comparison. CSVs: `agents/trace_best3_csv/` (`best3_c1_*`,
`best3_c2_*`), older ones in `agents/trace_c1_csv/`.

Windows:
- C1 window = 4-column full-vocab head end -> next one's end. KREG
  `kreg<Q5K,4>` has grid 62080 x 4 rows = 248,320.
- 81 C1 windows. Steady = 1-21 and 23-80 (79); window 22 contains a graph
  capture (45.6 ms).
- C2 cycle = `kreg_g<Q5K,6>` head end -> next one's end (73 cycles). Steady =
  65 cycles (4, 6-25, 27-33, 35-50, 52-72).
- C2 cycles 0-3, 5, 26, 34 and 51 contain graph re-capture (262 create calls
  each) and take 88.1-92.9 ms.

Profiled vs unprofiled:
- C1: 256 tokens in 82 windows (175 of 246 drafts accepted), so 3.12 tokens
  per window. The profiled window is 31.21 ms (= 100 tok/s, matching the
  logged 98.6). The coordinator's unprofiled 139 tok/s implies about 22.4 ms
  per window, so roughly 8.8 ms of each profiled window is profiler-induced.
- C2: unprofiled 77 tok/s per stream at about 2.6 tokens per stream per cycle
  implies roughly 33 ms per cycle, vs 40.10 profiled (inferred).
- Kernel durations are device timestamps and treated as reliable. Host gaps and
  in-graph dispatch gaps are the inflated part (see section 4).

## 0. What shrank (headline)

| Per window / cycle | hcq8 C1 | best3 C1 | S136 C2 | hcq8 C2 | best3 C2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Profiled wall (ms) | 43.13 | 31.21 (-27.6%) | 57.14 | 56.67 | 40.10 (-29.8% vs S136) |
| Card0 compute union | 32.18 | 21.06 | 42.65 | 42.06 | 26.80 |
| Card0 EP-wait only | 1.58 | 0.29 | 1.25 | 1.92 | 0.51 |
| Card0 eventResets (busy ms) | 1,934 (1.78) | 153 (0.14) | 2,435 (2.24) | 2,434 (2.23) | 209 (0.19) |
| Card0 kernel launches | 1,988 | 1,731 | 2,654 | 2,653 | 2,067 |
| Card0 idle | 7.59 | 9.72 | 10.99 | 10.45 | 12.60 |
| Card1 compute / spin | 7.61 / 19.88 | 4.92 / 16.82 | 9.75 / 26.80 | 10.07 / 26.49 | 6.91 / 19.82 |

## 1. C1 window phase timeline (79 steady windows)

Window 31.21 ms: min 30.24, p10 30.72, median 31.22, p90 31.68, max 32.04,
sd 0.38. Bins: 30 ms: 25, 31 ms: 53, 32 ms: 1.

Card0 (mean ms):

| Phase | hcq8 span | best3 start | best3 span | best3 busy | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Verifier tail | 4.29 (1,766 resets) | 0.00 | 0.09 | 0.09 | Only sample_tokens; reset storm gone |
| Host gap -> commit | 1.35 | - | 3.85 | 0 | Host sync returns ~2.4 ms after the last GPU op, then ~1.4 ms host work (host API timeline) |
| Commit graph (98 kernels, in-order) | 0.98 | 3.94 | 0.96 | 0.72 | Unchanged |
| Host gap + eager | 0.16 | - | 0.16 | 0 | |
| MTP step 1 / 2 / 3 | 1.91 / 1.78 / 1.78 | 5.06 | 0.83 / 0.72 / 0.72 | 0.73 / 0.63 / 0.63 | Draft head 0.995 -> 0.131 ms each (40,525-token subset) |
| Host gap -> verifier (PLE + submits) | 2.64 | - | 1.48 | 0 | |
| Verifier replay (1,476 kernels + 48 ep_wait) | 28.23 | 8.82 | 22.39 | 18.70 | Kernel union 26.34 -> 18.41; in-graph gaps 0.31 -> 3.69 |

Card1: idle 7.84 ms, verifier replay 22.46 ms (compute 4.92, spin 16.82,
in-graph gaps 0.72), then idle 0.92 ms.

Fractions of the best3 C1 window:

| Part | ms | Share | hcq8 share |
| --- | ---: | ---: | ---: |
| MTP drafter (3 steps) | 2.27 | 7.3% | 12.7% |
| Verifier replay | 22.39 | 71.7% | |
| ...of which kernel time | 18.41 | 59.0% | |
| ...of which in-order gaps | 3.69 | 11.8% | |
| Commit | 0.96 | 3.1% | |
| Host gaps | 5.5 | 17.6% | |
| Verifier tail | 0.09 | 0.3% | |

Drafter / verifier ratio is 0.10 (was 0.19).

Per layer (C1):

| | GDN layer | Attention layer |
| --- | ---: | ---: |
| Span (was) | 438 us (543) | 480 us (575) |
| Card0 local routed work after push | 137 us | 136 us |
| Card1 chain | 115 us (gaps 10, send 7.2; was 22) | 105 us |

Card1 is no longer on the critical path. Card0 waits only 7/3 us per layer,
0.29 ms per window.

## 2. Per-window family census (card0 unless noted; us per window)

| Family | hcq8 C1 | best3 C1 | best3 C1 n | S136 C2 | best3 C2 | best3 C2 n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HC projections (best3: down2q+up2q Q8 verifier; BF16 down2/up2 in MTP) | 4,504 | 3,306 (verifier 2,971) | 212 | 5,526 | 3,996 (verifier 3,554) | 218 |
| HC norm/reduce (separate kernels) | 1,133 | 0 (fused into v2) | 0 | 1,193 | 0 | 0 |
| Vocab head, full | 4,703 (4 heads) | 832 (C4 KREG) | 1 | 4,676 | 1,040 (C6 KREG_g) | 1 |
| Draft head, 40,525 subset | - | 392 (3 x 131) | 3 | - | 264 (2 x 132) | 2 |
| Routed GU, card0 | 2,927 | 2,273 | 48 | 4,365 | 3,040 | 48 |
| Routed down, card0 | 2,046 | 1,228 | 48 | 3,099 | 1,646 | 48 |
| MMVQ Q6K (KREG) | 3,261 | 1,981 | 71 | 3,827 | 2,374 | 71 |
| MMVQ IQ3S (multi_nc<21>, no KREG) | 1,639 | 1,586 | 55 | 1,929 | 2,003 | 55 |
| MMVQ IQ4XS (KREG) | 1,736 | 1,262 | 67 | 2,066 | 1,565 | 67 |
| MMVQ Q4K (KREG) | 1,733 | 758 | 45 | 2,311 | 1,021 | 45 |
| MMVQ IQ4NL (shared down, unchanged) | 667 | 658 | 48 | 797 | 783 | 48 |
| MMVQ Q5K (KREG) | 784 | 355 | 14 | 1,005 | 479 | 14 |
| MMVQ Q8 (MTP layer) | 549 | 544 | 28 | 737 | 726 | 38 |
| GDN (commit + verifier) | 1,603 | 1,593 (634 + 959) | 180 | 2,819 | 2,812 (1,272 + 1,540) | 324 |
| BF16 GEMV (router g512, indexer, gates) | 1,034 | 1,020 | 128 | 1,212 | 1,215 | 130 |
| Attention chunk/merge/select | 943 | 584 | 54 | 3,417 | 673 | 68 |
| Attention prep | 343 | 340 | 105 | 695 | 537 | 197 |
| Q8 activation quantize | 659 | 547 | 219 | 707 | 576 | 228 |
| EP plan/group | 576 | 547 | 108 | 677 | 650 | 112 |
| Router top-k | 388 | 393 | 51 | 400 | 403 | 52 |
| MoE combine (best3: ep_combine_psum) | 132 | 345 | 102 | 160 | 353 | 104 |
| EP wait spin | 3,405 | 292 | 48 | 3,196 | 511 | 48 |
| Sampling/select | 502 | 199 | 16 | 640 | 236 | 21 |
| EP push | 178 | 176 | 50 | 188 | 197 | 50 |
| eventReset | 1,776 | 136 | 153 | 2,242 | 190 | 209 |
| barrier | 138 | 49 | 7 | 164 | 17 | 2 |
| card1 routed GU | 3,619 | 2,883 | 48 | 4,614 | 4,100 | 48 |
| card1 routed down | 2,532 | 1,491 | 48 | 3,263 | 2,175 | 48 |
| card1 EP copy/send | 1,108 | 418 | 97 | 1,481 | 507 | 97 |
| card1 EP spin | 19,883 | 16,817 | 48 | 26,797 | 19,823 | 48 |

What shrank in C1 (card0, hcq8 -> best3, ms per window):

| Item | Saved |
| --- | ---: |
| Heads (draft subset -2.60, KREG C4 -0.89) | -3.48 |
| HC (v2 Q8: 54 -> 30 us per HC read) | -2.33 |
| Dense MMVQ (KREG; IQ3S/IQ4NL untouched) | -3.24 |
| EP wait (PSUM, linear graph) | -3.11 |
| eventResets | -1.64 |
| Routed on card0 (IQ ALU/SWAR decode) | -1.47 |
| Attention | -0.36 |
| Sampling | -0.30 |
| Combine | +0.21 |

## 3. Bandwidth (best3, verifier; percentages against 608 GB/s)

| Role / family | C1 MB | C1 ms | C1 GB/s (%) | C2 ms | C2 GB/s (%) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C4 / C6 vocab head (KREG) | 437 | 0.83 | 525 (86%) | 1.04 | 420 (69%) | At roofline (C1); C2 near |
| Draft head subset (KREG, 1-2 cols) | 71 | 0.131 each | 546 (90%) | 0.132 | 540 (89%) | At roofline |
| MTP Q8 wide32 (attn_q) | 33 | 0.060 | 558 (92%) | | 544 | At roofline |
| Dense qkv (Q6K/Q4K/IQ4XS/Q5K) | 626 | 1.47 | 427 (70%) | 1.75 | 358 (59%) | Near (Q6K 466, Q4K 448; IQ4XS 355) |
| Dense attn_q | 237 | 0.65 | 362 (60%) | 0.78 | 304 (50%) | Medium |
| Dense out-proj 6144->2560 | 543 | 1.55 | 351 (58%) | 1.91 | 284 (47%) | Q6K 433 (71%); IQ3S 71 (12%) |
| Dense gate 2560->6144 | 318 | 1.38 | 230 (38%) | 1.67 | 191 (31%) | IQ3S gate 95 GB/s, 0.71 ms (C1): target |
| Shared expert gate/up/down (144 launches) | 125 | 1.41 | 88 (15%) | 1.80 | 69 (11%) | Launch/latency bound: target |
| k/v 2560->512 | 23 | 0.13 | 173 (29%) | 0.18 | 128 (21%) | Small |
| All dense MMVQ (300 launches) | 1,871 | 6.60 | 284 (47%) | 8.08 | 231 (38%) | hcq8 was 9.84 ms / 190 GB/s |
| HC down2q + up2q (Q8, 194 launches) | 675 | 2.95 | 229 (38%) | 3.55 | 190 (31%) | Latency bound: target |
| Router BF16 GEMV g512 | 157 | 0.59 | 265 (44%) | 0.71 | 221 (36%) | Medium |

HC detail: `hc_down2q` g120 runs 17.4 us (200 GB/s) in C1 and 20.9 us in C2;
`hc_up2q` g160 runs 13.0 us (267 GB/s) in C1 and 15.6 us in C2. Cost still
scales with rows, so the kernels are issue/latency-bound.

Routed experts (estimate, upper bound on bytes: 50% of entries per card, no
de-duplication). Card0 shows 345-564 GB/s in C1 and up to 636 GB/s, i.e. 104%,
in C2. Exceeding the roofline proves the assumption overstates card0's bytes,
so card0 routed kernels should be treated as at or near roofline. Card1
estimates are 324-447 GB/s. Card1 is off the critical path now.

## 4. Idle / gap / reset accounting

C1, card0 idle 9.72 ms per window:
- Host gaps 5.51 ms: 3.85 after the verifier (sync wake ~2.4 ms + host ~1.4 ms),
  1.48 before the verifier, 0.16 near the eager copy.
- Verifier in-graph dispatch gaps 3.69 ms: 1,523 gaps, median 2.42 us, 87% in
  the 2-5 us bucket. This is the in-order-graph cost flagged in S136.
- Commit in-graph gaps 0.24 ms (median 2.48 us).
- MTP in-graph gaps 0.28 ms (median 0.23 us; still an out-of-order graph with
  49-55 resets per step).

Resets: 153 per window, all in the MTP steps, 0.14 ms (was 1,934, 1.78 ms).

C1, card1 idle 9.48 ms: 8.75 ms with no work submitted, plus 0.72 ms of
in-graph gaps (median 2.52 us).

C1 overlap (ms):
- Both cards compute: 4.12 (13.2%).
- Card0 only: 16.94 (54.3%).
- Card1 only: 0.80 (2.6%).
- Neither: 9.35 (30.0%), made up of 5.21 both idle, 3.92 card0 idle while card1
  spins (the in-order gaps and the pre-verifier host gap), 0.14 resets and
  0.09 card0 spin.

C2, card0 idle 12.60 ms per cycle:
- Host gaps 7.61 ms: 4.31 after the verifier, 2.41 before it, 0.89 around
  commit and copies.
- Verifier in-graph gaps 4.05 ms (1,659 gaps, median 2.44 us).
- Commit in-graph gaps 0.59 ms.
- MTP in-graph gaps 0.35 ms.

C2 overlap (ms): both 5.83 (14.5%), card0 only 20.97 (52.3%), card1 only 1.08
(2.7%), neither 12.22 (30.5%).

Profiler reality check (C1):
- Card0 GPU work is 21.35 ms per window (compute 21.06 + EP wait 0.29).
- The unprofiled window is about 22.4 ms.
- So at most about 1 ms of host and dispatch gaps survives without the
  profiler. The 3.85 ms host gap and the 3.69 ms of in-order gaps are mostly
  profiler cost; together they account for the 8.8 ms surplus. This is an
  inference; there is no unprofiled GPU timeline.
- In production, C1 is now about 95% card0-kernel bound. Kernels are the
  targets.

Kernel durations (card0):
- C1: 1,731 launches per window; 774 under 5 us (2.11 ms) and 1,045 under 10 us
  (3.96 ms).
- C2: 2,067 launches per cycle; 966 under 5 us (2.43 ms) and 1,225 under 10 us
  (4.18 ms).

Graph re-capture (outside the steady set):
- C2: 8 of 73 cycles carry about 262 create calls and run +50.6 ms (mean 90.7
  vs 40.1 ms). That is 405 ms, or about 12% of C2 decode wall in this trace,
  i.e. about 5.5 ms per cycle amortized. They occur mid-run (cycles 26, 34, 51)
  as new window shapes appear.
- C1: one capture window, +14.4 ms.
- Capture API cost is profiler-inflated, but recurring mid-run captures are
  real. Pre-capturing all window shapes at load would remove them.

## 5. Largest remaining costs, ranked

Status keys:
- ROOF = at or near roofline, not a target.
- LAT = latency/issue-bound kernel.
- LAUNCH = launch-floor-bound chain.
- IDLE = idle; profiler-suspect where noted.

"% of 22.4" uses the inferred unprofiled C1 window and applies to kernel
items only.

### C1, per window (31.21 ms profiled)

| Rank | Cost | ms | % of window | % of 22.4 | Status |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Host gap verifier -> commit (sync wake + host) | 3.85 | 12.3% | - | IDLE; mostly profiler (S142-type check needed) |
| 2 | Verifier in-order dispatch gaps (1,523 x 2.4 us) | 3.69 | 11.8% | - | IDLE/LAUNCH; mostly profiler (inferred) |
| 3 | HC projections, Q8 v2 (194 verifier + 18 MTP) | 3.31 | 10.6% | 14.8% | LAT: 38% of roofline |
| 4 | Routed GU, card0 | 2.27 | 7.3% | 10.1% | ROOF (estimate) |
| 5 | GDN (commit 0.63 + verifier 0.96) | 1.59 | 5.1% | 7.1% | LAT/compute |
| 6 | Dense out-proj (all formats) | 1.55 | 5.0% | 6.9% | Q6K 71% near; IQ3S share LAT |
| 7 | Host gap MTP -> verifier (PLE + submits) | 1.48 | 4.8% | - | IDLE; partly real (PLE ~0.67 ms unprofiled in S142) |
| 8 | Dense qkv | 1.47 | 4.7% | 6.5% | Near ROOF (70%) |
| 9 | Shared-expert GEMVs (144 launches) | 1.41 | 4.5% | 6.3% | LAT/LAUNCH: 15% |
| 10 | Dense gate 2560->6144 | 1.38 | 4.4% | 6.2% | LAT: IQ3S gate 0.71 ms at 16% |
| 11 | Routed down, card0 | 1.23 | 3.9% | 5.5% | ROOF (estimate) |
| 12 | BF16 GEMV (router g512, indexer, gates) | 1.02 | 3.3% | 4.5% | LAT: router 44% |

Next in line:
- Commit graph span 0.96 ms (LAT plus in-order gaps).
- C4 head 0.83 ms (ROOF, 86%).
- Attention 0.58 ms.
- Q8 quantize 0.55 ms (LAUNCH).
- EP plan 0.55 ms (LAUNCH).
- MTP Q8 MMVQ 0.54 ms (ROOF).
- Draft heads 0.39 ms (ROOF).
- Router top-k 0.39 ms.
- Combine 0.34 ms.
- EP wait 0.29 ms.

All IQ3_S multi_nc kernels together (gate, out, shared, attn_q) take 1.59 ms
at 10-16% of roofline. They are the only dense format left without KREG and
the single clearest kernel target.

### C2, per cycle (40.10 ms profiled; about 33 ms inferred unprofiled)

| Rank | Cost | ms | % of cycle | Status |
| ---: | --- | ---: | ---: | --- |
| 1 | Host gap verifier -> commit | 4.31 | 10.7% | IDLE; profiler-suspect |
| 2 | Verifier in-order dispatch gaps (1,659 x 2.4 us) | 4.05 | 10.1% | IDLE/LAUNCH; profiler-suspect |
| 3 | HC projections, Q8 v2 (6 rows) | 4.00 | 10.0% | LAT: 31% |
| 4 | Routed GU, card0 | 3.04 | 7.6% | ROOF (estimate) |
| 5 | GDN (commit 1.27 + verifier 1.54) | 2.81 | 7.0% | LAT/compute |
| 6 | Host gap MTP -> verifier | 2.41 | 6.0% | IDLE |
| 7 | Dense out-proj | 1.91 | 4.8% | Q6K 58%; IQ3S 10% LAT |
| 8 | Shared-expert GEMVs (144) | 1.80 | 4.5% | LAT/LAUNCH: 11% |
| 9 | Dense qkv | 1.75 | 4.4% | 59%, near |
| 10 | Dense gate | 1.67 | 4.2% | LAT: IQ3S gate 0.84 ms at 13% |
| 11 | Routed down, card0 | 1.65 | 4.1% | ROOF (estimate) |
| 12 | BF16 GEMV | 1.21 | 3.0% | LAT: router 36% |

Next in line:
- C6 head 1.04 ms (69%, near).
- MTP Q8 0.73 ms (ROOF).
- Attention 0.67 ms.
- EP plan 0.65 ms.
- Commit in-graph gaps 0.59 ms.
- Q8 quantize 0.58 ms.
- Attention prep 0.54 ms.
- EP wait 0.51 ms.

Outside the steady set: graph re-capture costs about 5.5 ms per cycle
amortized in this trace.

## 6. Verdict and leads

The night stack removed the reset storm, the EP wait, the full-vocab draft
heads, the per-row attention and most dense-MMVQ inefficiency. In production,
C1 is now about 95% card0-kernel bound. Remaining kernel targets, by expected
gain:

1. HC Q8 v2 at 31-38% of roofline: 3.3 / 4.0 ms. At 85% this would be about
   1.3 ms for the verifier HC, so roughly 1.7-2.0 ms (C1) / 2.2 ms (C2) saved.
2. IQ3_S dense (no KREG port), 1.59 / 2.0 ms at 10-16%: about 1.2 ms possible.
3. Shared-expert GEMVs, 144 launches at 11-15%: fuse gate/up/down. About
   1 / 1.3 ms.
4. GDN, 1.6 / 2.8 ms: compute/latency; commit plus verifier steps.
5. Router BF16 GEMV at 36-44%: about 0.3 ms.
6. Dense gate (IQ4XS KREG at 53%) and out-proj Q6K C2 at 58%: medium gains.

Not targets: vocab and draft heads, MTP Q8 MMVQ, card0 routed GU/down (by
estimate), and qkv Q6K/Q4K, which are all at or near roofline.

Idle items to verify unprofiled before acting:
- Host sync wake-up after the verifier (2.4 ms profiled).
- In-order dispatch gaps (2.4 us per node profiled).
- C2 mid-run graph re-captures (8 of 73 cycles at +50 ms). Very likely a real
  serving cost: pre-capture all window shapes.

## Limits

- Profiled single run.
- Unprofiled window times are inferred from tokens per window and the
  coordinator's tok/s, not measured on this binary's GPU timeline.
- Routed bytes are assumption-based.
- Shapes come from grid x rows-per-work-group (KREG last template scalar),
  checked against model shapes.
- Clock alignment is by activity.
