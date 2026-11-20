# Status (each lead keeps its own section current)

## Lead A (model path)  [updated 18:30 UTC]
- FIRST SERVE-LEVEL NUMBERS (18:26, worker S, workers/S/report.md + first-results.json): tensorfold-native OpenAI serve,
  TP2 GGUF IQ3_XXS + MTP, fn-serve c3ac6cf (= fn-int 46afffb + server wiring), :18150, unprofiled, greedy,
  256 tok, p0 tokens == solo serial (exact), chat coherent, health pre OK:
    C1 42.4 tok/s (T2) / 45.5 (T4); C2 66.5 agg (33-38/stream); C4 89.9 agg; C8 111.8 agg (T2; T4 worse at C>=4);
    TTFT C1 107 ms (short prompt); prefill 309 / 299 / 286 tok/s at 512 / 2k / 8k.
  Not merged yet into this number: HC kernels (H; ~18 ms/window GPU), recorded windows (R; ~17 ms/window host lag), KF4/KF5.
  Strata DD reference: C1 ~141, C2 ~92/stream, prefill ~2200. Re-measure after each merge with S/serve_bench.sh.
- Milestones: M-A1..M-A4 DONE. M-A5 (speed) in progress: correctness on the fast path DONE (16:35).
- Integration: b70/flashnext-2card (fn-int) = 46afffb = M-A4 + push transport + Strata-parity q8_1 kernels (row-invariant)
  + device PLE + per-card kernel-timestamp profiler. All gates pass on it: teacher vs Strata oracle 547/554 top-1, KL <= 0.008,
  deterministic; serial == T2 == T4 (8 prompts); C4 / stagger / 3-run soak == solo. q8_1 default.
- Critical path (diagnostic, profiled), C1 T2 55 ms/window per card: kernels 31.7 ms (HC 20.1 ms: fn_hc_down 16.7 ms = naive
  324-work-item GEMV; dense 5.0; experts ~3) + host submission lag 16.8 ms (~1720 appends/card) + event churn 5.2 ms; exposed
  AR 1.65 ms. C4 T4 116 ms/window. Prefill p7 287 tok/s (no XMX prefill yet).
- USER PRIORITY (17:25): serve-level numbers first. Worker S builds/benchmarks the OpenAI serve on :18150 (first numbers
  today); I6 = 3-min smoke gate + KF4 merge; merges to fn-int on smoke pass, full gates only at milestones.
- Workers now: R (fn-rec) recorded windows: R0 GO (recorded 32.3 ms vs 31.8 ms floor, host submit 13.8 ms -> 39 us, bitwise);
  R1 invariant-immediate in progress. H (fn-hc) HC kernels (target hc_down <= 20 us/call). KF (fn-model) round 4: Q6_K,
  IQ4_NL down (82.5 -> 70.1 us, now beats Strata 72), prepared shared q8_1 activations.
- Lead B: thanks for M-B2c, tracer v2, PLE-import analysis. No new requests.
- Next GPU need: both cards in short leased runs (R, H gates), card 0 (KF, H microbench).

## Lead B (runtime, latency, features)  [updated 17:58 UTC]
- SPEED DIRECTIVE (17:25) applied: kv8 and park2 wrapping up now -> merge fn-int 46afffb, one fast smoke each, HANDOFF.md for
  Lead A to merge (B/kv8/HANDOFF.md, B/park2/HANDOFF.md, ETA ~18:45). park2 delivers unprofiled park/restore ms at 1K/8K/32K.
  DPAS prefill next gets wired into fn-int prefill behind a flag (Lead B branch) for a real prefill tok/s on the serve.
- FOR LEAD A, ready now:
  * Tracer for your path: b70/fn-rt-trace2 d1a8a93 (based on fn-int bfb427f; default off, XPU_TRACE=<fresh dir>,
    XPU_TRACE_MODE=markers for baselines; analyzer: tools/xpu_trace/analyze.py TRACE --output OUT --window a4_window --streams N
    --rows R). Overhead 1.3-3.6%. First profile: JOURNAL [B] 13:20, B/trace2/RESULTS.md (C4: card 0 idles ~28 ms/window waiting
    for card 1, whose per-row GDN/attention appends are late).
  * Push all-reduce: b70/fn-runtime 611566d (TRANSPORT-B 4d).
  * Recorded-list transport (M-B2c) in b70/fn-runtime (TRANSPORT-B 4e: beginRecording / recordAllReduceF32 / finish, device-reset
    events, 10,000 replays clean at 10-160 KB x 96-192 hops, ~6 us host submit per window, flat p99). Counter-based events FAILED.
  * PLE host memory ANSWERED (TRANSPORT-B 4f, B/pushar/PLE_HOST.md): file-mapping import fails even with unlimited memlock
    (anonymous 4/64 MiB imports work). Use one zeMemAllocHost filled by O_DIRECT reads (3.85 GB/s measured, ~7.5 s for
    28.8 GB) + fadvise DONTNEED on the file range; the page cache is reclaimable, so the one USM copy is the only hard cost.
- Milestones: M-B1 DONE (one card 0.3-0.5% overhead; two-card profile done). M-B2, M-B2b, M-B2c DONE.
- M-B3: B3-1 + B3-2 DONE (b70/fn-runtime 620851f): 27B park/restore without recompute, bit-identical (bf16/int8, DFlash2,
  same/other slot, K=1/7/64); park 16K 155 ms (D2H 4.5 GB/s bound), restore 53 ms; concurrent T p50 +2.1% (gate 2%: narrow miss).
  B3-3 paged KV (codex-kv8, b70/fn-rt-kv8 from fn-int): bf16page chunk invariance (1/3/5/16/64) + 512-token seam bitwise ==
  legacy, 32/32 ids == legacy; int8 drafted == serial (p0, p1 full 256); M-A4 suites + teacher-forced quality gate running.
  B3-5 (Flash-Next two-card park/restore on the page pool): codex-park, b70/fn-rt-park2 from b70/fn-rt-kv8.
  B3-6 host side: b70/fn-rt-admit 853d273 accepted as prototype (18/18 host tests; sim [I]: C4x200K 0.25% lost, C8/C16x200K
  15.5% lost with full-copy parks -> write-behind mirror needed); fairness fixed + mirror model e0db34d [I: C16x200K lost 20.6% -> 9.3% with mirror].
  Prefill (B/prefill/CRITICAL_PATH.md): same lease, 2048 ids, chunk 512: pre-KF 55.8 -> fn-a3 integration 362.2 tok/s
  (6.49x; first token equal). Pre-KF 20K trace: dense GEMM 60% of busy, experts 25-28%, HC 6%, AR 2-5%, GDN 0.4%; the
  16-row tile kernels re-read weights per tile. ~6x more needed for 2200 tok/s.
- PROPOSAL TO LEAD A (prefill DPAS): DESIGN-A lists "DPAS tiles for R > 16 (prefill)" under M-A5. Since cold C4 TTFT is in
  M-B3 and KF is on decode gaps, Lead B has started a standalone, default-off DPAS prefill GEMM (dense first, then grouped
  experts) on b70/fn-rt-prefill3 with parity, determinism and row-invariance tests, card 1 only, no edits to your trees; you
  wire it. Say so in your section if you would rather own it or have KF do it, and Lead B will hand over the branch.
  DPAS (standalone, card 1, device timestamps): Q6_K dense M=512 7.4x / 8.5x vs Q8_1 (QKV 2560x5120 295 us, out 3072x2560
  156 us; GEMM 69-97 TOPS), M=2048 9.9-10.7x; fp64 rel L2 2.1e-4, bitwise row-invariant, repeatable. Grouped IQ2_XS/IQ3_XXS/
  IQ4_NL expert projections correct; expert speed vs baseline being measured.
- Next GPU need: both cards in short runs (kv8 gates, park2 gates/latency), card 1 short runs (DPAS prefill microbench).
