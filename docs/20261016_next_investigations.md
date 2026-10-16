# Next investigations (user, 2026-10-16): card balance, prefill, single-stream decode, C4 aggregate

The daily driver (Strata night-final-v3, 2x Arc Pro B70, expert parallel) measures C1 ~141 tok/s, C2 ~92/stream,
C4 51.3/stream (188 wall aggregate, 64K context), prefill ~2200 tok/s at 4K-32K and ~1600 at 200K
(docs/20261016_night_lab.md). The user's top item is the card imbalance: card 0 does nearly all the work while
card 1 mostly waits. Ranked list below; each item names the measured facts it starts from and what was already
tried. Measure every idea with the night harness (strata/flash-next/night/nl.py, fixed prompts, same binary A/B,
acceptance-normalised for non-bitwise changes) and the NLL gates (nl_ppl.py).

## 1. Card balance (top priority)

Facts (unitrace of the final stack, C1, strata/flash-next/night/agent-reports/trace_final_analysis.md):
- Per C1 verify window (29.3 ms profiled, ~22.9 ms unprofiled): card 0 computes ~20 ms (attention, GDN, HC,
  every dense GEMV, shared expert, vocab head, the MTP drafter, and its half of the routed experts) and is idle
  8.6 ms (host and launch gaps). Card 1 computes 4.9 ms (its routed experts) and spins 15.5 ms in ep_wait.
- C2 (S136 trace): card 0 computes 42.6 of a 57 ms cycle.
- Prefill at 32K: card 1 busy ~25 % before the halves pipeline (night start).
- Layer split instead of expert parallel balances better (card 0 71 %, card 1 56 %) but decodes slower (C1 81).
- Card 1's routed kernels run 20-30 % slower than card 0's; card 0 now waits 0.57 ms per C1 window on card 1.
- Tried, no gain: EP hot-expert share knob (STRATA_NIGHT_EP_SHARE0). pipeline-windows 2 crashes on SYCL (#2800).

Candidates, roughly by expected gain per effort:
a. Shared expert on card 1. Card 1 already receives each token's activations for its routed experts; computing
   the shared expert (1.14 ms per C1 window on card 0, latency bound at 18 % of roofline) there and returning it
   with the PSUM return adds no transfer. Fits inside card 1's 15.5 ms spin.
b. More routed experts on card 1 (static capacity rebalance, not the hot-expert share): card 0's routed GU + down
   cost ~3.5 ms per window at roofline; card 1 has ~15 ms of slack. Bounded by card 1's VRAM: account per card
   first (gguf_budget.py, the expert cache lines in the engine log).
c. Split large dense work across the cards per layer: HC projections (3.28 ms, the largest card-0 item, 38 % of
   roofline), the vocab head (0.83 ms), out-proj / gate / attn_q GEMVs. Needs a per-layer hidden-state exchange;
   measure the push latency per hop first (the EP exchange is already GPU-side) - 48 layers x hop must stay well
   under the saving.
d. MTP drafter or the vocab head on card 1 (2.25 ms + 0.83 ms per window), overlapped where the data dependence
   allows (the drafter needs the verify output; the head of slot B can overlap slot A at C2+).
e. Why card 1's routed kernels are 20-30 % slower (PCIe topology / root complex, clocks, kernel config): a fix is
   worth ~0.57 ms per C1 window directly.
f. Cross-window pipelining at C2+ (card 1 runs one slot's stage while card 0 runs the other's): the layer-split
   route needs #2800 fixed first.

## 2. Single-stream decode (C1 141 -> 150+)

Top remaining card-0 costs per C1 window (same trace): verifier in-order dispatch gaps 3.36 ms (1,575 launches,
683 under 5 us; partly profiler-inflated - unprofiled host idle is ~1-1.3 ms), HC Q8 v2 3.28 ms (local optimum
for one card; see 1c), host gaps verifier -> commit 3.17 ms and MTP -> verifier 1.40 ms (PLE ~0.67 ms real),
routed GU 2.26 ms (at roofline), dense out-proj 1.48 / qkv 1.47 / gate 1.34 ms, GDN 1.39 ms, shared expert 1.14 ms.
Candidates:
- Card balance items 1a-1d (the biggest lever left).
- IQ3_S dense at 1-4 columns: gate at 16 % of bandwidth; the IQ3S_LO kernel was bitwise in the bench but not in the
  engine - find the diverging call site, then re-measure.
- Deferred GDN commit fused into the next window's verify (deltas already saved by GDN2): ~-0.25 ms per window.
- Launch count: fewer, larger kernels in the verify graph (Q8 quantize 0.55 ms and EP plan 0.55 ms sit at the
  launch floor).
- MTP: `--parallel 1 --spec 6 --mtp-max-t 6` gives ~145 for a single user; acceptance work (draft vocab, deeper or
  tree drafts) with the 2-slot config.

## 3. Prefill (~2200 tok/s at 4K-32K, ~1600 at 200K)

- Card 1's share of the prompt path (item 1 applies: dense prompt GEMMs, attention and GDN all run on card 0).
- The XMX union prompt attention only pays where neighbouring selections overlap (even at a 5.2x union at 32K);
  long prompts (200K) need a profile of attention vs GDN vs experts per chunk.
- GDN chunkwise prefill: the FP32 version was 4.4x slower than the scan; a BF16 / XMX chunk form is untested.
- Prompt chunk size (12288) and the prefill dequant caches against VRAM; the first request after load is slower
  (1893 vs 2204 at 4K).

## 4. Aggregate decode at C4 (188 wall, 51.3/stream at 64K context)

- Card balance matters more at C4 (card 0 carries four slots' dense work).
- C4 runs batch MTP depth 1 with two MTP pairs; depth 2 with 12-row windows was slower. Kernels past 8 columns
  (the KREG family covers 1-8; MMVQ2 had a 9-16 variant) and batched attention at 4 slots are the levers.
- Context per slot is limited to 64K at C4 by VRAM; RAM parking covers idle conversations.

## 5. Reliability and follow-ups

- The SYCL port has no non-finite-logits guard (upstream #879 is CUDA/HIP only): port it (also an upstream PR).
- #2800 (`--pipeline-windows 2` on SYCL).
- Upstream reviews: Niko1221/Strata #2802 #2804 #2806 #2808 #2810 #2812 #2817 and SergiioB/b70-launcher#4; next
  upstream candidates: KREG BF16 GEMV + fused shared expert (stacked on #2804); the EP-specific changes once the
  expert-parallel PRs #1798/#2205 land.
- Quality: evals/abfast A/B of night-final-v3 against the 27B fallback (still pending).
- Measurements owed: C2 text vs images matched A/B (84.5 vs ~92 per stream, noisy; the user keeps images on);
  GLIBC_TUNABLES hugetlb effect; a localmaxxing re-run after the next speed gains.
