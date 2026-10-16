# Next investigations (user, 2026-10-16): elastic 4-slot KV (build next), card balance, prefill, single-stream decode, C4 aggregate

The daily driver (Strata night-final-v3, 2x Arc Pro B70, expert parallel) measures C1 ~141 tok/s, C2 ~92/stream,
C4 51.3/stream (188 wall aggregate, 64K context), prefill ~2200 tok/s at 4K-32K and ~1600 at 200K
(docs/20261016_night_lab.md). The user's top item is the card imbalance: card 0 does nearly all the work while
card 1 mostly waits. Ranked list below; each item names the measured facts it starts from and what was already
tried. Measure every idea with the night harness (strata/flash-next/night/nl.py, fixed prompts, same binary A/B,
acceptance-normalised for non-bitwise changes) and the NLL gates (nl_ppl.py).

## 0. Elastic KV: 4 slots up to 200K each, overflow parked to RAM (user, 2026-10-16: investigate and build next)

Serving rule (user, 2026-10-16): the DD runs either 2 x 200K (today) or 4 x 200K (once this lands). A reduced
per-slot context (e.g. 4 x 64K) is not a DD option.

Want: serve `--parallel 4` with `--max-context 200000` per slot. Most conversations stay far below 200K, so VRAM is
committed only for the cells a slot actually reaches. When the live slots together outgrow VRAM, the least recently
active slot's KV goes to system RAM and is restored later without recompute. Two full 200K contexts resident at once
remain the guaranteed floor (the goal in AGENTS.md); four long ones run with parking.

Why now (measured on the live DD, 2026-10-16 ~23:00 UTC): a burst of agent traffic held 13-14 requests in flight
against the 2 slots, so ~12 waited in queue. One prompt was still being read 57 s after it arrived, and prefix-cache
reuse fell from ~1000-1500 to ~70-140 tok/s as clients pushed each other's cached prefixes out. C4 (4 slots at 64K)
gives ~13 % more aggregate (185-188 vs ~164 wall) and twice the admission width, but its 64K cap breaks long agent
sessions.

Facts:
- KV is allocated per slot for the whole `--max-context` at load. Engine log: `--batch 2` slot sessions take
  5.63 GiB of VRAM on card 0 at 200K (expert cache 10.77 GiB, 6624 slots); `--batch 4` at 64K takes 4.01 GiB
  (expert cache 14.52 GiB, 8928 slots). A static 4 x 200K would need ~11.3 GiB and does not fit beside the expert
  cache.
- The existing `--kv-grow` (elastic KV, src-ep sycl/src/program/generate.cpp:6429 KvGrow / kvg_ensure,
  sycl/src/core/layer.cpp:579, docs/DETAILS.md) already does on-demand VMM mapping. It takes chunks from the expert
  cache and hands them back, and captured graphs stay valid because addresses never move. Its limits: one GPU only,
  and it turns itself off with `--batch`, `--vram-elastic` or `--peer-device`. All three limits apply to the DD (batch
  MTP, expert parallel across two cards).
- The conversation cache (`--conversation-cache-mib 32768 --conversation-cache-slots 4`) parks idle conversations to
  RAM and restores them without recompute (~415 ms per 200K restore, replay 0; parking ~49-75 ms per admission with
  PARK_MOVE). It does not evict a slot that is still running.

Investigate, then build (small, measured, default-off flag):
a. Read what blocks `--kv-grow` under `--batch` and expert parallel: per-slot KV pools versus one range, quiesce
   across both cards' queues, and the expert-cache slot hand-back on card 0 only (KV lives on card 0; card 1
   holds routed experts).
b. Per-slot elastic KV under `--batch 4`: grow in steps (STRATA_KV_GROW_STEP 8192), with a reserve that guarantees
   2 x 200K. Measure what each grow costs (quiesce + map) during decode of the other slots.
c. Pressure eviction: when a grow cannot be met, park the coldest slot (generating or waiting on a tool call) with
   the existing park path, unmap its chunks, and restore it on its next turn. The client sees TTFT, never a
   recompute or an error.
d. Decide the expert-cache trade with numbers: each GiB of KV costs ~615 expert slots on card 0. Measure C1/C2/C4
   decode with the cache shrunk to match 2 x 200K live.
e. Gates: C4 decode matched against night-final-v3-c4 (target: no regression at 64K), C2 200K needle x 2, a
   4-conversation 100K+ forced-eviction run (restore without recompute, NLL gate), the abfast agent suite under
   14-client load, and teardown/health.

## 1. Card balance (top priority after item 0)

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
