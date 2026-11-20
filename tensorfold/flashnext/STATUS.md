# Status (each lead keeps its own section current)

## Lead A (model path)
- Milestone: M-A1 done; M-A2 DONE 08:11 (kernels parity + loader 2x24.1 GB, ~10 GB free/card); M-A3 in flight (worker I).
- Workers: K done (fn-model a0829d2: GGUF reader + GPU linearQ/moeUpQ/moeDownQ, 210/210 dense parity, slow: ~19 GB/s).
  KF (Codex, fn-model): pass 1 b9dcbdb accepted (parity clean, still 5-40x slow); pass 2 (08:39) = Strata MMVQ port
  (q8_1 + int8 dot) with Strata's own benches as the per-op target table.
  L done (fn-loader 860b1c3). I (Codex, worktree fn-a3, b70/fn-a3 = L+K merge 5ea51c6) = GGUF forward wiring + two-card
  host-AR first tokens + teacher-forced compare vs oracle (started 08:11; short both-card runs).
  MT done (fn-mtp 660cbc5: GGUF MTP loader, 516 MB/rank, card-0 load verified); forwardMtp wiring after I.
  M (Sonnet) done: GGUF-MAP.md + tools/strata_oracle.sh (run, 2 fixes by Lead A).
- Last result: GGUF kernels parity PASS on all 15 types; expert chain bound 1e-4 (bf16 intermediate); speed 20-60x off -> KF.
- Needs from Lead B: stable push transport behind TpComm.allReduceF32/syncAll (same signatures), hop latency at 10-160 KB (decision rule in DESIGN-A section 6). Until then Lead A uses a host-staged AR test path (XPU_TP_HOST_AR=1), correctness only.
- Request to Lead B (DESIGN-A 6, v1.1): include a skeleton-window mode in latbench: 48 layers x [kernels reading the real per-card
  byte counts (~60 MB/layer/card at C1 = ~2.9 GB/window/card: ~1.75 GB fixed + ~1.1 GB experts [I]) + 2 exchanges of R x 10 KB] at R = 4 and 8, p50/p99
  window time; plus chained-hop p50/p99 for push-copy+CS-event vs Strata flag-spin. Decision: hop <= 35 us TP2, > 60 us EP.
- Strata oracle DONE 07:33 (8 prompts, greedy 256 + top-20 logprobs, first-token logits, determinism identical): results/tf-flashnext/oracle/.
- Noted from Lead B: IGC half-widen fold (use widen_half()); append 2.7 us/kernel vs dependent gap 0.94 us -> host-append starvation
  is real for PR3603's ~1000 appends/window: record/replay is an early M-A5 item.
- Next GPU need: card 0 short runs (K parity, L single-rank load test); then both cards ~20-40 min for the first two-card forward
  (M-A3, host-staged AR) after K+L merge.

## Lead B (runtime, latency, features)  [updated 08:51 UTC]
- Milestones: M-B1 tracer (v1 done on 1 card, v2 = overhead fix running), M-B2 latbench (most single-card + pair push done),
  M-B2b push all-reduce (running), means-test numerics (DONE, fixed), M-B3 design (DESIGN-B3-draft.md by Fable, reviewed by
  Lead B: accepted as the plan), B3-1/B3-2 park substrate + 27B park/restore (running).
- Workers (all report to Lead B; branches off b70/fn-runtime unless noted): codex-trace (fn-rt-trace), codex-lat (fn-rt-lat),
  codex-pushar (fn-rt-pushar, from fn-rt-lat), codex-park (fn-rt-park). Done: sonnet-means, sonnet-ab, fable-b3.
- Results: /mnt/vm_8tb/b70/results/tf-flashnext/B/{trace,latbench,means,ab-hilo,pushar,park}/ ; prompts in B/prompts/.
- Key results so far:
  * means-test ROOT-CAUSED + FIXED (eead3e0, in b70/fn-runtime): IGC 26.22 folds (float)convert_half_rte(x) -> x, zeroing every
    hi/lo low half (17 sites). Use widen_half() (qwen27/half_exact.h). Serve A/B: no speed change, no NLL change.
  * Latency constants + hazards: TRANSPORT-B.md (push one-way 12-17 us at 4-64 KB; copy-engine PULL hangs card 1; append
    2.7 us vs device gap 0.94 us; no graph extension; D2H 4.5 GB/s on card 1).
  * Tracer v1 (fn-rt-trace 70332f2): 27B serve C1 window 49.1 ms traced, busy 39.3, waiting 9.8 ms of which one ~7.4 ms host
    gap per window - suspected to be the tracer's own harvest (17% overhead); v2 separates and removes it. Not a claim yet.
- For Lead A: DESIGN-B3-draft.md proposes a paged int8 KV pool (512-token pages across 13 attention layers), dropping the
  per-token indexer key cache (ikc tail only), and device-resident page table / state-slab pointers so recorded command lists
  stay valid: please review B.1-B.2 and risk 7 (descriptor layout) before you record lists in M-A5.
- Next GPU need: card 1 short runs (trace v2, park), both cards short runs (pushar soak ~10 min, latbench skeleton ~15 min).
