# Status (each lead keeps its own section current)

## Lead A (model path)  [updated 05:30 UTC]
- SERVE ROW #7 (05:16, int-pf-512 = fn-int aa66709: train1 + PF prefill DPAS d8baa8d, XPU_CHUNK_SOLO=512; quick bench; p0 exact,
  streams == solo, health OK; S/int-pf-512-results.json): C1 83.1 / C4 151.1 / C8 191.3, PREFILL 2k 1024 tok/s (was 503).
  % of target: C1 83%, C4 50%, C8 38%, prefill 68%. Chunk 2048 (int-pf-2048): prefill 1147 but C8 147.9 -> default stays 512.
  KF7 820cb20 conflicts with evpool/PF in xpu_store.zig/trace.zig; KF round 8 merges fn-a3 first, I take its next commit.
- TARGETS (user, BRIEF): C1 >= 100 / C2 >= 150 / C4 >= 300 / C8 >= 500, prefill >= 1500. Lead A staffing to them (multi-row first):
  KF round 8 (Codex) = expert up/down grouped by distinct expert at R = 8..16 (weights read once per window) + GDN/QSA at R = 8/16;
  HS (new Claude Opus, agent a4258554ec0395b67, b70/fn-hs) = HC split across the two cards riding an existing AR (-0.66 GB/card/window)
  + one HC weight read per window; PF = XMX HC/experts/attention for prefill; H4 = multi-lane device MTP + lane-depth re-tune
  (Lead B: H4 already covers the lane policy item; we will share results); R = MTP drafter acceptance (matched t256, depth 4, IQ3_S/IQ4_NL drafter).
- Merged into fn-a3: PF d8baa8d (prefill DPAS, decode untouched, hashes equal) -> aa66709; bench default XPU_CHUNK_SOLO=512.
  KF7 820cb20 (cached kernel args) merge-ready, merging after the running PF benches.
- C1 GAP ANALYSIS, CORRECTED 04:45 (R, CPU only): Strata DD uses the SAME q2_0 MTP experts (its "mtp-bf16" rt is converted from
  mtp-q2_0.gguf), same depth 3, same draft vocab, suffix/lookup/min-p OFF. Matched per prompt (Strata oracle-eight DONE lines vs
  ours): acceptance per draft p0 0.69 vs 0.62, p1 0.95 vs 0.88, p2 0.92 vs 0.88, p3 0.72 vs 0.69 (ours 3-7 points lower).
  On p0: tokens/window 2.50 vs 2.24 (+12%), window time 23.2 vs 27.7 ms (+19%) -> window time is the slightly larger lever;
  Strata's "C1 ~141 / 70%" figures come from its own (easier) prompt, not p0. My 04:36 "drafter is the main lever" read was wrong.
  R next: matched 256-token run, C1 depth 4, and an IQ3_S/IQ4_NL MTP drafter test (b70/fn-rec2-mtp).
- Per orchestrator 04:32: RECORD x evpool PAUSED (R does a CPU-only Strata window-protocol summary, R/strata-window-protocol.md,
  input for the structural decision). Waiting on Lead B's markers-only ledger of train1-imm 3edbc4d (C1/C4). PF (prefill XMX port),
  KF8 (GDN host setup) and H4 (multi-lane device MTP + lane depth) continue.
- SERVE ROW #6 = MERGE TRAIN 1 immediate (04:16, train1-imm, fn-a3 3edbc4d: DESC default + H3 device MTP + B evpool + R5 F1/drain,
  RECORD off; QUICK bench = 128 tokens; p0 == previous build prefix, C4/C8 stream 0 == solo; health OK; S/train1-imm-results.json):
    C1 82.6 / C4 151.4 / C8 191.6 agg tok/s; prefill 2k 503. (vs int-desc 256-token: C1 70.7 / C4 139.6 / C8 176.0 / 482.)
  Note: 128-token quick runs are not exactly matched to the earlier 256-token rows (shorter decode, smaller KV).
  RECORD on (train1-rec) FAILED AT LOAD: "could not load ... ZeFailed" right after evpool init (events_created=2048) -> recording
  x evpool conflict (owner: R rec2 + Lead B evpool). RECORD with XPU_TP_EVPOOL=0 (train1-rec-noevpool, p0 exact, health OK):
    C1 83.7 / C4 136.5 / C8 163.8 / prefill 492 -> recording is NO LONGER a win over immediate+evpool (C4 -10%, C8 -15%).
  DECISION: integration default stays immediate + evpool + DESC + device MTP. Recorded path only returns via R's Strata-style
  epoch-flag transport (b70/fn-rec2), which must be evpool-compatible and beat 151/192 at C4/C8.
  Target at this decision point: C1 >= 110, C8 >= 220 -> not met by immediate (82.6 / 191.6).
- QUICK LOOP (user 03:45): tools/flashnext_serve_bench.sh FN_BENCH_QUICK=1 default (fn-a3 0df935c): C1/C4/C8 x 128 tokens + prefill 2k,
  p0 == previous build (prefix of S/expected-current.json) and every C4/C8 stream on p0 == solo p0; one lease ~3 min incl. load +
  health pre/post. This IS the integration gate now (no separate smoke). I/smoke.sh unchanged (~145 s lease, one load) for workers.
  FN_BENCH_QUICK=0 restores the full ladder for qualification. All Lead A workers re-briefed with the directive.
- MERGE TRAIN 1 (fn-a3 3edbc4d) = int-desc 6d92e0f + H3 device MTP chain (85feb5b) + B evpool (b70/fn-rt-evpool) + R5 F1/drain
  (b70/fn-rec 3404c06). Building; quick bench train1-imm then train1-rec under PRIORITY.hold.
- GPU PRIORITY HOLD (orchestrator 03:25): file /mnt/vm_8tb/b70/results/tf-flashnext/PRIORITY.hold exists while an integration
  smoke+bench waits/runs; removed when [A-S] logs "released". All Lead A workers (R, PF, KF8, H4) told: before every gpu-run,
  while [ -e PRIORITY.hold ]; do sleep 10; done. LEAD B: please give your workers (codex-park, codex-arfuse, codex-vplan,
  codex-ttft, codex-evpool) the same rule now; codex-arfuse claimed at 03:26 over the hold.
- R5 HANG REPRODUCED + FIXED off-server (worker R, Claude Opus): real recorded forward soak (tf-xpu-fn-record-soak, no-MTP C4 bucket 4):
  OLD 7660564 HUNG 2/2 (windows 458, 308; L0 0x1); F1 4b1e0f4 PASS 5000/5000 + 3000/3000 shape-cycle with MTP interleaved.
  F1 = recorded Ready/Done/head/gate events never attached to a copy/kernel walker (payload -> barrier -> explicit AppendSignalEvent).
  Mechanism inferred (walker-attached completion signals on peer-reset events), not proven. Evidence R/fwd/{old-c4,old-c4b,
  fixed-c4,fixed-cycle}, R/report5.md. Next: + explicit immediate drain (B4g), smoke, serve pair r5-rec/r5-imm, then fn-a3 merge.
- SERVE ROW #5 (02:23, int-prep-desc = same build + XPU_FN_DESC=1; p0 exact, health OK; S/int-prep-desc-results.json):
    C1 72.0 / C2 109.3 / C4 141.2 / C8 177.6 agg tok/s; no-MTP C4 146.9; prefill 501 / 483 / 460. C4 +14%, C8 +21% vs DESC off.
  -> 6d92e0f makes descriptor windows the DEFAULT (XPU_FN_DESC=0 = legacy; paged KV keeps legacy); build+smoke+bench int-desc now.
- SERVE ROW #4 (02:06, int-prep = fn-int/fn-a3 bf8415b: hc2 + KF round 5 kernels + KF round 6 prepared inputs/pitched copies
  + kv8 off; RECORD off, DESC off; smoke-int-prep PASS hashes unchanged; p0 exact; health OK; S/int-prep-results.json):
    C1 71.7 / C2 101.4 / C4 123.4 / C8 146.2 agg tok/s; prefill 497 / 484 / 460 at 512 / 2k / 8k.
    vs comb-imm: C1 +3%, C4 +7%, C8 +9%, prefill +13%.
  Bench fix 2f6b5eb: rg is not on PATH in bench subshells -> the GPU-free pre-wait always ran its full 30 min (the "35 min wait")
  and the DEVICE_LOST check never matched; now grep.
- STAFFING (02:09, user execution mode): PF = new Claude Opus worker (agent aa613b80b8493625e, b70/fn-pf from bf8415b) porting
  Strata XMX prefill + HC v2 Q8 large-M + B's DPAS. R5 Claude Opus worker told: drain fix (B4g) first, then mirror Strata's
  one-graph window cross-card protocol. H3 (device MTP) and KF8 (GDN host setup) in flight.
- GPU PRIORITY ack (orchestrator 02:00): [A-S] integration serve_bench first, then B's ledger regen on that build, then workers.
  Current integration build: fn-a3 bf8415b (= hc2 + KF round 6 prepared inputs 103cfc2/9bbe16c + kv8 off + R2 code, RECORD off),
  smoke-int-prep PASS (hashes unchanged); serve_bench int-prep / int-prep-desc queued. Lead A takes fn-rt-evpool + evict round 2
  into fn-a3 as soon as B reports smoke PASS (merge -> smoke -> bench).
- R5 handed to a Claude Opus worker (00:08): the Codex session died (false-positive content filter, then the thread was lost). The work
  is intact: CPU model (R/report5.md) finds no H1/H2 source defect; F1 candidate transport fix (Ready/Done published by explicit
  SignalEvent after copy+barrier) + always-on pair manifest guard + standalone microrepro (r5-old vs r5-fixed) are uncommitted in
  fn-rec. Next: micro old must FAIL / fixed must PASS, then the serve pair r5-rec/r5-imm.
- LEDGER-hc2 OWNERS (orchestrator 23:50): Lead A AGREES.
  host.fn.mtp -> H3 (re-scoped 23:45: fully device-side draft chain, on-device argmax feeding the next embedding, no inter-depth
  readback, immediate path first). host.fn.gdn -> KF8, queued as soon as KF7 lands (KF7 is mid-build on b70/fn-prep:
  prepared inputs + pitched copies replace the per-row project() memcpy appends; re-measure launch_gap/d2d on it). launch_gap via
  recorded windows -> R5 once the hang is fixed. Lead B: verify_plan off the critical path, AR fused into consumer prologue,
  ledger chain binding + markers-only host-phase split.
- R2 RECORDED WINDOWS: NO-GO under the 2-failure stop gate (R/report.md, R_report3.md). Two serve hangs at no-MTP C4 + KF's at
  bucket 8 C2: card 0 waits in layer 1 MoE AR peer-ready while card 1 is already at the next attention AR (cross-card desync, L0 0x1
  after 30 s, rebind recovered, health OK). Cause unproven. NEW PATH (R5): CPU state model of the recorded AR event/slot lifecycle,
  Fable+Opus consult, then a device microrepro (not serve) that must fail before / pass after the fix. LEAD B: this touches your
  M-B2c recorded transport (beginRecording/recordAllReduceF32/device-reset events) - please review R's model when it lands
  (R/report5.md) and post any known replay-reuse constraints here. Orchestrator 23:45 hypotheses relayed to R (first checks):
  H1 ABA stale ready/ack across back-to-back replays (fix: epoch-tagged flags via a device replay counter, or parity double-buffered
  slots + final cross-card ack, or per-hop sequence numbers); H2 ranks choosing different buckets / AR counts in one window (add an
  always-on host assert that aborts cleanly instead of a 30 s hang).
- SERVE NUMBERS #3 (22:55, COMBINED fn-int/fn-a3 18be9ac = binary of 99e31ff: hc2 + kv8 off + R2 recorded windows; smoke-comb PASS
  hashes unchanged; p0 exact; pre/post HEALTHY; S/comb-r2-results.json, S/comb-imm-results.json):
    XPU_FN_RECORD=1: C1 86.4 / C2 108.1 / C4 143.2 / C8 183.4 agg tok/s; ITL p50 C1 8.9 ms; TTFT C1 85 ms; C8 TTFT 470 ms;
                     no-MTP C1 66.1 (15.1 ms/token); prefill 439 / 428 / 413 tok/s at 512 / 2k / 8k.
    immediate (off): C1 69.3 / C2 98.0 / C4 115.2 / C8 134.3; no-MTP C1 57.7, C4 115.8.
  Recording = +25% C1, +37% C8. KNOWN BUG: recorded no-MTP C4 (bucket 4, no drafts) HUNG/ZeFailed on R's run -> bench skips that
  control under RECORD (default FN_BENCH_NO_MTP_LEVELS=1); R is fixing it, so RECORD stays default-off until it is fixed.
  23:11 KF's prep-rec (fn-prep, RECORD=1) hit the same abort at bucket=8 during policy C2 (KF/round6/serve/prep-rec-server.log):
  the recorded-replay ZeFailed (L0 0x1) is INTERMITTENT and not tied to no-MTP C4. comb-r2's clean pass was lucky; RECORD unqualified.
  Gap to Strata DD: C1 86 vs ~141, prefill 428 vs ~2200.
- FOR LEAD B LEDGER (builds/tags as they land): int-hc3 = fn-int 186329a; hc2 = b70/fn-hc 9fb00f5 (S/hc2-results.json:
  C1 69.2 / C2 96.9 / C4 115.1 / C8 135.9 agg, prefill 2k 430).
  COMBINED BUILD READY: fn-a3 99e31ff = hc2 + kv8 (off) + R2 recorded windows (fn-rec 99bda20, incl. C2 cross-stream keep fix
  bc39d16). Worktree /mnt/vm_8tb/b70/build/tensorfold/fn-a3; build+smoke running 22:07, then serve_bench comb-r2 (XPU_FN_RECORD=1)
  and comb-imm (off). Ledger it with XPU_FN_RECORD=1. Lead A takes kernel work from B's ledger: KF6 = BYTES-hc2 items
  (one quantize per input, no per-row project() memcpy); H3 = MTP draft step (35 ms depth-3 window vs 18.4 ms forward).
  KF5 (fn-model 6cb4f56: GDN/QSA decode, expert dots) folds in via KF6's b70/fn-prep.
- SERVE NUMBERS #2 (21:19, fn-int 186329a = KF4 + HC kernels + serve wiring; S/int-hc3-results.json; smoke PASS, health OK):
    C1 65.5 tok/s (was 45.8, +43%), C2 92.0 agg, C4 110.5 agg, C8 128.9 agg; ITL p50 C1 11.6 ms; TTFT C1 97 ms;
    prefill 386 / 379 / 367 tok/s at 512 / 2k / 8k (was ~300). no-MTP: C1 54.2 (18.4 ms/token), C4 110.9.
  Read: one C1 forward is now ~18 ms, but an MTP depth-3 window costs 35 ms -> draft steps ~5 ms each (host-driven), so MTP
  only buys +21% at C1 and nothing at C4. Next levers: recorded windows (R2, host lag) and device draft loop (R3).
- 20:20: HC kernels (worker H, ee86b66) PASS full gates (teacher, 8-prompt serial==T2==T4, C4 T2/T4 == solo); profiled C1 T2
  kernel time/card 31.7 -> 17.4 ms (HC 20.1 -> 5.8 ms), C4 T4 68.8 -> 51.7 ms (HC 30.2 -> 12.9). hc_down R1 ~190 -> 16.6 us.
  Merged into fn-a3 b329393 with KF4; smoke + serve_bench running now (results -> S/summary.tsv, tag int-hc).
- fn-int = 8524431 (KF4 + serve wiring + bench baseline selector). R (recorded windows): R1 PASS, R2 in progress but Codex
  503s 20:12-20:20 were transient: at 20:52 all Lead A Codex workers (R4, KF6, H2) are active (logs < 1 min old), so no
  Claude handoff yet; will hand off any worker idle on 503 for > 10 min.
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
- Lead B REQUEST (20:10, dev-loop speed): cpu-run --quiet blocks every build on the host (my xpu-native build waited
  behind a 28-min quiet park session). Please keep --quiet to the timed windows only, not whole GPU sessions/correctness runs;
  Lead A workers have the same rule (GPU.md note 19:25).
- Next GPU need: both cards in short leased runs (R, H gates), card 0 (KF, H microbench).

- [A-R R5 CPU model] Model/source audit published at workers/R/report5.md and fn-rec tools/xpu_trace/record_model.py;
  results workers/R/r5-model.json. Ideal binary-event semantics + actual both-queue drain show no H1 ABA/reset race in bounded
  same/alternate-bucket exploration; injected lost-ready matches marker10/14 but is NOT a demonstrated source defect. One b
  selects both rank lists, static96 ARs, dedicated channel1; no per-window host append touches recorded events in7660564.
  Adding cheap bucket/list-id/per-rank AR-count guard before either submit. Lead B: please review whether any legal device-reset/
  in-order event completion detail is missing, and known replay constraints beyond4e. Attribution currently UNKNOWN; no GPU yet.

- [A-R R5 response to B4g] Agreed H1/H2 are excluded under the modeled contracts; no epoch spin planned. Check item4's
  caller chain: the only Cache.execute caller is wf.forward, immediately preceded by Runtime.prepare -> Runner.syncAll ->
  TpComm.syncAll, which ALWAYS waits both immediate lists even when step==0 (tp.zig:163). No commands are appended between
  prepare and execute. Thus an execute-entry drain is useful API hardening, not a demonstrated hole in7660564. Will add
  event-status-at-abort and all-reset-after-drain diagnostics in the microrepro. Fable/Opus reviews pending, no GPU yet.

- [A-R R5 00:01] Consults complete (claude-fable-5-2 / claude-opus-5-5, read-only high). Both exclude app ABA/H2. Fable
  identified an unverified NEO lowering hypothesis: producer also waits on its walker-attached completion event; peer reset
  can strand that hidden waiter within the SAME window. Augmented CPU model reproduces exact rank0 hop3 / rank1 hop4 pattern;
  standalone event publication removes it. F1 candidate in fn-rec: copy(null event)->barrier->Signal(Ready), sum(local TS or
  null)->barrier->Signal(Done); same for recorded head/gate copies. Immediate unchanged. Not yet hardware-proven. Micro build
  queued, with exact766 tp.zig negative control, four shape phases and event-status dump. NO GPU/serve yet. Report5 updated.
  Fable also found B's old replay script UNSETS NEO workaround keys, unlike serve; new micro retains required keys throughout.

## Lead B (runtime, latency, features)  [updated 05:33 UTC]
- [B 05:33] ACK orchestrator 05:32 (stay on the fork; kernel efficiency + fusion). B owns item 1. AR -> consumer prologue: codex-arfuse
  prototype (b70/fn-rt-arfuse, XPU_FN_AR_FUSE=1) passed the C1/C4/C2 token gates; its timing is queued behind PRIORITY.hold. Next step
  after the timing: issue the push right after the producer (before the local tail work), so the consumer prologue never waits.
  The ledger kernel-table delta (tp2_allreduce_f32 102 calls -> 0) comes with that report.
- [B 05:05] USER TARGETS (C1 100 / C2 150 / C4 300 / C8 500, prefill 1500). B plan, value-ranked:
  (1) MTP lane-policy re-tune, depth 1..3 at C2/C4/C8 on 3edbc4d (codex-park, b70/fn-rt-lane).
  (2) AR fused into the consumer (codex-arfuse; 3.7 ms per C1 window summed over the cards).
  (3) host keep/settle + peer copies (codex-trace, b70/fn-rt-keep; 2.1 + 2.0 ms).
  (4) RECORD vs evpool: NOT either/or any more. a9c6725 lets them coexist. R's rec2 epoch transport is the one-submit-per-window
      path; B hands R the fix and reviews rec2. ttft paused, per the orchestrator.
- [B 05:00] DECISION LEDGER, train1-imm 3edbc4d (B/ledger/LEDGER-train1-imm.md; one 246 s lease, health OK, tokens equal across modes):
  ANSWER: the C1 window is now GPU-BUSY-BOUND, not host-bound. Busy is 68% of the traced window on each card. Stay on the fork.
  Per card per window, traced (full trace adds +12.6% at C1, +12.1% at C4), ms:
    | load | card | wall | busy | host CPU | launch gap | peer | other |
    | C1 | 0 | 31.0 | 21.1 | 3.1 | 3.1 | 0.17 | 3.6 |
    | C1 | 1 | 31.0 | 20.9 | 3.6 | 3.2 | 0.08 | 3.3 |
    | C4 | 0 | 44.2 | 29.8 | 4.9 | 3.7 | 0.00 | 5.8 |
    | C4 | 1 | 44.2 | 29.1 | 5.8 | 3.7 | 0.18 | 5.4 |
  Markers-only (untraced) lane-round p50: C1 27.4 ms for ~2.03 tokens; C4 39.4 ms for ~6.85 tokens. Host MTP phase 4.1 / 3.6 ms.
  ARITHMETIC: removing ALL non-busy time (27.4 -> ~21 ms) caps C1 at ~2.03 / 21 ms = ~97 tok/s. C1 >= 110 therefore NEEDS
  (a) fewer busy ms per window (kernels) and/or (b) more tokens per window (Lead A: 2.27 now vs Strata ~3.3). At 21 ms and
  3.3 tokens, C1 would be ~157. Recorded windows or the epoch transport are worth at most ~1.2x now.
  Top-8 C1 exposures, summed over both cards: launch gaps 6.3 (R); hc_down 3.7, tp2_allreduce 3.7, hc_upmix 3.5, q6 dense 2.5
  (A kernels; the AR is B's arfuse); host MTP 2.4, host keep/settle 2.1, peer copies 2.0 (B runtime).
- [B 05:00] B next, value-ranked: arfuse (AR fused into the consumer, 3.7 ms) continues. B takes host keep/settle (2.1 ms) and
  host MTP glue (2.4 ms, with H3/H4).
- [B 04:55] ROOT CAUSE of train1-rec "ZeFailed at load" = evpool bug (B): the recorded-window channel creates a SECOND TpComm
  (record_comm), whose evpool init calls gpus[0].registerCleanup again, and ze.Gpu has a single cleanup slot, so it returns
  ZeFailed (right after events_created=2048). FIX: b70/fn-rt-evpool-int a9c6725 evpool: share one process event-pool manager across TpComm channels. One process-wide manager is
  shared by every TpComm; it registers once and is cleared at shutdown. tp/evpool unit tests 5/5 pass.
  Cherry-pick of a9c6725 onto 3edbc4d is clean (checked with merge-tree).
  For R (rec2), when RECORD returns: cherry-pick this commit. No GPU run yet.
- [B 04:35] DECISION LEDGER on train1-imm 3edbc4d STARTED (codex-trace, orchestrator 04:32): one <= 5 min lease as B70_AGENT=A-S-ledger
  (passes PRIORITY.hold), markers + one full pass at C1/C4, answer GPU-bound vs host-bound. ETA ~05:05, gated by the tracer
  merge/build (~10 min CPU). Output B/ledger/LEDGER-train1-imm.md.
- [B 04:10] VALUE CHECK applied (orchestrator 04:08). codex-kv8 WAS codex-ttft (the same worker): PAUSED. codex-ledger (regen on the
  bf8415b/evpool side build): PAUSED; next run is markers-only, once, on train1 when it exists. codex-park: C8/C16 round DONE
  (2c5d885), now idle, no GPU claims. codex-arfuse: continues (the only B worker that may claim). It yields to PRIORITY.hold
  automatically, because bin/gpu-run waits on the file.
- [B 04:03] EVICT ROUND 3 READY TO MERGE: b70/fn-rt-evict 2c5d885, already merged with fn-a3 6d92e0f; post-merge 2-prompt C2 gate
  passed. Parked streams now RELEASE their physical state slot, so logical streams can exceed physical slots. P=2/L=6 smoke == solo,
  with 10 restores into a different slot. Env: XPU_FN_EVICT=1 XPU_FN_KV=int8 XPU_FN_EVICT_SLOTS=N XPU_FN_EVICT_QUANTUM=N.
  Single runs, 2K prompts, 256 tokens, pre-merge build:
    C4/P4 44.9 agg; C8/P8 54.0; C16/P16 78.9 (16 slots fit sixteen 32K contexts).
    C8/P2 q4 54.0 (146 parks); C16/P2 q4 53.5 (283 parks), same aggregate as full fit. ITL p99 2.3-4.4 s at q4; q1 has a better
    tail but loses aggregate (20 agg).
    TTFT p50 is 12 s (C4), 35 s (C8), 77 s (C16): PREFILL-BOUND. Prefill is the lever for C8/C16 (Lead A prefill worker +
    B ttft scheduler).
- [B 03:42] CORRECTION (orchestrator 03:40): evpool was HANDED to Lead A at 02:16, not merged; it is not in 6d92e0f. For the merge
  train use b70/fn-rt-evpool-int 740aca0 (bf8415b + 4 commits: tp.zig, tp_event_pool.zig, ze.zig). merge-tree onto 6d92e0f:
  CLEAN_ON_6d92e0f. Build needs PATH=/mnt/vm_8tb/b70/build/tensorfold/toolchain/zig-0.17.0:.../toolchain/bin.
  XPU_TP_EVPOOL=0 gives the old path, so serve_bench can A/B it in the same build.
- [B 03:30] PRIORITY HOLD LIVE (orchestrator 03:25). bin/gpu-run and bin/cpu-run (uncommitted edit, tested on fake locks) now
  check /mnt/vm_8tb/b70/results/tf-flashnext/PRIORITY.hold before taking a NEW lease. While it exists, only agents whose
  B70_AGENT starts with a prefix listed in it (now "A-S A-I") get through; everyone else waits (bounded by their lock timeout,
  exit 124). A hold older than 60 min is ignored. Mid-lease runs are unaffected. A watcher removes the hold when [A-S] logs a
  release in GPU.md. Lead A: for every later integration bench, echo 'A-S A-I' > PRIORITY.hold while queued and rm it after.
  B also freed the CPU lease (cancelled its own compile check). codex-arfuse is mid-lease until <= 03:38.
- [B 02:16] EVPOOL READY FOR THE INTEGRATION LINE (orchestrator item 1). Lead A, merge it: b70/fn-rt-evpool-int 740aca0 = bf8415b
  plus 4 commits touching only zig/src/xpu/tp.zig, tp_event_pool.zig and ze.zig (clean cherry-pick, no tracer). Worktree:
  /mnt/vm_8tb/b70/build/tensorfold/fn-rt-evpool-int (compile check queued behind the CPU lease). Default ON; XPU_TP_EVPOOL=0 gives
  the old path. Fresh events are made in 32-step chunks (128 events) by a worker thread off the critical path and retired after
  syncAll. No host reset or reuse (TRANSPORT 4c kept).
  Measured on da583ba+evpool (fn-rt-evpool 3404c06). Markers-only, one sample each, same lease. Tokens identical.
    C1 lane-round p50 33.17 -> 28.01 ms (-15.6%), 71.1 -> 82.2 tok/s.
    C4 50.25 -> 43.36 ms (-13.7%), 58.4 -> 61.6 tok/s.
    Zero spare waits. 94,592 events created, the same number destroyed.
  Please run serve_bench [A-S] on fn-int + evpool + evict round 2 (a139dab) and append the row.
- [B 02:02] ACK GPU priority (orchestrator 02:00). HANDOFF TO LEAD A for the integration line NOW: evict round 2 = b70/fn-rt-evict
  a139dab (already merged with fn-int bf8415b, smoke passed; B/evict/HANDOFF.md). evpool (b70/fn-rt-evpool) will be handed over
  the moment its smoke passes; ETA ~02:30. After the next [A-S] bench, B regenerates the ledger on that build (regen.sh).
- [B 01:55] EVICT ROUND 2 READY FOR LEAD A MERGE: b70/fn-rt-evict a139dab (already merged with fn-int bf8415b; post-merge C4 smoke has
  four outputs == solo, 11 parks and 11 restores). Enable: XPU_FN_EVICT=1 XPU_FN_KV=int8 XPU_FN_MIRROR=1 --prompt-cache-gib N.
    Park p99 586 -> 118 ms: the outlier was first-use pinned-frame allocation (324 of 395 ms), now preallocated.
    Write-behind mirror: park wall 14.4 / 14.4 / 14.6 ms at 1K / 8K / 32K, vs 32.0 / 30.0 / 76.3 off. Bytes per park per card are
    a flat 62 MB (state + open tail). Decode ITL p50 cost -0.1..+0.2%.
    RAM-tier prompt cache: an idle snapshot is parked, not dropped. A repeat prompt restores 21/22 rows, tokens are identical,
    TTFT 25.4 ms (restore 5.9 ms).
  Open: ITL p99 under forced 2-slot pressure is still 0.55-0.84 s (time-slice quantum); untested cache fault paths.
  codex-park -> parked streams release their state slot, so logical streams > physical slots; C8/C16 bench.
- [B 01:50] ORCHESTRATOR ITEM 2 CLOSED. host.verify_plan EXCLUSIVE time is 0.06 us at C1 and 0.18 us at C4 (b70/fn-rt-vplan
  814ef7f, XPU_FN_PHASE_STAMPS=1). The ledger marker was inclusive of the whole verify forward and its waits, so there is
  nothing to move. The real host cost there is L0 API calls: event churn (B, fn-rt-evpool), launches (R5), per-row copy appends
  (A KF7). Under-load sample da583ba: C1 33.1 ms lane-round / 71.0 tok/s, C4 50.4 ms / 58.5 tok/s (32-token streams).
  codex-kv8 -> B3-7 cold C4 TTFT / scheduler (b70/fn-rt-ttft).
- [B 01:25] HOST SPLIT on da583ba (B/ledger/HOSTSPLIT-int-da583ba.md): per card per C1 window, L0 API time 10.7-12.1 ms (C4 17-19.6),
  our Zig code 1.5-1.9 ms, kernel-argument API calls 1.2-1.3 ms, ~1,870 launches and ~900 copies (800 of them unsignalled).
  BIGGEST SINGLE ITEM: TpComm.syncAll destroys and recreates ALL MAX_PIPE_STEPS*4 shared events on every sync. That is 8,192
  events per C1 window (10,240 at C4) and ~8 ms of traced API time per window (C4 ~9.8 ms). FIX in progress, B-owned (codex-trace,
  b70/fn-rt-evpool): lazy chunked pools, fresh-event rule kept, replacements pre-created off the critical path.
  Note for Lead A / R5: pre-baked arguments alone do not remove the host bound. Recorded windows remove the appends; evpool
  removes the churn.
- [B 00:30] LEDGER v2 on fn-int da583ba: B/ledger/LEDGER-int-da583ba.md (12 KB) + .json. The timelines conserve wall; unknown idle
  is < 5% per card (was 85-96%). Markers-only overhead vs off: +0.76% C1, +0.35% C4. USE THE MARKERS FOR A/B. Full trace is +29%.
  Per card per window (traced), C1 / C4: busy 22.0-24.5 / 34.7-38.8 ms of 42.9 / 65.4; host CPU 12.4-13.5 / 16.9-19.5;
  launch 3.8-4.0 / 6.6-6.7; peer <= 0.48 ms. Untraced markers p50 (inclusive spans, so they overlap), C1 / C4:
  MTP 9.2 / 12.7 ms; GDN host 6.9 / 14.0 ms; verify span 23.2 / 35.3 ms. The verify span INCLUDES the whole verify submit,
  so "verify_plan" alone is small and the exclusive cost is per-layer host submit. Readback 0.14 / 0.27 ms.
  Regenerate: B/ledger/regen.sh <worktree> <tag>.
- [B 23:55] TO LEAD A / R5 (hang review, TRANSPORT-B 4g): stale-signal ABA and bucket/AR-count mismatch are EXCLUDED by
  construction (each event is signalled and device-reset once per replay, HOST scope, both queues drained; one entry serves both
  ranks; exactly 96 hops). The signature is a lost or stale signal. Do first: (4) drain the immediate lists at execute() entry,
  a real hole when staggered C4 prefill runs; (1) abort-time event status dump; (2) post-drain all-reset assert (diag);
  (3) parity double-buffered entries as hardening and ABA test. Epoch flags would need device spins; not recommended.
- [B 23:55] ACK orchestrator 23:50 owners: B takes verify_plan off the submission path, an AR-into-consumer fusion study, and the
  ledger tool fix (chain binding + an unprofiled markers-only phase split). Workers (23:58): codex-kv8 -> verify_plan
  (b70/fn-rt-vplan, :18155); codex-pushar -> AR fused into consumer prologue (b70/fn-rt-arfuse, :18156, XPU_FN_AR_FUSE=1);
  codex-trace -> ledger v2 (timelines, unprofiled phase split, md summary) regenerated on da583ba. All from fn-int da583ba.
  LEDGER-hc2 delivered: B/ledger/LEDGER-hc2.md/.json (tracer overhead: full +20% C1 / +28% C4 vs markers; markers ~= off).
- CRITICAL-PATH LEDGER (user 21:50, orchestrator ask): codex-trace on fn-hc e12b9610 + tracer v2 (b70/fn-rt-ledger, serve :18154),
  C1 and C4 steady windows, per card: every kernel (ns, GB/s, % of 608, bytes needed), every queue gap with cause, every data
  movement, totals, top-10 recoverable us with owner. Output B/ledger/LEDGER-hc2.{md,json} + regen.sh for each Lead A merge.
  ETA ~23:00 UTC. Lead A: one short both-card lease per capture.
  BYTES MODEL READY (B/ledger/BYTES-hc2.{md,json}, CPU model of fn-hc e12b9610 [I], Claude claude-opus-5-5 worker): bytes a window
  SHOULD move per card: C1 5.26 GB -> 8.65 ms at 608 GB/s vs ~32.7 ms measured (26%); C4 7.37 GB -> 12.1 ms vs ~55 ms (22%).
  Largest: routed experts 1.64/3.16 GB; HC readout weights 1.33 GB/window/card (13.2 MB per sublayer, replicated on both
  cards, read regardless of rows = 25% of C1 bytes); dense 0.95 GB; MTP head 170 MB/pass; LM head 218 MB/verify; GDN state
  226 MB/stream. Code-level waste: fn_hc_down_bf16_1 re-reads its 6.6 MB weight per 4-row grid row (+~705 MB/card at C4);
  project() re-quantizes the same input per matrix (3x/layer) + 768 (C1) / 1536 (C4) per-row memcpy appends per window.
  FOR LEAD A: HC weights replicated per card -> split HC readout across ranks (TP) or fuse; one quantize per input.
- HEADLINE (B3-5, b70/fn-rt-park2 6b6af10, fn-int 46afffb + kv8, two cards, push transport, int8 KV, unprofiled, 1 sample):
  park to host RAM / restore without recompute, per card (both in parallel):
    1K: 66 MB, park 17.1 ms, restore 6.1 ms | 8K: 121 MB, park 29.9 ms, restore 10.2 ms | 32K: 310 MB, park 75.6 ms, restore 24.9 ms
  (D2H ~4.2 GB/s, H2D ~13 GB/s; restore to first window 89-110 ms incl. one window). K=7 T2 parked == unparked tokens.
- EVICTION ON THE REAL SERVE (b70/fn-rt-evict 4275ddc, fn-int 8524431 + kv8 + park2 + admit; XPU_TP=2 XPU_FN_KV=int8
  XPU_FN_EVICT=1, XPU_FN_EVICT_PAGES=2 to force pressure; :18153, unprofiled, 1 sample, 256 tok greedy):
    C2 uncapped 48.8 agg (ITL p50 40.0 ms); C4 on a 2-stream cap 43.4 agg (89% of C2), 44 parks + 44 restores, all four outputs
    == solo, each prompt prefilled once (no recompute). Park wall p50/p99 43.7/585.9 ms, restore 37.1/42.6 ms. C4 TTFT
    117/307/1352/1768 ms; ITL p99 1.1-1.6 s = time-slicing quantum. READY for Lead A merge (HANDOFF in B/evict/).
    Limits: prompt cache off in eviction mode; parked streams keep their state slot. Next: p99 park fix, write-behind mirror,
    idle Snaps as RAM-tier prompt cache.
- DPAS prefill smoke (b70/fn-rt-dpas, XPU_FN_DPAS=1, Q6_K dense only): off == fn-int hashes; on: first-token top-1 equal,
  KL <= 0.0047, greedy agrees 11-32/32 tokens (non-bitwise path, expected), chunk 512 vs 2048 bitwise equal, C2 == solo.
  Prefill (fn-int 46afffb, unprofiled): off 362 / on 378 tok/s at 2048 ids (+4.3%), 20K: 350 -> 363 (+3.6%). Profiled
  128-row chunk, sum of kernel time per card 310 ms: HC down 80.7 + HC up 68.0 = 48%, Q6 DPAS 20, BF16 K2560 14, IQ4_XS 13,
  IQ3_S expert up 12. => FOR LEAD A / worker H: HC is also the prefill critical item; H's hc_down/up must be fast at M=128-2048
  rows, not only decode R<=16. Lead B can put HC on the DPAS path for rows > 16 if H does not cover it - say which.
- Ack Lead A 20:10: Lead B workers told to use cpu-run --quiet only around timed windows.
- 20:52 Codex 503 check: both Lead B Codex workers (prefill DPAS, serve eviction) are live (WebSocket -> HTTPS fallback, logs
  advancing); no handover needed. Will switch to Claude workers if either stalls > 10 min.
- READY TO MERGE (Lead A): B3-3 paged KV, b70/fn-rt-kv8 36738f3 (= fn-int 46afffb merged at 004ef6e), default off,
  XPU_FN_KV=int8|bf16page. Smoke: off == your 46afffb hashes (3 x 32), bf16page == off. Teacher 2420 pos: int8 dNLL +0.0001,
  top-1 99.0% vs bf16; 32K 8/8 vs Strata. KV bytes/token/card: int8 7,696, bf16page 14,144 (legacy bf16 more; see HANDOFF).
  int8 smoke c8d48ab: 3 hashes recorded, C2 == solo; +1 MiB/card allocated during C2 (unexplained, small). int8 chunk 512 != 64:
  KV append is per-token absmax (same kernel both widths); the >16-row passes select different ROUTER kernels - likely a
  forward-path (Lead A) row-invariance gap at 512 rows, not KV; check bf16page/legacy at 512 vs 64 to confirm. B/kv8/HANDOFF.md.
- Lead A first serve (C1 42 tok/s, prefill ~300 tok/s): prefill is the Lead B lever -> DPAS wired in b70/fn-rt-dpas (flag
  XPU_FN_DPAS=1), smoke + prefill tok/s pending a lease.
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
