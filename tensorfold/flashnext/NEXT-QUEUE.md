# NEXT-QUEUE.md -- ranked GPU tests for when the production serve comes down

Author: claude-opus-5-5 (Claude Opus 5.5), CPU-only research agent. First written 2026-11-21 17:58 UTC.
No GPU was touched and no build was run. Inputs: BRIEF, STATUS (Lead A 17:46, Lead B 07:42), JOURNAL (to 17:46),
LITERATURE, HARDWARE, TRANSPORT-B, DESIGN-A, B/ledger (LEDGER-train1-imm, BYTES-hc2), B/*/HANDOFF.md,
workers/H/r11/card1-queue.sh + PASS-GATES.txt, workers/S/int-rf2-results.json, workers/R/strata-window-protocol.md,
strata/flash-next/concurrency-0216/LEDGER.md, strata/flash-next/dd/hotschmoe-dd-0216.json, fn-a3 source (read only).
Tags: [M] measured (cited), [I] inferred / arithmetic, [L] literature claim.

## R. Re-rank vs 2cb54319 (INT, 2026-11-22 00:05 UTC; production config = tf-dd-prod.env, frozen 2cb54319)

Evidence: INT factor ablation on the frozen binary, :18150, one server per arm, 128 tokens greedy, oracle-eight prompts
(p0-p6 are 22-37 tokens, p7 is 4805 tokens, so C8 carries one long prefill), TF_LANE_TRACE time-weighted lane width
[M /mnt/vm_8tb/b70/results/tf-flashnext/workers/INT/abl/, abl-summary.json, abl_analyze.py]; final-2cb54319 window
server.log; SCHED a1 (C4 150 / C8 134 at EVICT_SLOTS=4); KV KVP1/KVP2 prefill profile. Table:
/mnt/vm_8tb/b70/results/tf-flashnext/B/ledger/LEDGER-prod-2cb54319.md.

Main finding: on the production config the C4/C8 loss is lane SCHEDULING (streams not sharing windows), not per-op
time. Production round ms by live width (1:21.3, 2:22.5, 3:29.4, 4:32.7) is within 3% of the bench config's widths,
yet prod C4 runs a time-weighted 2.31 live lanes (pc0/ctx16k: 3.55) and C8 2.89 of a 4-lane cap, with 54% round-busy
time because a 512-row prefill chunk holds the engine thread 340-450 ms with no decode round between chunks.

| Rank | Item (row) | Measured loss on production | Expected recovery | Owner | State |
|---|---|---|---|---|---|
| R1 | Eviction tier waits for idle compute lists (Runner.pleIdle) before any cache save/restore, park or restore, so admission serializes behind every decoder (new; supersedes the admission part of rows 11/12) | C4 136.6 vs 160.3 with pc0 (live 2.31 vs 3.55; 74 deferred fill calls; TTFT max 2.35 vs 0.44 s) | C4 +17% (to ~160); C8 TTFT seconds | SCHED (XPU_FN_EVICT_DRAIN, b70/dd-sched) | building/gating |
| R2 | Prefill blocks decode at XPU_CHUNK 512 (row 11 / B6, now first-order) | C8: 4.4-4.5 s of fill calls in a 9.4-10 s wall, round busy 53-59% in every arm | C8 up to ~1.8x if the 4.8K prefill interleaves (or shrinks: R5) | SCHED / Lead B ttft (B6); chunk64 arm pending | measuring |
| R3 | Lane cap: TF_PARALLEL=4 caps C8 at C4 by construction; EVICT_SLOTS=auto (3 physical at 200K) thrashes parks (slotsauto: 5 parks, restores up to 1273 ms, C8 did not finish in 170 s) | C8 = C4 ceiling | Keep EVICT_SLOTS=4 (env already); par8 arm pending | INT (config) + KV (8d1dd2d6 auto = state fit) | measuring |
| R4 | Park / cache copies polled once per engine round: 62 MB park in 1277-2093 ms, cache restore 6-129 ms, cache save 15-175 ms per prompt (new) | C4 tail TTFT; C8 parks | ~14 ms per 62 MB copy | KV (318e8b21 XPU_FN_EVICT_TIMING, then polling fix) | in progress: KV 0d6a331b submits cache copies at creation (k2a: restore 120 MiB = 4.9 ms device, 20-49 ms wall; save 13.5-28 ms device); park copies still per-round |
| R5 | Paged prefill attention: fn_page_attn_sparse 1601.7 ms vs fn_attn_sparse 699.1 ms over the same 1452 calls (row 14 relative) | PF2k 1390 vs 1556; 90K TTFT 88 s | PF to bench parity (+12%), 90K TTFT ~-25% | KV 38ccad07 (bitwise) | 38ccad07 gated bitwise (k1a), kernel only -9% (1601.7 -> 1462.3 ms, k2a): load-message bound; occupancy change dropped (no gain); 1856b3b7 vload_half4 scales: 1462.3 -> 682.3 ms (below legacy 699.1), PF 8K +14%, 32K +14%, bitwise (k5a) |
| R5b | Paged DECODE attention (traced pass, new): fn_dpage_sparse 1497 us vs fn_desc_attn_sparse 654 us per C1 window (2 cards, 30 launches) = 2.3x, C4 1612 vs 762 us; paged side kernels 414 vs 228 us; host CPU +1.4 ms and host syncs 17 vs 13 (C1) / 45 vs 31 (C4) per window | untraced lane round +3.2% (C1) / +3.8% (C4) | ~-0.5 ms per window per card (C1 +2-3%) | KV (same treatment as 38ccad07) | 38ccad07 already covers fn_dpage_sparse (not in 2cb54319); same fix in fn_dpage_sparse (1856b3b7): solo decode 95.5 -> 98.2 tok/s (k5a); traced per-window re-measure pending (INT) |
| R6 | Row 9 / B1 RESOLVED: DESC is on with paged int8 KV in 2cb54319; DESC=0 costs C1 -11% (92.2 -> 82.2), round width 4 32.7 -> 37.5 ms (+15%) | none (already on) | - | - | closed |
| R7 | B1' (was row 1 prerequisite): RECORD is refused with paged KV (xpu_forward.zig:137 RecordedAndPagedKvUnsupported); OOO is accepted; ATTN_DPAS is skipped when paged (xpu_window_runtime.zig:160) | rows 1 and 5 cannot run on production | row 1 / row 5 estimates (C4 +15%, C8 +17%; attention at long ctx) once ported | KV | after R4/R5 |
| R8 | C1 92 (this ablation) / 100 (window) vs bench 110: same 21.3 ms window, so tokens/window differs (int8 KV numerics vs drafter acceptance) | C1 -9..-16% | rows 6/8 (depth/conf, drafter numerics) | SAMP / R | KV sizing (2cb54319, ctx 16384, 4 oracle prompts x 256): paged bf16 == legacy bitwise (ids and events); int8 tokens/round -1.3% greedy (2.682 vs 2.719), -2.5% T0.6; p0 alone -5.9%. Systematic int8 cost 1-3%; bf16 MTP-layer KV (capped at that) ranked below R5b |
| R10 | NoHeldDrafts after RESTORE under forced parking (latent in 2cb54319; exposed by DRAIN / GDN defer / EVICT_SLOTS=2): Snapshot lacked Seq.device_drafts/count, lane rebinds l.seq (xpu_lane.zig:240) | all streams of the round fail | correctness | KV owner; fix = SCHED baa16cbc (KV 4c69045c, INT a52ef3f2) | s8fix 8 lanes 71 parks bitwise, s4fix PASS; every park gate now checks finish |
| R11 | Start-dependent output on the BENCH config (legacy bf16 KV, not paged), from SAMP via the orchestrator 2026-11-22: gen p2 gives one of two shas depending on the server start (df592884ff84 vs 12ba8def88a6), with shared top-K on or off. The prod config (paged int8) is stable across 8+ starts on 3 binaries | bench numbers and the legacy A/B control are not reproducible start to start; not a prod blocker | correctness (determinism) | KV, next release | CPU first: (1) get both p2 outputs from SAMP and find the first diverging token; (2) audit the legacy KV path (bf16 descriptor planes, KV_KV=off arm) for device buffers read before written (allocs with no zero-fill: planes, scratch, split-K partials, MTP layer KV) and for sizing that depends on mem_free at start (chunk, split counts, buckets, cache budget left_after_streams); (3) prepare a default-off debug flag that zero-fills every legacy KV/scratch alloc. GPU (queued): 4 starts of the bench config, gen p2 sha each, flag off vs on, plus the startup budget lines diffed between starts |
| R12 | Seeded-output drift, resolved by KRN at 12:00 (JOURNAL). Not RECORD and not cache provenance. KRN's co-save arms cs1-cs3 reproduce stack6's exact "prefill 23 / restore at=21 / prefill 1 / saved at=23" interleave and give OLD. INT stack9/10 (cache 2 and 6, bench-first) are all OLD and equal run to run. SAMP: the draw key is (seed, absolute position, id). KV's code review found no cache-save race. Two sources of the NEW shas 082752de/506dfd54: (1) 21d66fe8 (b70/dd-sched 945580ce, not on the release line) gives NEW in 4/4 starts even with PLAIN_ROUNDS=0, which violates default-off; suspect xpu_mtp_chain.submit's limits==0 early return moved after the loop (SCHED owns it). (2) A single stack6 observation (983cc068, RECORD off, after the full 30-step list), not reproduced. KV's marks/partition mechanism (xpu_lane_prompt.zig:89-96, prompt_cache marks fitted to the budget) is real in code but no arm showed it changing output | none for the release line | closes the seed-reproducibility question | SCHED (1); KV/INT (2), next window | (2): run the stack6 step list on ebdc73dd RECORD off, then bisect the steps; check marks per request if it moves |
| R9 | Rows 4 (HC_Q8), 7 (CHUNK_SOLO 1024), 13, 14-18 | per-op, after R1-R3 | as in section 1 | queue | unchanged order, below R1-R5 |

Reordered section-1 rows for the production config: R1, R2, R3, R4, R5 first (scheduling and tier copies), then
row 3 (matched prompts), row 6 (C1 depth), row 4, row 7, B1' (rows 1/5), row 10 long-context, row 13, 14-18.
Row 9 is closed (R6). Row 2 (ledger) is delivered as LEDGER-prod-2cb54319.md; its per-op part is the traced pass.

## 0. Baseline and what the targets demand

Production build int-rf2 = fn-a3 372bf1b2 (quick bench, 128 tokens, greedy, ctx 16384, XPU_CHUNK 64, CHUNK_SOLO 512,
XPU_MTP_CONF=0, paged KV/evict/prompt-cache OFF) [M workers/S/int-rf2-results.json]:

| Load | agg tok/s | event gap p50 | tokens/window/stream | target | window needed at today's tokens |
|---|---:|---:|---:|---:|---|
| C1 | 109.5 | 20.7 ms | 2.25 (128 tok / 57 events) | 100 (met) | Strata parity 133.5: 16.9 ms, or 2.76 tok at 20.7 ms |
| C2 | 139.5 | 22.1 ms | ~1.70 | 150 | 20.5 ms (-7%) |
| C4 | 168.0 | 31.8 ms | ~1.58 | 300 | 21.1 ms (-34%) |
| C8 | 209.0 | 47.6 ms | ~1.51 | 500 | 24.2 ms (-49%) |
| prefill 2k | 1560 | - | - | 1500 (met) | - |

Tokens/window/stream at C2-C8 = mean per-stream tok/s x event gap [I]; it is consistent with depth-1 acceptance
~0.55-0.6, so the 128-token bench aggregate is not dominated by ramp effects.

### 0.1 The single-stream "82% of Strata" is mostly a prompt mismatch, not window time [I from M]
- Strata's C1 133.5 is a 1500-token CODING request at ~3.12 tokens/window (concurrency-0216 LEDGER item 1:
  1500 tokens, 1443 drafts offered) -> 23.4 ms per window [I]. Strata's own matched-p0 window was 23.2 ms at 2.50
  tokens/window (R, STATUS 04:45) -> ~108 tok/s on p0.
- Ours on p0 today: 20.7 ms at 2.25 tokens/window = 109.5 tok/s. Our window is already ~11% SHORTER than Strata's.
- So on matched prompts we are at parity or ahead; the remaining single-stream gap is tokens/window only:
  acceptance parity on p0 (2.25 -> 2.50) is worth 2.50/20.7 = 121 tok/s (+10.6%); Strata's coding-prompt acceptance
  at our window would be 3.12/20.7 = 151 tok/s. Experiment Q3 measures this directly before anyone spends kernel
  time on "the C1 gap".
- Strata C1 recipe for reference: one recorded SYCL graph per card per row layout (LRU 64), EP layout (card 1 spins
  ~15.5 of ~20 ms), in-kernel flag-in-data push (ep_push/ep_wait, epoch counters), HC Q8, KREG MMVQ, --spec 4 /
  --mtp-max-t 4 (4-row solo window = depth 3), batch-MTP depth 1 at C2+, suffix/lookup drafting OFF
  (hotschmoe-dd-0216.json; R/strata-window-protocol.md). Nothing in its window structure is missing from TF except
  one-submit recorded windows, which TF has as XPU_FN_RECORD (+OOO) but does not serve by default.

### 0.2 Where C4/C8 time goes (last measured; needs Q2 to refresh)
- One-card layer slice (GDN+QSA pair, real rank-0 weights), us R1/4/8/16 [M JOURNAL 17:25]:
  recorded OOO 450/577/780/1189; immediate in-order (the SERVING default) last measured 676/697/928/1432
  (d97096cb, 10:33). Recorded OOO is -17% (R4), -16% (R8), -17% (R16) on the layers.
- Byte ideal per window per card: C1 8.65 ms, C4 12.1 ms, C8 ~16 ms [I BYTES-hc2 + B route]. C8 today is ~3x ideal.
- C8 ideal decomposition [I]: experts 4.6 GB (real union 54.6/card/layer, B route) = 7.6 ms; HC 1.33 GB = 2.2 ms;
  GDN state read+write 8 x 226 MB = 1.8 GB = 3.0 ms; dense ~1 GB = 1.6 ms; head/MTP/KV ~1.5 ms.
  => After launch/host overhead, GDN state traffic and kernel %BW at R16 are the C8 levers; expert union is already
  read-once (KF r9: default path reads weights once; GU 63-79% of BW at R8/R16).

## 1. Ranked queue (GPU, for when the serve comes down)

Gains are on the int-rf2 row and are NOT additive; "GPU min" is lease time incl. load/health. Every arm keeps the
standing gates (exact p0, C4/C8 p0 == solo, /v1/models identity, xpu-health pre/post). Rows 1-9 need no new code.

| # | Experiment | Hypothesis | Expected gain C1 / C2 / C4 / C8 / PF (arithmetic) | GPU min | Prereq build | Pass / fail |
|---|---|---|---|---|---|---|
| 1 | card1-queue 01-03: default vs XPU_FN_OOO=1 vs XPU_FN_RECORD=1+OOO, two-card serve | The one-card slice gap (imm in-order -> rec OOO -16..-17% on layers) survives two cards now that RECORD+evpool is fixed (H e0fa1982) and recording uses the fused paths (H 4376fe4d) | Layers ~70-80% of window: C1 20.7 -> ~18.2 ms = +12% (~123); C2 +12% (~156); C4 31.8 -> ~27.5 = +15% (~195); C8 47.6 -> ~40.5 = +17% (~245); PF 0 | 12 | fn-a3 tip (or 372bf1b2 to stay matched); H/r11/card1-queue.sh exists | PASS = gates + >= +5% C4 and C8 with C1 >= -1%. Prior two-card RECORD lost (C4 -10%, C8 -15%, evpool off), so this is the highest-value unknown |
| 2 | Two-card window ledger regen on the best mode of #1, C1/C4/C8, markers-only + one traced pass (Lead B regen.sh / LEDGER method) | Rank the remaining 20-48 ms: busy vs launch gap vs host vs AR peer wait vs GDN state, per card | Diagnostic; decides rows 10+ | 8 | codex-trace tracer cherry-pick on the chosen build | Conserved budget within 5% of markers wall; per-kernel us vs ideal at R4/R8/R16 |
| 3 | Matched-prompt C1/C2: Strata's 1500-token coding request + p0..p3, 512+ tokens, our build | 0.1: our window is shorter than Strata's; acceptance on code is ~3.1 tok/window | If TF reaches >= 125 tok/s on Strata's prompt, the C1 "gap" is closed by definition and C1 kernel work drops in rank | 5 | none (same server as #1 default) | Report tok/s + tokens/window on both prompt sets; Strata numbers from the concurrency-0216 ledger |
| 4 | card1-queue 06/07: main HC_Q8 (XPU_FN_HC_Q8) serve + teacher | Q8 HC halves HC bytes; Strata serves Q8 HC daily | HS slice: -0.95 ms (R1) / -2.25 ms (R16) per window: C1 +4.8% (19.75 ms), C4 ~+5%, C8 +5% (45.4 ms); PF small | 8 | merged (off) | Teacher argmax 4/4, KL <= 0.02, agreement >= 97% (PASS-GATES); speed >= +2% at C4 |
| 5 | card1-queue 08/09: XPU_ATTN_DPAS decode attention | Decode attention is chain/ALU bound 4-5x off ideal; DPAS fixes it, more so at long context | slice p2048 R1 561 -> 460 us per pair = -1.2 ms per C1 window at 2K ctx (+6%); R16 140 -> 48 us per QSA layer = -1.1 ms at C8 (+2.4%); at 16K-64K contexts larger (op p8192 149 -> 55 us) | 8 | merged (off) | Same KL gates; must also pass at an 8K+ prompt (row 9) |
| 6 | C1 depth/confidence sweep: XPU_MTP_DEPTH 3/4/5 x XPU_MTP_CONF {0, -0.4, -0.7}, on p0 and the coding prompt | With conditional acceptance c ~0.62 (p0), depth 4 adds c^4 = 0.15 tok (+6.6%) for one verify row + one draft step (~1.2-1.4 ms, +6%) = break-even; confidence stop makes it positive; on code (c ~0.8) depth 4-5 adds +12% | C1 +3..+8% (p0), up to +12% (code); C2 ~+2% | 15 | none (env only; XPU_MTP_DEPTH bypasses lane policy) | >= +3% on BOTH prompt sets, no p0 token change (greedy verify) |
| 7 | card1-queue 05: XPU_CHUNK_SOLO=1024 | Fewer expert passes per prompt | PF engine-level 512 -> 1024 was +15% at 2k (1378 -> 1580): PF 1560 -> ~1750; long-prompt TTFT -12% | 4 | none | C8 within -1% (chunk 2048 cost C8 -23% earlier), PF >= +5% |
| 8 | card1-queue 15/16: R drafter dump BF16 vs XPU_FN_DRAFT_RES_F32 + FP64 reference | Acceptance gap is numeric (bf16 drafter residual/HC/KV/embedding/handoff), find the first divergent op | Diagnostic; the fix is worth 2.25 -> 2.50 tok/window on p0 = C1 +11% (121), C2 +6%, C4/C8 +3-4% (depth 1) | 7 | merged; R/mtp/run_dump.sh | Non-empty dump; first-divergence op named; res-f32 arm accepted/drafted vs bf16 |
| 9 | Production-config A/B: XPU_FN_KV=int8 + XPU_FN_EVICT=1 + --prompt-cache-gib N + production context vs the bench config | Paged KV keeps the LEGACY (non-descriptor) window path: xpu_forward.zig:288 defaults XPU_FN_DESC off when XPU_FN_KV is set, and xpu_forward.zig:131 returns DescriptorAndPagedKvUnsupported for paged KV with DESC or RECORD [M source]. DESC was worth C4 +14% / C8 +21%, and row 1's RECORD+OOO cannot run with paged KV at all | If confirmed, production is at ~C4 147 / C8 173 and porting DESC to paged KV is the top build item (recovers +14/+21%) | 6 | none | Report both rows; if production uses the bench config, drop to low rank |
| 10 | Long-context decode sweep C1/C2 at prompt 1K / 16K / 64K / 128K (int8 KV) +- ATTN_DPAS | QSA indexer scan (L/4 pooled keys) + sparse attention + KV grow with context; agent traffic lives here | Unknown; if C1 at 64K < 90 tok/s, attention/indexer outranks HC/launch work for real users | 12 | needs XPU_CTX >= 128K build/config | tok/s and window ms per length; top-1 drift none (greedy) |
| 11 | Mixed prefill+decode (C2): stream A decoding, stream B arrives with 16K/32K prompt; XPU_CHUNK 64 / 256 / 512 | 64-row interleave chunks give poor prefill throughput and long TTFT; larger chunks trade A's ITL | TTFT of B and ITL p50/p99 of A; expected TTFT 2-3x solo at chunk 64 [I] | 10 | none (env) | Pick the chunk with B TTFT <= 1.3x solo and A ITL p99 <= 2x p50 |
| 12 | Multi-turn prefix reuse: turn 2 = turn 1 + answer + tool output, 8K-30K, prompt cache on | RAM-tier restore (25 ms, no recompute) only hits on an exact token prefix; chat-template thinking strip breaks it | Real-user TTFT 30K agent turn: ~20 s cold -> < 1 s on a hit | 8 | needs #9 config | Restored rows / prompt rows >= 0.9 on turn 2; tokens == cold |
| 13 | Long-generation steady state + fairness: C2/C4/C8 at 512-1024 tokens | 128-token quick runs under-weight steady windows; C4 min/max stream 42/58, C8 26/39 | Measurement correction only | 6 | none | Report window ms and per-stream spread |
| 14 | card1-queue 10/11: PF_ATTN+PF_ATTNS, PF_GDN | DPAS prefill attention: est -180 ms per sparse 2048 chunk per card | PF 2k 1.31 s -> ~1.13 s = ~1800 tok/s (+16%); more at 8K-32K | 6 | merged (off) | Exact p0, C4/C8 == solo, positive PF delta; teacher-KL later |
| 15 | MTP drafter arms 12-14: XPU_MTP_HC_Q8, XPU_MTP_Q8ACT | Cheaper draft steps (391 -> 371 us; experts 50.8 -> 36.3 us) | ~-0.1 ms per C1 window (+0.5%) unless acceptance moves | 6 | merged (off) | accepted/drafted not lower by > 1 point |
| 16 | Depth 2 at C4/C8 re-test (XPU_MTP_DEPTH_C4=2, _C8=2) on the best build after #1/#4 | Marginal verify row cost falls with R (E(R) marginal 4.4 -> 3.7 experts); lost before because multi-row kernels were slow | Lane sweep at 3edbc4d: d2 -7% (C4) / -14% (C8). Win only if windows got cheaper per row | 6 | none | >= +3% agg |
| 17 | HS1-3 XPU_HC_PREFETCH_AR off/C2/C5 | L2-warm HC weights during the AR peer wait | <= 1-2% | 8 | merged (off) | PASS-GATES HS rule (>= +1% C1 and C4) |
| 18 | KEEP_BATCH=0 control (card1-queue 04) | control only | 0 | 3 | none | diagnostic |

### 1.1 Build items to prepare on CPU now (so they are ready for the next down window), ranked

| # | Item | Why | Expected | Owner hint |
|---|---|---|---|---|
| B1 | Descriptor (and recorded) windows for paged int8 KV: remove DescriptorAndPagedKvUnsupported (xpu_forward.zig:131) | Long agent contexts need int8 paged KV (legacy bf16 KV >= 14 KB/token/card: 4 x 200K does not fit 8.5 GB free); evict and the RAM prompt cache require XPU_FN_KV=int8; so every bench win on the DESC/RECORD path is invisible to a long-context production config | C4 +14%, C8 +21% on the production config [M int-prep-desc delta], plus whatever row 1 wins | H / Lead B |
| B2 | Drafter acceptance fix from row 8 (likely fp32 residual/HC/KV or the hidden handoff) | Only lever for single stream that is not kernel time | C1 +11% on p0 | R |
| B3 | GDN replay records (ReplaySSM 2610.13778; verify never writes S; commit folded into next verify) | GDN state is ~3.0 ms of the ~16 ms C8 ideal (226 MB/stream/window read+write twice) | C8 -1.5 ms (+3-4%), C4 -0.75 ms (+2-3%) | HS |
| B4 | Data-parallel MTP drafter per card (LITERATURE #14) | No cross-card hops in draft chains at C >= 2; half the rows per card | C4/C8 MTP chain 0.65-1.33 ms -> about half; +2-3% | H |
| B5 | Prefill AR f16 payload + 2-microbatch wavefront (PF levers 1-2) | Prefill AR wait is wire time (~180 of 202-222 ms per 2048 chunk) | PF +8% (f16) .. +16% (wavefront); agent TTFT | PF (needs orchestrator OK: AR compression was ruled out of scope) |
| B6 | Mixed-batch scheduler: decode rows + one 256-512-row prefill chunk per window, shortest-remaining with aging (B ttft handoff) | Row 11 | Agent TTFT under C2 | Lead B ttft |
| B7 | Multi-turn checkpoint at message boundaries (park state at end of each user/assistant segment) | Row 12 if exact-prefix hits fail | Agent TTFT | Lead B |
| B8 | Cost-aware draft depth per stream (EcoSpec 2607.12696, Cascade 2506.20675, drafts_cost.zig already in TF for Metal/CUDA) | Stop drafting rows whose expert-union cost exceeds expected tokens | C4/C8 +5-10% when depth > 1 pays | Lead B lane |

### 1.2 Literature additions since LITERATURE.md (searched 2026-11-21 17:55, abstracts/snippets only) [L]
- FastMTP (2509.18362) and the Red Hat FastMTP-heads note: re-training a shipped MTP head on its own recursive
  outputs lifts deep-position acceptance (pos 2 0.476 -> 0.616 on Qwen3-Next-80B-A3B). Our drafter is used
  recursively at depth 3; a self-distilled drafter is an offline-training option if row 8 finds no numeric bug.
- MTP-D self-distillation (2603.23911): +3-4% acceptance, +7.5% cumulative at head 4.
- EcoSpec cost-aware drafting (2607.12696, up to 1.62x) and SpecFit (2610.15417): penalize drafts that grow the
  expert union; matches B8. AcceptMoE (2608.02989): lossy verifier expert sets, user decision only.
- SpecLA (2607.16673), TreeWY (2608.20961), HybridSpec (2611.04055): GDN rollback without full state copies; B3.

## 2. First 2 hours when the serve comes down

Pre-flight (CPU, before the stop): pin the binary (card1-queue "pin"), build the fn-a3 tip, write the Strata
1500-token coding prompt and an 8K/16K/32K agent-like prompt (from soak traffic if available) into workers/S/prompts.
Announce in GPU.md: cards 0+1, ~120 min, PRIORITY.hold "A-S A-I".

| Clock | Step | Rows | GPU min |
|---|---|---|---|
| 0:00-0:03 | bin/xpu-health both cards | - | 3 |
| 0:03-0:20 | CARD1_RELEASED=1 bash workers/H/r11/card1-queue.sh --run nq1 steps 01-05 (default, OOO, RECORD+OOO, KEEP_BATCH=0, CHUNK_SOLO=1024) | 1, 7, 18 | 15 |
| 0:20-0:25 | On the step-01 server: matched-prompt C1/C2 (Strata coding prompt + p0..p3, 512 tokens) + long-gen C4/C8 512 tokens | 3, 13 | 5 |
| 0:25-0:35 | Ledger regen, markers-only + one traced pass, C1/C4/C8, on the winning mode of 01-03 | 2 | 8 |
| 0:35-0:55 | card1-queue 06-09 (HC_Q8 serve + teacher, ATTN_DPAS serve + teacher) | 4, 5 | 16 |
| 0:55-1:10 | Depth/confidence sweep, 4 arms (d3/c0 control, d4/c-0.4, d5/c-0.4, d4/c-0.7), p0 + coding prompt | 6 | 15 |
| 1:10-1:20 | Production-config A/B (int8 KV + evict + prompt cache, production ctx) vs bench config | 9 | 6 |
| 1:20-1:35 | Long-context decode 16K/64K (128K if time) +- ATTN_DPAS; mixed prefill+decode chunk 64/256 | 10, 11 | 15 |
| 1:35-1:45 | card1-queue 15/16 R dump | 8 | 7 |
| 1:45-1:55 | STACKED ARM: every row that passed (mode from 1, HC_Q8, ATTN_DPAS, CHUNK_SOLO, depth/conf) in one quick bench + teacher | - | 8 |
| 1:55-2:00 | xpu-health, release, PRIORITY.hold removed, JOURNAL rows | - | 2 |

Stop rules: one DEVICE_LOST/hang -> bin/xe-reset + xpu-health, skip that arm; two consecutive failures stop the
session. No default flips during the window; the lead decides after the stacked arm.

Expected outcome if rows 1, 4, 5, 7 pass as estimated [I, multiplicative, 30% haircut on the sum of gains]:
C1 ~130-135 (with row 6 ~135-140), C2 ~160, C4 ~200-210, C8 ~250-265, PF ~1750. C4 300 / C8 500 still need the
structural items (B1-B4, B8) and per-kernel %BW at R8-R16; the row-2 ledger decides their order.

## 3. Soak observations

(appended below as /mnt/vm_8tb/b70/results/hotschmoe-dd/soak/ data arrives; at 17:58 UTC the directory does not
exist yet: hotschmoe-dd.service still names the Strata launcher and is inactive; a TF build is running under
/mnt/vm_8tb/b70/results/hotschmoe-dd/tf-build/.)

Row 10 measured (KV, 2026-11-22 06:40, b70/dd-kv 1856b3b7, prod KV config, C1 greedy, free-form 512-token reply,
distinct seeds, no cache hits): decode 96.4 / 91.3 / 90.0 / 84.9 tok/s at 1K / 16K / 64K / 128K prompts (-5% / -7% /
-12%); TTFT 0.76 / 11.6 / 51.0 / 110.5 s (prefill 1294 / 1333 / 1248 / 1160 tok/s). Above the 90 tok/s bar through 64K.
This sizes ATTN_DPAS over pages at about +5-12% C1 at long context (plus row 5's ~+6% at 2K): worth one kernel, gated by
teacher/KL. Copy-style replies (KV_LONG_ANSWER=1) inflate decode to 210-330 tok/s via MTP acceptance; do not use them for
speed. [M workers/KV/r10c, r10d]

SAMP verify row buckets (2026-11-22 07:05, next release): desc verify windows round up to power-of-2 row buckets
(xpu_window_abi.bucket: 1, 2, 4, 8, 16), so a depth-4 window (5 rows) pays the 8-row bucket. Approximate prod C1 verify
ms by bucket (2cb54319 prod config): bucket 2 ~15.3-15.9 (depth-1 arms, rows 2.0-2.3), bucket 4 ~16.7-17.6 (d3c0, rows
~4.1), bucket 8 ~20.4 (dconf d4, rows 4.75). The device confidence stop (c36f1e3d/535d9fe0, XPU_FN_MTP_DEVICE_CONF=1,
bitwise) at depth 4 gave +5.5% tok/round over d3c0 (greedy 3.010 vs 2.853, code 3.362 vs 3.024), but the 8-row bucket
ate it (window 21.5 -> 23.8 ms, tok/s -5.5%). ITEM: verify buckets 5/6 (or exact-row windows) -> unlocks depth 4+
drafting; re-run dconf (depth 4, conf -0.4) on top. Depth 3 + conf stop alone loses (tok/round -3.7%, window -0.35 ms,
tok/s -2.4%). [M workers/SAMP/perf/q-dconf, q-dconf3, armsA/d3c0]


## Next release items from KRN (2026-11-22 12:05)

### NR-1: b70/dd-sched 21d66fe8 changes sampled tokens with XPU_FN_PLAIN_ROUNDS=0 (default-off violation)

Repro: workers/INT/stack7 (ab_body.sh: fresh server, prod env, cache 6, FN_BENCH_QUICK prod bench, then prof_client2
--sets p --conc 1,2,4,8 --cfgs gen,t0.6 --c1-prompts 4 --tokens 256).
- On 21d66fe8 the new shas appear in 4 of 4 starts: t0.6 p1/p2 082752de/506dfd54, gen p1/p2 600e1ca0/d21c02a0. This holds
  with PLAIN_ROUNDS 0 and with 1.
- On ebdc73dd (= 983cc068 + B1') the same body gives the old shas at C1 and C8 (workers/KRN/or2, krn_arm_order.sh
  KRN_BENCH=1 ... gen,t0.6): t0.6 5c0f1986/d04ea442, gen 96e61817/f21fcfcd, which equal c6346c82.
- The env differs only in XPU_FN_PLAIN_ROUNDS=0 and the binary (diffed).
- Greedy p0 stays bitwise, which is why the KV/SCHED greedy gates passed.

Candidate, a weak one: xpu_mtp_chain.submit moved the per-stream `limits[i] == 0 -> return false` check from inside
the validation loop to after it. With no validation error this is equivalent; it differs only when a later request in
the batch would error (WindowTooWide, InvalidDraftPosition, checkIds) where the old code returned false (host chain).
The other dd-sched hunks (windows.plainCopies, shared_round.runPriced, lane cfg.plain_rounds) are all guarded by
cfg.plain_rounds, which is false at PLAIN_ROUNDS=0. So the next step is not a code read but a bisect:
0. FIRST CUT (SAMP owns it, a0376fc4c87a3b004): ReleaseSafe builds of 983cc068 and 21d66fe8 (undefined memory filled with
   0xAA). Run the stack7 body or p1 T0.6 seed 1001 on each. If the two ReleaseSafe builds agree with each other but differ
   from their ReleaseFast builds, the cause is an uninitialized read whose value moves with the lanes.Stream / Engine /
   Config layout. That read is latent on the release line too, so find it and initialize it. Run steps 1-2 only if this
   cut is clean.
1. Revert the submit hunk alone, then rerun the stack7 body for 1 lease.
2. If the shas still move, diff the build inputs of the 21d66fe8 and ebdc73dd trees (kernels and SPIR-V, toolchain,
   build flags). Also check that the startup "window by rows" calibration (it feeds the cost rule's depths) is the only
   run-to-run input.
Gate for dd-sched re-merge: the stack7 body sampled shas must equal ebdc73dd's at PLAIN_ROUNDS=0.

NR-1 SAMP CPU audit, 2026-11-22 12:05 UTC (a0376fc4c87a3b004):
- Matched-trace diff, done lines from server.log (prompt, cached, tokens, sha), stack7 ab-r2-pr0 (21d66fe8, NEW) vs KRN
  or2 (ebdc73dd, OLD). The bench traffic before the divergence is identical: the same greedy shas, the same cache
  events (p1 saved at=23 and p2 saved at=36 in the same interleave), the same resume points. The first divergent
  request is the first SAMPLED one, gen C1 p1: "restore id=0 at=23, prefill start=23 rows=1", then 256 tokens. One
  stream, no concurrency; e4c58b17 vs 12f89b02.
- Draws are pure (seed, absolute position, id), so either the restored p1 state at 23 or the target logits of that solo
  decode differ. Greedy from the same state at 23 stays equal (bench C1 p1: 882e89e6 in both), so any state difference
  is small.
- 983cc068 produced BOTH outcomes: stack6 NEW, INT a4 (bench first, cache 6) OLD. The outcome is therefore not fixed by
  the binary. 21d66fe8's 4/4 is one body on one binary; a timing bias is possible.
- Code read, flag off: the dd-sched hunks are token-equivalent (all gated by cfg.plain_rounds; submit returns the same
  false). No '= undefined' read past its filled length found in the paths I checked: xpu_mtp_chain.submit
  limits/steps, xpu_lane chooseFromRuns runs/cands, chainGathered local/global/cands, drawRows g/v32, xpu_sample
  Runs/Bufs (host_ov/oi/g read only for row < rows, j < k).
- Step 0 (ReleaseSafe 983cc068 + 21d66fe8 host suites + xpu-native) is building: workers/SAMP/rs_build.sh, output
  workers/SAMP/rs/.
- Proposed next-window GPU gates, one lease each (stack7 ab_body unless noted):
  G0 ReleaseSafe 21d66fe8 and ReleaseSafe 983cc068 (bins in workers/SAMP/rs/<c>-out). Pass: the shas equal each other.
  G1 21d66fe8 ReleaseFast, short body (prof_client2 only, no bench). OLD means the state at 23/36 saved during the bench
     is the variable, not the decode.
  G2 21d66fe8 ab_body with XPU_FN_EVICT_DRAIN=0, then with XPU_FN_GDN_DEFER=0. A move back to OLD points at the
     drain/defer path (cache save or restore overlapping compute).
SAMP's alternative, which can come before step 2: an uninitialized read exposed by the new lanes.Stream / Engine / Config
fields (layout shift). Build 983cc068 and 21d66fe8 in ReleaseSafe and run p1 at T0.6 with seed 1001 on each. If they
agree with each other but one differs from its ReleaseFast build, the read is latent on the release line too. See also
row R12 (KV).

Still open, not reproduced: workers/INT/stack6 (983cc068, RECORD off, full 30-step list) gave the new t0.6 p1/p2 shas.
workers/INT/stack8 (same list, RECORD=1) gave the old ones. Every fresh-server cut on 983cc068/ebdc73dd gives the old
shas: INT stack9/stack10 a1-a4, and KRN cs1-cs3, which include stack6's exact prefill/restore/save interleave, plus or1
and or2. Run the stack6 step list on ebdc73dd with RECORD off; if it moves, bisect the steps.

### NR-2 (next C1 lever): record the MTP drafter / device chain under RECORD (XPU_FN_OOO)

Under XPU_FN_RECORD=1 the device MTP chain is off: xpu_mtp_chain.available needs recordings.ooo (XPU_FN_OOO, default
false). Drafts therefore run the host chain (greedyImmediate on draft logits, uncoupled), and draftProbabilitiesMeasured
returns cache.gate, which changes the depth rule.
- This is bitwise (SAMP cpl0c/cpl1c: coupled == uncoupled == host chain) but costs C1. Prod C1 is 106.6 off and
  105.7 on, while C2-C8 gain 9-19%.
- Next: either record the draft steps (MTP forward + draft head) as their own bucketed lists beside the verify
  recordings, or qualify XPU_FN_OOO=1 so the immediate device chain runs between replays.
- Gates: the record gate (p0 bitwise), the sampled shas of prof_client2 t0.6/gen equal to RECORD off, conc == solo,
  and a forced park at 8 lanes.
- Size: compare C1 recorded window plus draft time against immediate with the decode-window profile
  (KRN_PROF_ALL, 8769fc0a).


## SCHED queue (added 2026-11-22 15:04:17 UTC; CPU-only since release 15:01, no GPU touched)

Evidence: JOURNAL [SCHED] 01:42, 04:12, 08:00, 08:55; workers/SCHED/. Port 18153; one arm per lease of 5 min or less.
1. Plain rounds on the release build (if INT's 8-lane arm did not run): XPU_FN_PLAIN_ROUNDS=1 vs off on the tf-dd-20261122
   binary, prod env, --parallel 4 and 8. Run b8.py C1-C8 at T0.6, card and greedy (steps_pr8.sh / steps_pr4.sh), plus prids
   (ids == drafts-off at C1/C8). Accept: C4/C8 >= off, C1/C2 within noise, ids equal, HEALTHY.
2. Plain-round cost breakdown (design input for a cheaper plain round): one profile at W=8 sampled with the gate forced
   plain (probe-only build or a gate margin of 0), against draft:false at W=8. Split the 40 ms vs ~32 ms gap into MTP absorb
   (layer + TP reduce), drafting-lane verify bookkeeping and the syncAll in absorb. Decides between dropping that syncAll
   (verify already host-syncs) and a gap-tolerant MTP cache (zero or mask the skipped rows) to reach the drafts-off ceiling
   (C8 254-283).
3. Prefill order for burst arrivals (INT long8, 7 x 60K at 8 lanes): default vs TENSORFOLD_PREFILL_ORDER=srpt vs
   TENSORFOLD_PREFILL_POLICY=fcfs. Gate: mean/max TTFT and the live decoder's ITL p50/p99; tokens == solo (order is
   bitwise). The knobs exist; no build is needed. Expected mean TTFT ~310 -> ~228 s at the same max (KV arithmetic).
4. Chunk 256 only beside a live decoder (XPU_CHUNK=256, XPU_CHUNK_SOLO=512): the row-11 stall fix (p99 1280 -> 270 ms)
   without the solo prefill loss. It needs its own conc == solo proof (the 256/512 chunk divergence is unexplained) and the
   teacher gate at 256 (90K first-token KL .026 > .02 failed on 2026-11-22). Low priority until a 256-row prefill is faster.

### NR-3 (INT): device memory growth under mixed long traffic on the RC

Release traffic rel-rc-20261122T1311Z (ebdc73dd + RECORD=1) fitted +1.21 GiB device memory over 60 min (cap 1.00, WARN):
VRAM per card 30.07 -> 30.79 GiB. The 4 "FN_RECORD scratch regrow" lines account for about 120 MiB.
- First, CPU only: read the traffic server log (workers/INT/rel/traffic-server.log) and the /metrics samples in
  workers/REL/rel-rc-20261122T1311Z. Find whether the growth is steps (allocation events) or a ramp, and which allocator
  logs line up.
- Then, GPU, next window: the same release_traffic run for 20 min with RECORD off and with RECORD=1 on one lease, to
  tell recording/scratch from KV/cache growth.
- Gates: growth below 1 GiB in 60 min, or a plateau that is bounded and explained.


### NR-4 (lever 2, KRN design, CPU only 2026-11-22 15:10): AR peer copy on the RECORDED path (prod is RECORD=1 now)

Prod now runs recorded windows (ebdc73dd + XPU_FN_RECORD=1), so the AR that matters is TpRecording.recordAllReduceF32
(zig/src/xpu/tp.zig:726, called from xpu_window_forward.zig:87). Per AR and per rank, the recording is:
1. barrier;
2. appendRecordedPush, which is a compute-list zeCommandListAppendMemoryCopy of send_buf to the peer inbox, then a
   barrier, then a signal;
3. a consume barrier that waits on the peer's event;
4. ar2_f32;
5. barrier and signal done.
At C1 that is 96 ARs of 40 KB. The copy is about 8.5 us (4.8 GB/s, latency-bound), and ar2_f32 has a median of about
12 us, which is mostly wait time (JOURNAL 2026-11-22 [KRN] decode-window profile).

Prior work, all on the IMMEDIATE path only and unmerged, on branch b70/fn-rt-arfuse (Lead B codex-arfuse):
- Consumer fusion (AR into the HC writeback_norm consumer): bitwise, 204 -> 12 AR launches per round. C1 was flat in a
  single pinned A/B.
- a231ef87 / 4c9d372d: opt-in peer stores in the TP2 MoE combine epilogue. Untested.
- Microbench (B/arfuse/peerstore, 40 KiB, producer start -> peer consumer end, p50/p99): copy push 16.0/19.3 us,
  producer local+peer stores 13.0/13.8 us. That is -3 us at p50 and -5.5 us at p99 per hop.

Design (default off, XPU_FN_AR_PEER=1, recorded and descriptor paths):
- The AR producers (attention/GDN output-projection f32 store and fn_moe_combine) gain a peer pointer argument. Each
  writes its rows both to local send_buf[slot] and to the peer's inbox[slot].
- Then append barrier + signal(event(pos,0,rank)), replacing appendRecordedPush. The ring slots and inboxes are fixed
  per position, so they are fixed buffers and the recording signature does not change.
- Sum order is unchanged: rank 0 computes send + inbox and rank 1 computes inbox + send, as today, so it should be
  bitwise by construction.
- Saving: one copy command and one barrier per AR per rank, about 3-5 us per hop, or about 0.3-0.5 ms per C1 window
  (1.5-2%). Most of it comes off the p99 tail, which is the prod +767 us item.
- Step 2: fuse ar2_f32 into the consumer (port the arfuse consumer fusion to the window path). It removes 96 launches and
  96 barriers per rank per window. That is device-side only under RECORD, because host append is already gone.

GPU tests for the next owner-approved window (each one lease of 5 min or less on port 18155):
- T0: one traced recorded C1 window with diagnostic_kernel_events. Get the per-AR split (copy, wait, ar2) under RECORD to
  size this; it is profile only.
- T1: XPU_FN_AR_PEER=1 through the record gate (p0 bitwise), conc == solo at C4/C8, SAMP prof_client2 t0.6/gen shas equal
  to off, and a forced park at 8 lanes.
- T2: prod bench A/B, off/on, C1/C2/C4/C8/PF. Keep it if C1 is at least +1.5% or the C8 p99 ITL drops, with no row loss.
CPU work now: port a231ef87 to the window kernels (w.k[rk]) behind the flag, plus host tests for slot/peer pointer
selection. The branch will be b70/krn-arpeer off ebdc73dd. No merge before T1/T2.
