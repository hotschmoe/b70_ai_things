Responding model: claude-opus-5-5 (Claude Opus 5.5), 2026-11-20. CPU-only static analysis; no GPU, build,
source or config change. Only this file was written.

# Strata 0.2.16 DD: request walk, cold-C4 serialization, restore TTFT, EP card split

## 0. Sources, citation key

- Source: /mnt/vm_8tb/b70/build/strata-0216/src, read as a git archive of commit cf1a2c359 (dd-0.2.16 tip at read
  time; the worktree had an unresolved Codex merge in iq_kernels.dp.cpp, so it was not read directly).
  Paths below are repo-relative:
  - gen = engine/sycl/src/program/generate.cpp (the SYCL main; engine/sycl/CMakeLists.txt:40, 338)
  - ver = engine/sycl/src/core/verify.cpp
  - pf = engine/sycl/src/prefill/prefill.cpp
  - lay = engine/sycl/src/core/layer.cpp
  - snap = engine/sycl/src/core/conversation_snapshot.cpp
  - mtp = engine/sycl/src/core/mtp.cpp
  - srv = terminal/serve/server.py; base = terminal/serve/server_parts/base.py; si = engine/sycl/serve/server_intel.py
  - cuda-gen = engine/src/program/generate.cpp (CUDA/HIP main; used only to show features SYCL lacks)
  - old-ver = /mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/core/verify.cpp (old night-final-v3 tree)
- Config: strata/flash-next/dd/hotschmoe-dd-0216.json plus launcher env from strata/flash-next/dd/nl_launch_0216.sh:32-34.
- Logs (read-only): /mnt/vm_8tb/b70/results/strata-0216/candidate-engine.log (= cand-log),
  old-night-engine.log (= old-log), candidate-context/engine-segment.log (= ctx-log),
  candidate-context/summary-reconciled.json, old-night.json.
- Tags: [R] = read in source or log at the cited line. [I] = inferred from what was read (arithmetic or reasoning).

## 0.1 Headline findings

1. Most of the C2/C4 decode regression versus old night is probably one porting error. [R+I]
   - Old night ran STRATA_NIGHT_ATTN_BATCH=1 (old-night.json env; old-log:8). It batched QSA select+attend per slot
     and ran one top-k and one attention launch per window, with streamed KV (old-ver:1372-1420; old-log:18 shows
     kv-resident 65536).
   - Its 0.2.16 replacement is STRATA_BATCH_ATTN_ROWS=1. That path is disabled whenever any row's KV is streamed
     (ver:1541 comment "Not with KV streaming"; ver:1546-1547 rows_ok requires kv_mode != 1). Under
     --kv-resident 65536 every slot session is kv_mode 1 (lay:722-740), so the DD falls back to the per-row path
     (ver:1567-1579): 4 launches per row per QSA layer.
   - update-0216/feature_map.md:73 says "streamed KV uses per-row path, as did the old batched path". That is
     contradicted by old-ver:1372-1420, which has no kv_mode gate and resolves KV per slot. [R]
   - Same row counts, old vs new, per window: [R]
     - C2, 3.70 rows: 23.42 ms (run 20.73) at old-log:881 vs 26.70 ms (run 23.93) at cand-log:882.
     - C4, 6.8 rows: 38.30 ms (run 32.41) at old-log:2002 vs 44.14 ms (run 38.38) at cand-log:1919.
     - commit and emit are unchanged (C4 2.80/3.09 old vs 2.76/3.01 new). The whole delta is in "run", and it grows
       with rows: +3.2 ms at C2, +6.0 ms at C4.
   - At C4 the per-row path is 8 rows x 12 QSA layers x 4 = 384 launches per window. Old mode 1 was about
     12 x (4 scores + 1 top-k + 4 resolve + 1 attention) = 120. [I]
2. Cold C4 serializes by design at three layers. [R]
   - Python admits one request at a time and holds the control lock through the admission's DONE/BADM.
   - The engine reads every prompt through the one main session on its single command thread.
   - A 12288-token chunk runs with every slot stopped.
   - Shared-expert-read coalescing (BCO), in-slot reads (STRATA_SLOT_READ) and a concurrent read thread
     (STRATA_PIPED_CONCURRENT) exist only in the CUDA engine.
   - For four equal ~20K prompts the minimum mean TTFT is bounded by prefill throughput (~17 s per prompt), not by
     scheduling.
3. Forced restore TTFT (median 2.08 s) breaks down as follows. [R+I]
   - About 0.53 s is outside the engine (Python).
   - About 0.44 s is the park of the outgoing main session, and 0.21 s is the native restore.
   - About 0.2 s is tail windows over a cold, fully non-resident KV map.
   - The second request of each pair also waits behind the first request's promotion (a 0.42-1.5 s park plus a
     0.16 s copy), held under the control lock.

## 1. The request walk (HTTP -> first token -> per-window loop)

### 1a. HTTP and Python

1. si wraps srv (xe telemetry, switcher, logprobs, narrator, two-card spec rule). Its _pump routes engine lines
   starting "BT " / "BDONE " into slot_q[slot] (si:173-178). [R]
2. Service path:
   - In parallel mode the request skips _inference_fifo (srv:4009; par branch srv:5359-5409). [R]
   - The prompt is chat-templated and tokenized incrementally against self.prompts (srv:4893-4907). [R]
   - The request runs in _run (srv:5232+). first_engine_token_s is stamped at srv:5461. [R]
3. StrataEngine (srv:694):
   - It has slot_q, slot_busy and slot_cv, and ctl = threading.Lock with the comment "one admission or solo request
     on the control lines at a time" (srv:982). [R]
4. generate_batched (srv:1870-2300):
   - Alone: a solo GEN on the main session (srv:1931-1947). This path uses the full --spec 4 / --mtp-max-t 4
     drafter. [R]
   - When another request waits, others() stops the solo stream (srv:1949-1951). Its state is promoted with
     prompt = ids + out (srv:1998). [R]
   - pick_slot (srv:2357-2388) uses _slot_prefix (srv:2307-2355). That is a binary search that slices Python lists of
     ids at each probe, O(n log n) element copies on 132K-token prompts; SLOT_PREFIX_MIN=512. [R; cost I]
     The pick loop releases ctl while it waits for a slot (srv:2006-2050). [R]
   - BGEN is sent (srv:2070-2082). ctl is released only after the admission's DONE/BADM line (srv:2152). [R]
   - The BT loop (srv:2162-2222) streams tokens from slot_q. [R]
   - Options: _coalesce/BCO (srv:1760-1802) is off by default (STRATA_COALESCE_PREFILL, base:275-291).
     ADMIT_SHORT_FIRST (srv:432-439, 1804-1814) and _may_go_solo (srv:1819-1829) exist. _shorter_waiting
     (srv:1865-1868) drives BYIELD, with YIELDS_MAX=2 (srv:1677). [R]
   - The DD config sets neither COALESCE_PREFILL nor ADMIT_SHORT_FIRST (hotschmoe-dd-0216.json env). [R]

### 1b. Engine command thread (SYCL)

1. Startup:
   - EP forces no_prefill_borrow, a per-layer expert cache, adapt_every=0 and no pipeline windows (gen:2436-2463).
     So --no-prefill-borrow in the config is redundant under EP. [R]
   - Slot sessions are made by session_init(max_context=200000) (gen:4406-4470). With --kv-resident 65536 < 200000
     cells, each QSA state is kv_mode 1 (streamed, authoritative host pool; lay:722-740, minimum 20480 at lay:755).
     [R]
   - cand-log:17: "65536 of 200000 cells per QSA layer in VRAM, the K/V in 2.36 GiB of pinned RAM". [R]
2. Main loop (gen:10417-10459):
   - batch_step runs only when no input line is pending. [R]
   - BGEN is rewritten to "GEN 1" and read through the main session ss (gen:10445-10458). [R]
3. GEN (gen:10703+):
   - Resume source selection: live session and checkpoints (gen:10974-10983), then a slot holding the prefix
     (slot_source, gen:10984-11005), then the conversation cache (gen:11006-11008). [R]
   - park_current(): the outgoing main-session conversation is parked before a restore (gen:11035; function at
     gen:8489-8506; ParkSource at gen:8325-8488; NIGHT_PARK_MOVE at gen:8338-8347). [R]
   - copy_from_slot (gen:11039-11088), or conversation_snapshot_restore (gen:11094-11155), or session_zero
     (gen:11179-11187); then mtp.kv_restore (gen:11223) and the RESUME line (gen:11255). [R]
   - Short reads (64 tokens or fewer, short_read at gen:747) run as windows, not a prompt pass (gen:11258-11268). [R]
   - read_part (gen:11576-11674) runs the chunked prompt loop.
     - Chunk = --prefill 12288 (gen:1979). [R]
     - At each chunk boundary the slots decode for STRATA_BATCH_DECODE_SHARE x the chunk's time (default 0.5,
       gen:11578-11582). [R]
     - BSTOP/BYIELD are scanned only at chunk boundaries. [R]
   - prompt_ms is measured from r0 (gen:10939) to gen:11818, so it includes park + restore + read. [R]
4. Prefill pass (pf):
   - STRATA_PREFILL_FIRST (pf:2930-2943) controls a short first chunk that overlaps the PLE IO. Its default is 256
     only when every expert is cached; the DD's 0 turns it off. For 20K prompts the effect is at most one chunk's
     PLE wait. [R; size I]
   - PLE is gathered by std::async double buffering (pf:2944-2958, 3106-3175). The chunk loop is at pf:2972. [R]
   - HALVES (pf:436-445, 3548-3556) is on. [R]
   - Host expert grouping happens per layer: a D2H copy of T*K*4 B of ids, then a host wait (hv_routed[sub].wait() /
     routed.wait() / m.cs->wait()), at pf:4259-4340. The peer's rows go last (pf:4341+). [R]
   - At T=12288, K=10: 480 KiB D2H x 48 layers = 23 MB per chunk, with 48+ host blocks per chunk. [I]
5. Admission tail (BADM, gen:13252-13315):
   - park_slot (gen:9975-10023, with a queues_wait at gen:9990). [R]
   - copy_to_slot (gen:9799-9868, several queues_wait_and_throw). With SLOT_COPY_DIRECT=1 the copy is direct
     (gen:9760-9765, 9773-9784, one queues_wait per stage). [R]
   - Under kv_mode 1 both pools are host memory. snap uses CPU memcpy for host-only copies because the BCS copy
     engine hung (snap:114-124). [R]
   - draft_first (mtp:2157, a .wait() at mtp:2166). DONE is printed at gen:13243. [R]
6. Per window, batch_step (gen:10026-10285):
   - Pack rows, 2 per slot with batch MTP depth 1 (gen:10035-10058). Set the slot histories (gen:10100-10105).
   - ver.run_slot_rows (gen:10107; ver:4047-4205):
     - stage_batch (ver:3911-4033): host PLE gather into h_ple_ (ver:3993-4005). A new row layout triggers a capture
       with LRU eviction (ver:3949-3981).
     - Submit the ep_q_ graph and then the cs_ graph. Wait on cs_->wait() (ver:4089) and copy_->wait(). Then ep_check.
     - sample_rows (ver:4294-4333): for non-greedy rows, one kernel per row, a memcpy().wait() per penalized row and
       a final wait. Results are read from mapped h_out_. [R]
   - Keep calc (gen:10114-10121). commit_slot_prefixes (gen:10138; ver:4218-4287): a graph launch, then
     cs_->wait() at ver:4258. [R]
   - Print BT per token (gen:10159). [R]
   - Per active slot: memcpy(final_R rows).wait() (gen:10184-10187). At G=2 rows that is 2*4*2560*4 = 81,920 B
     D2D per slot. [R; bytes I from the expression]
   - MtpPair::draft (gen:10208-10256; mtp:2403, cs_->wait() at mtp:2493 and mtp:2514). [R]
   - fflush (gen:10257). The timing summary line is built at gen:10260-10283. [R]

## 2. Why cold concurrent prefill serializes (TTFT 17/39/61/84 s on four ~20K prompts)

### 2a. Mechanism

1. Python. [R]
   - Only one request holds ctl for its whole admission (srv:982). It is released at srv:2152, after DONE/BADM.
   - Requests 2-4 queue behind the lock, or in the pick loop.
2. Engine. [R]
   - One command thread (gen:10417-10459).
   - Every prompt is read through the single main session ss ("GEN 1" rewrite, gen:10445-10458).
   - No SYCL path reads a prompt into a slot session, or reads two prompts at once.
   - grep for BCO, STRATA_SLOT_READ and STRATA_PIPED_CONCURRENT in gen returns 0 matches. All three exist in
     cuda-gen: BCO at cuda-gen:15929-15934 and 17347; SLOT_READ at cuda-gen:15340-15351 and 16558-16576;
     PIPED_CONCURRENT at cuda-gen:16611-16617 and 19802-19925.
3. Chunk granularity. [R]
   - Decode resumes only between 12288-token chunks, for 0.5x the chunk's time (gen:11576-11674, 11578).
   - BYIELD fires only for a waiting prompt shorter than half the current one (srv:1865-1868). Equal-length cold
     prompts never yield.
4. Measured timeline. [R]
   - The first prompt (19798 tokens) is read solo in 16.7-17.7 s (~1182 tok/s; cand-log cold C4, lines 941-1006 and
     1174-1237).
   - Promotion takes 150-165 ms (park ~105, copy 26, draft_first 21).
   - Each later prompt is "read in 2 parts, the slots decoding ~4980 ms between them" (12288 + 7510 tokens), 21.7 s
     total (909-912 tok/s effective).
   - Each admission also pays a park_current of ~110-135 ms.
5. The arithmetic matches. [I]
   - TTFT_k = 17 + (k-1) x (0.15 + 16.7 + 5.0 + ~0.1) = 17 / 39 / 61 / 83 s.
   - The cold window line confirms one slot is mostly decoding alone: "788 windows, avg 2.08 rows, 21.24 ms/window"
     at cand-log:1006. [R]

### 2b. What the three knobs do

1. --prefill 12288: the prompt chunk length (gen:1979). [R]
   - It sizes the prompt buffers: 5812 MiB on card 0 (cand-log). [R]
   - A smaller chunk gives decode more frequent breaks, with the same total share. It does not admit any request
     sooner. [I]
2. --no-prefill-borrow (gen:1991) is already forced by EP (gen:2436-2463). It is a no-op in this config. [R]
3. STRATA_PREFILL_FIRST=0 turns off the short PLE-overlap first chunk (pf:2930-2943). [R]
   - Its effect on 20K TTFT is small: one PLE gather, roughly 12288 x PLE row bytes from page cache. [I]

### 2c. What real interleaving needs (ordered by cost)

1. Fairness knob, config only. [I]
   - STRATA_BATCH_DECODE_SHARE=0 would make the TTFTs 17/34/51/68 (mean 42.5 instead of 50) but starves the running
     streams during long reads. A value of 0.25 is the middle ground (-2.5 s per later request).
   - This is the cheapest lever and needs no code.
2. Bounded prefill floor. [I]
   - With equal prompts the mean-TTFT optimum is FCFS without decode breaks, 17/34/51/68. Anything better needs
     higher prefill throughput.
   - Today: 1182 tok/s on the first chunk, and 4K/32K measured 889/1173 (BRIEF baseline).
   - prefill_analysis.md (night) traced it to two host blocks per layer and about 55-58 us of host enqueue per
     active peer expert. In the current source those are the per-layer grouping waits at pf:4259-4340.
3. Per-slot prompt reads. [I]
   - Port the CUDA STRATA_SLOT_READ idea (a slot's windows read a few prompt rows in place, beside the other slots).
     It needs piped groups and deferred commit, which the SYCL EP path does not have (gen:2436-2463 disables
     pipeline windows). Large.
4. Layer-major co-read (BCO). [I]
   - Port BCO from cuda-gen:15929-15934 and 17347; the Python side exists (srv:1760-1802). It reads the waiting
     prompts into idle slot sessions in the same pass, sharing each layer's expert reads.
   - At 12288-token chunks prefill is not weight-bound, so the gain is in host grouping amortisation and card-1
     occupancy, not in weight bytes.
   - It makes every TTFT about the batch's end, so the first request gets worse. It only helps C4 TTFT combined
     with a cap on co-read length.
5. Mixed windows. [I]
   - A verify window that carries N prompt rows of one admitting slot in addition to the decode rows. This is the
     vLLM-style chunked-prefill token budget.
   - Requires:
     - prompt rows in the batch graph layout (the layout key at ver:3949-3981 would multiply);
     - prompt rows writing a slot session that is not the main ss;
     - removing ctl's hold across BADM (srv:2152);
     - prompt buffers coexisting with slot windows (5.8 GiB today).
   - Largest change.

## 3. Forced RAM restore TTFT 2.08 s vs 211 ms native restore

### 3a. Data (ctx-log:270-365; summary-reconciled.json)

- The forced TTFTs are 4.648, 1.443, 2.717 and 1.449 s (median 2.083). Requests arrive in pairs (AB, CD). [R]
- The first of each pair gets ~1.44 s. The second waits for the first's whole admission plus promotion. [R+I]
- One forced request, in order: [R]
  1. "parked 131935 tokens in 435.1 ms" (park_current of the outgoing main session, snapshot 2.25 GB).
  2. "restored 132243 tokens (live) in 210.2 ms".
  3. 24 tokens read as windows in 153 ms, then 6 tokens in 43 ms.
  4. "31 read in 889 ms". prompt_ms includes the park and the restore (gen:10939 -> gen:11818).
  5. A first window of 19 ms.
  6. "KV streaming: 0.00% of 6168 block reads hit VRAM".
- The promotion BGEN of 132275 tokens then costs, with ctl held: [R]
  - park_slot 1509.4 or 422-445 ms;
  - copy 160-170 ms;
  - draft_first 11-210 ms.

### 3b. Budget, first request of a pair (1.44 s)

| Item | ms | Where | Tag |
| --- | ---: | --- | --- |
| Python: chat template + tokenize 132K + _slot_prefix slicing + pipe write + SSE | ~530 | srv:4893-4907, 2307-2355, 2070-2082 | I (TTFT minus engine time) |
| park_current of outgoing main session (2.25 GB host snapshot, CPU memcpy) | 435 (range 59-718) | gen:11035, 8489-8506; snap:114-141 | R time, I mechanism |
| native restore into main session (host pools -> host pools, CPU memcpy + kv_stream_reset) | 210 | gen:11094-11155; snap:229-263 | R |
| 24 + 6 tokens read as windows over a cold KV map (all blocks fault over PCIe) | 196 | gen:11258-11268; lay:1012-1015 | R time, I cause |
| other (mtp.kv_restore, checkpoints) | ~40 | gen:11223 | I |
| first window | 19 | | R |

The restore copies ~1.6 GB of int8 KV (12 QSA layers x 2 heads x 256 x 2 x 132K) plus ~115 MB of GDN state in
210 ms, about 8 GB/s, one CPU memcpy. [I] The 435 ms park moves a 2.25 GB snapshot at about 5 GB/s. [I]

### 3c. Second request of a pair (+1.27 s or +3.2 s)

- It waits on ctl (srv:982 / 2152) while the first does BADM: park_slot (gen:9975-10023), copy_to_slot
  (gen:9799-9868) and draft_first (mtp:2157). [R]
- Measured: park 422-1509 ms + copy 160-170 + draft_first 11-210. [R]
- The 1509 ms park and the 4.65 s TTFT come from the same AB pair. [R]

### 3d. Fixes (with checks)

1. Restore directly into a free slot. [I]
   - Skip the solo main-session read and the later promotion: restore the snapshot into a slot session, then read
     the ~31-token tail as slot windows. This removes park_current, park_slot and copy_to_slot from the critical
     path (about 0.6 s on the first request, 1.3-3.2 s on the second).
   - Code: gen:11094-11155 (restore target), gen:13252-13315 (BADM), srv:1931-1947 (solo path choice).
   - Check: compare the STRATA_TRACE RESUME and BADM lines in ctx-log against srv first_engine_token_s (srv:5461).
2. Async or deferred park_current. [I]
   - The outgoing main-session snapshot is not needed before the new request's first token. Move it to a worker
     thread after the RESUME, or skip it when the conversation cache already holds an identical copy
     (reused_kv_bytes 2.0 GB when retained, ctx-log).
   - Code: gen:8325-8506, snap:172-206.
   - Check: the "parked ... in X ms" times versus the prompt_ms delta.
3. Multithreaded host_only_copy. [I]
   - The 1950X has 4 DDR4 channels. A single-threaded memcpy reaches ~5-8 GB/s; 8-12 threads should reach 20-30 GB/s,
     which would cut park, restore and copy by 3-4x (0.85 s -> ~0.25 s for the first request).
   - Code: snap:114-141.
   - Check: the park/restore ms in the log before and after; no GPU change.
4. Prefetch the KV map after a restore. [I]
   - kv_stream_reset leaves nothing resident (lay:1012-1015; snap:229-263), so the tail reads fault every block
     (0.00% hits). Prefill the newest 65536 cells' map, or the last top-k selection, during the restore.
   - Code: snap:229-263, lay:1306 (qsa_kv_resolve).
   - Check: the KV streaming hit line after a restore, and the tail-window ms.
5. Python. [I]
   - Cache the rendered template and ids per conversation (encode is already incremental, srv:4893-4907).
   - Replace _slot_prefix slicing with an index comparison over a numpy or array.array (srv:2307-2355).
   - Check: py-spy dump during a forced restore; first_engine_token_s minus request arrival.

## 4. EP split, transfers, card-1 waits, balance options

### 4a. Placement and roles

- Placement is a snake (gen:5205-5250). Card 0's quota is 169 experts/layer and card 1 gets 343. The hottest experts
  are dealt alternately until card 0 is full; the cold rest go to card 1. [R]
- cand-log: [R]
  - card 0: 13.19 GiB of experts (8112 slots);
  - card 1: 16464 experts, 26.77 GiB, 3483 MiB free after 1260 MiB of exchange regions;
  - slot sessions: 4.93 GiB on card 0;
  - prompt buffers: 5812 MiB.
- Card 0 runs: [R]
  - embed/PLE, HC, GDN, QSA attention (KV lives on card 0), router and plan for both cards (ver:1640-1682, PLAN1);
  - the shared expert (ver:1716-1756, deferred by SH_LATE);
  - its own routed experts, the combine (ver:1957-1962), the head, sampling and the MTP drafter.
- Card 1 runs, on ep_q_, only: [R]
  - the ep_wait spin, ep_copy_uncached and native_expert_grouped for its experts;
  - ep_send_psum, which returns one fp32 row per token to card 0 (ver:1771-1814).
- Card 0 does ep_wait on the return flag (ver:1903-1905). The exchange regions are Level Zero IPC device allocations
  (engine/sycl/include/strata/core/ep_mem.hpp, EpRegion). [R]
- Startup log: "inbox 25.2 KiB per group, spin bound 2000000". [R]
- Card 1 waits in ep_wait spin kernels for card 0's push of each layer. Card 0 waits in ep_wait for card 1's psum
  only when card 1's expert chain is longer than card 0's own experts plus the late shared expert. [R+I]
- Old-night trace (strata/flash-next/night/agent-reports/trace_final_analysis.md:53, 86): at C1, card 1 computes
  4.92 ms and spins 15.54 ms of a 29.26 ms profiled window; card 0 waits 0.57 ms on card 1. [R]
- Card 1's routed kernels run 20-30% slower per call than card 0's at equal entries (moe_report.md 8.4). [R]
  The split balances entries but not distinct experts (C1 card 0 10 groups / 20 entries, card 1 18 groups / 20
  entries; moe_report.md:91), so card 1 reads more weight bytes. [R+I]

### 4b. Bytes per window over PCIe (Gen3 x16, ~12 GB/s each way) [I]

| Flow | per row per layer | C4 window (8 rows x 48 layers) | Note |
| --- | ---: | ---: | --- |
| push card0->card1: q8_1 row (2560 B + 320 B scales) + ids/weights (10 x 8 B) + plan | ~2.9 KB | ~1.1 MB (+ plan, inbox 25.2 KiB/group cap) | ver:1771-1814 |
| return card1->card0: fp32 psum row | 10,240 B | 3.9 MB | ep_send_psum |
| total | | ~5 MB, ~0.4 ms at bandwidth | 96 dependent small hops: latency-bound, not bandwidth-bound |

Other movements per C4 window:
- PLE H2D from a host gather: rows x PLE row bytes. [R site ver:3993-4005, size I]
- Final_R D2D: 4 x 81,920 B. [R]
- Sample readback through mapped h_out_. [R]
- KV misses: zero-copy reads of pinned host KV inside qsa_kv_resolve (lay:1306-1307; in the graph, ver:1586). [R]
  Decode after a cold admission reads about 53 MiB from RAM per request at ~69.5% hits (cand-log:949-1005). [R]

### 4c. Host syncs per C4 window (batch_step) [R sites, count I]

| # | Sync | Site |
| ---: | --- | --- |
| 1 | cs_->wait() after the verify graphs | ver:4089 |
| 2 | copy_->wait() | ver:4089 area |
| 3 | sample_rows final wait (+1 per penalized row; DD greedy-ish: usually 0) | ver:4294-4333 |
| 4 | commit graph cs_->wait() | ver:4258 |
| 5-8 | per-slot final_R memcpy().wait(), 81,920 B each | gen:10184-10187 |
| 9-12 | MtpPair::draft, 2 waits per pair x 2 pairs | mtp:2493, 2514 |
| - | fflush per window; one BT line per token | gen:10159, 10257 |

About 12 blocking host waits per C4 window. The measured host-serial share is commit 2.76 + emit 3.01 = 5.77 ms of
44.14 (13%) at C4, and 2.76 of 26.70 (10%) at C2 (cand-log:1919, 882). [R] Old night had the same commit and emit
(old-log:2002), so this is not the regression. [R]

### 4d. Card-balance options

1. Shared expert on card 1. [I]
   - Card 1 already receives each row's q8_1 activations in the push, so shared GU+down can use them, and its output
     can be added into the psum before ep_send_psum. No extra transfer is needed except the shared-gate scalar
     (4 B/row) if the sigmoid gate stays on card 0.
   - Weights: 48 x 3 x 2560 x 640 at ~4.5 b, about 0.13 GB. This fits in 3.4 GB free.
   - Gain: about 1.1 ms per C1 window (fused shared expert, trace_final_analysis.md 2), more at C4 where card 0's
     shared rows grow.
   - Code: ver:1716-1756 (shared on card 0), ver:1771-1814 (peer side), ver:1957-1962 (combine), gen:6586-6628
     (ep_cfg).
   - Blocker: SH_LATE already overlaps the shared expert with the psum wait, so the gain is real only while card 0 is
     the critical chain. At C1 it is, at 95% card-0-kernel bound (trace_best3_analysis.md).
2. Shift expert quota to card 1. [I]
   - Card 1 has 3.4 GB free, about 2000 experts at 1.66 MB each (26.77 GiB / 16464), about 40 per layer. Moving card 0
     from 169 to ~130 per layer cuts card 0's routed bytes and frees ~2.6 GB on card 0 (KV residency or prompt
     buffers).
   - Code: gen:5205-5250 (q0 quota), gen:5429-5484 (peer fill).
   - Blocker: card 1's per-call slowness (20-30%) and the spin-then-psum latency. Verify that card 1's chain stays
     shorter than card 0's.
3. Vocab head or drafter on card 1. [I]
   - The head is ~0.83 ms per C1 window. It needs the final hidden pushed (10 KB/row) and ids returned. The drafter
     depends on the verify output and holds both pairs' state on card 0 (pair requires same device).
   - Low gain without cross-window overlap.
4. Cross-window pipelining (card 1 runs pair B's expert stage while card 0 runs pair A). [I]
   - Blocked: EP disables pipeline windows (gen:2436-2463), #2800 crashes on SYCL, the host loop is serial
     (gen:10417-10459), and MTP drafting needs the previous verify.
5. Not possible: moving attention or GDN to card 1. All KV and GDN state is on card 0 (session_init, gen:4406-4470). [R]

## 5. Why per-stream decode falls (134.6 -> 65.4 -> 37.3, C1 -> C2 -> C4)

- C1 runs the solo GEN path with --spec 4 / --mtp-max-t 4: about 3.1 tokens per ~23 ms window. [R path srv:1931-1947;
  I tokens/window]
- At C2 and above, requests are promoted to batch slots with STRATA_BATCH_MTP_DEPTH=1 (config): 2 rows per slot, at
  most 2 tokens per window. [R]
- Measured per window and slot: [I]
  - C2: 65.4 x 0.0267 = 1.75 tokens per slot per window;
  - C4: 37.3 x 0.0441 = 1.65.
- C1 -> C2 is mostly the speculation depth (3.1 -> 1.75 tokens per window), not window cost. C2 aggregate (130.8) is
  below C1 (134.6). [I]
- Window growth with rows (cand-log): [R]
  - 21.2 ms at 2.08 rows;
  - 26.7 ms at 3.70 rows: +3.4 ms per row;
  - 44.1 ms at 6.81 rows: +5.6 ms per row.
- The superlinear part comes from: [I]
  - per-row QSA attention (finding 1);
  - expert-union growth;
  - host commit and emit, which double from C2 to C4 (2.76 -> 5.77 ms).

## 6. Ranked engineering candidates

Expected gains are inferred. Every item needs an unprofiled A/B on the same binary.

| Rank | Candidate | Target | Expected gain | Code | Cheap check (confirm/kill) |
| ---: | --- | --- | --- | --- | --- |
| 1 | Streamed-KV batched QSA attention: drop the kv_mode != 1 gate by grouping rows per slot, with qsa_kv_resolve per slot (as old-ver:1372-1420 did), then one top-k + one multi-pool attention launch per window | C2/C4 agg | recover most of +3.2 ms (C2) / +6.0 ms (C4) per window: C2 ~+12%, C4 ~+15% | ver:1539-1592; kernels qsa_decode_attn (old: qsa_decode_attn_multi, old src qsa_decode_attn.dp.cpp:680) | (a) unitrace one C4 window: count qsa_block_scores/topk/kv_resolve/attn launches (expect 384). (b) Config-only A/B: --kv-resident 0 at --max-context 65536, C4, BATCH_ATTN_ROWS 0 vs 1: if rows_ok engages and run drops ~5 ms, confirmed. |
| 2 | STRATA_BATCH_DECODE_SHARE lower (0.25 or 0) during cold bursts | C4 TTFT | -2.5 to -5 s per later request (mean 50 -> 42.5 s at 0) | gen:11576-11582 | cold C4 harness, env only; watch per-stream stalls |
| 3 | Restore-path fixes: restore into a slot, async park_current, multithreaded host_only_copy, map prefetch | restore TTFT | 2.08 -> ~1.0 s median; second-of-pair -1.3..-3 s | gen:8325-8506, 11035-11155, 13252-13315; snap:114-141, 229-263 | log park/restore ms; py-spy on Python 0.53 s; KV hit line after restore |
| 4 | Release ctl before BADM's park/copy (copy async on the engine side, Python waits only for READY of the slot) | C4 TTFT, restore pair | removes 0.6-1.7 s of queueing per paired arrival | srv:2152, 2070-2082; gen:13252-13315 | timestamp ctl acquire/release in srv; compare to engine BADM timings |
| 5 | Host-serial window: one batched final_R copy (one wait instead of 4), MTP pair waits merged, commit fused into the next window (CUDA has deferred commit) | C2/C4 agg | 1-3 ms of 44 at C4 (2-7%) | gen:10184-10187, 10208-10256; mtp:2493, 2514; ver:4218-4287 | STRATA_TRACE commit/emit split already logged; count host waits with an ltrace or perf probe on zeCommandQueueSynchronize / event host sync |
| 6 | Shared expert on card 1 | C1/C2/C4 | ~1.1 ms per C1 window, more at C4 | ver:1716-1814, 1957-1962 | unitrace: card 0 shexp_gu/down time per window vs card 1 spin; kill if card 0 is not critical at C4 |
| 7 | Expert quota shift to card 1 (169 -> ~130/layer) | C2/C4 agg; frees card-0 VRAM | ~0.5-1 ms per window, plus VRAM for KV residency | gen:5205-5250, 5429-5484 | unitrace card 0 vs card 1 routed chain per layer; card 0 ep_wait must stay ~0 |
| 8 | Batch MTP depth 2 at C2 (6 rows; GDN_COMMIT_LAYERS engages at S=6) | C2 per stream | uncertain; old night C4 depth 2 was slower | config STRATA_BATCH_MTP_DEPTH; mtp pair graph | C2 A/B, accepted tokens per window from STRATA_TRACE |
| 9 | Prefill throughput: per-layer host grouping waits and peer enqueue (48+ blocks per chunk) | C4 TTFT (floor) | prefill 1180 -> 1500+ tok/s would cut every cold TTFT proportionally | pf:4259-4340, 4341+ | STRATA_PREFILL_TIMING=1 phase table (pf:2680-2743): "host grouping" and "wait copy" share |
| 10 | Python long-prompt overhead | restore TTFT | ~0.3-0.5 s | srv:2307-2355, 4893-4907 | py-spy record during a 132K forced restore |
| 11 | Port BCO / in-slot reads to SYCL | C4 TTFT distribution | large effort; helps mixed-length bursts more than equal ones | cuda-gen:15929-15934, 17347, 15340-15351, 16558-16576; srv:1760-1802 | only after 2 and 9 are measured |

## 7. Open items not verifiable statically

- Whether the qsa_kv_resolve per-row launches or the attention per-row launches dominate the +6 ms at C4 (unitrace).
- PLE bytes per row and whether the host gather (ver:3993-4005) is on the critical path between graphs.
- Exact Python cost split of the ~0.53 s (py-spy was not run).
- Whether card 0 or card 1 bounds the expert stage at C4 (the night traces are C1/C2 only).
