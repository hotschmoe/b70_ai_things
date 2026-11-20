# DESIGN-B3 (draft): Strata serving features in TensorFold XPU Flash-Next (TP2, two B70s)

Author: worker fable-b3 (claude-fable-5-2) for Lead B, 2026-11-20. Status: M-B3 design study, CPU only, no code edited.
Tags: [R] read in source (file:line), [M] measured (cited), [I] inferred / arithmetic (a target, not a result).

Source keys. TF = /mnt/vm_8tb/b70/build/tensorfold/fn-runtime @ eead3e0 (PR3603 207aaae + hi/lo fix); Lead A's fn-a3
branch may have moved lines. Under zig/src/families/flashnext/: st = xpu_state.zig, fw = xpu_forward.zig, op = xpu_ops.zig,
sto = xpu_store.zig, en = xpu_engine.zig. nfn = zig/src/native/xpu_flashnext.zig, ln = zig/src/native/xpu_lane.zig,
lp = zig/src/native/xpu_lane_prompt.zig, mx = zig/kernels/xpu/flashnext/fn_mixers.cl, core/ = zig/src/core/.
Strata (MIT) = /mnt/vm_8tb/b70/build/strata-0216/src: q8 = engine/sycl/src/kernels/cuda/kv_q8.dp.cpp, q8h/ksh =
engine/include/strata/kernels/kv_q8.hpp / kv_stream.hpp, lay = engine/sycl/src/core/layer.cpp, snap =
engine/sycl/src/core/conversation_snapshot.cpp, gen = engine/sycl/src/program/generate.cpp.
Measured: PLAT = /mnt/vm_8tb/b70/results/tf-flashnext/B/latbench/PLATFORM_DRAFT.md (card 1 single, pair push; PROVISIONAL),
J = tensorfold/flashnext/JOURNAL.md, walk = concurrency-0216/analyst/opus_sched_prefill_restore_walk.md.

## 0. Decisions on one screen

1. One paged KV pool per card, int8 cells, page = 512 tokens spanning all attention layers. A cell is Strata's format:
   256 int8 codes + 4 fp16 scales (one per 64 channels) for K and for V = 528 B per token per layer per card (1 KV head
   per card on TP2). Scales are dynamic per position (absmax / 127), not offline calibrated: that is what Strata does [R].
2. Drop the per-token indexer key cache (ikc, 256 B/token/layer): only the last 3 rows are ever read. Without this, 4 x
   200K does not fit a card; with it, it fits on paper (section E) [I].
3. KV pages are append-only, so they are shared by refcount (prefix cache, parked streams) and mirrored to pinned host
   RAM write-behind by the copy engine. GDN/conv/PLE state is overwritten, so it is snapshotted (58 MB per card).
4. Park and prefix-cache entry are the same object (a "Snap": state blob + page id list + tails). Park = snapshot with
   page ownership moved, restore = resume into any free slot. Eviction cost is O(state) = ~14 ms, not O(context) [I].
5. Park/restore copies run on a separate in-order immediate list on queue group ordinal 1 (copy engine) per card, are
   appended only at round boundaries (after the round's existing readback sync) for immutable sources, and are observed
   by polling a completion event. The decode list never waits on them and the host never syncs for them.
6. Scheduler: first a config-only arm (busy chunk 1024, fcfs, decode share), then mixed passes (decode rows ride prefill
   chunks). Honest bound: for 4 equal cold prompts the TTFT tail is 80K tokens / prefill rate under any policy.
7. Order of work follows the user priority: eviction-to-RAM without recompute first, then slots/concurrency, scheduler,
   then decode items (pointer-swap GDN state, attention traffic), then MTP KV ring, then prefix page sharing polish.

## A. Per-stream state inventory (TP2, per card; nk=8, nv=24, nq=12, nkv=1 [R sto:74-99])

| Buffer (per stream, per card) | Shape / type [R] | Bytes | Write pattern | Class |
|---|---|---:|---|---|
| kc, vc x 12 attn layers | [cap, 1, 256] bf16 (st:36-37, 236) | 2 x 512 B/token/layer | written at pos by fn_attn_prep (mx:247-249) | append-only |
| ikc x 12 | [cap, 128] bf16 (st:38, 237) | 256 B/token/layer | written mx:226; read only by fn_qsa_pool for the 4 keys of a completing block (mx:262-265) | append-only, but only the last 3 rows are live [I] |
| pooled x 12 | [(cap+3)/4, 128] bf16 (st:39, 238) | 64 B/token/layer | one row per completed block of 4 (mx:274); all rows read by fn_qsa_scores (mx:293) | append-only |
| mtp_kc/vc/ikc/pooled | same as one attn layer (st:43-46, 272-277; attn_total st:239) | 1344 B/token | forwardMtp at mtp_pos0 (fw:441); draft chain writes ahead of pos (en:273) | append-only + speculative tail |
| GDN state x 36 | [24, 128, 128] f32 (st:33, 234) | 1.573 MB each, 56.6 MB | fully rewritten per pass (mx:180-182) | overwritten: snapshot |
| tmp_state x 36 | same (st:35) | 56.6 MB | verify end-state (op:109, 289); copied to state on full keep (fw:366) | rollback only |
| conv x 36 | [3, 5120] bf16 (st:32, 233) | 30.7 KB each, 1.1 MB | sliding window (fn_shift_window mx:77-86) | overwritten: snapshot |
| saved_proj x 36 | [16, 8240] bf16 (st:34) | 9.5 MB | verify rows, for partial-keep replay (fw:373-386) | rollback only |
| ple_tail | [9, 10240] bf16 (st:40) | 184 KB | sliding window (op:254; fw:359) | overwritten: snapshot |
| saved_ple_nrow, verify_streams | [16, 10240] bf16 (st:41, 48) | 328 KB each | per verify window | rollback / MTP feed |
| last_streams, mtp_out | [10240] bf16 (st:47, 49) | 20 KB each | per pass / per draft step | snapshot (tiny) |
| host: pos, history[2], win_* (st:52-64); Lane held/probs/pending/start/first_handle (ln:53-62) | host structs | < 1 KB | per round | snapshot (host copy) |

Totals today [I from st:232-241]: fixed 124.7 MB per stream per card; 17,472 B per token per card (13 layers incl. MTP,
bf16); 200K = 3.49 GB per stream per card. Head-split buffers (kc/vc, GDN state, conv) hold this card's half; indexer keys
(ikc, pooled) and ple_tail are replicated identically on both cards (DESIGN-A section 1) [R].

Facts that shape the design:
- Rollback of KV is implicit: a rejected verify row is simply overwritten by the next pass at the same position
  (op:327-331 writes at sq.pos; nothing reads above pos). So every K/V/pooled cell below the committed position is
  immutable forever [R]. Only GDN state, conv, ple_tail need copies.
- A new stream allocates full-context buffers and zero-fills each through a host sync (st:255-281 allocZero ->
  sto:280-307 zeroBuf sync). At 200K bf16 that is ~3.5 GB of H2D zeros per card (~270 ms at 13 GB/s) plus ~200 host
  syncs on the decode list per admission [I from R]. A preallocated page pool removes this.
- snapSave copies the whole KV prefix into fresh device allocations with two syncAll (st:369-417): at 20K tokens that is
  350 MB per card per snapshot today [I].
- keepRows on full accept copies tmp_state -> state for 36 layers (56.6 MB per card per stream per window, fw:363-366);
  D2D 64 MB measured 92-139 us append+sync [M PLAT; faster than 608 GB/s would allow for read+write, verify before use].
- parked_pages.zig is NOT a KV park tier (it reads /proc/zoneinfo per-CPU free pages) and disk_spill.zig is flag
  plumbing only [R zig/src/native/parked_pages.zig:1, disk_spill.zig:1]. XPU has no host tier today.

## B. int8 KV for the Flash-Next attention layers

### B.1 Format and layout
- Cell (one token, one layer, this card's KV head), 528 B, contiguous:
  [K codes i8 x 256][V codes i8 x 256][K scale fp16 x 4][V scale fp16 x 4].
  Scale group = 64 channels, scale = fp16(max|x| / 127), code = clamp(rint(x / stored_scale), -127, 127), as Strata
  (q8:83-91; q8h:1-13) [R]. Row-interleaving K and V differs from Strata's separate arrays (lay:841-844): the TF attention
  kernel reads K and V of every selected key in one loop (mx:435-446), so one contiguous 528 B read per key is better [I].
- Page = 512 consecutive tokens = 128 indexer blocks. Per layer a page holds 512 cells (270,336 B) + 128 pooled keys bf16
  (32,768 B) = 303,104 B. A page id addresses the same slot in all 13 layers (12 + MTP): pool[page][layer] contiguous,
  3.94 MB per page per card. One host-side allocator hands out the same ids on both cards (TF core kv_pool.Pool already
  has ids, holders, retain/release, promise: core/kv_pool.zig:223-340; Layout.token_bytes is "across every paged cache and
  layer", core/kv_pool.zig:12-24) [R].
- Per stream: a device page table PT i32[512] per card (key position p -> PT[p >> 9]); the host appends a small H2D of new
  entries on the decode list before the pass that first writes a page. Strata uses the same indirection with page = 4
  cells (ksh:1-8; q8:93-98) because it streams blocks; we page for sharing and parking, so 512 keeps copies above ~1 MB,
  where the measured copy cost is throughput and not the ~6-10 us fixed cost [M PLAT copy table].
- QSA compatibility: selection is unchanged. fn_qsa_select emits key positions in blocks of 4 (mx:399-411); the page lookup
  happens where a position is dereferenced. 512 % 4 == 0, so a block never straddles a page.
- ikc becomes ikc_tail [3, 128] bf16 per layer (a sliding window like conv, advanced with the existing fn_shift_window)
  plus the pass's own rows in scratch. Strata keeps exactly this (idx_tail, (idx_block-1) x idx_dim, lay:767) [R].
- Pooled keys stay bf16 (Strata keeps fp32, lay:769). They drive block selection, are 11% of a page, and must be fully
  resident for an active stream: 12.8 MB per layer per card at 200K [I].

### B.2 Kernels that change (all in zig/kernels/xpu/flashnext/fn_mixers.cl; Zig call sites op:124-185, 299-365, fw:441)
| Kernel | Change |
|---|---|
| fn_attn_prep (mx:216-253) | KV heads: after norm + RoPE (K) or raw projection (V), per 64-channel group absmax (4 subgroups combined through local memory, same pattern as wg_sum), write codes + scales into the cell at PT[pos >> 9]. Index-key head writes the pass scratch row instead of IKC[pos]. New args: POOL, PT, layer offset. |
| fn_qsa_pool (mx:256-275) | read the 4 keys of block b from ikc_tail (positions < pos0) or the pass scratch (>= pos0); write POOLED through the page table (block b -> page b >> 7). |
| fn_qsa_scores (mx:278-309) | read POOLED through the page table (one lookup per 64-block tile; a tile never straddles a page since 128 % 64 == 0). |
| fn_attn_sparse (mx:420-454) | key -> cell address via PT; d = sum over 4 groups of ks[g] * sum(q[i] * code[i]); o += p * vs[g] * code. fp32 math, no fp16 round of the product. |
| fn_shift_window (existing) | also advances ikc_tail on commit (inplace pass end; keepRows fw:359 analog). |
| fn_qsa_select, fn_attn_merge, fn_attn_gate | unchanged. |
Zig: st.zig RankSeq/Seq/Snap lose kc/vc/ikc/pooled pointers and gain {pt, pages lease, ikc_tail}; seqBytesPerRank and
snapBytes become page math; sto.zig Kernels unchanged in count. A build flag keeps the bf16 cell (1,024 B + pooled) on
the same page pool so the bf16 reference arm uses the same code path (needed for B.4).
Numerics hazard [M J 07:35]: "quantize against the stored fp16 scale" is an fp32 -> fp16 -> fp32 round trip. IGC 26.22
folds (float)convert_half(x) to x. Use widen_half() (zig/kernels/xpu/qwen27/half_exact.h) or store scales as bf16 via the
integer helpers bf()/to_bf() (flashnext.cl:9-11). Otherwise codes are quantized against a scale that is not the stored one.
Reference in tree: the 27B int8 KV (XPU_KV_INT8) uses one fp32 scale per position and KV head (store_kv8,
zig/kernels/xpu/qwen27/glue_attn.cl:43-52; zig/src/xpu/slot_states.zig:29-36) and reports perplexity +-0.04%, 15 of 24 greedy
answers identical (docs/xpu/README.md:76) [M-TF]. Strata's 4 scales per head is finer; we take Strata's.

### B.3 Rotation and clip options
- Hadamard rotation (Strata STRATA_KV_ROT=1; TF has fwht256 for the 27B study, glue_attn.cl:6-21): default OFF. Strata's own
  note: first-token KL vs fp16 improved on an NVFP4 pack (0.0066 -> 0.0051) but got worse on IQ2_XS (0.0022 -> 0.0054)
  (gen:2515-2516) [R of their M]. Keep as a kernel flag (rotate q, k, v in fn_attn_prep, rotate the output back in
  fn_attn_merge), enable only if B.4 shows a gain on our IQ3 pack. A rotated cell is a different format: tag it in the
  Snap header so it never restores into an unrotated server (snap:49-54 does the same) [R].
- Clip factor c (scale = c x absmax / 127, c in {1.0, 0.98, 0.95}): a one-line knob to test in B.4; default 1.0 [I].

### B.4 Memory math (per card, TP2) [I arithmetic from the sizes above]
| | B/token (13 layers) | 16K | 64K | 200K |
|---|---:|---:|---:|---:|
| bf16 today (kc, vc, ikc, pooled) | 17,472 | 286 MB | 1.15 GB | 3.49 GB |
| int8 cells, ikc kept | 11,024 | 181 MB | 722 MB | 2.20 GB |
| int8 cells + ikc_tail (this design) | 7,696 | 126 MB | 504 MB | 1.54 GB |
| same, MTP KV as a ring (later, Strata lay:732-734) | 7,104 | 116 MB | 466 MB | 1.42 GB |
200K x 2 streams = 3.1 GB, x 4 = 6.2 GB per card, against ~9.45 GB free after weights + MTP (9.97 GB Sysman after the GGUF
load minus 0.52 GB MTP [M J 08:11, 08:25]). Section E does the full fit with scratch and state.
Side effect on decode latency [I]: fn_attn_sparse launches one work-group per (key slice, query head, row) and each reads
the K and V of its 32 keys (mx:424-446), so 12 query heads re-read the same 2,051 keys. Unique bytes per row per layer per
card: 2.1 MB bf16 -> 1.08 MB int8; worst case with no cache reuse across heads 25 MB -> 13 MB (x 12 layers = 302 -> 156
MB per row). Where the truth sits between those is unknown: profile with the M-B1 tracer. A KV-head-major attention kernel
(read each key once for its 12 query heads) is the follow-up if the worst case is real.

### B.5 Validation plan
1. Kernel parity, CPU reference (no model): Zig reference of quantize/dequantize and of the sparse attention over int8
   cells; GPU vs reference bit-exact for codes and scales, <= 1e-6 rel for attention output. Includes scale-group, page
   boundary, PT indirection, ikc_tail and pooled-through-PT cases. One card, minutes.
2. Storage-only A/B at model level (needs Lead A's M-A3 forward): same engine, bf16 cells vs int8 cells, teacher-forced
   over Lead A's Strata oracle continuations (/mnt/vm_8tb/b70/results/tf-flashnext/oracle/{prompts.json,oracle.json}: 8
   prompts of 22-4805 tokens, 256 greedy tokens, top-20 logprobs [M J 07:33]) plus two long prompts (>= 32K and >= 120K) so
   the top-512 sparse selection (active only past 2,048 tokens, mx:327) and many-key softmax are exercised. Metrics per
   arm: NLL of the oracle tokens, top-1 agreement with the bf16 arm, mean KL of top-20. Gate: dNLL <= 0.005 nats and
   top-1 agreement >= 99% vs bf16 KV, and no worse agreement with the Strata oracle than the bf16 arm (the oracle ran
   fp16 KV). Report the SEM (Lead B's 27B A/B had SEM 0.0021 nats on 129 positions [M J 08:25]): use >= 2,000 positions.
3. Options sweep on the same harness: rotation on/off, clip c. Adopt only on a measured gain outside the SEM.
4. Contracts that must still hold in int8 mode: drafted == serial, concurrent == solo, chunk-invariant prompts (bitwise,
   int8 vs int8). int8 vs bf16 is NOT expected to be bit-equal.

## C. Park / restore to host RAM without recompute

### C.1 What is copied
A Snap (parked stream or prefix-cache entry) per card: (a) state blob = GDN state 56.6 MB + conv 1.1 MB + ple_tail 184 KB
+ ikc_tail x 13 (10 KB) + last_streams + mtp_out = 58.0 MB, a single contiguous device buffer (lay the per-stream state
out as one slab so one copy moves it); (b) the page id list (KV + pooled, main and MTP) up to pos; (c) host fields: pos,
history, Lane held drafts/probs/pending/start, sampling position, cell-format tag. Rollback buffers (tmp_state,
saved_proj, saved_ple_nrow, verify_streams) are never parked: a stream is parked only at a settled round boundary (after
keep and after the draft absorb, ln:209-214, 473-499), where they are dead [I from R].

### C.2 Page states and the write-behind mirror
Each page: VRAM-only, VRAM+RAM (mirrored), RAM-only. A page is CLOSED when its last token is < pos - 32 (two full
windows of guard for MTP draft-ahead writes, en:273); closed pages are immutable (section A). At each round boundary the
host queues a D2H copy of newly closed pages on the park list (3.94 MB each: 0.88 ms at 4.5 GB/s [I from M PLAT]). Mirror
traffic is tiny: prefill at 2,000 tok/s x 7.7 KB = 15 MB/s per card. Therefore:
- Park (evict) = D2H of the open tail page(s) + the state blob = ~4-8 MB + 58 MB = ~14-15 ms per card, then the VRAM
  copies of mirrored pages are released (refcount permitting). Without the mirror a 200K park is 1.54 GB = 342 ms [I].
- Strata comparison: Strata also keeps an authoritative host copy but writes it from inside the append kernel
  (q8:99-103) and parks by single-thread CPU memcpy (2.25 GB in 435 ms, walk 3a-3b) [R/M]. We write behind on the copy
  engine instead, off the decode list.

### C.3 Engines, ordering, no host waits on the decode list
- Per card one extra in-order immediate list on queue group ordinal 1 ("0 compute, 1 copy on the B70",
  zig/src/xpu/ze.zig:447-453; latbench found groups 0 = compute+copy, 1 = copy [M latbench NOTES single-card run]). Today
  every copy goes on the compute list (sto:309-312, d.list created with ordinal 0 at sto:167) [R].
- Ordering by construction, not by events: park-list copies are appended only at a round boundary, after the round's
  existing token readback sync (fw:323-324, op:524-525), and only for sources last written in that round or earlier
  (closed pages, the state slab of a stream that is leaving the rounds, a prefix snapshot slab). At that host time every
  writer has completed, so the copy needs no wait list. The decode list never gets a wait on a park event.
- A stream being parked is removed from the round set first; its state slab then has no writer. A stream that keeps
  running (prefix snapshot at a mark) gets a D2D copy of its state slab on the compute list in order (58 MB; ~0.1-0.2 ms
  if the measured D2D rate holds [M PLAT], well under 1 ms even at 100 GB/s), and that immutable copy is tiered later.
- Double buffering: (1) GDN state A/B per active stream with a pointer swap on full keep, and partial-keep replay writing
  the other buffer (replaces the 56.6 MB copy at fw:366 and the in-place replay at fw:380-382): the committed buffer is
  never written in place. (2) Restore lands in freshly leased pages and a fresh state slab; the stream's PT and slab
  pointers are published by the host at a round boundary after completion. Nothing visible to a running pass changes
  under it.
- Completion: the last copy of a batch signals a fresh event from a HOST_VISIBLE pool; the engine thread polls
  zeEventQueryStatus between rounds (busy-poll wake 4.3 us [M PLAT]) and never calls a blocking wait. No zeEventHostReset
  on events tied to in-order immediate lists that wait on each other (hang hazard, TRANSPORT-B section 2): recycle the
  event pool only when the park list is idle.
- Forbidden paths (both hung this driver): host <-> host copies through the copy engine (Strata: BCS engine reset,
  snap:114-128 [R]); copy-engine PULL from the peer card (BCS CAT error + reset on card 1, latbench 08:28 [M]). Park
  copies are always device-local <-> host on the owning card. Host-side moves are CPU memcpy.
- Throttle: restores are submitted in page units (0.3 ms each H2D) so the copy engine never holds a multi-100 ms burst;
  whether a 13 GB/s H2D raises TP2 hop p99 on the same PCIe link is unmeasured (acceptance test below).

### C.4 Host memory
- Pool: pinned host USM (zeMemAllocHost, made resident on both devices as st:151-158 does), allocated in 1 GiB slabs,
  carved into 3.94 MB page frames and 58 MB state frames. Largest proven host allocation in TF is the 256 MB staging
  buffer (sto:253-254) [R]; the per-allocation and total pinned limits of this driver are UNKNOWN: probe first.
- Budget against 121 GB [M free -g today: 8 GB used, 87 GB page cache, swap 7/7 GB used]:
  PLE table 28.8 GB (page cache; mlock it) + OS/other ~8 GB + server heap ~2 GB + KV mirror pool cap 48 GB + state frames
  4 GB (34 Snaps) = ~91 GB, ~30 GB headroom, with hotschmoe-dd stopped (it must be for research anyway). 48 GB holds 3.1 M
  tokens of context across both cards = 15 streams x 200K (a 200K stream is 3.08 GB over two cards) [I]. The admission
  ledger counts RAM frames as well as VRAM pages; TF's memory-pressure governor (core/lane_host_pressure.zig) stays on.
- D2H is 4.5 GB/s vs H2D 13 GB/s on card 1 (64 MB, both engines alike [M PLAT]); card 0 is unmeasured, the cause is
  unknown (the host is in UMA mode, each card on a different die). The design does not depend on D2H speed (write-behind).

### C.5 Restore latency estimates [I from M PLAT constants; cards restore in parallel, scaling unverified]
| Context | pages | H2D KV per card | + state | Strata reference [M walk 3a] |
|---|---:|---:|---:|---|
| 20K | 40 | 12 ms | 4.5 ms | |
| 64K | 128 | 39 ms | 4.5 ms | |
| 200K (132K in Strata's test) | 391 | 118 ms | 4.5 ms | forced RAM restore TTFT 2.08 s median; native restore 210 ms |
Target: resume-to-first-window <= 150 ms at 200K, <= 25 ms at 20K, with zero recompute. A later refinement (not in the
first stages) restores pooled keys + the newest pages first and streams the rest behind a residency check; it needs a
kernel-visible residency map (Strata ksh:1-20) and is only worth it if 120 ms matters.

### C.6 Correctness gates
1. Bit identity: stream S decoded N tokens never parked vs S force-parked and restored every K rounds (K = 1, 7, 64),
   into the same slot and into a different slot: identical token ids and identical top-64 logit bits at every position,
   greedy and seeded sampling, drafted and serial.
2. Byte identity: D2H then H2D of every Snap part compares equal on device (checksum kernel) for random contents.
3. Non-interference: with stream T decoding, park/restore S continuously; T's tokens == T solo (bitwise), and T's
   decode window p50/p99 from the M-B1 tracer within 2% / 5% of the no-park run; zero host waits attributed to park.
4. Fault drill: kill mid-restore -> stream fails cleanly, pages and frames return to the pools (census idle check as
   core/kv_pool.zig:219-220).

## D. Prefix cache (one mechanism with park)

- Keep TF's store and marks: prompt_cache.Store with the family Snapshots vtable (core/prompt_cache.zig:22-61), marks
  ending chunks (lp:57-65), restore at stream open (lp:24-31). Flash-Next rules are any-row cuts with lookahead 1
  (nfn:18) [R]. Only the family's save/restore/drop/bytes change.
- save(at): Snap = D2D copy of the state slab (58 MB per card) + retain() on every page below ceil(at / 512), including
  the partial boundary page. The owner keeps appending into that page above "at"; cells below "at" never change, so
  sharing it with the owner is safe [I from section A]. bytes(at) = 58 MB + 0 new KV; report it through the vtable's
  "charged" hook (core/prompt_cache.zig:33-34) so the budget reflects real use. Today the same save costs the full KV copy.
- restore into another stream: lease = shared full pages (retain) + ONE private copy of the boundary page (<= 3.94 MB D2D)
  + state slab copy. KV for the shared prefix is zero-copy. Resume by the same owner at its own end (next chat turn) is a
  truncate: no copy at all.
- Tiering: Snap parts follow section C. An idle Snap's state blob goes to a RAM frame; its pages lose their VRAM copy
  once mirrored and unreferenced by any resident stream (LRU by last use; the prompt cache's own eviction order decides
  which Snap dies, core/prompt_cache.zig:511-512). A hit on a RAM-tier Snap is a restore (C.5) instead of a miss.
- One mechanism: "parked stream" = Snap whose owner request is waiting; "kept reply end" (warm next turn) = Snap with
  no owner. The scheduler sees one resident-set problem.
- Later option: adopt core/prompt_radix.zig (longest-match over token pages, Pages vtable core/prompt_cache.zig:63-78,
  Rules.page core/prompt_rules.zig:23-24) for cross-request sharing without marks. The review notes it is not yet used
  for Flash-Next upstream; not needed for the first stages.
- MTP: its KV pages are part of the page list (same page ids, layer 13), so a prefix hit also restores the drafter
  context; lookahead 1 stays as is.

## E. Slots, admission, RAM tier

Per-card budget [I]: free after weights + MTP 9.45 GB [M J] - activation scratch 0.81 GB (st:160-230 at chunk_cap 2048,
max_len 200K; moe_down alone is 210 MB) - transport buffers 0.1 GB - margin 0.75 GB = ~7.8 GB for streams.
TF's fixed 2 GiB margin + 2 GiB default prompt cache (nfn:150-151) must become configurable or 200K x 4 cannot be admitted.
Resident stream = KV pages + state slab 58.0 MB + rollback set 66.8 MB (second GDN buffer, saved_proj, saved rows).

| Context | KV pages per card | resident stream | fit per card | with RAM tier |
|---|---:|---:|---:|---|
| 16K | 126 MB | 251 MB | 31 (engine cap 32, nfn:154) | all resident up to C16+ |
| 64K | 504 MB | 629 MB | 12 | C8 resident; C16 = 12 resident + 4 parked |
| 200K | 1.54 GB | 1.67 GB | 4 (6.66 of 7.8 GB; ~1.1 GB left for prefix Snaps) | C2/C4 resident; C8/C16 = 4 resident, rest parked |
These use worst-case full contexts. Admission should lease by need, as TF's ledger does (prompt + max_tokens + one
window, core/kv_pool.zig:53-56, core/kv_ledger.zig:21): real C8/C16 mixes are mostly short streams and will be resident.

Policy:
- Resident set = streams in the current rounds (decoding or filling). Idle kept states (between turns) are Snaps and
  go to RAM first, in LRU order, whenever a lease cannot be met; they cost nothing to keep in RAM.
- If active demand exceeds VRAM (C8/C16 at 200K): time-slice. Quantum 2 s of decode per resident stream; a swap is ~14 ms
  out + ~123 ms in, both asynchronous (the incoming stream restores while the outgoing one still decodes if pages are
  free; otherwise park first). Overhead ~7% at a 2 s quantum; each of 16 streams gets ~25% duty [I]. Never park a stream
  with an open window; never park the stream with the least remaining tokens (it frees its lease soonest).
- Limits to lift for C8/C16: verifyMulti packs at most 8 streams and 16 rows per forward (en:149-150, op:380, ln:417-428),
  so C16 already needs two full forwards per round. Raising the shared window to 32-64 rows is a Lead A item (kernel row
  buckets); the slot design does not block on it.
- Stream create/free become pool operations: no device allocation, no zero fill, no syncAll (compare st:287-331).

## F. Scheduler: mixing prefill chunks with decode rows

### F.1 What TF does today [R]
- One engine thread: admit, then fill.run (prompt steps), then one decode round (core/lane_host.zig:521-585).
- A long prompt advances one prefill_step per visit (core/lane_host_fill.zig:176-179; lp:38-73). The pick order is
  foreground, then least remaining rows, with aging after 8 skipped choices (core/lane_host_fill.zig:8, 22-42).
- Chunk rows: solo (2,048) only while NO other stream decodes, else busy rows (core/lanes/solo_chunk.zig:34-37 via
  lp:48-50); Flash-Next XPU busy rows default to 64 (XPU_CHUNK, nfn:117; sto:36-37). A decode-share debt can delay long
  steps (Share, core/lane_host_fill.zig:64-143) and an fcfs policy with a row hint exists
  (TENSORFOLD_PREFILL_POLICY/TOKENS, core/lane_host_fill.zig:182-199), but the hint can only lower the rows (lp:51).
- A Rounds planner that runs prompt pieces and decode rows in one forward exists in core (core/lane_rounds.zig:17-55;
  backend begin/piece/finish/pieces, core/lanes/backend.zig:148-155); the recipe describes that behavior for --parallel N
  (docs/recipes/qwen3.8-flash-next.md:427-433; which native family installs the planner was not traced here). The XPU
  backend does not implement pieces (ln:120-135) [R].

### F.2 What that means for 4 cold 20K prompts [I]
Let R = prefill rate with 2,048-row chunks, T1 = 20,000 / R.
- TF as is: prompt 1 fills solo in T1; as soon as it decodes, prompts 2-4 drop to 64-row passes. A 64-row pass still
  touches ~365 of 512 experts per layer [I: 640 draws] plus all dense weights, so its cost is close to a full weight
  sweep per 64 rows. Upstream reports a decode gap p95 of ~200 ms during prefill on their 4-card run (review 3.2), i.e.
  ~64 rows per 0.2 s = ~320 tok/s: 62 s per 20K prompt. TTFT ~ T1, T1+62, T1+125, T1+187 s: worse than Strata's
  17/39/61/84 s [M walk 2a]. This is the first thing to change.
- Bound: 80K prompt tokens need 80,000 / R seconds under any work-conserving order; for equal prompts the mean is
  minimized by FCFS. The scheduler removes waste; only R (Lead A, M-A5) or a shared prefix (section D) beats the bound.

| Policy | R = 1,200 | R = 2,000 (goal) | R = 4,000 | decoders during prefill |
|---|---|---|---|---|
| Strata measured | 17/39/61/84 s | | | ~5 s of decode per 12K chunk |
| FCFS, solo chunks, no share | 17/33/50/67 | 10/20/30/40 | 5/10/15/20 | stalled up to 1 s per chunk |
| proposed: FCFS, 1,024-row busy chunks, piggyback rows, share s = 0.15 | 17/36/56/76 | 10/22/34/45 | 5/11/17/23 | one window per chunk (<= 0.5 s gap at 2K) + 15% time |
Formula: TTFT_k = T1 x (1 + (k - 1) / (1 - s)). "Cold C4 TTFT << Strata" therefore needs R >= 2K for a 2x better tail and
R >= 4K for 4x; the scheduler's job is to not lose any of it and to keep decoders alive.

### F.3 Changes, in order
1. Config-only arm (no code): XPU_CHUNK=1024, XPU_CHUNK_SOLO=2048, TENSORFOLD_PREFILL_POLICY=fcfs,
   TENSORFOLD_PREFILL_TOKENS=1024, TENSORFOLD_PREFILL_SHARE=0.85 (prompt share p; decode share 0.15). Measure TTFT
   distribution, decode gap p95/p99 and prefill tok/s for chunk in {256, 512, 1024, 2048}. Note the XPU server does not
   parse --decode-share (review 3.2); the env path works (core/lane_host_fill.zig:93-99) [R].
2. Short-prompt bypass stays (<= 512 rows go whole and first, core/lane_host_fill.zig:165-175, 296-299).
3. Mixed pass ("piggyback"): generalize verifyMulti (op:367-528) to segments {seq, rows, inplace}: prompt segments are
   in place and up to chunk rows, decode segments are verify windows (not in place). The per-stream loops already exist
   for GDN, attention and PLE (op:272-295, 311-363, 441-455); HC, MoE and head are already batched over total rows. Expose
   it as backend "pieces" + keep the default loop, or attach the core Rounds planner. Decode streams then get one
   window per prefill chunk at < 1% extra rows instead of waiting for a decode-only round.
   Bitwise risk: a decode row inside a > 16-row pass runs the wide kernel variants (op:47-53, 78-92, 204-220) instead of
   the <= 16-row ones. Upstream declares prompts chunk-invariant (nfn:17), which implies wide == narrow bits for their
   Q4 kernels; Lead A's GGUF kernels must keep that property. Gate: mixed == solo bitwise, per kernel and end to end.
4. Admission: prefix wait already exists (core/lane_host_prefix.zig:1): a request sharing a cut that another prompt is
   filling waits for the Snap instead of recomputing. With section D the waiters then resume at ~5 ms each.
5. Fairness knob: decode share as a debt is kept; default 0.15, raised automatically when any decoder's gap exceeds
   a bound (e.g. 0.75 s). No starvation: aging already bounds every fill's wait (core/lane_host_fill.zig:10).

## G. Staged implementation plan (bounded worker tasks, priority order)

27B-first means: built and gated on the single-card Qwen3.8-27B XPU path that runs today (zig/src/xpu/, XPU_KV_INT8),
then ported. GPU needs are per run, through bin/gpu-run with health checks; "card 1" = single-card lease.

| ID | Task | Files (new *) | Acceptance | GPU |
|---|---|---|---|---|
| B3-1 | L0 park substrate, model-free: copy list on ordinal 1, pinned slab pool, batch copy + completion poll, probes for max host alloc / total pinned, D2H and H2D rate by size on BOTH cards, hop p99 under a concurrent 1 GB H2D | zig/src/xpu/host_tier.zig*, park_list.zig*, latbench mode | byte-identity round trip 1 MB-2 GB; no hang over 10k batches; numbers appended to TRANSPORT-B; health pre/post | card 1 15 min, then both 15 min |
| B3-2 | 27B-first park/restore of a slot to RAM (int8 KV mode): slot_states.zig saveSlot/restoreSlot gain a host tier | zig/src/xpu/slot_states.zig, runner_multi.zig, host_tier.zig | gates C.6-1/2/3 on the 27B (bit-identical continue, other slot == solo, tracer p99 within 5%) | card 1 20 min |
| B3-3 | Page pool + int8 cell + ikc_tail for Flash-Next (host logic + kernels), bf16-cell flag kept | xpu_pages.zig*, st.zig, op.zig, fw.zig (forwardMtp), mx (5 kernels), core kv_pool reuse | B.5-1 kernel parity; host unit tests of leases/refcounts/PT; seam + chunk-invariance tests bitwise in both cell modes | card 0 or 1, 15 min (after Lead A's M-A3 merge for model-level) |
| B3-4 | int8 KV model validation | xpu-fn-teacher driver arm, results under results/tf-flashnext/B/kv8/ | B.5-2/3/4 gates; journal entry with NLL, top-1, KL, SEM | both cards 20 min |
| B3-5 | Flash-Next park/restore + write-behind mirror (Snap object, state slab, page states) | st.zig (Snap), xpu_park.zig*, en.zig, ln.zig | gates C.6-1..4 on two cards; park <= 20 ms, 200K restore <= 150 ms, 20K <= 25 ms (unprofiled confirm) | both cards 30 min |
| B3-6 | Slots and admission on the pool: ledger by need, configurable margin/cache, resident-set policy, time-slice | nfn.zig, xpu_admit.zig, xpu_resident.zig*, core kv_ledger reuse | C4 x 200K admitted and decoding; C8/C16 x 16K-64K concurrent == solo; C8 x 200K completes with parks, zero recompute; host tests of the policy | both cards 30-40 min |
| B3-7 | Scheduler config arm (F.3-1) | none (env), bench scripts | TTFT table for 4 x 20K cold vs chunk size; decode gap p95; pick defaults | both cards 30 min |
| B3-8 | Mixed passes (F.3-3) | op.zig (forwardSegments), ln.zig/lp.zig (pieces), core planner hookup | mixed == solo bitwise; TTFT and gap vs B3-7 | both cards 30 min |
| B3-9 | Decode items: GDN A/B pointer swap (kills the 56.6 MB copy per stream per window), KV-head-major attention if the tracer shows the re-read cost | fw.zig keepRows, op.zig, mx | drafted == serial, concurrent == solo bitwise; window ms before/after (tracer + unprofiled) | both cards 20 min |
| B3-10 | MTP: drafter KV as a ring or page-shared (saves 592 B/token/card), drafter state in Snap verified | st.zig, fw.zig forwardMtp, en.zig | drafted == serial; acceptance rate unchanged on fixed prompts | both cards 20 min |
| B3-11 | Prefix cache on shared pages (section D) + RAM-tier hits | st.zig snap*, ln.zig snapshots vtable, charged hook | resume == full pass bitwise at marks; save cost 58 MB + 0 KV; RAM-tier hit resumes without recompute | both cards 20 min |
Order rationale: B3-1/2 can start now on card 1 without Lead A; B3-3 CPU work can start now, its GPU gates wait for
M-A3. B3-5 is the user's "eviction to RAM restored without recompute" milestone; after it come concurrency (B3-6..8),
decode (B3-9), MTP (B3-10). Prefix sharing (B3-11) is last because TF's existing full-copy snapshots keep working on
the page pool in the meantime (just more expensive). Each task is one worker, one branch off b70/fn-runtime, reviewed
and merged by Lead B.

## H. Unknowns and risks (ranked)

1. Fit margin [I]: 4 x 200K uses 6.66 of ~7.8 GB per card on paper. It depends on the measured 9.97 GB free (one loader
   run), on dropping ikc, and on shrinking TF's 2 GiB margin. Level Zero allocation granularity and fragmentation are
   not counted. If it misses, the MTP ring (B3-10) and a smaller solo chunk (moe_down scratch is 210 MB) recover ~0.4 GB.
2. Pinned host memory limits and D2H behavior are unmeasured beyond 64 MB on card 1; card 0 is unmeasured. B3-1 exists to
   remove this unknown before any design depends on it.
3. Copy-engine traffic vs TP2 hops on the same PCIe link: unmeasured. If hop p99 suffers, throttle restores to idle
   gaps between hops or to a duty cycle; park (D2H, write-behind) can always be delayed.
4. Driver hazards seen on this host: BCS resets on host-only copies and on peer PULL. The design uses neither, but the
   copy engine is the least-exercised engine here; soak it (B3-1) and keep a compute-engine fallback for the same copies
   (latbench shows the same rates on both engines [M PLAT]).
5. int8 accuracy on the IQ3 pack at long context is unmeasured in TF; Strata ships int8 KV on this pack daily, which is
   strong but indirect evidence. Rotation may hurt (Strata's IQ2_XS note). B3-4 decides.
6. Mixed passes need wide == narrow kernel bits from Lead A's new GGUF kernels; if not available, fall back to
   alternating rounds with 1,024-row chunks (B3-7), which needs no kernel property.
7. Recorded command lists (DESIGN-A v1.2 M-A5) will bake buffer addresses: the page pool, PT and state slabs must be
   device-resident arguments read at run time, not baked pointers. Agree the descriptor layout with Lead A before B3-3
   lands (PT base, state slab base, A/B selector per stream in the window descriptor).
8. All latency numbers in sections C and F are arithmetic from PROVISIONAL constants; every acceptance line above
   replaces one with a measurement. No speed or stability claim is made by this document.
