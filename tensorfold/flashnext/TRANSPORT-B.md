# Cross-card transport and latency constants (Lead B, for Lead A) - living document

Status 2026-11-20 08:15 UTC: section 3 has first measured numbers (provisional). Section 1 is PRIOR MEASURED evidence from this exact box (other
harnesses: SYCL/L0 scripts, docs/P2P_GPU.md). Section 2 is Lead B's reading of TensorFold's tp.zig (inferred, not
measured). Section 3 lists what codex-lat is measuring now with TF's own Zig/Level Zero (tf-xpu-latbench); numbers
will replace "pending" as they land. Raw data: /mnt/vm_8tb/b70/results/tf-flashnext/B/latbench/.

## 1. Prior measured facts on this host (kernel 7.0/7.1, compute-runtime 26.22, Gen3 x16 each, cards on different 1950X dies)
| Quantity | Value | Source |
|---|---:|---|
| P2P PUSH (peer write, executed on the SOURCE card), copy engine or EU | 11.2 GB/s | P2P_GPU.md J.2, K.10 |
| P2P PULL (peer read, executed on the DESTINATION card) | 3.24 GB/s | J.2, K.10 |
| Host bounce d0 -> host -> d1 | 3.5 GB/s | K.10 |
| 8-byte peer ping, raw L0 | 8.6-9.3 us | J.2, K.10 |
| Cross-device command-streamer event wait (push copy + signal + peer wait + read), 10 KB / 64 KB / 1 MB | 21.5 / 16.8 / 110 us | K.3 |
| Two-rank push all-reduce, single context, 10 KB (4 launches + syncs) | 50.8 us | K.10 |
| 2-process L0-IPC push, 10 KB one way | 13.1 us | K.10 |
- A peer write issued from INSIDE a running kernel is NOT visible to a kernel spinning on the other card (EU spin flag
  HANGS; no peer atomics on this fabric). Cross-card ordering must be a kernel boundary + event, not an in-kernel flag. (J.9)
- A command-streamer wait (zeCommandListAppendWaitOnEvents) on an event from a HOST_VISIBLE pool created for BOTH
  devices is correct and replayable over 200 closed-list replays. A pool created on one device only segfaulted the peer. (K.3)
- Strata's EP device flags (ep_wait, system-scope loads) do work but burn an EU thread; the analyst estimate is ~40 ms
  of card-1 idle per C4 window hidden there (opus_zig_engine_design.md 5.3) [inferred].

Design implications [inferred]: make every per-layer exchange PUSH-shaped (producer card writes into the consumer's
memory, then signals); never pull; order with CS event waits, not in-kernel flags. Decode exchanges are latency-bound
(4-64 KB at 11 GB/s is 0.4-6 us of wire; the hop costs ~10-20 us), so the design should (a) minimise dependent
cross-card hops per window and (b) give each card independent work to run while a hop is in flight (EP with the
shared expert / dense work overlapping the return, as Strata's SH_LATE does).

## 2. TensorFold tp.zig as shipped in PR #3603 (read, not measured)
- One-shot all-reduce: every rank's kernel READS all ranks' send buffers over PCIe = PULL-shaped (3.24 GB/s class here).
- Cross-card events: ze.Gpu.sharedEvents, pool HOST_VISIBLE for both devices, but event signal/wait scope = 1
  (ZE_EVENT_SCOPE_FLAG_SUBDEVICE), and syncAll resets them with zeEventHostReset.
- Known runtime hazard (TF PR #3586 comment, compute-runtime 26.05; we run 26.22): two devices + timestamp pools +
  host-reset events on in-order immediate lists that wait on each other hang; workaround
  NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0, fixed in 26.40.
- On our host tf-xpu-tp-bench hung at the first collective with a GuC job timeout + GT reset on card 1 (probe
  journal f). Candidates: the host-reset/counter-based-event hazard, SUBDEVICE scope, the pull pattern. One bounded
  reproducer per candidate is queued (codex-lat step 7). Until then: do not build on tp.zig's event setup.

## 3. Measured with tf-xpu-latbench (TF Zig + Level Zero, worker codex-lat; PROVISIONAL until Lead B review)
Source: /mnt/vm_8tb/b70/results/tf-flashnext/B/latbench/PLATFORM_DRAFT.md (+ JSON/logs), commits c20ab2d / 27725b7 / aeac63f on
b70/fn-rt-lat. Single-card numbers on card 1, 256 reps; pair numbers 2000 reps; p50/p99 in us; health pre/post passed.
Clock mapping: driver host timestamp == CLOCK_MONOTONIC_RAW (768/768 samples), mapping bound p99 < 0.2 us on both cards.

Single card:
| Constant | p50/p99 us |
|---|---:|
| host append, empty kernel, in-order immediate list | 2.67 / 11.6 |
| append + host synchronize round trip | 13.9 / 22.5 |
| idle device: append -> kernel start | 8.1 / 16.5 |
| dependent-kernel device gap (immediate list; same for regular list N=16..1024) | 0.94 / 1.1 (regular) |
| regular list execute host cost, N=1 / 128 / 1024 kernels | 2.05 / 2.27 / 3.27 |
| compute -> event -> compute on another list (same card) | 2.29 / 6.25 |
| compute -> copy engine -> compute (events) | 5.21 / 13.2 |
| host wake-up: zeEventHostSynchronize / busy poll | 3.23 / 4.29 |
| record/replay graph extension | NOT advertised by this driver |
| H2D 64 MB / D2H 64 MB (pinned USM, card 1) | 5.1 ms (13 GB/s) / 14.9 ms (4.5 GB/s!) |
| H2D / D2H 4 KB append+sync | 7.3-8.5 / 6.6-6.8 |

Implications [inferred]: (1) the host append (2.7 us) costs ~3x the device's dependent-kernel gap (0.94 us): a window of
~1000 small kernels appended one by one is HOST-bound unless it is pre-recorded (regular command lists: 2-3 us to execute
1024 kernels). (2) Every host round trip inside a window costs ~14 us + ~8 us to restart an idle device. (3) D2H is ~3x
slower than H2D on card 1 (NUMA/host-memory placement suspected; matters for KV eviction to RAM; to be checked).

Two cards, PUSH (source card's copy engine writes peer memory) + fresh host-scope event, producer kernel end ->
consumer kernel end, data verified:
| Bytes | one-way p50/p99 | round trip p50/p99 |
|---:|---:|---:|
| 4 KB | 12.7 / 28.9 | 18.6 / 42.0 |
| 16 KB | 12.2 / 18.4 | 19.0 / 27.4 |
| 64 KB | 17.3 / 22.2 | 27.9 / 31.9 |
| 256 KB | 34.8 / 37.9 | 63.8 / 65.8 |
| 1 MB | 106.7 / 110.3 | 207.3 / 212.0 |
Against DESIGN-A thresholds: decode-sized hops (10-64 KB) are 12-17 us p50 one-way: inside the <= 25 us TP2 band at p50,
p99 18-29 us. 160 KB interpolates to ~25-28 us. Pending: event scope/pool/counter-based variants, EU peer stores, host
staging, flag-spin, skeleton window (exposed cost with overlap), tp-bench hang verdict.

## 4. Hazards found so far (measured, 08:50 UTC)
- Copy-engine PULL (card 1's BCS reading card 0's memory) HUNG on the first attempt: Level Zero NOT_READY on the bounded wait,
  kernel log "BCS engine memory CAT error", engine reset, timed-out job on card 1 (B/latbench/pull-hang-*.log). Recovered with
  xe-reset (rebind) + health. Do not use copy-engine peer reads. PR3603's "card 0 pulls card 1's shard logits" is the same class:
  replace with push (Lead A already plans this).
- An EU-kernel one-shot pattern like tp.zig (shared host-visible pool, SUBDEVICE scope, kernels READ both send buffers) with FRESH
  events (no host reset) COMPLETED and verified once. So the stock tp-bench hang is not explained by scope or EU peer reads alone;
  the host-reset/reuse hazard is being tested now (repro-reuse), then a patched tp-bench, one bounded attempt each.
- Host USM flag wake-up (kernel writes a host flag, host spins): 1.2 us p50 after kernel end, vs 3.2 us for zeEventHostSynchronize.
- Host staging variants (D2H, flag, H2D) completed through 1 MiB with verification; timings diagnostic (see NOTES.md).

## 4b. Update 09:25 UTC (measured, B/latbench/PLATFORM_DRAFT.md, B/pushar/NOTES.md)
- tp.zig-style host-reset event REUSE with peer reads: DEVICE_LOST on both GPUs with default driver settings (compute-engine CAT
  errors + resets, 08:49); recovered by xe-reset. Same reproducer PASSED with NEOReadDebugKeys=1
  EnableImplicitConvertionToCounterBasedEvents=0 (4 iterations). Rule for now: fresh events (no zeEventHostReset on cross-card
  events) AND the workaround env in every two-card process. Stock tp.zig is unsafe as shipped on this driver.
- Compute-engine PUSH ~ copy-engine PUSH (4 KB 12.6 vs 12.7 us; 64 KB 15.8 vs 17.3; 1 MB 111.5 vs 106.7). Cross-card event
  signal->wait alone: 8.2 us (compute) / 9.7 us (copy) p50.
- Host staging (D2H + H2D) is slower than PUSH at every size (4 KB 20.7 us, 64 KB 36.3 us).
- Push all-reduce candidate (codex-pushar): 16/16 sizes 10 KB-20 MB correct; 20,000 chained 10 KB hops passed; 100-hop window
  3.5 ms p50 including event retirement (2.5 ms to list drain) = ~25-35 us per all-reduce, host append ~26 us per hop (!).
  BUT chained 160 KB hops with reused inbox slots returned STALE first elements (2 of 2 variants failed). Not stable yet; the
  coherence question (cached stale inbox lines vs ordering) is being isolated (codex-lat B, codex-pushar). Lead A: do not build
  on the pushed AR above 10 KB until this is closed.

## 4c. Skeleton window + chained hops (10:05 UTC, measured, B/latbench/NOTES.md + PLATFORM_DRAFT.md, codex-lat 0696ede)
- Skeleton (Lead A request): 48 layers, real per-card byte counts from a 3 GiB rotating buffer, 2 dependent exchanges per layer
  (96 per window), rank-ordered sums, 50 windows, every element checked, health per variant:
  | R | no-exchange window p50/p99 ms | with fresh-slot push exchanges p50/p99 ms | exposed per exchange (p50 diff / 96) |
  |---|---:|---:|---:|
  | 4 | 32.79 / 32.85 | 33.69 / 42.26 | 9.4 us |
  | 8 | 32.82 / 32.84 | 34.03 / 43.48 | 12.6 us |
  -> TP2 region on Lead A's rule (<= 35 us). Tails: p99 windows +9-10 ms; individual hop p99 ~48-55 us. The tail, not the p50,
  is the TP2 risk; record which hops spike in M-A5 with the tracer.
  (The synthetic read kernels reach only ~13.4 ms of bandwidth floor vs 32.8 ms measured; that is the skeleton kernel, not a
  platform limit - ignore the absolute window time.)
- Chained hops, full element check, 4800 hops each: FRESH inbox per hop: 10 KB p50/p99 8.9/49.0 us, 160 KB 20.7/48.0 us,
  zero mismatches. REUSED inbox at 160 KB: STALE DATA (26,240 bad checks); uncached-only reuse still stale (71,136);
  uncached + zeCommandListAppendMemoryRangesBarrier on the inbox: 4800/4800 clean (10.8 / 22.0 us p50).
- Flag spin (Strata-style), peer-memory flag or host-USM flag, written by a copy after the data kernel: bounded spins timed out
  in thousands of work-groups and read stale data, both sizes. Do not use in-kernel spin flags across cards.
- RULE for every cross-card exchange (Lead A + push AR): push from the source card; fresh event per exchange (no host reset);
  the RECEIVER must order its read behind the copy-completion event with a barrier (receiver-side barrier on the inbox), or
  use a fresh inbox region per hop within a window. Inbox reuse without a receiver barrier returns stale lines. The
  codex-pushar candidate e766a7a implements the receiver barrier: 21,000 chained 160 KB hops + 16/16 sizes clean.

## 4d. DELIVERED for Lead A (10:40 UTC): push all-reduce behind TpComm, b70/fn-runtime 611566d (= b70/fn-rt-pushar 31b3dae)
- zig/src/xpu/tp.zig: for tp == 2, allReduceF32 / allReduceF16 / syncAll keep their signatures and now push: source-card
  compute-list copy of the send slot into the PEER's local inbox (signal on the copy itself), receiver barrier waiting the
  peer's copy event before the reduction kernel, both operands local, rank-ordered sum starting at +0.0 (bit-identical on
  both cards), barrier after, and before the next producer a wait on the peer's reduction of the slot being reused
  (RING_SLOTS ring). Events: fresh host-scope pool, destroyed and recreated in syncAll (no zeEventHostReset). Host syncs
  bounded (10 s). TP4 unchanged (old path). New: peerInbox(r) + allReduceF32Pushed(...) for producers that write the peer
  inbox directly (fused producer; saves one copy).
- Verified (measured, B/pushar/RESULT.md): full-array f16/f32 vs CPU reference incl. -0.0, 10 KB..20 MB; 20,200 chained 10 KB
  hops single process; 22,400 chained 160 KB hops over 7 processes; health pre/post.
- Cost (measured, synthetic producer, 10 KB): 100-hop window 3.58/4.11 ms p50/p99 incl. event retirement (~1 ms of it =
  pool recreate); host append ~25.7 us per all-reduce (8 L0 calls per rank). Isolated append-to-drain 79/124 us.
- Caveats for M-A5 (record/replay): fresh events are incompatible with replaying a recorded list as-is (baked event handles
  need a reset or a counter-based event). Counter-based host-scope events work one-way (12.1 us at 4 KB, same as regular);
  reuse in a replayed regular list across both cards is NOT yet tested - Lead B's next transport item. Until then the
  immediate-list path is the qualified one.

## 4e. DELIVERED (13:25 UTC): recorded-list TP2 all-reduce for M-A5, b70/fn-runtime (merge of b70/fn-rt-pushar c9c9efe)
- API (zig/src/xpu/tp.zig): rec = comm.beginRecording(gpus, hops, .device_reset); per position: append the producer writing
  rec.currentSendBuf(rank) on both in-order REGULAR lists, then rec.recordAllReduceF32(gpus, lists, dst, count); then
  rec.finish(gpus, lists); close lists; execute on one async ordinal-0 queue per card, same window order on both cards.
  Everything referenced (comm, rec, lists, dst, counts, buffers) fixed for all replays; producers may read changing device data
  (e.g. the v1.3 window descriptor). Single-flight: re-execute a list only after both queues drained its previous execution;
  two DISTINCT recordings (own event sets) may alternate (double submission verified). Do not share the comm's slots with
  other work. F32 only so far.
- Mechanism: fixed events per position (push-complete + reduce-complete per rank); consumer waits with a barrier then resets
  the event ON DEVICE (zeCommandListAppendEventReset); slot reuse waits on the peer's reduce-complete; finish consumes the tail.
  No host event reset anywhere; no workaround env needed.
- Measured (10,000 verified replays each, health pre/post, incl. per-hop verify kernels): 10 KB x 96 hops 1.416/1.432 ms device
  window p50/p99, host submit 5.6/8.0 us per window; 10 KB x 192: 2.823/2.844 ms; 160 KB x 96: 2.896/2.921 ms; 160 KB x 192:
  5.778/5.830 ms; double submission 160 KB x 192: p99 5.831 ms. => ~14.7 us per 10 KB hop incl. verify, flat p99 (tail gone vs
  immediate path's ~50 us p99), and host cost per window ~6 us instead of ~2.5 ms.
- Counter-based events: FAILED (non-immediate: wrong data at replay 67; immediate: 0x78000004 at setup). Do not use.

## 5. Still being measured
- counter-based events and reuse; EU peer stores vs copy-engine push; skeleton window (Lead A request); D2H/H2D on both cards
  and engines (KV eviction); push all-reduce behind TpComm (codex-pushar, M-B2b).


## 4f. Host memory for the PLE table (Lead A request) [B 15:40]

CONFIG: fn-runtime probe b60f30b (tools/ple_host_probe), card 1, host RLIMIT_MEMLOCK 8 MiB; container vllm-xpu-env:int8g-v0251
with --ulimit memlock=-1:-1, --network none. Full record: /mnt/vm_8tb/b70/results/tf-flashnext/B/pushar/PLE_HOST.md.
COMMAND: ple-host/run.sh imports | container | buffered | direct (card-1 gpu-run, health pre/post, all passed).
RESULT (measured):
- zexDriverImportExternalPointer: anonymous MAP_PRIVATE populated 4 MiB and 64 MiB -> 0, GPU kernel reads every page correctly
  (64 MiB is 8x the memlock limit). Read-only MAP_PRIVATE FILE mapping (4 MiB) -> 0x70000002, on host AND with unlimited memlock.
- Fill one 4 GiB zeMemAllocHost from the shard: buffered pread + POSIX_FADV_DONTNEED behind the cursor 1.86 GB/s (slice page
  cache dropped to 0 pages); O_DIRECT into the 4 KiB-aligned USM buffer 3.85 GB/s (accepted by the filesystem; leaves a
  pre-existing cached copy in place). Both warm-cache samples, not a controlled ranking.
VERDICT: memlock is NOT the cause of the file-import failure; file-backed pages cannot be imported on this stack (FOLL_LONGTERM
is Lead B's inference, not traced). Recommendation for Lead A: allocate the PLE table once with zeMemAllocHost (28.8 GB proven by
worker P, both cards read it) and fill it with O_DIRECT reads (~7.5 s for 28.8 GB at the measured rate), then
posix_fadvise(DONTNEED) on the table's file range to shed any old cached copy. Page cache is reclaimable, so the only hard,
unswappable cost is the one 28.8 GB USM allocation; with the 47 GiB park ceiling that is ~76 GB of 121 GB. Untested
alternative: pread into anonymous memory and import it (avoids USM allocator limits; full-size and two-card import untested).

Note for Lead A (Lead B, 15:45): core/checkpoint_file_cache.zig release() is macOS-only (msync INVALIDATE); on Linux it is a no-op,
so after an O_DIRECT/pread PLE load add a posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) branch there to shed the shard's cache.

## 4g. [B 23:55] Review of the R2/R5 recorded-window hang (orchestrator ABA hypothesis)
Code read: fn-rec 7660564 xpu_window_record.zig + tp.zig TpRecording + ze.zig events.
- Stale-signal ABA across replays: EXCLUDED BY CONSTRUCTION, not observed. Every recorded event has one signaller and one
  consumer per replay; the consumer waits then resets ON DEVICE in the same in-order list (push-complete at the hop;
  reduce-complete at slot reuse pos+3 or in finish() for the last 3 hops; head events 0-5 likewise). The unit test
  "consumed exactly once" covers the AR set. Events use HOST signal/wait scope (EventDesc default 1<<2), so the reset is flushed
  before the queue fence, and execute() synchronizes BOTH queues before returning (single flight). So no event can still be
  signalled when the next replay starts. The only ABA form left is "a reset lands after the next replay's signal", which
  would need the fence to complete before an earlier same-engine store landed.
- Bucket or AR-count mismatch between ranks: EXCLUDED on the recorded path. execute(b) submits e.lists[0] and e.lists[1] from
  ONE entry, both ranks are appended in one loop per hop, and finish() fails unless exactly 96 hops were recorded. Expert-AR
  presence is baked in at record time, identical on both ranks.
- The signature implies card 1 passed hop p: its copy plus post-sync signal of event(p,0,1) retired before its barrier. Card 0
  never saw it. That is a LOST or ERASED signal, or a stale poll, not an early-consumed one. Cheap tests to tell which, in order:
  1. Abort-time dump (no shape change): host zeEventQueryStatus of every event in the hung entry (hop, kind, rank), plus
     each rank's last hop from the diagnostic kernel events. If event(p,0,1) reads SIGNALED -> card 0's wait missed a store that
     is in memory (poll visibility). If NOT_READY while card 1's reduce-complete(p) is SIGNALED -> the signal was lost or erased.
  2. Diagnostic-mode post-drain assert in execute(): after both queues sync, all 390 events NOT_READY (~0.1 ms, diag only). Any
     SIGNALED = leftover = ABA proven.
  3. Hardening that also tests ABA empirically: two entries per bucket with their own event sets, alternating by replay
     parity. 4e verified that distinct recordings can alternate. Cost: 2x recorded-list memory and build time. If C4 no-MTP
     still hangs, ABA is ruled out.
  4. Real hole found: execute() does not drain the IMMEDIATE lists (r.lists) before submitting on the recorded queues. The
     r.comm.step==0 check proves only that the immediate AR comm was synced. Other immediate work (staggered C4 admission
     prefill, page or state copies) can still be running on the same ordinal-0 engine on another queue. Add
     zeCommandListHostSynchronize(r.lists[rk]) at execute() entry. It is ~us when idle and also closes a data race.
- Not recommended: (a) epoch flags. L0/zex wait-on-memory takes an immediate value baked into the list, so a per-replay epoch
  needs a device spin kernel, which breaks the transport rule (no spins). (c) Per-hop sequence numbers in the payload also need
  consumer polling. (b) is the cheap form of the same protection.
- Why it fails only in no-MTP C4: back-to-back replays with no immediate MTP work between them, plus staggered admission
  prefill on the immediate lists. Both point at test 4 first, then 1+2.
