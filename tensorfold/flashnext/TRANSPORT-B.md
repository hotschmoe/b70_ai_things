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

## 5. Still being measured
- counter-based events and reuse; EU peer stores vs copy-engine push; skeleton window (Lead A request); D2H/H2D on both cards
  and engines (KV eviction); push all-reduce behind TpComm (codex-pushar, M-B2b).
