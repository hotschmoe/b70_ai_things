# hc agent report - hyper-connection (HC) read in decode

Worktree: /mnt/vm_8tb/b70/build/night-wt/hc/src-ep (base e1b7ee8a)
Patch:    /mnt/vm_8tb/b70/results/night-20261016/agents/hc.patch
Engine:   /mnt/vm_8tb/b70/results/night-20261016/builds/strata-hc-v2d (byte-identical to strata-hc-v2c / v2b; forced full recompile of fused_gr)
Probe:    /mnt/vm_8tb/b70/results/night-20261016/agents/hc_work/hc_probe.cpp (+ build_probe.sh, run_probe.sh)

## Flags (default OFF; original paths unchanged when unset)

- `STRATA_NIGHT_HC=1` (or `bf16`): new two-kernel HC read "v2" on the BF16 weights.
- `STRATA_NIGHT_HC=q8` together with `STRATA_HC_Q8=1`: native_dense stores the HC Q8_0 copies in a split
  layout (int8 quants [rows][K], then fp16 scales [rows][K/32], same bytes) and the read uses them (real Q8 read).
  Without STRATA_HC_Q8=1 it falls back to the v2 BF16 read. Inject rows stay BF16 in both.
- One stderr line when enabled: `strata: STRATA_NIGHT_HC=...: two-kernel hyper-connection read (...)`.
- Applies to every fused_gr_read_multi caller (verifier layers, final mixer, MTP body). Rows must be at the
  callers' fixed strides (verify/mtp/final mixer are); otherwise it prints once and uses the default read.

## Key finding 1: STRATA_HC_Q8=1 was a no-op for the GPU read

CONFIG: base source, fused_gr.dp.cpp. RESULT: `fused_gr_read_multi` never reads `q8_down/q8_up` ("the Q8_0 reads
are not built"); STRATA_HC_Q8=1 only allocated 0.63 GiB of unused Q8_0 copies (and switched the final-mixer branch
to the same BF16 kernels). VERDICT: the ~2% C1 delta attributed to HC_Q8 was not HC bandwidth. Fixed by
STRATA_NIGHT_HC=q8 (below).

## E1 baseline (graph replay, card 1)

CONFIG: probe includes the worktree fused_gr.dp.cpp; random BF16 weights (N(0,0.02)/N(0,0.05)), 8 distinct
weight sets (105 MB > L2), 48 reads per graph, apply=true, random R/bo/inj. COMMAND:
`run_probe.sh hc_probe_v1`. RESULT (us per HC read; 13.19 MB BF16 per read):

| T | default read | norm | down (sliced) | reduce | up | GB/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 38.7 | 5.8 | 14.2 | 2.7 | 15.4 | 340 |
| 2 | 41.0 | 5.8 | 15.2 | 2.7 | 16.9 | 322 |
| 3 | 45.6 | 5.8 | 16.8 | 2.8 | 19.8 | 289 |
| 4 | 50.8 | 5.9 | 18.5 | 2.8 | 23.0 | 260 |
| 6 | 62.0 | 5.9 | 22.7 | 2.8 | 30.2 | 213 |

Matches the C2 trace (verifier T=6: 7.5 + 22.7 + 3.5 + 29.9 us; 97 reads = 6.1 ms/cycle).
VERDICT: down/up degrade ~1.7/3 us per extra row -> issue/latency bound, not DRAM bound.

## E2 existing knobs (default read, env)

RESULT at T=6: STRATA_GR_MORE_GROUPS=1 80.3 us; UP_COLS=8 81.0; UP_COLS=4 75.4; UP_ILP=1 96.5 (T=1: 42-47 vs 38.7).
VERDICT: all worse on B70 (each extra group recomputes the lo prologue). Keep defaults.

## E3 v1 (upstream S23/V3-style, 2 kernels, last-group reduce) - rejected

RESULT T=6: 80.8 us BF16 / 83.1 us Q8_0 (34-byte blocks). ISA (IGC_ShaderDumpEnable): the by-value GrMulti
argument is 1.6 KB of cross-thread data = 25 GRF loaded in every thread; the up kernel spilled 2-8 KB/thread.
VERDICT: dead end as written; led to the compact strided argument struct (`NA`, ~2 GRF) used by v2.

## E4 v2 design (fused_gr_night.inc)

- night_down2: one ROW per lane (16-lane sub-groups), activations uniform across the sub-group (broadcast SLM reads,
  no per-row cross-lane reduction). Work-group = 32 rows (2 per lane) x 1280-column slab (half a stream); each
  stages R' * w_norm (unscaled) for the T rows and that slab's sums of squares; the 16 sub-groups split K (80
  columns) and add partials in SLM in fixed order. The LAST work-group of a row group (atomic counter, no waiting,
  counter reset by that group -> graph-safe, no forward-progress assumption) sums the slab partials per stream,
  applies rs (w . xn = sum_c rs[c] (w[:,c] . R'[c] g[c])), writes lo = silu(./4), inject_out, rs.
  Norm + down + reduce = ONE kernel.
- night_up2: one row per lane, 16 columns x 4 streams per work-group, K split in 4 quarters of 80 (5 for Q8),
  lo uniform; epilogue (pending write R_out, gate, stream mean) identical ops to the default.
- Q8 variants (down2q: 4 rows/lane x 32 columns; up2q: 5 K-fifths, 320 threads): w = d*q exact in FP32.
- ISA check: all v2 kernels spill-free for T <= 7 (T=8 down spills a little; T=8 is not used by C1/C2).

## E5 correctness (probe, identical inputs; FP64 CPU reference)

Max abs / max rel (rel denominators floored at 1e-3 of the max |ref|):

| path | lo | rs | inject | mixed | R_out |
|---|---|---|---|---|---|
| default vs FP64 (T=6) | 5.3e-7 / 4.4e-5 | 5.7e-8 | 7.8e-7 | 1.6e-7 / 3.0e-5 | - |
| v2 bf16 vs FP64 (T=6) | 2.9e-7 / 6.5e-5 | 8.0e-8 | 1.0e-6 | 1.9e-7 / 4.7e-5 | bitwise default |
| v2 bf16 vs default (T=6) | 6.0e-7 / 7.1e-5 | 6.0e-8 | 1.1e-6 | 1.8e-7 / 4.2e-5 | 0 diffs |
| v2 q8 vs FP64 with Q8 weights (T=6) | 2.6e-7 / 3.5e-5 | 5.7e-8 | 8.9e-7 | 1.6e-7 / 3.0e-5 | bitwise default |
| v2 q8 vs FP64 with BF16 weights (T=6) | 6.8e-3 / 2.1 | 5.7e-8 | 8.9e-7 | 1.6e-3 / 0.38 | - |

Relative RMS error of v2 q8 vs the BF16-weight reference (the Q8_0 weight rounding): lo 5.6e-3, mixed 7.2e-4
(v2 bf16: lo 2.0e-7, mixed 7.8e-8). VERDICT: v2 bf16 = summation-order change only, FP32 rounding level (same size
as the default read's own error vs FP64; not bitwise). v2 q8 arithmetic is FP32-exact on its weights; its only
error is the Q8_0 quantization of the HC weights (synthetic Gaussian weights - quality must be judged on the model).
Determinism: repeated runs bitwise identical; stress (25 x 48 graph-replayed reads, T=1/4/6, both paths): 0 output
differences vs a single run (last-group counter + device-scope fences hold under replay).

## E6 timing (graph replay, card 1, same probe as E1)

COMMAND: `run_probe.sh hc_probe_v6 night`. RESULT, us per HC read:

| T | default | v2 bf16 | (down2 / up2) | v2 q8 | (down2q / up2q) |
|---|---:|---:|---|---:|---|
| 1 | 38.7 | 34.8 (-10%) | 17.7 / 16.1 | 24.7 (-36%) | 12.7 / 8.2 |
| 2 | 41.0 | 36.8 (-10%) | 18.7 / 17.2 | 26.2 (-36%) | 13.5 / 8.7 |
| 3 | 45.6 | 39.3 (-14%) | 20.0 / 18.3 | 28.3 (-38%) | 14.8 / 9.4 |
| 4 | 50.8 | 41.5 (-18%) | 20.9 / 19.7 | 29.3 (-42%) | 15.4 / 10.0 |
| 6 | 62.0 | 47.1 (-24%) | 23.4 / 22.8 | 34.4 (-45%) | 18.2 / 12.1 |

Down2 shape sweep at T=6 (rows/lane, columns/sub-group): (1,80) 30.6, (2,40) 26.9, (1,40) 30.2, (2,80) 23.6 us
-> (2,80) is the default. Kernels per read: 4 -> 2.

## Estimated savings (probe deltas x read counts; not yet measured end-to-end)

- C2 cycle (97 verifier reads at T=6 on card 0): v2 bf16 -1.45 ms; v2 q8 -2.68 ms (of 6.1 ms HC).
  MTP-body reads (BF16 MTP weights, T=1-3) another ~0.05 ms.
- C1 window (verify T=4: 97 reads; 3 draft steps x 3 MTP reads at T=1): v2 bf16 ~-0.94 ms; v2 q8 ~-2.1 ms
  (~5% of a ~43 ms window).

## Not done

- Router BF16 GEMV (bf16_gemv_mmvf_multi g{512}, 11.9 us at T=6 in the trace) + top-10 fusion: not started (time
  went to the HC read). The same row-per-lane / uniform-activation structure should apply to the 2560->512 GEMV.
- Engine-level serving validation (coherence, tok/s) is for the orchestrator: suggested A/B
  `STRATA_NIGHT_HC=1` and `STRATA_NIGHT_HC=q8 STRATA_HC_Q8=1` against the current config.

# Mission 2 (router / small BF16 GEMVs / GDN per-slot work)

Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-hc-gdn1 (HC v2 + GDN slots; base e1b7ee8a)
Patch (this mission only, applies on top of hc.patch): /mnt/vm_8tb/b70/results/night-20261016/agents/hc_gdn_slots.patch
Full worktree diff vs e1b7ee8a: /mnt/vm_8tb/b70/results/night-20261016/agents/hc_full.patch

## G1 STRATA_NIGHT_GDN_SLOTS=1 (verify recurrence of every slot group in one launch) - KEEP

CONFIG: verify batch window (batch_rec_), STRATA_GDN_SPLIT unset/0 (the config). The per-slot loop launched
gdn_step_norm_multi_kernel<true,false> once per slot group (48 blocks of 512 threads each); the night launcher runs
the same kernel body on grid (head, slot) with per-slot state / row offsets (select chain, no runtime-indexed arg).
COMMAND: hc_work/gdn_probe (36 layers of distinct 3.1 MB states per slot, graph replay, card 1).
RESULT (us per GDN layer, y bitwise equal in every case):

| window | per-slot launches | one launch | per 36 layers |
|---|---:|---:|---:|
| C2: 2 slots x 3 rows | 25.4 | 19.5 | -0.21 ms |
| 2 slots x 4 rows | 29.1 | 21.6 | -0.27 ms |
| C4: 4 slots x 2 rows | 43.1 | 33.9 | -0.33 ms |
| C1: 1 slot x 4 rows (not batch_rec_) | 14.7 | 14.9 | n/a (path not used) |

VERDICT: bitwise, -0.21 ms per C2 cycle, -0.33 ms per C4 cycle. Falls back to the per-slot loop if GDN_SPLIT=1,
more than 4 slot groups, or a group > 8 rows.

## R1 router (STRATA_NIGHT_ROUTER) - NOT integrated (negative result, code kept in hc_work/)

CONFIG: 2560 -> 512 BF16 router GEMV + shared-gate row (2560 -> 1) + native_router_top10_multi, 16 weight sets,
graph replay. COMMAND: hc_work/router_probe (+ NIGHT_ROUTER_MODE builds). RESULT at T=6 (us): default GEMV 11.4,
gate 2.9, top-10 7.5, chain 22.5. STRATA_MMVF_ROWS rows kernel (the S26 LFUSE aux path's GEMV): 30.2 (14-31 at T=2-8,
1.3-2.7x slower than the default on B70). Fused one-launch variants (all bitwise, stress-checked):
- GEMV + gate row as block 513 of the default-arithmetic GEMV: 11.6 (vs 11.4 + 2.9) - the only win (-2.7 us/layer),
  but it needs the LFUSE gate-deferred plumbing (sg_ready / gated combine) to drop the shared expert's own gate launch.
- + last-block top-10: 25-27.5 (worse than the 22.5 chain). The device-scope arrival counter alone cost ~5 us over 512
  blocks (two-level counting did not help: the acq_rel RMWs, not contention), and the top-10 tail costs ~8 us inside
  the last block: native_router_top10's 7.5 us is its own serial 10-round argmax latency, not launch overhead.
VERDICT: no engine change. Next step if pursued: a faster top-10 (parallel rank-by-count selection, bitwise ties to
the lower id) - the selection, not the launch, is the cost; and enable the gate row only together with LFUSE.
Files: hc_work/night_router.inc, hc_work/night_router.hpp, hc_work/night_router_native_bf16.diff.

## Not done (time)

- gdn_ab_multi (ssm_alpha/beta 2560->48 x2, 36 x 11.2 us per C2 cycle): only 96 sub-groups in flight; bitwise needs
  each lane's 10 chunks in order, so the lever is ILP (hoisted loads), not more groups.
- indexer q/k pair GEMV (2560->512 + 2560->128 share the input; default mmvf kernel per output is bitwise-movable
  into one launch: ~1 launch + ~4 us per QSA layer, 12 layers).
- commit batching across slots/layers (gdn_step_commit 72 x 16.1 us per C2 cycle, already ~390 GB/s: mostly
  bandwidth; S145 layer batching is gated by STRATA_GDN_COMMIT_LAYERS, not re-investigated).

# Mission 3 (batch slot admission: host round trip -> device to device)

Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-hc-admit1 (also carries HC v2 and GDN slots, default off)
Patch (this mission only): /mnt/vm_8tb/b70/results/night-20261016/agents/hc_admit_d2d.patch
Full worktree diff vs e1b7ee8a: /mnt/vm_8tb/b70/results/night-20261016/agents/hc_full.patch

Flags: `STRATA_NIGHT_ADMIT_D2D=1` (D2D copies + split timing in the admission log line),
`STRATA_NIGHT_ADMIT_VERIFY=1` (with D2D: after each stage / draft copy, runs the OLD saves from the source and compares
every region the old restores write against the slot: running state via conversation_checkpoint_verify_device,
every QSA layer's K/V via conversation_kv_verify (authoritative pools + the draft ring's resident pages); logs
`strata: STRATA_NIGHT_ADMIT_VERIFY: ... PASS` or `FAIL: <why>` once per copy; slow, for one request).

## A1 design

- conversation_checkpoint_copy_device(from, to, g, ids, err, sync_end) (conversation_state.cpp): save's targets check
  of `from`, restore's payload validation against `to`, then queue memcpys on the current device's in-order queue in
  the pair's order: GDN rows, PLE history, per owned QSA layer idx_tail / idx_dead / idx_block_pos and (non-empty
  prefix) the pooled row ids.size()/idx_block from idx_dead; ple_prev set from ids. No host buffer, no per-piece wait.
- conversation_kv_copy_device(from, to, g, upto, index, err, sync_end) (conversation_snapshot.cpp): layout/valid of
  both, the image-vs-target checks of restore (format incl. rotation, cells, page size, pooled rows, part sizes,
  target pools), the five parts [0, size) from pools(from) to pools(to) (host-authoritative pools for kv_mode 1/2, as
  save/restore use), then restore's fix-ups on the same queue: kv_mode 1 kv_stream_reset, kv_mode 2 kv_ring_restore.
- generate.cpp: copy_to_slot and copy_from_slot (main <-> slot sessions on every stage, then the MTP draft K/V) use
  them behind the flag: one device-wide sync before (as before), one after each stage, one after the draft copy.
  copy_from_slot with a checkpoint `at` still restores that checkpoint's running state from its host copy (it has no
  device source), then copies the K/V D2D. The pooled-row order is preserved (checkpoint row first, K/V pooled after,
  same in-order queue), so the final bytes match.
- Admission log with the flag: `strata batch: slot S takes N tokens (copied in X ms: park P, copy C, draft_first D ms)`;
  the yield path logs `strata batch: yield copy: park P ms, copy C ms`. park_slot (RAM cache) is unchanged.

## A2 validation

CONFIG: conversation_snapshot_test (existing fixtures) + new d2d_admission(fmt, mode): source A, targets B (D2D) and
C (old save + restore pair, generate.cpp's order), B and C initialised identically (every allocation), A different.
COMMAND: `nl_wt.sh ninja hc conversation_snapshot_test`, run on card 1. RESULT: every allocation of B (K/V pools,
host pools, scales, pooled rows, page tables / streaming map / ring slots, GDN rows, PLE, indexer tail/dead/pos)
byte-identical to C for F16 / INT8 / Q4 x kv_mode 0/1/2 and hybrid K8V4 mode 0, upto in {0,1,3,4,5,63,64,65,96};
ple_prev equal; conversation_checkpoint_verify_device and conversation_kv_verify agree. "4640 checks passed".
VERDICT: the D2D path writes exactly the bytes of the save + restore pairs. Not measured end to end (no servers):
expected copy time ~1 ms per admission (about 120 MB of GDN state at device bandwidth plus the K/V) against
388-620 ms now; park_slot time is reported separately.

# Mission 4 (HC Q8 toward bandwidth; park_slot)

Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-hc-park1 (HC v2 unchanged from the merged version;
adds STRATA_NIGHT_PARK_MOVE). Patch (this mission, generate.cpp only): agents/hc_park_move.patch.

## H1 HC Q8 v2: where the time goes (probe, card 1, graph replay; us per read / per kernel)

| T | read (DRAM) | read (weights L2-resident) | down2q DRAM / L2 | up2q DRAM / L2 |
|---|---:|---:|---|---|
| 1 | 24.8 | 14.2 | 12.8 / 8.8 | 8.2 / 5.6 |
| 4 | 29.6 | 19.7 | 15.6 / 12.0 | 10.0 / 7.4 |
| 6 | 34.4 | 26.5 | 18.3 / 16.3 | 12.2 / 9.7 |
| 8 | 46.6 | 36.5 | 18.3 / 16.4 (T=8 spills) | 12.3 / 9.6 |

Only ~25-40% of a read is DRAM time; the rest is fixed per-kernel latency (the down kernel's chain staging ->
dots -> SLM partial sum -> global partial -> arrival counter -> last-group reduce, the ~4 us kernel boundary between
down and up) plus per-row issue that grows ~1.5 us per row in down2q even from L2.

Tried (all correct: FP32-rounding vs the FP64 reference on the Q8 weights; PF bitwise = no PF), none faster:
- up3q (one row per lane over K/2 or all of K, one wave of 80/40 groups instead of 1.6 waves): KP2 = up2q
  (24.4 vs 24.8 T=1, 34.1 vs 34.4 T=6), KP1 worse (+1..+10 us).
- L2 prefetch of the up kernel's Q8 bytes from inside down2q (sycl prefetch, L2 hint): no change (24.8 / 34.5).
- down2q shapes (rows per lane, columns per sub-group): (4,32) 13.0/15.5/18.2 us at T=1/4/6 stays best; (2,32)
  +2-3 us, (1,32) +5-10, (1,160) +4-24, (2,160) +14-26; GRF256 (8,32) +0.7 us, GRF256 (4,32) +1-3 us.
VERDICT: v2 Q8 is at a local optimum of this structure; nothing integrated. A further step needs a different
structure (fold the last-group reduction and the down/up boundary away, e.g. one persistent kernel with a device
barrier, or overlap the up weights' DRAM stream with the down kernel through a concurrent queue), not tuning.
Experiment code kept in hc_work/fused_gr_night_v3exp.inc and hc_probe.cpp (modes v3, d2q).

## P1 park_slot: why ~190 ms for a 104-token conversation (probe hc_work/park_probe.cpp, card 1)

The parked image holds the conversation's GDN running state (36 layers x 3.27 MB = 117.7 MB, independent of tokens)
and the slot's checkpoints (each another full 117.7 MB running state). Measured per piece:

| piece (117.7 MB) | default glibc | GLIBC_TUNABLES=glibc.malloc.hugetlb=1 |
|---|---:|---:|
| fresh std::vector resize (zero fill + 4 KB page faults; checkpoint_save's c.gdn.resize) | 63-64 ms | 15 ms |
| D2H device -> pageable vector (pinned is the same: ~4.4 GB/s on this link) | 27 ms | 27 ms |
| one checkpoint's host copy (SavedConversation.checkpoints = view.checkpoints) | 75-78 ms | 21 ms |
| sum of the three | ~170 ms (the observed 184-200 ms) | ~63 ms |

So the cost is host page faults on two fresh 118 MB allocations, then the PCIe read; the K/V of a short conversation
is noise.

Changes / settings:
- `STRATA_NIGHT_PARK_MOVE=1` (generate.cpp park_source / park_slot): a slot's checkpoints are MOVED into the parked
  image instead of copied (the slot is overwritten right after its park in both callers: admission and the prompt
  yield). The view used for validation and the budget estimate still includes them; the saved image is identical
  (same checkpoint objects); main-session parks (park_current) and layer splits keep the copy. Removes the 21-78 ms
  checkpoint copy.
- Recommended env (no code): `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` on the engine process: malloc uses
  madvise(MADV_HUGEPAGE) for its large mmap chunks (host THP mode is [madvise]), so the 118 MB allocations fault
  2 MB pages: resize 63 -> 15 ms, host copies 77 -> 21 ms. Process-wide; restores read the same vectors.
- Expected park with both: ~15 (allocation) + 27 (D2H) + K/V ~ 45 ms instead of ~190 ms. Not measured in serving.
- Not done: async D2H (D2D snapshot into a staging device buffer, host copy in the background) - it would hide
  the remaining ~27-42 ms but needs VRAM proportional to the conversation's K/V (up to GBs at 200K) and deferred
  cache insertion; the RAM-parking semantics are unchanged by the two changes above.

# Mission 5 (persistent single-kernel HC read, STRATA_NIGHT_HC=q8p) - NEGATIVE, not integrated

Code kept outside the engine: hc_work/fused_gr_night_q8p.inc (kernel), hc_work/hc_probe.cpp mode `p`,
hc_work/coop_probe.cpp (co-residency / barrier probe). Engine source is unchanged from strata-hc-park1.

## C1 co-residency on this stack (coop_probe, card 1)

- `max_num_work_groups` (sycl kernel_queue_specific) answers 256 for a 256-thread group with 0 or 64 KB SLM: not an
  occupancy answer; unusable as the co-residency bound.
- root_group barrier (use_root_sync -> cooperative launch): works eagerly (3000 launches, 0 wrong) but a SYCL graph
  containing it fails at finalize/replay: `level_zero backend failed with error: 45 (UR_RESULT_ERROR_INVALID_ARGUMENT)`.
  The verify window is a graph, so the runtime cannot GUARANTEE co-residency where it would be used.
- Device: 32 Xe-cores x 8 EUs x 8 threads, 128 KB local memory. The q8p grid was therefore derived from the device:
  Xe-cores x min(threads / 16, 128 KB / kernel SLM) / margin (2 default; 1 tested) -> 64 / 120 groups.
- Hand-made device-scope barrier: sense reversal (arrival count + generation, last arrival resets the count). An
  earlier monotonic-target barrier DEADLOCKED in the probe when two grid sizes shared one counter (fixed design;
  card 1 healthy afterwards, xpu-health OK). Barrier cost (one-barrier kernel minus an empty one, graph replay):
  ~1.2-1.5 us at 64 groups, ~2.9-3.8 us at 120 groups, vs ~0.6-0.8 us for a kernel boundary between empty kernels.

## C2 q8p kernel and result

Phase A = down2q's units and arithmetic (incl. its per-row-group last-block reduction); each block then issues its
up unit's Q8 weight loads (up3q<2> layout) into registers, arrives at the grid barrier, and runs the up unit.
Correctness: lo and R_out bitwise = v2q; mixed FP32-rounding vs v2q (max rel 1.7e-5) and vs FP64 on the Q8 weights
(3.0e-5); stress 48 reads x 100 graph replays at T=1/4/6 per grid size: 0 bad replays, no hang.

| T | v2q us/read | q8p grid 120 | q8p grid 64 |
|---|---:|---:|---:|
| 1 | 24.8 | 28.0 | 31.1 |
| 4 | 29.4 | 33.6 | 39.3 |
| 6 | 34.8 | 39.4 | 52.7 |

VERDICT: slower at every T. The grid barrier at 120 groups costs more than the kernel boundary it removes, and the up
phase loses the 160-group spread; a smaller (safer) grid serialises phase A. Without a runtime co-residency guarantee
inside graphs the design is also a hang risk in serving. Not integrated; STRATA_NIGHT_HC=q8p does not exist in the
engine.

INCIDENT (for the coordinator): the deadlocked barrier probe at 12:39-12:46 outlived its `timeout` (the docker
client was killed, the container kept running) after gpu-run released card 1; agent wt-attn acquired card 1 at
12:46:04 while the hung kernel still occupied it (killed at ~12:46, card 1 HEALTHY). Measurements on card 1 around
12:40-12:47 may be disturbed. run_probe.sh now names the container and kills it before releasing the lease.
