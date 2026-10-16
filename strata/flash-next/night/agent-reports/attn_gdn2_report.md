# attn mission 4: GDN decode verify/commit (STRATA_NIGHT_GDN2)

Agent: Claude Opus 5.5 (claude-opus-5-5), worktree `attn`. Base: main 1c360526, synced into the worktree (local
commit 39f8b4ef). 2026-10-16, 14:16-15:45 UTC. GPU: card 1 only, under the lease.

Deliverables:
- Patch: /mnt/vm_8tb/b70/results/night-20261016/agents/attn_gdn2.patch (against main 1c360526; applies cleanly with offsets to 14405cdc).
- Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-attn-g3.
- Bench: new ninja target `gdn2_bench` (sycl/src/kernels/gdn2_bench.cpp); logs in attn_work/gdn2*.log; ISA in
  attn_work/isa/gdn/dump.

Flag:

    STRATA_NIGHT_GDN2=1   (default off; latched; one stderr line; it reports itself off under STRATA_GDN_SPLIT=1
                           or STRATA_QFUSE=1, whose verify paths do not save deltas)

## 1. What the kernels do today (source + ISA)

- The state is 48 value heads x 128 x 128 FP32 = 3 MB per layer per slot (not 16 heads / 1 MB). 36 GDN layers: 108 MB
  per slot.
- Verify recurrence (`gdn_step_norm_multi_kernel<true,false>`, C1 through the GRF256 launch): one 512-thread
  work-group per head (128 columns x 4 row groups). Each thread loads its 32 state rows ONCE into registers and
  keeps them across all T rows; the state is NOT re-read per row and NOT written. Per row: 4 barriers (k/q staging,
  kv row-group sum, o row-group sum, norm).
  - ISA, GRF256: numGRF=256, no spill, 1,897 instructions, 77 loads.
  - The C2 slots launch (`night_gdn_step_norm_slots_k`, GRF128) spills 1,280 B (832 B before this patch's extra
    pointer argument; the moe agent measured GRF256 slower there).
- Commit (`gdn_step_commit_kernel`, 192 work-groups of 128): re-reads the 3 MB state and REPEATS each accepted row's
  kv reduction (32-row fma chains, red[0] + red[1] + red[2] + red[3], 3 barriers per row), then writes the 3 MB back.
  - ISA: 128 GRF, no spill, 1,553 instructions, 59 loads.
  - Per C1 window: 573 us (36 x 15.9 us).

## 2. Change: commit from the verifier's deltas (bitwise)

For the accepted prefix, the verify and the state-only commit run identical arithmetic from the same state:
- the same kv fma chain over each 32-row group, and the same red[0] + red[1] + red[2] + red[3];
- delta = (v - g kv) beta;
- s = fma(g, s, k delta).

So the verify's deltas ARE the commit's deltas. With the flag on:
1. Every verify launch (C1 GRF256, the plain all-output and partial-output paths, the C2 GDN_SLOTS launch) stores
   each row's delta (48 x 128 floats per row per layer, a store only) into a per-layer buffer `delta_L_`. That is
   36 x MT x 6144 floats, taken from the verifier arena only when the flag is on.
2. The commit becomes `gdn_commit_delta`: per state element, n times s = fma(g_t, s, k_t[row] * delta_t[col]).
   There are no reductions, barriers or local memory; 768 work-groups x 128, 8 rows per lane.
   - ISA: 128 GRF, no spill, 485 instructions, 14 loads, 0 barriers.
3. `gdn_commit_layers` (C2, all 36 layers in one launch) takes the same delta body when given the deltas.
4. Untouched: one-token windows the verify self-commits (no commit graph); conv history commits.

## 3. Correctness

CONFIG -> card 1, `gdn2_bench`. Model shapes: S=128, h_k=16, h_v=48, conv channels 10240. Random state (0.03 N),
q/k 0.08 N, v 0.7 N, beta in [0.1, 0.9], log decay -0.05 (1 + |N|), z/gamma as the gdn_parity fixture.
STRATA_NIGHT_GDN_GRF=1 as in the stack.

COMMAND -> `run_bin.sh gdn2_bench gdn2b_T4_s1 STRATA_NIGHT_GDN_GRF=1 -- --T 4 --slots 1`, the same with
`--T 3 --slots 2` (gdn2c_T3_s2).

RESULT:
- Verify outputs are bitwise with and without the delta store: C1 T=4 (24,576 floats) and C2 3x2 slots
  (36,864 floats).
- Committed state, delta commit vs state-only commit: bitwise for n = 1..4 (C1) and n = 1..3 per slot (C2).
- `gdn_commit_layers` with vs without deltas: bitwise for n = 1..3, all 36 layers, conv history included
  (29.4 M floats).
- PASS, 0 failures.

VERDICT -> bitwise.

## 4. Timing (graph replay, 36 layers with distinct DRAM-resident states, us per layer)

| shape | verify old -> with delta store | commit old -> delta | saved |
| --- | --- | --- | ---: |
| C1, T=4 (n_keep 4) | 13.01 -> 13.12 | 16.99 -> 13.84 (455 GB/s) | -0.11 ms per window |
| C2, 3 rows x 2 slots, per-slot commits | 19.88 -> 19.98 | 32.31 -> 26.92 | -0.19 ms per cycle |
| C2, gdn_commit_layers (36 layers, per owner) | - | 487.6 -> 472.4 us per owner | -0.03 ms per cycle |

The trace has 13.9 us verify and 15.9 us commit in C1, matching the bench.

In the final stack (C2 on `gdn_commit_layers`), C2 gains only ~0.03 ms. The batched launch already overlaps 36
layers and runs at ~480 GB/s on 226 MB, so the reductions it drops were hidden there. C1 gains ~0.11 ms per window
(~0.5% of a ~22.9 ms window).

## 5. What is left: the commit is at the state-bandwidth floor

Every window moves each layer's state three times: the verify reads it, and the commit reads and writes it. That is
6.3 MB of commit traffic per layer, about 13 us at ~480 GB/s. GDN2 brings the commit to that floor.

The remaining lever is structural: a deferred commit fused into the next window's verify. The verify would read the
stale state, apply the previous window's accepted rows from the saved deltas in registers (this patch's arithmetic),
run its T rows, and write the committed state back.
- That removes one 3 MB read per layer and the separate commit launch (~-6.5 us per layer, about -0.25 ms per C1
  window, plus a shorter commit phase).
- It needs a flush wherever the state is read outside the verify: prefill continuation, conversation snapshots, slot
  admission and moves. That is engine-wide state-ownership work I did not attempt in this window.

Other observations:
- Verify per-row cost: 4 barriers per row with 48 work-groups only (one per head). A 128-row-per-thread layout
  (SIMD16, no kv/o cross-thread reduction, 1 barrier per row) could save ~3-5 us per layer. It needs a bitwise
  emulation of the 32-lane norm butterfly across two SIMD16 sub-groups. Not done.
- The C2 slots verify spills 1,280 B at GRF128. A 256-thread x 2 work-groups-per-head layout would fit 2
  work-groups per core at GRF256; not attempted.
