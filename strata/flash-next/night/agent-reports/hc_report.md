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
