# mmvq: dense small-batch MMVQ toward bandwidth roofline (night 2026-10-16)

Status: v2 (supersedes v1). Flag: `STRATA_NIGHT_MMVQ=1` (default off; one stderr line when enabled).
Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-mmvq-v2 (sha256 4491772b10e4a8cc...)
        (v1 = strata-mmvq-v1, Q4_K/Q5_K/Q6_K only, sha256 9c4b0f877310f284...)
Patch:  /mnt/vm_8tb/b70/results/night-20261016/agents/mmvq.patch (worktree /mnt/vm_8tb/b70/build/night-wt/mmvq/src-ep)
Covers: Q4_K, Q5_K (dense AND the 2560x248320 head, verifier C6 and MTP C2 calls), Q6_K, IQ4_XS (v2), ncols 1..8.
Not covered (original kernels): IQ4_NL (shared down 640->2560), Q8_0 (MTP drafter), IQ3_S (mmvq_multi_nc<21>), Q2_0.
IQ4_XS ncols=1 with n_out < 2048 stays on the original kernel (night measured 0.91-0.97x there).

## What the original kernels do wrong (ISA evidence, not guesses)

Dumped ISA (IGC_ShaderDumpEnable, AOT bmg-g31, engine flags) of `native_mmvq_wide_kernel<WideQ4K,6>`:
the loop body is ~850 SIMD32 instructions per 4-block step, 468 of them `mov`; every activation load
(`load.ugm.d32x4`/`d16u32`) targets the SAME register block r4:8 and is consumed before the next one is
issued, i.e. ~24 serialized L1 loads per step for 6 columns (each row re-reads every column's activation:
~10x the weight bytes in L1 traffic). Repeating ONE matrix (L2-resident, PROBE_NMAT=1) the original Q4_K
nc6 kernel still takes 82 us vs 94 us from DRAM: it is issue/latency bound, not bandwidth bound.

## New kernel family: KREG (K split across the work-group, activations in registers)

- One work-group spans the whole K of a row; lane l owns 32 weights of every row (8 lanes per 256-block, the
  wide kernels' per-lane split). SIMD16 sub-groups (2x registers per lane).
- Each lane loads its NCOLS activation pieces ONCE (plus precomputed min-term sums for Q4_K/Q5_K).
- Tiles of R rows: R independent weight loads per lane (Q4_K/Q5_K: one 16 B header load dm+scales and one
  16 B qs load; Q5_K +16 B qh; Q6_K: the A2 aligned loads), R x NCOLS dots (same dp4a math), transposed
  sub-group butterfly (each lane ends with R*NCOLS/16 finished sums), double-buffered SLM sum across
  sub-groups, one SLM-only barrier per tile.
- Two ISA-found pitfalls fixed: `cond ? v[i] : v[i+H]` on array lvalues makes clang select ADDRESSES and
  spills v[] to private (global) memory; `sycl::group_barrier` also fences global memory (fence.ugm).
  Branchless K-quant scale decode (lanes g<4 and g>=4 share a sub-group).
- Per (format, ncols) tile shape (from the sweep): Q6_K R=4 (nc<=3) / R=2 (nc>=4), GRF128; Q4_K/Q5_K
  nc1 R=4; nc2 R=8 GRF256 (n_in<=4096) else R=4; nc3-4 R=4; nc>=5 R=8 GRF256 (Q5_K n_in>4096: R=2).
- Numerics: different summation order than the wide kernels -> not bitwise; max |err| / max|y| <= 3e-7,
  max elementwise rel err (|y| > 1e-3 max) <= 1.2e-4 (FP32 rounding level over 2560-6144-term sums).

## E1 baseline microbench (original serving kernels)

CONFIG -> card 1, verbatim copies of the serving kernels (WIDE A2 Q6_K, WIDE_K RPW1, image e41d59c0),
random valid blocks (finite fp16 scales ~2^-10), Q8_1 activations, in-order queue graph replay of NMAT
distinct weight copies (>= 640 MB per replay, L2 cannot hold), median of 7-11 replays.
COMMAND -> mmvq_work/probe/run.sh mmvq_probe <fmt> all 1,2,3,4,6 0 (probe source: mmvq_work/probe/)
RESULT -> e.g. Q4_K qkv nc6 93 us (26%), Q5_K head nc6 2426 us (30%), nc2 1251 us (57%); matches the
C2 trace (head C6 2259-2525 us, Q4K qkv 99.9 us). VERDICT -> baseline reproduced.

## E2 engine A/B (the shipped code, through the public native_mmvq / native_mmvq_head API)

CONFIG -> worktree build, night_mmvq_bench (new, sycl/src/kernels/night_mmvq_bench.cpp), card 1, same
protocol as E1, original dispatch (night off) vs STRATA_NIGHT_MMVQ path (toggle native_mmvq_set_night).
COMMAND -> mmvq_work/run_engine_bench.sh 14,13,12 1,2,3,4,6   (raw: mmvq_work/engine_ab_v1.txt)
RESULT (us per call, % of 608 GB/s for night):

| fmt  | shape (n_in x n_out) | nc1 orig->night | nc2 | nc3 | nc4 | nc6 |
|------|----------------------|-----------------|-----|-----|-----|-----|
| Q6_K | qkv 2560x10240  | 55.6->41.0 (86%) | 62.9->41.5 (85%) | 68.5->42.7 (83%) | 74.4->46.5 (76%) | 89.5->58.0 (61%) |
| Q6_K | gate 2560x6144  | 38.8->27.0 (79%) | 42.5->27.2 (78%) | 45.5->28.0 (76%) | 48.7->30.2 (70%) | 57.9->37.3 (57%) |
| Q6_K | out 6144x2560   | 47.1->28.1 (76%) | 50.3->28.1 (76%) | 52.4->28.6 (74%) | 55.4->30.4 (70%) | 65.6->37.3 (57%) |
| Q6_K | attn_q 2560x12288 | 69.2->49.1 (87%) | 77.1->49.7 (85%) | 83.7->50.9 (83%) | 90.3->56.2 (76%) | 107.7->70.1 (61%) |
| Q6_K | kv 2560x512     | 6.2->5.1 | 9.0->5.1 | 10.3->5.2 | 10.5->5.1 | 12.3->5.9 |
| Q6_K | sh_gu 2560x640  | 6.2->5.0 | 8.6->5.1 | 10.2->5.3 | 10.7->5.6 | 12.7->6.5 |
| Q5_K | qkv             | 49.2->34.7 (85%) | 61.2->36.0 (82%) | 74.2->36.9 (80%) | 87.0->40.6 (73%) | 114.1->52.2 (57%) |
| Q5_K | gate            | 33.1->22.6 (79%) | 40.3->23.8 (75%) | 47.8->24.5 (73%) | 55.1->26.6 (67%) | 70.7->33.9 (52%) |
| Q5_K | out             | 46.6->23.0 (77%) | 54.8->23.3 (76%) | 62.1->24.4 (73%) | 70.1->25.9 (69%) | 90.4->37.8 (47%) |
| Q5_K | attn_q          | 60.6->41.5 (86%) | 75.2->42.4 (84%) | 90.8->43.8 (81%) | 105.9->48.2 (74%) | 137.9->62.8 (57%) |
| Q5_K | HEAD 2560x248320 | 1061->761 (94.5%) | 1366->774 (92.9%) | 1697->795 (90%) | 2006->886 (81%) | 2660->1166 (62%) |
| Q4_K | qkv             | 44.5->28.8 (84%) | 55.7->29.6 (82%) | 66.7->32.1 (76%) | 80.7->32.1 (76%) | 108.7->40.7 (60%) |
| Q4_K | gate            | 30.7->19.0 (76%) | 37.1->19.7 (74%) | 43.7->21.4 (68%) | 51.5->21.7 (67%) | 67.2->27.0 (54%) |
| Q4_K | out             | 39.7->19.5 (75%) | 46.7->19.4 (75%) | 55.9->21.8 (67%) | 65.4->21.5 (68%) | 80.6->31.0 (47%) |
| Q4_K | attn_q          | 53.7->34.2 (85%) | 67.2->34.7 (84%) | 80.0->38.0 (77%) | 96.8->37.9 (77%) | 130.7->48.2 (60%) |
| Q4_K | kv / sh_gu      | 5.0->3.7 / 5.1->3.9 | | | | 11.4->5.4 / 11.8->5.7 |

All rows: max|err| <= 1.2e-4 abs on outputs of magnitude 50-620 (see raw file), no non-finite values.
VERDICT -> >=70% of roofline reached for every 6-13 MB matrix at nc<=3 (and most nc4); nc6 at 47-61% (up
from 19-40%); head 2.3x at C6 (2660 -> 1166 us) and 1.77x at C2 (1366 -> 774 us, 93%). Small (1 MB)
matrices are at the ~4-5 us launch/ramp floor (1.2-2.1x faster).

Estimated C2 cycle effect on card 0 (trace census x measured ratios, not a serving measurement): Q6_K dense
3.83 -> ~2.30 ms, head C6 2.26 -> ~0.99 ms, MTP C2 heads 2.42 -> ~1.37 ms, Q5_K dense ~1.0 -> ~0.45,
Q4_K 2.31 -> ~0.91 ms: ~5.8 ms per C2 cycle. Needs a serving A/B (orchestrator) with STRATA_NIGHT_MMVQ=1.

## E3-E4 (probe) tile-shape sweep and fixes

- First KREG version was SLOWER (ISA: v[] in private memory via address selects, fence.ugm in the barrier,
  divergent scale branch). After the fixes nc6 Q4_K qkv 72 -> 40 us; nc1 45 -> 28.6 us.
- Persistent grids (cap 256/512 groups) did not beat one group per tile; GRF256 helps only at R=8.

## E5 IQ4_XS (probe, then integrated in v2)

The IQ4 codebook lookup (iq4_lut4, a select chain per nibble) dominates; KREG with it is slower than the
original (0.86-1.0x). Two cheaper lookups tried: SWAR mux tree with byte masks (m*) and a 256-entry SLM
table code(lo)|code(hi)<<16 (s*). RESULT -> s8/s4: nc1 1.22-1.42x, nc2 1.3x, nc4 1.3-1.4x, nc6 1.23-1.39x
over the original (still only 41-68% of roofline). VERDICT -> integrated in v2 (SLM table, R=8 at nc1, R=4 above).

## E6 engine A/B v2 (all four formats, raw: mmvq_work/engine_ab_v2.txt)

CONFIG -> as E2, v2 worktree build. COMMAND -> run_engine_bench.sh 14,13 1,2,3,4,6 ; 12,23 1,2,3,4,6
RESULT -> Q4_K/Q5_K/Q6_K rows identical to E2 within noise (<1%). IQ4_XS (us orig -> night, % roofline):

| shape | nc1 | nc2 | nc3 | nc4 | nc6 |
|-------|-----|-----|-----|-----|-----|
| qkv    | 42.9->33.5 (68%) | 47.7->37.8 (61%) | 52.0->40.1 (57%) | 56.7->41.6 (55%) | 67.8->49.2 (47%) |
| gate   | 29.1->23.7 (58%) | 31.2->25.5 (54%) | 33.2->26.5 (52%) | 35.6->27.2 (51%) | 42.3->34.0 (40%) |
| out    | 31.4->21.8 (63%) | 32.9->25.9 (53%) | 35.1->26.9 (51%) | 36.6->26.7 (51%) | 45.0->31.7 (43%) |
| attn_q | 54.6->40.0 (69%) | 59.9->45.7 (60%) | 64.4->48.0 (57%) | 69.6->49.5 (56%) | 82.7->60.3 (46%) |
| kv/sh_gu | 5.0->5.0 (orig) | 5.2->4.6 | 5.9->4.7 | 6.3->4.9 | 7.9->7.0 / 8.1->7.0 |

max|err| <= 7.6e-5 (outputs up to ~560), max rel 3.7e-5. VERDICT -> 1.23-1.44x on the large IQ4_XS shapes;
still ALU-bound (the codebook decode), below the 70% target.

## Estimated C2 cycle effect (card 0; trace census x measured nc6/nc2 ratios; NOT a serving measurement)

| family (C2 trace ms/cycle) | night estimate |
|---|---|
| Q6_K dense 3.83 | ~2.30 |
| Q5_K head C6 2.26 | ~0.99 |
| Q5_K MTP heads C2 2 x 1.21 | ~1.37 |
| Q5_K dense ~1.0 | ~0.45 |
| Q4_K 2.31 | ~0.91 |
| IQ4_XS 2.07 | ~1.58 |
| total ~12.7 | ~7.6 (-5 to -6 ms per ~46-57 ms cycle) |

C1 benefits too (verify nc<=4: 1.4-3x; drafts nc1-2; head nc1 1.40x, nc4 2.26x). Needs the orchestrator's
serving A/B with STRATA_NIGHT_MMVQ=1 (matched config, coherence, C1/C2 tok/s).

## Flags

STRATA_NIGHT_MMVQ=1 enables the KREG path for Q4_K/Q5_K/Q6_K (incl. head) at ncols 1-8. Other formats and
STRATA_Q6_PACKED-registered tensors keep the original kernels. STRATA_HEAD_Q5K_RPW2 / *_RUNTIME_ONES (if set)
still take precedence on the head.

## E7 (probe only, NOT in the engine) software-pipelined persistent KREG for nc6

CONFIG -> kreg_pf_kernel (probe/kern_night.hpp): persistent grid (cap groups), the next tile's raw weight loads
issued before this tile's decode+dots, Q4_K, nc6. Why: the L2-resident (repeated single matrix) time of the
nc6 KREG kernel is 28.4 us vs 40.3 us from DRAM (Q4_K qkv): the per-group chain act-load -> weight-load ->
math is not overlapped. Per-row instruction counts from ISA (loopmix.py): Q4_K nc6 204/row, Q6_K nc6 343/row
(load16_a2 selects), Q6_K nc1 223/row, IQ4_XS-SLM nc6 249/row.
COMMAND -> run.sh mmvq_probe 12 {qkv,out} 6 192,384,768 base,p4gc192,... (raw: probe/pf_q4k.txt)
RESULT -> qkv 2560x10240: p4 GRF256 cap192 34.8 us (69.7%) vs KREG g8 40.0 (60.6%) = -13%; out 6144x2560:
31.8 us vs 31.0 (no gain; 12-thread groups fit badly at GRF256). Q6_K 4-byte-aligned loads (KQ6Ka4): nc6
-3..-6%, nc1-4 slightly worse.
VERDICT -> worth integrating for n_in=2560 shapes at nc>=5 (Q4_K/Q5_K, likely the head too) in a follow-up;
not done tonight.

## Next steps (ranked by measured headroom)
1. nc6 (C2 verify) is still 47-62% of roofline for K-quants: integrate E7 for n_in=2560 shapes; for n_in=6144
   try a 2-blocks-per-lane mapping (6-thread groups) so GRF256 occupancy is not wasted.
2. IQ4_XS remains ALU-bound in the codebook decode (40-69%).
3. Uncovered formats: IQ3_S dense (mmvq_multi_nc<21>, 1.93 ms/C2 cycle), IQ4_NL shared down 640->2560
   (40 x 14.5 us), Q8_0 MTP drafter (0.74 ms).
4. Small (0.5-1.3 MB) matrices are at the 4-7 us launch/latency floor; only fusion would help further.

## Reproduce
- Probe: mmvq_work/probe/{kern_base.hpp (verbatim serving kernels), kern_night.hpp, mmvq_probe.cpp, build.sh,
  run.sh, loopmix.py}; ISA dumps: ISA=1 ./build.sh <src> <out> -> probe/isa_<out>/.
- Engine A/B: nl_wt.sh ninja mmvq night_mmvq_bench; mmvq_work/run_engine_bench.sh <types> <ncols> [shape].
