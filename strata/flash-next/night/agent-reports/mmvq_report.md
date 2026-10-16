# mmvq: dense small-batch MMVQ toward bandwidth roofline (night 2026-10-16)

ROUND 4 (latest, BF16 GEMVs): engine /mnt/vm_8tb/b70/results/night-20261016/builds/strata-mmvqm-bf16b (sha256
c69534036ba33123..., built from a reflink copy of main's working tree taken ~13:00 incl. its then-staged T12 changes,
+ this round), flag STRATA_NIGHT_BF16=1. Delta vs main's CURRENT working tree (13:20; only this round's edits, main's
later T12 guards preserved): agents/mmvq_bf16_vs_main.patch (git apply --check passes).
See "Round 4" at the end.

ROUND 3 (latest, shared expert): engine /mnt/vm_8tb/b70/results/night-20261016/builds/strata-mmvq-shexp2
(sha256 a9b00e320da52c34...), flag STRATA_NIGHT_SHEXP=1. Delta vs main (60b7f900):
agents/mmvq_shexp_vs_main.patch (applies cleanly); full worktree diff vs e1b7ee8a: agents/mmvq.patch. See "Round 3".

ROUND 2 (latest): engine /mnt/vm_8tb/b70/results/night-20261016/builds/strata-mmvq-r2b (sha256 a4b0ff80a16e197e...),
patch /mnt/vm_8tb/b70/results/night-20261016/agents/mmvq.patch (round 1 + round 2, vs base e1b7ee8a).
New flag STRATA_NIGHT_MMVQ2=1 (implies STRATA_NIGHT_MMVQ=1; one stderr line). Round-1 behaviour (STRATA_NIGHT_MMVQ=1
alone) is unchanged. Round-1 serving A/B by the orchestrator: C1 107.2 -> 131.7 tok/s, C2 63.9 -> 73.1 tok/s/stream.
See "Round 2" at the end.

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


# Round 2 (STRATA_NIGHT_MMVQ2=1)

## What it adds
1. Software-pipelined persistent KREG (night_pf_kernel): one group per resident slot (Xe-cores x (32 or 64)/threads
   per group), the next tile's raw weight loads issued before this tile's decode + dots; same lanes/butterfly/SLM
   order as round 1 (bitwise equal to round-1 KREG at the same NCOLS/R). Used for n_in <= 4096 only:
   Q4_K/Q5_K nc5-8 R4 GRF256, nc9-12 R2 GRF256; IQ4_XS nc5-6 R2 GRF128, nc7-8 R4 GRF256, nc9-16 R2 GRF256;
   Q6_K nc6-8 R2 GRF256 with 4-byte-aligned raw loads (needs matrix base % 4 == 0 and an even block count per row;
   else round 1). The 6144-wide projections keep round 1 (12-thread groups fit badly at GRF256; no win measured).
2. ncols 9-16 for Q4_K/Q5_K/Q6_K/IQ4_XS (incl. head): direct where measured (above), otherwise two calls of
   ceil(n/2) + floor(n/2) columns. The night hooks now run before validate_shape (which still caps the
   original kernels at 8).
3. IQ3_S dense (type 21, previously mmvq_multi_nc<21>): KREG lane = sub-block, iq3s grid in a per-group SLM table,
   sign masks by shifts; tiles R2 for n_out <= 1024, else nc1-2/5-6 R4, nc3-4/7-8 R8 GRF256 (from the probe2 helper,
   mmvq_work/probe2/, verbatim baseline kern_iq3s_base.hpp).
4. IQ4_NL with n_in <= 1024 (shared-expert down 640 -> 2560): one lane per 32-value block, R4, select-chain codebook.
5. Q8_0 (MTP drafter): NOT changed - probe2 measured 0.94-1.17x (already 63-91% of roofline).

## E8 engine A/B round 2 (night_mmvq_bench now times orig / level 1 / level 2; above 8 columns orig and level 1 are
two calls of 8 + rest; card 1, same protocol; raw: mmvq_work/engine_ab_r2{,_c,_d,_q6}.txt)

us per call (orig / lv1 / lv2), % = lv2 of 608 GB/s:

| case | nc5 | nc6 | nc8 | nc12 | nc16 |
|---|---|---|---|---|---|
| Q4_K qkv 2560x10240 | 86.3/37.7/34.0 | 105.1/40.3/35.3 (69%) | 133.8/56.0/37.5 (65%) | 203/80.2/56.3 | 258/99.6/67.9 |
| Q4_K gate 2560x6144 | 54.6/24.9/21.9 | 64.9/26.2/22.8 (64%) | 80.6/36.1/24.3 (60%) | 122/53.6/35.2 | 155/64.1/42.2 |
| Q4_K attn_q 2560x12288 | 104/44.8/40.1 | 127/47.7/41.4 (70%) | 160/66.2/43.9 (66%) | 246/96.1/68.1 | 312/121/80.6 |
| Q5_K gate | 62.5/29.7/28.1 | 70.1/33.6/27.6 (64%) | 84.8/40.9/33.5 | 129/61.2/37.9 | 160/71.9/56.6 |
| Q5_K attn_q | 121/54.1/52.2 | 137/62.3/51.1 (70%) | 167/75.4/59.8 | 261/111/72.7 | 324/139/108 |
| Q5_K HEAD 2560x248320 | 2266/997/957 | 2632/1160/948 (76%) | 3316/1383/1036 (69%) | 5360/2290/1352 | 6667/2767/2043 |
| Q6_K gate | 51.3/30.7/31.0 (lv1 path) | 56.1/36.2/33.6 (63%) | 97.7/40.3/35.2 (60%) | 148/64.6/57.2 | 200/76.0/59.6 |
| Q6_K qkv / attn_q nc6 | | 86.1/56.4/54.5, 105.9/69.4/67.1 | 160/63.8/57.4, 194/76.5/69.4 | | |
| IQ4_XS qkv | 56.5/40.7/36.5 | 65.4/48.6/39.6 (58%) | 78.5/55.0/44.4 | 138/88.7/52.9 | 164/104/56.4 |
| *_K / IQ4_XS out 6144x2560 | lv2 = lv1 (round-1 tiles) | | | | |

IQ3_S (orig -> lv2; lv1 = orig): gate 2560x6144 nc1 61.6->52.4, nc2 67.7->57.0, nc4 72.7->67.6, nc6 84.7->67.5,
nc8 90.9->72.6; out 6144x2560 nc1 87.5->50.7, nc6 110.3->68.9; qkv nc6 139->109; attn_q nc6 167->127;
sh_gu 2560x640 nc6 12.4->11.0; kv nc6 10.3->9.1. Still only 15-23% of roofline (issue/latency-bound: ~180
instructions per row-thread, 8 dependent SLM grid gathers; probe2 report).
IQ4_NL 640x2560 (orig -> lv2): nc1 8.82->6.59, nc2 9.60->6.71, nc4 11.42->7.12, nc6 13.39->8.54, nc8 16.07->9.19.

Errors (lv2 vs original): max abs <= 1.2e-4 on outputs up to ~620 (K-quants), IQ3_S <= 9.5e-6 (outputs ~50),
IQ4_NL <= 1.9e-6; max rel <= 7e-5; no non-finite values. Level 1 vs level 2 at nc <= 4 are identical paths.

## Cost per extra column (ncols 9-16, for a 12-16 row verify window)
Measured lv2, us: Q4_K gate nc8 24.3 -> nc12 35.2 -> nc16 42.2 (+2.7 us/col to 12, +1.8 us/col to 16);
Q5_K head 1036 -> 1352 -> 2043 (16 = two calls of 8); Q6_K gate 35.2 -> 57.2 (6+6) -> 59.6 (8+8);
IQ4_XS qkv 44.4 -> 52.9 -> 56.4 (direct). One 12-row call vs two 6-row windows: Q4_K gate 35.2 vs 45.6 (-23%),
Q5_K head 1352 vs 1896 (-29%), Q6_K gate 57.2 vs 67.2 (-15%), IQ4_XS qkv 52.9 vs 79.3 (-33%).
Direct KREG above 12 columns runs out of registers (activations 10-12 regs/col/lane): nc16 direct d2g was 2-3x
slower than two calls (probe r2_wide.txt), so 13-16 columns are split.

## Estimated C2 effect of round 2 on top of round 1 (card 0, trace census x lv1->lv2 nc6 ratios; not measured
in serving): Q5_K head C6 -0.21 ms, Q4_K/Q5_K dense -0.15, Q6_K -0.1, IQ4_XS -0.28, IQ3_S ~-0.4 (1.93 -> ~1.53),
IQ4_NL shared down -0.21 (0.58 -> ~0.37): ~-1.3 ms per C2 cycle. C4 depth1 (nc8) gains more (Q4_K nc8 -32%,
head nc8 -25%, Q6_K nc8 -10%, IQ4_XS nc8 -19% vs level 1).

## E9 negative / parked
- Q6_K pipelined with 16-byte-aligned raw chunks (A2): slower (spills, select-heavy funnel), not used.
- Q6_K pipelined at nc5: slower than round 1 (excluded); 6144-wide projections: no pipelined win.
- One open oddity: at level 2 an (8 pipelined GRF256) + (4 round-1 GRF128) column split ran ~25% slower than the
  sum of its parts (Q6_K nc12); splitting into equal halves (6+6, both pipelined) fixed it. Not root-caused.

## Flags summary
STRATA_NIGHT_MMVQ=1: round 1 (Q4_K/Q5_K/Q6_K/IQ4_XS, head, ncols 1-8). STRATA_NIGHT_MMVQ2=1: round 1 + round 2
(pipelined nc5-12, ncols 9-16, IQ3_S, IQ4_NL<=1024 wide). Both default off.


# Round 3: fused shared expert (STRATA_NIGHT_SHEXP=1)

## Before (per layer, verifier, native bf16 gate path, x_q8_1 ready as with STRATA_NIGHT_SH_LATE)
6 launches: gate mmvq, up mmvq, swiglu+q8_1, down mmvq, bf16 gate row (2560 -> 1), sigmoid+scale rows. Breakdown
(card 1, (gate,up,down) = (IQ4_XS, IQ3_S, IQ4_NL), nc1): gate 5.1 us, up 7.8, swiglu+q8 2.0, down 6.7, gate row 1.9,
scale ~1.5. The pack mixes formats: gate/up types differ in 31 of 48 layers (13 (gate, up) pairs: Q4_K/Q5_K/Q6_K/
IQ3_S/IQ4_XS); down IQ4_NL x40, Q2_0 x8.

## After: 3 launches
1. gate + up in ONE KREG launch (both matrices' tiles in one group; each matrix keeps its own format's lanes and
   activation registers, shared when the types match; per-format SLM tables side by side), epilogue h = silu(gate) *
   up (the fused quantizer's expression) -> h; NCOLS extra groups compute the shared gate row bit for bit like
   bf16_f32_mmvf_kernel<256> (virtual-thread fma order + replayed xor-butterfly trees).
2. native_quantize_q8_1(h) (same q bytes/ds as native_swiglu_quantize_q8_1 on gate/up).
3. down (IQ4_NL one lane per block; Q2_0 lane = Q20Traits iqs) with the sigmoid scale in its epilogue.
Tiles: same-type pairs R2 (GRF128 nc<=3, GRF256 above); mixed pairs R2 (GRF256 for nc>=3).
Tried and rejected (measured): in-kernel q8 quantization via a grid handoff (device-scope atomic per 32-row chunk;
STRATA_NIGHT_SHEXP=2 keeps it): +6-7 us per call (fences + tail); mixed pairs at R1: slower than two launches;
the gate row as ONE extra group with per-column serial loops: +5-10 us at nc>=4 (fixed: one group per column, loads
hoisted).

## E10 per-layer A/B (night_shexp_bench, card 1, 48 distinct layers per graph, in-order graph replay)
CONFIG -> shared_expert_multi(nc, x, nw, gate_inp_bf16, ..., x_q8_1_ready, lfuse=0), native bf16 on, original vs
STRATA_NIGHT_SHEXP (toggle night_shexp_set). Baseline MMVQ level 2 (raw shexp_ab_v6.txt) and level 1 (shexp_ab_v6_lv1.txt).
COMMAND -> mmvq_work/run_shexp_bench.sh 1,2,4,6,8 {2,1}
RESULT (us per layer call, orig -> night, MMVQ level 2 baseline; level-1 baseline in parentheses):

| (gate, up, down) x layers | nc1 | nc4 | nc6 | nc8 |
|---|---|---|---|---|
| IQ4_XS, IQ3_S, IQ4_NL x8 | 24.9 -> 18.7 (28.1) | 26.4 -> 24.2 (32.6) | 31.8 -> 27.0 (38.9) | 47.3 -> 29.1 (50.4) |
| IQ3_S, IQ3_S, IQ4_NL x7 | 27.6 -> 21.1 (32.0) | 29.9 -> 26.2 (38.4) | 36.8 -> 28.4 (44.6) | 42.5 -> 30.2 (59.0) |
| Q4_K, IQ3_S, IQ4_NL x8 | 23.7 -> 18.2 (26.9) | 25.9 -> 24.5 (32.0) | 39.7 -> 28.0 (58.9) | 47.2 -> 31.8 (72.5) |
| IQ4_XS, IQ4_XS, IQ4_NL x6 | 22.1 -> 15.7 (24.3) | 22.9 -> 20.3 (26.9) | 26.8 -> 23.2 (33.6) | 37.1 -> 24.5 (42.0) |
| IQ3_S, IQ4_XS, IQ4_NL x3 | 24.9 -> 18.2 | 26.4 -> 24.5 | 31.7 -> 26.9 | 41.9 -> 29.3 |
| IQ4_XS, Q4_K, IQ4_NL x2 | 21.0 -> 14.9 | 22.4 -> 20.7 | 29.8 -> 23.9 | 37.5 -> 25.9 |
| IQ4_XS, Q6_K, IQ4_NL x2 | 22.0 -> 15.7 | 23.6 -> 20.7 | 31.0 -> 23.6 | 37.7 -> 25.4 |
| Q4_K, Q4_K, IQ4_NL x2 | 19.8 -> 14.2 | 21.9 -> 17.8 | 28.7 -> 20.3 | 36.8 -> 22.1 |
| IQ3_S, IQ3_S, Q2_0 x2 | 42.3 -> 22.5 | 43.0 -> 28.2 | 53.1 -> 31.2 | 62.9 -> 36.1 |
| IQ4_XS, Q6_K, Q2_0 x1 | 36.8 -> 17.2 | 36.8 -> 22.5 | 47.5 -> 26.3 | 58.0 -> 31.4 |

nc2: 1.27-1.64x. Speedup vs level 2: 1.06-1.66x (never slower); vs level 1: 1.29-2.37x.
Correctness: vs the level-2 path every IQ4_NL-down layer is BITWISE identical (0 differing output bits at every nc;
the KREG sums do not depend on R/NCOLS, the h/q8 expression and the scale factor are the originals, the gate row
replays bf16_f32_mmvf_kernel's order). Q2_0-down layers: KREG vs the multi kernel's order, max abs 3e-5 of 269
(1.1e-7 relative). Vs level 1: max abs <= 1.95e-3 of 1.5e4 (1.3e-7 relative), FP32 rounding level.
VERDICT -> ship behind STRATA_NIGHT_SHEXP=1. Estimated from the layer mix (not measured in serving): about -160 us
per C1 window (nc4) and about -350 us per C2 cycle (nc6) against the level-2 path (more against level 1).
Note: the fused path uses the KREG traits for IQ3_S/IQ4_NL regardless of STRATA_NIGHT_MMVQ2(_MASK).

## Flags (all default off)
STRATA_NIGHT_MMVQ=1 (round 1), STRATA_NIGHT_MMVQ2=1 (+ round 2; orchestrator's STRATA_NIGHT_MMVQ2_MASK kept),
STRATA_NIGHT_SHEXP=1 (round 3; =2 = experimental in-kernel q8 handoff, slower), STRATA_NIGHT_SHEXP_MIXED_MAXC=n caps
the column count for mixed-type gate/up pairs (default 8).


# Round 4: BF16 x FP32 GEMVs (STRATA_NIGHT_BF16=1)

## What it does
- Drop-in inside bf16_gemv_fp32_mmvf_multi (T = 2..8) and bf16_gemv_fp32_mmvf (T = 1, only n_in = 640): every caller
  of the BF16 mmvf path (MoE router 2560->512, QSA indexer q 2560->512 / k 2560->128, PLE, ...) takes it.
- BITWISE equal to the existing kernel by construction: the old kernel's virtual thread t of BLOCK (mmvf_block_size:
  256 for 2560, 160 for 640) walks pairs t, t + BLOCK, ... with two ordered fmas; here one SIMD16 sub-group is one
  32-thread warp, lane l holds virtual threads 2l, 2l+1 (adjacent pairs: one 8-byte weight load, one float4
  activation load per step), the warp's xor butterfly is replayed for its lane 0 (levels 16..2 = xor 8..1 shuffles,
  level 1 in-lane) and the partials[32] stage is replayed in its exact operand order. One group (BLOCK/2 lanes) spans
  K, the activations are loaded ONCE into registers, R = 4 output rows per tile (the old kernel re-read every
  activation row for every output row: n_out x T x n_in floats through L1/L2).
- night_bf16_gemv_pair(): two BF16 matrices on the same rows in one launch (rows [0, n1) from w1, [n1, n1+n2) from
  w2). verify.cpp (main) now computes the QSA indexer k_proj AND q_proj in one launch at the k_proj site and skips
  the later q_proj call when it ran (12 QSA layers).
- ISA lesson: the first version guarded every load (p < npairs): IGC put each load in its own divergent block (65
  goto/join + 83 sync per tile) and ran 0.4-0.8x the old kernel; shapes with whole BLOCK steps only (2560: 5 x 256,
  640: 2 x 160) as compile-time KMAX/NW removed all of it. T = 1 at n_in = 2560 stays on the old kernel (break-even).

## E11 A/B (night_bf16_bench, card 1, >= 32 distinct weight copies per graph, two runs agree within 1%; raw
bf16_ab_v3_{1,2}.txt). us per call orig -> night (% of 608 GB/s), all outputs 0 differing bits:

| shape | T1 | T2 | T3 | T4 | T6 | T8 |
|---|---|---|---|---|---|---|
| router / idx q 2560x512 | 7.26 (orig) | 7.90 -> 6.44 (67%) | 8.79 -> 6.71 | 9.73 -> 7.40 (58%) | 11.65 -> 8.03 (54%) | 20.34 -> 8.95 (48%) |
| idx k 2560x128 | 3.68 (orig) | 3.93 -> 3.09 | 4.28 -> 3.35 | 4.64 -> 3.84 | 5.32 -> 4.40 | 10.72 -> 5.10 |
| 2560x48 (ssm_ab shape*) | 2.94 (orig) | 3.31 -> 2.21 | 3.48 -> 2.68 | 3.67 -> 2.85 | 4.08 -> 2.70 | 9.20 -> 3.06 |
| 2560x2560 | 28.4 (orig) | 30.1 -> 24.3 (89%) | 33.3 -> 25.0 | 37.4 -> 25.4 (85%) | 45.6 -> 26.5 | 71.9 -> 28.8 |
| 640x2560 | 10.82 -> 7.90 | 12.06 -> 10.71 | 13.89 -> 7.68 | 15.82 -> 8.23 | 20.30 -> 9.44 | 33.30 -> 10.90 |
| idx k+q pair (one launch) | - | 11.82 -> 7.75 | 13.07 -> 8.05 | 14.35 -> 9.20 | 17.00 -> 9.97 | 31.06 -> 11.80 |

(*) ssm_alpha/ssm_beta go through gdn_ab_multi (fused, changed by agent moe), not this path; the row shows the shape.
Estimated per C1 window (T = 4, router x48 + indexer pair x12): about -105 us router, -62 us indexer; per C2 cycle
(T = 6): about -175 us router, -84 us indexer; C4 (T = 8): about -550 us router, -230 us indexer. Not measured in serving.

## On the round-2 pipelined regression in-engine (hypothesis, not tested)
The pipelined kernels launch exactly one group per resident slot (Xe-cores x 32 / threads-per-group at GRF256) and
walk a STATIC tile list (t += groups). On card 0 other queues run at the same time (EP push / copy kernels, the
ep_wait spin kernel, the shared-expert queue). Any slot they hold delays a persistent group's start, and that group
then finishes its whole fixed share late (no dynamic balancing), while the one-group-per-tile round-1 kernels are
balanced by the hardware dispatcher. GRF256 groups also cannot share an XVE's thread slots the way GRF128 groups do.
C1 (fewer concurrent kernels on the EP path) being neutral fits. Tests that would decide it: (a) the pipelined
kernels with an atomic dynamic tile counter instead of the static stride; (b) fewer persistent groups; (c) the probe
with a concurrent spin kernel on a second queue.
Knob added for (b): STRATA_NIGHT_MMVQ2_CAP_PCT (default 100 = unchanged; 0 = one group per tile). Isolated (card 1,
nothing concurrent) it only costs: Q4_K gate nc6 22.8 us at 100, 30.1 at 50, 33.7 at 0 (round 1: 26.0); nc8 24.3 /
33.1 / 38.2 (round 1: 35.8); outputs unchanged (same per-row arithmetic). If C2 serving improves at 50 or 0 despite
that, slot starvation by the concurrent EP kernels is the cause.

## Flags (all default off)
STRATA_NIGHT_MMVQ=1, STRATA_NIGHT_MMVQ2=1 (+ STRATA_NIGHT_MMVQ2_MASK), STRATA_NIGHT_SHEXP=1, STRATA_NIGHT_BF16=1
(STRATA_NIGHT_BF16_R=1|2|4|8 overrides the rows per tile, bench knob).
