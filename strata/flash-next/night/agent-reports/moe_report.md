# Night lab: EP MoE critical path (worktree `moe`)

Agent: Claude Opus 5.5 (claude-opus-5-5), 2026-10-16. Single-card fixtures on card 1 only (lease), no servers.
Worktree: /mnt/vm_8tb/b70/build/night-wt/moe/src-ep (branch night-20261016, base e1b7ee8a).
Patch: /mnt/vm_8tb/b70/results/night-20261016/agents/moe.patch. Bench: `build-sycl-ep/ep_night_bench`
(source sycl/src/kernels/ep_night_bench.cpp). Raw outputs: agents/moe_work/*.txt.

STATUS: final. Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-moe-ep2 = base e1b7ee8a + moe.patch
(NOT the SH_LATE / MtpPair commits that later moved this worktree's shared HEAD to ecd09e6f; the patch was generated
against e1b7ee8a and `git apply --check` passes on ecd09e6f). Flags, all default off, independent, latched, one stderr
line each: STRATA_NIGHT_EP_PSUM=1, STRATA_NIGHT_EP_ROWS=2, STRATA_NIGHT_EP_PLAN1=1. None was run in a two-card server
here (rules): the orchestrator must run the matched coherence / identity / teardown check.

## 1. Per-layer EP decode sequence (C2 verify window, 6 rows, cap = 60 entries; from trace cycle 5 layers 7/8)

card 0 (primary), after the post-attention HC:
| # | kernel | grid | us | data / signal |
| --- | --- | --- | ---: | --- |
| 1 | bf16_gemv_mmvf_multi (router logits) | 512 | 11.6-11.9 | BF16 2560x512 (2.6 MB) x 6 rows |
| 2 | native_router_top10_multi | 6 | 7.7-7.8 | ids/w 6x10 |
| 3 | resident_plan_ep role 1 | 1 WG | 6.3 | plan of card 0's own entries |
| 4 | quantize_q8_1_rows | 60 | 2.7-2.8 | 6 x 2560 -> 17.3 KB q8_1 |
| 5 | ep_push | 1 WG | 3.8 | ids 240 B + q8_1 17.3 KB over PCIe into card 1's inbox; then card 1's in-flag := epoch |
| 6 | routed_gu_port<TG> | 40 x 60 | 79-91 | own groups (hot experts, ~2 entries each) |
| 7 | swiglu_q8_1 (native_expert_grouped) | 150 | 3.3 | |
| 8 | routed_down_port<TD> | 80 x 60 | 54-68 | |
| 9 | ep_wait (1 work-item spin on the return flag) | 1 | 0-161 (mean 26 when waiting, 76% of layers) | runs beside the shared expert (forked stream, 7 kernels, ~53 us, starts only after #8) |
| 10 | native_moe_combine_multi | 5 x 6 | 1.8-2.2 | parts_ (60 rows: own + the peer's) |

card 1 (peer), in-order graph, ~2.6 us gap between kernels (x6 = ~13 us/layer):
| # | kernel | grid | us | data / signal |
| --- | --- | --- | ---: | --- |
| 1 | ep_wait (in-flag ring) | 1 | ~360 | |
| 2 | ep_copy_uncached | 18 | 1.4-1.6 | inbox slot -> local (17.5 KB) past the caches |
| 3 | resident_plan_ep role 2 | 1 WG | 5.4-5.5 | |
| 4 | routed_gu_port<TG> | 40 x 60 | 86-96 | |
| 5 | swiglu_q8_1 | 150 | 2.3-2.6 | |
| 6 | routed_down_port<TD> | 80 x 60 | 60-80 | |
| 7 | ep_send_rows | 60 | 29-30 | ~33 fp32 rows x 10 KB = ~330 KB over PCIe into card 0's parts_; last WG raises card 0's return flag |

Return latency send-end -> card 0 wake 1.4-2.9 us; push latency 3.0-4.75 us.

## 2. STRATA_NIGHT_EP_PSUM=1 (pre-summed return, item B)

CONFIG -> card 0 pushes the 60 routing weights with the ids (ep_push_w; inbox layout ids | w | q8_1 under the flag);
card 1 replaces ep_send_rows by ep_send_psum: per token, the weighted sum of its own entries' rows (entry order, first
product rounded, then FMA), ONE fp32 row per token (6 x 10 KB = 60 KB instead of ~330 KB) into a new card-0 EpRegion
(ep_psum_), same done-counter + flag protocol. Card 0's combine becomes ep_combine_psum: own entries (res0[id] >= 0)
in entry order + psum (read with uncached L1/L3 loads) + shared (plain or LFUSE-gated). Off: byte-identical original
path (separate kernels/functions; layout only changes when the flag is on). Requires K == 10 (else silently off).

COMMAND -> `ep_night_bench send 10` (card 1, single-card fixture: res0 = 145 of 512 experts, ~half the entries
own, random rows/weights; old = ep_send_rows + native_moe_combine_multi[_hits_gated]; new = ep_send_psum +
ep_combine_psum; the peer buffer local, and pinned host memory as a PCIe-write proxy).

RESULT (agents/moe_work/lanes.txt):
| T | combine | old vs new max abs | max rel | old vs fp64 ref | new vs fp64 ref |
| --- | --- | ---: | ---: | ---: | ---: |
| 6 | plain | 1.49e-08 | 5.3e-06 | 1.49e-08 | 1.49e-08 |
| 6 | gated | 1.49e-08 | 6.5e-06 | 1.49e-08 | 1.49e-08 |
| 4 | plain | 1.49e-08 | 3.5e-06 | 1.49e-08 | 1.49e-08 |
| 4 | gated | 1.49e-08 | 2.8e-06 | 1.49e-08 | 1.49e-08 |
| 1 | plain/gated | 1.49e-08 | 3.3e-06 | 1.49e-08 | 1.49e-08 |
(|out| max ~0.15-0.25: the difference is 1 ulp-level FP32 reordering; both paths are equally far from fp64.)

| T | peer entries | send old -> new, device-local | send old -> new, pinned host (PCIe proxy) | bytes | combine old -> new |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 33 | 2.78 -> 4.98 us | 242.0 -> 45.5 us | 330 -> 60 KB | 1.17 -> 3.75 (v1) / 4.36 (final) us |
| 4 | 20 | 2.08 -> 4.07 us | 146.4 -> 31.6 us | 200 -> 40 KB | 1.17 -> 3.91 (v1) / 4.35 (final) us |

The host-memory proxy is much slower than card-to-card P2P (trace: 29.5 us for ~330 KB), but it confirms the send
is byte-bound: 5.3x less time for 5.5x fewer bytes. At the trace's P2P rate (~11 GB/s) 60 KB is ~5.5 us, so the
card-1 chain shortens by ~22-24 us per layer. The new combine costs +3.2 us per layer on card 0 (not explained:
hoisting all loads and switching psum to plain loads - the contract parts_ already relies on; uncached kept behind
-DSTRATA_NIGHT_PSUM_UNCACHED=1 - did not help; the res0[id] gather is the remaining suspect; an own-mask from the plan
would remove it). 48 x 3.2 = 0.15 ms per cycle is charged below.

VERDICT -> correct to FP32 rounding (1.5e-8 abs, both paths equally close to fp64); expected saving is capped by
card 0's pure EP wait (1.25 ms / C2 cycle in the trace; card 0 waits a mean 26 us in 76% of layers): estimate
0.5-0.75 ms per C2 cycle before the combine charge, i.e. ~0.35-0.6 ms net; C1 (4 rows, 200 -> 40 KB): ~0.3-0.4 ms
net per window. Capped by the shared expert: card 0 runs it only after its own routed down (trace), so once card 1
is ~25 us faster card 0's serial routed + shared path (~203 us/layer) becomes the limiter. Needs a two-card
serving check (coherence + identity) by the orchestrator: the P2P region/flag protocol is the same as ep_send_rows'
but untested across cards here.

## 3. Routed expert kernels, standalone (card 1)

CONFIG -> `ep_night_bench grouped`: native_expert_grouped exactly as verify calls it (cap = rows x 10 groups/entries,
grid_groups 0), 48 layers per graph, each layer its own expert blobs (DRAM-streamed), random valid weights,
STRATA_GU_LANES=8 STRATA_IQ2XS_MULTI=1. Scenarios: C2 card0 = 14 groups / 30 entries (sizes 4,3x3,2x7,1x3),
C2 card1 = 26 groups / 30 entries (2x4,1x22), C1 card0 = 10 g / 20 e, C1 card1 = 18 g / 20 e. us per call;
GB/s = groups x 1.665-MiB-class blob bytes / call time.

RESULT (baseline, agents/moe_work/grouped_base.txt):
| gate/up / down | scenario | gu+swiglu us | down us | all us | GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| IQ2_XXS/Q2_0 | C2 card0 14g/30e | 99.0 | 60.0 | 158.1 | 116 |
| IQ2_XXS/Q2_0 | C2 card1 26g/30e | 105.1 | 86.6 | 190.6 | 178 |
| IQ2_XXS/Q2_0 | C1 card0 10g/20e | 73.7 | 41.5 | 115.2 | 113 |
| IQ2_XXS/Q2_0 | C1 card1 18g/20e | 76.1 | 60.0 | 135.4 | 174 |
| IQ2_XS/IQ4_NL | C2 card0 | 105.2 | 61.8 | 170.3 | 154 |
| IQ2_XS/IQ4_NL | C2 card1 | 111.1 | 93.6 | 205.8 | 236 |
| IQ3_S/Q2_0 | C2 card0 | 86.9 | 60.3 | 148.0 | 177 |
| IQ3_S/Q2_0 | C2 card1 | 127.7 | 87.1 | 213.9 | 227 |
| IQ2_S/Q2_0 | C2 card0 | 76.4 | 60.1 | 137.8 | 154 |
| IQ2_S/Q2_0 | C2 card1 | 108.5 | 86.8 | 196.5 | 200 |
| IQ3_XXS/IQ4_NL | C2 card0 | 79.6 | 62.5 | 145.7 | 209 |
| IQ3_XXS/IQ4_NL | C2 card1 | 114.9 | 94.2 | 209.7 | 270 |
(IQ2_XS rows above without STRATA_IQ2XS_MULTI; with it, C2 card0 139.7 / card1 198.7 us.)

Findings: (a) grid_groups = exact group count instead of cap changes nothing (dead work-groups are free);
(b) time tracks ENTRIES more than weight bytes (card0 14g/30e ~= card1 26g/30e for IQ2_XXS GU), i.e. the kernels
are bound by per-entry work (each lane re-loads a part's activation words - 8 x 4-byte loads + scale - for every
row), not by DRAM: 116-270 GB/s = 19-44% of 608 GB/s; (c) STRATA_DOWN_LANES 16/32 and STRATA_GU_LANES 16/32 are
all slower than 8/8; (d) staging the weight tile in local memory first (tried, STRATA_NIGHT_EP_STAGE, bitwise) is
0-30% SLOWER - weight-load latency is not the limiter.

RESULT (multi-row lanes, STRATA_NIGHT_EP_ROWS=2; agents/moe_work/final1.txt, same harness, bitwise check vs base):
Each lane computes 2 rows for the same lane parts, so a part's activation words are loaded once per 2 rows; every
(row, entry) keeps the k order, Multi::finish and lanes_sum<8>: BITWISE equal to the base kernels in all 20 cases
(76,800 / 51,200 output floats each). SIMD16 (SIMD32 spilled 40-60 regs; 4 rows spilled 100-290).
| gate/up / down | scenario | base gu+sw / down / all | R2 gu+sw / down / all | all delta |
| --- | --- | --- | --- | ---: |
| IQ3_S/Q2_0 | C2 card0 | 84.6 / 60.0 / 147.8 | 76.0 / 58.3 / 131.4 | -11% |
| IQ3_S/Q2_0 | C2 card1 | 124.3 / 88.2 / 214.2 | 114.3 / 83.6 / 193.7 | -10% |
| IQ3_S/Q2_0 | C1 card0 | 62.5 / 41.6 / 106.6 | 53.5 / 40.8 / 93.2 | -13% |
| IQ3_S/Q2_0 | C1 card1 | 85.6 / 61.0 / 148.9 | 77.7 / 59.5 / 134.8 | -9% |
| IQ2_S/Q2_0 | C2 card0 / card1 | 137.4 / 194.6 | 126.4 / 183.7 | -8% / -6% |
| IQ3_XXS/IQ4_NL | C2 card0 / card1 | 144.8 / 209.7 | 133.0 / 194.2 | -8% / -7% |
| IQ3_XXS/IQ4_NL | C1 card0 / card1 | 103.3 / 147.3 | 93.4 / 133.5 | -10% / -9% |
| IQ2_XS/IQ4_NL (GU not taken, down only) | C2 card0 / card1 | 140.5 / 198.5 | 135.4 / 191.0 | -4% / -4% |
| IQ2_XXS/Q2_0 (no Multi<16>: down only) | C2 card0 / card1 | 158.9 / 191.8 | 155.2 / 185.4 | -2% / -3% |
Down alone: Q2_0 -3..-6%, IQ4_NL -6..-12%. IQ2_XS gate/up with 2 rows was 1.7-2x SLOWER (excluded in code);
4 rows slower than 2 (removed). Control: the unchanged IQ2_XXS / IQ2_XS gate/up kernel reads +3-5 us in the R2 column
(run order / clock drift), so the R2 gains are if anything understated.

VERDICT -> keep as opt-in: bitwise, -2..-13% per call depending on the layer's formats. Weighted over this model's 48
layers (GU: 13x IQ3_S, 10x IQ2_S, 6x IQ3_XXS, 9x IQ2_XS, 9x IQ2_XXS; down: 30x Q2_0, 18x IQ4_NL): card 1 chain
~-0.45 ms and card 0 own experts ~-0.3 ms per C2 cycle. The kernels remain ALU/issue-bound (19-44% of DRAM BW):
the real fix is a decode-once dequant (or a Multi<16> for IQ2_XXS, which re-decodes per entry today), not tiling.

## 4. STRATA_NIGHT_EP_PLAN1=1 (card 0 makes card 1's plan, item C part)

CONFIG -> resident_plan_ep_pair: one launch on card 0, work-group 0 = role 1 (card 0's plan, as before), work-group 1 =
role 2 with card 1's arena base and a card-0 copy of card 1's slot offsets (generate.cpp copies them under the flag);
ep_push_w carries the 406 plan words (1.6 KB) in the inbox slot before the q8_1 rows; card 1's ep_copy_uncached (same
call, longer span) brings them in and card 1 SKIPS resident_plan_ep (5.4-5.5 us + one in-order gap ~2.6 us).

COMMAND -> `ep_night_bench plan 10`: 200 random cases (1-8 rows, 100-200 experts on card 0, 20 on both cards, hot-set
repeats, with and without slot offsets) vs resident_plan_ep role 1 and role 2 called separately.

RESULT -> 200/200 cases: every one of the 406 words equal for both roles (counts, start, dst, tok, ptr). Launch cost
in a graph: role-1 only 4.68 us, pair 4.70 us (the second work-group runs beside the first).

VERDICT -> ~-8 us per layer on card 1's chain (~-0.38 ms per C2 cycle while card 1 is critical), +~0.2 us on card 0
(1.6 KB more over PCIe). Two-card check needed (the pushed pointers are card 1 addresses computed on card 0).

## 5. Rejected

- STRATA_NIGHT_EP_STAGE (weight tile staged in local memory, 16-byte loads, then the same dots; bitwise): 0-30% slower
  (e.g. IQ3_S C2 card1 GU 124 -> 163 us). Removed from the patch.
- grid_groups = live groups (no dead work-groups): no change (dead groups exit for free).
- STRATA_GU_LANES / STRATA_DOWN_LANES 4/16/32: all slower than 8/8 at these shapes (agents/moe_work/lanes.txt).
- 4 rows per lane: spills, slower than 2.

## 6. Estimated effect (per layer -> per C2 cycle, 48 layers; unprofiled cycle ~46.5 ms)

Model from the trace: per layer the EP section ends at max(card 0 own routed ~150 us + shared expert ~53 us (it starts
only after own down), card 1 chain ~219 us (push latency, copy, plan, GU, swiglu, down, send, 6 in-order gaps)).
| change | card 1 chain | card 0 path | per layer | per C2 cycle |
| --- | ---: | ---: | ---: | ---: |
| baseline | ~219 us | ~203 us | 219 | - |
| PSUM | -23 | +3.2 (combine) | 206.2 | -0.6 ms |
| + PLAN1 | -8 | +0.2 | 206.4 (card 0 now limits) | ~0 more |
| + ROWS=2 | -9.5 | -6.5 | ~200 | -0.3 ms |
| all three | ~179 | ~200 | ~200 | ~-0.9 ms (~2%) |
C1 window (4 rows): same structure, smaller rows; expect ~-0.5 ms per window. After these, card 0's serialized shared
expert (~53 us/layer, ~2.5 ms/cycle) is THE critical item: the next lever is running it concurrently with card 0's own
routed GU (graph order) or giving it to card 1 together with PSUM (card 1 has ~40 us of slack per layer after these
changes). Second lever: a Multi<16> (IQ2_XXS, 9 layers) and a decode-once expert kernel.

## 7. Flags and files

- STRATA_NIGHT_EP_PSUM=1: ep_push_w / ep_send_psum / ep_combine_psum (ep_kernels.*), verify.cpp ep_init/pre/post.
- STRATA_NIGHT_EP_ROWS=2: native_gu_kernel_rows / native_down_kernel_rows (iq_kernels.dp.cpp), Multi formats except
  IQ2_XS gate/up; any native_expert_grouped call (card 0 own, card 1, remote experts).
- STRATA_NIGHT_EP_PLAN1=1: resident_plan_ep_pair (verify_kernels.dp.cpp), VerifyEp::slot_off1_on0 (generate.cpp).
- Bench / parity: ep_night_bench {grouped|send|plan} (ninja target ep_night_bench; nl_wt.sh ninja moe ep_night_bench).
- Suggested serving A/B: ep-stack1 config + STRATA_NIGHT_EP_PSUM=1 STRATA_NIGHT_EP_PLAN1=1 STRATA_NIGHT_EP_ROWS=2 on
  strata-moe-ep2, one flag at a time if the stack fails coherence.

## 8. IQ decode, instruction level (STRATA_NIGHT_IQ=1), second mission

FINAL engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-moe-iq2 (base e1b7ee8a + agents/moe_iq.patch =
the EP patch + this). Incremental patch on top of the committed EP patch (7fae90ec): agents/moe_iq_incremental.patch
(iq_kernels.dp.cpp, iq_kernels.hpp, new night_tables.inc, ep_night_bench.cpp).
Flags: STRATA_NIGHT_IQ=1 together with STRATA_NIGHT_EP_ROWS=2 (all 7 expert formats; without ROWS=2 only IQ2_XXS
gate/up and the two down formats switch, the others measured slower at 1 row). Sub-switches for A/B only:
STRATA_NIGHT_IQ_FMTS=<ggml ids>, STRATA_NIGHT_IQ_ALU=0 (table sign masks for IQ3_XXS/IQ3_S/IQ2_S: slower),
STRATA_NIGHT_IQ4NL_TAB=0 (bit-slice IQ4_NL: ~4% slower).

### 8.1 ISA method and where the time goes

CONFIG -> relinked ep_night_bench without the ocloc cache and with IGC_ShaderDumpEnable=1
IGC_DumpToCustomDir=<dir> (agents/moe_work/isa_link.sh; AOT bmg-g31, 26 s), 511 kernels; demangled index
agents/moe_work/isa_index.txt, loop classifier agents/moe_work/isa_loops.py (counts instruction classes per
backward-branch loop body). Production kernels (GU_LANES=8, EXPERT_SPLIT=0): native_gu_port<TG,8,1> and the IQ2_XS
native_gu_iq2xs_multi_s100 kernel; native_down_port<TD,8,1>. (native_gu_multi_kernel / row_dot_80_sub16 are the
STRATA_EXPERT_SPLIT=1 path, not served.) All SIMD32, 128 GRF, no spills.

Findings (single-entry inner loop = one lane-part: 32 weights for the IQ/Q2_0 formats, 16 for IQ4_NL):
- Tables: iq2xxs/iq2xs/iq2s/iq3xxs/iq3s grids are dpct::global_memory -> global memory; every lookup is a
  load.ugm.d64/d32.a64 with a 64-bit address (shl + 2 x add.q per SIMD32) - 4 lookups per part (IQ2) or 8 (IQ3).
  ksigns is computed (cbit popcount + 32-bit multiply by 0x01010101 = mul+macl pairs), not looked up.
- Sign expansion swar_ne4 (or/shift/and chain -> bfn + shr) + mul 255, and swar_sub4, per 4 weights: ~10 ALU.
- Addressing: kbx * sizeof(block) and kbx * 36 recomputed each iteration as 32x32->64 products (mul/mach/macl) +
  64-bit adds: ~30 instructions per part; weights read as 16-bit loads (get_int_b2: 2 sends per int).
- Q2_0 down: the dpct emulation of byte_level_permute (16 per part) dominates: 476 instructions per part, mov 157,
  shr 89, and 73, shl 56 - for 8 DP4As.
- IQ4_NL down: iq4nl_lut4's 3 cmp + 3 sel per nibble: cmp 50 + sel 48 of 279 per 16 weights.
- DP4A is native (dp4a instruction) but is 2-3% of the loop.
- Cost model check: kernel time ~= loop instructions / (~300 G SIMD32 instructions/s): e.g. Q2_0 down at C2 card1
  (1.54 M part-dots x 476 / 32 lanes = 23 M instr -> 76 us predicted, 87 us measured); IQ2_XXS GU 90 vs 102 us;
  IQ4_NL down 89 vs 93 us. The expert kernels are ALU (issue) bound, 19-44% of DRAM bandwidth.

| kernel (production) | loop instr per part | dp4a | loads | notes |
| --- | ---: | ---: | ---: | --- |
| IQ2_XXS GU, 1 entry (no Multi: every entry re-decodes) | 282 | 8 | 12 | mul+macl+mach 42, bfn 40, cbit 4 |
| IQ2_XS GU (iq2xs multi), 1 / 2 entries | 302 / 344 | 8 / 16 | 13 / 16 | |
| IQ3_XXS GU, 1 / 2 entries | 304 / 375 | 8 / 16 | 18 / 19 | 8 grid lookups |
| IQ3_S GU, 1 / 2 entries | 317 / 371 | 8 / 16 | 20 / 21 | |
| IQ2_S GU, 1 / 2 entries | 300 / 366 | 8 / 16 | 14 / 17 | |
| Q2_0 down, 1 / 2 / 4 entries | 476 / 501 / 546 | 8 / 16 / 32 | 8 / 11 / 17 | byte_level_permute emulation |
| IQ4_NL down, 1 / 2 / 4 entries (16 weights) | 279 / 302 / 338 | 4 / 8 / 16 | 6 / 9 / 15 | cmp 50 + sel 48 |

### 8.2 Changes (all bitwise: same integers per part, same finish / float order)

NQ<TY> decode traits + native_gu_kernel_n / native_down_kernel_n (1 or 2 rows per lane):
- loop over block pointers (LANES % ipb == 0: a lane's part index is fixed), no per-iteration 64-bit products;
- 8-byte aligned-word loads (load8_a2) instead of 16-bit loads;
- ksigns/sign bytes -> 16-byte masks from a 2 KB / 4 KB table {m_lo, m_hi, m_lo & 0x01.., m_hi & 0x01..};
- signed grid bytes (g ^ m) + (m & 0x01010101) (exact: every grid byte is 1..62, checked);
- Q2_0: fields spread by shifts/masks, (t + 0x7F7F7F7F) ^ 0x80808080 (= the {-1,0,1,2} permute lookup);
- IQ4_NL: 4-level bit-slice select (bfn) instead of cmp/sel;
- IQ2_XXS gets a multi-entry path (decode once per part for up to 4 entries).

| kernel (new, 1 row / SIMD32) | loop instr per part (1 / 2 / 4 entries) | was |
| --- | --- | --- |
| IQ2_XXS GU | 179-210 / 192-256 / 254-300 | 282 / 564 / 1128 (re-decode) |
| IQ2_XS GU | 189-233 / 224-287 / 306-348 | 302 / 344 / - |
| IQ3_XXS GU | 227-278 / 244-320 | 304 / 375 |
| IQ3_S GU | 246-308 / 268-356 | 317 / 371 |
| IQ2_S GU | 224-318 / 241-323 | 300 / 366 |
| Q2_0 down | 142-203 / 152-228 / 212-278 | 476 / 501 / 546 |
| IQ4_NL down | 236 / 254 / 297 | 279 / 302 / 338 |
(two values = the two SIMD32 compiles IGC emitted; the new 1-row SIMD32 kernels spill 7-30 registers; the 2-row
SIMD16 kernels used by default do not spill. IQ4_NL with tables (default) and the ALU-sign IQ3 forms were not
re-dumped.)

### 8.3 Timings (card 1, ep_night_bench grouped, 48 layers/graph, us per call; bitwise equal in every case)

agents/moe_work/iq1a.txt, iq1b.txt. base = current kernels; R2 = ROWS=2 (old decode); IQ = new decode 1 row;
IQR2 = new decode 2 rows. gu = gate/up+swiglu, dn = down.

| GU / down | scenario | base gu / dn / all | IQR2 gu / dn / all | all delta |
| --- | --- | --- | --- | ---: |
| IQ2_XXS/Q2_0 | E1 30g x1 | 105.8 / 103.3 / 209.1 | 94.3 / 39.2 / 133.5 | -36% |
| IQ2_XXS/Q2_0 | E2 15g x2 | 112.9 / 58.4 / 171.3 | 59.0 / 27.0 / 86.0 | -50% |
| IQ2_XXS/Q2_0 | E3 10g x3 | 110.5 / 68.6 / 179.1 | 68.6 / 27.5 / 96.1 | -46% |
| IQ2_XXS/Q2_0 | C1 card0 / card1 | 114.9 / 135.4 | 60.3 / 84.3 | -48% / -38% |
| IQ2_XXS/Q2_0 | C2 card0 / card1 | 159.1 / 192.9 | 85.8 / 121.2 | -46% / -37% |
| IQ2_XXS/Q2_0 | C4 card0 / card1 | 213.8 / 249.2 | 111.3 / 147.1 | -48% / -41% |
| IQ2_XS/IQ4_NL | C2 card0 / card1 | 138.3 / 197.3 | 120.1 / 173.5 | -13% / -12% |
| IQ2_XS/IQ4_NL | C4 card1 | 242.0 | 209.9 | -13% |
| Q2_0 down alone | C2 card0 / card1 | 63.7 / 92.3 | 26.4 / 35.2 | -59% / -62% |
| IQ4_NL down alone | C2 card0 / card1 | 64.9 / 95.9 | 55.8 / 80.3 | -14% / -16% |
| IQ3_S/Q2_0 (GU new is slower; dn only) | C2 card1 | 216.4 | 158.7 (GU old R2 ~114 + dn 34 -> ~148) | -27..-32% |

First pass: IQ3_S, IQ3_XXS, IQ2_S gate/up with TABLE sign masks were 5-20% SLOWER than the current kernels despite
10-25% fewer instructions (4 more gathers per part, cheap ALU signs replaced; SIMD32 spills at 1 row). Second pass
(NQ<118/121/122>): block-pointer loop + aligned loads but ggml's ALU sign masks: 1-6% faster than ROWS=2 alone
(iq2.txt) -> default. IQ4_NL through 4 x 256 x 8-byte position tables (NQ<120>, 4 gathers per 8 weights): ~4% faster
than the bit-slice select -> default (cheap gathers win when they replace ~40 ALU; lose when they replace ~6).

### 8.4 Final default configuration (STRATA_NIGHT_IQ=1 + STRATA_NIGHT_EP_ROWS=2), bitwise in all 30 cases

agents/moe_work/iq3a.txt, iq3b.txt (us per call; columns run base, R2, IQ, IQR2 in that order; unchanged kernels
read 5-8% slower in later columns - clock drift - so the IQR2 gains are conservative):
| GU / down | C1 card0 | C1 card1 | C2 card0 | C2 card1 | C4 card0 | C4 card1 |
| --- | --- | --- | --- | --- | --- | --- |
| IQ2_XXS/Q2_0 base -> IQR2 | 114.8 -> 59.8 | 135.1 -> 83.8 | 158.7 -> 85.8 | 192.3 -> 121.5 | 213.1 -> 112.3 | 248.7 -> 147.3 |
| IQ3_S/Q2_0 | 105.1 -> 70.3 | 147.9 -> 100.1 | 147.4 -> 99.3 | 215.2 -> 147.2 | 194.8 -> 130.8 | 256.9 -> 175.1 |
| IQ2_S/Q2_0 | 97.0 -> 67.2 | 135.5 -> 91.5 | 136.4 -> 94.5 | 195.5 -> 133.1 | 181.1 -> 123.1 | 237.3 -> 161.9 |
| IQ2_XS/IQ4_NL | 99.2 -> 82.8 | 135.6 -> 119.6 | 138.1 -> 117.7 | 196.7 -> 169.9 | 181.4 -> 151.6 | 242.6 -> 205.4 |
| IQ3_XXS/IQ4_NL | 102.9 -> 88.9 | 146.6 -> 127.9 | 142.7 -> 126.0 | 208.2 -> 183.9 | 189.7 -> 162.9 | 253.4 -> 221.3 |
Q2_0 layers -31..-48%, IQ4_NL layers -12..-17% per call; decode GB/s 115-275 -> 220-340.

VERDICT -> keep: bitwise, every format faster. Layer-weighted vs ROWS=2 alone (30 Q2_0 + 18 IQ4_NL down; GU 9 IQ2_XXS,
9 IQ2_XS, 6 IQ3_XXS, 13 IQ3_S, 10 IQ2_S): C2 card 1 chain ~-2.3 ms per cycle (Q2_0 down alone -52 us x 30), card 0
own experts ~-1.7 ms per cycle. With card 0 (own experts + the serialized ~53 us shared expert) the limiter after the
EP patch, expect ~-1.5 to -1.8 ms per C2 cycle (~3.5% of 46.5 ms) and more at C4 (IQ2_XXS multi-entry -47%, 4-entry
Q2_0 passes). The remaining expert time is now IQ4_NL down (~78 us/call card 1 C2) and the IQ3/IQ2_S gate/up
(~100-110 us card 1 C2, ~2-4% better than ROWS=2): next levers are fewer grid gathers (8 per IQ3 part) and the
IQ4_NL down at 16 weights per part. Serving check (coherence, identity, C2/C4 throughput) needed.

## 9. Third mission: GDN verify kernels and the router top-10 (card 0 latency-bound per-layer kernels)

Base: main tree 60b7f900 (worktree reset to it; my earlier work is in it). Engine:
/mnt/vm_8tb/b70/results/night-20261016/builds/strata-moe-gdn1 = 60b7f900 + agents/moe_gdn.patch (applies on 60b7f900).
Flags (default off, independent, latched, one stderr line each): STRATA_NIGHT_GDN_AB=1, STRATA_NIGHT_TOPK10=1,
STRATA_NIGHT_GDN_GRF=1. All bitwise (checked below). Bench: ep_night_bench {router|gdnab|gdnstep}.

### 9.1 ISA findings (relinked ep_night_bench with IGC_ShaderDumpEnable, agents/moe_work/isa3, isa4)

- native_router_top10_multi (route<512>): 2,770 instructions, fully unrolled, one sub-group per token: mov 1,517 -
  every dpct::experimental::permute_sub_group_by_xor (and sycl::permute_group_by_xor too: same code) is lowered to an
  a0-indirect gather (mov a0 / add a0 / mov r[a0] per 16 lanes) - plus 117 goto/join pairs from the short-circuit
  tie rule `other > best || (other == best && ...)` and the 15-step serial per-lane argmax. A latency chain on one
  sub-group per token (the card is otherwise idle), so instruction count and dependency depth are the cost.
- gdn_ab_multi: one sub-group per output row (96 rows: 96 sub-groups on a 32-core card) looping over the T tokens:
  T accumulator chains in one lane, 1,511 instructions at T=6.
- gdn_step_norm_multi_kernel (verify recurrence): every variant SPILLS at 128 GRF: 704 B (per-slot <true,false>),
  1,152 B (q8 <true,true>), 832 B (GDN_SLOTS launch), 8.9-9.7 KB (the commit-side <false,*> variants, not used when
  STRATA_GDN_COMMIT_SPLIT routes commits to gdn_step_commit, which does not spill). 32 state rows x SIMD32 = 64 GRF of
  state per thread alone; per-t loop 463 instructions incl. 44 loads and spill sends.

### 9.2 Results (card 1, graph replay; bitwise in every case)

CONFIG/COMMAND -> agents/moe_work/rg1.txt, rg3.txt, gs1.txt.

| kernel | change | check | T=1 | T=4 | T=6 | T=8 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| router top-10 (STRATA_NIGHT_TOPK10) | per-lane best by 4-level tree (first max = lower id, as the scan), no dynamic register indexing (winner lane clears by unrolled selects, recomputes only its best), branch-free (value, id) tie rule; same softmax prologue, same xor trees, same per-lane selected_sum order | 1,800 token rows (random, many exact ties, flat, 40 tied leaders): ids + weights memcmp-equal | 7.21 -> 5.35 us | 7.16 -> 5.20 | 7.17 -> 5.20 | 7.17 -> 5.19 |
| gdn_ab_multi (STRATA_NIGHT_GDN_AB) | one sub-group per (row, token): T x more sub-groups; each lane's chunk order / FMA chain / xor tree / epilogue unchanged; 2 chunks' loads issued ahead | T=1..8 x 3 weight sets: gate + beta memcmp-equal | 4.25 -> 3.76 us | 6.03 -> 3.32 | 7.83 -> 3.69 | 15.68 -> 4.06 |
| GDN verify recurrence, per slot (STRATA_NIGHT_GDN_GRF) | same kernel with grf_size<256> (no spills) | y and the q8_1 image memcmp-equal, T=2..8 | - | 14.74 -> 13.38 us (T=3: 12.87 -> 11.75) | 18.46 -> 16.62 | 22.20 -> 19.88 |

GDN_GRF on the 2-slot GDN_SLOTS launch was 1-12% SLOWER (halved occupancy for 96 x 512-thread groups), so the flag
leaves that launch alone; it applies to the per-slot path (C1 windows; C2/C4 with GDN_SLOTS keep their launch).

Per window/cycle estimates (trace_best3 counts):
- TOPK10: -2.0 us x 48 layers = -0.10 ms per C1 window and per C2 cycle (+ 3-4 MTP calls: -0.006 ms).
- GDN_AB: trace 9.2 us (C1, T=4) / 11.9 us (C2, T=6) per call x 36 -> ~3.3-3.7 us: -0.2 ms per C1 window,
  -0.3 ms per C2 cycle (the bench's default is faster than the trace, so the in-graph saving may be larger).
- GDN_GRF: -1.4 us x 36 = -0.05 ms per C1 window.
Total ~-0.35 ms per C1 window (~1.6% of ~22.4 ms), ~-0.4 ms per C2 cycle.

### 9.3 Not done / next

- gdn_step_commit (72 x 15.9 us per C2 cycle, 1.14 ms; 377 GB/s): its 768 SIMD32 threads (48 heads x 128 cols x 4 row
  groups) fit in one wave, so load (3 MB), recurrence and store (3 MB) phases serialize; the two slots' commits run
  back to back in the in-order commit graph. A one-launch commit for all slots (as GDN_SLOTS did for verify) would
  overlap them: estimated ~-9 us per layer, ~-0.3 ms per C2 cycle. Not started: the engine already has a batched
  commit (gdn_commit_layers, all 36 layers per owner in one launch, STRATA_GDN_COMMIT_LAYERS, S145 "-26% private",
  on HOLD at 0 in the serving config) - re-qualifying that is the cheaper route than a new slot-merged commit.
- route_night<256> (the 256-expert model path) compiles and follows the same rule but was not exercised (model: 512).
- Router BF16 GEMV 2560->512 (11.9 us, 36-44% of roofline): not attempted (hc found row-per-lane slower).
- The xor shuffle lowering (a0-indirect, ~6 instr per 16 lanes) affects every warp reduction in the engine
  (lanes_sum, warp_sum, GDN norms, quantizers); a native sub-group reduce or SLM tree may be cheaper in the
  latency-bound kernels - worth an ISA experiment.

## 10. Fourth mission: 12-row verify windows (STRATA_NIGHT_VERIFY_T12=1) for C4 x depth 2

Base: main dc023c67 (worktree reset to it). Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-moe-t12a
= dc023c67 + agents/moe_t12.patch (applies on dc023c67; main's uncommitted generate.cpp change is NOT included).
Flag: STRATA_NIGHT_VERIFY_T12=1 (runtime, latched, one stderr line). Off: every new branch is behind it (and > 8
rows), host arrays are only larger, the window cap stays 8, so C1/C2/C4-depth1 run the same code (bitwise).

### 10.1 What limited the window to 8 rows (audit)

- Engine caps: kVerifyMaxT = 8 for batch_pack's window, the verifier's max_t (batch MTP), Verifier::init's check
  (also cpu::MAXT = 8, the CPU expert pool's arrays), host arrays in generate.cpp's batch loop (rows/tok/outb/pos/
  first/active/glen/pend, BSlot::draft), the sampler history (kHistSlots), run_slots/batch_launch rows[8],
  ple_rows[kVerifyMaxT * PLE_N_HEADS] (x2), pl_ple_rows_, and FusedGrArgs fa[kFusedGrMaxT] (x4: the per-layer HC read
  and three head mixers).
- Kernel entry points that throw / exit above 8 rows: native_mmvq (validate_shape, MAX_NCOLS = 8; only the KREG
  formats Q4_K/Q5_K/Q6_K/IQ4_XS take 9-16 columns, with STRATA_NIGHT_MMVQ2 bit0), native_quantize_q8_1 (the same
  validate_shape), bf16_gemv_fp32_mmvf_multi (router, indexer, PLE, shared gate: 1..8), shared_expert_multi (1..8),
  fused_gr_read_multi (kFusedGrMaxT = 8; HC v2 inside), gdn_ab_multi and gdn_conv_l2_multi (kVerifyMaxT).
- Already fine at 12 rows: router top-10 (any n), resident_plan* (120 entries <= kResidentPlanMax 128; 16 rows would
  not fit), native_expert_grouped and the night IQ kernels (any entries per group), EP push/psum/plan1/combine (sized
  by max_t_), native_moe_combine_multi, gr_write_multi, quantize_q8_1_rows, norm/rope (rows <= 65535), QSA
  attention (per slot / qsa_decode_attn_multi any n_q, 4 pools = 4 slots), GDN_SLOTS (4 slots x <= 8 rows), sampler,
  every buffer allocated from max_t_. gdn_commit_layers (S145) requires MT == 8 and turns itself off at 12 (it is
  off in the serving config anyway).

### 10.2 Changes

- verify_kernels.hpp: kVerifyCapT = 16 (host array capacity), verify_t12(), verify_max_t() (12 with the flag).
- generate.cpp: batch arrays -> kVerifyCapT; batch_pack and the batch-MTP verifier max_t -> verify_max_t().
- verify.cpp: init accepts up to verify_max_t(); more than cpu::MAXT rows only with expert parallel (no CPU pool);
  the fixed arrays above -> kVerifyCapT.
- Chunking at the entry points (only when > 8 rows and the flag): rows [0, 8) then [8, n) with offset pointers:
  native_mmvq (KREG formats first try the whole 12 columns - one weight read - and chunk only if not taken),
  native_quantize_q8_1, bf16_gemv_fp32_mmvf_multi, shared_expert_multi (its q8 scratch reused, pre-quantized input
  offset), fused_gr_read_multi (not with STRATA_QFUSE counters), gdn_ab_multi, gdn_conv_l2_multi (second half with
  t_begin = 8: same history/qkv buffers).

### 10.3 Checks

CONFIG -> `STRATA_NIGHT_VERIFY_T12=1 ep_night_bench t12` on card 1, with the default MMVQ kernels and again with
STRATA_NIGHT_MMVQ=1 MMVQ2=1 SHEXP=1 GDN_AB=1 (agents/moe_work/t12a.txt).
RESULT -> for every entry point: the 12-row call is memcmp-equal to the same entry called on rows [0, 8) and
[8, 12) separately (offsets right); native_mmvq for Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, IQ4_NL, Q8_0, Q2_0 (2560 ->
1024) is also equal to two 6-row calls (max rel diff 0: the column arithmetic does not depend on the call width,
also for KREG taking all 12 columns). Engine builds and links; flag-off path unchanged by construction.
NOT RUN: a whole 12-row window against two 6-row windows (needs the model on both cards = a server run).

VERDICT -> ready for the serving coherence check: C4, --batch 4, STRATA_BATCH_MTP=1 STRATA_BATCH_MTP_DEPTH=2
(+ the rest of the night stack) + STRATA_NIGHT_VERIFY_T12=1: batch_pack should now take all four 3-row slots in one
12-row window. Watch the engine log for "verify: the window must hold" (init) and for VRAM (max_t 12 enlarges the
per-window buffers by ~1.5x, roughly +20-40 MB on card 0). Rows 0-7 / 8-11 chunking means MMVQ IQ3_S / IQ4_NL / Q8_0,
the router / indexer BF16 GEMVs, the shared expert and the HC read read their weights twice in a 12-row window; the
KREG formats (most dense bytes) read them once.

### 10.4 Fix t12b (startup abort "native MMVQ requires 1 <= ncols <= 8")

Cause (found by listing every validate_shape caller): Verifier::init sizes xq_ / sh_xq_ with
native_q8_1_bytes(max_in, T), T = max_t_ = 12, and that size helper ran validate_shape(ncols <= 8) - an uncaught throw
at init. Two more startup paths would have exited next: Verifier::warm() captured one-request windows T = 1..max_t_
(9..12 run the non-batch path, whose gdn_step_norm_multi exits above 8 rows), and capture_commit() recorded the
one-request commit with n_tok = MT = 12 (same exit).
Fix (agents/moe_t12b.patch, on main 5fb1411f): native_q8_1_bytes accepts up to kVerifyCapT columns with the flag;
warm() captures one-request windows only up to kVerifyMaxT (they never exceed --spec <= 8; batch windows capture on
first use); capture_commit passes min(MT, 8) rows (n_keep <= T <= 8 there). Diagnostic: with the flag, validate_shape
prints "strata night T12: validate_shape(n_in, ncols, block) is about to throw (caller %p)" (first 4; not for the
expected, caught whole-width try in native_mmvq). Engine: builds/strata-moe-t12b. Flag-off behavior unchanged.

### 10.5 Fix t12c (SIGSEGV after the first 12-row window capture)

Cause: Verifier members still sized 8 and written per row of a batch window: brow_[8] (capture_batch, the slot of
each row), last_rows_[8], last_tokens_[8], last_pos_b_[8] (stage_batch / commit_slot_prefixes / sample_rows) and
b_out_[8] (both include/ and sycl/include verify.hpp). The 9-row window already overflowed them by one (stack-free
member overflow into the next members, silent); the 12-row one wrote 4 past and crashed. Fix: all five -> 16
(kVerifyCapT), plus a host guard in capture_batch (before recording) and stage_batch that prints
"strata night T12: ... with N rows exceeds the host row arrays" and fails the window instead of overflowing.
Audit of the rest of the per-window host path: generate.cpp batch_step arrays (rows/tok/outb/pos/first/active/glen/
pend, BSlot::draft) were already kVerifyCapT; the MtpPair args are per pair (2); exec_/ep_exec_/groups_/seg_[9] are
indexed by one-request window size only (<= 8 after the warm() cap); mapped buffers, commitb_ (CB = 2 + max_t_) and
hist snapshots are sized from max_t_; mtp.hpp's [8] arrays are the drafter's chain (<= --mtp-max-t).
Patch agents/moe_t12c.patch (on main f498be11), engine builds/strata-moe-t12c.
