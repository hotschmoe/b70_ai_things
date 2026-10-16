# prefill agent report (night 2026-10-16) - EP prefill

Worktree: /mnt/vm_8tb/b70/build/night-wt/prefill/src-ep (branch night-20261016).
Probe: sycl/src/kernels/night_ep_expert_bench.cpp (ninja target night_ep_expert_bench), run on card 1 only via
/mnt/vm_8tb/b70/results/night-20261016/agents/prefill_work/run_probe.sh. Raw outputs: prefill_work/probe{1..4}.txt.

Index (all flags default off; final engine for all three: see section 9 / the summary):
- Mission 1 (sections 1-5): STRATA_NIGHT_PREFILL_EARLY (doorbell instead of the host drain), STRATA_NIGHT_PREFILL_F16SUM
  (FP16 peer sums). Served: control 4K 1175/1288, 32K 1640/1605; EARLY 1207/1325, 1672/1639; +F16SUM 1213/1342,
  1746/1711 tok/s (orchestrator).
- Mission 2 (sections 6-7c): STRATA_NIGHT_PREFILL_GROUPED (+ _MAX): grouped IQ-decode + XMX expert kernel, one launch per
  projection for all small experts of a layer on each card.
- Mission 3 (sections 8-9): STRATA_NIGHT_PREFILL_HALVES: two-half per-layer pipeline across the cards.

## 1. Card-1 expert loop microbenchmark (first result, 07:40)

CONFIG -> card 1 alone, random IQ blobs (d = 0.0625 per block), FP16 activations N(0, 0.25), synthetic row counts
(lognormal): "12288" = T 12288, 245 experts, 66,965 rows, 142 experts <= 64 rows; "4K" = T 3783, 261 experts,
27,144 rows, 183 experts <= 64 rows. Gate/up IQ3_S + down Q2_0 unless noted. GPU time = wall(busy kernel + loop)
- wall(busy kernel), so the whole loop is queued before the GPU reaches it.

COMMAND -> run_probe.sh night_ep_expert_bench "12288 245 60 1.88 21 42 64 3" "3783 261 40 1.5 21 42 64 3"

RESULT (ms per layer-chunk of card-1 work, best of 3):

| variant | launches | host enqueue | wall | GPU only | vs ORIG numerics |
| --- | ---: | ---: | ---: | ---: | --- |
| 12288 ORIG (engine today: 5 launches/expert) | 1225 | 13.0 | 17.4 | 17.3 | - |
| 12288 GRP (<=64-row experts in iq_xmx_grouped, 16/launch) | 543 | 5.5 | 19.4 | 19.2 | small experts rel 5.8e-5 of max |
| 12288 WAVE (1 gather + 1 grouped dequant + 1 swiglu per 16 experts, oneMKL per expert) | 544 | 8.6-8.9 | 19.8 | 19.9 | bitwise equal |
| 12288 PAR2 / PAR3 (ORIG split over 2 / 3 host threads + queues) | 1225 | 13.3 / 11.8 | 19.7 / 19.6 | - | bitwise equal |
| 4K ORIG | 1305 | 14.5 | 14.6 | 12.1 | - |
| 4K GRP | 431 | 4.5 | 14.4 | 14.7 | rel 5.2e-5 |
| 4K WAVE | 579 | 9.5 | 15.2 | 15.3 | bitwise equal |
| 4K PAR2 / PAR3 | 1305 | 14.4 / 12.6 | 14.5 / 15.2 | - | bitwise equal |

Per-launch host cost (launches queued behind a busy kernel) and back-to-back time, M = 32:

| launch | host us | back-to-back us |
| --- | ---: | ---: |
| gather_rows16 | 3.8 | 3.8 |
| iq_dequant_expert_f16 (IQ3_S/Q2_0) | 4.5 | 10.3 |
| iq_dequant_expert_f16 (IQ2_XXS/IQ4_NL) | 4.6 | 34.2 |
| oneMKL f16 GU via dpct (Gemm::f16) | 18.2 | 17.8 |
| oneMKL f16 down via dpct | 17.8 | 17.4 |
| oneMKL f16 GU direct column_major::gemm | 17.7 | 16.7 |
| swiglu_interleaved | 4.1 | 4.1 |
| new iq_dequant_experts_f16 x16 (one launch) | 4.8 | 348 (21.7/expert) |
| xmx::grouped_f16 GU 16 experts x 32 rows | 7.8 | 355 |
| iq_xmx_grouped GU 16 x 32 | 6.1 | 509 |
| oneMKL gemm_batch (group API) GU 16 groups | 1951-3155 | 1877 |

VERDICT:
- The host cost per peer expert is ~51 us, and 70% of it is the two oneMKL calls (~17-18 us each, dpct adds only
  ~1 us). This matches the analysis' 55-58 us per expert.
- Host submission does not parallelize: 2 threads with 2 queues gave the same host time (13.3 vs 13.0 ms); 3 threads
  only -9%. Some lock serializes submissions process-wide (consistent with S7's negative async result). Not pursued.
- oneMKL gemm_batch (group API) costs ~2 ms host per call: dead.
- The grouped XMX kernels in the tree (xmx::grouped_f16, iq_xmx_grouped) are 2-4x slower on the GPU than
  dequant + oneMKL even at 32 rows per expert; GRP cuts the host to 5 ms but the GPU time rises to 19 ms (12288)
  and the wall time is flat at 4K. Not a win.
- Batching dequants (WAVE) is bitwise identical but loses L2 locality (16 dequantized experts = 157 MB instead of
  one 9.8 MB expert that oneMKL reads from cache): GPU +2.5 ms at 12288. Not a win.
- => Launch batching with the existing kernels does not pay. The real lever for the 4K host bound is a GEMM that
  does not go through oneMKL per expert (a fast fused IQ-decode + DPAS grouped kernel), which is a multi-hour kernel
  project. Moving on to the critical-path items that do not need it: earlier card-1 start (doorbell instead of the
  full host wait) and the FP16 return push.
- Side finding: IQ2_XXS/IQ4_NL dequant is 3.3x slower than IQ3_S/Q2_0 (34 vs 10 us per expert): 9 layers use
  IQ2_XXS gate/up, so card 1 spends ~5-6 ms more per such layer-chunk at 12288.

## 2. Engine changes delivered (all default OFF)

Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-prefill-a3 (sha256 fef9b3bcee944c68..., built 08:19 from
the worktree = base e1b7ee8a + /mnt/vm_8tb/b70/results/night-20261016/agents/prefill.patch).
With no STRATA_NIGHT_* variable set every code path is the original one (the shared-expert block only moved into a
lambda called at the same place).

### 2a. STRATA_NIGHT_PREFILL_EARLY=1 - card 1 starts on a doorbell, not after the host's full drain

What changes per MoE layer (EP IPC path only):
- card 0 queue: router, route, ids/weights to host, [barrier event], activation push, doorbell, shared expert + gate.
  Before: router, route, shared expert + gate, ids, push, weights, then `m.cs->wait()` (pf ~3494) drained everything.
- host: waits only on the barrier event (ids + weights ready), routes, uploads card 1's tables and enqueues card 1's
  experts while card 0 is still pushing and computing the shared expert.
- card 1 queue: a bounded spin kernel (ep_wait_value on a 4 KB IPC region on card 1) before ep_copy_uncached.
- the push is a new kernel ep_push_signal: the copy_f32_wide copy (same bytes, same grid), every work-group fences at
  system scope, the last one raises the doorbell (as ep_send_rows: the source notes a flag in a separate later kernel
  was seen before the rows arrived).
- if card 1 ever hits the spin bound (50M loads, tens of seconds) the prefill fails with a clear error instead of
  computing on stale rows.
Numerics: same kernels on the same inputs -> bitwise identical outputs expected.
Expected gain (model, analysis numbers): card 1 starts earlier by the shared expert (~1.5 ms at 12288, ~0.5 ms at 4K)
plus host wake + routing + table uploads (0.6-4.6 ms at 12288): ~2-6 ms per layer-chunk, ~0.3-0.5 s per 32K request,
~0.05-0.2 s at 4K.

CONFIG -> card 1 alone, night_ep_sum_bench (single card: a cross-queue spin deadlocks on one compute engine, so only
the semantics can be checked here; the cross-card handshake is the decode path's sys_store/sys_load pattern).
COMMAND -> run_probe.sh night_ep_sum_bench "12288 0.7 100"
RESULT -> ep_push_signal 62.9 MB local in 0.507 ms, copy bit-exact vs source, flag = value, group counter back to 0;
ep_wait_value returns at once on a flag at/after the value (err 0) and gives up with err = code at the bound.
host->device copies from pageable memory return in 0.12 ms even behind a busy queue (card 1's table uploads do not
block the host); device->pageable-host copies block until executed (card 0's ids copy: harmless, the host waits for
the ids anyway).
VERDICT -> semantics verified; end-to-end needs the orchestrator's two-card run (see section 3).

### 2b. STRATA_NIGHT_PREFILL_F16SUM=1 - FP16 x per-token power-of-two scale peer sums (needs REDUCE=1)

- card 1: ep_prefill_reduce_h computes the same FP32 sums (same FMA order per token), scales each token row by 2^-e so
  its max lands in [2^14, 2^15), writes FP16 (T*N*2 bytes) plus T float scales into card 0's sum region (PCIe bytes
  126 -> 63 MB per 12288 chunk).
- card 0: ep_prefill_finish_h reads the FP16 rows and scales straight from the region (uncached loads), replacing
  ep_copy_uncached + ep_prefill_finish; result = (primary sum + peer) + shared * sigmoid(gate), the original order.
- STRATA_NIGHT_PREFILL_F16SUM_GROUPS (default 64, as the FP32 push) tunes the push grid if the link is not saturated.

CONFIG -> card 1 alone, synthetic: T rows x 10 routed entries, 70% peer, rows N(0, 0.05) with 0.1% outliers x100
(x1000 in the second case), every 997th row x1e-6.
COMMAND -> run_probe.sh night_ep_sum_bench "12288 0.7 100" "3783 0.7 1000"
RESULT:

| | 12288 | 3783 |
| --- | ---: | ---: |
| FP32 reduce, push grid, local target (ms) | 2.26 (125.8 MB) | 0.69 (38.7 MB) |
| FP16 reduce_h, push grid, local target (ms) | 2.79 (63.0 MB) | 0.86 (19.4 MB) |
| card 0: copy_uncached + finish (ms) | 2.73 | 0.86 |
| card 0: finish_h (ms) | 1.32 | 0.41 |
| peer sum rel error: mean / max | 1.76e-4 / 8.7e-4 | 1.76e-4 / 2.1e-2 |
| layer MoE output: rms err / rms | 1.58e-4 | 1.73e-4 |
| layer MoE output: max abs (max abs value) | 1.8e-3 (5.1) | 1.6e-2 (43.8) |

(max relative error above 2^-11 = 4.9e-4 only on values that cancel to below 2^-28 of their row's max: FP16
subnormals, absolute error < max * 2^-39.)
VERDICT -> kernels correct; on PCIe the reduce stays link-bound at half the bytes: expected -6.3 ms per 12288 chunk
on card 1's critical path plus -1.4 ms of card 0 finish (-2.1 / -0.45 ms at 4K): ~0.9-1.0 s per 32K request,
~0.12 s at 4K. Numerics change at FP16 level (rms 1.6e-4 relative on the MoE output): needs the quality gate.

## 3. Orchestrator recipe (two cards, server)

Binary: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-prefill-a3 in place of strata-s147, config base-ep
(REDUCE=1, PUSH=1, OWNED_PRICE=exact, --prefill 12288).

1. A0 control: no new flags (must reproduce s147: 4K ~1138-1172, 32K ~1568-1637 tok/s).
2. A1: STRATA_NIGHT_PREFILL_EARLY=1 (bitwise arm). Expect the stderr line "STRATA_NIGHT_PREFILL_EARLY=1: card 1
   starts on a doorbell...". Correctness: run the same 4K prompt in A0 and A1 with
   STRATA_PREFILL_DUMP_R=/mnt/vm_8tb/b70/results/night-20261016/runs/<tag>.dumpr and compare with
   python3 /mnt/vm_8tb/b70/results/night-20261016/agents/prefill_work/cmp_dump_r.py A0.dumpr A1.dumpr
   -> "bitwise equal: True" expected (the dump is the final residual every 64th position of the last prompt).
3. A2: EARLY=1 + STRATA_NIGHT_PREFILL_F16SUM=1 (numerics arm): cmp_dump_r.py gives the error; then the usual
   coherence/identity checks. If the 32K gain is below ~0.6 s, retry with STRATA_NIGHT_PREFILL_F16SUM_GROUPS=128.
4. Optional: + STRATA_EP_PREFILL_GATHER_ALIAS=1 (existing, S38 +1.8% at 4K; compatible with both flags).

## 4. Negative results (do not repeat)

- Launch batching with the kernels in the tree does not pay (section 1): xmx::grouped_f16 and iq_xmx_grouped are
  2-4x slower than dequant + oneMKL per expert even at 32 rows; batched dequant into 16 slots loses L2 locality
  (+2.5 ms GPU per 12288 chunk); oneMKL gemm_batch costs ~2 ms host per call.
- Multi-threaded submission (2-3 threads, own queues and oneMKL handles, same card): host time unchanged / -9%.
- IQ4_NL down dequant: register table, and 16-bit quant loads, are bitwise identical but 2-3 us SLOWER per expert
  (22.4-23.0 vs 20.1 us); reverted. Per-format expert dequant (back-to-back, us): IQ2_XS/Q2_0 9.3, IQ2_S/Q2_0 9.8,
  IQ3_S/Q2_0 10.3, IQ2_XS/IQ4_NL 17.5, IQ3_S/IQ4_NL 18.7-20.1, IQ2_XXS/Q2_0 26.7, IQ3_XXS/IQ4_NL 27.4,
  IQ2_XXS/IQ4_NL 34.2 - the floor is the 9.8 MB FP16 write (~9.3 us); IQ2_XXS, IQ3_XXS and IQ4_NL decoders cost
  8-17 us more per expert (~90 ms per 12288 chunk on card 1 summed over layers); cause not found in the time box.

## 5. What would move 4K toward 2000 (not done)

The 4K MoE phase is host-bound at ~51 us per peer expert, 35 us of it in two oneMKL calls that do not parallelize
across threads. Only a non-oneMKL GEMM path removes it: a grouped, device-driven expert kernel (IQ decode in
registers feeding DPAS, counts read on the device) that matches oneMKL's per-expert GPU time (~7 us GU / ~5.5 us down
at M <= 64 from a cached FP16 copy). With that, card 1's host enqueue drops from ~13-15 ms to <2 ms per layer-chunk
and the device-side routing + doorbell (2a) removes the remaining host drain.

# Mission 2 (09:00-): grouped prefill expert GEMM that bypasses oneMKL (STRATA_NIGHT_PREFILL_GROUPED)

## 6. Kernel: night_grouped_f16 (sycl/src/kernels/cuda/night_grouped_expert.inc, header night_grouped.hpp)

One launch per (projection, row-tile size) covers every small expert of a layer on a card. A device table of entries
{blob pointer, first row, rows <= 16/32/64} drives work-groups (entry, 64-column tile); each sub-group owns 16 output
columns (16 weight rows: gate/up interleaved 2r / 2r+1 as iq_dequant_gu_f16), each lane one weight row. Per 32-value K
step a lane loads its row's block fields (2-byte loads; the IQ blocks are only 2-byte aligned) for the NEXT step,
decodes the current 32 values bit-exactly as the dq_* decoders (sign masks as shifts, ksigns computed by parity
instead of two table loads per value) into its slot of the sub-group's local slice, and the sub-group runs
joint_matrix 8x16x16 FP16 -> FP32 DPAS against the entry's FP16 rows (joint_matrix_load_checked for row tails).
Formats: gate/up IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22; down IQ4_NL 20, Q2_0 42 (all 48 layers).
ISA (IGC dump, bmg-g31 AOT): dpas.8x8 present (2 x MT per step), SIMD16, no register spills (flag spills only),
A via load_block2d, B via SLM d32x4 loads.

Iterations (GPU ms for 32 experts of M rows, gate/up IQ3_S unless noted; ref = iq_dequant_gu_f16 + oneMKL per expert):

| version | change | M=16 | M=64 | note |
| --- | --- | ---: | ---: | --- |
| ref | dequant + oneMKL per expert | 0.56 | 0.68 | host 0.75-0.85 ms (2 launches/expert) |
| v1 | lane = row, 32-value step, dq_run decoders, 4 SG/WG | 0.50 | 0.75 | IQ2_XXS 2.9x faster than ref |
| v2 | + split K over 4 SGs, SLM reduction | 0.60 | 1.01 | worse: not latency-bound that way |
| v3 | lanes = runs of one row (coalesced), 256-value steps | 1.07 | 1.54 | worse |
| v4 | v1 + 32-value decoders (shared fields once) | 0.54 | 0.75 | IQ2_XXS 3.6x, IQ4_NL down 2.7x |
| v5 | v4 + next-step loads first, all 7 formats decoded in registers | 0.89/64e | 1.35/64e | (64 experts) |
| v5 SGN 2/8 | sub-groups per WG | within 4% | | kept 4 |

Mode split (v4, 64 experts, M=16, NIGHTG_MODE): IQ3_S gate/up full 0.86 ms = decode-only 0.71 + DPAS-only 0.26:
the IQ decode (~5 int/float instructions per value, ~450 instructions per 32-value step) is the bound, not XMX.

v5, 64 experts each of M rows (GPU ms, host ms ~0.008 for the grouped launch vs 1.4-1.6 for ref):

| projection | M=16 ref | M=16 grouped | M=64 ref | M=64 grouped |
| --- | ---: | ---: | ---: | ---: |
| gate/up IQ3_S | 1.17 | 0.89 (1.32x) | 1.41 | 1.35 (1.04x) |
| gate/up IQ2_XXS | 3.14 | 0.66 (4.76x) | 3.44 | 1.10 (3.12x) |
| gate/up IQ2_XS | 1.05 | 0.68 (1.53x) | 1.28 | 1.16 (1.11x) |
| gate/up IQ2_S | 1.08 | 0.84 (1.28x) | 1.32 | 1.34 (0.99x) |
| gate/up IQ3_XXS | 1.74 | 0.77 (2.27x) | 1.98 | 1.20 (1.65x) |
| down Q2_0 | 0.68 | 0.29 (2.35x) | 0.74 | 0.53 (1.39x) |
| down IQ4_NL | 1.26 | 0.53 (2.39x) | 1.31 | 0.71 (1.85x) |

Above ~128 rows per expert oneMKL wins on the GPU (IQ3_S M=128 0.79x, M=256 0.62x): the engine keeps dequant +
oneMKL for those.

Numerics: gate/up vs oneMKL rms rel 2.2-2.4e-7, max abs 3e-4 at |y| ~600 (FP32 summation order); down bitwise equal
to oneMKL in every case measured. Through SwiGLU's FP16 H the layer output of small experts differs at rms rel
1.2-1.4e-5 (1-ULP flips of H), large experts bitwise.

## 7. Engine integration: STRATA_NIGHT_PREFILL_GROUPED=1 (engine strata-prefill-b2)

- card 1 (peer): its held experts with <= STRATA_NIGHT_PREFILL_GROUPED_MAX rows (default 64) take the first offsets
  of its row block; one FP16 gather (GROUPED implies the S38 GATHER_ALIAS layout); large experts run per expert from
  the highest offset as before; then batches of small experts (<= T_max rows, the GUf/Hh capacity) from the last
  batch down: grouped gate/up (<= 3 launches) -> one swiglu_interleaved -> grouped down into the FP32 rows in place.
  The reduce, slot tables and card 0's finish are unchanged (rows only moved).
- card 0: its own small experts first in its local rows (only when all of them are resident and the layer takes the
  plain FP16 path: no MMQ, no STRATA_PF_XMX, nothing streamed this layer); one grouped pass before the per-expert loop.
- entry tables: 2 x 512 x 16 entries on each card (384 KB), one host->device copy per layer per card.

CONFIG -> card 1 alone, night_grouped_bench pipe: the engine's card-1 layer code path (offsets small-first, one gather
into the FP32 rows region, large experts in place from the top, small batches from the last) against every expert on
the reference path; lognormal routing as section 1.
COMMAND -> run_probe.sh night_grouped_bench "pipe <gt> <dt> <T> <experts> <median> <sigma> <G>"
RESULT (card-1 ms per layer-chunk):

| layer formats, chunk | G | small experts | per-expert host / GPU | grouped host / GPU | large rows | small rms rel |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| IQ3_S/Q2_0, 4K | 64 | 164/261 | 14.69 / 12.00 | 5.59 / 10.96 | bitwise | 1.3e-5 |
| IQ2_XS/Q2_0, 4K | 64 | 164/261 | 14.13 / 11.48 | 5.49 / 10.21 | bitwise | 1.4e-5 |
| IQ2_S/IQ4_NL, 4K | 64 | 164/261 | 14.15 / 14.09 | 5.59 / 12.24 | bitwise | 1.2e-5 |
| IQ2_XXS/IQ4_NL, 4K | 128 | 206/261 | 16.27 / 22.52 | 3.66 / 13.76 | bitwise | 1.3e-5 |
| IQ3_S/Q2_0, 12288 | 64 | 130/245 | 14.20 / 18.54 | 6.79 / 18.93 | bitwise | 1.2e-5 |
| IQ3_S/Q2_0, 12288 | 128 | 159/245 | 13.38 / 18.59 | 4.81 / 19.33 | bitwise | 1.3e-5 |
| IQ3_XXS/IQ4_NL, 12288 | 64 | 130/245 | 14.44 / 23.20 | 6.67 / 21.51 | bitwise | 1.4e-5 |
| IQ2_XXS/Q2_0, 12288 | 64 | 130/245 | 13.17 / 26.34 | 6.38 / 22.44 | bitwise | 1.4e-5 |

VERDICT -> card 1's host enqueue per layer-chunk drops 2.1-4.4x (the 4K bound), its GPU time drops 2-40% except
IQ3_S at 12288 (+2%). Expected end to end (model): 4K MoE phase from ~24 ms host-bound to ~11-14 ms GPU-bound per
layer (with card 0's own small experts grouped too): roughly -0.4 to -0.5 s of 3.3 s (4K ~1400-1450 tok/s, more on
IQ2_XXS/IQ3_XXS/IQ4_NL layers); 32K: card 1's critical path -0 to -4 ms per layer-chunk (~-0.1 to -0.3 s).
Needs the two-card run: flags below.

Flags (engine /mnt/vm_8tb/b70/results/night-20261016/builds/strata-prefill-b2, also contains EARLY / F16SUM):
- A3: STRATA_NIGHT_PREFILL_GROUPED=1 (implies the one-gather alias layout)
- A3b: + STRATA_NIGHT_PREFILL_GROUPED_MAX=128 (more experts off oneMKL: better when host-bound, 4K)
- with EARLY=1 (+ F16SUM=1) for the stacked arm.
Correctness: STRATA_PREFILL_DUMP_R A/B vs A0 -> expect "bitwise equal: False" with rms rel ~1e-5 level (small experts'
FP32 order + FP16 H), no non-finite; then the usual coherence gate.

### 7b. v6 (negative) and the final build

v6 staged each 256-value superblock of a sub-group's 16 rows into local memory with coalesced 2-byte loads and the
grid table into local memory (testing whether the scattered per-lane block reads / L1 gathers were the bound): GPU
0.87-1.02 ms vs v5 0.66-0.89 ms for 64 experts at M=16 on IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS (IQ3_S equal, M=64 slightly
better) -> reverted; v5 is final. The remaining bound is the per-value decode arithmetic (byte extract, int->float,
scale, sign select, FP16 convert: ~6 instructions per weight, decode-only 0.71 of 0.86 ms); the next lever is
int8 DPAS on the grid integers with the scales applied per 32-value block after the product (changes numerics at
the FP16-weight-rounding level, ~1e-4 relative) or a decode shared by 2+ row tiles for 65-256-row experts.

Final engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-prefill-b3 (10:22; sha256 prefix in the summary)
= base e1b7ee8a + /mnt/vm_8tb/b70/results/night-20261016/agents/prefill.patch (cumulative: mission 1's EARLY /
F16SUM hunks, already merged into main, plus mission 2's files night_grouped.hpp, night_grouped_expert.inc, the
include in iq_kernels.dp.cpp, the GROUPED hunks in prefill.cpp, CMake bench targets). Sanity rerun with the final
sources: pipe IQ3_S/Q2_0 4K host 13.53 -> 5.15 ms, GPU 12.01 -> 11.00 ms, large rows bitwise, small rms rel 1.3e-5.
Bench-only env knobs in the kernel file (NIGHTG_MODE decode/DPAS split, NIGHTG_SGN) are inert when unset.

### 7c. 128-row tiles (MT 16) and the FINAL engine strata-prefill-b4

Experts of 65-128 rows now take one 128-row tile (16 accumulators, 256 GRF) instead of two 64-row tiles (decode
once): 64 experts, GPU ms ref -> 64-row tiles -> 128-row tiles: IQ3_S M=96 1.55 -> 2.32 -> 1.74; IQ3_S M=128 2.01 ->
2.57 -> 2.18; IQ2_XXS M=128 4.01 -> 2.15 -> 2.04; Q2_0 down M=128 0.84 -> 0.97 -> 0.93 (host ~0 vs ~1.6 ms).
Engine pipe with G=128 (now 128-row tiles): IQ3_S/Q2_0 4K host 15.05 -> 3.37 ms, GPU 12.12 -> 11.64 ms; 12288 host
13.94 -> 4.94, GPU 18.52 -> 19.31; IQ2_XXS/IQ4_NL 4K host 15.31 -> 3.49, GPU 22.53 -> 13.83; IQ3_XXS/Q2_0 12288 G=64
host 14.21 -> 6.92, GPU 20.95 -> 19.78. Large rows bitwise, small rms rel 1.3e-5, no non-finite.

FINAL: engine /mnt/vm_8tb/b70/results/night-20261016/builds/strata-prefill-b4 (built 10:34) = base e1b7ee8a +
agents/prefill.patch (cumulative). Suggested arms: A3 GROUPED=1 (G 64); A3b GROUPED=1 GROUPED_MAX=128; A4 EARLY=1 +
GROUPED=1 (+ F16SUM=1 for the numerics arm).

# Mission 3 (10:45-): two-half pipeline (STRATA_NIGHT_PREFILL_HALVES)

## 8. Timeline model first (ms, uninstrumented per-kernel numbers from section 1 / the analysis / sections 2-7)

Dependencies: a half's attention at layer l needs only the earlier tokens' GDN state / K,V at layer l (the first
half's, queued earlier); its FFN input needs only its own tokens; layer l+1 of a half needs layer l of that half.
So per layer card 0 runs P(l,B) while card 1 runs the experts of (l,A), and card 0 runs Q(l,A) + P(l+1,A) while
card 1 runs (l,B). Card 0 order: P(l,A) P(l,B) | Q(l,A) P(l+1,A) | Q(l,B) P(l+1,B) | ...; P = HC, attention, HC,
router, push; Q = own experts, wait for that half's sums (device doorbell), finish, HC write.

| 12288-token chunk, ctx 12-24K | today (EARLY+F16SUM+GROUPED) | halves |
| --- | ---: | ---: |
| GDN layer: card 0 pre-MoE / push | 54.7 / 5.75 | 2 x (27.4 + 2.9) |
| card 1 experts + FP16 reduce/push | ~16 + 6.4 on the critical path | 2 x (~9.5 + 3.2), under card 0's other half (~31 ms window) |
| card 0 own experts + finish | ~6 + 1.3 (overlap card 1) | 2 x (~3.2 + 0.65) |
| GDN layer period | ~84 | ~69 (-18%) |
| QSA layer period (attention 259 ms) | ~322 | ~307 |
| per chunk / per 32K request | | -0.72 s / ~-1.9 s: 32K ~1750 -> ~1950 tok/s |
| 4K (one chunk, halves of 1891): layer period | ~38 (host + card 1 bound) | ~28 (card 0 bound) |
| per 4K request | | ~-0.45 s: ~1210 -> ~1400-1450 tok/s |

Costs inside the model: expert decode / dequant runs once per half (card 1 +~3-4 ms per layer, hidden; card 0 own
experts +~0.5 ms, exposed), dense GEMMs at M = T/2 (12288: negligible; 4K: a few %), 2x the host routing and card-1
enqueue per layer (must stay under one half's card-0 time: ~30 ms at 12288, ~10 ms at 4K with GROUPED).

## 9. Implementation (engine strata-prefill-h2, flag STRATA_NIGHT_PREFILL_HALVES=1, default off)

- The layer body of Prefill::run_impl is now a lambda layer_step(l, sub, stage); flag off = layer_step(l, -1, 0) for
  every layer (the original code path, same kernels in the same order).
- Halves: a RAII HalfView offsets every token-indexed chunk buffer (R, grs, xn16 (with the chunk's padded stride),
  lo, lo16, gated, inj, mixed, mixed_bf, mixed_h, bo, emb, ple_emb, steps_dev, the mapped grouping tables) by the
  half's first row, and moves the routing weights, shared-expert rows and shared gate (they live in the scratch region
  the other half's attention reuses) to dedicated carry buffers on card 0 (T_max x (K + N + 1) floats, 126 MB);
  T / p0 / c0 are the half's; gdn_index / qsa_index are derived from l; QSA's host step table is offset; QSA's
  next-layer KV prefetch is skipped (each step stages its own cells; the pre-loop prefetch is consumed by the first
  half).
- stage 1 (P): HC read, attention, HC write, FFN HC read, router, ids into mapped memory (per-half rows), weights into
  mapped host memory (copy kernel: no host block), a barrier event, the half's activation push (ep_push_signal into
  that half's rows of card 1's inbox, per-half epoch), the shared expert.
- stage 2 (Q): the host waits only for that barrier event, routes, enqueues card 1 (spin on that half's push epoch,
  its rows, experts, reduce into that half's rows of the sum region; the last work-group raises card 0's new sum
  doorbell with a per-half epoch), card 0's own experts, then card 0 spins on the device for that half's sums (no host
  wait on card 1 anywhere in the layer loop), finish, HC write.  Card 1 is drained once at the end of the chunk.
- Driver: P(0,A) P(0,B), then per layer Q(l,A) P(l+1,A) Q(l,B) P(l+1,B).
- Taken only when: expert parallel IPC, STRATA_EP_PREFILL_REDUCE=1, chunk >= 256 tokens, all layers on this stage,
  nothing streamed in the chunk, mapped grouping tables, no MMQ, no peer timing; else one stderr note and the original
  path.  Every doorbell wait is bounded (50M loads) and reported as an error instead of a hang.
- Not measurable on one card (both directions of the handshake are cross-card); compiled for bmg-g31 (h2, 11:12).

Validation recipe (two cards):
1. Bitwise: a prompt of exactly 24576 (or 12288) tokens, same env both runs, GROUPED off in both (the reference cannot
   use the alias layout at --prefill 6144):
   A0: --prefill 6144, flag off;  A1: --prefill 12288 + STRATA_NIGHT_PREFILL_HALVES=1; both with
   STRATA_PREFILL_DUMP_R=<file>; cmp_dump_r.py A0 A1 -> "bitwise equal: True" expected (halves = 6144-token chunks
   computed in an interleaved order).  EARLY / F16SUM may be on in both.
2. Speed: the standard 4K / 32K bench, base-ep (+EARLY +F16SUM +GROUPED) with and without HALVES=1.

### 9b. Independent review (read-only subagent, model claude-opus-5-5) and fixes -> engine strata-prefill-h3

Review findings and what changed (h3, 11:32):
- BUG fixed: the host staging vectors of card 1's tables (src_ipc, reduce_slot_host) and of both grouped-entry tables
  were refilled by the other half while the first half's copy could still be pending (the original relied on the
  per-layer host wait on card 1). Now per-half vectors (reserved once at T_max * K, no reallocation); their reuse at
  the next layer of the same half is ordered by card 1's copy -> reduce doorbell -> card 0's wait -> routed event ->
  host.
- BUG fixed: card 0's sum-doorbell timeout wrote card 1's error word (cross-context USM). Card 0 now has its own
  mapped error word; the chunk end drains card 0 too (m.cs->wait()) and reports either word as an error.
- Risk fixed: card 1 no longer copies the routing weights straight out of card 0's mapped memory; the host copies them
  into a per-half vector first (245 KB at 12288).
- Risk fixed: HALVES now requires STRATA_EP_PREFILL_PUSH=1 (with PUSH=0 card 0 would read the sums from card 1's
  memory right after a mid-kernel doorbell).
- Accepted: relies on immediate command lists (as EARLY does); stats count per half; QSA KV prefetch overlap is lost
  after the first QSA layer (staged-KV layouts only); an error return mid-layer can leave card 1 spinning up to the
  bound.
- Confirmed by the review: HalfView covers every token-indexed buffer of carve with matching strides; only w / shared
  / sg cross stage 1 -> 2 from the aliased scratch region (carried); no wait can precede its signal (epochs monotonic,
  regions zeroed); stage 0 identical to the original; P(1,A)'s ple_land copies the whole chunk; the pre-loop KV
  prefetch is consumed by half A and half B re-stages including A's rows.

Bitwise test conditions (from the review): prompt length a multiple of 12288; STRATA_PREFILL_FIRST=0 (as in base-ep);
compare R / KV / GDN state (STRATA_PREFILL_DUMP_R), not the MTP drafter's chunk consumers; check the baseline is
run-to-run reproducible first. Caveat: the reference must run with --prefill 6144, which changes T_max-derived
choices (PLE sub-batch size from the scratch region, the alias / GROUPED eligibility): if A0/A1 are not bitwise
equal, expect FP32-rounding-level differences from PLE GEMM batching, not structure.

Final engine for all three missions: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-prefill-h3
(base e1b7ee8a + agents/prefill.patch; contains EARLY, F16SUM, GROUPED, HALVES).

## 10. GROUPED quality flag (orchestrator NLL gate + DUMP_R): diagnosis so far (11:55)

Data (orchestrator): 8377-token prompt, GROUPED off vs on, else identical: final residual rms rel 8.7e-2.
My analysis of the two dumps (/mnt/vm_8tb/b70/results/night-20261016/runs/grpbug/): per-position rms rel median
3.7e-2, p90 7.9e-2, max 0.69; position 0 only 5.4e-4; already 2-4% at positions 64-320; spread evenly over the chunk
(quarters 3.4/4.2/3.3/3.7e-2) and over the 4 HC streams. Position 0 cannot see other tokens and is routed through
small experts too, so its 5e-4 is the rounding-level effect amplified over 48 layers; the %-level elsewhere comes
from cross-token propagation. Two explanations remain: (a) a defect that corrupts some rows but not a token's first
row in each expert (token 0 is always row 0 of its experts), or (b) chaotic amplification of the FP32-order / FP16-H
rounding difference through discontinuities (MoE top-10 routing flips and later QSA top-k flips).

Checked and excluded: main's prefill.cpp is identical to my pre-halves version (no merge damage); main changed no
decode table or dq_* decoder my kernel uses (iq_kernels.dp.cpp: additions only); the kernel file is identical; the
card-1 engine path (small-first offsets, one gather, in-place alias, batches) was emulated with lognormal routing on
real formats: large rows bitwise, small rows rms rel 1.2-1.6e-5. Not emulated: card 0's small-expert path, real
weights/activations, real routing.

New engine strata-prefill-c2 (11:52; GROUPED + HALVES + EARLY + F16SUM code as h3) adds:
- STRATA_NIGHT_PREFILL_GROUPED_CHECK=1 (debug, slow): every layer, on each card, the grouped small experts are
  re-run on the per-expert reference path (dequant + oneMKL, from saved inputs on card 1) and compared; one line per
  layer and card: "strata prefill GROUPED_CHECK: card N chunk@p layer l (gu, down): experts, max abs, rms rel, worst
  expert ... <-- BAD" (BAD above 1e-3 rms rel; expected ~1e-5).
- STRATA_NIGHT_PREFILL_GROUPED_CARDS=1 (card 1 only) / 2 (card 0 only) / 3 (both, default).

Requested runs (same 8377-token context, DUMP_R each):
1. GROUPED=1 + GROUPED_CHECK=1: if every line is ~1e-5 the kernels and both integrations are right (then (b)); a BAD
   line names the card, layer, formats and expert.
2. Control for (b): reference twice (run-to-run determinism) and F16SUM=1 vs off (a known rounding-only change ~10x
   larger than GROUPED's per-expert one): if F16SUM's dump divergence is similar, GROUPED's is chaotic sensitivity.
3. If 1 shows BAD on one card only: GROUPED_CARDS=1 / 2 to confirm, and GROUPED_CARDS=<good card> is the safe sub-flag.

## 11. Round 4 (12:20-): dequant costs and the 4K path (engine strata-prefill-d1, main dc023c67 + prefill_r4.patch)

Worktree re-synced to main HEAD dc023c67 (git archive); incremental patch vs main:
/mnt/vm_8tb/b70/results/night-20261016/agents/prefill_r4.patch. All three flags default off, all bitwise.

### 11a. STRATA_NIGHT_PREFILL_DQ32=1: the large experts' FP16 dequant (iq_dequant_expert_f16)

CONFIG -> card 1, night_grouped_bench dq: 16 random experts per format pair, 64 back-to-back dequants, profiling
events; bitwise compare of both FP16 matrices of every expert vs the original kernel.
Iterations: v1 one work-item per 32 values (scattered 64-byte stores) 26-35 us; v2/v3 one per 8-value run reading
a register array by a runtime index (private memory) 43-63 us; v4 per-run decoders that load only the run's fields
(2-byte words, computed IQ2 signs, shift masks, one grid load); v5 = v4 for IQ2_XXS / IQ3_XXS, the original run
decoders for the rest, in the same launch; v6 IQ4_NL table in local memory: slower (24.5 vs 19.6 us), reverted.
RESULT (v5, us per expert, gate/up + down, original -> DQ32, all 0 of 32 matrices differ):

| gate/up / down | Q2_0 down | IQ4_NL down |
| --- | --- | --- |
| IQ3_S | 10.95 -> 10.22 | 19.64 -> 19.59 |
| IQ2_XXS | 27.71 -> 14.73 (1.88x) | 36.34 -> 25.62 (1.42x) |
| IQ2_XS | 9.89 -> 9.18 | 18.53 -> 18.71 |
| IQ3_XXS | 19.58 -> 14.94 (1.31x) | 28.17 -> 24.54 (1.15x) |
| IQ2_S | 10.22 -> 9.17 | 19.10 -> 18.91 |

VERDICT -> bitwise, never slower; IQ2_XXS / IQ3_XXS layers gain 4-13 us per large expert (~15 layers). With GROUPED
only experts over 64 rows (half of them at 4K) still take this path: ~20 ms per 12288 chunk on card 1 (hidden under
HALVES) and ~8 ms on card 0. Small end-to-end gain; safe to stack. IQ4_NL down (+9 us per expert over Q2_0) stays the
cost: its table gathers resisted three rewrites.

### 11b. STRATA_NIGHT_PREFILL_DDQ=1: the dense FP16 dequant (Gemm::native's dequant_f16, K-quant / IQ4_XS types)

The original writes one 2-byte value per store from one work-item per 32 values; DDQ decodes the same group32 into
local memory and writes 16 bytes per lane, consecutive lanes on consecutive bytes.
RESULT (card 1, random blocks, GPU ms, bitwise equal in all):

| type, shape | original | DDQ |
| --- | --- | --- |
| Q4_K 10240 x 2560 (qkv) | 0.375 (179 GB/s) | 0.223 (302 GB/s), 1.69x |
| Q5_K 6144 x 2560 (z) | 0.229 (184 GB/s) | 0.132 (320 GB/s), 1.73x |
| Q6_K 2560 x 6144 (out) | 0.410 (108 GB/s) | 0.314 (141 GB/s), 1.31x |
| IQ4_XS 12288 x 2560 (attn_q) | 0.476 (167 GB/s) | 0.293 (271 GB/s), 1.62x |
| Q6_K 640 x 2560 (shared) | 0.044 | 0.035, 1.28x |

VERDICT -> ~40% off the dense dequant (~64 ms per chunk -> ~40 ms; doubled under HALVES): ~25 ms per 4K prompt
without HALVES, ~50 ms with.

### 11c. STRATA_NIGHT_PREFILL_DQCACHE=1 (with HALVES): one dense dequant per layer for both halves

HALVES runs every dense projection twice per layer (once per half), each one dequantizing its weight again. DQCACHE
gives Gemm::native a 132 MB arena on card 0: the first half's stage 1 dequantizes into it (one entry per weight, at
most 16), the second half's stage 1 finds the entry (same blocks, N, K) and runs its GEMM on the same FP16 values
(bitwise); reset at each layer's first half. Expected: the second dequant pass gone, ~40-64 ms per 4K prompt
(~2-3%), ~1% at 32K. Needs the two-card run.

### 11d. Not done: HC traffic (analysis item 7)

A fused HC up-projection + mix (gr_upmix exists only for HIP gfx11) would save ~2.5 ms per half-layer at 12288
(~0.6 s per 32K, ~70 ms per 4K): a BF16 XMX GEMM with the gr_mix_r epilogue. Not started in this round.

Flags to try on top of the current best (GROUPED + HALVES + EARLY + F16SUM + XMX attention): STRATA_NIGHT_PREFILL_DDQ=1,
STRATA_NIGHT_PREFILL_DQ32=1, STRATA_NIGHT_PREFILL_DQCACHE=1 (all bitwise vs without them: STRATA_PREFILL_DUMP_R must be
identical).
