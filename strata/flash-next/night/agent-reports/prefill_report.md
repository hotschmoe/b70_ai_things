# prefill agent report (night 2026-10-16) - EP prefill

Worktree: /mnt/vm_8tb/b70/build/night-wt/prefill/src-ep (branch night-20261016).
Probe: sycl/src/kernels/night_ep_expert_bench.cpp (ninja target night_ep_expert_bench), run on card 1 only via
/mnt/vm_8tb/b70/results/night-20261016/agents/prefill_work/run_probe.sh. Raw outputs: prefill_work/probe{1..4}.txt.

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
