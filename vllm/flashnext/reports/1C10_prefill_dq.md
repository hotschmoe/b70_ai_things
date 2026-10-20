# C10 prefill dequant + oneDNN (2026-10-20)

Card 0, `_C.c10.so` (8294752 bytes, sha256
`55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0`,
linked 08:28, not rebuilt). Dequant is `csrc/dequant_fast.cpp:42`:
one work-item, one 32-wide group, `b70_decode32`, bf16 or fp16 store.
Bench: `tests/c10_dq_bench.py` via `scripts/c10_run.sh` under
`bin/gpu-run --card 0`. Log
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c10b-dq-bench.log`.
Queued 09:10 behind H1's health probe, `gpu-run` exit 0 at 09:13
in 43s, python quiet-window 12s (SYCL cache already warm).
`HEALTH_BEFORE:0`, `HEALTH_AFTER:0`.
No xe-reset. Times below are event-min microseconds unless noted.
Wall-min picks the same winner on every row. Roof used for GB/s is
608 GB/s.

## Correctness

Pass bit (`tests/c10_dq_bench.py` `bf16_ulp_ok`, and the dense check
that requires both):

- `prefill_dequant` bf16 bits match CPU round-to-nearest-even and the
  existing `dequant_bf16`. 0 mismatches on all 13 types at 8x256, plus
  Q2_0 K=640, IQ3_XXS K=2560, Q4_K K=2560.
- `dense_dq` matches oneDNN `torch.mm` of those bf16 weights within 1
  bf16 ulp. This run: `mm_ulps_bad=0`, `mm_max_abs=0`.
- That output against an fp32 matmul of the same bf16 weights is within
  8 bf16 ulps (`bf16w_ulps_bad=0`). Max abs was 63.4 on the Q4_K case.
- Non-finite values fail.

The old atol=rtol=5e-2 check against the fp32 dequant is logged and is
not the pass bit. Same Q4_K case: `vs_f32_max_abs=7.085e+01`,
`vs_f32_max_rel=2.353e+01`. One bf16 ulp near 2e4 is 128, so an abs
error of ~70 fails 5e-2 on the smaller lanes while the bf16 matmul
matches. Dispatch with `B70_PREFILL_DQ_M=0` matched the same bf16
matmul (`mm_max_abs=0`).

MoE, E=4 T=5 topk=2 H=256 F=64, one invalid id. bmm and loop:
`bf16_ulps_bad=0` against the bf16-weight reference (max abs 0.289),
and `bmm_vs_loop` max abs 0. Against the fp32 dequant reference,
`vs_f32_max_rel=1.777e+01` (the 5e-2 check would fail). `grouped_dq`
max abs 0.244, 0 bad ulps. `gemv` still links and is finite.
`F.grouped_mm` runs on XPU (probe shape (16, 16)); it was not timed.

## Dequant bandwidth

Large buffers land at 250-278 GB/s (41-46% of 608). A tiny Q2_0
matrix is launch-bound.

| shape | event us | GB/s | % roof |
| --- | ---: | ---: | ---: |
| Q2_0 2560 x 640 | 107.0 | 34.9 | 5.7 |
| Q5_K head 248320 x 2560 | 6141.6 | 278.2 | 45.8 |
| IQ4_XS GDN qkv 10240 x 2560 | 262.7 | 252.6 | 41.6 |
| IQ3_S GDN gate 6144 x 2560 | 154.0 | 248.1 | 40.8 |
| IQ4_XS GDN out 2560 x 6144 | 153.0 | 260.2 | 42.8 |
| Q6_K attn out 2560 x 6144 | 277.0 | 160.1 | 26.3 |
| IQ3_XXS all expert gate/up | 7799.6 | 256.3 | 42.2 |
| Q2_0 all expert down | 7400.6 | 258.6 | 42.5 |

Expert gate+up+down dequant is about 23 ms if the three packs run
back to back. Gate and up share one scratch, so they stay ordered on
the stream.

## Dense crossover, N=12288 K=2560

Q4_K. dq is dequant plus `torch.mm`. mm is the GEMM with the weight
already bf16. dq wins at every measured M. The gap at M=512 is
already 3.5x on wall time (763 vs 2664 us), so the crossover is below
512. M below 512 was not measured and stays on the fused kernel.

| M | fused us / TF | dq us / TF | mm us / TF |
| --- | ---: | ---: | ---: |
| 512 | 2553 / 12.6 | 663 / 48.6 | 320 / 100.7 |
| 1024 | 3092 / 20.8 | 957 / 67.3 | 528 / 122.1 |
| 2048 | 5161 / 25.0 | 1573 / 81.9 | 963 / 133.7 |
| 4096 | 10277 / 25.1 | 2807 / 91.8 | 1882 / 136.9 |
| 8192 | 20471 / 25.2 | 5356 / 96.2 | 3807 / 135.4 |
| 12288 | 30581 / 25.3 | 7928 / 97.5 | 5822 / 132.8 |

IQ4_XS, the real `attn_q` type, same N,K. dq event us:
673.6 / 2803.1 / 7929.9 at M=512 / 4096 / 12288 (47.8 / 91.9 / 97.5
TFLOPS). Fused stays near 25 TFLOPS (23.7 at M=512). Same winner.

fp16 dequant+mm at M=4096 only: 2238 us, 115.2 TFLOPS, faster than
bf16 dq (2807 us). No fp16 numeric check was run, so the wired path
is bf16.

## K=640 down, Q2_0 N=2560

This is the short-K shape C8 called out. Fused wins at M=512. dq wins
at M=4096 and M=12288. mm at M=12288 is 93 TFLOPS; the K=2560 mm
above is 133. The regression is real, and dequant does not hide it
at M=512 because the GEMM is too small to pay for the unpack.

| M | fused us / TF | dq us / TF | mm us / TF | wall winner |
| --- | ---: | ---: | ---: | --- |
| 512 | 143 / 11.7 | 248 / 6.8 | 190 / 8.8 | fused (238 vs 356 us) |
| 4096 | 588 / 22.8 | 369 / 36.3 | 194 / 69.3 | dq (463 vs 683 us) |
| 12288 | 1709 / 23.6 | 836 / 48.2 | 432 / 93.2 | dq (971 vs 1809 us) |

## Experts, E=512 topk=10 H=2560 F=640

`grouped_mm` is available and was not the timed path. bmm pads each
expert to a multiple of 8 and calls `torch.bmm`. loop is one
`torch.mm` per occupied expert. hybrid keeps fused gate/up and uses
bmm for the K=640 down. Fused is the C8 s64 kernel in this .so
(split at T=12288: GU 45.9 ms, down 18.9 ms, total 74.5 ms).

| T | fused ms | bmm ms | loop ms | hybrid ms | winner |
| --- | ---: | ---: | ---: | ---: | --- |
| 512 | 10.82 | 35.04 | 108.86 | 20.68 | fused |
| 4096 | 26.76 | 44.24 | 112.15 | 40.16 | fused |
| 12288 | 74.34 | 66.97 | 122.71 | 91.58 | bmm |

48-layer expert-only tok/s from the winner's event time:
986 / 3189 / 3823 at T=512 / 4096 / 12288. C8's T=12288 budget used
76.7 ms (3351 tok/s). bmm is 1.11x that fused call. Loop and hybrid
lose at every T. The K=640 down is 18.9 ms of the 74.3 ms fused
layer; bmm's win is the whole layer, not a down-only fix.

## Head, Q5_K N=248320 K=2560

M=12288 was not run (the bf16 output is about 6 GiB, and `dense_dq`
then widens it to fp32). M=512 dq 12.39 ms (52.5 TFLOPS) vs fused
24.58 ms. M=4096 dq 55.32 ms (94.1 TFLOPS) vs fused 197.98 ms.
Dequant of the head is 6.14 ms and does not grow with M. Projected
dq at M=12288 holds that dequant fixed and scales the rest by 3:
(55.318 - 6.142) * 3 + 6.142 = 153.7 ms. Labeled projected.

## Dispatch

`b70_gguf_xpu/prefill_dq.py:82` `dense_wants_dq`, used by
`prefill.py:78`:

- M >= 4096: dequant + `torch.mm` for every K.
- 512 <= M < 4096: dq only when K > 640.
- M < 512: fused.

`prefill.py:99` `moe_wants_dq`: full layer uses dequant + bmm at
T >= 12288. Below that the fused layer stays. One grouped GEMM
(`prefill_grouped`) stays fused (`DQ_GROUP_M_MIN` at
`prefill_dq.py:35`) because that op was not the measured win; the
expert win is the full layer. Env overrides: `B70_PREFILL_DQ_M`,
`B70_PREFILL_DQ_T`, `B70_PREFILL_DQ_GROUP_M` (`off` / `0` / N).
CPU check of the default policy matches the table above. The GPU
run forced the env off so both kernels were timed, then forced
`B70_PREFILL_DQ_M=0` for one dense dispatch call.

`prefill_check.py` correctness uses M=7 and M=130 and T=3, all under
these cuts, so those 5e-2 checks still run the fused kernel.

## Chunk size

`max_num_batched_tokens` is M for the dense GEMMs and T for the
expert layer. Recommend **12288**.

- It is the only measured T where expert bmm beats fused.
- It is the best GEMM-only tok/s below (2887).
- A cap of 4096 still clears a 2000 tok/s GEMM budget (2495 tok/s,
  406 ms of slack) and correctly keeps fused experts, but the slack
  for attention plus the GDN recurrence is much smaller than at 12288.
- Below 512 the new dense path does not engage.

## Full-model GEMM budget

Same pieces as `reports/1C8_prefill3.md` (36 GDN, 12 full attn, 48
layers). Attention, GDN recurrence, conv, norms, PLE, and indexer
top-k are outside the sum. Quantized long-K projections scale the
measured IQ4_XS dq time at N=12288 K=2560 by `(N*K)/(12288*2560)`.
Shared and attn K=640 downs use the measured K=640 row, not that
scale. Routed experts use the winner above, times 48. Head M=12288
is the projection in the head section. HC and router/indexer are C8's
bf16 milliseconds times this run's Q4 `torch.mm` event divided by
C8's (320.1/323.0, 1881.9/1855.5, 5822.0/5481.5 us).

| piece | T=512 | T=4096 | T=12288 |
| --- | ---: | ---: | ---: |
| routed experts, measured winner | 519.38 | 1284.53 | 3214.53 |
| GDN qkv+gate+out, dq scaled | 44.46 | 185.00 | 523.37 |
| full attn q+k+v+output, dq scaled | 12.80 | 53.26 | 150.67 |
| shared expert | 10.23 | 31.74 | 79.75 |
| head (12288 projected) | 12.39 | 55.32 | 153.67 |
| HC + output mixer, torch rate | 4.66 | 27.08 | 118.43 |
| router + indexer, torch rate | 0.79 | 4.97 | 15.29 |
| GEMM sum | 604.71 | 1641.90 | 4255.71 |
| GEMM-only tok/s | 847 | 2495 | 2887 |

C8's GEMM sum was 612 / 2557 / 7265 ms (836 / 1602 / 1691 tok/s).
The T=12288 change is mostly the dense projections and the head
(GDN 2026 to 523 ms, attn 583 to 151, head 620 to 154). Experts
move 3680 to 3215 ms.

Per layer at T=12288: experts 67.0 ms, one GDN layer 14.5 ms, one
full-attn layer 12.6 ms, shared 1.7 ms. Head once, 154 ms projected.

2000 tok/s on a 12288-token chunk is 6.14 s. This GEMM sum is 4.26 s,
so 1.89 s remains for attention and the GDN recurrence. That slack
is not a measurement of those kernels. The 2000 tok/s end-to-end
goal is not claimed. At T=512 the GEMM sum is 605 ms against a
256 ms budget for 2000 tok/s.

Small-N mats (attn K and V, shared gate/up) are NK-scaled from the
N=12288 dq time. They are a few tens of milliseconds together. A
2x error there does not change the tok/s column. The head at M=12288
is projected; doubling it moves 2887 tok/s to about 2786.

## VERDICT

Dequant is bit-exact. bf16 dequant + oneDNN matches `torch.mm` of the
bf16 weights exactly on the Q4 check, and the 5e-2 fp32-weight gate
still fails (`max_rel` 23.5) for the reason C10 stopped on. Dense
prefill should take this path at M>=512 for K>640 and at M>=4096 for
K=640: 97 TFLOPS at M=12288 against 25 for the fused kernel. Experts
should take it only at T>=12288 (67.0 vs 74.3 ms); fused remains
faster at 512 and 4096. Set `max_num_batched_tokens=12288`.
GEMM-only Flash-Next prefill at that chunk projects to 2887 tok/s,
from C8's 1691, with 1.89 s left before attention and GDN recurrence.
The expert layer is still 75% of the GEMM sum.
