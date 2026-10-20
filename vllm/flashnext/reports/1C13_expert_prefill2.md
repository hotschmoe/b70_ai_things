# C13 expert prefill: one layer, measured

Card 0, 2026-10-20. Library `b70_gguf_xpu/_C.c13.so` (2635416 bytes,
12:59). It does not replace `_C.so`, `_C.c10.so`, `_C.c11.so`, or
`_C.c12.so`. Log: `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c13-bench5.log`.

The one-layer path is `prefill_fast.moe_fast`
(`prefill_fast.py:49`). Dispatch is `prefill.py:89`: mode 0 and
`T >= DQ_T_MIN` (12288, `prefill_dq.py:34`) and `wants_fast()`
(`prefill_fast.py:19`). `B70_PREFILL_DQ_LAYER=bmm` keeps the C10
split. The daily `_C.so` has no `prefill_dequant_wide`, so a serve
that does not point `B70_GGUF_SO` at `_C.c13.so` stays on the C10/C11
path. `DQ_T_MIN` stays 12288.

## CONFIG

Flash-Next expert layer, E=512, top-10, H=2560, F=640. Gate and up
IQ3_XXS, down Q2_0, for the timed layer. Dequant bandwidth also covers
IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, and IQ4_NL at the full expert shape.
icpx 2026.1, `-DB70_PREFILL_SMOKE=1`, SYCL cache on
`/mnt/vm_8tb/b70/build/gguf-xpu-sycl-cache`. Health via
`bin/xpu-health --card 0` inside `scripts/c13_run.sh`.

```
B70_AGENT=C13 B70_GPU_LOCK_TIMEOUT=21600 B70_PREFILL_TIMEOUT=7200 \
  B70_CPU_LOCK_TIMEOUT=1800 bin/gpu-run --card 0 scripts/c13_run.sh
```

## Dequant

Wide dequant (`dequant_wide.cpp:213`) is bit-exact against
`prefill_dequant` on every tuned type, including the packed gate+up
layout (0 mismatches). One subgroup owns 512 outputs. Each lane
decodes one 32-wide group and stores 32 bf16 (`dequant_wide.cpp:252`).

| type | bytes in+out | event ms | GB/s | old GB/s |
| --- | ---: | ---: | ---: | ---: |
| IQ3_XXS | 1.999e9 | 7.014 | 285.0 | 255.8 |
| IQ3_S | 2.038e9 | 7.115 | 286.4 | |
| IQ2_XXS | 1.894e9 | 6.949 | 272.6 | |
| IQ2_XS | 1.920e9 | 7.010 | 273.9 | |
| IQ2_S | 1.946e9 | 6.975 | 279.0 | |
| Q2_0 | 1.914e9 | 7.079 | 270.3 | 258.8 |
| IQ4_NL | 2.150e9 | 7.651 | 281.0 | |

A bf16 copy of one gate (3.355e9 bytes, read+write) is 1147 GB/s.
The tuned kernels land at 270-286 GB/s. The 500 GB/s target is not
met. Two earlier kernels were measured and dropped: an ESIMD flat
load aborted in IGC with an internal floating-point crash (health
stayed rc=0), and a striped `SubgroupBlockWrite` of 8 bf16 per lane
was bit-exact but slower (IQ3_XXS 238 GB/s, layer 63.4 ms at
T=12288, log `c13-bench4.log`).

## One layer

Real layer time, event min of 2 iters after 1 warmup. Routing hits
every expert (empty=0). mmax is the padded expert rows.

| T | mmax | mean | fast overlap ms | fast seq ms | C10 bmm ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 109 | 80 | 40.13 | 40.13 | 45.12 |
| 12288 | 287 | 240 | 56.97 | 56.92 | 66.16 |

C11 fused layer was 26.76 ms at T=4096 and 74.34 ms at T=12288. C11
bmm layer was 66.97 ms at T=12288. At T=4096 the fused kernel remains
the winner, so `DQ_T_MIN` stays 12288. At T=12288 this path is the
winner. `prefill_moe_layer` at T=12288 matches `moe_fast` bit for bit
(0 mismatches). Overlap of the down dequant on a second stream saves
about 0 ms at both sizes.

Sequential profile at T=12288 (one call, milliseconds):

| piece | ms |
| --- | ---: |
| group (histogram + host mmax) | 1.80 |
| dequant gate | 8.94 |
| dequant up | 7.11 |
| dequant down | 7.15 |
| pad to mmax | 2.89 |
| bmm gate+up | 8.45 |
| SiLU * mul | 1.50 |
| bmm down | 4.67 |
| unpad scatter | 3.15 |
| weighted fp32 sum (`prefill_dq.py:227`) | 11.47 |
| sum | 57.13 |

Dequant of the three packs is 23.2 ms. The two GEMMs are 13.1 ms.
Pad, unpad, SiLU, the routing sync, and the fp32 top-10 combine are
20.8 ms. C11's 35.55 ms figure was the three dequants plus two bare
GEMMs. The combine alone is a cast of the bf16 slots to fp32 and a
sum over top-10.

## Scratch and 2x200K

One layer of bf16 experts is `3 * 512 * 2560 * 640 * 2` = 5033164800
bytes = 4.6875 GiB. Two layers are 9.375 GiB.

From `reports/02_model_support.md`: IQ3_XXS weights 43.797 GiB plus
2x200K BF16 KV 9.155 GiB is 52.95 GiB, about 11 GiB free before
vision and MTP. A second layer of bf16 experts (9.375 GiB) leaves
about 1.7 GiB, under the ~4 GiB servable floor. With mmproj 0.845 GiB
and about 2 GiB of MTP the free pool is about 8.2 GiB, and 9.375 GiB
does not fit. One resident layer fits the 11 GiB pool and is tight
once vision and MTP are in (about 3.5 GiB left). The measured overlap
saves nothing, so a second buffer has no latency return on this card.

## Full-model GEMM budget

Non-expert pieces are C10's (`reports/1C10_prefill_dq.md`). Attention,
GDN recurrence, conv, norms, PLE, and indexer top-k stay outside the
sum. T=512 and T=4096 keep C11's fused expert winner. T=12288 uses
56.97 ms times 48 layers.

| piece | T=512 | T=4096 | T=12288 |
| --- | ---: | ---: | ---: |
| routed experts | 519.38 | 1284.53 | 2734.56 |
| GDN qkv+gate+out | 44.46 | 185.00 | 523.37 |
| full attn q+k+v+output | 12.80 | 53.26 | 150.67 |
| shared expert | 10.23 | 31.74 | 79.75 |
| head | 12.39 | 55.32 | 153.67 |
| HC + output mixer | 4.66 | 27.08 | 118.43 |
| router + indexer | 0.79 | 4.97 | 15.29 |
| GEMM sum | 604.71 | 1641.90 | 3775.74 |
| GEMM-only tok/s | 847 | 2495 | 3255 |

C10's T=12288 GEMM sum was 4255.71 ms (2887 tok/s). The expert term
moves from 3214.53 ms to 2734.56 ms.

2000 tok/s on a 12288-token chunk is 6.144 s. GEMM accounts for
3.776 s. That leaves 2.37 s for attention and the rest of the step.
Those pieces were not timed in this run. This is a GEMM budget, not
an end-to-end prefill rate.

## bf16 gate

The 5e-2 threshold is unchanged in `prefill_check.py` and in
`c11_bench.py`.

Scale-1 synthetic pack, fused SGN=4 versus a bf16 `torch.mv`
reference: max_rel 1.963 at got=2.801, ref=-2.696 (a sign
disagreement at magnitude about 2.8). sig_max_rel 1.963. frac_gate
0.945. frac_near 0. The largest lane is got=6128, ref=6097. This is
an fp16 DPAS plus fp16 SiLU store versus a bf16 matmul, on a
moderate lane. The new bf16 layer on the same pack has max_rel
8.7e-3 and frac_gate 1 (largest-lane error 1.5 on a value of 470).

Real rows, shard-1 `blk.36`, gate and up IQ3_XXS, down Q2_0, first 4
experts, T=8, topk=2, fp16 randn. Dequant of those bytes is bit-exact.
Fused versus the same bf16 reference, unscaled: max_rel 1.993 at
got=1.34e-4, ref=-1.35e-4, max_abs 1.89e-3, frac_gate 1, sig_max_rel
0. The new layer on the same rows: max_rel 0.365 at got=-9.5e-5,
ref=-1.5e-4, max_abs 8.8e-4, frac_gate 1, sig_max_rel 0. The largest
new-layer lane is got=-0.072, ref=-0.071. No real-row output reaches
magnitude 1. Strict max_rel trips on a lane near 1e-4 because the
denominator floor is 1e-6. Every lane passes
`abs <= 5e-2 + 5e-2 * max(|got|, |ref|, 1e-6)`.

Decision: leave 5e-2 in place. The synthetic fused miss is an
fp16-versus-bf16 gap at magnitude about 3, and it stays a miss. The
real-row strict max_rel misses are absolute errors under 0.002 on
lanes below 2e-4. The logged `VERDICT failures=1` is that strict check on
the new layer (`fast real x1`). It is the small-lane metric. Dequant
and the large lanes agree with the bf16 reference.

## VERDICT

The T>=12288 dispatch is live on `_C.c13.so`. One expert layer is
40.13 ms at T=4096 and 56.97 ms at T=12288. T=4096 stays on the fused
kernel. GEMM-only prefill at a 12288-token chunk is 3255 tok/s, with
2.37 s of the 6.14 s 2000-tok/s budget still unmeasured (attention and
the non-GEMM step). Dequant is 270-286 GB/s. Health before and after
rc=0.
