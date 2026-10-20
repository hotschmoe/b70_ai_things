# C1 MoE decode -- floor, fused layer, Q2_K / Q3_K

Owner: C1. Plugin: `vllm/flashnext/plugins/b70_gguf_xpu`. Card 0 only
(`bin/gpu-run --card 0`, `ZE_AFFINITY_MASK=0`). Device: Intel Arc Pro B70.
Roofline used below: 608 GB/s. Latency is the metric. Both-card
`bin/xpu-health` was skipped: card 1 stayed leased to other agents, and
this work is a single-card kernel launch.

unitrace and onetrace are not installed. Queue `enable_profiling` does
not stick. Device time is `torch.xpu.Event` (`submit_profiling_tag`).
`a2` syncs each call. `behind` queues the call after a burn matmul.
When `behind` is larger than `a2` on a short kernel, the burn leaked
into the interval; trust `a2` there. When they agree, that number is
device time.

Step 1 is journaled at 03:40. Empty kernel behind a burn: 2.60 / 3.16 /
4.48 us (n=32). The A2 `moe_gemv` a2 of ~122 us is a 30-50 us host/tag
gap on a real 64-87 us scalar kernel.

## What the three binaries are

All three passed `tests/gpu_check.py` and `tests/moe_layer_bench.py`
with `failures=0`. Dequant stayed bit-exact (0 bf16 mismatches),
including Q2_K and Q3_K. GEMV, moe, and grouped stayed inside atol/rtol
5e-2. `moe_layer` vs the fp32 dequant + SiLU reference (x rounded to
bf16 first) stayed inside 8e-2. Worst gemv abs on the installed binary
is IQ4_XS M=8 at 3.278e-2.

| binary | schedule | measured |
| --- | --- | --- |
| A2 | scalar GEMV, three `moe_gemv` launches per layer | `reports/1A2_gguf.md` |
| b1 | fused `moe_layer`; DP4A vec form; one row per subgroup; no dense SLM | card 0, `c1_run.sh`, 72s, exit 0 |
| b2 | night `sgn`; 2-row moe/fused; dense SLM + M-chunk 4; shift-form DP4A | card 0, 04:27, 93s, exit 0, quiet CPU lock |
| b3 | b2 schedule with vec-form DP4A restored and memcpy 4/8-byte IQ loads | card 0, 05:28, 70s, exit 0. Quiet CPU lock was busy, so this run did not take it. Installed. `_C.so` is 6391032 bytes |

b3 is the plugin `.so` left on disk. The shift form in b2 is the one
that produced the Q5_K M=8 regression below. It is not installed.

`moe_layer` is one host call. The queue gets Q8(x), fused gate+up+SiLU
over top-k and the shared expert (expert ids read on device; an id
outside `[0, E)` adds nothing), Q8(hidden), down, then the routing sum.
No host sync between those submits. Signature is the C1 section of
`plugins/INTERFACE.md`.

DP4A is the vec-convert form that matches `dpct::dp4a`
(`strata/sycl_math.hpp:8`). A shift-and-mask form was measured in b2
and reverted. Finish formulas are unchanged and matched the GPU
tolerance on every binary: IQ2_XXS `sum*ls/8`; IQ2_XS and IQ2_S
`(s0*ls0+s1*ls1+(s0+s1)/2)/4` with integer division toward zero;
IQ3_XXS `(ls*sum+sum/2)/2`; IQ3_S `sum*sc`; IQ4_NL and Q2_0 plain
`sum * d * q8_d`.

Ported from Strata `iq_kernels.dp.cpp` (worktree line numbers in
`reports/03_strata_inventory.md`): ALU sign path at the night latch
`:5011-5076` (IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S use ALU masks; the ks
tables are the slower path and were left out), `sgn` `:2426`,
`q2_spread4` `:2431`, `iq4nl_mux4` `:2437`, unaligned load note
`:2416`. KREG schedule is `native_mmvq.dp.cpp:2236`. Not ported, on
purpose: pipelined KREG (MMVQ2 mask bit 0, measured regression in the
night lab), `STRATA_NIGHT_IQ3S_LO`, the 8 KB `kNightNl` IQ4_NL table,
cooperative / graph launch (night lab: a grid barrier was slower than
kernel boundaries; `reduce_over_group` is not a graph op).

## Dense GEMV

Shapes are the A2 bench. us is the a2 timer. % is percent of 608 GB/s.
`behind` is quoted only where it agrees with a2, which is the small-N
KREG rows. IQ3_S `behind` was 1692 us on b3 against an a2 of 1094 us
(burn leak); the a2 column is the one to use.

| op | shape | A2 | b1 | b2 | b3 |
| --- | --- | ---: | ---: | ---: | ---: |
| gemv Q5_K | M=1 N=248320 K=2560 | 808.4 us, 88.9% | 807.5, 89.0% | 833.4, 86.2% | 808.6, 88.9% |
| gemv Q5_K | M=8 same | 4303.6, 16.7% | 4330.5, 16.6% | 6917.0, 10.4% | 4316.4, 16.7% |
| gemv IQ3_S | M=1 N=248320 K=2560 token_embd | 1593.5, 28.2% | 2082.7, 21.6% | 1273.9, 35.3% | 1094.4, 41.1% |
| gemv IQ3_S | M=8 same | 6338.2, 7.1% | 16659.3, 2.7% | 3215.6, 14.0% | 2255.2, 19.9% |
| gemv IQ4_XS | M=1 N=12288 K=2560 | 128.1, 21.5% | 130.9, 21.0% | 144.2, 19.1% | 125.1, 22.0% (behind 128.3, 21.4%) |
| gemv IQ4_XS | M=8 same | 267.2, 10.3% | 266.2, 10.3% | 322.4, 8.5% | 267.3, 10.3% |
| gemv Q6_K | M=1 N=12288 K=2560 | 126.6, 33.5% | 127.6, 33.3% | 150.6, 28.2% | 121.4, 35.0% (behind 129.2, 32.8%) |
| gemv Q6_K | M=8 same | 281.5, 15.1% | 281.5, 15.1% | 329.6, 12.9% | 282.2, 15.0% |
| gemv Q4_K | M=1 N=12288 K=2560 | 128.0, 22.7% | 128.3, 22.7% | 156.5, 18.6% | 125.0, 23.3% (behind 128.4, 22.7%) |
| gemv Q4_K | M=8 same | 224.3, 13.0% | 224.3, 13.0% | 327.0, 8.9% | 210.9, 13.8% |

80% of 608 on these byte counts:

| weight | bytes | 80% us | b3 |
| --- | ---: | ---: | --- |
| Q5_K head M=1 | 437043200 | 898 | 808.6 us, 88.9%. At the target |
| IQ3_S embd M=1 | 273152000 | 561 | 1094 us, 41.1% |
| IQ3_S embd M=8, weight read once | 273152000 | 561 | 2255 us, 19.9% |
| IQ4_XS N=12288 M=1 | 16711680 | 34 | behind 128 us, 21.4% |
| Q6_K N=12288 M=1 | 25804800 | 53 | behind 129 us, 32.8% |
| Q4_K N=12288 M=1 | 17694720 | 36 | behind 128 us, 22.7% |

The head Q5_K M=1 row was already at the 80% goal in A2 (808 us vs
Strata nc1 760 us, 95% of 608, `mmvq_report.md`). b2's shift DP4A moved
the same weight at M=8 from 4330 us to 6917 us. b3 restored 4316 us.
Small-N KREG device time is the ~125-129 us number: `behind` matches
`a2`, so it is not the host gap from step 1. Strata's pipelined KREG
regressed in the night lab and was not ported.

IQ3_S moved. A2 M=1 was 1594 us. b1 dropped the SLM row stage and went
to 2083 us (M=8 16659 us). b2 put the row back in SLM and reused it
across M=4 chunks: 1274 / 3216 us. b3's vec DP4A plus the wide loads
landed at 1094 / 2255 us. M=8 is 2.1x the M=1 time, so the reuse is
real. 41% and 20% are short of 80%. The remaining time is the IQ3_S
decode arithmetic at M=1. Grouped XMX on one expert at M=128 is still
1.2-6.9 GB/s of the packed weight (`gpu_check` on b3), so DPAS is the
wrong tool for this GEMV.

## MoE layer

Flash-Next expert shape from the ISTA IQ3_XXS shard. Gate IQ2_XS,
up IQ2_XXS, down IQ4_NL, E=32, topk=10, I=640, K=2560, plus one shared
expert of the same packs. Routed+shared weight bytes for one token:
19993600. At 608 GB/s that is 33 us. At 80% it is 41 us.

`moe_gemv` a2 at T=1 stays 110-128 us on every binary. That is the
launch gap from step 1. The device column is `behind`.

IQ2_XS gate, T=1, topk=10, N=640, K=2560 (4.7 MB):

| timer | A2 scalar (step 1) | b1 | b2 | b3 |
| --- | ---: | ---: | ---: | ---: |
| behind us | 71.9 | 60.5 | 62.1 | 68.2 |
| a2 us | ~122 | 119.0 | 171.7 | 113.2 |

Fused `moe_layer`, same layer, T tokens, topk=10, one shared expert.
b1 and b2 held the quiet CPU lock. b3 did not.

| T | b1 a2 / behind | b2 a2 / behind | b3 a2 / behind |
| --- | ---: | ---: | ---: |
| 1 | 215.9 / 367.6 | 215.2 / 358.1 | 618.3 / 234.1 |
| 2 | 408.0 / 412.3 | 459.6 / 410.4 | 404.1 / 437.6 |
| 4 | 788.8 / 746.9 | 751.2 / 722.8 | 758.4 / 732.8 |
| 8 | 1424.8 / 1424.3 | 1356.4 / 1340.9 | 1373.8 / 1365.5 |

T=1 `behind` on b1 and b2 is the burn leak (behind > a2). The steady
number there is a2 216 us. On b3 the T=1 a2 minimum of 8 synced calls
is 618 us while T=2 a2 is 404 us and the T=1 behind sample is 234 us.
T=2, T=4, and T=8 agree between a2 and behind and match b1/b2. The
618 us sample is a host-timer spike on the run that missed the quiet
CPU lock. It is not a new device cost. The fused layer did not get
faster.

The printed GB/s in `moe_layer_bench` divides the one-token byte count
by the T-token time, so the T=8 figure (14.6 GB/s, 2.4%) is not the
bytes the kernel touched. Use the microseconds.

Fusion itself did what step 1 asked. Three A2 `moe_gemv` calls are
about 3 * 122 us = 366 us of a2 time. One `moe_layer` is 216 us a2 on
the quiet runs (b1, b2). The device work inside that call is still
about the sum of the three projections.

## Strata equivalent, and the 1.2x target

Sources, night-20261016, single-card microbenches. These are not
end-to-end tok/s.

Routed experts, `results/night-20261016/agents/moe_report.md`. C1
card0 is 10 groups / 20 entries (one card's share of T=4, topk=10).
Time tracks entries. 19-44% of 608.

| gate/up / down | case | us |
| --- | --- | --- |
| IQ3_S / Q2_0 | C1 card0, R2 | gu+sw 53.5, down 40.8, all 93.2 |
| IQ3_S / Q2_0 | C1 card1, R2 | all 134.8 |
| IQ2_XXS / Q2_0 | C1 card0, baseline | gu+sw 73.7, down 41.5, all 115.2 |
| IQ2_XXS / Q2_0 | C1 card1, baseline | all 135.4 |

Shared expert, `mmvq_report.md` round 3, us per layer at nc1 after
`STRATA_NIGHT_SHEXP=1`: IQ4_XS/IQ3_S/IQ4_NL 24.9 -> 18.7; IQ3_S/IQ3_S/IQ4_NL
27.6 -> 21.1. Trace medians in `trace_c1_analysis.md` (earlier binary):
IQ3_S shared gate/up 12.4 us, IQ4_NL shared down 12.4 us, Q2_0 shared
down 21.3 us.

Our fused call is 10 entries plus one shared expert, on one GPU. The
published Strata routed number at the nearest larger point is 93 us
for 20 entries (IQ3_S/Q2_0, C1 card0, R2), and their own report says
the time tracks entries. Half of that, plus the fused shared expert at
~19 us, is about 65 us. 1.2x of that estimate is about 80 us. 1.2x of
the published 93 us (20 entries, no shared) is 112 us. The quiet-run
fused T=1 number is 216 us. b3 T=2 (20 entries plus shared) is 404 us
a2 / 438 us behind, against 93 us for 20 routed entries. That is about
4x the published 20-entry figure, and about 3x the 65 us estimate for
our exact shape. The 1.2x target is missed.

The gate dot is the reason. Templated DP4A moved IQ2_XS T=1 from 71.9 us
to 60.5 us. Two-row Q8 reuse, `sgn`, and the memcpy loads left it at
62 us and then 68 us. Each pair still decodes two weight rows. The Q8
reload is 48 bytes. The decode is hundreds of instructions. Strata saw
the same bound: staging the weight tile in SLM was 0-30% slower, and
GU_LANES 16/32 was slower than 8.

## Q2_K and Q3_K

ggml id 10 and id 11. Host dequant matches gguf-py bit-exact. On b3,
card 0, UD-IQ3_XXS `blk.40.ffn_gate_exps` (Q2_K, K=2048) and
`blk.40.ffn_down_exps` (Q3_K, K=512): dequant 0 bf16 mismatches;
gemv M=1/3/8 max abs under 1.4e-2; moe under 1.0e-2; grouped under
5.3e-4. Same pass on b1 and b2. The float decode path was not retuned.
These types are the MTP expert tensors (on UD-IQ4_XS the MTP assignment
is swapped; both ids decode).

## Wall

The memcpy loads were the last ALU cut that the night source pointed
at for this schedule. The gate behind stayed at 68 us and the fused
layer at T=2/4/8 stayed on the b1/b2 line. Stopping here.

Next step, not started:

- IQ4_NL `kNightNl` (`iq_kernels.dp.cpp:5015`, NQ<120>) is the +4%
  down-only table. It does not move the IQ2 gate.
- A wider subgroup does not cut T=1 when each output row is a different
  expert. Strata's R2 win needs two rows that share an activation, which
  this kernel already does, and it measured as no change.
- XMX grouped GEMM at M=128 on one expert is 1-7 GB/s. It does not
  replace the M=1 dot.
- Small-N KREG at N=12288 is a ~128 us device time (behind matches a2)
  against a 34-53 us 80% target. The head at N=248320 M=1 is already
  88.9%. Pipelined KREG regressed in the night lab.
- IQ3_S dense is 1094 us at M=1 (41% of 608) and 2255 us at M=8 (20%).
  The next gain has to be fewer instructions in the IQ3_S decode, or a
  different numeric format. Another occupancy tweak on the current
  prep/apply split is not justified by the b2 to b3 gate result.
