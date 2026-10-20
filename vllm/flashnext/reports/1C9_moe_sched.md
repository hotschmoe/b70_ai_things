# C9: grouped decode schedule vs Strata, one card

Card 0, 2026-10-20. C5 showed the 198 us plugin fused layer versus the 93.5 us Strata grouped call is kernel schedule. This stage ports that schedule into `b70_moe_layer` and times the whole MoE layer on one card.

## Metric

Per-token whole-MoE-layer device time at Flash-Next decode: top-10 routed gate+up+SiLU+down, plus the shared expert, T=1..4. One card runs every entry. K=2560, I=640, router weight 0.1, shared weight 1.

Live work (ids are dense; E equals the group count):

| T | routed groups | routed entries | shared entries |
|---:|---|---:|---:|
| 1 | 10 x 1 | 10 | 1 |
| 2 | 6 x 2 + 8 x 1 | 20 | 2 |
| 3 | 4 x 3 + 6 x 2 + 6 x 1 | 30 | 3 |
| 4 | 2 x 3 + 8 x 2 + 18 x 1 | 40 | 4 |

T=4 is the published card0 histogram plus the published card1 histogram, run on one card. Gate and up are the same ggml type on both sides, because `native_expert_grouped` takes one gate/up type. The eager plugin bench further down uses the split pair IQ2_XS / IQ2_XXS / IQ4_NL.

## How each column was timed

Strata R2 whole layer is the sum of three medians. Each piece is 48 launches, 5 trials, `q.wait()` after every launch, so the number includes that sync. Pieces are Q8(x), `native_expert_grouped` (night rows 2, IQ off), and the routing sum. The T=1 grouped piece is 92.22 us (IQ3_S/Q2_0) and 91.05 us (IQ2_XS/IQ4_NL). A command graph of the full plan, shared expert included, resets the device on replay. That graph was left as a wall. The control graph (no shared, Q8 outside the timer, cap 40) does replay: phase-1 60.14 us, base 105.06 us, R2 92.39 us (`c9/fair_ctrl.log`). C5's binary on this card today is R2 92.55 us, IQR2 69.69 us, gu 60.03 us, all 103.24 us (`c9/c5_anchor.log`).

The FAIR lines print `groups=40` and `entries=43/54/65/76`. Those are padded launch caps (`cap_g = max(live groups, 40)`, `cap_e = max(live entries + 32, 40)`), not the live counts in the table above. The swiglu loop stops at the live entry count.

IQR2 is the same rows=2 schedule with `STRATA_NIGHT_IQ=1`. On these weights R2 and IQR2 dumps differ in 0 bits. IQR2 is faster. This plugin ports the lanes-8 R2 launch, not the IQ=1 schedule.

Plugin columns are `torch.xpu.XPUGraph` replay of 48 `moe_layer` calls, median of 5, device events, microseconds per call. `B70_GGUF_SO=_C.c9.so`. sched=1 is the new schedule (env unset or 1). sched=0 is the C1 fused kernels. The latch is per process.

## Whole layer, microseconds per call (per token)

IQ3_S gate+up, Q2_0 down. Strata log `c9/fair_all5.log`. Plugin logs `c9/ours6_sched1.log`, `c9/ours6_sched0.log`.

| T | Strata R2 | per tok | Strata IQR2 | plugin sched=1 | per tok | plugin sched=0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 112.89 | 112.89 | 87.80 | 186.32 | 186.32 | 192.80 |
| 2 | 148.10 | 74.05 | 116.62 | 278.53 | 139.26 | 351.71 |
| 3 | 202.93 | 67.64 | 156.15 | 400.90 | 133.63 | 494.24 |
| 4 | 261.64 | 65.41 | 206.20 | 534.72 | 133.68 | 660.14 |

IQ2_XS gate+up, IQ4_NL down.

| T | Strata R2 | per tok | Strata IQR2 | plugin sched=1 | per tok | plugin sched=0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 110.79 | 110.79 | 99.98 | 196.31 | 196.31 | 239.78 |
| 2 | 168.87 | 84.43 | 132.26 | 289.93 | 144.96 | 422.77 |
| 3 | 231.10 | 77.03 | 179.39 | 417.92 | 139.31 | 584.28 |
| 4 | 310.50 | 77.62 | 236.45 | 571.06 | 142.76 | 807.23 |

Strata piece split, R2, microseconds (q8 + grouped + reduce = whole):

| pair | T | q8 | grouped | reduce | routed grouped, no shared |
|---|---:|---:|---:|---:|---:|
| IQ3_S/Q2_0 | 1 | 9.58 | 92.22 | 11.09 | 88.45 |
| IQ3_S/Q2_0 | 2 | 9.18 | 127.50 | 11.42 | 122.50 |
| IQ3_S/Q2_0 | 3 | 9.44 | 180.83 | 12.66 | 155.21 |
| IQ3_S/Q2_0 | 4 | 9.77 | 239.01 | 12.86 | 218.92 |
| IQ2_XS/IQ4_NL | 1 | 8.95 | 91.05 | 10.79 | 87.68 |
| IQ2_XS/IQ4_NL | 2 | 9.20 | 148.49 | 11.17 | 137.93 |
| IQ2_XS/IQ4_NL | 3 | 9.43 | 208.91 | 12.76 | 173.64 |
| IQ2_XS/IQ4_NL | 4 | 9.64 | 287.89 | 12.96 | 247.71 |

sched=1 over sched=0 on the fair graphs: IQ3_S/Q2_0 is 1.03x / 1.26x / 1.23x / 1.23x at T=1/2/3/4. IQ2_XS/IQ4_NL is 1.22x / 1.46x / 1.40x / 1.41x. Strata R2 over sched=1: IQ3 1.65x / 1.88x / 1.98x / 2.04x, IQ2 1.77x / 1.72x / 1.81x / 1.84x.

## Agreement with Strata's output

Same weights, same ids, same x (already rounded to bf16). Plugin y versus the Strata R2 dump:

| pair | T | max abs | mean abs | cosine | bulk max rel | fraction of bulk over 8e-2 |
|---|---:|---:|---:|---:|---:|---:|
| IQ2_XS/IQ4_NL | 1 | 14.66 | 1.70 | 0.999999 | 0.304 | 0.0063 |
| IQ3_S/Q2_0 | 1 | 1.039 | 0.131 | 0.999999 | 0.172 | 0.0043 |
| IQ2_XS/IQ4_NL | 2 | 29.67 | 3.89 | 0.999992 | 1.153 | 0.0197 |
| IQ3_S/Q2_0 | 2 | 2.725 | 0.442 | 0.999994 | 1.105 | 0.0203 |
| IQ2_XS/IQ4_NL | 4 | 28.82 | 3.16 | 0.999995 | 1.120 | 0.0170 |
| IQ3_S/Q2_0 | 4 | 3.845 | 0.348 | 0.999996 | 0.847 | 0.0166 |

sched=0 prints the same abs and cosine as sched=1 on each dump. The two plugin schedules agree with each other. They agree with Strata at cosine 0.99999. They are not bit-identical. Bulk max relative error (elements with |ref| at least 1% of mean |ref|) reaches about 1.15, and about 2% of those elements exceed relative 8e-2. The 8e-2 gate is the plugin against its own CPU reference, which passed. Strata's Q8 block is half2 (`block_q8_1`); the plugin block is a float scale plus 32 int8 codes. SiLU on the Strata path uses `sycl::native::exp`. The plugin uses `sycl::exp`.

## What landed in the plugin

`csrc/sched_moe.hpp`, called from `b70_moe_layer` when the fast path matches. Signature unchanged (`plugins/INTERFACE.md`, C9 section).

- Subgroup 16, 8 lanes, two rows per rib (row and row+32). Chunks of 4, then 2, then 1 entries. Gate and up share one load of the activation.
- 36-byte Q8: `struct { float d; int8_t qs[32]; }`. The KREG 48-byte block is unchanged.
- Expert ids are grouped by a device kernel. Invalid ids contribute 0. The shared expert is one extra group, routing weight 1. Duplicate ids on one token stay separate entries.
- `reserve_decode` also reserves this scratch. A grow during capture throws.
- T=1 stays on this path. The eager A/B at T=1 is a tie; the fair graph is a small win.

Build output for this work is `b70_gguf_xpu/_C.c9.so`, sha256 `8a76f559e65981fe056c358e9094e4252f77d75915e67a9f2a9bb006a96ac63b`, 7574840 bytes, 07:39. C8's later link of the same `kernels.cpp` installed the schedule into the default `_C.so`, sha256 `086595a01babbc366b3d5366147464ebd24022fba0b65d83b1ff530489c70d55`, same size, 07:40. That file was not reverted. `B70_MOE_SCHED=0` still selects the C1 kernels. Prefill sources were not edited.

## Gates

On `_C.c9.so`, earlier the same day, card 0:

- `gpu_check.py` atol/rtol 5e-2, `c9/gpu_check.log`, failures=0.
- `moe_layer_bench.py` atol/rtol 8e-2, `c9/moe_layer_bench.log`, failures=0.
- `compare_cpu.py` bit-exact dequant, `c9/compare_cpu.log`, 13/13 mismatches=0.
- `graph_probe.py`, `c9/graph_probe.log`, GRAPH_PROBE_OK.

Clean eager A/B after the card recovered from earlier device-lost events. Shape E=32, topk=10, shared expert, IQ2_XS / IQ2_XXS / IQ4_NL, I=640, K=2560. failures=0 both processes.

| T | sched=1 a2 us | sched=0 a2 us |
|---:|---:|---:|
| 1 | 230.7 | 234.7 |
| 2 | 289.9 | 397.6 |
| 4 | 492.7 | 735.6 |
| 8 | 877.3 | 1426.8 |

Logs: `c9/moe_bench_sched1c.log`, `c9/moe_bench_sched0c.log`.

## 35B serve

Not run. A header scan of `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (753 tensors) gives shared experts Q6_K x120 and Q8_0 x3. Routed experts are IQ2_S x78, IQ3_S x39, IQ4_XS x3, Q2_K x2, Q3_K x1. The fast path requires every used type, including the shared expert, to be DP4A (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0). Q6_K and Q8_0 stay on the C1 kernels, so `b70_sched_try` returns false on every layer of that file. C7's one-card FULL_DECODE_ONLY serve (JOURNAL 07:21, C1 ITL 11.11 ms) uses `_C.graph.so`, which does not contain this schedule.

## Walls

- Full-plan Strata command graph (Q8 + grouped + reduce, shared expert, 48 layers) finalizes and then loses the device on replay. An unwaited burst of the same launches also lost the device before the memcpy fix. After each expert-blob copy waits, layer 0, 1, 2, and 47 return, and the synced-eager table above completes (`FAIR_DONE rc=0`).
- Passing `-fsycl-default-sub-group-size=32` on the Strata link line rewrote SG16 kernels and lost the device. The working link matches C5: the flag is on the compile of `fair_strata.cpp` only.
- Plugin versus Strata is cosine 0.99999, not an 8e-2 max-rel match. Closing the remaining 1.7x-2.0x needs the Strata inner loop (and, for the IQR2 column, the IQ=1 schedule), not another subgroup-size flag.

Health: `bin/xpu-health --card 0` HEALTHY before and after the plugin graphs (`c9/run_ours7.outer.log`, gpu-run exit 0 in 44 s). No xe-reset. Card 1 was D3's for the whole window.
