# C12 decode schedule: types, graph shape, 35B serve

Owner: C12. Card 0 only, via `bin/gpu-run --card 0`. No commit.

## Installed binary

`_C.so`, `_C.graph.so`, and `_C.c12.so` are the same bytes:

```
sha256 b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979
10165216 bytes
```

Nothing had those files mapped at the copy. Preserved side copies:

| File | sha256 | What it is |
|---|---|---|
| `c12/_C.c12.measured.so` | `d46f8017…` | Per-type kernels, 1D grid. Gates passed. Graph flat vs C9. |
| `c12/_C.c12.ec3.so` | `457a262d…` | 3D + `use_root_sync` + one kernel per entry width. Graph instantiate failed. |
| `c12/_C.c12.fused.so` | `6a2e412b…` | 3D + `use_root_sync` + entry widths inside one kernel. Graph instantiate failed. |
| `c12/_C.c12.noroot.so` | `b3c15e14…` | Installed binary. |
| `c12/_C.c12.ecsplit.so` | `cd2882dc…` | 3D, no root sync, one kernel per entry width. Slower. |

## Fast path types

`moe_layer` signature is unchanged. `B70_MOE_SCHED=0` still selects the C1 kernels. With the schedule on, T<=8, topk<=16, I and K multiples of 32 and at most 8192, these types take it:

Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0.

Q8_0 and the seven C9 types use DP4A. The K-quants and IQ4_XS decode 32 weights and fma against the 36-byte Q8 block (`float d` + `int8 qs[32]`). Host dequant of all 14 types was bit-exact vs gguf-py (`c12/compare_cpu.log`, failures=0).

The 35B UD-IQ3_XXS routed experts are IQ2_S, IQ3_S, IQ4_XS, Q2_K, and Q3_K. All of those now hit this path. Shared experts in that file are Q6_K (40 layers) and one Q8_0 triple. The live runner still passes empty shared tensors into `moe_layer` and applies shared itself (`moe_runner.py`). This binary accepts shared Q6_K and Q8_0 when the caller passes them. Fusing them into `moe_layer` without disabling that runner path would double-count, so the serve was not changed.

## Gates

On the per-type binary (`d46f8017`, card 0, `c12/gates.log`, 340s, exit 0):

- `xpu-health` HEALTHY card 0 before and after.
- `gpu_check.py` failures=0 (atol 5e-2), including Q2_K and Q3_K.
- `moe_layer_bench.py` failures=0 (atol 8e-2).
- `c12_types.py` failures=0, 12 types, Q8_0 dequant mismatches=0.

`decode.hpp`, `dequant_fast.cpp`, and `kernels.cpp` were not edited after that gpu_check. Later binaries only changed `sched_moe.hpp`. Each of those reran `moe_layer_bench` and `c12_types` (failures=0, sched=1) before the graph number. No xe-reset on any of these runs.

## Fair graph, one card, K=2560 I=640, top-10 + shared

Microseconds per call, median of 5, cosine 0.99999 vs the Strata dump. `B70_MOE_PHASE` unset. Not under `cpu-run --quiet`.

IQ3_S/Q2_0 (gu=21, dt=42):

| T | Strata R2 | C9 / per-type 1D | Installed 3D | Per-entry-width |
|---|---:|---:|---:|---:|
| 1 | 112.89 | 186.95 | 180.17 | 185.76 |
| 2 | 148.10 | 277.77 | 267.04 | 320.79 |
| 3 | 202.93 | 398.45 | 385.36 | 378.78 |
| 4 | 261.64 | 535.64 | 515.13 | 595.23 |

IQ2_XS/IQ4_NL (gu=17, dt=20): installed 191.86 / 280.62 / 402.88 / 560.55, versus per-type 198.51 / 288.44 / 414.10 / 571.82.

Installed vs Strata R2 is about 1.60x at T=1 and 1.97x at T=4. Logs: `c12/fair4.log` (installed), `c12/fair5.log` (per-entry-width), `c12/gates.log` (per-type).

## What Strata does

R2 `native_expert_grouped` is three timed pieces, not one fused dot. IQ3_S/Q2_0 T=1: Q8(x) 9.58 us, grouped (gate/up, swiglu, Q8(hidden), down) 92.22 us, routing sum 11.09 us. The reduce is not fused into the down kernel. R2 does not use SLM. `STRATA_EXPERT_V2` is AMD-only and the SYCL path ignores it.

`launch_gu_port` tries the night IQ kernels first, then `launch_gu_rows` when `STRATA_EXPERT_ROWS=2`. The 112.9 us numbers are that rows kernel: subgroup 16, 8 lanes, 2 rows per lane, 3D grid `nd_range<3>(range(1, groups, gx*256), range(1, 1, 256))`, `use_root_sync`. Entry widths 4, 2, and 1 are constexpr passes inside the one kernel. The k-loop always dots both rows. The sub-warp kernels (`row_dot_80_sub16`, `row_dot_40_sub8`, `row_dot_20_sub16`) are not launched on that path.

`b70_prep` is already the night ALU decode (NQ masks, `q2_spread4`, `iq4nl_mux4`). Porting that prep again does not explain a 1.60x gap. IQR2, which is the same rows kernel with night IQ on, is 87.80 us at T=1, only about 1.28x under R2.

## What was tried

1. One specialized kernel per gate/up type and per down type, 1D grid. Graph time matched C9. The fair case has gate type equal to up type, so it was already one kernel.
2. `use_root_sync` on the 3D grid. Eager checks passed. `c9_fair` died in `capture_end`: `Failed to instantiate native UR executable graph` (`fair2.log`, `fair3.log`). Dropping the property made the same 3D grid capture (`fair4.log`).
3. One kernel per entry width, without `use_root_sync`, so the T=1 span-1 case would not carry the 4-wide register file. Graph captured. It was slower than the fused kernel at T=1, T=2, and T=4 (`fair5.log`). Empty width-4 and width-2 launches cost more than they save.
4. The 3D grid with both rows computed on every k step, entry widths inside one kernel, no `use_root_sync`. About 4 percent under the C9 number. This is the installed binary.

`B70_MOE_PHASE=1` prints host submit plus `event.wait()` per stage. It is not device time. The torch XPU queue does not keep profiling on. Warmed eager sums sit well above the graph number and are not the comparison. The graph microsecond is the matched figure.

At T=1 the host group count equals the live groups, so empty groups do not explain the 1.60x. At T=4 about 30 percent of the host groups can be empty, which is only part of the extra ratio.

## 35B graph serve

C7 recipe, card 0, container `vllm-fn-qwen36-gguf-c0`, port 18104:

- `GRAPH=1`, `FULL_DECODE_ONLY`, `FLASH_ATTN`, compilation mode 0, capture sizes 1,2,3,4,8
- `B70_XPU_CG_RECLAIM=500`, `B70_EAGER_QK=0`, `B70_GGUF_REF=0`
- `B70_GGUF_SO` = the installed `_C.graph.so`
- `max_num_seqs=4`, `max_model_len=33792`, `gpu_memory_utilization=0.90`
- Bench: `harness/bench_latency.py --model hotschmoe-dev --alias qwen3.6-35b-a3b-ud-iq3-xxs --decode-tokens 64 --prefill 1024,8192,32768`

The driver is `c12/serve_35b.sh`. Log `c12/serve_driver.log`, JSON `c12/bench_35b.json`. Health before the serve: HEALTHY card 0. Graph capture completed (FULL, then the decode sizes) and the server logged `Application startup complete`. No `DEVICE_LOST`. The bench ran under `cpu-run --quiet` (246s, exit 0).

Identity: `hotschmoe-dev` first, alias `qwen3.6-35b-a3b-ud-iq3-xxs`. Coherence: both samples `OK`, text ` Paris.\nThe capital of France is Paris.\nThe capital of France is`. MTP off.

| | ITL p50 | p90 | p99 | tok/s | TTFT |
|---|---:|---:|---:|---:|---:|
| C1, 64 decode tokens | 11.02 ms | 11.23 | 11.47 | 90.89 | 169.3 ms |
| C2 stream 0 | 16.61 ms | 17.02 | 75.01 | 55.24 | 271.9 ms |
| C2 stream 1 | 16.61 ms | 17.00 | 17.35 | 62.08 | 389.9 ms |

Prefill tok/s: 177.43 at 1017 tokens, 175.43 at 8185, 173.05 at 32762. TTFT 5.73 s, 46.66 s, 189.32 s.

C7's earlier graph serve on the old `_C.graph.so` (no C9 schedule) was C1 ITL p50 11.11 ms (90.16 tok/s) and C2 about 16.5 ms/stream. This binary matches that. The fair-layer 4 percent does not show up as an end-to-end ITL change on this 35B shape (K=2048, I=512, topk=8). C2 stream 0 p99 is one slow chunk (75 ms); the p50 matches stream 1.

The container was `docker stop`ped after the bench. Post-serve `xpu-health` waited behind that lease and then behind H1, then reported HEALTHY card 0 (13s, exit 0). Driver exit 0. The container is gone. No xe-reset.

## Shared diffs

`kernels.cpp`: one `B70_DISPATCH(B70_Q8_0)` line. Q8_0 is not added to the KREG runtime list (that else-path is IQ4_XS and would be wrong). `dequant_fast.cpp`: one `case B70_Q8_0`. `decode.hpp`, `iq_dot.hpp`, `row_moe.hpp`, `types.py`, `compare_cpu.py` gained Q8_0. `grouped_prefill.cpp` and `bind.cpp` were not edited by C12.
