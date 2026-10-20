# O3: GGUF MoE decode gap vs Strata, root cause and fix

Owner: O3. Card 0 only (`bin/gpu-run --card 0`, ZE_AFFINITY_MASK=0). No commit.
Work dir: `/mnt/vm_8tb/b70/build/vllm-flashnext/o3/`.

## Verdict

The 1.6x-2.0x gap was not the schedule. It was the weight loads. `b70_prep`
(csrc/iq_dot.hpp) read the 8- and 4-byte fields of each IQ/Q2_0 block with
`__builtin_memcpy` (`b70_loadu64` / `b70_loadu32`). Every block starts only
2-byte aligned (block sizes 110, 98, 82, 74, 66, 18 bytes). IGC lowered those
copies to single `load.ugm.d64` / `load.ugm.d32` LSC messages at misaligned
addresses. Strata reads the same fields with `load8_a2` / `load4_a2`
(iq_kernels.dp.cpp:371): two or three aligned dword loads and a funnel shift.
With Strata's loads, our gate/up kernel goes from 123 us to 44 us at T=1. The
same geometry, subgroup, lanes, and dot are kept.

The second cost was the plan kernel. It was a `single_task` doing the O(n^2)
grouping on one work-item: 8.5 us at T=1 and 74 us at T=4. It is now one
parallel work-group, and that launch also quantizes x. SiLU is fused with the
hidden Q8. A layer is now 5 launches; it was 7.

Whole layer in the fair graph, IQ3_S/Q2_0: 180/267/386/516 us went to
69/100/141/179 us at T=1..4. Strata R2's whole layer is 113/148/203/262 us. On
device time alone, ours is 1.07x Strata's best schedule (IQR2) at T=1 and
1.01x at T=4. The 35B UD-IQ3_XXS graph serve's C1 ITL went from 11.02 ms to
9.34 ms.

## Evidence: per-kernel device time

`unitrace -d` (pti-gpu build in strata-ab). Eager calls, 400 calls per process,
fair dump IQ3_S gate/up and Q2_0 down, 10 routed experts plus the shared
expert, K=2560, I=640. Strata ran `fair_strata` (C9 bench with a T/pair filter,
`o3/fair_strata.cpp`) in image e41d59c0, on the same card, the same day.
Values are average microseconds per call.

T=1:

| kernel | plugin before (C12 b3c15e14) | plugin after (O3 be76661b) | Strata R2 | Strata IQR2 |
|---|---:|---:|---:|---:|
| plan | 8.54 (single_task) | 2.40 (plan + q8 fused) | host plan | host plan |
| q8(x) | 1.89 | (fused) | 1.82 | 1.82 |
| gate/up | 123.11 | 44.42 | 45.28 (gu_rows<21,8,2>) | 42.73 (gu_n<121>) |
| silu (+q8h) | 0.89 + 3.17 | 1.30 (fused) | 2.09 | 2.12 |
| down | 38.03 | 16.49 | 33.62 (down_rows<42>) | 14.71 (down_n<42>) |
| reduce | 3.49 | 3.66 | 2.35 | 2.35 |
| sum | 179.1 | 68.3 | 85.2 | 63.7 |

T=4 (28 routed groups / 40 entries, plus shared):

| kernel | before | after | Strata R2 | Strata IQR2 |
|---|---:|---:|---:|---:|
| plan (+q8) | 73.64 + 1.98 | 5.51 | host | host |
| gate/up | 319.68 | 114.70 | 121.78 | 120.97 |
| silu + q8h | 1.37 + 3.41 | 1.70 | 3.03 | 2.83 |
| down | 107.08 | 45.89 | 91.24 | 41.31 |
| reduce | 6.00 | 6.32 | 4.61 | 4.61 |
| sum | 513.2 | 174.1 | 222.7 (incl. q8 2.0) | 171.7 |

Logs: `o3/ut_ours_base_T{1,4}.*.txt`, `o3/ut_strata_T{1,4}.1.txt`,
`o3/ut_mb_f1_T{1,4}.*.txt`, `o3/mb2.log`.

ISA check (CPU only). The SPIR-V was dumped with `-save-offload-code`, then
`ocloc compile -device bmg-g31` with `IGC_ShaderDumpEnable`. In
`b70_sched_gu_k<21>` the IQ3_S `qs` read is a `load.ugm.d64.a64` at
`row + j*110 + 2 + 8*ib`, an address that is only 2-byte aligned. The fp16
scale is two `d8u32` byte loads, and the software `b70_fp16_to_fp32` adds a
denormal-normalize loop (`jmpi` back-edge) per block. Static instruction count
(578 vs 523) and dp4a count (112 each) are close to Strata's
`launch_gu_rows_k<21,8,2>`. Spills, private memory, SLM, and GRF (128) are
identical. The gap is per-message memory cost, not instruction count. Dumps
are in `o3/isa/dump_*`.

## A/B that isolates the cause

Standalone microbench `o3/mb/mb.cpp` (no torch). It links `kernels.cpp`,
reads the C9 fair dumps, and reports graph replay of 48 calls (median of 5)
and agreement with Strata's y. IQ3_S/Q2_0, us per call (`o3/mb1.log`,
`o3/mb2.log`, `o3/mb3.log`):

| variant | T=1 | T=2 | T=3 | T=4 |
|---|---:|---:|---:|---:|
| base (= C12 source) | 184.5 | 272.2 | 388.5 | 519.6 |
| fp16 hw convert only (a0h1) | 179.4 | 266.2 | 399.5 | 521.8 |
| aligned loads only (a1h0) | 90.1 | 136.7 | 195.3 | 271.7 |
| aligned + hw half (a1h1) | 84.2 | 127.8 | 181.0 | 255.4 |
| + parallel plan/q8, silu/q8h fused (f1) | 70.0 | 103.4 | 145.9 | 189.8 |
| f1 + `#pragma unroll 2` k-loop | 71.6 | - | - | 195.4 |
| f1 with 16 lanes per row | 76.9 | - | - | 211.5 |

The aligned loads alone account for about 2x. The hardware half convert adds
about 7 percent once the loads are fixed. The fusion removes the plan
serialization. Unroll 2 and 16 lanes were slower, so the installed binary
does not use them. The outputs of all variants agree with the base bit for
bit (same cos and max_abs vs the Strata dump); 16 lanes changes k order and
is not bit-identical.

## Strata decode path, item by item

Source: `strata-ab/src-ep/sycl/src/kernels/cuda/iq_kernels.dp.cpp` (night-20261016).

- Work distribution. 3D grid `(1, groups, gx*256)`, WG 256, SG16. Each rib of
  8 lanes takes 2 rows, RB=32 apart (`native_gu_kernel_rows`, line 2302, RPL=2).
  Groups are experts, and entries in a group take constexpr passes of 4, then
  2, then 1. The plugin had the same geometry since C9/C12. Not the cause.
- k loop. Lane `sub` takes 32-weight part `sub` of each 256-superblock
  (`k = sub; k += 8`), so the 8 lanes of a rib read one superblock per step.
  The plugin used the same mapping. Not the cause.
- Activation. `block_q8_1` is 36 bytes (half2 ds plus 32 int8). The plugin's
  B70Q8S is 36 bytes (float d plus 32 int8). The activations take 8 aligned
  int loads in both. Not the cause.
- Dot. `dpct::dp4a` byte-unpack emulation, which IGC matches to DP4A, in both:
  112 dp4a in both kernels. Not the cause.
- Weight loads. Strata uses `load8_a2` / `load4_a2` (aligned dwords) and
  `sycl::half` -> float. The plugin used unaligned `memcpy` d64/d32 and
  software fp16. This is the cause (above).
- Fusion and launches. Strata does Q8(x), gu, swiglu+Q8(h), down, and reduce:
  5 launches, with the plan built outside the timed region. The plugin did
  plan (single_task), Q8, gu, silu, Q8(h), down, and reduce: 7 launches, with
  a serial plan. That is now 5, with a parallel plan.
- Night IQ (`STRATA_NIGHT_IQ=1`, `NQ<121>`). It uses the table/ALU sign masks
  and walks block pointers. The plugin's prep already had the ALU masks. On
  gate/up that schedule gains only 45.3 -> 42.7 us. On the Q2_0 down it gains
  33.6 -> 14.7 us. Most of that is the same aligned-load fix (`down_rows`
  still uses `Multi<42>`'s `byte_level_permute` emulation). The plugin's down
  is now 16.5 us.

## What changed (plugin source)

- `csrc/iq_dot.hpp`: added `b70_ld8_a2`, `b70_ld4_a2` (Strata load8_a2 /
  load4_a2), and `b70_h2f` (aligned 16-bit load + `sycl::half` convert). All
  DP4A preps (Q8_0, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL, Q2_0) use them.
  `-DB70_LOAD_A2=0` / `-DB70_HW_HALF=0` restore the old loads for A/B.
  `b70_prep` also feeds the C1 fused kernels (`B70_MOE_SCHED=0`) and the dense
  and moe GEMV paths, which get the same fix.
- `csrc/sched_moe.hpp`: `b70_sched_plan_q8` is work-group 0 building the plan
  (ids in SLM, one work-item per slot, the same group and entry order as the
  old single_task) while the other work-groups quantize x, 8 lanes per block.
  `b70_sched_silu_q8` is SiLU*up plus Q8 of the hidden row, 8 lanes per block,
  with no float hidden write. Both give byte-identical Q8 to the old kernels.
  `-DB70_SCHED_FUSE=0` restores the separate launches.
- `csrc/kernels.cpp`: `b70_check_a2` throws on an odd weight base pointer in
  gemv, moe_gemv, and moe_layer (the aligned loads need an even block address).
- Unchanged: `decode.hpp`, `row_moe.hpp`, `grouped_prefill.cpp`, `bind.cpp`,
  the op signature, `B70_MOE_SCHED`, and `B70_MOE_PHASE` (the q8/q8h stamps
  are now 0 because those stages are fused).

## Gates (`o3/gates.log`, card 0, `_C.o3.so`)

- `xpu-health --card 0` HEALTHY before and after.
- `gpu_check.py` VERDICT failures=0.
- `moe_layer_bench.py` failures=0 (sched=1), and failures=0 with `B70_MOE_SCHED=0`.
- `c12_types.py` failures=0 covered=12 (all 14 sched types incl. Q8_0 shared).
- `c9_fair.py`: cos 0.999992-0.999999 vs the Strata R2 dump, with max_abs,
  mean_abs, and frac_bulk identical to C12's to every printed digit.
- `graph_probe.py` GRAPH_PROBE_OK (XPUGraph capture and replay).
- `compare_cpu.py` (CPU, under cpu-run) VERDICT failures=0 types=14/14.
- FULL_DECODE_ONLY replay: the 35B serve below captured sizes 1,2,3,4,8 and
  decoded coherently.

## Fair graph, timed under `cpu-run --quiet` (`o3/fair_quiet.log`)

Same process window, `_C.o3.so` then the pre-O3 `_C.so` (C12, b3c15e14).
`c9_fair.py`: XPUGraph replay of 48 calls cycling 8 weight copies, median of
5, us per call. Strata R2 is the C9 published whole layer (q8 + grouped +
reduce, each with a host wait).

IQ3_S gate/up, Q2_0 down:

| T | Strata R2 | before | after | after / Strata | before / after |
|---:|---:|---:|---:|---:|---:|
| 1 | 112.89 | 180.10 | 68.64 | 0.61 | 2.62 |
| 2 | 148.10 | 267.01 | 100.32 | 0.68 | 2.66 |
| 3 | 202.93 | 385.82 | 141.41 | 0.70 | 2.73 |
| 4 | 261.64 | 515.81 | 178.59 | 0.68 | 2.89 |

IQ2_XS gate/up, IQ4_NL down:

| T | Strata R2 | before | after | after / Strata |
|---:|---:|---:|---:|---:|
| 1 | 110.79 | 192.19 | 81.36 | 0.73 |
| 2 | 168.87 | 281.41 | 117.39 | 0.70 |
| 3 | 231.10 | 403.84 | 168.33 | 0.73 |
| 4 | 310.50 | 565.37 | 216.73 | 0.70 |

The target was within 1.15x of Strata at T=1..4. Against the stated R2
numbers ours is now 0.61-0.73x. Strata's R2 figure includes three host waits,
so the device-only sums above are the stricter check: 68.3 vs 63.7 us (1.07x)
at T=1 and 174.1 vs 171.7 us (1.01x) at T=4 against Strata's best (IQR2).

## 35B UD-IQ3_XXS graph serve (C12 recipe, `o3/serve_35b.sh`)

The recipe is C12's: card 0, GRAPH=1, FULL_DECODE_ONLY, FLASH_ATTN,
compilation mode 0, capture sizes 1,2,3,4,8, `B70_XPU_CG_RECLAIM=500`,
`B70_GGUF_REF=0`, `B70_EAGER_QK=0`, max_num_seqs 4, max_model_len 33792,
gpu_memory_utilization 0.90. `B70_GGUF_SO` = the installed `_C.graph.so`
(be76661b). The bench is `bench_latency.py --decode-tokens 64` under
`cpu-run --quiet`. Prefill was 1024 only, to keep the window short; C12 also
ran 8192 and 32768.

Identity: `hotschmoe-dev` first, alias `qwen3.6-35b-a3b-ud-iq3-xxs`.
Coherence: both samples OK (` Paris.\nThe capital of France is Paris...`).
MTP off. No DEVICE_LOST. Health HEALTHY before and after.

| | C12 (b3c15e14) | O3 (be76661b) |
|---|---:|---:|
| C1 ITL p50 | 11.02 ms | 9.34 ms |
| C1 p90 / p99 | 11.23 / 11.47 | 9.50 / 9.60 |
| C1 tok/s | 90.89 | 107.03 |
| C2 ITL p50 (stream 0 / 1) | 16.61 / 16.61 | 12.59 / 12.60 |
| C2 tok/s (stream 0 / 1) | 55.24 / 62.08 | 73.46 / 81.26 |
| prefill 1024 tok/s | 177.43 | 177.14 |

The 35B routed experts are IQ2_S/IQ3_S (DP4A, fixed) plus Q2_K/Q3_K/IQ4_XS
(float decode32 path, unchanged). The shared Q6_K expert is still applied by
the runner. C1 gains 1.68 ms per token (-15 percent). C2 gains 4.0 ms per
step (-24 percent).

The C12 driver's post-serve health check waits on the lease that the serve
container itself still holds. `o3/serve_35b.sh` now stops the container
before that check. This run's container was stopped by hand, and health then
passed.

## Installed library

`plugins/b70_gguf_xpu/b70_gguf_xpu/_C.so`, `_C.graph.so`, and `_C.o3.so` are
the same bytes:

```
sha256 be76661bd120aa2f6c7523ae3c08813ba56d6114660f3d7f31183c32e5e861eb
10135104 bytes
```

Installed with cp + mv (new inode); no process had the old files mapped. The
previous binary stays as `_C.c12.so` (b3c15e14...) for revert. Note: C7b's
Flash-Next graph window (`logs/c7b-fn-graph-window.sh`) points at
`_C.graph.so` and had not started when this was installed. It will load the
O3 binary.

## Stage 2 (16:05): K-quants and IQ4_XS on DP4A, bandwidth tries

The coordinator committed stage 1 as ad52e8ef. Stage 2 targets the types
that still went through `b70_decode32` (decode.hpp): byte-wise loads, 32
floats decoded per part, and 32 fma per entry. Those are Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, and IQ4_XS.

Change, `csrc/iq_dot.hpp` only (decode.hpp is shared with the host dequant
and is untouched):

- `b70_prep` now has those six types. It builds 8 signed-byte words per
  32-weight part with aligned-dword loads (`b70_ld4_a2`), plus integer
  per-16 scales, mins, and the fp16 `d`/`dmin`. The byte maps are exact:
  - Q3_K: `q + 4*hbit - 4` via `(x + 0x7C) ^ 0x80`.
  - Q6_K: `q6 - 32` via `(x + 0x60) ^ 0x80`.
  - IQ4_XS: the IQ4_NL mux with `d*(ls-32)`.
- `b70_apply_u` computes `d*(s0*sc0 + s1*sc1) - dmin*(a0*m0 + a1*m1)`. The
  `a` sums are dp4a against 0x01010101, as in ggml `vec_dot_*_q8_1`.
- `b70_dp4a_type` includes them, so the sched path, the C1 fused path
  (`B70_MOE_SCHED=0`), and the non-KREG dense GEMV all use the DP4A form.
  `-DB70_KQ_DP4A=0` restores decode32 + fma.
- `sched_moe.hpp`: the R-row dot is now generic (`b70_sched_rows<TYPE, EC,
  R>`, `-DB70_SCHED_RPL`). It is the same code at R=2, with the same outputs
  and timings as stage 1.

Microbench (`o3/kq2.log`). Synthetic K-quant dumps (`o3/gen_kq.py`): K=2048,
I=512 (the 35B shape), top-10 plus shared, one type for every matrix, ids and
x from the C9 dumps. Graph replay, us per layer. The old column is the same
source built with `-DB70_KQ_DP4A=0`. New vs old output cosine is 1.000000
for every case.

| type | T=1 old | T=1 new | T=4 old | T=4 new |
|---|---:|---:|---:|---:|
| Q2_K | 485.5 | 49.3 | 1179.3 | 126.0 |
| Q3_K | 268.2 | 79.4 | 684.8 | 196.4 |
| Q4_K | 96.8 | 59.2 | 251.6 | 148.7 |
| Q5_K | 119.4 | 78.7 | 302.1 | 193.2 |
| Q6_K | 581.9 | 88.4 | 1396.6 | 213.1 |
| IQ4_XS | 110.5 | 81.8 | 280.4 | 198.7 |

Gates on `_C.o3b.so` (`o3/gates_b.log`, card 0). gpu_check, moe_layer_bench
(sched 1 and 0), c12_types (12 cases), and graph_probe all have failures=0.
compare_cpu is 14/14. Health was HEALTHY before and after. The c9_fair
IQ3_S/Q2_0 graph is 68.4/99.8/141.4/176.8 us, unchanged from stage 1 as
expected.

35B UD-IQ3_XXS graph serve, same recipe (`o3/serve_driver_o3b.log`,
`o3/bench_35b_o3b.json`). Identity and coherence OK. C1 ITL p50 is 9.30 ms
(107.45 tok/s), and C2 is 12.53/12.54 ms per stream. Stage 1 was 9.34 and
12.6. The change is within noise: that file has only 6 K-quant/IQ4_XS
routed tensors (Q2_K x2, Q3_K x1, IQ4_XS x3 of 123), and its Q6_K shared
expert is applied by the runner, not by `moe_layer`. The gain matters for
layers and models that are mostly K-quant, such as the 35B shared Q6_K once
it is fused into `moe_layer`.

Installed: `_C.so` and `_C.graph.so` are `_C.o3b.so`, sha256
`e9e5282d9d4cc3def64d17516e847021d623735fde357fef690208ab291befa1`
(10221368 bytes). `_C.o3.so` (be76661b, stage 1) is left as is because X1's
proxy window pins it. `_C.c12.so` is still b3c15e14.

Gate/up bandwidth push, microbench graph us, IQ3_S/Q2_0, T=1/2/3/4
(`o3/kq1.log`, `o3/mb3.log`):

| variant | T=1 | T=2 | T=3 | T=4 |
|---|---:|---:|---:|---:|
| installed (RPL 2, 8 lanes) | 69.9 | 103.7 | 146.5 | 190.3 |
| RPL 4 (one wave at T=1) | 70.8 | 115.1 | 158.2 | 213.2 |
| k-loop unroll 2 | 71.6 | - | - | 195.4 |
| 16 lanes per row | 76.9 | - | - | 211.5 |

None of the three helps. RPL 4 has no GRF spills (ISA check `o3/isa/r4`,
flag spills only) and still loses. Gate/up stays at about 44 us for 15.5 MB
(about 350 GB/s), the same as Strata's best (night gu_n 42.7 us). Going past
that needs a different load scheme, such as a cooperative subgroup block
load of whole superblocks plus a shuffle. That is a larger change, not a
cheap one, so it is left as a work item (estimate: 1-2 days).

## Stage 3 (19:45): shared expert fused, gate/up ALU cut, batch-invariant reduce

The coordinator committed stage 2 as 952f1971. Stage 3 work dir:
`o3/src`. Logs are under `o3/`.

### (a) Shared expert in the decode op

Shared-expert types, from the GGUF headers:

- 35B UD-IQ3_XXS: Q6_K gate/up/down on 40 layers and Q8_0 on one. Gate
  input `ffn_gate_inp_shexp` is F32 (BF16 on one layer).
- Flash-Next IQ3_XXS: mixed per layer. Gate is IQ3_S/IQ4_XS/Q4_K/Q5_K/Q6_K,
  up is IQ3_S/IQ4_XS/Q4_K/Q5_K/Q6_K, down is IQ4_NL/Q2_0. Gate input is
  BF16 [2560]. All of these are sched types.

Plugin changes:

- New op `b70_gguf::moe_layer_sh` (`bind.cpp`, `kernels.cpp`
  `b70_moe_layer2`, `ops.py` wrapper and fake). It computes the routed
  experts, the shared expert, and the per-token shared gate
  sigmoid(x . w_sg) in one call. It returns bf16 [2, T, K]: [0] is the
  routed sum, [1] is the gated shared output.
- The plan launch gets T extra work-groups that compute the gate dot.
  `moe_layer` is unchanged.

Fork change (`vllm/models/qwen4_exp/gguf/moe.py`, A1's file; small and
opt-in):

- With `B70_FUSE_SHARED=1`, `mk_can_overlap_shared_experts` is True. The
  runner then skips its own shared forward (MK_INTERNAL_OVERLAPPED order),
  and `apply()` stores [1] as the shared output. The runner still adds the
  two outputs.
- Anything not fusable runs the shared module inside `apply()` instead:
  prefill (T>8), an EP expert_map, a non-b70 shard, a ref path, or a shape
  mismatch.
- A one-line note per decision is printed (`b70_gguf: shared-expert
  FUSED/separate`).
- Default is off. EP (Flash-Next DP=2 EP) always takes the separate path,
  because fusing there would add the shared expert on every rank.

Correctness (`tests/o3_shared.py`): [0] vs moe_layer routed, and [1] vs
sigmoid * (moe_layer with shared - routed), for bf16 and fp32 gate
weights at T=1 and T=4 on both pairs. Cosine is >= 0.999998, relative max
is <= 2.7e-3, 8/8 OK.

Layer time, XPUGraph of 48 calls. "Before" is moe_layer + gemv shared MLP
+ F.linear gate + sigmoid + mul + add, the unfused shape.

| pair | T=1 before | T=1 after | T=4 before | T=4 after |
|---|---:|---:|---:|---:|
| IQ3_S/Q2_0 | 110.2 | 63.4 | 205.4 | 158.6 |
| IQ2_XS/IQ4_NL | 124.0 | 79.9 | 248.6 | 205.2 |

### (b) Gate/up and down: measured steps

ISA showed the gate/up loop is ALU-bound, not load-bound. The IQ3_S EC=1
k-loop was 528 instructions per iteration for 2 rows, with 16 dp4a. The
time adds up to about 385 instructions per 32-weight part. Steps, each
bit-identical to the previous output unless noted:

1. Sign masks: 4 sign bits become per-byte LSBs with one multiply
   (shifts 0/7/14/21 cannot carry), and mask = lsb * 0xFF. This replaces
   swar_ne4 (`B70_SIGN_MUL`). Loop: 528 -> 405.
2. Codebook grid staged in SLM per work-group, giving 32-bit gather
   addresses (`B70_GRID_SLM`). Loop: 405 -> 372.
3. Block-pointer walk: one pointer add per iteration instead of 64-bit
   (blk/P)*BB math (`B70_PTR_WALK`). Loop: 372 -> 361.
4. K-quants and IQ4_XS: branch-free funnel loads for 2-aligned runs
   (`b70_ldrun_a2`, Q6_K, Q3_K) and plain aligned loads where block and
   offset are multiples of 4 (Q2_K, Q4_K, Q5_K, IQ4_XS). EC=1 loop: Q6_K
   646 -> 391, Q4_K 504 -> 193, Q3_K 735 -> 509, IQ4_XS 900 -> 726.

Tried and rejected, all measured:

- SLM sign-mask table: fewer instructions (361 -> 320) but slower,
  66.3 vs 61.6 us. Off by default (`B70_SIGN_TAB`).
- Branch-free loads for the IQ types: 69.2 vs 61.6 us. Kept only for the
  K-quant runs.
- Down with 4 lanes per row: mixed results. Kept at 8.
- Stage 2's RPL 4, unroll 2, and 16 lanes, from before.

Load-only probe (`B70_PROBE_LOADONLY`, IQ3_S weight loads with almost no
ALU). Gate/up drops from 43.9 us to 30.4 us at T=1. That is about 510
GB/s, which is the floor of this per-lane access pattern. The full kernel
is now about 40 us. Of the remaining gap to 608 GB/s, about 10 us is ALU
and about 4 us is the access pattern (cooperative superblock loads). That
step was not taken; it needs subgroup shuffles that add ALU back.

Microbench graph us, T=1:

| type | stage 2 | stage 3 |
|---|---:|---:|
| IQ3_S/Q2_0 | 70.3 | 61.1 |
| IQ2_XS/IQ4_NL | 83.6 | 77.6 |
| Q2_K | 49.6 | 38.8 |
| Q3_K | 79.4 | 65.3 |
| Q4_K | 59.2 | 51.9 |
| Q5_K | 78.7 | 63.5 |
| Q6_K | 88.4 | 78.1 |
| IQ4_XS | 81.8 | 68.4 |

At T=4: IQ3_S/Q2_0 189.9 -> 161.3, IQ2_XS/IQ4_NL 226.0 -> 207.2.

### Batch-invariant reduce (coordinator add-on, X1 JOURNAL 18:12)

The routing sum used to walk the entries in grouped first-occurrence
order across all tokens. A token's float order therefore depended on the
other tokens in the batch.

The plan now records `emap[t*(topk+1)+j]`, the entry of (token, router
slot) with the shared expert at slot topk and -1 for an invalid id or no
shared expert. Both plan builders fill it: the parallel plan_q8 and the
serial `B70_SCHED_FUSE=0` plan. `b70_sched_red` FMAs in ascending router
slot order and adds the shared expert last. This applies to both out_mode
0 (fp32 combined) and out_mode 1 (bf16 split).

Every per-entry partial was already batch-independent: per-token Q8,
per-entry k order, EC passes that do not mix entries, and per-entry SiLU
and Q8(h).

Gate `tests/o3_batchinv.py`: token 0 alone vs the same token as row 0 of
T=2..8 batches whose other tokens reuse half of its experts in a different
order. It covers both pairs, with and without an invalid slot, for
moe_layer, moe_layer+shared, and both moe_layer_sh outputs. Result: 28/28
OK, 0 differing elements.

Cost is negative: the reduce at T=4 went from 6.55 to 3.90 us (unitrace).
The B70_SCHED_FUSE=0 build (serial plan) also runs, with cos 0.999999 on
the fair dumps.

### Gates and installed library

On `_C.o3g.so` (`o3/gates_g.log`), all failures=0, HEALTHY before and
after:

- gpu_check
- moe_layer_bench, sched 1 and 0
- c12_types (12 cases)
- graph_probe
- o3_shared
- o3_batchinv

compare_cpu is unaffected: the host dequant is untouched; it was last
14/14 at stage 2. The c9_fair graph (not quiet) is below; it includes the
weight-1 shared expert as before.

| T | Strata R2 IQ3_S | stage 2 | stage 3 | / Strata | Strata R2 IQ2_XS | stage 3 | / Strata |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 112.89 | 68.38 | 60.04 | 0.53 | 110.79 | 76.39 | 0.69 |
| 2 | 148.10 | 99.84 | 85.84 | 0.58 | 168.87 | 105.96 | 0.63 |
| 3 | 202.93 | 141.43 | 117.04 | 0.58 | 231.10 | 150.12 | 0.65 |
| 4 | 261.64 | 176.83 | 152.97 | 0.58 | 310.50 | 197.79 | 0.64 |

`_C.so` and `_C.graph.so` are now `_C.o3g.so`, sha256
`38f7176ac290cb264268185c864da2dea1fcc36d07535fff021fb636d6538dcc`.
Revert copies:

- `_C.o3.so` (be76661b, stage 1; X1 and H1e pin it)
- `_C.o3b.so` (e9e5282d, stage 2)
- `_C.c12.so` (b3c15e14)

### 35B UD-IQ3_XXS graph serve

Card 0, C12 recipe, prefill 1024 only. Identity `hotschmoe-dev` first, and
coherence OK in every run.

| binary | B70_FUSE_SHARED | C1 ITL p50 | C1 tok/s | C2 ITL p50 per stream |
|---|---:|---:|---:|---:|
| stage 1 o3 | - | 9.34 | 107.0 | 12.60 |
| stage 2 o3b | - | 9.30 | 107.5 | 12.54 |
| o3f (stage 3 without the K-quant loads) | 0 | 9.21 | 108.5 | 12.26 |
| o3f | 1 | 9.26 / 9.27 | 109.7 / 109.9 | 11.79 / 11.78 |
| o3g (installed) | 1 | 8.76 | 116.2 | 11.12 / 11.17 |

The serve log confirms the fused path on every decode size: `FUSED T=1..8
types=14,14,14 gate_inp=bf16`. One layer has no packed shared shards and
runs separately.

With o3f, fusing was neutral at C1 and gained 4 percent at C2. The fused
Q6_K shared expert then ran at our slower Q6_K decode, against the
runner's KREG Q6_K GEMV. The K-quant load fix (o3g) brings C1 from 9.27 to
8.76 ms. Overall since C12 this is 11.02 -> 8.76 ms C1 (-20.5 percent) and
16.61 -> 11.1 ms C2.

## Next levers (not done)

- Gate/up runs at about 350 GB/s and down at about 310 GB/s; the card peak is
  about 608. Unroll-2 and 16 lanes did not help. Try RPL=4, or software
  prefetch of the next superblock.
- Done in stage 2 (above): the K-quants and IQ4_XS on DP4A with aligned loads.
- Done in stage 3: the shared expert is fused into the decode op
  (opt-in B70_FUSE_SHARED=1, not under EP).
- IQ4_NL / IQ4_XS: the codebook mux is still the fattest decode (about
  780 instructions per EC=1 iteration). A table approach needs SLM gathers,
  which measured slow for the sign table.
- Cooperative superblock loads: about 4 us of gate/up per layer at T=1 by
  the load-only probe.
- Flash-Next shared under EP: fuse on the rank that owns the token's
  output, or run it in the EP sidecar.
- The reduce is 3.7 us vs Strata's 2.3 us. It could be fused into down for
  T=1.
