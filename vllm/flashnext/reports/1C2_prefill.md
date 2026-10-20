# C2 prefill GEMM

Card 0, Arc Pro B70, icpx 2026.1. Measured library
`plugins/b70_gguf_xpu/b70_gguf_xpu/_C.full.so` (same bytes as `_C.bench.so`,
sha256 edefe35868158518bbe5cbfd11e7097f6bdc8c73b893e246a910af8b3282f533,
6386936 bytes). `scripts/prefill_gpu.sh` points `B70_GGUF_SO` at that file.
Package `_C.so` is C1's 6391032-byte decode binary from 2026-10-20 05:33
(sha256 1e484faeeade967701bf56b8abeb24acd230b28eec79212b28a101ffe4bc3cac).
It was not left as the prefill build.

## What was built

New files, not edits to `kernels.cpp`, `ops.py`, or `__init__.py`:

- `csrc/grouped_raw.hpp` -- `b70_load_raw` / `b70_decode_raw`, next-block preload.
- `csrc/grouped_prefill.cpp` -- device sort, gather, one GEMM launch per projection, swiglu, weighted reduce.
- `csrc/grouped_api.hpp` -- the four op declarations.
- `csrc/raw_selftest.cpp` -- host only, not linked into the `.so`.
- `b70_gguf_xpu/prefill.py`, `tests/prefill_check.py`, `scripts/prefill_gpu.sh`.

Shared diffs, kept small:

- `csrc/bind.cpp` -- a `TORCH_LIBRARY_FRAGMENT` plus a second XPU impl and a
  `CompositeExplicitAutograd` impl for `prefill_split_us`, after C1's block.
- `scripts/build.sh` -- compiles `grouped_prefill.cpp`. `B70_PREFILL_SMOKE=1`
  writes `_C.smoke.so`. `B70_PREFILL_OUT` writes a side `.so`.

Ops: `prefill_grouped`, `prefill_dense`, `prefill_moe_layer`, `prefill_split_us`.
Mode 0 is fp16 DPAS 8x16x16. Mode 1 is bf16 DPAS 8x16x16. Mode 2 is int8
DPAS 8x16x32 with per-32 scales. Mode 0 is instantiated for all 11 Flash-Next
types at MT=4 and MT=16. Modes 1 and 2 are MT=16 only, for Q4_K, Q5_K, Q6_K,
IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0. IQ2_XXS / IQ2_XS / IQ2_S are fp16 only.
Q2_K and Q3_K are not in this prefill set.

## Strata schedule, and what this port actually launches

Strata source is `/mnt/vm_8tb/b70/build/strata-ab/src-ep`.

- `sycl/include/strata/kernels/night_grouped.hpp:1` describes one launch
  covering every entry of a layer on a card. `night_grouped_f16` is at
  line 24. An entry is `rows` of 1..8*mt. The header lists types
  16, 17, 18, 20, 21, 22, 42.
- The caller is `sycl/src/prefill/prefill.cpp:4121`. It cuts each expert
  into row tiles, buckets those tiles by mt = 2, 4, 8, 16 (`2 << t` for
  `t` in 0..3, caps 16/32/64/128 rows), and launches once per populated
  bucket. Gate/up is one launch (`gu=true`, N = 2*640, even column = gate,
  odd = up). Then `swiglu_interleaved`. Then down. A layer is several
  launches, not one kernel for gate+up+down.
- The kernel is `sycl/src/kernels/cuda/night_grouped_expert.inc:235`,
  included from `iq_kernels.dp.cpp:6686`. Default SGN is 4 (64 columns),
  line 341; the comment says SGN 2 and 8 were within 4% of 4. LDB is 40
  (line 242). The next-block preload is the comment at line 275. DPAS is
  fp16 8x16x16 into fp32 (lines 297-305). GRF is 256 when MT>4 (line 247).
  A is loaded from global, not staged in SLM. `xmx_gemm_iq` at
  `iq_kernels.dp.cpp:6653` is the dense IQ GEMM, separate from grouped.

This port launches once per projection for every expert, with one host-chosen
MT (average rows <= 24 selects MT=4, otherwise MT=16). An expert with more
rows than 8*MT re-reads its weights inside the same work-group. Gate and up
share one launch when their types match. Down is a second launch. That is
two launches per layer, not Strata's per-bucket launches, and not one
launch for the whole layer. Activations are sorted on device
(histogram, scan, scatter) and gathered into expert-major order. The
work-group owns many output columns and stages the activation tile in SLM
so those columns do not each reload X. Weights are decoded from the GGUF
block into SLM with the next 32-wide block prefetched, then consumed by
`joint_matrix`.

Device query, printed once: `local_mem=131072 max_wg=1024 max_sg=64`.
`grf_size<256>` (MT>4) rejects a work-group above 512 even though the
device reports 1024. The first bench hit `Total number of work-items in a
work-group cannot exceed 512` on every MT=16 shape that picked 640 or
1024 columns. The picker now caps MT>4 at 512 and takes the largest
multiple of 16 that divides N and fits SLM. MT=4 is unchanged and did
launch at 640. `SPLIT` `cols` is the gate/up tile (`grouped_prefill.cpp`
sets `g_split.cols` from the GU layout only). For N=2560 and MT=16 the
same picker returns 512 for down.

`unsigned short` is `matrix_type::uint16`. The first bf16 attempt threw
`joint_matrix with parameters matrix_type::uint16, use::a, Rows=8, Cols=16
is not supported on this device`. Mode 1 stores `bfloat16` bits
(`sycl::ext::oneapi::bfloat16`, `matrix_type::bf16`) and that combination
runs.

## CONFIG

Correctness: real IQ3_XXS shard
`/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`,
first 128 rows of each of the 11 types, fp32 reference from gguf-py
dequant (Q2_0 via `libdequant_cpu.so`). fp16 and bf16 gate is atol=rtol=5e-2.
int8 fails only on NaN or far_frac >= 0.30. Grouped case: three copies of
the slice, ids with a 40-row expert, a 7-row expert, and one invalid id.
Layer: shard-0 `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`, T=3,
topk=2, one id forced to -1. Reference rounds the hidden vector through
fp16 before the down GEMM.

Bench: synthetic blocks with fp16 scale 1.0. Event min of 8 (4 at
T=12288) after warmup. `B70_PREFILL_SPLIT=1` is a separate call, not inside
the event min. GB/s uses unique weight bytes, the same numerator A2 used.
It is not multiplied by the column-group or row-tile re-read. Layer shape
E=512, topk=10, H=2560, F=640, gate/up IQ3_XXS, down Q2_0. Dense attn
shapes N=12288 K=2560. Head Q5_K N=248320 K=2560 M=512.
`expert_only_48layer_tok/s = T / (us * 48 * 1e-6)`. Attention, norms, GDN,
and HC are not in that number.

Host raw selftest (all 11 types, K in {block, 640 where legal, 2560}, plus
row 1 flat index) printed `VERDICT fail_types=0` before the binary was
removed.

## COMMAND

```
B70_AGENT=C2 B70_GPU_LOCK_TIMEOUT=0 bin/gpu-run --card 0 \
  vllm/flashnext/plugins/b70_gguf_xpu/scripts/prefill_gpu.sh
```

Log: `/mnt/vm_8tb/b70/build/c2-prefill-gpu-correct.log`. gpu-run done in 74s,
exit 0.

```
B70_AGENT=C2 B70_GPU_LOCK_TIMEOUT=0 B70_CPU_LOCK_TIMEOUT=0 \
  PYTHONUNBUFFERED=1 B70_PREFILL_TIMEOUT=7200 B70_GGUF_SO=.../_C.bench.so \
  bin/gpu-run --card 0 bin/cpu-run --quiet \
  vllm/flashnext/plugins/b70_gguf_xpu/scripts/prefill_gpu.sh --bench-only
```

Log: `/mnt/vm_8tb/b70/build/c2-prefill-gpu-bench2.log`. cpu-run quiet 39s,
gpu-run 75s, exit 0. The earlier log
`/mnt/vm_8tb/b70/build/c2-prefill-gpu-bench.log` is the uncapped binary:
MT=16 launches with cols>512 failed, and the T=512 layer (MT=4, cols=640)
still ran.

## RESULT

Health before and after both runs: `xpu-health: HEALTHY (cards 0)`,
`HEALTH_BEFORE:0`, `HEALTH_AFTER:0`. In-process 64x64 matmul probe passed.

Correctness, `VERDICT failures=0` (61 OK lines). fp16 mode 0, 44 checks,
max abs 2.682e-07 to 1.487e-03. bf16 mode 1, 8 checks, max abs 1.378e-03
to 1.093e-02. int8 mode 2, 8 checks, far_frac 0.009 to 0.013, max abs
8.304e-03 to 3.772e-02. Layer
`OK layer IQ2_XS->IQ4_NL T=3 E=2 H=2560 F=640 max_abs=1.214e-04`.
High max_rel on tiny reference values is the 5e-2 relative gate, not a
lane swap. N=128 on these checks, so the column tile was 128.

Bench, `VERDICT failures=0`. Times are the event minimum.

| op | shape | us | unique GB/s | % of 608 | cols | mt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| grouped | IQ4_NL M=128 N=2560 K=640 | 344.4 | 2.68 | 0.44 | 512 | 16 |
| grouped | IQ3_XXS M=128 N=640 K=2560 | 646.8 | 0.97 | 0.16 | 320 | 16 |
| dense | Q4_K M=512 N=12288 K=2560 | 5295.8 | 3.34 | 0.55 | 512 | 16 |
| dense | Q4_K M=4096 N=12288 K=2560 | 42015.7 | 0.42 | 0.07 | 512 | 16 |
| dense | Q4_K M=12288 N=12288 K=2560 | 126036.2 | 0.14 | 0.02 | 512 | 16 |
| dense | Q5_K M=512 N=12288 K=2560 | 7532.6 | 2.87 | 0.47 | 512 | 16 |
| dense | Q5_K M=4096 N=12288 K=2560 | 59898.6 | 0.36 | 0.06 | 512 | 16 |
| dense | Q6_K M=512 N=12288 K=2560 | 8352.4 | 3.09 | 0.51 | 512 | 16 |
| dense | IQ4_XS M=512 N=12288 K=2560 | 1768.4 | 9.45 | 1.55 | 512 | 16 |
| dense | IQ3_S M=512 N=12288 K=2560 | 1447.0 | 9.34 | 1.54 | 512 | 16 |
| head | Q5_K M=512 N=248320 K=2560 | 124874.0 | 3.50 | 0.58 | 512 | 16 |

A2's grouped IQ4_NL M=128 N=2560 K=640 was 136 us (6.8 GB/s, 1.1%).
A2's grouped IQ3_XXS M=128 N=640 K=2560 was 306 us (1.7 GB/s, 0.3%).
This kernel is slower on those two single-expert shapes.

Layer, IQ3_XXS gate/up + Q2_0 down, E=512 topk=10 H=2560 F=640. `cols` in
the split is the GU tile.

| T | us | unique GB/s | % of 608 | one-layer tok/s | expert-only 48-layer tok/s | GU cols | mt |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 13884.8 | 63.25 | 10.40 | 36875 | 768 | 640 | 4 |
| 4096 | 27598.9 | 31.82 | 5.23 | 148412 | 3092 | 320 | 16 |
| 12288 | 66129.1 | 13.28 | 2.18 | 185818 | 3871 | 320 | 16 |

T=512 split us: sort 84.7, gather 36.0, GU 10268.4, swiglu 67.9, down
3440.1, reduce 186.7, total 14117.9.
T=4096: sort 161.3, gather 617.5, GU 19562.5, swiglu 563.4, down 5085.0,
reduce 1472.1, total 27758.1.
T=12288: sort 331.3, gather 2320.0, GU 45207.1, swiglu 1654.6, down
11710.2, reduce 4449.9, total 66566.8.

bf16 layer T=512: 24708.5 us (35.54 unique GB/s). int8 layer T=512:
29103.5 us (30.17 unique GB/s). Both are slower than fp16 on this shape.
fp16 is the path to keep.

Unique bytes for one layer are 878 MB (gate+up 642 MB, down 236 MB).
Actual traffic is higher. At T=512, GU cols=640 so N=1280 is read twice:
642 MB * 2 / 10.27 ms = 125 GB/s. Down cols=640, N=2560 is read four
times: 236 MB * 4 / 3.44 ms = 274 GB/s. At T=4096 the average expert has
80 rows, one MT=16 tile, GU cols=320 (four column groups): about 131 GB/s
on the GU bytes. At T=12288 the average expert has 240 rows, so the
in-group row loop reads weights twice on top of the four column groups:
about 114 GB/s on the GU bytes. IQ3_XXS gate/up decode sits near 120 GB/s,
about 20% of 608. Q2_0 down is higher, about 200-270 GB/s.

Strata's published prefill numbers are end-to-end tok/s, not an isolated
expert kernel. Grouped XMX alone moved 4K from 1342 to 1507 and 32K from
1708 to 1738. Grouped plus XMX attention reached about 2002-2047, and the
later stack about 2200 (`reports/03_strata_inventory.md`). At 2200 tok/s a
12288-token chunk is 5.59 s, about 116 ms per layer for attention, GDN,
HC, and the expert GEMM together. This expert layer is 66.1 ms at
T=12288. Expert-only 3871 tok/s is not a 2000 tok/s prefill result.

Sort and the reduction are small beside the GEMM. Gather grows with T
(2.3 ms at T=12288) and is still under the GU time.

## VERDICT

The prefill path is correct on the 11 GGUF types, on a shuffled
multi-expert group with an invalid id, and on a fused layer. fp16, bf16,
and int8 DPAS all run. fp16 is the fastest of the three on the T=512 layer.

It does not hit 2000 tok/s prefill. Expert time alone is 27.6 ms at
T=4096 and 66.1 ms at T=12288, against about 116 ms/layer for Strata's
whole 2200 tok/s layer at T=12288. The GU IQ3_XXS decode is the critical
path, near 120 GB/s of actual weight traffic.

Next step, not started: two schedule trials on the same bench.

1. Dense and the head. One work-group walks every row tile, so M=12288
   is 96 serial passes over the weights (24 work-groups). Split row tiles
   into parallel entries the way `prefill.cpp:4121` does, and remeasure
   Q4_K M=512 and M=12288.
2. Expert GU. Strata's SGN=4 (64 columns, X from global) beat this wide
   SLM tile on the A2 single-expert shapes (136 us vs 344 us). Rerun the
   T=4096 and T=12288 layer with that schedule and compare GU microseconds.
   Do not add attention to the tok/s claim until it is in the same timer.
