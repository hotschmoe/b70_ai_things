# 1A2 -- GGUF / IQ XPU kernels

Plugin: `vllm/flashnext/plugins/b70_gguf_xpu`. Ops: `torch.ops.b70_gguf.*`.
The contract A1's fork calls is the section above the A2 marker in
`vllm/flashnext/plugins/INTERFACE.md`. Signatures match that contract.
The A2 section records the implementation.

## CONFIG

- Card 0 only. `bin/gpu-run --card 0` with `ZE_AFFINITY_MASK=0`. One visible
  device: Intel Arc Pro B70. Card 1 was left to other agents.
- Torch 2.14+xpu from `/mnt/vm_8tb/b70/build/vllm-flashnext/venv`.
  Host `venv/bin/python` points at a missing `/usr/bin/python3.12`, so the
  tests use uv CPython 3.12.13 plus `PYTHONPATH` to that venv.
- Compiler: host icpx 2026.1.1
  (`.../oneapi/compiler/2026.1/bin/icpx`). strata-sycl-dev:020 is icpx
  2026.1.0 and its entrypoint is `bash -lc`, so the extension was built on
  the host against the venv `libsycl.so.9`.
- The host has `libstdc++.so.6` (GCC 15) and no g++ headers (no sudo).
  Headers, crt, and the libstdc++ link input were copied from
  strata-sycl-dev:020 to `/mnt/vm_8tb/b70/build/gguf-xpu-toolchain`.
  `scripts/build.sh` passes `--gcc-toolchain` at that tree.
- Build: `B70_AGENT=A2 MAX_JOBS=4 bin/cpu-run scripts/build.sh`. One
  translation unit, about 38 s inside the lock. Device code is SPIR-V in
  `_C.so` and JIT-compiled by IGC on first launch
  (`SYCL_CACHE_PERSISTENT=1`, cache
  `/mnt/vm_8tb/b70/build/gguf-xpu-sycl-cache`).
- Weights: Flash-Next shard 1
  `/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`.
  Also the A3 devmodel GGUFs under `/mnt/cache/b70/devmodel/unsloth/`.
- Roofline used for GB/s: 608 GB/s. Numerator is weight bytes touched.
  Grouped issued bytes equal unique bytes: MT=16 covers 128 rows in one
  weight read.
- Strata reference (reports/03): KREG `native_mmvq.dp.cpp:2236`, head nc1
  760 us (575 GB/s, 95% of 608), nc6 1166 us. Grouped prefill is
  `night_grouped_expert.inc` joint_matrix 8x16x16, MT up to 16, GRF 256
  when MT > 4.

Ops, all XPU, contiguous:

- `dequant_bf16(q uint8, ggml_type, rows, cols) -> bf16 [rows, cols]`
- `gemv(x bf16 [M, K], q, ggml_type, N, K) -> fp32 [M, N]`, M chunked by 8.
  Q4_K, Q5_K, Q6_K, and IQ4_XS with K divisible by 256 quantize x to Q8_1
  once, then the KREG schedule. Other types stay on the scalar GEMV.
- `moe_gemv(...)` stays on the scalar GEMV. Flash-Next experts are IQ and
  Q2_0, and expert-down K=640 is not a multiple of 256.
- `grouped_gemm(...)` is joint_matrix 8x16x16, MT=16, GRF 256. Requires
  `N % 64 == 0` and `K % 32 == 0`. bf16 activations are cast to fp16 in
  the binding. One entry of 128 contiguous rows reads the weight once.
  `nrows = 1` reads it for a single row.

Types: Q4_K 12, Q5_K 13, Q6_K 14, IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18,
IQ4_NL 20, IQ3_S 21, IQ2_S 22, IQ4_XS 23, Q2_0 42.

## COMMAND

```
B70_GGUF_HOST_ONLY=1 scripts/build.sh
python3 tests/compare_cpu.py

B70_AGENT=A2 MAX_JOBS=4 bin/cpu-run scripts/build.sh
SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/mnt/vm_8tb/b70/build/gguf-xpu-sycl-cache \
  B70_AGENT=A2 B70_GPU_LOCK_TIMEOUT=0 bin/gpu-run --card 0 scripts/gpu_test.sh
```

`compare_cpu.py` dequants 2 rows per type with `libdequant_cpu.so` and
compares uint32 bits to llama-pinned gguf-py. Q2_0 has no gguf-py
dequant class; that row is compared to a separate Python transcription of
`dequantize_row_q2_0`.

`gpu_check.py` bit-compares GPU bf16 dequant to round-to-nearest-even of
the fp32 reference (64 rows). GEMV at M=1,3,8 is compared to fp32 matmul
of the bf16 activation. K-quant types (Q4_K, Q5_K, Q6_K, IQ4_XS) use
atol 5e-2 and rtol 5e-2, because the kernel dots a Q8_1 quantization of
x (amax/127 per 32 values). Other types use atol 2e-2 and rtol 2e-3.
MoE uses the tight tolerance. Grouped GEMM at M=16 and M=32 uses atol
5e-2 and rtol 5e-2. Bench is the min of 10 timed launches after 3
warmups, `torch.xpu.Event` milliseconds converted to us. The whole
`gemv()` call is inside the event, including the Q8 quant kernel.

## RESULT

CPU dequant, 0 mismatched bits:

- Flash-Next shard 1, all 11 types, 2 rows each (Q2_0 against the local
  transcription, every other type against gguf-py).
- Devmodel `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`: Q6_K, IQ3_S, IQ2_S, IQ4_XS
  (the types that file actually stores; a scan did not find a ggml
  IQ3_XXS tensor). Flash-Next `blk.35.ffn_gate_exps` is IQ3_XXS and is
  bit-exact. Also bit-exact on the UD-IQ4_XS and UD-Q4_K_M files for the
  types those files store (Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS).

GPU, Flash-Next, 64 rows, final binary, `VERDICT failures=0`:

- Dequant bf16: 0 mismatches on all 11 types.
- Scalar GEMV and MoE (IQ and Q2_0): max abs about 6e-8 to 1e-6.
- KREG GEMV (Q8_1 of the real bf16 x): max abs 8.6e-3 (Q4_K M=1) to
  3.3e-2 (IQ4_XS M=8). All under the 5e-2 gate. A swapped-lane layout
  would miss by a large fraction of y.
- Grouped GEMM (fp16 XMX vs fp32 reference) max abs about 1.5e-4 to
  1.1e-3. Relative error can exceed 1 on outputs near zero; every tile
  passed atol 5e-2 + rtol 5e-2.

Bench of the shipped binary. Weight GB/s = bytes / seconds. 608 GB/s = 100%.
Three earlier binaries of the same shapes, for the head only: fixed 32 KB
SLM 3403 us; scalar GEMV with SLM sized to the row 1739.4 us / 251.3 GB/s;
KREG that re-quantized x inside every work-group 857.6 us / 509.6 GB/s.

| op | type | shape | us | GB/s | % of 608 |
| --- | --- | --- | --- | --- | --- |
| gemv | Q5_K | M=1 N=248320 K=2560 (output.weight, 437043200 B) | 808.4 | 540.6 | 88.9 |
| gemv | Q5_K | M=8 same | 4303.6 | 101.6 | 16.7 |
| gemv | IQ3_S | M=1 N=248320 K=2560 (token_embd, 273152000 B) | 1593.5 | 171.4 | 28.2 |
| gemv | IQ3_S | M=8 same | 6338.2 | 43.1 | 7.1 |
| gemv | IQ4_XS | M=1 N=12288 K=2560 (16711680 B) | 128.1 | 130.5 | 21.5 |
| gemv | IQ4_XS | M=8 same | 267.2 | 62.5 | 10.3 |
| gemv | Q6_K | M=1 N=12288 K=2560 (25804800 B) | 126.6 | 203.9 | 33.5 |
| gemv | Q6_K | M=8 same | 281.5 | 91.7 | 15.1 |
| gemv | Q4_K | M=1 N=12288 K=2560 (17694720 B) | 128.0 | 138.2 | 22.7 |
| gemv | Q4_K | M=8 same | 224.3 | 78.9 | 13.0 |
| moe_gemv | IQ4_NL | T=1 topk=10 N=2560 K=640 (9216000 B) | 122.9 | 75.0 | 12.3 |
| moe_gemv | IQ3_S | T=1 topk=10 N=640 K=2560 (7040000 B) | 122.8 | 57.3 | 9.4 |
| moe_gemv | IQ3_XXS | T=1 topk=10 N=640 K=2560 (6272000 B) | 122.6 | 51.2 | 8.4 |
| moe_gemv | IQ2_S | T=1 topk=10 N=640 K=2560 (5248000 B) | 121.8 | 43.1 | 7.1 |
| moe_gemv | IQ2_XS | T=1 topk=10 N=640 K=2560 (4736000 B) | 122.7 | 38.6 | 6.4 |
| moe_gemv | Q2_0 | T=1 topk=10 N=2560 K=640 (4608000 B) | 121.9 | 37.8 | 6.2 |
| moe_gemv | IQ2_XXS | T=1 topk=10 N=640 K=2560 (4224000 B) | 122.3 | 34.5 | 5.7 |
| grouped | IQ4_NL | M=128 N=2560 K=640 unique=921600 | 136.0 | 6.8 | 1.1 |
| grouped | Q2_0 | M=128 N=2560 K=640 unique=460800 | 135.8 | 3.4 | 0.6 |
| grouped | IQ3_S | M=128 N=640 K=2560 unique=704000 | 308.7 | 2.3 | 0.4 |
| grouped | IQ2_S | M=128 N=640 K=2560 unique=524800 | 306.0 | 1.7 | 0.3 |
| grouped | IQ3_XXS | M=128 N=640 K=2560 unique=627200 | 352.3 | 1.8 | 0.3 |
| grouped | IQ2_XS | M=128 N=640 K=2560 unique=473600 | 364.1 | 1.3 | 0.2 |
| grouped | IQ2_XXS | M=128 N=640 K=2560 unique=422400 | 348.9 | 1.2 | 0.2 |

Strata's head Q5_K nc1 number is 760 us, 575 GB/s, 95% of 608. This
binary is 808.4 us, 540.6 GB/s, 88.9% of 608, 1.06x that figure. The
in-kernel quant binary was 857.6 us. Quantizing x once, in a kernel
queued ahead of KREG on the same stream, removed the 62080-way repeat
of the K=2560 activation quant and closed 49 us. The remaining 48 us
was not closed by another occupancy change. Strata nc6 on the head is
1166 us; our M=8 on the same weight is 4304 us (the in-kernel-quant
KREG was 4699 us, the scalar GEMV was 6609 us). Short N=12288 rows sit
near a 110-130 us floor; Q4_K M=1 there is 128 us, slower than the
scalar binary's 108 us.

Grouped, same shapes, MT=4 (weight read 4 times at M=128) was 192 us
(IQ4_NL down), 151 us (Q2_0 down), and 510-748 us on the K=2560 gates.
MT=16 reads the weight once. Downs moved to 136 us. Gates moved to
306-364 us, about 1.2-2.3 GB/s of the packed weight. The kernel is
correct and still decode-bound: one expert gate is hundreds of
microseconds for a few hundred KB.

Side probe, not the shipped op. Dequant the expert to bf16 and
`torch.mm` it, with x already bf16: gate 193-195 us, down 188-189 us.
The matmul alone, on the 3.2 MB bf16 tile, is 151 us (21 GB/s), a
launch floor. An in-op version that host-reads `entries` and casts
fp16 x measured about 307 us on the gate, widened K-quant grouped
error from ~1e-3 to ~1e-2, and would sync once per call. The fork
emits `nrows = 1` (INTERFACE.md), so that host loop would run once
per token. It was not shipped. Joint_matrix stays the grouped op.

Card 0 started with a 128-class matmul probe (`PROBE matmul ok`,
device_count=1) and the suite ended `gpu-run` exit 0. No DEVICE_LOST
and no xe-reset.

## VERDICT

Installable package with dequant, GEMV (M=1..8), expert-indexed MoE
GEMV, and grouped joint_matrix GEMM for all 11 Flash-Next GGUF types.
Dequant matches gguf-py bit for bit on real rows (Q2_0 matches a ggml
transcription, because gguf-py does not implement it). IQ and Q2_0
GEMV and MoE match an fp32 reference to about 1e-6 absolute. K-quant
GEMV matches a Q8_1 dot of the bf16 activation to a few 1e-2 absolute
against the true bf16 matmul.

Decode GEMV on the output head is 808 us, 541 GB/s, 89% of 608 GB/s,
1.06x Strata KREG's 760 us. Expert MoE GEMV for one token and top-10
is about 122 us per projection. Grouped prefill of one packed expert
at M=128 is 136 us on the downs and 306-364 us on the gates.

## Next step

1. Head GEMV, the last 48 us (808 vs 760) and the M=8 gap (4304 vs
   Strata nc6 1166). The quant is already once per call. The remaining
   lever is Strata's pipelined persistent round-2 (`night_pf` in
   `native_mmvq.dp.cpp`, after the round-1 KREG at line 2236), plus a
   check that IGC emits DP4A for `b70_dp4a` rather than the byte-unpack
   mad. Another occupancy or quant tweak is not the gap.
2. Grouped prefill. MT=16 already reads a 128-row entry once and the
   tile is still a few GB/s. Next lever inside the kernel is the
   next-superblock preload in `night_grouped_expert.inc` (load the
   next block's raw fields before decoding this one). The dense
   fallback is launch-bound at ~150 us for a 3.2 MB tile, so it only
   wins if the IQ decode is free. Serving does not see the 128-row
   number until the caller packs one expert's rows into one entry;
   `nrows = 1` repeats the weight read per row.
3. Q2_K (ggml 10) and Q3_K (ggml 11) are not in the Flash-Next
   histogram. A1's note: three Qwen3.6 MTP expert tensors use them.
   Not implemented.
