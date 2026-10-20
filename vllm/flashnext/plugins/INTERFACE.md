# Plugin op interface

The section above the A2 marker is the contract the vLLM fork calls
(branch `b70/flashnext`, `vllm/models/qwen4_exp/gguf/ops_interface.py`).
The block below the marker is A2's XPU implementation record. A2's ops
are registered for XPU only (`plugins/b70_gguf_xpu/csrc/bind.cpp:94`).
CPU dequant uses `libdequant_cpu.so`
(`b70_gguf_xpu/b70_gguf_xpu/libdequant_cpu.so`, symbol
`b70_dequant_f32_cpu`). There is no gather op and no `linear` / `moe` /
`dequant` name. The fork indexes packed rows itself and calls `dequant_bf16`.

## A1 contract (what the fork calls) -- 2026-10-20

Namespace: `torch.ops.b70_gguf`. `ggml_type` is the ggml integer id.

```
dequant_bf16(Tensor q, int ggml_type, int rows, int cols) -> Tensor
  q: uint8, numel >= rows * row_bytes(ggml_type, cols)
  returns bf16 [rows, cols]

gemv(Tensor x, Tensor q, int ggml_type, int N, int K) -> Tensor
  x: bf16 [M, K]
  q: uint8 [N, row_bytes]
  returns fp32 [M, N]
  A2 chunks M by 8, so one op covers decode and dense prefill

moe_gemv(Tensor x, Tensor q, Tensor expert_ids, int ggml_type, int N, int K) -> Tensor
  x: bf16 [T, K]
  q: uint8 expert pack, numel == E * N * row_bytes (E is inferred)
  expert_ids: int32 [T, topk], each value in [0, E)
  returns fp32 [T, topk, N]
  one GEMV per (token, slot) against the same x[t]
  gate and up can call it directly
  down cannot, unless the caller flattens tokens to [T*topk, I]
  and ids to [T*topk, 1]

grouped_gemm(Tensor x, Tensor q, Tensor entries, int ggml_type, int N, int K) -> Tensor
  x: fp16 or bf16 [M, K] (A2 converts bf16 to fp16 in the binding)
  q: uint8 expert pack, same layout as moe_gemv
  entries: int32 [n_ent, 3] = (expert, row0, nrows)
  returns fp32 [M, N], zero-initialized, then each entry overwrites its rows
  requires N % 64 == 0 and K % 32 == 0, otherwise the op raises
  gate and up are two calls, not one fused GEMM
  overlapping entries on the same output rows race
```

Python composition in `ops_interface.py` (this is the fork, not a new op):

- `linear` calls `gemv`. It casts `x` to bf16, flattens leading dims, and
  casts the fp32 result back to `x.dtype`.
- `moe` is SwiGLU in Python. Gate and up stay separate packed blobs because
  IQ3_XXS stores different ggml types on them. Expert id -1 is clamped to 0
  and that topk weight is multiplied by 0 (A2 rejects -1). `n_local_experts`
  is checked by the caller and is not passed to the op.
  Decode (`x.shape[0] <= 8`, or N/K fail the grouped_gemm modulo) uses
  `moe_gemv` for gate, up, then down. Down sees `[T*topk, I]` and ids
  `[T*topk, 1]`.
  Prefill (`x.shape[0] > 8` and both projections satisfy the modulo: gate
  needs intermediate % 64 == 0 and hidden % 32 == 0, down needs the swap,
  so both dims must be multiples of 64) uses `grouped_gemm`. The fork emits
  one entry per expanded row (`nrows = 1`). That is correct and slow.
- `dequant` accepts a 2D logical shape `(rows, cols)`. On CPU it tries
  `dequant_bf16` and, if the kernel is missing, calls `dequant_cpu_so`.
  On XPU it calls `dequant_bf16`. A 3D tensor raises.
- `gather_rows` indexes `table` on `table.device`, moves only that slice to
  `row_ids.device`, then dequants. Embedding lookup and the PLE host table
  use it. The 26.8 GiB IQ4_NL table stays on CPU.

Fused `ffn_gate_up_exps` (qwen3_5 schema, one ggml type) is split on the N
axis before the calls. The first half is gate and the second half is up
(llama.cpp `llama-graph.cpp` views). The halves are made contiguous because
A2's pack is `E * N * row_bytes`. The Qwen3.6 UD-IQ3_XXS file has no fused
`ffn_gate_up_exps` tensor (header check, 2026-10-20). Gate, up, and down
are three tensors.

Types the fork can size (`gguf/types.py`) include A2's set plus F32, F16,
BF16, Q4_0, Q5_0, Q8_0, and the two ids A2 does not implement:

| name | id | block elems | block bytes | A2 kernel |
| --- | --- | --- | --- | --- |
| Q2_K | 10 | 256 | 84 | no |
| Q3_K | 11 | 256 | 110 | no |

Those two are the Qwen3.6 MTP expert tensors only:
`blk.40.ffn_gate_exps.weight` and `blk.40.ffn_up_exps.weight` (Q2_K),
`blk.40.ffn_down_exps.weight` (Q3_K). A load can size the rows. A matmul
of them is an A2 work item (medium). Sampled base blocks blk.0 and
blk.39 use IQ2_S, IQ3_S, and IQ4_XS on the expert weights. The only
Q2_K and Q3_K tensors in the file are those three MTP expert weights.

Measured expert shapes that satisfy grouped_gemm (N and K multiples of 64):
Qwen3.6 gate/up raw ne `(2048, 512, 256)`, down `(512, 2048, 256)`.
Flash-Next down logical width 640 is also a multiple of 64
(`packed_row_bytes(640, 20)` in the CPU test). If a future tensor fails
the modulo, `moe` stays on `moe_gemv`.

## A2 proposal (b70_gguf_xpu) -- 2026-10-20

Namespace: `torch.ops.b70_gguf`. Load with `torch.ops.load_library` on
`b70_gguf_xpu/_C.so`, or `import b70_gguf_xpu; b70_gguf_xpu.register()`
(entry point group `vllm.general_plugins`). Every tensor argument is a
contiguous XPU tensor. `ggml_type` is the ggml integer id.

Quant layout: one row is K logical values, `row_bytes = (K / block_elems) * block_bytes`.
An expert pack is E such matrices concatenated, expert e at byte offset
`e * N * row_bytes`. ggml ids:

| name | id | block elems | block bytes |
| --- | --- | --- | --- |
| Q4_K | 12 | 256 | 144 |
| Q5_K | 13 | 256 | 176 |
| Q6_K | 14 | 256 | 210 |
| IQ2_XXS | 16 | 256 | 66 |
| IQ2_XS | 17 | 256 | 74 |
| IQ3_XXS | 18 | 256 | 98 |
| IQ4_NL | 20 | 32 | 18 |
| IQ3_S | 21 | 256 | 110 |
| IQ2_S | 22 | 256 | 82 |
| IQ4_XS | 23 | 256 | 136 |
| Q2_0 | 42 | 64 | 18 |

```
dequant_bf16(Tensor q, int ggml_type, int rows, int cols) -> Tensor
  q: uint8, numel >= rows * row_bytes(ggml_type, cols)
  returns bf16 [rows, cols]
  bf16 rounding is round-to-nearest-even (ggml / strata f2bf)

gemv(Tensor x, Tensor q, int ggml_type, int N, int K) -> Tensor
  x: bf16 [M, K], M >= 1 (host loops M in chunks of 8)
  q: uint8, numel >= N * row_bytes(ggml_type, K)
  returns fp32 [M, N]
  y[m, n] = dot(x[m], dequant(row n))

moe_gemv(Tensor x, Tensor q, Tensor expert_ids, int ggml_type, int N, int K) -> Tensor
  x: bf16 [T, K]
  q: uint8 expert pack, numel == E * N * row_bytes (E is inferred)
  expert_ids: int32 [T, topk], value in [0, E)
  returns fp32 [T, topk, N]
  one GEMV per (token, slot) against expert expert_ids[t, slot]

grouped_gemm(Tensor x, Tensor q, Tensor entries, int ggml_type, int N, int K) -> Tensor
  x: fp16 or bf16 [M, K] (bf16 is converted to fp16 in the binding)
  q: uint8 expert pack, same layout as moe_gemv
  entries: int32 [n_ent, 3] = (expert, row0, nrows)
    the entry consumes x[row0:row0+nrows] and writes y[row0:row0+nrows, :]
  returns fp32 [M, N], zero-initialized, then each entry's rows are overwritten
  requires N % 64 == 0 and K % 32 == 0, otherwise the op raises
  gate and up are two calls; they are not a fused interleaved GEMM
  overlapping entries on the same output rows are a race
```

Python helpers with the same names live in `b70_gguf_xpu.ops`.

Implementation note (A2, 2026-10-20, signatures unchanged):

- `gemv` for Q4_K, Q5_K, Q6_K, and IQ4_XS with K divisible by 256 quantizes x to Q8_1 once per call, then runs the KREG schedule. Other types, and MoE, stay on the scalar GEMV. K=640 expert-down is not a KREG shape.
- `grouped_gemm` is joint_matrix 8x16x16, MT=16, so one entry of 128 contiguous rows reads its weight once. An entry with `nrows = 1` reads that expert's weight for a single row. Packing one expert's rows into one entry is what the bench measures.
