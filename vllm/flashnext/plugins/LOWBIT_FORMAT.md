# Low-bit WNA16 layout for B2 (auto_round:auto_gptq)

Produced by B1. Packing was checked against Intel auto-round
`auto_round/utils/bit_packing.py` (`pack_bitstream`) and against the on-disk
Ornith GPTQ int4 expert `model.language_model.layers.0.mlp.experts.0.down_proj`.
Repacking that matrix with this code is byte-identical for both `qweight` and
`qzeros`. Signed codes recovered with zero-point 8 were `[-8, 7]`.

This is the classic AutoGPTQ / auto_round little-endian bitstream. It is not
AWQ order and it is not the MXINT pair packing in `qlinear_int.py`.

## Grid

Symmetric, group along K (the last dim of an HF Linear weight `[N, K]`).

```
maxq   = 2 ** (bits - 1)
signed = clamp(round(w / scale), -maxq, maxq - 1)
scale  = per-group absmax / maxq          # during SignRound, absmax is min/max-tuned
```

| bits | signed range | maxq (subtract this) | unsigned stored | qzeros lane |
|------|--------------|----------------------|-----------------|-------------|
| 4    | [-8, 7]      | 8                    | signed + 8      | 7           |
| 3    | [-4, 3]      | 4                    | signed + 4      | 3           |
| 2    | [-2, 1]      | 2                    | signed + 2      | 1           |

`group_size` is 128. `desc_act` is false. `g_idx` is int32 `[K]` with
`g_idx[i] = i // 128`. Groups are contiguous, so a kernel may ignore `g_idx`
and use `k // 128`.

Stored scale is fp16, shape `[K / 128, N]` (group, out). Tuning runs in fp32
and the export casts that scale to fp16. Dequant in fp32 is:

```
unsigned = unpack(qweight)          # [K, N], values in [0, 2*maxq)
signed   = unsigned - maxq          # NOT unsigned - qzeros
w        = signed * scale[k // 128, n]
```

`qzeros` is the AutoGPTQ "zeros minus 1" convention. Every lane holds
`maxq - 1` (7, 3, or 1). A CUDA GPTQ kernel that adds 1 recovers `maxq`.
The XPU oneDNN symmetric int4 path ignores `qzeros` and subtracts the scalar
8 (`vllm-xpu-kernels` `transpose_onednn_woq_format`, `is_sym`). Do the same
for int3 (subtract 4) and int2 (subtract 2). Subtracting the raw qzeros lane
is off by one: on the reference expert, `(unsigned - 7) * scale` has rel L2
0.530 against BF16, while `(unsigned - 8) * scale` has rel L2 0.104.

## Bit stream

`qweight` is int32, shape `[K * bits / 32, N]`. The pack axis is K.
`K` and `N` are multiples of 32 (and of 128 on this model).

Little-endian over blocks of 32 values. Value `i` in the block occupies bits
`[i * bits, (i + 1) * bits)`. Word `(i * bits) // 32`, offset `(i * bits) % 32`.
If `offset + bits > 32` the value straddles two int32 words. The int32 is a
signed view of the same bits (values with the top bit set are negative int32).

For bits in {2, 4, 8} nothing straddles. That is `32 / bits` values per word,
lowest K index in the lowest bits:

```
word[k0 // (32/bits), n] |= (unsigned[k0 + j, n] & mask) << (j * bits)
j = 0 .. 32/bits - 1
```

For bits == 3, 32 values occupy 3 words. 10 values, one straddling value,
10 values, and so on. Shape is `K * 3 / 32` rows (K = 2048 -> 192 rows,
K = 512 -> 48 rows). This matches humming `common_pack_weight` and the legacy
AutoGPTQ 3-bit pack.

Stock vLLM INC builds `pack_factor = 32 // bits`, which is 10 for 3-bit and
does not divide K. Do not size int3 `qweight` as `[K / 10, N]`. Use
`K * bits / 32`. `INCConfig.pack_factor` is a `Fraction` already; the inner
WNA16 scheme still does integer `32 // weight_bits`.

`qzeros` is int32 `[K / 128, N * bits / 32]`, same bitstream packed along N
(dim 1), constant lane `maxq - 1`.

## Tensor names

Per Linear, four tensors, prefix = the HF module name:

```
{prefix}.qweight   int32
{prefix}.qzeros    int32
{prefix}.scales    fp16
{prefix}.g_idx     int32
```

Routed experts are unfused, one Linear per expert, matching
`models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2`:

```
model.language_model.layers.{L}.mlp.experts.{E}.gate_proj.*
model.language_model.layers.{L}.mlp.experts.{E}.up_proj.*
model.language_model.layers.{L}.mlp.experts.{E}.down_proj.*
```

`gate_proj` is the first half of the fused BF16 `experts.gate_up_proj`
(`[E, 1024, H]` -> gate `[E, 512, H]`, up `[E, 512, H]`). `down_proj` is
`[H, 512]` per expert (`[E, H, I]` in the fused parameter). This matches the
HF `Qwen3_5MoeExperts` chunk, not a cosine check against a second source.

vLLM `Qwen3_5ForCausalLMBase.packed_modules_mapping` stacks on the N axis:

| packed name     | pieces                         | all pieces |
|-----------------|--------------------------------|------------|
| `qkv_proj`      | `q_proj`, `k_proj`, `v_proj`   | int4 g128  |
| `gate_up_proj`  | `gate_proj`, `up_proj`         | same bits  |
| `in_proj_qkvz`  | `in_proj_qkv`, `in_proj_z`     | int4 g128  |
| `in_proj_ba`    | `in_proj_b`, `in_proj_a`       | int4 g128  |

Stack `qweight` on dim 1, `scales` on dim 1, `qzeros` on the packed-N dim.
Every piece in a stack has the same K, bits, and group. `g_idx` is identical
across the stack (it indexes K).

`self_attn.q_proj` is `[8192, 2048]`, not `[4096, 2048]`. The extra half is
the attention output gate fused into `q_proj` (chunked after the projection,
then `attn_output * sigmoid(gate)` before `o_proj`). Quantize it as one int4
matrix. `q_norm` / `k_norm` stay bf16 and apply to the query/key half only.

`linear_attn.in_proj_a` and `in_proj_b` have N = 32. That is the minimum the
bitstream allows. `shared_expert_gate` has N = 1 and stays bf16.

## What stays bf16

Router `mlp.gate`, `shared_expert_gate`, input and post-attention RMSNorms,
`linear_attn.norm`, `q_norm`, `k_norm`, `conv1d`, `A_log`, `dt_bias`,
embeddings, `lm_head`, final norm, the whole vision tower, and the whole MTP
block (including MTP experts). `extra_config` marks those with `"bits": 16`.
A skipped module has no `qweight`. The loader must take the raw `.weight`.

## Recipes and mixed bits

`config.json` `quantization_config.quant_method` is `auto-round` (vLLM renames
that to INC). `packing_format` is `auto_round:auto_gptq`. `sym` true.
`data_type` `int`. Default `bits` 4, `group_size` 128.

- `ornith-w4a16-g128`: every quantized Linear is int4, including experts.
- `ornith-w3a16-g128-experts-w4a16-trunk`: routed language experts (not MTP)
  are int3. Attention and the shared expert are int4.
- `ornith-mixed-int3-int2-experts-w4a16-trunk`: same trunk. Each routed
  expert is int3 or int2. The bit is in `expert_bits.json` (`bits` int8
  `[40, 256]`, plus `counts`). Exact `extra_config` keys override a regex,
  so int2 experts have an exact entry and the expert regex says 3. The
  `qweight` row count is the other copy of the bit width:
  `rows = K * bits / 32`.

A fused MoE module whose experts do not share one bit width makes stock INC
raise (`Fused MoE layer requires consistent quant config`). Recipe (b) cannot
be one fused WNA16 scheme. Read `expert_bits.json` and the per-expert tensors.
Recipes (a) and (c) are uniform per layer and can be stacked on the expert
axis after the N-axis gate/up stack.

There is no XPU int3 or int2 MoE kernel in the tree this file was written
against. int4 MoE is the stock path. int2 linear needs ARK, which is not
installed. B2 owns those kernels.

## Dequant check (reference expert, not a quality claim)

Ornith GPTQ int4 expert-0 down_proj versus the matching BF16 row:

- `(unsigned - 8) * scale`: rel L2 0.103923, cosine 0.994586
- `(unsigned - 7) * scale`: rel L2 0.530263
- RTN absmax/8 on the same BF16 matrix: rel L2 0.09966

The checkpoint's own quantizer is GPTQ, so the 0.104 figure is GPTQ error,
not SignRound error.
