# 02 -- Qwen3.8-Flash-Next model support (vLLM main)

Date: 2026-10-19. CPU only. No GPU.

## Sources verified

GGUF metadata was read with `python3 -I` and `GGUFReader` from
`/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py` (numpy injected on
`sys.path` because `-I` hides user site-packages). Tensor payloads were not
copied; the reader memory-maps the header.

| File | Bytes | Role |
|---|---|---|
| `/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47039860096 | IQ3_XXS weights. Read. |
| `.../IQ3_XXS/...-00002-of-00002.gguf` | symlink, 105-byte link text | Points at the shared shard 2 below. |
| `/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28800138432 | Shared n-gram shard. Same sha256 for XXS and IQ3_S in `strata/flash-next/model-lock-ista-iq3.json`. Read. |
| `models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54817524224 | IQ3_S weights. Not on `/mnt/cache`. Read. |
| `models/files/.../ista-gsq-rco-iq3-s/mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907543008 | Vision tower. Read. |
| `models/files/.../unsloth-ud-q4-k-xl/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` | 2786204800 | MTP sidecar. Read. |
| `models/files/.../unsloth-ud-q4-k-xl/UD-Q4_K_XL/*-00001-of-00004.gguf` | 10946624 | Metadata-only split shard (0 tensors). Read. Shards 2-4 were not re-histogrammed. |

A second IQ3_XXS shard-1 copy exists at
`models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs/` with the same byte
size. It was not hashed against the cache copy.

No `config.json` and no `*.safetensors` for this model under
`/mnt/cache/b70`, `models/files/qwen3.8-flash-next`, or `/mnt/vm_8tb/b70/models`.
`compat-bf16.json` under `/mnt/vm_8tb/b70/models/flashnext-native-source-pack-*`
is a 21 KB dtype map of the Unsloth GGUF, not a BF16 checkpoint.
HF BF16 MTP tensors were already extracted (shapes only used below) at
`/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/mtp-inventory.json`. The served
Strata pack is `/mnt/vm_8tb/b70/build/strata-ab/packs/ista-gsq-rco-iq3-xxs`
and `/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt`.

vLLM tree: `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm` at
`5fd62688ffe995c0c11c8708fe51dbff53235b74` (`main`).
GGUF support is no longer in that tree. It is the plugin
`vllm-project/vllm-gguf-plugin`, shallow-cloned for this report to
`/mnt/vm_8tb/b70/build/vllm-flashnext/vllm-gguf-plugin-ro` at
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8` (2026-09-24, base version 0.0.5).

## Architecture

`general.architecture = qwen4exp`. `general.description` is "A Preview of the
Qwen4 Architecture". `general.size_label = 512x56B`. Tokenizer pre-tokenizer
is `qwen35` (GPT-2 BPE). This is not a `qwen3next` or `qwen35moe` GGUF.

| Item | Value | Evidence |
|---|---|---|
| Layers | 48 | `qwen4exp.block_count`. Tensor names `blk.0`..`blk.47`. |
| Hidden | 2560 | `embedding_length`. |
| Vocab | 248320 | `token_embd.weight` shape `[2560, 248320]`. Untied: embd is IQ3_S (XXS) or IQ4_XS (IQ3_S); `output.weight` is Q5_K (XXS) or Q6_K (IQ3_S). |
| Bos / pad / eos | 248044 / 248044 / 248046 | `tokenizer.ggml.*`. `add_bos_token` false. |
| Context | 262144 | `qwen4exp.context_length`. 200k is inside this. |
| RMS eps | 1e-6 | `attention.layer_norm_rms_epsilon`. |
| RoPE | dim 64, base 1e7, sections `[11, 11, 10, 0]` | mRoPE-style. 11+11+10 = 32 = rotary_dim/2. |
| Routed experts | 512, top-10 | `expert_count`, `expert_used_count`. |
| Expert FFN | 640 | `expert_feed_forward_length`. Gate/up `[2560, 640, 512]`, down `[640, 2560, 512]`. |
| Shared expert | 1, FFN 640 | `expert_shared_feed_forward_length`. Gate vector `[2560]` (`ffn_gate_inp_shexp`). |
| Router | BF16 `[2560, 512]` every layer | `ffn_gate_inp.weight`. |
| Linear / GDN | 36 layers | `blk.{0,1,2,4,5,6,...}` have `attn_qkv`, `attn_gate`, `ssm_*` and no `attn_q`. |
| Full attn + indexer | 12 layers | `blk.{3,7,11,...,47}`. `full_attention_interval = 4`. `compress_ratios` is 0 on linear layers and 4 on these. |
| Full-attn heads | Q 24, KV 2, head 256 | `head_count`, `head_count_kv`, `key_length`, `value_length`. |
| Q packing | `[q \| gate]` per head | `attn_q` is `[2560, 12288]` = 24 * (256+256). `attn_k`/`attn_v` are `[2560, 512]` = 2*256. ggml comment: `constants.py:2943` in the strata gguf-py. vLLM sets `attn_output_gate = True` unconditionally (`qsa.py:409-411`). |
| Indexer (QSA) | 4 query heads, key 128, top-2048 | `attention.indexer.*`. Tensors only on the 12 full layers: `indexer.q_proj [2560, 512]`, `indexer.k_proj [2560, 128]`, both BF16, plus F32 norms. |
| GDN (stored as `ssm_*`) | 16 key groups, state 128, inner 6144, conv 4, dt rank 48 | `qwen4exp.ssm.*`. `attn_qkv` `[2560, 10240]`, `attn_gate` `[2560, 6144]`, `ssm_out` `[6144, 2560]`. 10240 = 2*16*128 + 6144, so 16 key heads of 128 and 48 value heads of 128. |
| Hyper-connections | count 4, low rank 320 | `hyper_connection.*`. Width 10240 = 4*2560. Per layer, attn and mlp each have down `[10240, 320]`, up `[320, 10240]`, inject `[10240, 4]`, norm F32 `[10240]`, all BF16 except the norm. Same trio on the output mixer (`output_hc_*`). No `output_norm` / `attn_norm` tensors. |
| PLE | layer block 1 only | `ple.layers = [1]`. `blk.1.ple_{key,value,conv1d,norm_*}` only. Conv kernel 4, ngram size 3, 8 heads per ngram, 16 hash vocabs. |
| PLE table | 51,200,245,760 params | Shard 2 tensor `per_layer_token_embd.weight`, shape `[160, 320001536]`, type IQ4_NL. 16 vocabs sum to 320001536 (`ple.head_vocab_sizes`, each ~20,000,0xx). Row is 90 bytes. `embedding_length_per_layer_input = 160`. |
| Vision | separate mmproj, not in the text GGUF | `clip`, projector `qwen3vl_merger`. 27 blocks, width 1152, FFN 4304, 16 heads, patch 16, image 768, merge 2, proj dim 2560. Deepstack flags all false. Weights F32 norms/bias + BF16 mats. `general.file_type = 32` (`MOSTLY_BF16`). Size label 449M. Chat template mentions video; the mmproj KV has no video or audio encoder. |
| MTP | not in the ISTA text GGUF | No `nextn.*` tensors in either ISTA shard. Unsloth sidecar is also `qwen4exp`, `block_count = 49`, `nextn_predict_layers = 1`, tensors on `blk.48` (full attn + indexer + MoE + HC) plus `nextn.{eh_proj,enorm,hnorm,hc_head_*}`. HF names in `mtp-inventory.json` (`mtp.layers.0.*`, 512 experts, shared expert, indexer, HC) match one full-attn block. |

Layer pattern, verified from tensor names, is linear, linear, linear, full,
repeated. Block 1 is linear plus PLE. Full layers are the only ones with
`attn_{q,k,v,output}`, q/k norms, and `indexer.*`.

Element counts from the reader:

| Pool | Elements | Notes |
|---|---|---|
| Routed experts, 48 layers * 3 mats | 120,795,955,200 | 96.1% of shard 1. Per expert: 3 * 2560 * 640 = 4,915,200. |
| Active routed params / token | 2,359,296,000 | 10/512 of the routed pool. |
| Shared experts | 235,929,600 | Always on. |
| Rest of shard 1 | 4,947,698,560 | Attn, GDN, HC, indexer, PLE projections, embeddings, head. |
| Shard 1 total | 125,743,653,760 | Same element count in IQ3_XXS and IQ3_S. |
| PLE table | 51,200,245,760 | Lookup, not a matmul. |

Inferred GDN dims (16 key heads * 128, 48 value heads * 128) are the unique
split of `ssm.group_count`, `ssm.state_size`, `ssm.inner_size`, and the
`attn_qkv` width 10240. vLLM's state formula is
`gated_delta_net_state_shape` in `mamba_utils.py:281-302`.

## Quant-type histograms

`general.file_type` is a single label, not the mix. XXS file_type 23 =
`MOSTLY_IQ3_XXS`. IQ3_S file_type 26 = `MOSTLY_IQ3_S`
(strata `gguf/constants.py:5751` and `:5754`). The histograms below are the
reader's `GGMLQuantizationType` counts. They match the RCO allocation text
shipped beside the IQ3_XXS weights.

### IQ3_XXS shard 1 (1223 tensors, 47028835840 tensor bytes = 43.797 GiB)

| Type | Count | Where it matters |
|---|---|---|
| BF16 | 484 | Every router, shared-expert gate, HC matmul, indexer proj, `ssm_alpha`/`ssm_beta`. |
| F32 | 292 | Norms, `ssm_a`, `ssm_dt`, `ssm_conv1d`, `ssm_norm`. |
| IQ3_S | 82 | Mixed dense (attn, shared up, token embd) plus 13 layers of gate/up experts. |
| Q6_K | 71 | Mostly `attn_qkv` / `ssm_out` / full-attn out. |
| IQ4_XS | 67 | Dense attn and shared gate. |
| IQ4_NL | 58 | 18 of 48 `ffn_down_exps`, 40 of 48 shared downs. Block 32. |
| Q4_K | 45 | Dense. |
| Q2_0 | 38 | 30 of 48 `ffn_down_exps`, 8 shared downs. ggml type id 42, block (64, 18). |
| IQ2_XS | 20 | 10 layers of `ffn_gate_exps` and 10 of `ffn_up_exps` (paired). |
| IQ2_S | 20 | Same pairing. |
| IQ2_XXS | 18 | Same pairing, 9 layers. |
| Q5_K | 15 | Dense, including `output.weight`. |
| IQ3_XXS | 12 | 6 layers of gate experts and 6 of up experts. |
| F16 | 1 | `blk.1.ple_conv1d.weight`. |

Expert down is only IQ4_NL (18) or Q2_0 (30). Expert gate and up are only
IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S. There is no Q8_0 and no Q8_1 in
this file.

### IQ3_XXS / IQ3_S shard 2 (1 tensor, 28800138240 bytes = 26.822 GiB)

`per_layer_token_embd.weight`: IQ4_NL, 51,200,245,760 elements. Identical
file for both quants.

### IQ3_S shard 1 (1223 tensors, 54806499840 tensor bytes = 51.043 GiB)

Same shapes and the same 48/36/12/PLE layout. Different mix:

| Type | Count | Where it matters |
|---|---|---|
| BF16 | 484 | Same fixed tensors as XXS. |
| F32 | 292 | Same. |
| Q6_K | 129 | Most full-attn and many `attn_qkv` / `ssm_out`. |
| IQ4_NL | 86 | 39 of 48 expert downs, 47 of 48 shared downs. |
| Q4_K | 47 | Dense. |
| IQ4_XS | 45 | Dense, plus 1 expert gate/up layer and the token embd. |
| IQ2_S | 40 | 20 layers of expert gate and 20 of expert up. |
| Q5_K | 35 | Dense. |
| IQ3_XXS | 34 | 17 layers of expert gate and 17 of expert up. |
| IQ3_S | 20 | 10 layers of expert gate and 10 of expert up. |
| Q2_0 | 9 | 9 expert downs. |
| F16 | 1 | `ple_conv1d`. |
| Q8_0 | 1 | One `ffn_down_shexp`. The only Q8 in the ISTA text weights. |

### mmproj

334 tensors, 907523008 tensor bytes (0.845 GiB). F32 x224, BF16 x110.
No IQ/K quant.

### Unsloth MTP Q4_K_M

34 tensors, 2775258112 tensor bytes (2.585 GiB), file_type 15
(`MOSTLY_Q4_K_M`). Q4_K x13, F32 x11, Q6_K x3, Q5_0 x3, Q8_0 x2, BF16 x2.
The two Q8_0 tensors are `ffn_down_exps` (891289600 bytes) and
`ffn_down_shexp`. The file also carries its own `token_embd` and `output`,
so the draft block alone is smaller than 2.585 GiB. ISTA does not ship this
file. Strata's runtime MTP pack (`mtp-bf16/rt`) is a different quant of the
HF BF16 MTP tensors: `dense.bin` 116099072 bytes plus `experts.bin`
707788800 bytes.

### Unsloth UD-Q4_K_XL

Shard 1 is metadata only (`split.count = 4`, `split.tensors.count = 1224`,
same `qwen4exp` keys as ISTA, including PLE). Shards 2-4 are
49859583136 + 49376141504 + 12087983520 bytes. All four files sum to
111334654784 bytes (103.691 GiB). Per-tensor types of shards 2-4 were not
read in this pass.

## Does vLLM main have this model?

Yes. The class is `Qwen4Exp`, not `Qwen3Next` and not `Qwen3.5`.

Registered at `vllm/model_executor/models/registry.py`:

- `Qwen4ExpForCausalLM` line 116-119, loader `vllm.models.qwen4_exp`
- `Qwen4ExpForConditionalGeneration` line 607-609
- `Qwen4ExpMTP` line 708

`Qwen3NextForCausalLM` is line 115. `Qwen3_5ForConditionalGeneration` /
`Qwen3_5MoeForConditionalGeneration` are lines 602-606. Those are the parents
the Qwen4Exp file imports (`nvidia/model.py:47-60`): GDN attention, the
Qwen3-Next MoE block, and the Qwen3.5 multimodal shell. Qwen4Exp then adds
hyper-connections, PLE, and QSA.

XPU is refused before any of that runs:

```
vllm/models/qwen4_exp/__init__.py:30-31
  if current_platform.is_xpu() or current_platform.is_tpu():
      raise NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")
```

ROCm has a parallel implementation under `vllm/models/qwen4_exp/amd/`. CUDA
is `nvidia/`. There is no XPU subdirectory.

What the CUDA class implements, matched to this checkpoint:

- Hybrid layers. `linear_attention` uses `QwenGatedDeltaNetAttention`.
  Full layers use `Qwen4ExpQSAAttention` when `indexer_n_heads` is set
  (`model.py:227-254`). Accepted full-layer type strings are
  `qwen_sparse_attention` and `indexed_attention` (`model.py:92-94`).
- MoE with a shared expert (`model.py:171-188`).
- HC on attn, mlp, and the head (`model.py:264-278`).
- PLE as an n-gram table that can live in pinned host memory and be gathered
  through UVA (`common/ngram_embedding.py:1-7` and `:626-646`). The pinned
  path calls `torch.cuda.Stream` (`:660`).
- MTP class `Qwen4ExpMTP`.
- Vision via `Qwen3_VisionTransformer` (`model.py:55-60`), which matches the
  mmproj's `qwen3vl_merger` layout.

QSA constraints that matter for the 2x200k target, all in
`nvidia/qsa.py`:

- Line 252-253: requires FlashAttention (`is_flash_attn_varlen_func_available`).
- Line 357-358: model dtype must be BF16.
- Line 359-369: main KV is BF16 or FP8-e4m3. A quant-config `kv_cache_scheme`
  is rejected ("does not support KV quantization").
- Line 552-560: main cache is a full `FullAttentionSpec` of `num_kv_heads` by
  `head_dim` for K and V. Compression does not shrink this tensor.
- Indexer side state is separate and small: a per-request ring of raw BF16
  keys (`qsa_cache.py:842-860`) plus one compressed key per
  `indexer_compress_ratio` tokens (`qsa_cache.py:863-874`).

PLE attachment hazard: weights exist only on `blk.1`. vLLM enables PLE when
`(layer_idx + 1) in config.ple_layer_ids` (`model.py:211-212`). Copying
`qwen4exp.ple.layers = [1]` straight into `ple_layer_ids` would attach PLE
to `layer_idx` 0, which has no `ple_*` tensors. No local HF config exists
to show which convention the official checkpoint uses.

`Qwen3Next` / `Qwen3.5-MoE` cannot load this GGUF as-is. The checkpoint has
no `attn_norm` (HC replaced it), packs a sigmoid gate into `attn_q`, and
adds indexer, HC, and PLE tensors those classes do not own.

## Can the GGUF loader load this arch and these types?

Not on this tree, and not on XPU.

In-tree GGUF is gone. `vllm/model_executor/layers/quantization/` has no
`gguf.py`. `setup.py:1748` puts it in the extra `vllm-gguf-plugin>=0.0.2`.
`XPUPlatform.supported_quantization` (`vllm/platforms/xpu.py:113-131`) lists
awq, gptq, moe_wna16, inc, fp8, mxfp4, mxfp8, modelopt, compressed-tensors,
and others. It does not list `gguf`. A search of
`/mnt/vm_8tb/b70/build/vllm-flashnext/vllm-xpu-kernels` found no GGUF, IQ, or
ggml dequant sources.

The plugin (commit above):

- Adapter registry is Gemma3, Gemma4, OLMoE, Qwen3.5, Qwen3.5-MTP
  (`weights_adapter/__init__.py:19-33`). There is no Qwen4Exp adapter.
- `Qwen35GGUFAdapter.matches` is only `qwen3_5`, `qwen3_5_text`,
  `qwen3_5_moe`, `qwen3_5_moe_text` (`qwen3_5.py:29-43`, `:189-190`). Its
  rename table covers GDN `ssm_*` names and MoE experts. It does not rename
  `hc_*`, `ple_*`, `indexer.*`, or `per_layer_token_embd`.
- Config parsing does not build a config from GGUF keys. A local `.gguf`
  path uses the parent directory as the HF config source
  (`config_parser.py:61-63`). These directories have no `config.json`.
- The generic fallback maps `config.model_type` onto `gguf.MODEL_ARCH_NAMES`
  (`transformers.py:126-132`). The arch string is `qwen4exp`. A Transformers
  `model_type` of `qwen4_exp` would not match that string. Even if it did,
  the installed `gguf` package has to know the Qwen4Exp tensor map; the
  plugin does not vendor it. The map that matches these files is the strata
  llama-pinned gguf-py (`tensor_mapping.py:2772-2829`, `constants.py:2928-2978`),
  which is not what the plugin imports.

Kernel coverage in the plugin, for the types that actually occur:

| Type in the ISTA files | CUDA GEMV (`ops.py:54-76`) | CUDA GEMM (`ops.py:77-90`) | Triton (`triton/gemm/utils.py:49-70`) |
|---|---|---|---|
| Q4_K, Q5_K, Q6_K, Q8_0 | yes | yes | yes |
| Q5_0 (MTP only) | yes | yes | yes |
| IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | yes | no, falls through to Triton | yes |
| Q2_0 (ggml type 42) | no | no | no. Not in the type tables. |
| BF16, F16, F32 | not quant kernels; plain tensors | | |

Triton is not an XPU path. Dequant and GEMM both reject non-CUDA tensors:

- `triton/dequantize/utils.py:81-82` `if not W.is_cuda: raise ValueError("Triton dequant kernels require CUDA tensors")`
- `triton/gemm/utils.py:230-231` same check on W and X
- `triton/fused_moe/utils.py` requires `is_cuda` on the MoE operands

So IQ expert GEMMs are CUDA-or-Triton-on-CUDA. Q2_0 has no kernel at all.
IQ3_XXS uses Q2_0 for 30 of 48 expert down projections. IQ3_S uses it for 9.
A loader that otherwise understood `qwen4exp` would still fail on those
tensors. IQ2, Q5_K, Q6_K, and the single Q8_0 have CUDA kernels; they do
not have XPU kernels.

IQ layouts are lookup grids, not packed int4. They cannot be handed to the
XPU AWQ/GPTQ/INC int4 kernels without a dequant-and-requant.

## VRAM budget (2x 32 GiB = 64 GiB)

Main KV, from `FullAttentionSpec` and the GGUF head geometry. Only the 12
full layers store token KV. K and V are each 2 heads * 256.

- BF16: 12 * 2 * 256 * 2 * 2 bytes = 24576 bytes/token.
  One 200k stream = 4.578 GiB. Two streams = 9.155 GiB.
- FP8-e4m3, which QSA allows: 4.578 GiB for two streams.

QSA compressed side cache, inferred from `MLAAttentionSpec` with
`tokens_per_state = 4`, 1 head, dim 128, 12 layers: 0.286 GiB BF16 for
2x200k, half of that in FP8. The raw-key ring is a few dozen KiB per
request. GDN recurrent state, inferred by plugging the GGUF dims into
`gated_delta_net_state_shape` and assuming FP32: about 0.22 GiB for two
streams. PLE conv state is negligible (one layer). These three are under
0.6 GiB combined. They are not the budget problem.

PLE table is 26.822 GiB at IQ4_NL, or 95.368 GiB if it were BF16. vLLM's
pinned-host PLE path exists for CUDA UVA, not for XPU, and the model class
never gets that far on XPU. Strata already keeps this shard as a sparse
lookup. Budgets below leave it in host RAM. Putting it in VRAM blows every
option except a heavily quantized table plus a much smaller weight set.

Activations, graph scratch, and allocator fragmentation are not measured
here. A fit that leaves under ~4 GiB is treated as not servable at 200k.

| Format | Shard-1 / weight bytes | + 2x200k BF16 KV (9.155 GiB) | Fits 64 GiB with PLE in host? |
|---|---|---|---|
| IQ3_XXS tensor payload | 43.797 GiB | 52.95 GiB, ~11 GiB left before vision/MTP | Size yes. Kernels and the XPU model class no. |
| IQ3_XXS + mmproj 0.845 + ~2 GiB MTP | ~46.6 GiB | ~55.8 GiB | Size still yes. Same blockers. |
| IQ3_S tensor payload | 51.043 GiB | 60.20 GiB, ~3.8 GiB left | Too tight once vision, MTP, or a prefill activation is added. FP8 KV brings it to ~55.6 GiB. |
| Uniform int4, group 128, FP16 scales, all shard-1 elements | 58.556 + 1.830 = 60.386 GiB | 69.54 GiB | No. FP8 KV still 64.96 GiB before activations. |
| Uniform int3, same scale assumption | 43.917 + 1.830 = 45.747 GiB | 54.90 GiB | Size yes. No XPU MoE kernel. |
| Uniform int2, same scales | 29.278 + 1.830 = 31.108 GiB | 40.26 GiB | Size yes. XPU linear path allows symmetric int2. Quality is the weak end. |
| FP8 or W8A8 of shard 1 | 117.15 GiB | over | No. |
| BF16 shard 1 | 234.215 GiB | over | No. Plus PLE BF16 95.368 GiB = 329.58 GiB (353.89 decimal GB), which matches the card's 354 GB BF16 line. |
| Intel AutoRound W4A16, published size | 181196007016 bytes = 168.752 decimal GB | over | Not on disk. The 2026-10-09 note only fetched a header. PLE left in BF16 explains the jump over a 60 GiB int4. |
| Unsloth UD-Q4_K_XL, all 4 shards | 103.691 GiB | over | No, even if one shard is the PLE table and stays on host. |

## Ranked weight options

1. Keep the ISTA IQ3_XXS GGUF as the weight source (43.797 GiB) and teach
   vLLM to run it. This is the only artifact that is local, is the current
   daily-driver quality point, and fits in 64 GiB beside 2x200k BF16 KV with
   PLE in host RAM (~11 GiB theoretical headroom, less after the 0.845 GiB
   mmproj and an MTP block). It is not loadable today. The work is an
   XPU Qwen4Exp port plus a GGUF adapter and XPU kernels for every type in
   the histogram, including Q2_0. CUDA GGUF kernels do not cover Q2_0, and
   the Triton fallback checks `is_cuda`.

2. Do not convert the IQ grids into a vLLM int4 checkpoint. The grids are
   not nibbles, so the conversion is dequant-then-requant (a second quality
   loss) and the int4 result is 60.386 GiB. With the KV this target needs,
   that does not fit. XPU can run symmetric int4: INC WNA16 lists bits
   `{2, 4}` (`inc/schemes/inc_wna16_scheme.py:21`), oneDNN backends are
   int4 only (`:29-30`), and the ARK MoE kernel rejects anything except
   symmetric int4 (`inc/schemes/inc_wna16_moe.py:38-40`). The kernel exists.
   The budget does not.

3. An int3 checkpoint (INC config allows bits `{2,3,4,5,6,7,8}` at
   `inc/inc.py:37`, and compressed-tensors WNA16 has a uint3 type) is about
   45.75 GiB plus 9.16 GiB KV, which fits. On CUDA the odd widths go to the
   humming kernel (`inc_wna16_scheme.py:22-26`). On XPU the MoE kernel is
   int4-only, so the 120.8B expert params would not run. There is also no
   local BF16 to quantize from. Dequantizing IQ3_XXS and then quantizing to
   int3 spends the quality twice to land in a format XPU cannot execute.

4. Int2 on the XPU INC linear path fits easily (~40 GiB with BF16 KV). It
   is a worse operating point than the ISTA card's own Q2_0 build, and the
   XPU MoE kernel still requires int4, so the expert mats are the unsolved
   part. Not a serving candidate for this target.

5. Requantizing from BF16 is blocked on disk. No backbone BF16 or safetensors
   copy is local. The card's 354 GB figure matches the element math. The
   published AutoRound W4A16 (168.8 decimal GB, PLE left in BF16) is not
   local and does not fit. Downloading either file does not change the
   64 GiB result.

6. Unsloth UD-Q4_K_XL is local and is the same `qwen4exp` architecture, but
   103.7 GiB does not fit. Leave it paused.

Vision and MTP are add-ons, not alternate backbones. The mmproj is 0.845 GiB
BF16 and matches the Qwen3-VL merger the Qwen4Exp class already calls. MTP
weights exist as the Unsloth Q4_K_M sidecar (2.585 GiB including a private
embedding and head) and as the smaller Strata q8 runtime pack. The ISTA text
GGUF has zero MTP tensors. `Qwen4ExpMTP` is registered, and it is under the
same XPU refusal as the base class.

## Bottom line

Qwen3.8-Flash-Next is a 48-layer Qwen4Exp MoE: 36 Gated-DeltaNet layers,
12 QSA full-attention layers (24/2 heads, dim 256, indexer top-2048,
compress ratio 4), 512 experts of width 640 with top-10 and one shared
expert, hyper-connections of width 4, and a single PLE n-gram table on
block 1. vLLM main has that class for CUDA and ROCm and refuses it on XPU.
The GGUF plugin has neither a Qwen4Exp adapter nor an XPU kernel, and it
has no kernel at all for Q2_0, which IQ3_XXS uses on 30 expert downs.
The weight file that fits 2x32 GiB with 2x200k KV is the existing IQ3_XXS
GGUF with the 26.8 GiB PLE table left in host memory. A native int4
conversion of the same weights does not fit that KV.
