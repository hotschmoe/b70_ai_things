# G3 -- Flash-Next model correctness (CPU)

## Verdict

The fork's XPU hyper-connection mix and PLE block match Strata's formulas
on the ISTA IQ3_XXS weights, in fp32, once the GGUF loader subtracts 1
from the stored norm. Sixteen gathered IQ4_NL rows from the first token
of the daily-driver math prompt match gguf-py with zero bit mismatches.
The XPU decoder does not run the QSA indexer. Dense GQA selects the same
cells as QSA only while the KV length is at most 2051. The Unsloth
`eh_proj` MTP head is a different graph from the daily driver's wide
fc stage.

No GPU was used. Expert pools and the GDN interior were not dequantized,
so a mismatch there will not show up in these checks.

## CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: CPU only. uv CPython 3.12.13. PYTHONPATH is the fork
`/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`, the venv site-packages,
`llama-pinned/gguf-py`, then `vllm/flashnext`. Shard 1 is the IQ3_XXS
GGUF (43.8 GiB mmap). Shard 2 is the PLE table (26.8 GiB mmap, 16 rows
copied). Tokenizer is
`/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext`. The pack tokenizer
dir is a llama.cpp dump and does not load as a Hugging Face tokenizer.
Fork `HyperConnection` and `Qwen4ExpPLE` imported on CPU (XPU device
count zero, `vllm._C_AVX2` absent). Activations are fp32. The daily
driver runs HC at Q8 (`STRATA_NIGHT_HC=q8`, `STRATA_HC_Q8=1`); that
rounding is not in this compare.

COMMAND: `bin/cpu-run python -m flashnext_ref.compare_layers --stage all`
(26 s of work after the CPU lock; log in the session). Collector:
`python flashnext_ref/collect_strata_ref.py` over 2619 JSON files.

RESULT: Numbers below. Raw dump is
`vllm/flashnext/flashnext_ref/last_compare.json`.

VERDICT: Use the checks as the localization ladder for C7. Layer-0 HC
and layer-1 PLE are the same formula as Strata under the loader
contract. QSA and the MTP sidecar are the places a long-context or
draft mismatch should be looked for first.

## PLE

Header vs the transcribed constants in `flashnext_ref/geometry.py`:
multipliers, 16 vocab sizes, and 16 offsets all match. Offsets are the
prefix sums. Vocab sizes sum to 320001446 of 320001536 table rows.
`qwen4exp.ple.layers` is `[1]`, the 0-based block.

`Qwen4ExpTextConfig(ple_layer_ids=[2], ...)` constructs. `layer_types[1]`
is `linear_attention`, `num_attention_heads` is 24,
`linear_num_value_heads` is 48, `ple_embed_dim` is 2560. HF ids are
1-based: blk.1 is id 2. Putting the GGUF value `[1]` into
`ple_layer_ids` attaches layer 0, which has no `ple_*` tensors.

EOS. Strata's cut is `PLE_EOS_TOKEN_ID = 248044`
(`include/strata/kernels/ngram.hpp:37`). The HF text config
`eos_token_id` is 248044. The tokenizer eos, and one of the generation
stop ids, is 248046. XPU `_compute_rows` uses `config.eos_token_id`
(`xpu/model.py:256`, fill at 346-351). The chat prompt
"Return only the integer: 17 * 19." with `enable_thinking=false` is 25
tokens and contains 248046, not 248044. That 25 matches the daily-driver
log's `prompt_tokens`.

On those 25 ids, strata's walk and the XPU transcription with eos
248044 differ in 0 of 400 row ids. The same transcription with eos
248046 differs in 32 of 400. The fork's `_compute_rows` matches the
transcription on the first 4 tokens (0 of 64). A loader that replaces
the text eos with the tokenizer eos changes the hash.

A fresh sequence matches: Strata rewrites a missing predecessor
(`TOKEN_NULL`, -1) to 248044 before the multiply, and XPU initialises
history to eos. A stored predecessor of -1 does not match (8 of 16
heads, the 3-gram half). `_ensure_state` never writes -1, so this is
latent on the text path.

Token id 0 is a real token. Every in-vocab product fits in signed
int64: `248319 * 23703573157769 = 5886047582964040311 < 2^63`. XOR keeps
the high bit clear, so the NVIDIA CPU `torch.remainder` path matches
the uint64 modulo on ordinary tokens. Measured 0 diffs on this prompt.
A negative id sets the high bit; that is the same latent -1 case.

IQ4_NL table `per_layer_token_embd.weight`: ggml ne `[160, 320001536]`,
numpy data `(320001536, 90)`, 28800138240 bytes, type IQ4_NL. The 16
rows for token 0 of the math prompt match `gguf.quants.dequantize` with
0 bit mismatches and max abs 0. Nibble order is split-half (byte j holds
elements j and j+16). No -8. Codebook is the standard IQ4_NL table.

PLE block, those 16 rows concatenated, real `blk.1` weights, one random
hidden row. Shapes: `ple_key` numpy `(10240, 2560)` BF16, `ple_value`
`(2560, 2560)` BF16, `ple_conv1d` `(10240, 4)` from ggml ne `[4, 10240]`
F16. Fork path with weights stored as `(file - 1)`, then
`_grouped_norm` / `_dilated_conv`: max abs 4.77e-7 against the numpy
block that multiplies the stored gamma. Using `(1 + stored)` instead
moves the block by max abs 1.46.

Norm means (the file, fp32):

| tensor | mean | min | max | `(1+(w-1))-w` |
| --- | --- | --- | --- | --- |
| blk.1.ple_norm_key | 0.893307 | 0.674 | 1.863 | 0 |
| blk.1.ple_norm_query | 0.893309 | 0.674 | 1.863 | 0 |
| blk.1.ple_norm_conv | 0.890633 | 0.543 | 1.275 | 0 |

`0.893 - 1 = -0.107`, which is the deviation mean the XPU comment
quotes. `ple.hpp:52` says the kernel does no `(1+w)` add. Read that as
"multiply the stored value". The GGUF value is already the folded
gamma. `mapping.py` `TRANSFORM_DEVIATION` subtracts 1 on load
(`mapping.py:623`) and `_grouped_norm` adds 1 (`model.py:316`). The
net is the stored value. Adding 1 a second time is the 1.46 error.

`qwen4exp.ple.image_token_id` is 248056, same as the HF config.
`_compute_rows` does not branch on it. An image cell hashes like any
other id once 248056 is in the token stream. No image prompt was run.

## Hyper-connections

Layer-0 attention HC, real BF16 weights, two fp32 tokens from
`Generator(1)`.

| weight | ggml ne | numpy shape |
| --- | --- | --- |
| hc_attn_down | [10240, 320] | (320, 10240) |
| hc_attn_up | [320, 10240] | (10240, 320) |
| hc_attn_inject | [10240, 4] | (4, 10240) |
| hc_attn_norm | [10240] | (10240,) |

`gr.hpp:83` stores `w_norm` as `(1+w)`. `HyperConnection.mix`
(`model.py:168-192`) does `gamma = 1 + weight`. With the parameter set
to `stored - 1`, mix vs numpy `gr_read(stored)` is max abs 1.31e-6
(mean abs 6.0e-8). Inject is max abs 1.01e-6. That is gemm order, not
a different formula. Leaving the stored gamma in the parameter
(deviation transform skipped) is max abs 1.278, the same gap as numpy
`(1+stored)` vs `stored`.

Other norm means, same round-trip of 0:

| tensor | mean | note |
| --- | --- | --- |
| blk.0.hc_attn_norm | 0.936453 | deviation about -0.064 |
| blk.0.hc_ffn_norm | 0.890454 | |
| blk.1.hc_attn_norm | 0.910808 | |
| blk.2.hc_attn_norm | 1.062375 | min -0.820, so some channels are negative |
| output_hc_norm | 3.749722 | min 0.633, max 13.375 |

`output_norm.weight` is not in the file. The final mixer is
`output_hc_*`, and `mapping.py:429` still applies the deviation
transform. Do not skip it because the mean is far from 1. The
round-trip on this tensor is exact, and Strata multiplies the stored
value (`gr.hpp:83` is the HC rule; the final mixer uses the same
container).

`gr_write` with a zero inject is a plain residual add (scale
`2*sigmoid(0) = 1`). Checked on a random residual, max abs 0.

## QSA / indexer

From the GGUF header, checked against tensor shapes:

- `attention.head_count` 24, `head_count_kv` 2, `key_length` 256.
- `rope.dimension_count` 64, `freq_base` 1e7, `dimension_sections`
  `[11, 11, 10, 0]`. HF `mrope_section` is the first three,
  `partial_rotary_factor` 0.25. `attention.layer_norm_rms_epsilon` is
  9.99999997e-7, the fp32 spelling of 1e-6. Strata hardcodes 1e-6
  (`qsa.hpp:120`).
- `blk.3.attn_q` ggml ne `[2560, 12288]` = 24 * (256 + 256). That is
  `[q|gate]` per head, which is what `Qwen3NextAttention` chunks
  (`qwen3_next.py:444-445`). `attn_k` and `attn_v` are `[2560, 512]`.
- Indexer tensors exist on blk.3: `q_proj` `[2560, 512]` BF16 (4 * 128),
  `k_proj` `[2560, 128]` BF16, `q_norm` and `k_norm` F32 `[128]`.
  `indexer.top_k` 2048, compress ratio 4, one KV head.

The XPU full-attention layer is `Qwen3NextAttention`, not the CUDA QSA
module (`xpu/model.py:538-548`). Indexer weights are created only when
the quant name is `b70_gguf`, so the loader consumes the tensors. The
forward does not read them. The comment at `model.py:102-103` says
these layers carry no indexer tensors. The file does.

Selection width is `min(n_kv, 2048 + 4 - 1) = 2051` (`qsa.hpp` comment
above `qsa_rms_eps`, and the CUDA indexer returns
`token_topk + compress_ratio - 1`). Synthetic checks in
`compare_layers.py --stage self`: width at 200000 is 2051; relu is per
head then summed (a two-head case gives 1.0, relu-of-sum gives 0);
the incomplete tail cell gets +1e9; GQA head 12 of 24 reads KV head 1,
not `12 % 2` as a different map would; the gate is the second half of
each head; RoPE position 0 is the identity. Below 2051 tokens the
selected set is every cell, so dense GQA matches QSA. At the 200k
target it does not.

Daily driver KV is `--kv int8` (`strata/flash-next/dd/hotschmoe-dd.json`).
A bf16-KV vLLM serve can flip greedy tokens even on a short prompt.
That flip was not measured.

## MTP

Daily driver: `--mtp .../mtp-bf16/rt`, `--spec 4`, `--mtp-max-t 4`.
The args do not include `--mtp-hnorm`. Strata's default is one RMS over
the full `hc * H` vector, then the same `fc_hidden` on each stream, with
the embedding broadcast onto every stream. XPU `_fc_stage`
(`xpu/mtp.py:182-202`) calls `forward_wide` (`mtp.py:93-106`), which is
that pooled RMS, with `gamma = 1 + weight`.

The GGUF sidecar is one `eh_proj` of shape `(H, 2H)`.
`try_gguf_fc_stage` (`gguf/mtp_proj.py:34-54`) mean-collapses the wide
norm and broadcasts one vector to all four streams. On one random draw
the wide path and the eh path differ by max abs 3.97, the eh streams
are identical, and the wide streams are not. Pooled RMS vs per-stream
RMS (the opt-in `--mtp-hnorm stream` graph) differs by max abs 0.024
on that draw. The `mtp-bf16/rt` bytes were not dequantized; this is the
layout, not a weight dump.

Target greedy tokens do not depend on the draft when the verifier is
greedy on the target. Draft acceptance will not match the daily driver
if C7 loads the Unsloth sidecar.

## Greedy references for C7

`vllm/flashnext/harness/strata_reference.json`. Collected from existing
logs. No live server call. 14 temperature-0 texts. Stored sha256 matches
the text when the log has one.

Short ones, model `hotschmoe-dd`, stack labelled `ista-iq3-dd`:

| prompt | text | prompt tokens | completion tokens | source |
| --- | --- | --- | --- | --- |
| Return only the integer: 17 * 19. | 323 | 25 | 4 | `lowlevel_probe/paired_whole_graph_serving_s106/clean/mode0/functional/concurrent-math.json` |
| Return only the integer: 23 * 29. | 667 | 25 | 4 | same file |
| Compute 23 times 17. Give just the result after reasoning. | 391 | 68 | 70 | `.../functional/thinking.json` (reasoning_content is 229 chars, the visible answer is 391) |
| Look up the build status of project b70-demo... | The build status for project b70-demo is passed, with 17 tests passed and 0 tests failed. | 379 | 25 | `.../functional/tool-final.json` |

A canary that prefixes the 17*19 prompt still answers 323, with 31
prompt tokens. Longer bench completions (256 tokens, prompts of 4k and
32k) are clipped to 2000 characters; the file keeps the sha256 of the
full text and the token counts.

No "capital of France" completion is in the daily-driver logs. The
paused UD-Q4 campaign is a different stack and is not mixed in.
`/mnt/vm_8tb/b70/results/hotschmoe-dd/engine.log` is 11634818 bytes.
The first 2 MB contains no `"content"` and no `completion_tokens`.

## What a layer mismatch will not catch

`blk.0.attn_qkv` is ggml ne `[2560, 10240]`. The split into GDN q, k, v
was inferred from the sizes by A1 and was not checked on a numeric
payload here. Routed experts (512, top-10, FFN 640) were not
dequantized. There is no Strata hidden-state dump for layers 0-2 in
this run. The numpy reference is the localization tool for HC, PLE,
the indexer score, and the MTP fc layout. It is not a full forward.

## For C7

1. The HF directory to serve is
   `/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext`. Its text config
   already has `ple_layer_ids [2]`, `ple_embed_dim 2560`, 24 heads, 48
   value heads, eos 248044, rope theta 1e7, partial rotary 0.25,
   `mrope_section [11, 11, 10]` interleaved. A skeleton with the same
   fields is `flashnext_ref/qwen4exp_text_config.json`. Do not copy
   GGUF `ple.layers = [1]` into `ple_layer_ids`.
2. Keep the text eos at 248044 for PLE. 248046 is the stop token and
   it occurs inside the 25-token math prompt. Using it as the PLE cut
   changes 32 row ids.
3. Set `B70_EAGER_QK=1` when comparing attention numbers. The fused
   Triton qk-norm-rope-gate path is off on CPU and is the wrong place
   to debug a GGUF mismatch (`qwen3_next.py:391`).
4. Dense attention matches QSA selection only up to 2051 tokens. The
   200k target needs the indexer, which this XPU forward does not run.
5. Daily-driver draft behavior is the wide fc path in `mtp-bf16/rt`,
   not `eh_proj`.
6. GGUF HC and PLE norms are already `(1+w)`. The loader has to
   subtract 1. Skipping that changes the layer-0 mix by max abs 1.28
   and the layer-1 PLE block by max abs 1.46.
7. First greedy checks: `17 * 19` -> `323`, `23 * 29` -> `667`. The
   daily driver uses int8 KV. A bf16 KV serve is a separate variable.
