# 16 - Model-store prepack audit (W4A8 int-quantized)

Audit of `/mnt/vm_8tb/b70/models/` (+ `models/archive/`) on the GPU host (Unraid @ 192.168.10.5)
for compressed-tensors W4A8 models that store 4-bit weights UNPACKED as int8 and therefore
benefit from (or require) offline pre-packing.

Date: 2026-06-20. Auditor: model-store auditor agent. GPU NOT touched (CPU-only inspect + prepack).


## Background -- why prepack

compressed-tensors W4A8 uses the **`int-quantized`** format. It stores the 4-bit weights
UNPACKED as int8 (1 byte/elem) on disk. On load, vLLM moves all those I8 weights to the GPU,
then packs them to int4 [out, in/8] int32 via `XPUW4A8IntLinearKernel._pack_int4_weight`
(`contrib/vllm_int8_xpu .../xpu.py`). That on-load step creates a large GPU transient (unpacked
I8 + the new packed int4 held simultaneously). For big models it OOMs/hangs the 32 GiB B70.

Offline fix: `/mnt/vm_8tb/b70/offline_prepack_w4a8.py` packs the I8 weights -> int32 [out, in/8]
OFFLINE (byte-matching the kernel), writes a `-prepacked` dir, and sets
`quantization_config.is_prepacked_w4a8=true`. vLLM then loads the small packed weights directly,
no on-load pack, no I8 transient.

Rules:
- ONLY `format=int-quantized` with **4-bit weights** (W4A8) prepacks.
- `pack-quantized` (W4A16 / W8A16) is ALREADY packed -> do NOT prepack.
- `int-quantized` **W8A8** is genuinely int8 (8-bit weights, not 4-bit) -> does NOT prepack.
- auto-round / fp8 / gguf / fp16-bf16 -> different formats, not candidates.


## Full model table

A "weight" is quantized iff a sibling `.weight_scale` exists; the 14B-W4A8 header shows
280 I8 quantized weights + 163 BF16 (norms/embeddings/ignored layers).

```
name                                       size   quant_method        format          scheme    prepack?
------------------------------------------ ------ ------------------- --------------- --------- -----------------------------
models/
  Qwen3-14B-W4A8-gptq                       16G    compressed-tensors int-quantized   W4A8      YES (beneficial) -> DONE
  Qwen3-14B-W4A8-gptq-prepacked            9.3G    compressed-tensors int-quantized*  W4A8      (output of this audit)
  Qwen3-14B-W4A16-gptq                     9.3G    compressed-tensors pack-quantized  W4A16     no (already packed)
  Qwen3-14B-W8A16                           16G    compressed-tensors pack-quantized  W8A16     no (already packed, 8-bit)
  Qwen3-14B-W8A8-gptq                       16G    compressed-tensors int-quantized   W8A8      no (8-bit weights, genuine I8)
  Qwen3-14B-W8A8-gptq512                    16G    compressed-tensors int-quantized   W8A8      no (8-bit weights, genuine I8)
  Qwen3.6-27B-W4A16                         25G    compressed-tensors pack-quantized  W4A16     no (already packed)
  Qwen3.6-27B-W4A8-rtn                      27G    compressed-tensors int-quantized   W4A8      already prepacked below (skip)
  Qwen3.6-27B-W4A8-rtn-prepacked           15G    compressed-tensors int-quantized*  W4A8      DONE PREVIOUSLY (fit-critical 27B)
  Qwen3.6-27B-W8A8-INT8-RTNtest             33G    compressed-tensors int-quantized   W8A8      no (8-bit weights, genuine I8)
  Intel_Qwen3.6-35B-A3B-int4-AutoRound      21G    auto-round          -               int4      no (auto-round, not comp-tensors)
  Lorbus_Qwen3.6-27B-int4-AutoRound         18G    auto-round          -               int4      no (auto-round, not comp-tensors)
  Qwen_Qwen3.6-27B-FP8                      29G    fp8                 -               fp8       no (fp8, not comp-tensors)
  Qwen_Qwen3.6-27B                          72G    none                -               bf16      no (full precision)
  Qwen_Qwen3-0.6B                          1.5G    none                -               bf16      no (full precision)
  google_gemma-4-12B-it                     23G    none                -               bf16      no (full precision)
  bartowski_Qwen2.5-7B-Instruct-GGUF       4.4G    -                   gguf            -         no (gguf)
  unsloth_Qwen3.6-27B-GGUF                  16G    -                   gguf            -         no (gguf)
archive/  (parked dups -- not served per CLAUDE.md)
  Qwen3-14B-W4A16                          9.3G    compressed-tensors pack-quantized  W4A16     no (already packed)
  Qwen3-14B-W4A8-INT                        16G    compressed-tensors int-quantized   W4A8      int-quantized-W4A8 but NOT prepacked (archived dup)
  Qwen3-14B-W8A8-INT8                       16G    compressed-tensors int-quantized   W8A8      no (8-bit weights, genuine I8)
```
\* `is_prepacked_w4a8=true`; the 280 quantized weights are now I32 (packed int4 [out, in/8]).


## Which need / benefit from prepacking

Prepack candidates (format=int-quantized AND 4-bit weights = W4A8):

1. **Qwen3-14B-W4A8-gptq** -- BENEFICIAL (not fit-critical).
   14B at I8 is a ~15.4 GiB weight payload; on-load the unpacked-I8 transient + packed int4
   peaks well under 32 GiB (est. ~16-18 GiB incl. activations), so it FITS the B70 either way.
   Prepacking still wins: disk 16G -> 9.3G, faster load, and removes the on-load pack step.
   -> PREPACKED in this audit. No `-prepacked` existed before.

2. **Qwen3.6-27B-W4A8-rtn** -- FIT-CRITICAL. The 27B unpacked-I8 load peak (>~28-30 GiB I8 +
   packed int4 held together) OOMs/hangs the 32 GiB card. Already solved:
   `Qwen3.6-27B-W4A8-rtn-prepacked` exists (27G -> 15G, is_prepacked_w4a8=true).
   -> SKIPPED (already done; instructed not to re-prepack the 27B).

3. **archive/Qwen3-14B-W4A8-INT** -- int-quantized W4A8, WOULD be a candidate, but it is an
   ARCHIVED, parked dup (CLAUDE.md: archive models are not served). Prepacking it would only
   spend disk on something that will never be served. -> NOT prepacked (archived dup).

NOT candidates (and why):
- W4A16 / W8A16 (`pack-quantized`): weights already packed by compressed-tensors. No prepack.
- W8A8 (`int-quantized`, 8-bit weights): genuinely int8, NOT 4-bit. No int4 pack exists/needed.
  (Qwen3-14B-W8A8-gptq, -gptq512, Qwen3.6-27B-W8A8-INT8-RTNtest, archive/Qwen3-14B-W8A8-INT8.)
- auto-round / fp8 / gguf / bf16: not compressed-tensors int-quantized.


## What was prepacked in this audit

One model, run on CPU (non-GPU container, no `--device /dev/dri`), 121 GiB RAM free:

```
docker run --rm \
  -v /mnt/vm_8tb/b70/models:/models \
  -v /mnt/vm_8tb/b70/offline_prepack_w4a8.py:/prepack.py \
  -e SRC=/models/Qwen3-14B-W4A8-gptq \
  -e DST=/models/Qwen3-14B-W4A8-gptq-prepacked \
  --entrypoint python vllm-xpu-env:int8g /prepack.py
```

Script output:
```
[prepack] tensors=723 quantized_weights_to_pack=280
[prepack] packed 280 weights | bytes 15.4 -> 9.2 GiB
[prepack] DONE -> /models/Qwen3-14B-W4A8-gptq-prepacked (is_prepacked_w4a8=True)
```

Verification:
- du: `Qwen3-14B-W4A8-gptq` 16G -> `Qwen3-14B-W4A8-gptq-prepacked` 9.3G (quantized part halved).
- safetensors header dtypes after pack: BF16: 163, I32: 280 (was BF16: 163, I8: 280).
- `quantization_config.is_prepacked_w4a8` = true.


## Summary

- Models prepacked in this audit: 1 (`Qwen3-14B-W4A8-gptq` -> `-prepacked`, BENEFICIAL).
- Fit-critical W4A8 already prepacked previously: `Qwen3.6-27B-W4A8-rtn-prepacked` (the 27B).
- int-quantized-W4A8 NOT prepacked: `archive/Qwen3-14B-W4A8-INT` (archived/parked dup, never
  served -> intentionally skipped).
