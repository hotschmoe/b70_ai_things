# AWQ-4bit checkpoint of Qwen3.6-27B for SGLang Intel-XPU (vision-retaining)

VERIFIED recipe (2026-06-27, agent research corroborated against image source + on-disk configs).
WHY: `awq_dequantize` is the ONLY registered XPU quant GEMM (see PERF.md). AWQ W4A16 (dequant-4bit->fp16,
then torch.matmul) is the only way to cut the bf16 decode-bandwidth bottleneck on this box. 4-bit fits one
card -> TP=1 -> no all-reduce -> DP=2 possible.

## GO/NO-GO: GO. Tool = Intel AutoRound exporting format="auto_awq".
- AutoAWQ pkg: NO-GO (no qwen3_5 arch entry; CUDA-oriented).
- llm-compressor AWQModifier: NO-GO (only emits compressed-tensors -> WNA16/Marlin -> dead on XPU).
- AutoRound `pip install auto-round`: runs on XPU OR CPU (no CUDA), proven on qwen3_5 in this repo
  (Lorbus int4 was AutoRound), and its AWQ exporter emits the exact AutoAWQ gemm layout the XPU kernel reads.

## sglang XPU AWQ loader requirements (config.json -> quantization_config)
```json
{ "quant_method":"awq", "bits":4, "group_size":128, "version":"gemm",
  "zero_point":true,
  "modules_to_not_convert":["linear_attn","visual","lm_head","mtp","embed_tokens"] }
```
(top-level `torch_dtype:"float16"`; serve `--quantization awq --dtype float16`.)
Per-layer tensors (gemm layout; gemv NOT accepted):
  qweight int32 [in, out//8]  (packed along out, factor 8, AWQ lane interleave [0,4,1,5,2,6,3,7])
  qzeros  int32 [in//128, out//8]
  scales  fp16  [in//128, out]
Constraints (all satisfied by this model): in%128==0, out%8==0.

## THE CRITICAL GOTCHA (silent NaN if wrong)
`modules_to_not_convert` is **substring-matched** (`module_name in prefix`). Quant tools serialize the
RESOLVED exclusion as ~330 ENUMERATED names like `model.language_model.layers.0.linear_attn.in_proj_a`.
At serve time sglang prefixes are `model.layers.N.linear_attn.*` (VLM load rewrites language_model.->model.).
Enumerated names are NOT substrings -> match nothing -> GDN gets quantized -> garbage/NaN. So POST-EXPORT
you MUST overwrite the list with the 5 short stems above. (QUANTS_TODO.md:320-330.)

## Module map (what stays fp16)
- GDN linear-attn (all under `linear_attn.` prefix): in_proj_qkvz/ba, out_proj, conv1d (already quant=None),
  norm, A_log, dt_bias  -> "linear_attn" stem. (correctness: GDN must stay fp16/bf16.)
- Vision tower (all under `visual.` prefix): blocks/attn/mlp/merger/patch_embed -> "visual" stem. (vision.)
- lm_head, embed_tokens (defensive), mtp (head dropped at load).
- Quantized = the dense self_attn (qkv_proj,o_proj) + MLP (gate_up_proj,down_proj) of the 64 decoder layers.

## fp16 / GDN RISK (#1 unknown -- MUST validate)
AWQ pins act dtype fp16. Every prior GDN validation here was fp32-SSM + bf16. mamba_ssm_dtype stays float32
(SSM recurrence fp32) but in/out projections + conv now see fp16. After producing ANY awq ckpt, run
contrib/gdn_nan_repro/ under --dtype float16 (esp. mixed prefill+decode) BEFORE trusting output.

## Path 1 -- FAST repack (CPU, ~30min): text-only W4A16 -> AWQ (de-risk serve path + fp16/GDN)
The on-disk Qwen3.6-27B-W4A16 is compressed-tensors pack-quantized, SYMMETRIC int4, g128, NO zero-point.
Transcode is numerically EXACT: CT stores u=q_signed+8 (offset 8) unsigned nibbles; AWQ dequant w=(u-z)*scale;
set z=8 -> w=q_signed*scale, bit-identical. Per quantized layer:
  1. u = unpack_int32(weight_packed,4)  (unsigned nibble [out,in], = q_signed+8; do NOT subtract offset)
  2. transpose -> [in,out]; AWQ-gemm-pack -> qweight [in,out//8] int32 (interleave [0,4,1,5,2,6,3,7])
  3. qzeros = const 8 packed -> [in//128, out//8]
  4. scales = weight_scale.t().fp16 -> [in//128, out]
  config zero_point=false (symmetric). VISION ABSENT (source is text-only) -> smoke test only.

## Path 2 -- PRODUCTION vision-retaining (GPU job, ~1-2h with algorithm=awq, else 4-10h)
AutoRoundMLLM on the FULL VLM (AutoModelForImageTextToText) -> vision retained; W4A16 scheme; format="auto_awq".
ignore regex for layer_config: `(visual|\.mtp|mtp\.|linear_attn)` + lm_head. nsamples=128-256 seqlen=1024.
to_quant_block_names=["model.language_model.layers"]. Adapt scripts/_autoround_w8a8.py + scripts/59_autoround_2xpu.sh.
POST-EXPORT FIXES (mandatory): (1) overwrite modules_to_not_convert -> 5 stems; (2) graft VLM wrapper config
(architectures Qwen3_5ForConditionalGeneration, model_type qwen3_5, text_config+vision_config, torch_dtype float16).
Patterns: w4a8/fix_27b_vlm_config.py, scripts/87_fix_autoround_vlm_config.py.

## Validation gate (do not skip)
1. At load, confirm GDN/vision Linears -> UnquantizedLinearMethod (no qweight under linear_attn.*/visual.*).
2. gdn_nan_repro under --dtype float16 incl. mixed prefill+decode.
3. bench2048.sh: is 4-bit AWQ decode actually faster than bf16 9.0 t/s? (the whole point.)
