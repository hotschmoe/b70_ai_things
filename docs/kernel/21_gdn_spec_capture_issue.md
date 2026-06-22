# 21 -- DRAFT upstream issue: `_xpu_C.gdn_attention` rejects spec_query_start_loc under FULL_DECODE_ONLY capture

Ready-to-file bug report for **vllm-project/vllm-xpu-kernels** (or vllm with the `xpu` label). Filing this is the only
path to unblock FULL-capture MTP on Battlemage (it would lift single-card MTP past the current PIECEWISE 1.79x ceiling).
The exact assert string is untracked upstream as of 2026-06-22.

---

## Title
[XPU] `gdn_attention` (Gated-DeltaNet) asserts `spec_query_start_loc must have size [num_spec_decodes + 1]` during FULL_DECODE_ONLY cudagraph capture with MTP speculative decoding

## Environment
- Image: `vllm-xpu-env:v0230` = vLLM **0.23.0** + **vllm_xpu_kernels 0.1.9**, torch 2.11.0+xpu
- HW: 1x Intel Arc Pro B70 (Battlemage / Xe2, 32 GB)
- Model: Qwen3.6-27B int4-AutoRound (qwen3_5, Gated-DeltaNet hybrid + MTP head), `quantization=inc`

## What works
- `cudagraph_mode=PIECEWISE` + MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`) WORKS and is
  a **1.79x** single-card decode speedup (55.28 vs 30.84 t/s). So MTP wiring + the GDN spec path are correct in eager/PIECEWISE.

## What fails
- `cudagraph_mode=FULL_DECODE_ONLY` (+ `--attention-backend TRITON_ATTN`) + the SAME MTP config crashes during
  decode-FULL graph capture:
```
Capturing CUDA graphs (decode, FULL):   0%|          | 0/3
...
File ".../vllm/_xpu_ops.py", line 151, in _gdn_attention_core_xpu_impl
    torch.ops._xpu_C.gdn_attention( ... )
RuntimeError: spec_query_start_loc must have size [num_spec_decodes + 1]
```
- Capture sizes include `1 + num_spec` (here 6 for spec=5), i.e. the dummy decode batch IS spec-aligned.
- `--attention-backend TRITON_ATTN` does NOT avoid it: the GDN *decode* core always routes through the baked
  `torch.ops.vllm.gdn_attention_core_xpu` -> `_xpu_C.gdn_attention` op regardless of attention backend.

## Bisection (rules out the vLLM Python dispatcher)
The upstream `CudagraphDispatcher._create_padded_batch_descriptor` hard-asserts
`num_tokens_padded % uniform_decode_query_len == 0` (uniform_decode_query_len = 1+num_spec), which can also fail. We
ported the vllm-ascend #7148 dispatcher fix (gate that divisibility instead of asserting). With that patch active, capture
proceeds PAST the dispatcher and reaches the kernel op above -> **the residual assert is inside `_xpu_C.gdn_attention`
itself**, i.e. the kernel builds/validates `spec_query_start_loc` with the wrong expected size during capture-time dummy
runs. Likely the dummy spec metadata constructed for FULL capture doesn't match what the kernel expects
(`num_spec_decodes + 1`), or the kernel's capture-path shape check is too strict.

## Ask
Either (a) relax/fix the `spec_query_start_loc` size handling in `gdn_attention` for the FULL-capture dummy-metadata
path, or (b) document that GDN + spec-decode is PIECEWISE-only on XPU. A minimal repro serve command is available.

## Repro (serve)
```
vllm serve <qwen3.6-27b-int4> --quantization inc --trust-remote-code --max-model-len 8192 --max-num-seqs 8 \
  --attention-backend TRITON_ATTN --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,6,8,16,32]}'
```
