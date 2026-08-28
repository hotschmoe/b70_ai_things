# Qwen3.8 NVFP4 SGLang TP2 single-stream status - 2026-08-28

## Result

The RadixArk mixed ModelOpt checkpoint now runs on the refreshed SGLang XPU
stack at TP=2. The current winner keeps its NVFP4 MLP and lm_head weights in
packed E2M1 form, uses the source-built Torch 2.13 XPU NVFP4 operator, and
routes the checkpoint's FP8 attention and Gated DeltaNet projections through
the XPU W8A16 operator only for M=1 decode.

| Route | Median post-first decode | Median including TTFT | Request |
| --- | ---: | ---: | --- |
| Stock static-FP8 projections | 30.1665 tok/s | 27.6079 tok/s | c1, p879/o512 |
| FP8 weight-only W8A16 projections | 32.6206 tok/s | 29.7873 tok/s | c1, p879/o512 |

The matched gain is 8.14%. The 40 tok/s objective is not met, so this remains
a research winner rather than a live-shelf promotion.

## Exact stack and identity

- Host kernel: 7.1.0-070100.
- Image: `b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`.
- Torch: 2.13.0+xpu.
- SGLang: `bede6bc`.
- sgl-kernel-xpu: `2d10888`.
- Compute Runtime: 26.22.
- XPU operator SHA256: `96e33b4e66f4eba6a2108c5a4f3aef5fba505f3696ba876e60b6ddeb08a87549`.
- GDN sidecar SHA256: `323547ed36f4821ccba6fbbc75ced8fd6e9837e268891d6488d62825002279a8`.
- Local RadixArk cache revision: `554ebba9b5f1b79dc11246341960360e6ef05ef4`.
- Winning served ID: `qwen3.8-27b-NVFP4-radixark-sglang-w8a16-full-tp2`.

Both accepted arms used bf16 KV cache, Triton linear attention, FULL decode
capture at batch size one, prefill graph disabled, chunked prefill 128,
maximum one running request, P2P disabled, pidfd IPC, SYCL collective kernels,
and text-only runtime configuration.

## Operator accounting and selection

Per rank and target token, the model issues 129 NVFP4 calls, 128 FP8 calls,
48 tiny bf16 linear calls, 129 all-reduces, and one logits all-gather. The
compulsory weight and scale stream is approximately 8.197 GiB per token per
rank. At 32.6206 tok/s this corresponds to about 287 GB/s of useful weight
traffic; 40 tok/s would require about 352 GB/s before other traffic.

The stock FP8 path quantizes the activation and then calls
`torch._scaled_mm` for every FP8 projection. The W8A16 route removes 128
activation-quant kernels per target token. Exact real-weight M=1 oracles found
the XPU W8A16 GEMM faster on all three fused decode shapes, with cosine at
least 0.9999965 and relative L2 at most 0.00264 against dequantized-weight
references. FULL XPUGraph replay was bit-identical to eager execution.

## Rejected controls

- A source-built ESIMD NVFP4 M=1 GEMV was correct and deterministic but was
  3.6x to 4.8x slower than the current oneDNN NVFP4 operator on gate/up, down,
  and lm_head shapes.
- The current native Intel XPU Gated DeltaNet backend returned two different
  greedy byte sequences for the same request. It was rejected before timing.
- A current-stack oneCCL P2P A/B passed bit-exact direct and graph oracles, but
  P2P enabled measured 0.3650 ms per graph iteration versus 0.3512 ms with P2P
  disabled. No model arm was attempted.
- vLLM 0.28 FULL capture loaded and compiled the model but stalled at the
  graph/collective handoff. The SGLang route is the current TP2 execution path.

## Evidence

- Stock result JSON SHA256: `618e99288361be4dfa88119bc2ef4a71bac52fca1a3c38d1f31a9c2dddc7bece`.
- W8A16 result JSON SHA256: `71b18391e8fe545b52c8f16a640fcb93888a88e32e16ebaa208ab856e8853a99`.
- Stock runtime log SHA256: `e09f3995fcc289b8d98c7280b095d4e462342a07a972e3d280e49818932dc217`.
- W8A16 runtime log SHA256: `6cbb837e8c9e8b0fda5107e12ddb3800e688ee524f43d41ff258abfaaba1829d`.
- P2P-off oracle JSON SHA256: `f1bdeb63163b46e9aea1a59573ea65f9a22379ca46cbfd39190cdb704f5fca40`.
- P2P-on oracle JSON SHA256: `a5fb6015c699ea5e9ece783bf8cbf18f1feb580dff013f6d8169d0ce79d1849b`.

The W8A16 runtime log contains a post-warmup `/freeze_gc` connection-refused
traceback caused by SGLang calling its own endpoint before it was reachable.
The endpoint subsequently opened, passed identity and coherence gates, served
all benchmark requests, and stopped normally. Host logs show no OOM, reset,
hang, fault, or reboot. Final per-card and compiled P2P-off collective health
passed.
