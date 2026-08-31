# Qwen3.8 W8A8 native INT8 result

Date: 2026-08-31

Status: closed as a useful research control, not promoted. The native INT8
GEMM unlocks FULL decode capture and is much faster than the old Triton eager
baseline, but target-only decode remains behind FP8 W8A16 and the checkpoint's
unquantized MTP layer is incompatible with useful speculative serving.

## Native kernel and runtime identity

CONFIG -> Qwen3.8-27B compressed-tensors W8A8 GPTQ, FP16 activations and BF16
KV cache, TP=2 on two B70 cards, vLLM `ac7509e2b`, direct oneCCL P2P where
noted, and a native oneDNN `s8 x s8` GEMM registered ahead of vLLM's
`TritonInt8ScaledMMLinearKernel`.

The runtime image is
`b70-local/vllm-openai-xpu:qwen38-w8a8-int8-mtp1-r03`, immutable ID
`sha256:5dad53a3ef34f6615ab35e7ea984ca6062d48c4c7e3050c07f74d49091948639`.
It was built from `vllm-xpu-kernels` commit
`1e90ffa672ba02f17a909da11838a4c55b199783`, patched tree
`6c944faae2af17ada2123acacfdf540ce43b2255`. The extension SHA256 is
`0c9d19a875089d157f2b30c4e487b9c51792aaebcd6bd649b97d7b15afc4d9dc`.

COMMAND -> build the pinned tree with
`vllm/w8a8/qwen38_mtp1/build.sh`, then run the standalone real-shape oracle
through `bin/gpu-run --card 0`. Test both FP16 decode rows M=1 and M=4 for the
four TP2 projection shapes.

RESULT -> the weight scales were exact, all tested FP16 M=1 activation bytes
were exact against vLLM's Triton quantizer, and GEMM cosine was at least
0.99999988 against the dense reference. With the input dependency barrier
disabled, native quant plus GEMM measured 0.1616 ms for `[1,5120] x
[5120,8704]`, 0.1733 ms for `[1,17408] x [17408,2560]`, 0.08184 ms for
`[1,5120] x [5120,6144]`, and 0.06535 ms for `[1,6144] x [6144,5120]`.

VERDICT -> the local FP16 scale/layout math and decode-sized native quantizer
are numerically sound. This does not imply full-model output identity; that is
qualified separately below.

## FULL target and MTP results

CONFIG -> FULL decode capture, direct P2P, maximum four sequences, 32,768
batched tokens, 0.96 GPU-memory utilization, 237,568 maximum context, FP16
target activations, BF16 KV cache, and no prefix cache. The MTP1 arm used the
checkpoint's unquantized BF16 `model-mtp.safetensors`.

COMMAND -> run the 12-prompt cold realistic suite at temperature zero and
then the 32-request concurrent quality canary. Compare target-only, MTP1, and
the opt-in native activation quantizer. Inspect `/metrics` for speculative
acceptance. Surround each risky TP2 lifecycle with card and compiled
collective health; use `bin/xe-reset` after failed graph initialization.

RESULT ->

| Arm | Scope | First-100 tok/s | Interval tok/s | Full post-TTFT tok/s | Quality |
| --- | --- | ---: | ---: | ---: | --- |
| Native GEMM, MTP1 | complete 12-prompt suite | 22.7345 | 22.5071 | 21.2707 | 24/32 |
| Native GEMM, MTP0 | one-prompt screen | 26.5623 | 26.2967 | 26.2476 | 32/32 |
| Native GEMM and native quant, MTP0 | one-prompt screen | 26.0765 | 25.8157 | 25.7720 | not run |

The complete MTP1 performance artifact is under
`/mnt/vm_8tb/b70/results/w12_qwen38_w8a8_native_int8_mtp1_full/20260831T220900Z/`.
Its performance JSON SHA256 is
`2ca0cb3042bcfc1e0c70f2c5345bd6aa4510ab91b5d4ac456c7c0ccd6b64e831`.
The target-only screen and 32/32 canary are under
`/mnt/vm_8tb/b70/results/w13_qwen38_w8a8_native_int8_target_full_nobarrier/20260831T222600Z/`;
their SHA256 values are
`6d4d8fd82112dacfb4d486421797f0b703bde8212c43e1c722d4e7c44edd3776`
and `8aaf46ab9d9df386b4b50bf259d670020ff8acea4269690b68cc0200b61f7841`.
The native-quant screen SHA256 is
`a0d7970f946347387ac02da3e04bfd73cc181e0935b80ac9254ff28290e3522b`.

RESULT -> MTP drafted 6,076 tokens and accepted zero. It reduced target-only
screen speed by about 14 percent and deterministically failed the arithmetic
case in all eight canary rounds (`54` instead of `60`). With MTP removed, all
32 quality requests passed. The exact 262,144-token MTP1 attempt also failed
the capacity gate: 8.69 GiB of KV was required but only 7.90 GiB was
available, for an estimated 237,952-token maximum. At 237,568, MTP1 fit
exactly; target-only exposed 295,110 KV tokens with Triton activation quant
and 303,330 with native quant.

RESULT -> the matched stock-dispatch FULL arm selected
`TritonInt8ScaledMMLinearKernel` but failed during the first profile run at
Torch FX graph partitioning: `free_symbols()` rejected an empty-arguments
`TreeSpec`. The established stock eager denominator remains 3.53 tok/s. The
custom registered op therefore supplies both a graph boundary and a speed
improvement, even though the overall INT8 route remains slower than FP8.

VERDICT -> retain `target_full` as the best INT8 research profile and remove
MTP1 from any daily-driver default. Do not describe either one-prompt result
as a promotion-grade throughput claim. Native activation quantization did not
win this screen. The next INT8 work should wait for either a matching
quantized MTP artifact or a fused norm-plus-quant/deduplicated activation path.

## Why FP8 W8A16 remains ahead

CONFIG -> compare the W8A8 path with the already qualified FP8 W8A16 route.
The dense W8A8 checkpoint dynamically quantizes activations before about 160
linear projections per generated token. FP8 W8A16 keeps activations in FP16
and stores only weights in FP8, so it avoids those reductions, scale writes,
and quantization launches while retaining one-byte weight traffic.

RESULT -> the local FP8 route measures about 30.90 tok/s target-only FULL and
46.60 tok/s with a useful MTP1. The best W8A8 target-only screen is 26.56
tok/s, and its MTP artifact has zero acceptance. Native quantization alone did
not close the gap.

VERDICT -> FP8 W8A16 remains the daily-driver and qualification focus. The
INT8 kernel work is not wasted: it proved native XMX dispatch, FULL graph
compatibility, real-shape numerical correctness, and the exact remaining
bottlenecks. Resume INT8 only around a fused activation pipeline and a
compatible MTP model, not by tuning the standalone GEMM further.
