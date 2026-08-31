# Qwen3.8 W8A8 native INT8 research route

This directory preserves the 2026-08-31 vLLM experiment that moved the local
compressed-tensors W8A8 GPTQ checkpoint from Triton INT8 linears to a native
oneDNN/XMX `s8 x s8` GEMM. It is a research control, not a shelf entry.

The native GEMM made FULL decode graph capture possible and raised the bounded
target-only screen to 26.56 tok/s. It did not beat the qualified FP8 W8A16
route. The checkpoint's BF16 MTP layer accepted 0 of 6,076 draft tokens and
must not be used as a serving accelerator.

Profiles:

- `control`: target only, P2P off, eager, 8K.
- `target_full`: target only, direct P2P, FULL decode, measured 237,568-token
  envelope, Triton activation quantization, and native INT8 GEMM.
- `mtp1_diagnostic`: the rejected MTP1 configuration retained only for
  reproduction.

Inspect or start the bounded winning control:

```bash
PROFILE=target_full bash vllm/w8a8/qwen38_mtp1/serve.sh --print-config
bin/gpu-run env PROFILE=target_full \
  bash vllm/w8a8/qwen38_mtp1/serve.sh run
```

Rebuild from the pinned `vllm-xpu-kernels` source and tracked patch:

```bash
bash vllm/w8a8/qwen38_mtp1/build.sh
```

Run the standalone numerical oracle on one leased card. Use
`--require-quant-exact` for the decode-sized byte-exact gate; the default also
allows the observed rare one-LSB differences on larger rows.

```bash
bin/gpu-run --card 0 docker run --rm --device /dev/dri:/dev/dri \
  --group-add render --env ZE_AFFINITY_MASK=0 \
  --env ONEAPI_DEVICE_SELECTOR=level_zero:0 --entrypoint python \
  --volume "$PWD/vllm/w8a8/qwen38_mtp1/bench_int8_ops.py:/bench_int8_ops.py:ro" \
  b70-local/vllm-openai-xpu:qwen38-w8a8-int8-mtp1-r03 \
  /bench_int8_ops.py --dtype float16 --rows 1,4 --require-quant-exact
```

Full evidence and limitations are in
`docs/20260831_qwen38_w8a8_native_int8_result.md`.
