# vllm_wna16_xpu -- last-resort XPU dequant fallback for compressed-tensors WNA16 (FUTURE / not wired in)

`xpu_wna16_dequant.py` is a drop-in `MPLinearKernel` that dequantizes int4 (uint4/uint4b8) -> bf16/fp16 at
load and runs a dense GEMM at apply. It accepts ANY input dim (no %32 / no group-divisibility constraint),
unlike the stock `XPUwNa16` kernel. Register it as the last XPU entry in
`vllm/model_executor/kernels/linear/__init__.py` `_POSSIBLE_KERNELS[PlatformEnum.XPU]` (and the scheme's
`assert input_size_per_partition % group_size == 0` in `compressed_tensors_wNa16.py` would need to be
relaxed to ceil-groups for genuinely-odd-K layers).

WHY IT IS NOT USED for the 27B W4A16 (rdy_to_serve/qwen36-27b-w4a8... -w4a16): that checkpoint is
TEXT-ONLY (architectures Qwen3_5ForCausalLM, zero vision tensors) -- there are no vision weights to
dequant at all. The real fix there is the arch shim (load the text-only class, never build the vision
tower). Keep THIS for a FUTURE compressed-tensors model that genuinely quantizes odd-dim layers and
ships their weights (then it is the right tool). Verified to LOAD past the kernel chooser on v0230 XPU;
correctness of the dequant math (int4 unpack + group scales) is UNVERIFIED on a real workload.
