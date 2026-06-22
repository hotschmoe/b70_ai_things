# Dense W8A8 int8 on v0230 -- TEST (task #12)

Makes a DENSE `CompressedTensorsW8A8Int8` model (e.g. Qwen3-14B-W8A8-autoround) use the task-c
`XPUInt8TritonScaledMMLinearKernel` (true int8 -> DPAS/XMX via triton_scaled_mm) on `vllm-xpu-env:v0230`.
v0230 stock has NO XPU entry in `_POSSIBLE_INT8_KERNELS` -> dense W8A8 fails/dequants.

`sitecustomize.py` wraps `init_int8_linear_kernel` (called by BOTH Quark and CompressedTensors W8A8 at
model-load) to run the quark.py `_b70_register_xpu_int8_kernel()` first, so the dense path finds the kernel.

Serve (GPU1, outside the gpu-run lease while GPU0 is busy): mount the task-c quark.py over the vllm quark.py
path (so the register fn is importable) + `/b70site` on PYTHONPATH (the sitecustomize), `B70_INT8_LINEAR=triton`,
`ZE_AFFINITY_MASK=1`, port 18081. SUCCESS = log `Selected XPUInt8TritonScaledMMLinearKernel for CompressedTensorsW8A8Int8`.
