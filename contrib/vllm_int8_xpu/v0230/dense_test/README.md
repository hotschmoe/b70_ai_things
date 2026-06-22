# Dense W8A8 int8 on v0230 -- TEST (task #12)

Makes a DENSE `CompressedTensorsW8A8Int8` model (e.g. Qwen3-14B-W8A8-autoround) use the task-c
`XPUInt8TritonScaledMMLinearKernel` (true int8 -> DPAS/XMX via triton_scaled_mm) on `vllm-xpu-env:v0230`.
v0230 stock has NO XPU entry in `_POSSIBLE_INT8_KERNELS` -> dense W8A8 fails/dequants.

`sitecustomize.py` wraps `init_int8_linear_kernel` (called by BOTH Quark and CompressedTensors W8A8 at
model-load) to run the quark.py `_b70_register_xpu_int8_kernel()` first, so the dense path finds the kernel.

Serve (GPU1, outside the gpu-run lease while GPU0 is busy): mount the task-c quark.py over the vllm quark.py
path (so the register fn is importable) + `/b70site` on PYTHONPATH (the sitecustomize), `B70_INT8_LINEAR=triton`,
`ZE_AFFINITY_MASK=1`, port 18081. SUCCESS = log `Selected XPUInt8TritonScaledMMLinearKernel for CompressedTensorsW8A8Int8`.

## RESULT 2026-06-23 -- SUCCESS (tested on GPU1, in parallel with the GPU0 autoround eval)
Served Qwen3-14B-W8A8-autoround (compressed-tensors int-quantized) on `vllm-xpu-env:v0230`, GPU1:18081,
`B70_INT8_LINEAR=triton`, the sitecustomize hook + the task-c quark.py mounted. Logs:
```
[quark.py:98] b70: registered XPU int8 Triton scaled-mm kernel (DPAS int8 fastpath)
[b70 int8-dense TEST] registered XPU int8 triton kernel before init_int8_linear_kernel
Selected XPUInt8TritonScaledMMLinearKernel for CompressedTensorsW8A8Int8     <-- dense W8A8 picks the TRUE int8 kernel
Model loading took 15.34 GiB     <-- int8 weights stay int8 (NOT dequant-to-bf16 ~28GB)
Triton kernel JIT compilation: scaled_mm_kernel   <-- the int8->int32 DPAS GEMM runs
```
Generation = COHERENT (correct iterative fib + main()). **v0230 CAN serve dense W8A8 true-int8 -> resolves the
v0.23-vs-:int8 tension.** PRODUCTIONIZE: bake the hook into the image (or a clean mounted patch, like the quark serve),
register for compressed-tensors + quark. PERF (TODO): eager triton is ~25% slower than dequant (per-forward act-quant);
the meaningful test is WITH graph capture (act-quant fuses into the graph) vs the :int8 oneDNN kernel.
