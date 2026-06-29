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

## PERF VERDICT 2026-06-23 -- works, but TOO SLOW; keep :int8 for W8A8 perf
With the opaque-custom-op fix, PIECEWISE graph capture SUCCEEDS (triton-wrap errors 142->0; "Graph capturing
finished 2.47 GiB"; coherent gen). BUT decode = **~1.7 t/s** (TTFT-cancelled, warm) vs **:int8 oneDNN W8A8 ~23.5**
captured -- ~13x SLOWER. Not a JIT-churn artifact (only 1 inference-time JIT, a vLLM internal) and capture is real.
Root cause = the path does **un-fused plain-torch per-token int8 act-quant on EVERY dense linear** (~280 of them;
amax+div+round+clamp+to-int8, XPU lacks `_C.scaled_int8_quant`), AND the kernel is an OPAQUE custom op so inductor
can CAPTURE but cannot FUSE it. The :int8 image has a FUSED int8 act-quant kernel + oneDNN s8s8s32 GEMM -> ~13x faster.
**Conclusion:** the v0230 dense-W8A8 int8 patch is a CORRECTNESS / portability / memory proof (dense W8A8 runs true-int8
on the modern v0.23 stack, weights stay int8, capture-safe) -- a valid FALLBACK if :int8 ever breaks -- but it is NOT a
perf path. We CANNOT drop :int8 for W8A8. To make v0.23 competitive: need (a) a FUSED XPU int8 act-quant op (not torch),
and (b) the int8 GEMM inductor-fusable (not opaque) -- that is real kernel work (RESEARCH_TODO Track 1), a separate project.
