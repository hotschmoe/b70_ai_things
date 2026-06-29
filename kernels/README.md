# kernels/ -- shared custom-kernel SOURCE (built per-backend)

The custom oneDNN int8/int4 gemm ops are **one shared source**, compiled **separately per
serving backend** because the prebuilt `.so` is ABI-locked to the torch it was built against
(`torch::Library::_def` fails across versions).

## What's here

- `int8_gemm_kernel.patch` -- adds the fused int8 activation gemm types (`f16_int8`,
  `bf16_int8`) to `vllm-xpu-kernels`' `csrc/xpu/onednn/onednn_ext.h`.
- `int8_gemm_w8a16.h` -- decode op (int8 weight, fp16 activation).
- `int8_gemm_w8a8.h` -- prefill op (int8 weight, int8 activation).
- (int4 gemm ops `int4_gemm_w4a8` / `int4_gemm_w4a16` are upstream in `vllm-xpu-kernels`
  itself, gated by `XPU_SPECIFIC_KERNELS_ENABLED=ON`; no repo patch needed.)

## How it builds (per backend)

The same source compiles against each backend's torch into ABI-specific binaries. The built
`.so`s are git-ignored runtime artifacts under `/mnt/vm_8tb/b70`, NOT repo content.

- **vLLM**: built into the `vllm-xpu-env:int8g` image (`XPUInt8ScaledMMLinearKernel`).
  Build context: `vllm/images/int8g/`.
- **sglang**: built against sglang's torch 2.12 into a runtime `_xpu_C.abi3.so`
  (`/mnt/vm_8tb/b70/w8a8_kernel/_xpu_C.abi3.so`) via `build_xpu_c.sh`, loaded at serve time
  with `B70_XPU_C_SO=...` + the `sglang/patches/w8a8_shim.py` shim that routes int8 linears
  to the fused ops. See `research/w8a8/W8A8_BUILD.md`.

## Rule

Edit the kernel op once HERE. A change means a rebuild on BOTH backends and a re-bench of
every affected `(model, quant, backend)` -- log it in `research/LESSONS.md`.
