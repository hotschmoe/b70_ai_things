# 02 - XPU INT8 W8A8 kernel: STATUS + how to use (2026-06-18)

What we built this session: the **first working INT8 W8A8 inference path on Intel Arc Pro B70
(Battlemage / Xe2) in vLLM**, plus a fused activation-quant kernel that makes it competitive with
FP8. Stock vLLM `KeyError: PlatformEnum.XPU`-crashes loading a W8A8-INT8 checkpoint on XPU; our
kernel makes it load, run, and beat FP8 ~1.6x in prefill.

See the design in [01_int8_w8a8_blueprint.md](01_int8_w8a8_blueprint.md); the why/hardware in
[../literature/06_xpu_kernel_fastpaths.md](../literature/06_xpu_kernel_fastpaths.md).

## What it is

| Piece | Where | What |
|---|---|---|
| `int8_gemm_w8a8` | vllm-xpu-kernels `csrc/xpu/onednn/int8_gemm_w8a8.h` (+ `onednn_ext.h` `s8_s8` joint dtype, `onednn_matmul.cpp`, `torch_bindings.cpp`, `ops.h`) | oneDNN s8 x s8 -> s32 GEMM, per-token act scale + per-channel weight scale -> f16/bf16. Symmetric. |
| `dynamic_per_token_int8_quant` | vllm-xpu-kernels `csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp` | Fused SYCL per-token int8 quant (1 work-group/row, sub-group absmax reduction). Replaces the slow `@torch.compile` `_ref`. |
| `XPUInt8ScaledMMLinearKernel` | vLLM `model_executor/kernels/linear/scaled_mm/xpu_int8.py` (= `contrib/vllm_int8_xpu/xpu_int8_kernel.py`) | The linear kernel: per-token dynamic-symmetric int8 quant (fused, w/ `_ref` fallback) -> int8_gemm_w8a8. |
| registry + chooser fix | vLLM `model_executor/kernels/linear/__init__.py` | `_POSSIBLE_INT8_KERNELS[XPU] = [XPUInt8ScaledMMLinearKernel]` + harden `possible_kernels[...]` -> `.get()` (also fixes the GDN-FP8 KeyError family). See `contrib/vllm_int8_xpu/registry_patch.md`. |

Scope: **dynamic per-token symmetric int8 activations + per-channel/per-tensor int8 weights** (the
common compressed-tensors/Quark W8A8 case). Static + asymmetric (AZP) schemes are rejected in
`can_implement` (a phase-2 extension; lower value since dynamic-symmetric dominates).

## Results (Qwen3-14B-W8A8-INT8, single B70, vLLM v0230)

| Metric | INT8 W8A8 (ours) | FP8 | Note |
|---|---|---|---|
| Loads / serves | **YES** (was KeyError-crash on stock) | yes | `Selected XPUInt8ScaledMMLinearKernel` |
| Prefill @4096 | **6353 tok/s** | 3997 | **1.59x** (native s8s8s32 vs FP8's conversion path) |
| Prefill @2048x8, @4096x8 | 14648 / 10888 | 9176 / 6802 | **1.60x** |
| TTFT @512 | 142 ms | 158 ms | better |
| Decode (batch-1) | **22.6 tok/s** | ~29 | ~78% of FP8 (was 13 before fused quant = 1.7x gain) |
| Numerical | gemm err 2.4e-4; quant q-match 100% | - | fp16-accurate |

Bottom line: **INT8 W8A8 is the prefill/throughput champion on B70 (1.6x FP8), and now nearly
matches FP8 in decode.** Remaining decode gap (22.6 vs 29) = the M=1 oneDNN int8 GEMM single-row
path -- diminishing returns; vectorizing the quant K-loop / an M=1 GEMM path are optional follow-ups.

## How to use it

**Reusable image** `vllm-xpu-env:int8` bakes in everything (the .so + class + registry patch).
Built by `scripts/47_build_int8_image.sh`. Serve any compressed-tensors W8A8-INT8 checkpoint with a
plain `vllm serve` (no graft/patch):

```bash
docker run -d --name vllm_int8 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p 18080:18080 -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 \
  -e ZE_AFFINITY_MASK=0 --entrypoint vllm vllm-xpu-env:int8 \
  serve /mnt/vm_8tb/b70/models/Qwen3-14B-W8A8-INT8 --served-model-name qwen3-14b-w8a8 \
  --host 0.0.0.0 --port 18080 --dtype float16 --enforce-eager --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --trust-remote-code
```

Verify it engaged: `docker logs vllm_int8 | grep "Selected XPUInt8ScaledMMLinearKernel"`.

**Rebuild from source** (e.g. after editing the kernel): `scripts/44_build_int8_kernel.sh` builds the
`_xpu_C` extension only (minimal-target profile -> minutes, not the 1-2h full build). The repo lives
on the box at `/mnt/vm_8tb/b70/vllm-xpu-kernels` (forked head 11f42aa + our edits). The vLLM Python
patch is applied at image-bake time by `contrib/vllm_int8_xpu/apply_patches.py` (resolves the real
vLLM dir via `import vllm`).

## Scripts

- `40_quantize_w8a8.sh` / `43_quantize_w4a8.sh` -- make W8A8 / W4A8 checkpoints (data-free RTN, CPU).
- `44_build_int8_kernel.sh` -- build the `_xpu_C` extension (minimal target, ccache).
- `45_patch_serve_int8.sh` -- graft + patch + serve (ephemeral; for dev iteration).
- `46_bench_prefill.sh` -- prefill/large-batch + decode bench.
- `47_build_int8_image.sh` -- bake + commit `vllm-xpu-env:int8` + serve-verify.

## Open follow-ups (not done)

- **Upstream (task #13):** PR 1 to vllm-project/vllm-xpu-kernels (int8_gemm_w8a8 + s8_s8 joint dtype +
  fused quant); PR 2 to vllm-project/vllm (XPUInt8ScaledMMLinearKernel + registry + `.get()` hardening,
  which is standalone-worthy). RFC #37979 currently omits INT8 W8A8 for XPU.
- **make-it-right (task #12):** asymmetric/AZP + static-scale schemes (needs an asym checkpoint; lower value).
- **make-it-faster:** vectorize the quant K-loop; an M=1 int8 GEMM fast path to close the last decode gap.
