# vllm-xpu-env:int8g -- our INT8 W8A8 kernel image + graph-capture-traceable custom ops

THE real low-precision compute image for the B70 (Xe2 has no native FP8). Used by the int8-kernel
shelf models: `rdy_to_serve/qwen3-14b-w8a8`, `qwen3-14b-w4a8`, `qwen36-27b-w4a8`.

## Lineage + digests (recorded 2026-06-23)
```
  vllm-xpu-env:v0230   sha256:04e26c1c7f89...   base, vLLM 0.23.0+xpu
    -> :int8           sha256:77ae629c04ad...   + oneDNN INT8 W8A8 GEMM .so (contrib/vllm_int8_xpu) baked,
                                                 registered as XPUInt8ScaledMMLinearKernel in
                                                 _POSSIBLE_INT8_KERNELS[XPU] (apply_patches.py). scripts/47.
    -> :int8g          sha256:8e25c7582871...   + register_fake on _xpu_C.int8_gemm_w8a8 /
                                                 dynamic_per_token_int8_quant so VLLM_XPU_ENABLE_XPU_GRAPH=1
                                                 (PIECEWISE capture) can trace them. THIS dir's build.sh.
```
Op check at bake: `int8_gemm: True  fused_quant: True`.

## (Re)build
```bash
ssh root@192.168.10.5 && cd /mnt/vm_8tb/b70
DATE=20260623 bash /mnt/vm_8tb/b70/images/int8g/build.sh    # or sync this dir + run build.sh
```
`build.sh` bakes a DATED immutable tag (`:int8g-YYYYMMDD`) and moves the convenience tag `:int8g` to it
(ORGANIZATION.md: tags immutable; record the new digest here). It needs `:int8` present
(`scripts/47_build_int8_image.sh` rebuilds that from `:v0230` if it is ever lost).

## What is and isn't pinned (honest status)
- TRACKED + rebuildable: the bake (build.sh) + the Python kernel sources (`contrib/vllm_int8_xpu/`,
  host `contrib_int8/xpu_int8.py`).
- NOT yet a pure Dockerfile chain: `:v0230` and `:int8` are still `docker commit`-built, and the compiled
  `_xpu_C.abi3.so` (with `int8_gemm_w8a8`) is a host binary from `vllm-xpu-kernels/` (built via its own
  `Dockerfile.xpu`). Full Dockerfile-ization of the chain is the ORGANIZATION.md follow-up. For now, the
  recorded digests above let you VERIFY you are on the right image even though a rebuild is multi-step.

## GDN note (for the Qwen3.6-27B W4A8 model)
The `_xpu_C.abi3.so` baked here ships `GDN_KERNELS_ENABLED=OFF` -> no `gdn_attention` (fine for dense 14B;
the 27B uses gated-delta-net and needs it). The 27B serve mounts the GDN-enabled kernel `.so` (+ sibling
`libgdn_attn_kernels_xe_2.so`) from `vllm-xpu-kernels/` over the baked one at run time -- see
`rdy_to_serve/qwen36-27b-w4a8/serve.sh`.
