# Building a patched libtorch_xpu.so (the submit_without_event NEO-abort fix)

Goal: rebuild ONLY `libtorch_xpu.so` from pytorch source with the `submit_without_event`
patch (docs/upstream_pytorch_xpugraph_replay_leak.md), ABI-matched to the prebuilt
torch 2.12.0+xpu in `vllm-xpu-env:int8g-v0240`, and drop it in. No IPEX involved.

Build tree (git-ignored, runtime): `/mnt/vm_8tb/b70/torch_xpu_build/`
- `pytorch/` -- source @ 7661cd9 (exact torch 2.12.0+xpu SHA) + submodules, patch applied.
- `build-torch-xpu/` -- cmake build dir (output `lib/libtorch_xpu.so`).
- `build_torch_xpu.sh` -- the build script (runs INSIDE a container off the image).

## Key gotchas (learned 2026-07-07)
1. **setvars.sh `return`s at `bash -c` top level** (acts like exit) -> capture its env from a
   subshell (`bash -c 'source setvars.sh; env'`) and re-export, don't source inline.
2. **Image cmake is 4.3.4, too new** -- it removed `cmake_minimum_required(<3.5)` compat (breaks
   pytorch 2.12's FP16/NNPACK/XNNPACK/qnnpack subdirs) and the single-arg `FetchContent_Populate`
   torch-xpu-ops uses. Fix: `pip install "cmake==3.31.*"` in the build container.
3. **ABI match**: build with `_GLIBCXX_USE_CXX11_ABI=1`, gcc-13/g++-13 (host), oneAPI 2025.3 (SYCL),
   Release, at the EXACT SHA so XPUGraphImpl layout is identical -> the .so drops in.
4. **Scope reduction (ABI-safe)**: `USE_XNNPACK/QNNPACK/NNPACK/FBGEMM/FLASH_ATTENTION=OFF` --
   these are CPU/CUDA-only; libtorch_xpu.so does not reference their symbols, so disabling them
   speeds the build without changing the .so's NEEDED/exported deps. `torch_xpu` still forces
   building `torch_cpu` (link dep) + torch-xpu-ops + ATen codegen -- that is the bulk of the time.

## Run
```
docker run -d --name torch_xpu_build -v /mnt/vm_8tb/b70/torch_xpu_build:/work \
  --entrypoint bash vllm-xpu-env:int8g-v0240 -c '/work/build_torch_xpu.sh 2>&1 | tee /work/build.log'
docker logs -f torch_xpu_build     # watch; output = build-torch-xpu/lib/libtorch_xpu.so
```

## Drop-in + validate
- Preserve original, compare ELF (`readelf -d` SONAME/NEEDED/RPATH), normalize RPATH with patchelf.
- Mount the patched .so over `/opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_xpu.so`
  in the serve container (a `-v` bind, like the GDN .so overlay), then serve NVFP4 TP=2 captured+MTP
  (NO drafter-eager) and soak past 40k tokens. SUCCESS = no abort AND decode stays ~43 t/s flat
  (the 43->20 degradation is gone).

The 4-line patch: `vllm/patches/xpugraph_submit_without_event.patch`.
