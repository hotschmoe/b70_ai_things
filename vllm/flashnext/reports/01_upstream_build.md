# 01 -- upstream vLLM XPU pull and build path

Date: 2026-10-19. Evidence is from the clones, `docker images` / `docker inspect` /
`importlib.metadata` inside existing images (no `/dev/dri`), and a CPU-only
container build. No GPU work was run. Nothing here was served or timed.

Clone root: `/mnt/vm_8tb/b70/build/vllm-flashnext/`. Both clones are full
history (`git rev-parse --is-shallow-repository` = false). vLLM has 463
remote-tracking branches.

## SHAs and versions

| Tree | HEAD | Committer date | Note |
| --- | --- | --- | --- |
| vllm main | `5fd62688ffe995c0c11c8708fe51dbff53235b74` | 2026-10-19T14:37:31-07:00 | `[Docs] Add Gemma-2, Granite SWA, Qwen3-0.6B to batch invariance validated models (#61254)` |
| newest GitHub release | tag `v0.32.0` at `5df63a7203cdb9328d824a27b682e77c7b9928be` | published 2026-10-18T06:50:48Z; commit 2026-10-17T11:57:10-07:00 | Tag `v0.32.1rc0` exists and sorts newer; it is not a GitHub release |
| vllm-xpu-kernels | `19c6ddca8f6ef7a0f1de1bce1bc4dea9505cf8d4` | 2026-10-16T16:39:36+08:00 | `git describe`: `v0.1.16-2-g19c6ddc`. Tag `v0.1.16.1` is `8b95ea665de4c737f99d5aa9b76d9f23c9a0d10f`. Tag `v0.1.15` is `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |

Repo: https://github.com/vllm-project/vllm-xpu-kernels (exists; pushed 2026-10-16).

### What main requires

From `requirements/xpu.txt` and `docker/Dockerfile.xpu` on this HEAD:

- Python 3.12 (`Dockerfile.xpu` ARG, kernels README).
- `torch==2.14.0` from `https://download.pytorch.org/whl/xpu`. The index has
  `torch-2.14.0+xpu-cp312-cp312-manylinux_2_28_x86_64.whl` (also 2.14.1+xpu).
  `torchaudio`, `torchvision`, `torchcodec>=0.14`.
- `triton==3.8.0+xpu` from `https://wheels.vllm.ai/xpu/triton/` (wheel present).
  This is the Intel Triton build, published as package `triton` with a local
  version, not a separate `triton-xpu` project name.
- `vllm_xpu_kernels==0.1.15.4` (PyPI wheel
  `vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl`).
  The git tag series has no `v0.1.15.4`; HEAD is already past `v0.1.16`.
  The vLLM install uses the wheel, not this clone.
- `auto_round_lib==0.16.0`, `numba==0.65.0`.
- Kernels README: oneAPI 2026.1, CMake >= 3.26, Ninja, to build kernels
  from source.
- Dockerfile.xpu pip-pins the runtime (not the host packages):
  `intel-sycl-rt==2026.1.1`, `dpcpp-cpp-rt==2026.1.1`,
  `intel-opencl-rt==2026.1.1`, `oneccl==2022.1.2`, `oneccl-devel==2022.1.2`,
  `intel-openmp==2026.1.1`, `intel-pti==1.1.0`,
  `intel-cmplr-lib-rt/ur` and `intel-cmplr-lic-rt` 2026.1.1.
- Dockerfile.xpu also vendors userspace inside the image: IGC 2.38.2,
  compute-runtime `26.27.39122.11`, gmmlib `22.10.0`, Level Zero loader
  `1.32.0` (`docker/Dockerfile.xpu` lines 74-86).

### This host (verified, not changed)

- Kernel `7.1.0-070100-generic`.
- `intel-opencl-icd` / `libze-intel-gpu1` `26.22.38646.4-0`.
- Level Zero loader `libze1` `1.28.2-2` (`libze_loader.so.1.28.2`).
- RAM 121 GiB, about 78 GiB available at start. Disk `/mnt/vm_8tb` 5.7T free.
  Docker root is `/mnt/vm_8tb/docker`. `nproc` is 32.
- Docker client/server 29.1.3. `docker build` is buildx and has no
  `--memory` flag (help lists `--shm-size`, `--ulimit`, `--cgroup-parent`).

The upstream image's UMD 26.27 / L0 1.32 is newer than this host's 26.22 /
L0 1.28.2. A later GPU run should keep the host userspace. This build does
not install those debs.

## XPU feature matrix (code on main, not GPU-tested)

| Area | Status | Evidence |
| --- | --- | --- |
| Platform | supported | `vllm/platforms/xpu.py:103` `XPUPlatform`. Device `xpu`, control env `ZE_AFFINITY_MASK` (`xpu.py:105-112`). Worker `vllm.v1.worker.xpu_worker.XPUWorker` (`xpu.py:428`). |
| Native vLLM extensions | missing by design | `setup.py:1557-1559` skips `vllm.spinloop` and `vllm.fs_io_C` on XPU. `setup.py:1551-1555` builds CUDA/HIP cmake extensions only. `import_kernels` does not import `vllm._C` (`xpu.py:134-137`). Kernels come from the `vllm_xpu_kernels` wheel. |
| FlashAttention | supported via vllm-xpu-kernels | Default backend (`xpu.py:238-248`). `fa_utils.py:30-37` binds `flash_attn_varlen_func` to `vllm._xpu_ops`, which imports `vllm_xpu_kernels.flash_attn_interface` (`vllm/_xpu_ops.py:8`). Block size multiple of 64 (`flash_attn.py:358-376`). No FA4 (`xpu.py:214-217`). fp32 falls back to Triton (`xpu.py:232-237`). |
| Triton attention | supported via triton 3.8.0+xpu | Registry path `vllm.v1.attention.backends.triton_attn.TritonAttentionBackend` (`registry.py:48`). Selected for batch-invariance, multimodal prefix-LM, and fp32 (`xpu.py:191-237`). JIT is the `triton` +xpu wheel, not a second repo. |
| Triton MLA / FlashAttn MLA | supported, with fallback | `xpu.py:160-190`. Sparse MLA: `XPU_MLA_SPARSE` (`xpu.py:157-159`), Triton op in `vllm/v1/attention/ops/xpu_mla_sparse.py`. |
| ViT attention | partial | FLASH_ATTN, TRITON_ATTN, TORCH_SDPA (`xpu.py:251-255`). fp32 forces Triton (`xpu.py:265-270`). |
| Unquantized MoE | supported | Priority `XPU` then `TRITON` (`fused_moe/oracle/unquantized.py:96-103`). `XPUExperts` calls `vllm_xpu_kernels.fused_moe_interface.XpuFusedMoe` (`experts/xpu_moe.py:31,52`). Triton is the batch-invariance fallback (`unquantized.py:97-99`). |
| FP8 MoE | supported | XPU backend moved first (`oracle/fp8.py:134-137`); classes `XPUExpertsFp8`, `XPUExpertsMxFp8`, `XPUExpertsBlockFp8` (`fp8.py:237-244`). |
| INT4 WNA16 MoE | supported | XPU-only priority (`oracle/int_wna16.py:130-131`). `XPUExpertsWNA16` uses `xpu_fused_moe(is_int4=True)` for symmetric int4 (`xpu_moe.py:273-307`). |
| MXFP4 MoE | supported | XPU priority (`oracle/mxfp4.py:446-447`), class `XPUExpertsMxFp4` (`xpu_moe.py:311`). |
| NVFP4 MoE / linear | missing | `select_nvfp4_moe_backend` lists CUDA backends only (FlashInfer, CUTLASS, Marlin, Humming, emulation) (`oracle/nvfp4.py:203-212`). No XPU branch. |
| FP8 linear | supported | `supports_fp8` True (`xpu.py:551-552`). Fakes for `_xpu_C.fp8_gemm` and `fp8_gemm_w8a16` (`_xpu_ops.py:29-58`). Compressed-tensors FP8 on XPU is W8A8 only when `--linear-backend` is `xpu` or `torch`; otherwise W8A16, except block scales which always use W8A8 (`compressed_tensors.py:879-893`). |
| INT4 linear (GPTQ/AWQ/AutoRound/INC) | supported | Allow-list includes `awq`, `gptq`, `auto_awq`, `auto_gptq`, `inc` (`xpu.py:113-119`). INC XPU method calls `torch.ops._xpu_C.int4_gemm_w4a16` (`inc_wna16_linear.py:307-314`). W4A8 is opt-in `VLLM_XPU_INC_WNA16_BACKEND=w4a8` via `int4_gemm_w4a8` (`inc_w4a8_linear.py:10-28`). INC routes `RoutedExperts` to `scheme.get_moe_method` (`inc/inc.py:353-357`). |
| INT8 W8A8 GEMM | missing | No `int8_gemm` in `vllm-xpu-kernels/csrc` (FP8 W8A8 and INT4 GEMMs exist). `int8_utils.py:22-26` is only a Triton rounding helper (`tl.extra.intel.libdevice.round`). Docs hardware table marks llm-compressor INT8 W8A8 unsupported on Intel GPU (`docs/features/quantization/README.md:72`). |
| GGUF | missing | Not in `supported_quantization` (`xpu.py:113-131`). In-tree GGUF moved to `vllm-gguf-plugin` (`docs/features/quantization/gguf.md:7-12`, `setup.py` extra `vllm-gguf-plugin>=0.0.2`). Hardware table: Intel GPU column is unsupported (`quantization/README.md:77`). Plugin README requires a CUDA or ROCm toolkit and a CUDA extension; a code search of that repo found no XPU hits. |
| MXFP4 / MXFP8 quant | partial | Allow-listed (`xpu.py:122-123`). MX quant helpers take an XPU path (`fused_moe/utils.py:392-393`, `422-423`). Linear MXFP4 dispatch exists (`quantization/utils/mxfp4_utils.py:549`). Not GPU-tested. |
| Speculative MTP / Eagle / DFlash | supported at the engine | Methods include `qwen3_next_mtp`, `qwen3_5_mtp`, `eagle`, `eagle3`, `dflash` (`config/speculative.py:38-72`). No XPU rejection under `vllm/v1/spec_decode/`. Draft layers use the same XPU worker. GDN prefill on XPU prefers the SYCL kernel `torch.ops._xpu_C.gdn_attention` when present, else Triton (`qwen_gdn_linear_attn.py:167-179`). |
| XPU graph | partial | `support_static_graph_mode` True (`xpu.py:516-517`). Disabled when the torch build lacks it, or when sleep mode is on (`xpu.py:362-377`). Fusion passes AllReduce+RMSNorm, attn+quant, act+padding, RoPE+KV, RoPE+KV+MLA are forced off (`xpu.py:382-397`). Wrapper class is still `CUDAGraphWrapper` (`xpu.py:326-327`). Torch 2.14.0 `aten/src/ATen/xpu/XPUGraph.cpp:215` still calls `queue.ext_oneapi_graph` (see patches). |
| Collectives | supported as xccl | `dist_backend = "xccl"` (`xpu.py:111`). Communicator `XpuCommunicator`; warns if `torch.distributed.is_xccl_available()` is false (`xpu.py:540-548`). Multiproc method forced to `spawn` (`xpu.py:438-440`). Shutdown timeout forced to 5s (`xpu.py:442-449`). |
| KV offload to CPU RAM | supported for attention blocks | `CPUOffloadingSpec.get_worker` allows XPU (`v1/kv_offload/cpu/spec.py:213-219`). Shared mmap region is used on XPU (`spec.py:174-176`). Copy path is `ops.swap_blocks_batch`, not the Triton UVA kernel, because XPU cannot `tl.load` a CPU pointer (`gpu_worker.py:50-55`). Implementation is `torch.ops._C_cache_ops.swap_blocks_batch` (`_custom_ops.py:3029-3030`), registered by vllm-xpu-kernels (`csrc/torch_bindings.cpp:361-364`). Hybrid groups are described in the worker (`gpu_worker.py:586-589`). |
| UVA weight offload | partial | Platform detects UVA weight offload and disables Inductor's static Triton launcher (`xpu.py:399-420`). That is weight offload, not KV restore. |
| NixlConnector | partial | Registered (`kv_connector/factory.py:177-181`). Dockerfile.xpu builds UCX `--with-ze=yes` and NIXL 1.5.0 (`Dockerfile.xpu:173-202`). This attempt did not compile UCX/NIXL. |
| Other KV connectors | not XPU-specific | Factory also registers LMCache, Mooncake, HiSparse, MoRIIO, FlexKV, HF3FS, SimpleCPUOffload (`factory.py:153-248`). No XPU audit of those imports. |
| Hybrid (Mamba/GDN) prefix cache | supported, with a block constraint | `support_hybrid_kv_cache` returns True (`xpu.py:511-513`). Prefix caching derives mamba mode `"align"` (`config/cache.py:189-195`). If any layer's backend name is `GDN_ATTN`, attention block size is raised to a multiple of 64 (`xpu.py:462-499`). Eagle/MTP does not drop a mamba cache block (`kv_cache_coordinator.py:860-865`). |
| Batch invariance | partial | Triton attention is selected (`xpu.py:194-212`). Quantized models and quantized KV are rejected (`xpu.py:334-353`). Unquantized MoE falls back to Triton (`unquantized.py:97-99`). |

`vllm/platforms/xpu.py:11-13` imports `vllm_xpu_kernels._C`, `._moe_C`, and
`._xpu_C` at platform import. Those extensions are the SYCL kernels
(attention, GDN, grouped GEMM, int4/fp8/fp4 oneDNN GEMM, cache swap, LoRA).
Triton kernels that ship inside vLLM (`triton_attn.py`, Triton MoE, Triton
GDN fallback) compile at runtime with triton 3.8.0+xpu.

## How this repo previously built and served XPU vLLM

Pattern: pin an image, overlay Python or a rebuilt `vllm-xpu-kernels` wheel.
No live Dockerfile in this repo builds current main from scratch.

Package versions below were read with `importlib.metadata` inside the image
(`docker run --network none --memory 2g --cpus 1`, entrypoint
`/opt/venv/bin/python`, no devices).

| Image | Created | vLLM | torch | triton dist | vllm-xpu-kernels | oneAPI seen in the image |
| --- | --- | --- | --- | --- | --- | --- |
| `vllm-xpu-env:int8g-v0251` `30c04ad77c73` | 2026-07-16 | `0.25.1+xpu` | `2.12.0+xpu` | no dist named `triton` | `0.1.10.1` | Env shows oneAPI 2025.3, oneCCL 2021.15, MKL/DNNL 2025.3 (`/opt/intel/oneapi`) |
| `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff...` | 2026-08-14 | `0.27.2rc1.dev77+gac7509e2b.xpu` | `2.13.0+xpu` | no dist named `triton` | `0.1.12.3` | Official image. `VLLM_TARGET_DEVICE=xpu` |
| `vllm/vllm-openai-xpu:v0.28.0` `4756b66a0776` | 2026-08-25 | `0.28.0+xpu` | `2.13.0+xpu` | `3.7.2+xpu` | `0.1.13.2` | Official release image |
| `vllm/vllm-openai-xpu@sha256:1db27a8b75ae...` | 2026-09-09 | `0.29.0+xpu` | `2.13.0+xpu` | `3.7.2+xpu` | `0.1.14.1` | Pulled as `v0.29.0` (`vllm/int4/20260910_runtime_freshness.md`) |

Use on this host:

- 0.25.1: local tag `vllm-xpu-env:int8g-v0251`. Serve pin in
  `rdy_to_serve/vllm/qwen36-27b-nvfp4/serve.sh`. NVFP4 image build base in
  `vllm/nvfp4/build_nvfp4_f8scale_gdn.sh`. No Dockerfile for that tag remains
  in the live tree. `rdy_to_serve/README.md` still names vLLM 0.25.1 as the
  old default TP=1 stack.
- 0.27.2: digest pin in `vllm/gptq_int4/serve_qwen38_gptq_int4_v0272.sh:10`.
  JOURNAL entries (for example 2026-08-28) record
  `0.27.2rc1.dev77+gac7509e2b` and torch `2.13.0+xpu`. Cookbook MTP patches
  target this image (`vllm/patches/cookbook/README.md`).
- 0.28.0: digest pin in `vllm/gptq_int4/serve_qwen38_gptq_int4_v028.sh` and
  `vllm/nvfp4/serve_qwen38_v028.sh`. Kernel rebuild inside that image:
  `vllm/nvfp4/build_nvfp4_v028_inside.sh` (`MAX_JOBS` default 16, SYCL, AOT
  `bmg`). That script is a GPU-toolchain build and was not rerun here.
- Later local work (fp8, int4, w8a8) stacks small Dockerfiles `FROM` those
  images and copies Python or a kernels wheel
  (`vllm/fp8/Dockerfile.qwen38-fp8-mixed-gdn`,
  `vllm/w8a8/qwen38_mtp1/Dockerfile.runtime`).

## Build recipe for this host

`docker build --memory` is not available on Docker 29.1.3 buildx, and
`Dockerfile.xpu` runs `make -j$(nproc)` in the UCX stage. The recipe below
keeps the vLLM XPU install, caps jobs, and does not install UMD 26.27.

Do not pass `--device /dev/dri` or `--gpus`. `import torch` / `import vllm`
can open Level Zero if the device nodes are visible.

Scripts (scratch, not the git repo):

- `/mnt/vm_8tb/b70/build/vllm-flashnext/build_host.sh`
- `/mnt/vm_8tb/b70/build/vllm-flashnext/build_inside.sh`

Host command:

```
B70_AGENT=task01-upstream B70_CPU_LOCK_TIMEOUT=0 \
  bin/cpu-run /mnt/vm_8tb/b70/build/vllm-flashnext/build_host.sh
```

`build_host.sh` runs:

```
docker run --name vllm-fn-main-build \
  --memory 90g --memory-swap 90g --cpus 4 \
  -v $ROOT/vllm:/src/vllm \
  -v $ROOT/venv:/opt/venv \
  -v $ROOT/cache/uv:/root/.cache/uv \
  -v $ROOT/cache/cargo:/opt/cargo \
  ubuntu:24.04 bash /build_inside.sh
```

Inside the container (`MAX_JOBS=4`, `CARGO_BUILD_JOBS=4`, `MAKEFLAGS=-j4`,
`VLLM_TARGET_DEVICE=xpu`, `UV_INDEX_STRATEGY=unsafe-best-match`):

1. apt: build-essential, cmake, ninja, protobuf-compiler, python3.12-dev,
   libssl-dev, libaio-dev.
2. uv venv at `/opt/venv` on the bind mount (Python 3.12).
3. rustup toolchain `1.95` from `rust-toolchain.toml` (Rust is the only
   native compile; XPU `setup.py` builds no SYCL/CUDA extensions).
4. `uv pip install -r requirements/xpu.txt` (torch 2.14.0+xpu, triton
   3.8.0+xpu, vllm_xpu_kernels 0.1.15.4, auto_round_lib 0.16.0).
5. `uv pip install --no-build-isolation --no-deps .`
6. The Dockerfile.xpu oneAPI pip pins (sycl-rt / oneccl 2026.1.1 / 2022.1.2).
7. Verify with `importlib.metadata` only. Do not import torch.

Venv path: `/mnt/vm_8tb/b70/build/vllm-flashnext/venv`. The 2026-10-19
run left it populated. `requirements/xpu.txt` first installed oneAPI
2026.1.0 and oneccl 2022.1.1; step 6 replaced those with 2026.1.1 and
2022.1.2. `torch==2.14.0` resolved to `torch==2.14.0+xpu` under
`UV_INDEX_STRATEGY=unsafe-best-match`.

Not in this attempt, on purpose:

- UMD 26.27 / L0 1.32 debs from `Dockerfile.xpu` lines 74-86.
- UCX `--with-ze` and NIXL (`Dockerfile.xpu` lines 168-205). Needed only
  for NixlConnector. If built later, force `make -j4`, not `nproc`.
- A from-source `vllm-xpu-kernels` rebuild. Requirements pin the 0.1.15.4
  wheel. Source build needs oneAPI 2026.1 and would be a second CPU compile
  (`vllm-xpu-kernels` README). Do that only to move off the wheel pin, with
  `MAX_JOBS=4`, and still without opening `/dev/dri`.

Serving later (not done): bind host `/dev/dri`, keep host UMD 26.22 on
`LD_LIBRARY_PATH` ahead of any pip-shipped loader, `ZE_AFFINITY_MASK` per
card, and `bin/gpu-run`. The pip oneAPI 2026.1.1 libs can still shadow the
host loader; check `ldd` on `libze_loader` before the first GPU process.

## Build result

STATUS: finished. Container `vllm-fn-main-build` exited 0. `bin/cpu-run`
(agent `task01-upstream`) reported `done in 986s (exit 0)`.
`HOST_BUILD_START 2026-10-19T22:21:50+00:00`,
`HOST_BUILD_DONE 2026-10-19T22:38:16+00:00`. Watcher printed
`DONE: vllm main xpu container build finished`.

Stage markers: apt 22:21:56Z, uv 22:29:12Z, venv 22:29:13Z, rustup
22:29:14Z (`rustc 1.95.0`, `59807616e 2026-04-14`), requirements
22:29:28Z, vllm-install 22:31:21Z (wheel build 5m 58s), oneapi-pip-pins
22:37:32Z, verify 22:38:10Z, chown 22:38:11Z, done 22:38:13Z.

`importlib.metadata` inside the container (torch was not imported):

| Package | Version |
| --- | --- |
| vllm | 0.32.1rc1.dev94+g5fd62688f.xpu |
| torch | 2.14.0+xpu |
| torchaudio | 2.11.0+xpu |
| torchvision | 0.29.1+xpu |
| triton | 3.8.0+xpu |
| vllm-xpu-kernels | 0.1.15.4 |
| auto-round-lib | 0.16.0 |
| oneccl | 2022.1.2 |
| intel-sycl-rt | 2026.1.1 |

The pin step also set `dpcpp-cpp-rt`, `intel-opencl-rt`, `intel-openmp`,
`intel-cmplr-lib-rt`, and `intel-cmplr-lib-ur` to 2026.1.1, `intel-pti`
to 1.1.0, and `oneccl-devel` to 2022.1.2. `torchcodec` landed as
`0.17.0+cpu`. Python is 3.12.3. uv is 0.13.2.

Rust frontend artifact (host path, owner hotschmoe, 57 MiB, ELF
x86-64): `/mnt/vm_8tb/b70/build/vllm-flashnext/venv/lib/python3.12/site-packages/vllm/vllm-rs`.
No `vllm/_C*.so` under that site-packages tree. `vllm_xpu_kernels` and
`triton` (plus dist `triton_xpu` 3.8.0) are installed as wheels.

No `/dev/dri` was passed. No GPU process was started. The pip
`intel-opencl-rt` package is inside the venv and is not the host UMD
26.22. A later serve still has to check `ldd` on `libze_loader` before
the first GPU process.

Log: session terminal capture
`call-0c7b5492-f395-4d32-ae02-67fe3a2d6565-105.log`.

CONFIG -> ubuntu:24.04, no GPU devices, `--memory 90g --memory-swap 90g
--cpus 4`, `MAX_JOBS=4`, `CARGO_BUILD_JOBS=4`, source
`5fd62688ffe995c0c11c8708fe51dbff53235b74`, `VLLM_TARGET_DEVICE=xpu`.
COMMAND -> `B70_AGENT=task01-upstream B70_CPU_LOCK_TIMEOUT=0 bin/cpu-run
/mnt/vm_8tb/b70/build/vllm-flashnext/build_host.sh`.
RESULT -> exit 0 in 986s. vLLM `0.32.1rc1.dev94+g5fd62688f.xpu` (main
HEAD, 94 commits past the 0.32.1rc1 base, not the v0.32.0 release),
torch `2.14.0+xpu`, triton `3.8.0+xpu`, kernels wheel `0.1.15.4`,
`vllm-rs` present, no `vllm._C`.
VERDICT -> CPU install of main for XPU succeeded. Not a GPU health,
identity, or speed result.

## Old `vllm/patches` and contrib vs main

| Local change | Verdict |
| --- | --- |
| `vllm/patches/kv_cache_utils_gcd.py` | Obsolete as a vendored file. Main computes hash block size with `math.gcd` (`vllm/v1/core/kv_cache_utils.py:891`, `kv_cache_planning.py:1183`). |
| `vllm/fp8/patches/offload-merged-52771-52807-54288.patch` | Obsolete. The issue #52735 comments are in `kv_connector/v1/offloading/scheduler.py:244`, `:465`, `:904`. |
| `vllm/int4/patches/r276-gdn-prefill-phase.patch` and `v029-gdn-prefill-phase.patch` | Obsolete as diffs. Main rewrote the no-spec split and passes `treat_short_extends_as_decodes=False` (`gdn_attn.py:357-377`). Not runtime-compared. |
| `vllm/int4/patches/r276-mamba-eagle-drop.patch` | Obsolete. Coordinator gives mamba no eagle margin and says the finder never drops (`kv_cache_coordinator.py:860-865`). |
| `vllm/int4/patches/r276-accepted-count-order.patch` | Partial. A D2H sync of `num_accepted_tokens_event` is at `gpu_model_runner.py:2065-2067`, with an async/non-align skip (`:2062-2064`). The patch's earlier sync is not present. Revalidate before reapplying. |
| `vllm/int4/patches/r276-gdn-request-trace.patch` | Not re-checked line by line. Treat as a debug overlay, not an upstream feature. |
| `vllm/patches/cookbook/patch_mtp_nightly.py`, `patch_mtp_bf16_draft_v0260.py` | Still a lab overlay if a checkpoint cannot exclude MTP weights. No `B70_MTP_BF16_DRAFT` hook on main. Anchors are 0.26.x and will not apply. |
| `vllm/patches/cookbook/patch_mtp_boundary.py` | Still needed until a partial final MTP group is shown safe on XPU GDN. Main forces GDN block multiple of 64 (`xpu.py:462-472`) but does not contain this patch. |
| `vllm/patches/cookbook/patch_gdn_mixed_split_v5.py` | Obsolete as a diff against 0.27 `gdn_attn.py`. Main's spec-mask path is the rewrite at `gdn_attn.py:337-355`. |
| `vllm/patches/cookbook/patch_draft_lmhead_int4.py`, `patch_draft_mtp_int4.py` | Still needed for the runtime RTN-INT4 draft memory trick. Not in main. |
| `vllm/patches/qwen3_dflash_swa.py` | DFlash itself is upstream (`speculative.py:69`, `gpu_model_runner.py` `DFlashProposer`). The `B70_DFLASH_SWA=1` uniform sliding window (`qwen3_dflash_swa.py:176-183`) is still local. |
| `vllm/patches/xpugraph_submit_without_event.patch` | Still needed, and it is a PyTorch patch, not a vLLM patch. Torch 2.14.0 `aten/src/ATen/xpu/XPUGraph.cpp:215` is still `queue.ext_oneapi_graph(*graph_exec_)`. The patch header targets torch 2.12 `7661cd9c`. Rebase onto 2.14 before using it. |
| `vllm/contrib/vllm_moe_xpu` | Obsolete. It fixed INC on 0.23.0 returning None for `RoutedExperts`. Main routes those layers at `inc/inc.py:353-357` and has `XPUExpertsWNA16`. |
| `vllm/contrib/vllm_int8_xpu` | Still needed for a real INT8 W8A8 GEMM. Main has no XPU int8 GEMM (see matrix). |
| `vllm/contrib/vllm_wna16_xpu` | Still a last-resort dequant for odd-K compressed-tensors WNA16. README says it is not wired. Main's fast path is `int4_gemm_w4a16` / `XPUExpertsWNA16`, which still have alignment rules (`inc_w4a8_linear.py:35-37` for the w4a8 variant). |
| `vllm/contrib/vllm_push_allreduce` | Still needed for the local TP=2 PUSH collective. Upstream XPU communication is xccl (`xpu.py:111`, `:540-548`). AGENTS.md still says the loaded native push graph stalls at first submission; do not treat this as a serving default. |
| `vllm/nvfp4/` SYCL port and sitecustomize | Still needed for NVFP4 on XPU. Main's NVFP4 MoE selector has no XPU backend (`oracle/nvfp4.py:203-212`). The 0.28 in-image rebuild script is tied to kernels override `0.1.13.2` and image `v0.28.0`. |
