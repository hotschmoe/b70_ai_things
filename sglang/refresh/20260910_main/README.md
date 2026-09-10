# Pinned current-main SGLang CPU refresh

CONFIG -> Base image `intel/sglang-dev@sha256:1a13d3d2b0adc63422a643eb5c59524842577d99ca3935b4b8c2c432e739e127`.
Hold its torch 2.13.0+xpu, triton-xpu 3.7.2, container UMD 26.18.38308.1,
and oneCCL libraries. The host remains unchanged. No device nodes are exposed.

Pinned source archives:

- SGLang: `2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed`.
- sglang-kernel-xpu: `3802d7d87829073535fe5f84d3aef4ed76e7ef90`.
- BMG sycl-tla: `87f6850680a580654b9ea2c80dbc01aeb36ad231`.
- torch_memory_saver: `a5c99f11b18ebb8e9fda71a68812e476ae49e417`.

COMMAND -> `python3 sglang/refresh/20260910_main/run_build.py`.

The driver uses a dedicated container with 4 CPUs, 24 GiB memory/no extra
swap, and a 90-minute deadline. Build parallelism is four throughout. CMake
receives `-DDPCPP_SYCL_TARGET=bmg`; the inspected `BuildFlags.cmake` then skips
Level Zero discovery. The pinned local sycl-tla source replaces FetchContent's
network checkout. torch_memory_saver uses `TMS_PLATFORM=xpu`. Build-time
package installation uses `--no-deps` to prevent replacing runtime libraries.
All required native extensions are rebuilt from the supplied source. No
quarantined source or binaries are used.

RESULT -> Build started; raw log, source archive hashes, exact Docker command,
base-image identity, and held runtime hashes are under
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-refresh/`.

VERDICT -> In progress, not serving or GPU-qualified. A successful build must
preserve the held runtime hashes, resolve required dependencies without
changing other layers, and pass new-stack paired health before any serving.
The stopped build container is retained for inspection and image creation;
no image is automatically promoted.

CONFIG -> First attempt used pinned setuptools-rust/scikit-build-core installed
without transitive dependency resolution.

COMMAND -> Initial `run_build.py` invocation, raw `build/`.

RESULT -> Metadata preparation stopped before native compilation because the
build-only `pathspec` module was absent. No runtime libraries changed.

VERDICT -> Add pinned pathspec 0.12.1 and retry in raw `build2/` using
`run_build.py --attempt build2`; preserve the first attempt evidence.

CONFIG -> Dense Qwen3.8-27B scope approved by the coordinator. MoE and MLA are
unrelated to this model's dense attention/GDN/MTP path.

COMMAND -> Stop the owned full build container, verify it is not running,
then `run_build.py --attempt dense1 --profile dense`.

RESULT -> The deferred full build is preserved in `build2/`. A clean dense
build starts with USE_MOE=OFF, USE_MLA=OFF, USE_FMHA=ON, USE_GDN=ON and
USE_SYCL_JIT=OFF. CMake confirms explicit BMG and generates 663 actions instead
of 865. No partial compiler artifacts from the full build are reused. Each
new attempt mounts an immutable snapshot of the recipe from its output tree.

VERDICT -> This candidate targets dense Qwen only. It is not an Ornith/MoE or
MLA replacement, and unchanged runtime-JIT dispatch avoids adding another
kernel-path change. Native build and GPU qualification remain outstanding.

CONFIG -> Resource-only increase approved after checking host headroom.
Source, image, compiler ABI, BMG target and dense feature flags remain fixed.

COMMAND -> Stop/verify the owned `dense1` container, then run:

```text
python3 sglang/refresh/20260910_main/run_build.py --attempt dense16 \
  --profile dense --cpus 16 --memory-gib 64 \
  --resume-work /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-refresh/dense1/work
```

RESULT -> Preserve 40 completed objects and CMake cache at the same container
path `/out/work`. The new output `dense16/` contains its own recipe snapshot,
logs and runtime identity. Resume validates matching image, source manifest
and dense profile. CPU, Ninja and Cargo limits increase to 16; memory/no-extra-
swap increases to 64 GiB. `dense1/RESOURCE_RESUME.json` records the boundary.

VERDICT -> Only resources changed. Partial full-MoE build artifacts remain
separate and unused. No change to kernel dispatch or host runtime.

CONFIG -> Dense text-only Triton-attention candidate approved separately.
Preserve pinned source, compiler/PyTorch/UMD ABI, explicit BMG and 16 CPU/64 GiB
bounds. Disable FMHA as well as MoE/MLA; retain GDN and SYCL_JIT=OFF.

COMMAND -> A separate CPU-only CMake configure and Ninja dry-run proved 176
clean actions versus 663 with FMHA. Stop/verify `dense16` (false/137), preserve
its 196 objects and logs, then run without object reuse:

```text
python3 sglang/refresh/20260910_main/run_build.py --attempt triton-dense1 \
  --profile triton-dense --cpus 16 --memory-gib 64
```

RESULT -> Build started in separate `triton-dense1/`. CMake filters the FMHA
base source and all generated FMHA instances; native `fwd` registration is
macro-guarded. Both Python `torch.ops.sgl_kernel.fwd` lookups occur inside
function bodies, not during import. GPTQ and GDN remain selected. The build
now includes an actual installed-package CPU import/API gate, native hashes,
and a separate dependency report before its held-runtime hash check.

VERDICT -> This deliberately reduced backend requires Triton text attention.
It cannot execute native Intel FMHA or vision FMHA inference and is not a
universal backend replacement. Prefix cache, graph, MTP and calibrated FP8 KV
still require their respective runtime qualification. No GPU was exposed;
paired new-stack health remains mandatory before serving.

CONFIG -> `triton-dense1` completed all native kernel, memory-saver and four
Rust extension builds with the intended pinned source and held ABI.

COMMAND -> Build SGLang wheel under no-build-isolation, then require the
explicit source-bearing version filename for installation.

RESULT -> The base lacked setuptools-scm. Wheel metadata silently fell back
to version 0.0.0; the expected-version install guard rejected that filename.
This attempt exited 1 before CPU import/final identity gates, so no candidate
was qualified. Native output and the rejected wheel are preserved.

VERDICT -> Add build-only setuptools-scm 8.2.0 and resume the exact clean
Triton-dense source/profile/object tree in separate `triton-dense2/` output.
This repairs metadata tooling; it changes no native source, compiler flags,
PyTorch, UMD, dispatch or model configuration.

CONFIG -> Metadata-corrected `triton-dense2` retains the same native source,
profile and ABI. The thin runtime image starts from the original immutable
base and installs only the three rebuilt wheels, after uninstalling old
`sgl-kernel`. Build trees and Cargo caches are excluded.

COMMAND -> Resume `triton-dense1/work` with the same profile/resources and
pinned setuptools-scm 8.2.0. Package the three verified wheels using the tracked
Dockerfile, then run `cpu_import_gate.py` and `runtime_identity.py` in a fresh
CPU-only container by immutable image ID.

RESULT -> Runnable image:
`sha256:bdc51c5f083fdcbacd59a74bfeb8389fa6d62c8478e066ed5c62f9bffcfaa4cf`.
Tag: `b70/sglang-main-triton-dense:20260910`.
All 63 installed native hashes equal the rebuilt wheels: 58 kernel libraries,
one memory-saver binary and four Rust extensions. All four Rust extensions
and the memory-saver library load successfully. The final process loads 57
kernel libraries with no untracked native payload; the old distribution is
absent. Native GDN and held PyTorch GPTQ XPU dispatch are registered; native
FMHA `fwd` is absent as intended. The 3820 packaged Python source files match
the pinned archives, apart from the separately generated version module.
Held torch/UMD/CCL hashes and versions are unchanged. `pip check` exactly
matches the base's pre-existing xgrammar dependency on the distribution name
`triton`; the Intel package is named `triton-xpu`. CUDA Triton was not installed.
All owned CPU build/check containers are stopped or removed.

VERDICT -> CPU build, import, source, native-payload and held-runtime gates
passed. `verified_image.json` records hashes and raw evidence. This is current
main at the recorded pin, not the older four-file port, and is ready for
new-stack health. It has no GPU qualification yet.

The reviewed pair-preflight helper is copied with its ownership, timeout,
cleanup and recovery guards intact. It pins the strict health-probe hash and
uses a separate, non-overwriting result root. Its four CPU ownership/timeout
checks pass. The following plan is prepared only; the coordinator must run it
when both cards are released:

```text
B70_AGENT=sglang-main-triton-dense-preflight bin/gpu-run \
  python3 sglang/refresh/20260910_main/preflight.py
```

Output: `/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-triton-dense-preflight/`.
The plan runs strict per-card health and the compiled two-rank collective with
P2P=0, CCL root `/opt/venv`, and bounded owned cleanup. Only after it passes may
a separate card lease run FP16-KV/eager/MTP0 viability with explicit
`--attention-backend triton --linear-attn-backend triton`. Calibrated FP8,
graphs, MTP, cache qualification and TP2 serving remain separate gates.
