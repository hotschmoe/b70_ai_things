#!/usr/bin/env bash
# Run only inside the owned CPU-only build container, with /inputs read-only.
set -euo pipefail
[ ! -e /dev/dri ] || { echo 'Refusing exposed GPU devices'; exit 2; }
mkdir -p /out/wheels /out/work
python3 /recipe/runtime_identity.py /out/runtime-before.json
cp -a /inputs/. /out/work/
export MAX_JOBS="${B70_BUILD_JOBS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" CARGO_BUILD_JOBS="$MAX_JOBS"
export BUILD_TARGET_DEVICE=bmg TMS_PLATFORM=xpu
export CMAKE_ARGS='-DDPCPP_SYCL_TARGET=bmg -DSGL_BUILD_MEMORY_MONITOR=OFF -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/out/work/sycl-tla'
export USE_MOE="${B70_BUILD_MOE:-ON}" USE_FMHA="${B70_BUILD_FMHA:-ON}" USE_MLA="${B70_BUILD_MLA:-ON}"
export USE_GDN=ON USE_SYCL_JIT=OFF
export CC=gcc CXX=g++
icpx --version
cargo --version
rustc --version
python3 -m pip install --no-deps setuptools-rust==1.12.0 setuptools-scm==8.2.0 semantic-version==2.10.0 scikit-build-core==0.11.6 pathspec==0.12.1
python3 -m pip freeze > /out/pip-build-tools.txt
python3 -m pip wheel --verbose --no-build-isolation --no-deps --wheel-dir /out/wheels /out/work/sgl-kernel-xpu
python3 -m pip uninstall -y sgl-kernel
python3 -m pip install --no-deps /out/wheels/sglang_kernel_xpu-0.2.0-*.whl
python3 -m pip wheel --verbose --no-build-isolation --no-deps --wheel-dir /out/wheels /out/work/torch_memory_saver
python3 -m pip install --no-deps /out/wheels/torch_memory_saver-*.whl
cp /out/work/sglang/python/pyproject_xpu.toml /out/work/sglang/python/pyproject.toml
python3 - <<'PY'
from pathlib import Path
p=Path('/out/work/sglang/python/pyproject.toml')
s=p.read_text();old='sglang-kernel-xpu @ git+https://github.com/sgl-project/sgl-kernel-xpu.git'
assert s.count(old)==1
p.write_text(s.replace(old,'sglang-kernel-xpu==0.2.0'))
PY
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.19.dev0+g2f7393f0d
python3 -m pip wheel --verbose --no-build-isolation --no-deps --wheel-dir /out/wheels /out/work/sglang/python
python3 -m pip install --no-deps --force-reinstall /out/wheels/sglang-0.5.19.dev0+g2f7393f0d-*.whl
python3 /recipe/cpu_import_gate.py
python3 /recipe/runtime_identity.py /out/runtime-after.json
python3 - <<'PY'
import json,pathlib
p=pathlib.Path('/out')
a=json.loads((p/'runtime-before.json').read_text());b=json.loads((p/'runtime-after.json').read_text())
assert a['held_packages']==b['held_packages']
assert a['held_files']==b['held_files']
assert a['driver_packages']==b['driver_packages']
(p/'BUILD_PASS').write_text('Pinned source builds completed; held torch/UMD/CCL hashes unchanged. GPU qualification pending.\n')
PY
