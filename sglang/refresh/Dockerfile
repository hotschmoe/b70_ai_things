# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG SGLANG_COMMIT=bede6bc37c5d9638099ebb948d93b9e2a7799f10
ARG SGLANG_TREE=938cf2b1b71bbc60e5d18d8388f1388ca0eff5a7
ARG SGL_KERNEL_COMMIT=2d10888c069350ff20a192338d568dec945c9594
ARG SGL_KERNEL_TREE=3f975153d4d430535c759e57cb176e141a1b25c8
ARG BUILD_JOBS=8

LABEL org.opencontainers.image.title="B70 exact SGLang XPU refresh"
LABEL org.opencontainers.image.revision="${SGLANG_COMMIT}"
LABEL b70.sglang.tree="${SGLANG_TREE}"
LABEL b70.sgl-kernel-xpu.commit="${SGL_KERNEL_COMMIT}"
LABEL b70.sgl-kernel-xpu.tree="${SGL_KERNEL_TREE}"
LABEL b70.compute-runtime="26.22.38646.4"
LABEL b70.igc="2.36.3+21719"
LABEL b70.gmm="22.10.0"
LABEL b70.level-zero-loader="1.28.2"
LABEL b70.torch="2.13.0+xpu"
LABEL b70.triton-xpu-alias="3.7.2"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/opt/intel/oneapi/compiler/2026.0/bin:${PATH}"

USER root

# Replace the base image's rolling GPU userspace with the exact host-matched
# boundary. Asset hashes make the GitHub release downloads content-pinned.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates ccache cmake curl git ninja-build \
        python3 python3-dev python3-venv && \
    old_pkgs=() && \
    for package in \
        intel-opencl-icd libze-intel-gpu1 intel-ocloc \
        intel-igc-core-2 intel-igc-opencl-2 libigdgmm12 \
        libze-dev libze1 level-zero-devel level-zero; do \
      if dpkg-query -W "$package" >/dev/null 2>&1; then \
        old_pkgs+=("$package"); \
      fi; \
    done && \
    if ((${#old_pkgs[@]})); then apt-get remove -y "${old_pkgs[@]}"; fi && \
    mkdir -p /tmp/gpu-stack && \
    curl -fsSLo /tmp/gpu-stack/level-zero.deb \
        'https://github.com/oneapi-src/level-zero/releases/download/v1.28.2/level-zero_1.28.2%2Bu24.04_amd64.deb' && \
    curl -fsSLo /tmp/gpu-stack/level-zero-devel.deb \
        'https://github.com/oneapi-src/level-zero/releases/download/v1.28.2/level-zero-devel_1.28.2%2Bu24.04_amd64.deb' && \
    curl -fsSLo /tmp/gpu-stack/intel-igc-core.deb \
        'https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-core-2_2.36.3%2B21719_amd64.deb' && \
    curl -fsSLo /tmp/gpu-stack/intel-igc-opencl.deb \
        'https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-opencl-2_2.36.3%2B21719_amd64.deb' && \
    curl -fsSLo /tmp/gpu-stack/libze-intel-gpu.deb \
        'https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/libze-intel-gpu1_26.22.38646.4-0_amd64.deb' && \
    curl -fsSLo /tmp/gpu-stack/intel-opencl-icd.deb \
        'https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/intel-opencl-icd_26.22.38646.4-0_amd64.deb' && \
    curl -fsSLo /tmp/gpu-stack/intel-ocloc.deb \
        'https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/intel-ocloc_26.22.38646.4-0_amd64.deb' && \
    curl -fsSLo /tmp/gpu-stack/libigdgmm12.deb \
        'https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/libigdgmm12_22.10.0_amd64.deb' && \
    printf '%s\n' \
        'df339abcc8872b3d30887b8845c62a9768db943debcdd96a900b573a02a12cae  /tmp/gpu-stack/level-zero.deb' \
        'ff5f04c2d934acdb9b6d1cf6aa72359160d501881f440823bce55aa4a47737cf  /tmp/gpu-stack/level-zero-devel.deb' \
        '9e0975ac75015b431ebb2da81a802b9fd1e28a3c270313a97569cd1e6a6c6048  /tmp/gpu-stack/intel-igc-core.deb' \
        '350a52331e784bb7fb9ed42e993b5c44b7e6562fc74d2cf3102b29b6a576fa85  /tmp/gpu-stack/intel-igc-opencl.deb' \
        '8bef9f24e03f826f93c076081bda13c6ac3afbd9e42b9fb8f298fab652330e2f  /tmp/gpu-stack/libze-intel-gpu.deb' \
        '6fdac2e8a2aacf844ebfd90521bf7102b3ebb44f69c1bced1a9785a7ce96a3c2  /tmp/gpu-stack/intel-opencl-icd.deb' \
        '25745841e66279c7ff1ff6c2d4da9ec911685c12e1c9c5609d99e4d54a364dc0  /tmp/gpu-stack/intel-ocloc.deb' \
        '6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  /tmp/gpu-stack/libigdgmm12.deb' | sha256sum -c - && \
    apt-get install -y --no-install-recommends --allow-downgrades \
        /tmp/gpu-stack/level-zero.deb \
        /tmp/gpu-stack/level-zero-devel.deb \
        /tmp/gpu-stack/libigdgmm12.deb \
        /tmp/gpu-stack/intel-igc-core.deb \
        /tmp/gpu-stack/intel-igc-opencl.deb \
        /tmp/gpu-stack/libze-intel-gpu.deb \
        /tmp/gpu-stack/intel-opencl-icd.deb \
        /tmp/gpu-stack/intel-ocloc.deb && \
    apt-mark hold level-zero level-zero-devel libigdgmm12 intel-igc-core-2 \
        intel-igc-opencl-2 libze-intel-gpu1 intel-opencl-icd intel-ocloc && \
    python3 -m venv "${VIRTUAL_ENV}" && \
    rm -r /tmp/gpu-stack && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        pip==25.2 setuptools==80.9.0 setuptools-scm==9.2.0 wheel==0.45.1 \
        build==1.3.0 scikit-build-core==0.11.6 ninja==1.13.0 && \
    python -m pip install --no-cache-dir \
        torch==2.13.0+xpu torchvision==0.28.0+xpu torchaudio==2.11.0+xpu \
        --index-url https://download.pytorch.org/whl/xpu

COPY sgl-kernel-xpu /src/sgl-kernel-xpu
COPY sglang /src/sglang

RUN test "$(git -C /src/sgl-kernel-xpu rev-parse HEAD)" = "${SGL_KERNEL_COMMIT}" && \
    test "$(git -C /src/sgl-kernel-xpu rev-parse HEAD^{tree})" = "${SGL_KERNEL_TREE}" && \
    test -z "$(git -C /src/sgl-kernel-xpu status --porcelain)" && \
    test "$(git -C /src/sglang rev-parse HEAD)" = "${SGLANG_COMMIT}" && \
    test "$(git -C /src/sglang rev-parse HEAD^{tree})" = "${SGLANG_TREE}" && \
    test -z "$(git -C /src/sglang status --porcelain)"

# Build every kernel feature, including MoE needed by the Ornith lane. Disable
# only the source tree's 4 GiB-per-file kill wrapper and bound parallel jobs.
RUN set +u && \
    source /opt/intel/oneapi/setvars.sh --force && \
    set -u && \
    CC=gcc CXX=g++ \
    MAX_JOBS="${BUILD_JOBS}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" \
    CMAKE_ARGS='-DSGL_BUILD_MEMORY_MONITOR=OFF' \
    USE_MOE=ON USE_FMHA=ON USE_MLA=ON \
    python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /opt/wheels /src/sgl-kernel-xpu && \
    python -m pip install --no-cache-dir --no-deps \
        /opt/wheels/sgl_kernel-0.11.0-*.whl

# Replace the moving git-main kernel requirement with the local exact wheel.
# The metadata copy dirties the checkout, so fix the computed source version.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.19.dev443+gbede6bc37c
RUN cp /src/sglang/python/pyproject_xpu.toml \
       /src/sglang/python/pyproject.toml && \
    sed -i \
        's#"sgl-kernel @ git+https://github.com/sgl-project/sgl-kernel-xpu.git",#"sgl-kernel==0.11.0",#' \
        /src/sglang/python/pyproject.toml && \
    grep -Fx '  "sgl-kernel==0.11.0",' \
        /src/sglang/python/pyproject.toml && \
    python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /opt/wheels /src/sglang/python && \
    python -m pip install --no-cache-dir \
        /opt/wheels/sglang-0.5.19.dev443+gbede6bc37c-*.whl \
        --extra-index-url https://download.pytorch.org/whl/xpu

# XGrammar is imported unconditionally by current function-call detectors.
# Its generic Triton dependency is supplied by Torch XPU under the distribution
# name triton-xpu; install an empty metadata alias without replacing its files.
COPY --from=b70-refresh triton_xpu_alias /src/triton-xpu-alias
RUN python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /opt/wheels /src/triton-xpu-alias && \
    python -m pip install --no-cache-dir apache-tvm-ffi==0.1.13.post3 && \
    python -m pip install --no-cache-dir --no-deps \
        /opt/wheels/triton-3.7.2-*.whl xgrammar==0.2.1

# The upstream compressed-tensors loader recognizes W8A8 INT8 but rejects it
# on XPU. Keep the B70 port isolated and environment-gated at Python startup.
COPY --from=b70-refresh b70_xpu_w8a8.py \
    /opt/venv/lib/python3.12/site-packages/b70_xpu_w8a8.py
COPY --from=b70-refresh b70_xpu_w8a8.pth \
    /opt/venv/lib/python3.12/site-packages/b70_xpu_w8a8.pth

RUN mkdir -p /opt/b70-build-manifest && \
    python -m pip check && \
    python -m pip freeze | LC_ALL=C sort \
        > /opt/b70-build-manifest/pip-freeze.txt && \
    dpkg-query -W | LC_ALL=C sort \
        > /opt/b70-build-manifest/dpkg-query.txt && \
    sha256sum /opt/wheels/*.whl \
        > /opt/b70-build-manifest/wheel-sha256.txt && \
    python - <<'PY'
import importlib.metadata
import torch

import sgl_kernel
import sglang

print("torch", torch.__version__, torch.version.git_version)
print("sglang", importlib.metadata.version("sglang"), sglang.__file__)
print("sgl-kernel", importlib.metadata.version("sgl-kernel"), sgl_kernel.__file__)
PY

WORKDIR /sgl-workspace
CMD ["bash", "-lc", "source /opt/intel/oneapi/setvars.sh --force && exec bash"]
