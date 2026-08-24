# Repo-owned production build recipe for qwen38-b70.
# build_image.sh stages the two pinned optimization-lab patches and entrypoint
# from the 0xSero checkout, then adds the repo-owned quant instrumentation.
FROM intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
        python3-pip python3-venv ffmpeg \
    && pip3 install --break-system-packages "huggingface_hub[cli]" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone -q https://github.com/mndodd/llama.cpp llama.cpp \
    && cd llama.cpp \
    && git checkout -q 4302fb59969a5d8cf9f8e5f55fdd4506d0ed2126
COPY patches/ /patches/
RUN cd /build/llama.cpp \
    && git apply --check /patches/tp2-full-stack.patch \
    && git apply /patches/tp2-full-stack.patch \
    && git apply --check /patches/q4k-increment.patch \
    && git apply /patches/q4k-increment.patch \
    && git apply --check /patches/quant-census.patch \
    && git apply /patches/quant-census.patch \
    && git apply --check /patches/quant-timing.patch \
    && git apply /patches/quant-timing.patch \
    && git diff --check

SHELL ["/bin/bash", "-c"]
RUN cd /build/llama.cpp \
    && . /opt/intel/oneapi/setvars.sh --force \
    && cmake -G "Unix Makefiles" -S . -B build-sycl \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/cc \
        -DCMAKE_CXX_COMPILER="$(command -v icpx)" \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
        -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
        -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=OFF -DGGML_SYCL_DNN=OFF \
        -DGGML_SYCL_HOST_MEM_FALLBACK=OFF \
        -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
    && cmake --build build-sycl --target llama-server -j"$(nproc)"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MODEL_REPO=ggml-org/Qwen3.8-27B-GGUF \
    MODEL_REV=0669b98607d47046c7c2b3f801011d54a08cfccf \
    MODEL_FILE=Qwen3.8-27B-Q4_K_M.gguf \
    MODEL_SHA256=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34 \
    MTP_FILE=mtp-Qwen3.8-27B-Q4_0.gguf \
    MMPROJ_FILE=mmproj-Qwen3.8-27B-Q8_0.gguf

EXPOSE 8010
ENTRYPOINT ["/entrypoint.sh"]
