#!/usr/bin/env bash
# Offline CPU-only compilation. Initialize oneAPI BEFORE strict shell options.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
command -v icpx sha256sum ldd grep cp mkdir >/dev/null
[[ -w /out && -r /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h && -r /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3 ]]
cd /out
sha256sum -c /prep/expected-container.sha256
mkdir /out/frozen
mkdir /out/frozen/include /out/frozen/provenance
cp /work/src-ep/build-sycl-ep/ggml/src/libggml-base.a /out/frozen/libggml-base.a
cp /work/llama-pinned/ggml/include/ggml.h /out/frozen/include/ggml.h
cp /work/llama-pinned/ggml/src/ggml.c /work/llama-pinned/ggml/src/ggml-quants.c /work/llama-pinned/ggml/src/ggml-impl.h /work/llama-pinned/ggml/src/ggml-common.h /out/frozen/provenance/
sha256sum -c /prep/expected-frozen.sha256
icpx --version > compiler-version.txt
printf '%s\n' 'icpx -O2 -std=c++20 -fp-model=precise -ffp-contract=off -frounding-math -I/out/frozen/include -I/opt/intel/oneapi/mkl/latest/include /prep/reference_s80.cpp /out/frozen/libggml-base.a -L/opt/intel/oneapi/mkl/latest/lib -Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib -l:libmkl_rt.so.3 -pthread -ldl -lm -o /out/reference_s80' > compile-command.txt
icpx -O2 -std=c++20 -fp-model=precise -ffp-contract=off -frounding-math \
 -I/out/frozen/include -I/opt/intel/oneapi/mkl/latest/include \
 /prep/reference_s80.cpp /out/frozen/libggml-base.a \
 -L/opt/intel/oneapi/mkl/latest/lib -Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib \
 -l:libmkl_rt.so.3 -pthread -ldl -lm -o /out/reference_s80
ldd /out/reference_s80 > linked-libraries.txt
if grep -Eiq 'libsycl|libze_loader|libOpenCL' linked-libraries.txt;then echo 'FAIL CPU-only linkage';exit 2;fi
sha256sum /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h /opt/intel/oneapi/mkl/latest/include/mkl_service.h /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3 > MKL-files.sha256
sha256sum -c /prep/expected-container.sha256
sha256sum -c /prep/expected-frozen.sha256
sha256sum /prep/reference_s80.cpp /prep/reference_bound.hpp /out/reference_s80 > build-sha256.txt
