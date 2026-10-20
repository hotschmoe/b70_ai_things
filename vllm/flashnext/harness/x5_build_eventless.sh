#!/usr/bin/env bash
# CPU-only late-load variant of X1's validated replay body. cpu-run/MAX_JOBS4.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT=/mnt/vm_8tb/b70/build/vllm-flashnext/x5/eventless-lazy
VENV=/mnt/vm_8tb/b70/build/vllm-flashnext/venv
TORCH=$VENV/lib/python3.12/site-packages/torch
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1
TC=/mnt/vm_8tb/b70/build/gguf-xpu-toolchain
mkdir -p "$OUT/tmp"
export TMPDIR="$OUT/tmp" MAX_JOBS=4
cp "$REPO/vllm/flashnext/harness/patches/x1_eventless.cpp" "$OUT/helper.cpp"
cat >> "$OUT/helper.cpp" <<'CPP'
extern "C" void b70_x5_eventless_impl(void* self) {
    // Qualified call is essential: virtual dispatch returns to the stub.
    static_cast<at::xpu::XPUGraphImpl*>(self)->at::xpu::XPUGraphImpl::replay();
}
CPP
cat > "$OUT/helper.map" <<'MAP'
{ local: _ZN2at3xpu12XPUGraphImpl6replayEv; };
MAP
"$ONEAPI/bin/compiler/clang" --gcc-toolchain="$TC" -std=c11 -O2 -fPIC -shared "$REPO/vllm/flashnext/harness/patches/x5_eventless_stub.c" \
  -Wl,--as-needed -ldl -o "$OUT/libx5_eventless_stub.so"
if [[ "${STUB_ONLY:-0}" == 1 ]]; then
  readelf -d "$OUT/libx5_eventless_stub.so" | rg NEEDED
  sha256sum "$OUT/libx5_eventless_stub.so"
  exit 0
fi
"$ONEAPI/bin/icpx" -fsycl -std=c++20 -O2 -fPIC -shared \
  --gcc-toolchain="$TC" -isystem "$TC/include/c++/15" \
  -isystem "$TC/include/x86_64-linux-gnu/c++/15" \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 "$OUT/helper.cpp" \
  -L"$TORCH/lib" -L"$VENV/lib" -ltorch_xpu -ltorch_cpu -lc10_xpu -lc10 -lsycl \
  -Wl,--version-script,"$OUT/helper.map" -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$VENV/lib" -Wl,-rpath,"$ONEAPI/lib" \
  -o "$OUT/libx5_eventless_helper.so"
# Localize only replay: its wrapper must not recurse into the preload stub.
# Do not use Bsymbolic: SYCL's inline/weak runtime symbols must interpose normally.
readelf -d "$OUT/libx5_eventless_stub.so" | rg NEEDED
sha256sum "$OUT"/*.so
echo X5_LAZY_EVENTLESS_BUILD_OK
