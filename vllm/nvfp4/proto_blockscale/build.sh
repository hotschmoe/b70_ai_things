#!/usr/bin/env bash
# AOT-build the block-scaled DPAS kernels for BMG-G31 inside the toolchain container.
# Compile-only (no GPU). Also dumps IGC Gen-ISA/vISA asm to confirm native s8 DPAS.
# Usage: bash build.sh [src.cpp ...]   (default: all bs_dpas_*.cpp)
set -o pipefail
IMG=vllm-xpu-env:int8g-v0240
DIR=/mnt/vm_8tb/b70/nvfp4_blockscale
SRCS=${*:-$(cd "$DIR" && ls bs_dpas_*.cpp)}

docker run --rm -v "$DIR":/work --entrypoint bash "$IMG" -c '
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
which icpx >/dev/null || { echo "NO icpx"; exit 2; }
icpx --version | head -1
cd /work
for src in '"$SRCS"'; do
  name=$(basename "$src" .cpp)
  echo "=================== BUILD $name ==================="
  rm -rf dump_$name; mkdir -p dump_$name
  IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/work/dump_$name \
  icpx -fsycl -std=c++17 -O2 \
    -fsycl-targets=intel_gpu_bmg_g31 \
    "$src" -o "$name" 2> build_$name.err
  rc=$?
  echo "compile rc=$rc"
  if [ $rc -ne 0 ]; then echo "--- errors (tail) ---"; tail -40 build_$name.err; continue; fi
  echo "--- dpas mnemonics in dump_$name (native encoding evidence) ---"
  grep -rIhoE "dpas[.a-z0-9_]*" dump_$name 2>/dev/null | sort | uniq -c | head -20
done
echo "=== DONE ==="; ls -la /work/bs_dpas_* 2>/dev/null | grep -v "\.cpp\|\.err"
'
