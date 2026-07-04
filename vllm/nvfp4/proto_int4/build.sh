#!/usr/bin/env bash
# Compile-only (no GPU lease). Builds s8 control + s4 (+ s2) ESIMD DPAS probes
# AOT for Battlemage BMG-G31 and dumps IGC/vISA asm for precision-field evidence.
# Run: bash build.sh   (invokes the toolchain container itself)
set -o pipefail
IMG=vllm-xpu-env:int8g-v0240
SRC=/mnt/vm_8tb/github/b70_ai_things/vllm/nvfp4/proto_int4
OUT=/mnt/vm_8tb/b70/int4_dpas_build

mkdir -p "$OUT"
docker run --rm \
  -v "$SRC":/src -v "$OUT":/out \
  --entrypoint bash "$IMG" -c '
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
which icpx || { echo "NO icpx"; exit 2; }
icpx --version | head -1

build() {
  local prec=$1 name=$2
  echo "=================== BUILD PREC=$prec ($name) ==================="
  rm -rf /out/dump_$name; mkdir -p /out/dump_$name
  # AOT to BMG-G31, ESIMD. Dump IGC shader asm/vISA for the DPAS encoding.
  IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/out/dump_$name \
  icpx -fsycl -std=c++17 -O2 -DPREC=$prec \
    -fsycl-targets=intel_gpu_bmg_g31 \
    /src/int4_dpas.cpp -o /out/int4_dpas_$name 2> /out/build_$name.err
  local rc=$?
  echo "compile rc=$rc"
  if [ $rc -ne 0 ]; then echo "--- build errors (tail) ---"; tail -40 /out/build_$name.err; fi
  # find dpas in dumped asm
  echo "--- dpas mnemonics in dump_$name ---"
  grep -rIl "dpas" /out/dump_$name 2>/dev/null | head
  grep -rIhoE "dpas[.a-z0-9_]*" /out/dump_$name 2>/dev/null | sort | uniq -c | head -40
  return $rc
}

build 8 s8
build 4 s4
build 2 s2
echo "=== DONE ==="
ls -la /out/int4_dpas_* 2>/dev/null
'
