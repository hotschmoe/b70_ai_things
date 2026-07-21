#!/usr/bin/env bash
# AOT-build the dot32+correction block-scaled DPAS kernel for BMG-G31, all 3 modes.
# COMPILE-ONLY (no GPU touch). Dumps IGC Gen-ISA to confirm native s8 DPAS.
# Usage: bash build.sh          (builds DOT32, BS, PC)
# The .cpp is copied into the git-ignored runtime dir so the container mount matches
# the existing proto_blockscale build layout.
set -o pipefail
IMG=vllm-xpu-env:int8g-v0240
REPO=/mnt/vm_8tb/github/b70_ai_things/vllm/nvfp4/proto_blockscale/dot32
DIR=/mnt/vm_8tb/b70/nvfp4_blockscale/dot32
mkdir -p "$DIR"
cp "$REPO/bs_dpas_dot32.cpp" "$DIR/"

docker run --rm -v "$DIR":/work --entrypoint bash "$IMG" -c '
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
which icpx >/dev/null || { echo "NO icpx"; exit 2; }
icpx --version | head -1
cd /work
build_one() {
  local km=$1 name=$2
  echo "=================== BUILD $name (KMODE=$km) ==================="
  rm -rf dump_$name; mkdir -p dump_$name
  IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/work/dump_$name \
  icpx -fsycl -std=c++17 -O2 -DKMODE=$km \
    -fsycl-targets=intel_gpu_bmg_g31 \
    bs_dpas_dot32.cpp -o "$name" 2> build_$name.err
  local rc=$?
  echo "compile rc=$rc"
  if [ $rc -ne 0 ]; then echo "--- errors (tail) ---"; tail -40 build_$name.err; return; fi
  echo "--- dpas mnemonics (native encoding evidence) ---"
  grep -rIhoE "dpas[.a-z0-9_]*" dump_$name 2>/dev/null | sort | uniq -c | head
}
build_one 32 dot32
build_one 16 bs
build_one 0  pc
echo "=== DONE ==="; ls -la /work/{dot32,bs,pc} 2>/dev/null
'
