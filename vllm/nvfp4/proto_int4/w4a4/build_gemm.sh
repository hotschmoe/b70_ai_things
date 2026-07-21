#!/usr/bin/env bash
# Compile-only (no GPU lease). Builds the tiled s4s4 W4A4 GEMM microbench AOT for
# Battlemage BMG-G31 and dumps IGC/vISA asm to confirm native dpas.s4.s4 is
# emitted inside the tiled mainloop. Run: bash build_gemm.sh
# Optional: GEMM_M=1024 NSUB=4 bash build_gemm.sh
set -o pipefail
IMG=${IMG:-vllm-xpu-env:int8g-v0240}
SRC=/mnt/vm_8tb/github/b70_ai_things/vllm/nvfp4/proto_int4/w4a4
OUT=/mnt/vm_8tb/b70/int4_dpas_build/w4a4
GEMM_M=${GEMM_M:-512}
NSUB=${NSUB:-4}

mkdir -p "$OUT"
docker run --rm \
  -e GEMM_M="$GEMM_M" -e NSUB="$NSUB" \
  -v "$SRC":/src -v "$OUT":/out \
  --entrypoint bash "$IMG" -c '
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
which icpx || { echo "NO icpx"; exit 2; }
icpx --version | head -1
echo "=== BUILD s4s4 GEMM  M=$GEMM_M NSUB=$NSUB ==="
rm -rf /out/dump_gemm; mkdir -p /out/dump_gemm
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/out/dump_gemm \
icpx -fsycl -std=c++17 -O2 -DGEMM_M=$GEMM_M -DNSUB=$NSUB \
  -fsycl-targets=intel_gpu_bmg_g31 \
  /src/s4s4_gemm_microbench.cpp -o /out/s4s4_gemm 2> /out/build_gemm.err
rc=$?
echo "compile rc=$rc"
if [ $rc -ne 0 ]; then echo "--- build errors (tail) ---"; tail -50 /out/build_gemm.err; exit $rc; fi
echo "--- native dpas mnemonics emitted in the mainloop ---"
grep -rIhoE "dpas[.a-z0-9_]*" /out/dump_gemm 2>/dev/null | sort | uniq -c | head
echo "--- s4 register operands present? (expect r..:s4) ---"
grep -rIhE "dpas.*:s4" /out/dump_gemm 2>/dev/null | head -3
ls -la /out/s4s4_gemm
echo "=== DONE (run under: gpu-run --card 1 /out/s4s4_gemm) ==="
'
