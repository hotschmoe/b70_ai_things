#!/usr/bin/env bash
# K1 isolated kernel matrix on Qwen3.8-27B shapes. Card 1 only.
# Campaign: docs/20260820_qwen38_w4a8_campaign.md section 8.2.
# Does not need the 3.8 W4A8 file. 3.6 w4a8-sqgptq is the packed-int4 stand-in.
# P2PACCESS left unset (0). Do not start DD.
# int8g-v0260 image Entrypoint is leftover `sleep` -- always --entrypoint bash.
set -uo pipefail
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
CARD="${CARD:-1}"
SO="${B70_XPU_C_SO:-$ROOT/w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG:-$REPO/results/logs/k1_w4a8_shapes_${TS}.log}"
CSV="${CSV:-$REPO/results/logs/k1_w4a8_shapes_${TS}.csv}"
mkdir -p "$REPO/results/logs"
echo "=== K1 matrix card=$CARD img=$IMG so=$SO ===" | tee "$LOG"
echo "=== log=$LOG csv=$CSV ===" | tee -a "$LOG"

docker rm -f k1_w4a8_shapes 2>/dev/null || true
docker run --rm --name k1_w4a8_shapes \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  --entrypoint bash \
  -e CUDA_VISIBLE_DEVICES="" \
  -e ZE_AFFINITY_MASK="$CARD" \
  -e OMP_NUM_THREADS=8 \
  -e B70_XPU_C_SO="$SO" \
  -e CKPT=/models/qwen3.6-27b/w4a8-sqgptq \
  -e OUT_CSV="$CSV" \
  -e ONLY_MS="${ONLY_MS:-}" \
  -e ONLY_SHAPES="${ONLY_SHAPES:-}" \
  -e INCLUDE_LMHEAD="${INCLUDE_LMHEAD:-0}" \
  -e GROUP="${GROUP:-128}" \
  -v "$ROOT:$ROOT" \
  -v "$REPO:$REPO" \
  -v "$REPO/models/files:/models" \
  "$IMG" \
  -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
    python3 /mnt/vm_8tb/github/b70_ai_things/vllm/w4a8/bench_w4a8_shapes.py' \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== K1 python matrix rc=$RC ===" | tee -a "$LOG"

# Path S microbench (item 6): native s4 DPAS tile vs s8 control. Not a serve.
# Binaries already AOT at /mnt/vm_8tb/b70/int4_dpas_build from proto_int4.
echo "=== K1 Path S proto_int4 benches (card $CARD) ===" | tee -a "$LOG"
if [ -x "$ROOT/int4_dpas_build/bench_s4" ]; then
  docker rm -f k1_w4a8_paths 2>/dev/null || true
  docker run --rm --name k1_w4a8_paths \
    --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --entrypoint bash \
    -e CUDA_VISIBLE_DEVICES="" \
    -e ZE_AFFINITY_MASK="$CARD" \
    -v "$ROOT/int4_dpas_build:/out:ro" \
    vllm-xpu-env:int8g-v0240 \
    -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
      for n in s8 s4 s2; do echo "==== bench_$n ===="; /out/bench_$n || echo FAIL_$n; done' \
    2>&1 | tee -a "$LOG" || echo "=== Path S docker failed (non-fatal for K1 table) ===" | tee -a "$LOG"
else
  echo "=== Path S binaries missing at $ROOT/int4_dpas_build -- skip ===" | tee -a "$LOG"
fi

if grep -q "DONE_K1_MATRIX" "$LOG"; then
  echo "=== K1 DONE log=$LOG csv=$CSV ===" | tee -a "$LOG"
  exit 0
fi
echo "=== K1 python matrix did not print DONE_K1_MATRIX rc=$RC log=$LOG ===" | tee -a "$LOG"
exit "${RC:-1}"
