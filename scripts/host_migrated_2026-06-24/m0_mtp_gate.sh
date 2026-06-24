#!/usr/bin/env bash
# M0 -- MTP wiring sanity gate (MTP_TODO). Serve 27B + PIECEWISE graph + MTP spec, wrapped so the gpu-run
# flock is held for serve+probe+stop (clean GPU before/after). Default vehicle = Lorbus W4A16 int4-AutoRound:
# the most-proven v0230+GDN serve AND the exact Lorbus 45.2 t/s precedent -> isolates MTP wiring from the
# W4A8 :int8g/KERNEL_SO/prepack confounders. Override MODEL/SERVED/IMG/KERNEL_SO/PREPACK for the W4A8 variant.
# Q: does v0230 load the MTP head, draft, and NOT crash on the GDN + speculative-mask path?
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
SPECTOK="${SPECTOK:-3}"
IMG="${IMG:-vllm-xpu-env:v0230}"
MODEL="${MODEL:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"
SERVED="${SERVED:-qwen36-27b-int4}"
METHOD="${METHOD:-mtp}"
GRAPHV="${GRAPHV:-1}"
LOGF="$ROOT/results/m0_mtp_${SERVED}_spec${SPECTOK}_g${GRAPHV}.log"
mkdir -p "$ROOT/results"
echo "=== M0: $SERVED  IMG=$IMG  spec($METHOD)=$SPECTOK  GRAPH=$GRAPHV ==="
# COMPILESZ= (set empty) -> omit compile_sizes (avoids the spec-pad '[1] would be padded' rejection).
env IMG="$IMG" MODEL="$MODEL" SERVED="$SERVED" \
    GRAPH="$GRAPHV" DTYPE=auto UTIL=0.92 MAXLEN=8192 MAXSEQS=8 \
    CAPSIZES=1,2,4,8 COMPILESZ= NOMM=1 NAME=vllm_m0 \
    ${KERNEL_SO:+KERNEL_SO="$KERNEL_SO"} ${PREPACK:+PREPACK="$PREPACK"} ${KVDTYPE:+KVDTYPE="$KVDTYPE"} \
    SPEC="{\"method\":\"$METHOD\",\"num_speculative_tokens\":$SPECTOK}" \
    bash ./30_serve_w4a8_graph.sh 2>&1 | tee "$LOGF"
if curl -sf http://localhost:18080/health >/dev/null 2>&1; then
  echo "=== M0 HEALTHY -- gen probe (greedy, 128 tok) ==="
  curl -s --max-time 90 http://localhost:18080/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"Write a Python function that returns the nth Fibonacci number, with a short docstring.\",\"max_tokens\":128,\"temperature\":0}" | head -c 900; echo
  echo "--- spec-decode / accept-length / draft log signals ---"
  docker logs vllm_m0 2>&1 | grep -iE "spec|draft|accept|mtp|num_spec|speculat|reject|propos" | grep -viE "respect|prospect" | tail -30
  echo "M0_VERDICT=PASS"
else
  echo "=== M0 NOT HEALTHY -- crash signature ==="
  docker logs vllm_m0 2>&1 | grep -iE "error|traceback|notimplement|spec_sequence|gdn|assert|no attribute|KeyError|raise|Unsupported|topk|moe_C" | tail -45
  echo "--- last 25 raw ---"; docker logs vllm_m0 2>&1 | tail -25
  echo "M0_VERDICT=FAIL"
fi
docker stop vllm_m0 2>/dev/null || true
echo "=== M0 done; log $LOGF ==="
