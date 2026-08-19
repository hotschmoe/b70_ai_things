#!/usr/bin/env bash
# L62: MTP via 5b XPUExperts map, then TP=2 GRAPH push-AR.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
SERVE="$REPO/vllm/nvfp4/serve_nvfp4_moe_35b.sh"
PROBE="$REPO/vllm/nvfp4/g1_probe.py"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR="$REPO/results/logs/ornith_nvfp4_l62_${STAMP}"
mkdir -p "$OUTDIR"
HOST=192.168.10.5 PORT=18080 NAME=ornith_nvfp4_sweep
export IMG=vllm-xpu-env:int8g-v0260 NAME PORT
export CKPT=/models/ornith-1.5-35b-a3b/nvfp4-modelopt
export MODE=fused KV_FP8=0 LANGONLY=1 MAXSEQS=8 MAXLEN=8192 UTIL=0.90
export P2PACCESS=0
export FUSED_SO=/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so
export GDN_LIB=/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/libgdn_attn_kernels_xe_2.so
export B70_EXTRA_ENV="B70_FP8_CHANNEL_INT8XMX=0 B70_NVFP4_F8_SCALE_M_MAX=8"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUTDIR/ledger.txt"; }

wait_healthy() {
  local i status
  for i in $(seq 1 180); do
    status=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
    if [ "$status" = "exited" ] || [ "$status" = "missing" ]; then return 1; fi
    if curl -sf --max-time 3 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}

run_one() {
  local arm="$1"
  export SERVED="ornith-1.5-35b-A3B-NVFP4-l62-${arm}"
  log "===== $arm TP=$TP GRAPH=$GRAPH MTP=${MTPTOK:-off} PUSH_AR=${PUSH_AR:-0}/${PUSH_AR_GRAPH:-0} ====="
  NAME="$NAME" bash "$SERVE" stop >/dev/null 2>&1 || true
  local t0 hs
  t0=$(date +%s)
  NAME="$NAME" PORT="$PORT" IMG="$IMG" bash "$SERVE" >"$OUTDIR/${arm}_boot.txt" 2>&1
  if ! wait_healthy; then
    docker logs "$NAME" >"$OUTDIR/${arm}_docker.log" 2>&1 || true
    grep -E 'nvfp4-shim|ValueError|AssertionError|event.wait|XPUExperts|5b' \
      "$OUTDIR/${arm}_docker.log" | tail -40 >"$OUTDIR/${arm}_err.txt" || true
    log "$arm BOOTFAIL"
    echo "$arm BOOTFAIL" >> "$OUTDIR/summary.txt"
    return 1
  fi
  hs=$(( $(date +%s) - t0 ))
  log "$arm HEALTHY ${hs}s"
  python3 "$PROBE" "http://${HOST}:${PORT}/v1" auto | tee "$OUTDIR/${arm}_g1.json"
  mid=$(curl -s "http://${HOST}:${PORT}/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
  python3 "$REPO/vllm/nvfp4/bench_code.py" \
    "http://${HOST}:${PORT}/v1" "$mid" 1 256 3 | tee "$OUTDIR/${arm}_code.txt"
  docker logs "$NAME" >"$OUTDIR/${arm}_docker.log" 2>&1 || true
  grep -E 'nvfp4-shim \(5b\)|XPU Unquantized|push-AR|PUSH_AR' \
    "$OUTDIR/${arm}_docker.log" | head -12 | tee -a "$OUTDIR/ledger.txt" || true
  echo "$arm HEALTHY=${hs}s" >> "$OUTDIR/summary.txt"
  return 0
}

export TP=1 CARD=0 GRAPH=0 CAPSIZES= MTPTOK=3 PUSH_AR=0 PUSH_AR_GRAPH=0
run_one mtp3e || true

if [ -f "$OUTDIR/mtp3e_g1.json" ] && grep -q '"ok": true' "$OUTDIR/mtp3e_g1.json"; then
  export GRAPH=1 CAPSIZES=1,2,4,8 MTPTOK=3
  run_one mtp3g || true
fi

export TP=2 GRAPH=1 CAPSIZES=1,2,4,8 MTPTOK= PUSH_AR=1 PUSH_AR_GRAPH=1
run_one t2push || true

NAME="$NAME" bash "$SERVE" stop >/dev/null 2>&1 || true
log "DONE $OUTDIR"
cat "$OUTDIR/summary.txt" 2>/dev/null || true
