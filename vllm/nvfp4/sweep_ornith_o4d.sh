#!/usr/bin/env bash
# O4d: GRAPH no-MTP Ornith NVFP4 + sidecar M1_KERNEL vs hold 34.9.
# Same ckpt as o3. Stops o3 (bench 21.1 exists). Leaves ornith_o4d up on GO.
# Run under: ./bin/gpu-run --card 0 bash vllm/nvfp4/sweep_ornith_o4d.sh
# P2P=0. No DD. M1 python/STICKY stay 0. Do not demote 34.9 on a miss.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
SERVE="$REPO/vllm/nvfp4/serve_nvfp4_moe_35b.sh"
PROBE="$REPO/vllm/nvfp4/g1_probe.py"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR=/mnt/vm_8tb/b70/lmx_overnight
STATUS=$LOGDIR/STATUS
HOST=192.168.10.5
PORT=18080
NAME=ornith_o4d
SO_HOST=/mnt/vm_8tb/b70/lmx_overnight/o4_m1/b70_nvfp4_m1_gemv.so
mkdir -p "$LOGDIR"

export IMG=vllm-xpu-env:int8g-v0260
export NAME PORT
export CKPT=/models/ornith-1.5-35b-a3b/nvfp4-modelopt
export MODE=fused KV_FP8=0 LANGONLY=1 MAXSEQS=8 MAXLEN=8192 UTIL=0.90
export TP=1 CARD=0 GRAPH=1 CAPSIZES=1,2,4,8
export MTPTOK=
export P2PACCESS=0 PUSH_AR=0
export FUSED_SO=/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so
export GDN_LIB=/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/libgdn_attn_kernels_xe_2.so
export B70_NVFP4_M1_SO_HOST="$SO_HOST"
export SERVED=ornith-1.5-35b-A3B-NVFP4-o4d-graph-m1k
export B70_EXTRA_ENV="B70_FP8_CHANNEL_INT8XMX=0 B70_NVFP4_F8_SCALE_M_MAX=8 B70_NVFP4_MOE_STICKY=0 B70_NVFP4_MOE_M1=0 B70_NVFP4_MOE_M1_KERNEL=1 B70_NVFP4_M1_SO=/opt/nvfp4_m1/b70_nvfp4_m1_gemv.so"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOGDIR/o4d_${STAMP}.log"; }
st() { echo "$(date -u +%Y-%m-%dT%H%MZ) O4D_STATUS=$*" | tee -a "$STATUS"; }

wait_healthy() {
  local i status
  for i in $(seq 1 180); do
    status=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
    if [ "$status" = "exited" ] || [ "$status" = "missing" ]; then return 1; fi
    if curl -sf --max-time 3 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

bootfail() {
  local why="$1"
  st "BOOTFAIL $why"
  docker logs "$NAME" >"$LOGDIR/o4d_docker_${STAMP}.log" 2>&1 || true
  tail -80 "$LOGDIR/o4d_docker_${STAMP}.log" | tee -a "$LOGDIR/o4d_${STAMP}.log" || true
  docker rm -f "$NAME" ornith_o3 >/dev/null 2>&1 || true
  exit 2
}

[ -f "$SO_HOST" ] || { echo "MISSING $SO_HOST"; st "NO_SO"; exit 3; }

st "START stamp=$STAMP graph=1 m1k=1 no-mtp"
log "stop leftover ornith_o3/o4d"
docker rm -f ornith_o3 ornith_o4d >/dev/null 2>&1 || true
NAME=ornith_o3 bash "$SERVE" stop >/dev/null 2>&1 || true
NAME=ornith_o4d bash "$SERVE" stop >/dev/null 2>&1 || true
sleep 2

t0=$(date +%s)
log "start $NAME SERVED=$SERVED"
NAME="$NAME" PORT="$PORT" IMG="$IMG" bash "$SERVE" \
  >"$LOGDIR/o4d_serve_${STAMP}.log" 2>&1 || bootfail serve_rc
st "SERVE_STARTED log=$LOGDIR/o4d_serve_${STAMP}.log"

if ! wait_healthy; then
  bootfail not_healthy
fi
hs=$(( $(date +%s) - t0 ))
st "SERVE_OK healthy=${hs}s"
log "HEALTHY ${hs}s"

docker logs "$NAME" >"$LOGDIR/o4d_docker_pre_${STAMP}.log" 2>&1 || true
if ! grep -q 'm1k=1' "$LOGDIR/o4d_docker_pre_${STAMP}.log"; then
  log "m1k=1 missing in shim install"
  grep -E 'nvfp4-shim|m1_gemv|load_library' "$LOGDIR/o4d_docker_pre_${STAMP}.log" \
    | tail -20 | tee -a "$LOGDIR/o4d_${STAMP}.log" || true
  bootfail m1k_not_loaded
fi
if ! grep -q 'm1_gemv dispatch' "$LOGDIR/o4d_docker_pre_${STAMP}.log"; then
  log "WARN no m1_gemv dispatch yet (may appear at first decode)"
fi
grep -E 'nvfp4-shim \(7\)|m1_gemv' "$LOGDIR/o4d_docker_pre_${STAMP}.log" \
  | head -20 | tee -a "$LOGDIR/o4d_${STAMP}.log" || true

log "G1"
python3 "$PROBE" "http://${HOST}:${PORT}/v1" auto \
  | tee "$LOGDIR/o4d_g1_${STAMP}.log"
g1rc=$?
st "G1 rc=$g1rc log=$LOGDIR/o4d_g1_${STAMP}.log"
[ "$g1rc" = 0 ] || bootfail g1

mid=$(curl -s "http://${HOST}:${PORT}/v1/models" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
log "bench_code mid=$mid"
python3 "$REPO/vllm/nvfp4/bench_code.py" \
  "http://${HOST}:${PORT}/v1" "$mid" 1 256 3 \
  | tee "$LOGDIR/o4d_code_${STAMP}.txt"
st "BENCH_CODE out=$LOGDIR/o4d_code_${STAMP}.txt mid=$mid"

docker logs "$NAME" >"$LOGDIR/o4d_docker_${STAMP}.log" 2>&1 || true
grep -E 'nvfp4-shim \(7\)|m1_gemv' "$LOGDIR/o4d_docker_${STAMP}.log" \
  | head -20 | tee -a "$LOGDIR/o4d_${STAMP}.log" || true
st "DONE stamp=$STAMP healthy=${hs}s leave_up=$NAME"
log "DONE leave $NAME up"
exit 0
