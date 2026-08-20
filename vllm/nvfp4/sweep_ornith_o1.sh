#!/usr/bin/env bash
# O1: LocalMaxxing-method C1 of the Ornith 34.9 GRAPH no-MTP hold.
# Restarts o4d (32.2 exists) onto STICKY=0 M1=0 M1_KERNEL=0. Measurement only.
# Run under: ./bin/gpu-run --card 0 bash vllm/nvfp4/sweep_ornith_o1.sh
# P2P=0. No DD. Do not demote 34.9.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
SERVE="$REPO/vllm/nvfp4/serve_nvfp4_moe_35b.sh"
PROBE="$REPO/vllm/nvfp4/g1_probe.py"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR=/mnt/vm_8tb/b70/lmx_overnight
STATUS=$LOGDIR/STATUS
HOST=192.168.10.5
PORT=18080
NAME=ornith_o1
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
unset B70_NVFP4_M1_SO_HOST || true
export SERVED=ornith-1.5-35b-A3B-NVFP4-o1-graph
export B70_EXTRA_ENV="B70_FP8_CHANNEL_INT8XMX=0 B70_NVFP4_F8_SCALE_M_MAX=8 B70_NVFP4_MOE_STICKY=0 B70_NVFP4_MOE_M1=0 B70_NVFP4_MOE_M1_KERNEL=0"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOGDIR/o1_${STAMP}.log"; }
st() { echo "$(date -u +%Y-%m-%dT%H%MZ) O1_STATUS=$*" | tee -a "$STATUS"; }

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
  docker logs "$NAME" >"$LOGDIR/o1_docker_${STAMP}.log" 2>&1 || true
  tail -80 "$LOGDIR/o1_docker_${STAMP}.log" | tee -a "$LOGDIR/o1_${STAMP}.log" || true
  docker rm -f "$NAME" ornith_o4d >/dev/null 2>&1 || true
  exit 2
}

st "START stamp=$STAMP graph=1 m1k=0 no-mtp hold-recipe"
log "stop leftover ornith_o4d/o1"
docker rm -f ornith_o4d ornith_o1 >/dev/null 2>&1 || true
NAME=ornith_o4d bash "$SERVE" stop >/dev/null 2>&1 || true
NAME=ornith_o1 bash "$SERVE" stop >/dev/null 2>&1 || true
sleep 2

t0=$(date +%s)
log "start $NAME SERVED=$SERVED"
NAME="$NAME" PORT="$PORT" IMG="$IMG" bash "$SERVE" \
  >"$LOGDIR/o1_serve_${STAMP}.log" 2>&1 || bootfail serve_rc
st "SERVE_STARTED log=$LOGDIR/o1_serve_${STAMP}.log"

if ! wait_healthy; then
  bootfail not_healthy
fi
hs=$(( $(date +%s) - t0 ))
st "SERVE_OK healthy=${hs}s"
log "HEALTHY ${hs}s"

docker logs "$NAME" >"$LOGDIR/o1_docker_pre_${STAMP}.log" 2>&1 || true
if grep -q 'm1k=1' "$LOGDIR/o1_docker_pre_${STAMP}.log"; then
  log "m1k=1 leaked into hold recipe"
  bootfail m1k_on
fi
grep -E 'nvfp4-shim \(7\)|m1_gemv' "$LOGDIR/o1_docker_pre_${STAMP}.log" \
  | head -10 | tee -a "$LOGDIR/o1_${STAMP}.log" || true

log "G1"
python3 "$PROBE" "http://${HOST}:${PORT}/v1" auto \
  | tee "$LOGDIR/o1_g1_${STAMP}.log"
g1rc=$?
st "G1 rc=$g1rc log=$LOGDIR/o1_g1_${STAMP}.log"
[ "$g1rc" = 0 ] || bootfail g1

mid=$(curl -s "http://${HOST}:${PORT}/v1/models" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
log "phase_bench mid=$mid p512/g128 n=5"
python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
  --base "http://${HOST}:${PORT}" \
  --model "$mid" \
  --prompt-tokens 512 --gen-tokens 128 --n 5 \
  --label o1-hold \
  --out "$LOGDIR/o1_phase_${STAMP}.json" \
  | tee "$LOGDIR/o1_phase_${STAMP}.log"
prc=$?
st "BENCH rc=$prc out=$LOGDIR/o1_phase_${STAMP}.json"

# If entropy prompt EOS-short, force g128 on the live serve (same pick).
comp=$(python3 -c "import json; d=json.load(open('$LOGDIR/o1_phase_${STAMP}.json')); print(d.get('median_completion_tokens') or 0)" 2>/dev/null || echo 0)
if python3 -c "import sys; sys.exit(0 if float('${comp:-0}') < 100 else 1)"; then
  log "short completions ($comp); retry --ignore-eos"
  python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
    --base "http://${HOST}:${PORT}" \
    --model "$mid" \
    --prompt-tokens 512 --gen-tokens 128 --n 5 \
    --ignore-eos \
    --label o1-hold-ieos \
    --out "$LOGDIR/o1_phase_ieos_${STAMP}.json" \
    | tee "$LOGDIR/o1_phase_ieos_${STAMP}.log"
  st "BENCH_IEOS out=$LOGDIR/o1_phase_ieos_${STAMP}.json"
fi

log "bench_code hold-check"
python3 "$REPO/vllm/nvfp4/bench_code.py" \
  "http://${HOST}:${PORT}/v1" "$mid" 1 256 3 \
  | tee "$LOGDIR/o1_code_${STAMP}.txt"
st "BENCH_CODE out=$LOGDIR/o1_code_${STAMP}.txt mid=$mid"

st "DONE stamp=$STAMP healthy=${hs}s leave_up=$NAME"
log "DONE leave $NAME up"
exit 0
