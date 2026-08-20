#!/usr/bin/env bash
# W8b: k1bar rematch with PREFIXCACHE=1 B70_MRV2=0 (hold-faithful).
# LOOP 18 was PREFIXCACHE=0 which enabled MRV2 and got code c1 28.8 vs 31.9.
# Same ckpt/SO. P2P=0. No DD. Do not demote 31.9 on a miss.
# Run under: ./bin/gpu-run bash vllm/dflash/sweep_lmx_w8b_k1bar.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
SERVE="$REPO/vllm/dflash/serve_qwen38_w8a8_dspark.sh"
PROBE="$REPO/vllm/nvfp4/g1_probe.py"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR=/mnt/vm_8tb/b70/lmx_overnight
STATUS=$LOGDIR/STATUS
HOST=192.168.10.5
PORT=18080
NAME=qwen38_w8a8_dspark
mkdir -p "$LOGDIR"

export IMG=vllm-xpu-env:int8g-v0260
export NAME PORT
export TP=2 GRAPH=1 SPECTOK=4 MAXLEN=122880
export W8A16_M_MAX=0 PREFIXCACHE=1 B70_MRV2=0 KV_FP8=0
export P2PACCESS=0 PUSH_AR=1 PUSH_AR_GRAPH=1 CGRECLAIM=0
export GDN_SO=/mnt/vm_8tb/b70/w8a8_kernel_v0260_k1barrier/_xpu_C.abi3.so
export GDN_LIB=/mnt/vm_8tb/b70/w8a8_kernel_v0260_k1barrier/libgdn_attn_kernels_xe_2.so
export SERVED=qwen3.8-27b-W8A8-gptq-dspark4-k1bar-pc1
export B70_EXTRA_ENV="VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER=1 PUSH_AR_ALLGATHER_ASYNC=1"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOGDIR/w8b_${STAMP}.log"; }
st() { echo "$(date -u +%Y-%m-%dT%H%MZ) W8B_STATUS=$*" | tee -a "$STATUS"; }

bootfail() {
  local why="$1"
  st "BOOTFAIL $why"
  docker logs "$NAME" >"$LOGDIR/w8b_docker_${STAMP}.log" 2>&1 || true
  tail -80 "$LOGDIR/w8b_docker_${STAMP}.log" | tee -a "$LOGDIR/w8b_${STAMP}.log" || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit 2
}

[ -f "$GDN_SO" ] || { echo "MISSING $GDN_SO"; st "NO_SO"; exit 3; }

st "START stamp=$STAMP graph=1 spectok=4 pc=1 mrv2=0 barrier=1 p2p=0"
log "stop leftover $NAME"
docker rm -f "$NAME" >/dev/null 2>&1 || true
sleep 2

log "xpu-health both cards"
if ! "$REPO/bin/xpu-health" --img "$IMG" --timeout 90 \
    >"$LOGDIR/w8b_health_${STAMP}.log" 2>&1; then
  tail -40 "$LOGDIR/w8b_health_${STAMP}.log" | tee -a "$LOGDIR/w8b_${STAMP}.log"
  bootfail xpu_health
fi
st "HEALTH_OK"

t0=$(date +%s)
log "start $NAME SERVED=$SERVED PREFIXCACHE=1 B70_MRV2=0"
if ! bash "$SERVE" start >"$LOGDIR/w8b_serve_${STAMP}.log" 2>&1; then
  bootfail serve_rc
fi
hs=$(( $(date +%s) - t0 ))
st "SERVE_OK healthy=${hs}s"
log "HEALTHY ${hs}s"

if ! curl -sf --max-time 3 "http://${HOST}:${PORT}/v1/models" >/dev/null; then
  bootfail not_healthy
fi
if grep -q 'VLLM_USE_V2_MODEL_RUNNER=1' "$LOGDIR/w8b_serve_${STAMP}.log"; then
  log "WARN MRV2 leaked into hold-faithful arm"
fi

log "G1"
python3 "$PROBE" "http://${HOST}:${PORT}/v1" auto \
  | tee "$LOGDIR/w8b_g1_${STAMP}.log"
g1rc=$?
st "G1 rc=$g1rc log=$LOGDIR/w8b_g1_${STAMP}.log"
[ "$g1rc" = 0 ] || bootfail g1

mid=$(curl -s "http://${HOST}:${PORT}/v1/models" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
log "bench_code mid=$mid vs hold 31.9"
python3 "$REPO/vllm/nvfp4/bench_code.py" \
  "http://${HOST}:${PORT}/v1" "$mid" 1 256 3 \
  | tee "$LOGDIR/w8b_code_${STAMP}.txt"
st "BENCH_CODE out=$LOGDIR/w8b_code_${STAMP}.txt mid=$mid"

log "phase_bench p512/g128 n=5 --ignore-eos"
python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
  --base "http://${HOST}:${PORT}" \
  --model "$mid" \
  --prompt-tokens 512 --gen-tokens 128 --n 5 \
  --ignore-eos \
  --label w8b-k1bar-pc1 \
  --out "$LOGDIR/w8b_phase_${STAMP}.json" \
  | tee "$LOGDIR/w8b_phase_${STAMP}.log"
st "BENCH out=$LOGDIR/w8b_phase_${STAMP}.json"

st "DONE stamp=$STAMP healthy=${hs}s leave_up=$NAME"
log "DONE leave $NAME up"
exit 0
