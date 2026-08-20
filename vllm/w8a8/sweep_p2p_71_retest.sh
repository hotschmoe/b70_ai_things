#!/usr/bin/env bash
# 7.1 P2P-in-vLLM-TP2 retest. Operator override 2026-08-20: kernel 7.1
# cured the GuC wedge; H.13 oneCCL P2P-in-serve was never retested.
# ONE attempt. I_KNOW_P2P_WEDGES=1. Do not chain a second P2P start
# without xpu-health GO (and xe-reset if not).
#
# Repro is H.13: vLLM TP=2 warmup all_reduce with CCL_TOPO_P2P_ACCESS=1.
# GRAPH=0 NOMTP so we hit warmup fast. G1 + short phase_bench if it lives.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
LOGDIR="${LOGDIR:-$ROOT/lmx_overnight}"
mkdir -p "$LOGDIR"
NAME="${NAME:-qwen38_w8a8_p2p71}"
PORT="${PORT:-18080}"
STATUS="$LOGDIR/STATUS"
PEER_LOG="$LOGDIR/p2p71_peer_${STAMP}.log"
SERVE_LOG="$LOGDIR/p2p71_serve_${STAMP}.log"
G1_LOG="$LOGDIR/p2p71_g1_${STAMP}.log"
BENCH_OUT="$LOGDIR/p2p71_phase_${STAMP}.json"
HEALTH_PRE="$LOGDIR/p2p71_health_pre_${STAMP}.log"
HEALTH_POST="$LOGDIR/p2p71_health_post_${STAMP}.log"

set_status(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H%MZ)" "$*" | tee -a "$STATUS"; }
cd "$REPO"

# Graceful stop of any leftover (Pliny TP2 wait holder). docker stop not rm -f first.
for n in qwen38_oblit_q8 lmx_w1_d38 qwen38_w8a8 "$NAME"; do
  docker stop -t 30 "$n" >/dev/null 2>&1 || true
  docker rm -f "$n" >/dev/null 2>&1 || true
done
# Drop docker-wait lease holders if they survived.
pkill -f "docker wait qwen38_oblit_q8" >/dev/null 2>&1 || true
pkill -f "docker wait lmx_w1_d38" >/dev/null 2>&1 || true
sleep 2

set_status "P2P71_STATUS=PREFLIGHT kernel=$(uname -r)"
./bin/xpu-health >"$HEALTH_PRE" 2>&1
hc=$?
tail -8 "$HEALTH_PRE"
if [ "$hc" -ne 0 ]; then
  set_status "P2P71_STATUS=UNHEALTHY_PRE rc=$hc"
  exit 10
fi

set_status "P2P71_STATUS=PEER_PROBE"
B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run bash -c '
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
    -v "'"$REPO"'/vllm/w8a8/peer_probe.py:/peer_probe.py:ro" \
    -e ZE_FLAT_DEVICE_HIERARCHY=FLAT \
    --entrypoint python \
    vllm-xpu-env:int8g-v0260 \
    /peer_probe.py
' >"$PEER_LOG" 2>&1
prc=$?
cat "$PEER_LOG"
if [ "$prc" -ne 0 ]; then
  set_status "P2P71_STATUS=PEER_FAIL rc=$prc"
  exit 11
fi

set_status "P2P71_STATUS=SERVE_P2P1"
# ONE try. GRAPH=0 so warmup all_reduce is the first real collective.
I_KNOW_P2P_WEDGES=1 P2PACCESS=1 TP=2 GRAPH=0 B70_NOMTP=1 \
  NAME="$NAME" PORT="$PORT" \
  SERVED=qwen3.8-27b-W8A8-gptq-p2p71 \
  MAXLEN=16384 MAXSEQS=4 \
  B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run bash "$REPO/vllm/w8a8/serve_qwen38_27b.sh" start \
  >"$SERVE_LOG" 2>&1
src=$?
tail -40 "$SERVE_LOG"
if [ "$src" -ne 0 ]; then
  set_status "P2P71_STATUS=SERVE_FAIL rc=$src -- DO NOT retry P2P without health+reset"
  docker logs "$NAME" 2>&1 | tail -80 >"$LOGDIR/p2p71_serve_tail_${STAMP}.log" || true
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  ./bin/xpu-health >"$HEALTH_POST" 2>&1 || true
  tail -20 "$HEALTH_POST"
  exit 4
fi
set_status "P2P71_STATUS=SERVE_OK"

python3 "$REPO/vllm/nvfp4/g1_probe.py" "http://127.0.0.1:${PORT}/v1" auto | tee "$G1_LOG"
g1=${PIPESTATUS[0]}
set_status "P2P71_STATUS=G1 rc=$g1"
if [ "$g1" -ne 0 ]; then
  echo "G1 FAIL -- stop serve, health, do not chain P2P"
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  ./bin/xpu-health >"$HEALTH_POST" 2>&1 || true
  exit 5
fi

python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
  --base "http://127.0.0.1:${PORT}" \
  --model qwen3.8-27b-W8A8-gptq-p2p71 \
  --prompt-tokens 512 --gen-tokens 128 --n 3 \
  --out "$BENCH_OUT"
brc=$?
set_status "P2P71_STATUS=BENCH rc=$brc out=$BENCH_OUT"

# Leave up if GO so a later fire can A/B P2P=0 without a second P2P start.
if [ "${STOP:-0}" = 1 ]; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  ./bin/xpu-health >"$HEALTH_POST" 2>&1 || true
  set_status "P2P71_STATUS=STOPPED health_post=$?"
fi
exit "$brc"
