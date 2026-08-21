#!/usr/bin/env bash
# Qwen3.8-27B W4A8 research serve (NOT a shelf). Clone of
# rdy_to_serve/vllm/qwen36-27b-w4a8/serve.sh via exec, 3.8 env overrides.
# Campaign: docs/20260820_qwen38_w4a8_campaign.md. GRAPH=0 first, Paris+fib.
# Served id must encode method+scheme. P2PACCESS=0. DD stays parked.
#
#   GRAPH=0 NOMM=1 B70_NOMTP=1 PORT=18081 NAME=qwen38_w4a8_rtn \
#     ./bin/gpu-run --card 1 bash vllm/w4a8/serve_qwen38_w4a8.sh start
#   bash vllm/w4a8/serve_qwen38_w4a8.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHELF="$REPO/rdy_to_serve/vllm/qwen36-27b-w4a8/serve.sh"
ACTION="${1:-start}"

if [ "$ACTION" = stop ]; then
  NAME="${NAME:-qwen38_w4a8_rtn}"
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

[ -f "$SHELF" ] || { echo "missing 3.6 shelf serve $SHELF"; exit 1; }

export CKPT="${CKPT:-/models/qwen3.8-27b/w4a8-rtn-gdn}"
export SERVED="${SERVED:-qwen3.8-27b-W4A8-rtn-gdn}"
export NAME="${NAME:-qwen38_w4a8_rtn}"
export PORT="${PORT:-18081}"
export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
export TP="${TP:-1}"
export DEVICE="${DEVICE:-${CARD:-1}}"
export GRAPH="${GRAPH:-0}"
export DTYPE="${DTYPE:-float16}"
export NOMM="${NOMM:-1}"
export P2PACCESS="${P2PACCESS:-0}"
export UTIL="${UTIL:-0.85}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-2}"
export B70_W4A8_HYBRID="${B70_W4A8_HYBRID:-0}"
# First smoke: no MTP. Path H hybrid is a later A/B; HYBRID=0 proves int4_gemm_w4a8.
export B70_NOMTP="${B70_NOMTP:-1}"
export MTPTOK="${MTPTOK:-}"
export P2PACCESS="${P2PACCESS:-0}"
# K15: 3.6 W4A8 shelf has no PUSH_AR block (it is TP=1). Inject via B70_EXTRA_*
# which the shelf APPENDs. GRAPH=0 -> PUSH_AR_GRAPH=0 (eager + captured-decode
# push is the !!!! trap). Never CCL_TOPO_P2P_ACCESS=1.
if [ "${TP:-1}" -gt 1 ] && [ "${PUSH_AR:-1}" = 1 ]; then
  _PAR="$REPO/vllm/contrib/vllm_push_allreduce"
  if [ "${GRAPH:-0}" = 1 ] && [ "${PUSH_AR_GRAPH:-1}" = 1 ]; then
    _SO="$_PAR/prebuilt/libxpu_push_ar_graph.so"
    export PUSH_AR_GRAPH=1
    export PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-0}"
  else
    _SO="$_PAR/prebuilt/libxpu_push_ar_torch.so"
    export PUSH_AR_GRAPH=0
    export PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-65536}"
  fi
  [ -f "$_SO" ] || { echo "[!] PUSH_AR=1 but missing $_SO (set PUSH_AR=0)" >&2; exit 1; }
  export B70_EXTRA_MOUNTS="${B70_EXTRA_MOUNTS:+$B70_EXTRA_MOUNTS }${_PAR}:/opt/push_ar:ro ${_SO}:/opt/push_ar_so/libxpu_push_ar_torch.so:ro"
  export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }PYTHONPATH=/opt/push_ar:/opt/mtp_shim PUSH_AR_CHAIN_SITECUSTOMIZE=/opt/mtp_shim/sitecustomize.py PUSH_AR_SO=/opt/push_ar_so/libxpu_push_ar_torch.so PUSH_AR_DISABLE=0 PUSH_AR_GRAPH=${PUSH_AR_GRAPH} PUSH_AR_MIN_NUMEL=${PUSH_AR_MIN_NUMEL}"
  echo "=== PUSH_AR overlay ON GRAPH=${PUSH_AR_GRAPH} MIN_NUMEL=${PUSH_AR_MIN_NUMEL} P2PACCESS=0 ==="
else
  echo "=== PUSH_AR overlay OFF (TP=$TP PUSH_AR=${PUSH_AR:-0}) ==="
fi

echo "=== 3.8 W4A8 research serve SERVED=$SERVED GRAPH=$GRAPH NOMM=$NOMM DEVICE=$DEVICE PORT=$PORT TP=$TP ==="
echo "=== IMG=$IMG CKPT=$CKPT P2PACCESS=$P2PACCESS HYBRID=$B70_W4A8_HYBRID NOMTP=$B70_NOMTP ==="
exec bash "$SHELF" "$ACTION"
