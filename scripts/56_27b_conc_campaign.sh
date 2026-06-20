#!/usr/bin/env bash
# 27B (Qwen3.6-27B int4 AutoRound = w4a16) CAPTURED concurrency campaign on the B70.
# Two configs, both fp16 KV + PIECEWISE graph capture (batch sizes 1..64 captured):
#   A) normal ctx (8k)   -> concurrency sweep
#   B) BIG ctx fp16 KV   -> tries 256k, falls back to 128k if KV won't fit
# Runs ON the GPU host (/mnt/vm_8tb/b70). Wrap the WHOLE thing in one gpu-run lease:
#   ./gpu-run bash scripts/56_27b_conc_campaign.sh   (or from host root: ./gpu-run bash 56_27b_conc_campaign.sh)
# See docs/SERVING.md for the canonical serve recipe this encodes.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
cd "$ROOT"
SERVE=./30_serve_w4a8_graph.sh
SWEEP=./35_sweep_bench.sh
MODELC=/models/Lorbus_Qwen3.6-27B-int4-AutoRound   # container path
SERVED=qwen36-27b-int4
COMMON="IMG=vllm-xpu-env:v0230 MODEL=$MODELC SERVED=$SERVED GRAPH=1 DTYPE=auto UTIL=0.92 MAXSEQS=64 CAPSIZES=1,2,4,8,16,32,64 NOMM=1 NAME=vllm_w4a8"

stop_srv () { docker stop vllm_w4a8 >/dev/null 2>&1 || true; docker rm -f vllm_w4a8 >/dev/null 2>&1 || true; }
serve_ok () { local MAXLEN="$1"; stop_srv; echo ">>> SERVE MAXLEN=$MAXLEN fp16-KV captured"; env $COMMON MAXLEN="$MAXLEN" bash "$SERVE"; curl -sf http://localhost:18080/health >/dev/null 2>&1; }
sweep () { local LABEL="$1"; echo ">>> SWEEP $LABEL  CONC=1 2 4 8 16 32 64"; env NAME=vllm_w4a8 MODEL=$SERVED LABEL="$LABEL" TOKPATH=$MODELC CONC="1 2 4 8 16 32 64" bash "$SWEEP"; }

echo "############ verify served-model id (CLAUDE.md rule) after each serve ############"

echo "############ CONFIG A: normal ctx (8k), fp16 KV, captured ############"
if serve_ok 8192; then
  curl -s http://localhost:18080/v1/models | grep -oE '"id":"[^"]*"' | head -1
  sweep "27b-w4a16-cap-ctx8k-fp16kv"
else
  echo "!!! CONFIG A serve FAILED"; docker logs vllm_w4a8 2>&1 | tail -40
fi
stop_srv

echo "############ CONFIG B: BIG ctx, fp16 KV, captured (256k -> 128k fallback) ############"
BIG_OK=0
for ML in 262144 131072; do
  if serve_ok "$ML"; then
    echo ">>> BIG ctx SERVED at MAXLEN=$ML (fp16 KV)"
    curl -s http://localhost:18080/v1/models | grep -oE '"id":"[^"]*"' | head -1
    sweep "27b-w4a16-cap-ctx${ML}-fp16kv"
    BIG_OK=1; stop_srv; break
  else
    echo "!!! MAXLEN=$ML did NOT serve (KV likely won't fit at fp16); trying next"
    docker logs vllm_w4a8 2>&1 | grep -iE "max.*model.*len|kv cache|can.?t fit|out of|memory|too large|ValueError|Error" | tail -15
    stop_srv
  fi
done
[ "$BIG_OK" = 0 ] && echo "!!! BIG ctx did NOT fit at fp16 KV even at 128k -- would need fp8 KV (KVDTYPE=fp8_e5m2)"

echo "############ CAMPAIGN COMPLETE ############"
for f in $(ls -t "$ROOT"/results/sweep_27b-w4a16-cap-*.csv 2>/dev/null | head -3); do echo "== $f =="; cat "$f"; done
