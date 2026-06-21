#!/usr/bin/env bash
# Dual-B70 TP=2 perf campaign. Serve Qwen3.6-27B int4 AutoRound (=w4a16) at TP=2 and sweep concurrency,
# to compare against the BANKED single-card (TP=1) captured curve:
#   TP=1: C1 28.1agg/30.9per, C2 52.0/29.3, C4 87.8/26.7, C8 134.3/21.7, C16 178.3/14.5, C32 216.7/8.4, C64 234.7/6.7
# Tries captured (GRAPH=1, the vLLM #41663 stable stack includes XPU graph ON) first; eager fallback.
# Wrap in ONE gpu-run lease:  ./gpu-run bash 58_tp2_campaign.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
SERVE=./30_serve_w4a8_graph.sh
SWEEP=./35_sweep_bench.sh
MODELC=/models/Lorbus_Qwen3.6-27B-int4-AutoRound
SERVED=qwen36-27b-int4-tp2
NAME=vllm_w4a8
CONC="${CONC:-1 2 4 8 16 32 64}"

stop_srv(){ docker stop "$NAME" >/dev/null 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1 || true; }
linkstate(){ echo "  [link] $(date +%T) 0a=$(cat /sys/bus/pci/devices/0000:0a:00.0/current_link_speed 2>/dev/null|tr -d '\n')x$(cat /sys/bus/pci/devices/0000:0a:00.0/current_link_width 2>/dev/null|tr -d '\n')  44=$(cat /sys/bus/pci/devices/0000:44:00.0/current_link_speed 2>/dev/null|tr -d '\n')x$(cat /sys/bus/pci/devices/0000:44:00.0/current_link_width 2>/dev/null)"; }
serve(){ # $1=GRAPH
  stop_srv
  env IMG=vllm-xpu-env:v0230 MODEL=$MODELC SERVED=$SERVED GRAPH="$1" DTYPE=auto UTIL=0.92 \
      MAXLEN=8192 MAXSEQS=64 CAPSIZES=1,2,4,8,16,32,64 NOMM=1 TP=2 NAME=$NAME bash "$SERVE"
  curl -sf http://localhost:18080/health >/dev/null 2>&1
}
sweep(){ env NAME=$NAME MODEL=$SERVED LABEL="$1" TOKPATH=$MODELC CONC="$CONC" bash "$SWEEP"; }

echo "######## PCIe link BEFORE (idle) ########"; linkstate

echo "######## TP=2 CAPTURED attempt (GRAPH=1) ########"
MODE=FAILED
if serve 1; then
  curl -s http://localhost:18080/v1/models | grep -oE '"id":"[^"]*"' | head -1
  echo "  warmup request to ramp PCIe..."; curl -s http://localhost:18080/v1/completions -H 'Content-Type: application/json' -d '{"model":"'"$SERVED"'","prompt":"Count to twenty:","max_tokens":64}' >/dev/null 2>&1
  echo "######## PCIe link UNDER LOAD (just served, warmup) ########"; linkstate
  sweep "27b-w4a16-TP2-cap"; MODE=captured
else
  echo "!!! TP=2 captured did NOT serve -- last logs:"; docker logs $NAME 2>&1 | tail -25
  echo "######## TP=2 EAGER fallback (GRAPH=0) ########"
  if serve 0; then
    curl -s http://localhost:18080/v1/models | grep -oE '"id":"[^"]*"' | head -1
    sweep "27b-w4a16-TP2-eager"; MODE=eager
  else
    echo "!!! TP=2 eager ALSO failed"; docker logs $NAME 2>&1 | tail -25
  fi
fi
echo "######## PCIe link AFTER sweep ########"; linkstate
stop_srv
echo "######## TP=2 CAMPAIGN DONE (mode=$MODE) ########"
for f in $(ls -t "$ROOT"/results/sweep_27b-w4a16-TP2-*.csv 2>/dev/null | head -2); do echo "== $f =="; cat "$f"; done
