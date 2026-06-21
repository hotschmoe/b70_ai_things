#!/usr/bin/env bash
# DATA-PARALLEL 2-replica bench: the no-comms dual-GPU serving play. One 27B int4 (=w4a16) captured
# replica per B70 (card0 -> :18080, card1 -> :18081), independent, ZERO inter-GPU traffic. Measures:
#   (1) dp0 SOLO curve (dp1 idle)            = clean single-card baseline on THIS run
#   (2) dp0 + dp1 CONCURRENT                  = true 2-card aggregate; sum the out_tok_s
# DP scaling = (dp0 under concurrent load) / (dp0 solo). ~1.0 => perfect scaling (~2x aggregate, no contention).
# Wrap the WHOLE session in ONE gpu-run lease (holds both cards for its lifetime):
#   B70_GPU_LOCK_TIMEOUT=0 ./gpu-run bash 64_dataparallel_2rep.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:v0230
MODELC=/models/Lorbus_Qwen3.6-27B-int4-AutoRound
SERVE=./30_serve_w4a8_graph.sh
SWEEP=./35_sweep_bench.sh
CONC="${CONC:-1 8 32 64}"
COMMON=(IMG=$IMG MODEL=$MODELC GRAPH=1 DTYPE=auto UTIL=0.92 MAXLEN=8192 MAXSEQS=64 CAPSIZES=1,2,4,8,16,32,64 NOMM=1)

stop(){ docker rm -f vllm_dp0 vllm_dp1 2>/dev/null || true; }
trap stop EXIT
stop

echo "######## DATA-PARALLEL 2-REPLICA BENCH (27B int4 captured) $(date +%T) ########"
echo "== serve replica 0 -> card 0 :18080 =="
env "${COMMON[@]}" SERVED=qwen36-27b-int4-dp0 DEVICE=0 PORT=18080 NAME=vllm_dp0 bash "$SERVE"
echo "== serve replica 1 -> card 1 :18081 =="
env "${COMMON[@]}" SERVED=qwen36-27b-int4-dp1 DEVICE=1 PORT=18081 NAME=vllm_dp1 bash "$SERVE"

echo "== health + served-id check (CLAUDE.md rule) =="
curl -sf http://localhost:18080/health >/dev/null && echo "  dp0 :18080 HEALTHY" || { echo "  dp0 NOT HEALTHY"; docker logs vllm_dp0 2>&1 | tail -20; exit 1; }
curl -sf http://localhost:18081/health >/dev/null && echo "  dp1 :18081 HEALTHY" || { echo "  dp1 NOT HEALTHY"; docker logs vllm_dp1 2>&1 | tail -20; exit 1; }
curl -s http://localhost:18080/v1/models | grep -oE '"id":"[^"]*"' | head -1
curl -s http://localhost:18081/v1/models | grep -oE '"id":"[^"]*"' | head -1

echo "######## PHASE 1: dp0 SOLO (dp1 idle) -- single-card baseline ########"
env NAME=vllm_dp0 PORT=18080 MODEL=qwen36-27b-int4-dp0 LABEL=dp-solo TOKPATH=$MODELC CONC="$CONC" bash "$SWEEP"

echo "######## PHASE 2: dp0 + dp1 CONCURRENT -- true 2-card aggregate ########"
env NAME=vllm_dp0 PORT=18080 MODEL=qwen36-27b-int4-dp0 LABEL=dp-conc0 TOKPATH=$MODELC CONC="$CONC" bash "$SWEEP" &
P0=$!
env NAME=vllm_dp1 PORT=18081 MODEL=qwen36-27b-int4-dp1 LABEL=dp-conc1 TOKPATH=$MODELC CONC="$CONC" bash "$SWEEP" &
P1=$!
wait $P0; wait $P1

echo "######## RESULTS ########"
for L in dp-solo dp-conc0 dp-conc1; do
  f=$(ls -t "$ROOT"/results/sweep_${L}_*.csv 2>/dev/null | head -1)
  echo "== $L : $f =="; [ -n "$f" ] && cat "$f"
done
echo "######## INTERPRET: sum(dp-conc0 + dp-conc1) out_tok_s per C vs dp-solo. ~2x => DP scales clean. ########"
echo "######## reference banked single-card: C32 ~217 agg / C64 ~235 agg ########"
stop
echo "######## DATA-PARALLEL BENCH DONE $(date +%T) ########"
