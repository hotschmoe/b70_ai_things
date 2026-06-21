#!/usr/bin/env bash
# PP=2 vs TP=2 on the x1-link rig. Pipeline-parallel passes ONE hidden state per token across the stage
# boundary (vs TP's ~128 all-reduces/token), so PP should dodge most of the Gen1-x1 comms tax. Serve 27B
# int4 PP=2 (eager) + single-stream decode probe (CONC 1,2). Wrap the WHOLE thing in one gpu-run lease.
set -uo pipefail
cd /mnt/vm_8tb/b70
MODELC=/models/Lorbus_Qwen3.6-27B-int4-AutoRound
SERVED=qwen36-27b-int4-pp2
docker rm -f vllm_multi 2>/dev/null || true
MODEL=$MODELC SERVED=$SERVED QUANT=none TP=1 PP=2 \
  IMG=vllm-xpu-env:v0230 UTIL=0.92 MAXLEN=8192 MAXSEQS=8 KVDTYPE=auto NAME=vllm_multi \
  EXTRA='--limit-mm-per-prompt {"image":0,"video":0}' \
  bash ./43_serve_multi.sh
if curl -sf http://localhost:18080/health >/dev/null 2>&1; then
  curl -s http://localhost:18080/v1/models | grep -oE '"id":"[^"]*"' | head -1
  env NAME=vllm_multi MODEL=$SERVED LABEL=27b-w4a16-PP2-eager TOKPATH=$MODELC CONC="1 2" bash ./35_sweep_bench.sh
fi
echo "######## PP=2 decode probe DONE (serve LEFT UP for client-side eval; docker stop vllm_multi when done) ########"
cat "$(ls -t results/sweep_27b-w4a16-PP2-eager*.csv 2>/dev/null | head -1)" 2>/dev/null || echo "(no CSV)"
