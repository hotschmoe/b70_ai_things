#!/usr/bin/env bash
# Persistent serve of Qwable int4 for the HumanEval+ eval. Run under gpu-run: it serves (30_serve waits health
# internally) then holds the lease via `docker wait` until the container is stopped (docker stop vllm_eval).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
docker rm -f vllm_eval 2>/dev/null || true
env IMG=vllm-xpu-env:v0230 MODEL=/models/Qwable-5-27B-Coder-int4-AutoRound SERVED=qwable-27b-int4 \
  GRAPH=1 DTYPE=auto UTIL=0.92 NOMM=1 MAXLEN=8192 MAXSEQS=8 CAPSIZES=1,2,4,8,16,32,64 NAME=vllm_eval PORT=18080 \
  bash ./30_serve_w4a8_graph.sh 2>&1 | tail -8
if curl -sf http://localhost:18080/health >/dev/null 2>&1; then
  echo "=== q8_eval_serve HEALTHY; holding lease (docker stop vllm_eval to release) ==="
  docker wait vllm_eval
else
  echo "=== q8_eval_serve FAILED to become healthy ==="; docker logs vllm_eval 2>&1 | tail -15; docker stop vllm_eval 2>/dev/null
fi
echo "=== q8_eval_serve exit ==="
