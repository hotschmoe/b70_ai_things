#!/usr/bin/env bash
# Persistent serve of Qwen3-14B-W8A8-autoround for the W8A8 autoround-vs-gptq HumanEval+ eval.
# Run under gpu-run; 36_serve.sh serves (waits health internally), then hold the lease via docker wait.
set -uo pipefail
cd /mnt/vm_8tb/b70
QUANT=/models/Qwen3-14B-W8A8-autoround SERVED=qwen3-14b-w8a8-autoround IMG=vllm-xpu-env:int8 \
  MAXLEN=4096 MAXSEQS=4 NAME=vllm_w8a8eval bash 36_serve.sh
if curl -sf http://localhost:18080/health >/dev/null 2>&1; then
  echo "=== w8a8_eval_serve HEALTHY; holding lease (docker stop vllm_w8a8eval to release) ==="
  docker wait vllm_w8a8eval
else
  echo "=== w8a8_eval_serve FAILED healthy ==="; docker logs vllm_w8a8eval 2>&1 | tail -20; docker stop vllm_w8a8eval 2>/dev/null
fi
echo "=== w8a8_eval_serve exit ==="
