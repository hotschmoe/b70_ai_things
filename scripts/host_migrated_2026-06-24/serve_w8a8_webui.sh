#!/usr/bin/env bash
# Long-lived interactive serve of the W8A8 TP=2 MTP recipe for OpenWebUI (holds the lease until stop).
# DTYPE=float16: the int8 W8A8 kernel produces float16 output; bf16 (--dtype auto) yields GARBAGE.
set -uo pipefail
cd /mnt/vm_8tb/b70
PORT=18080 DTYPE=float16 MAXLEN=32768 MAXSEQS=8 REASONPARSER=qwen3 \
  bash rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/serve.sh start
docker wait vllm_qwen36-27b-w8a8-sqgptq-mtp
