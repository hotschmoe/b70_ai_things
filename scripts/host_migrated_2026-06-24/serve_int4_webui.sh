#!/usr/bin/env bash
# Long-lived interactive serve: Qwen3.6-27B int4 (Lorbus, KNOWN-GOOD/coherent) + native MTP spec=4, for OpenWebUI.
# Single-card (card 0). Holds the lease until the container stops.
set -uo pipefail
cd /mnt/vm_8tb/b70
PORT=18080 DEVICE=0 MTPTOK=4 MAXLEN=32768 MAXSEQS=8 CAPSIZES=1,2,4,8 COMPILESZ= \
  bash rdy_to_serve/qwen36-27b-int4/serve.sh start
docker wait vllm_qwen36-27b-int4
