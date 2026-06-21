#!/usr/bin/env bash
# Capacity-unlock demo: serve the FULL BF16 27B (Qwen_Qwen3.6-27B, ~54 GB text weights -> CANNOT fit one
# 32 GB card) across BOTH B70s at TP=2. Proves the 2nd card enables a model that's physically too big for one.
# Eager (capture is blocked at TP=2 per the oneCCL sycl_graph finding). VLM -> limit mm so the vision-encoder
# dummy-profiling doesn't crash on XPU. Tight on VRAM: 28 GB/card weights -> small MAXLEN/KV. gpu-run wraps it.
set -uo pipefail
cd /mnt/vm_8tb/b70
MODEL=/models/Qwen_Qwen3.6-27B SERVED=qwen36-27b-bf16-tp2 QUANT=none TP=2 \
  IMG=vllm-xpu-env:v0230 UTIL="${UTIL:-0.95}" MAXLEN="${MAXLEN:-4096}" MAXSEQS="${MAXSEQS:-4}" \
  KVDTYPE="${KVDTYPE:-auto}" NAME=vllm_multi \
  EXTRA='--limit-mm-per-prompt {"image":0,"video":0}' \
  bash ./43_serve_multi.sh
