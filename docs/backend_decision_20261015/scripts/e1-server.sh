#!/usr/bin/env bash
# E1 CPU llama-server. Usage: e1-server.sh NAME [extra llama-server args...]
NAME=$1; shift
M=/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs
exec docker run --rm --name "$NAME" --memory 110g --network host --user $(id -u):$(id -g) \
  -v /mnt/vm_8tb/b70/build/backend-decision/llama.cpp:/src:ro -v "$(readlink -f $M)":/m:ro \
  -v /mnt/cache:/mnt/cache:ro -v /mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/MTP:/mtp:ro \
  -e LD_LIBRARY_PATH=/src/build-cpu/bin --entrypoint /src/build-cpu/bin/llama-server \
  b70/flashnext-llamacpp-build:oneapi2026.1-de7fa0a \
  -m /m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -t 16 -tb 16 \
  --host 127.0.0.1 --port 18190 -c 8192 -np 2 --cache-ram 8192 -ctxcp 8 --jinja "$@"
