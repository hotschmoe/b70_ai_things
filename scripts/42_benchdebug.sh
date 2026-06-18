#!/usr/bin/env bash
# Debug: run one vllm bench serve and show full output (find why metrics were empty).
NAME="${NAME:-vllm_qwen3}"; MODEL="${MODEL:-qwen36-int4}"; TOK="${TOK:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"; PORT=18080
docker exec -i "$NAME" vllm bench serve --backend vllm --model "$MODEL" --tokenizer "$TOK" \
  --base-url "http://localhost:${PORT}" --endpoint /v1/completions --dataset-name random \
  --random-input-len 256 --random-output-len 64 --num-prompts 8 --max-concurrency 4 --ignore-eos 2>&1 | tail -35
