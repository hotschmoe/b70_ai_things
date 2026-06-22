#!/usr/bin/env bash
# n-gram speculative-decode A/B on the 27B W4A8 (Seguin's cheap Qwen decode lever, docs/literature/10).
# Serves W4A8 (TP=1, graph) WITH --speculative-config ngram, sweeps ctx2048, stops. Compare decode vs the
# no-spec baseline (W4A8 TP=1 c1 20.7 / Q3). Route via gpu-run. n-gram helps when output echoes prompt n-grams.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
KSO=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so
NAME=vllm_ngram
MDIR=Qwen3.6-27B-W4A8-sqgptq-prepacked
SPEC='{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'
SPECTOK="${SPECTOK:-3}"   # allow num_speculative_tokens override
SPEC="{\"method\":\"ngram\",\"num_speculative_tokens\":$SPECTOK,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"

docker rm -f "$NAME" 2>/dev/null || true
echo "=== serve W4A8 + ngram (k=$SPECTOK) ==="
env IMG=vllm-xpu-env:int8g MODEL="/models/$MDIR" SERVED=qwen36-27b-w4a8-sqgptq \
    GRAPH=1 MAXLEN=2560 MAXSEQS=8 UTIL=0.95 NAME="$NAME" NOMM=1 DTYPE=auto PREPACK=1 TP=1 \
    KERNEL_SO="$KSO" SPEC="$SPEC" bash "$ROOT/30_serve_w4a8_graph.sh" || true

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "=== HEALTHY -> sweep ctx2048 (ngram k=$SPECTOK) ==="
  env NAME="$NAME" MODEL=qwen36-27b-w4a8-sqgptq LABEL="qwen36-27b-w4a8-ngram${SPECTOK}" \
      TOKPATH="/models/$MDIR" IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 2 4 8}" \
      bash "$ROOT/35_sweep_bench.sh" || true
else
  echo "=== NOT HEALTHY -- ngram serve failed; dumping tail ==="
  docker logs "$NAME" 2>&1 | tail -25 || true
fi
docker stop "$NAME" 2>/dev/null || true
echo "=== ngram bench done ==="
