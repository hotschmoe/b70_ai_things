#!/usr/bin/env bash
# Serve each model in MODELS captured (GRAPH=1), run a ctx-IN concurrency sweep, then stop -- the whole
# ladder in ONE gpu-run session (so the flock lease is held start-to-finish). Pollable: each model writes
# results/sweep_<SERVED>_<stamp>.csv (cols: concurrency,req_s,out_tok_s,mean_ttft_ms,mean_tpot_ms,per_stream_decode_tok_s).
#
# Run on host (under gpu-run):  cd /mnt/vm_8tb/b70 && MODELS="..." setsid ./qrun.sh <NAME> bash 66_bench_ladder.sh
#
# MODELS = entries separated by ';'. Each entry = "MODELdir|SERVED|TOKdir|IMG|SERVEFLAGS"
#   MODELdir / TOKdir : path under $ROOT/models (mounted at /models/<...>); TOKdir usually == MODELdir.
#   IMG               : docker image (blank -> vllm-xpu-env:int8g).
#   SERVEFLAGS        : extra "KEY=VAL KEY=VAL" env for 30_serve_w4a8_graph.sh (e.g. "NOMM=1 DTYPE=auto").
# Env: IN(2048) OUT(128) CONC("1 2 4 8") MAXLEN(4096) MAXSEQS(64) UTIL(0.90)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IN="${IN:-2048}"; OUT="${OUT:-128}"; CONC="${CONC:-1 2 4 8}"
MAXLEN="${MAXLEN:-4096}"; MAXSEQS="${MAXSEQS:-64}"; UTIL="${UTIL:-0.90}"
NAME=vllm_bench
CAPS="$(echo "$CONC" | tr ' ' ',')"   # capture sizes cover the sweep levels
: "${MODELS:?set MODELS=\"MODELdir|SERVED|TOKdir|IMG|SERVEFLAGS;...\"}"

echo "=== bench ladder :: IN=$IN OUT=$OUT CONC=[$CONC] CAPS=$CAPS MAXLEN=$MAXLEN MAXSEQS=$MAXSEQS ==="
IFS=';' read -ra ENTRIES <<< "$MODELS"
for entry in "${ENTRIES[@]}"; do
  [ -z "${entry// }" ] && continue
  IFS='|' read -r MDIR SERVED TOKDIR IMG SFLAGS <<< "$entry"
  IMG="${IMG:-vllm-xpu-env:int8g}"; TOKDIR="${TOKDIR:-$MDIR}"
  echo; echo "========== BENCH $SERVED  (model=$MDIR img=$IMG flags=[${SFLAGS:-}]) =========="
  docker rm -f "$NAME" 2>/dev/null || true
  if [ ! -d "$ROOT/models/$MDIR" ]; then echo "SKIP $SERVED -- missing $ROOT/models/$MDIR"; continue; fi
  # serve (30 does docker run -d + health wait + model-id print)
  env IMG="$IMG" MODEL="/models/$MDIR" SERVED="$SERVED" GRAPH=1 MAXLEN="$MAXLEN" MAXSEQS="$MAXSEQS" \
      UTIL="$UTIL" CAPSIZES="$CAPS" NAME="$NAME" ${SFLAGS:-} bash "$ROOT/30_serve_w4a8_graph.sh" || true
  if ! curl -sf "http://localhost:18080/health" >/dev/null 2>&1; then
    echo "SKIP $SERVED -- NOT healthy after serve"; docker logs --tail 40 "$NAME" 2>&1 | tail -40 || true
    docker rm -f "$NAME" 2>/dev/null || true; continue
  fi
  echo "--- served id ---"; curl -s "http://localhost:18080/v1/models" 2>/dev/null | grep -oE '"id":"[^"]*"' | head -2
  # sweep
  NAME="$NAME" MODEL="$SERVED" LABEL="$SERVED" TOKPATH="/models/$TOKDIR" IN="$IN" OUT="$OUT" CONC="$CONC" \
      bash "$ROOT/35_sweep_bench.sh" || echo "sweep failed for $SERVED"
  docker stop "$NAME" 2>/dev/null || true; docker rm -f "$NAME" 2>/dev/null || true
done
echo; echo "===== BENCH LADDER DONE ====="
