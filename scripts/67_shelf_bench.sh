#!/usr/bin/env bash
# Bench the whole rdy_to_serve shelf on the NEW Ubuntu/kernel-7.0 install: validates each model still
# serves + generates coherently, and produces PP/TTFT/TG. Runs each model's OWN serve.sh recipe (its real
# GRAPH/TP/MTP defaults) -> serve -> health -> coherence-gated gen probe -> concurrency sweep -> stop.
# ctx IN=2048 (meaningful TTFT/prefill), c=1 and c=4. One gpu-run session holds the lease start-to-finish.
#   ./gpu-run bash 67_shelf_bench.sh             # all models
#   ./gpu-run bash 67_shelf_bench.sh <model>     # one model (e.g. qwen36-27b-w8a8-sqgptq-mtp)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
RTS="$ROOT/rdy_to_serve"
export IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 4}"
ONLY="${1:-}"    # prefix filter, e.g. "qwen36" matches all qwen36-* models
echo "=== SHELF BENCH :: IN=$IN OUT=$OUT CONC=[$CONC] :: $(date '+%F %T') ==="
declare -a NAMES RES
for d in "$RTS"/*/; do
  m=$(basename "$d"); case "$m" in _*) continue;; esac
  [ -f "$d/serve.sh" ] || continue
  [ -n "$ONLY" ] && [[ "$m" != "$ONLY"* ]] && continue
  # Auto-skip if this model's docker image isn't loaded (e.g. v0230moe not yet recovered).
  img=$(grep -hoE "vllm-xpu-env:[a-z0-9]+" "$d/serve.sh" | head -1)
  if [ -n "$img" ] && ! docker image inspect "$img" >/dev/null 2>&1; then
    echo; echo "## SKIP $m -- image $img not loaded"; NAMES+=("$m"); RES+=("SKIP_NOIMG"); continue
  fi
  echo; echo "##################################################################"
  echo "## $m  (img=$img)  ::  $(date '+%T')"
  echo "##################################################################"
  if env IN="$IN" OUT="$OUT" CONC="$CONC" bash "$d/serve.sh" run; then
    NAMES+=("$m"); RES+=("OK")
  else
    NAMES+=("$m"); RES+=("FAIL")
    env bash "$d/serve.sh" stop >/dev/null 2>&1 || true
  fi
done
echo; echo "================ SHELF BENCH SUMMARY ================"
for i in "${!NAMES[@]}"; do printf "  %-8s %s\n" "${RES[$i]}" "${NAMES[$i]}"; done
echo "Per-model CSVs: $ROOT/results/sweep_*.csv   ($(date '+%F %T'))"
