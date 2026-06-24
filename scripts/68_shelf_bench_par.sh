#!/usr/bin/env bash
# Parallel shelf bench. TP=1 (single-card) models run 2-up -- one pinned to card 0 (port 8000), one to
# card 1 (port 8001) via ZE_AFFINITY_MASK -- to cut wall-clock. TP=2 models run solo (both cards). Each
# instance takes its OWN per-card gpu-run lease (--card N); TP=2 takes the whole-box lease. ctx IN=2048,
# c=1 and c=4.  Run:  bash 68_shelf_bench_par.sh [name-prefix]   (no outer gpu-run -- it leases per-instance)
# NOTE: TP=1 numbers are measured with the other card busy (co-resident); compare card-isolated metrics
# (decode tok/s, TTFT) to the known solo baseline to confirm no skew. TP=1 cache (/vllm_cache) is shared
# but model-keyed, so the two concurrent compiles write disjoint entries.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
RTS="$ROOT/rdy_to_serve"
GPURUN="$ROOT/gpu-run"
export IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 4}"
PREFIX="${1:-qwen36}"

declare -a TP1 TP2
for d in "$RTS"/"$PREFIX"*/; do
  m=$(basename "$d"); s="$d/serve.sh"; [ -f "$s" ] || continue
  img=$(grep -hoE "vllm-xpu-env:[a-z0-9]+" "$s" | head -1)
  if [ -n "$img" ] && ! docker image inspect "$img" >/dev/null 2>&1; then echo "SKIP $m (no image $img)"; continue; fi
  if grep -qE 'TP="\$\{TP:-2\}"|TP:-2|tensor-parallel-size[ =]*2' "$s"; then TP2+=("$m"); else TP1+=("$m"); fi
done
echo "=== PARALLEL SHELF BENCH :: IN=$IN OUT=$OUT CONC=[$CONC] :: $(date '+%F %T') ==="
echo "TP1 (run 2-up, one per card): ${TP1[*]:-none}"
echo "TP2 (solo, both cards):       ${TP2[*]:-none}"

run_one() {  # model card port
  local m="$1" card="$2" port="$3"
  echo ">>> START $m  card=$card port=$port  $(date '+%T')"
  env DEVICE="$card" PORT="$port" NAME="vllm_${m}" IN="$IN" OUT="$OUT" CONC="$CONC" \
    "$GPURUN" --card "$card" bash "$RTS/$m/serve.sh" run > "$ROOT/results/log_${m}.txt" 2>&1
  echo "<<< DONE  $m (exit $?)  $(date '+%T')"
}
run_solo() {  # model
  local m="$1"
  echo ">>> START $m  TP=2 solo  $(date '+%T')"
  env NAME="vllm_${m}" IN="$IN" OUT="$OUT" CONC="$CONC" \
    "$GPURUN" bash "$RTS/$m/serve.sh" run > "$ROOT/results/log_${m}.txt" 2>&1
  echo "<<< DONE  $m (exit $?)  $(date '+%T')"
}

# Let a just-freed card's VRAM fully release before the next model grabs it at high UTIL
# (a back-to-back start otherwise OOMs engine-init -- seen on w4a8 in wave 2). 15s is ample.
SETTLE="${SETTLE:-15}"

# TP=1: two at a time on card 0 / card 1
i=0
while [ $i -lt ${#TP1[@]} ]; do
  a="${TP1[$i]}"; b="${TP1[$((i+1))]:-}"
  echo; echo "### WAVE: $a (card0)  +  ${b:-<none>} (card1)"
  run_one "$a" 0 8000 & pa=$!
  pb=""; [ -n "$b" ] && { run_one "$b" 1 8001 & pb=$!; }
  wait $pa; [ -n "$pb" ] && wait $pb
  i=$((i+2))
  echo "... settle ${SETTLE}s (VRAM release) ..."; sleep "$SETTLE"
done

# TP=2: solo (settle before each -- it needs BOTH cards fully free)
for m in "${TP2[@]}"; do echo; sleep "$SETTLE"; run_solo "$m"; done

echo; echo "================ PARALLEL SHELF BENCH DONE $(date '+%F %T') ================"
echo "logs: $ROOT/results/log_<model>.txt   CSVs: $ROOT/results/sweep_*.csv"