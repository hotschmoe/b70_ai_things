#!/usr/bin/env bash
# 121 -- quick 2048-ctx bench (PP / TTFT / TG @ c1,c4) for the 4-bit daily-driver candidates.
# Single-card (card 0) serves -> no TP=2 BCS wedge risk; leaves card 1 free.
# RUN:  ./bin/gpu-run --card 0 bash scripts/121_4bit_bench.sh
# Compares int4 captured vs int4 cudagraph=NONE (the campaign's stability/speed lesson) + w4a16.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
LOGDIR="$REPO/agentic-eval/results/logs"
TS="$(date +%Y%m%d_%H%M%S)"
RES="$LOGDIR/bench_4bit_${TS}.tsv"
export DEVICE="${DEVICE:-0}"; export B70_LOGDIR="$LOGDIR"
IN="${IN:-2048}"; OUT="${OUT:-128}"; CONC="${CONC:-1 4}"

# label | serve_dir | served_id | ckpt_container_path | env
CELLS=(
  "int4_graph|qwen36-27b-int4|qwen36-27b-int4|/models/Lorbus_Qwen3.6-27B-int4-AutoRound|GRAPH=1"
  "int4_none|qwen36-27b-int4|qwen36-27b-int4|/models/Lorbus_Qwen3.6-27B-int4-AutoRound|GRAPH=1 CGMODE=NONE"
  "int4_eager|qwen36-27b-int4|qwen36-27b-int4|/models/Lorbus_Qwen3.6-27B-int4-AutoRound|GRAPH=0"
  "w4a16_graph|qwen36-27b-w4a16|qwen36-27b-w4a16|/models/Qwen3.6-27B-W4A16|GRAPH=1"
)

printf '# 4-bit single-card bench  IN=%s OUT=%s CONC="%s"  %s\n' "$IN" "$OUT" "$CONC" "$(date)" | tee "$RES"
printf '# label\tc\treq_s\tout_tok_s\tttft_ms\ttpot_ms\tdecode_tps\tprefill_tps\n' | tee -a "$RES"

for row in "${CELLS[@]}"; do
  IFS='|' read -r label dir served ckpt env <<<"$row"
  echo "==================== $label ($env) ===================="
  slog="$LOGDIR/bench4_${label}_serve_${TS}.log"
  if ! env $env DEVICE="$DEVICE" PORT=18080 MAXLEN=8192 bash "$REPO/rdy_to_serve/$dir/serve.sh" start >"$slog" 2>&1; then
    echo "  [!] serve FAILED ($label) -- tail:"; tail -6 "$slog"
    printf '%s\tSERVE_FAIL\t-\t-\t-\t-\t-\t-\n' "$label" | tee -a "$RES"
    env DEVICE="$DEVICE" bash "$REPO/rdy_to_serve/$dir/serve.sh" stop >/dev/null 2>&1 || true
    continue
  fi
  cname="$(docker ps --format '{{.Names}}' | grep -iE 'vllm' | head -1)"
  echo "  serving as $cname; benching IN=$IN OUT=$OUT CONC='$CONC' ..."
  csv="$(NAME="$cname" MODEL="$served" TOKPATH="$ckpt" PORT=18080 IN="$IN" OUT="$OUT" CONC="$CONC" \
         bash "$REPO/bin/35_sweep_bench.sh" 2>&1)"
  echo "$csv" >"$LOGDIR/bench4_${label}_bench_${TS}.log"
  echo "$csv" | awk -F, -v L="$label" -v IN="$IN" 'NR>1 && $1 ~ /^[0-9]+$/ {
      ttft=$4; pp=(ttft>0)? (IN*1000.0)/ttft : 0;
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.0f\n", L,$1,$2,$3,$4,$5,$6,pp }' | tee -a "$RES"
  env DEVICE="$DEVICE" bash "$REPO/rdy_to_serve/$dir/serve.sh" stop >"$LOGDIR/bench4_${label}_stop_${TS}.log" 2>&1 || true
done
echo "=== done: $RES ==="
echo "----- RESULTS -----"; cat "$RES"
