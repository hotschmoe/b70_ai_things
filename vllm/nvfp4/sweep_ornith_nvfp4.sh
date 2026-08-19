#!/usr/bin/env bash
# sweep_ornith_nvfp4.sh -- L59 Ornith 35B NVFP4 G1 matrix.
# KV is always auto (bf16-class). Do NOT pass --kv-cache-dtype bfloat16.
# Run under: B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run bash vllm/nvfp4/sweep_ornith_nvfp4.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
SERVE="$REPO/vllm/nvfp4/serve_nvfp4_moe_35b.sh"
PROBE="$REPO/vllm/nvfp4/g1_probe.py"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTDIR="${OUTDIR:-$REPO/results/logs/ornith_nvfp4_sweep_${STAMP}}"
mkdir -p "$OUTDIR"
SUMM="$OUTDIR/summary.tsv"
LEDGER="$OUTDIR/ledger.txt"
HOST="${HOST:-192.168.10.5}"
PORT="${PORT:-18080}"
NAME="${NAME:-ornith_nvfp4_sweep}"
IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
FUSED_SO_DEFAULT="$ROOT/nvfp4_fused_kernel_gdn/_xpu_C.abi3.so"
F8SCALE_SO="$ROOT/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so"
ORNI="/models/ornith-1.5-35b-a3b/nvfp4-modelopt"
OFFI="/models/qwen3.6-35b-a3b/nvfp4-modelopt"

echo -e "arm\ttp\tgraph\tmode\tint8xmx\tso\tf8m\tckpt\tserved\thealthy_s\tverdict\thas_paris\thas_391\tbangs\ttps_paris\tnote" > "$SUMM"
{
  echo "L59 Ornith NVFP4 sweep $STAMP"
  echo "KV=auto (KV_FP8=0) LANGONLY=1 IMG=$IMG PORT=$PORT"
  echo "P2PACCESS=0  no DD  no bfloat16 enum"
} | tee "$LEDGER"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LEDGER"; }

wait_healthy() {
  local i status
  for i in $(seq 1 180); do
    status=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
    if [ "$status" = "exited" ] || [ "$status" = "missing" ]; then
      return 1
    fi
    if curl -sf --max-time 3 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

stop_serve() {
  NAME="$NAME" bash "$SERVE" stop >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 3
}

start_serve() {
  # args via env already set
  log "start NAME=$NAME TP=$TP GRAPH=$GRAPH MODE=$MODE SERVED=$SERVED"
  NAME="$NAME" PORT="$PORT" IMG="$IMG" \
    bash "$SERVE" >"$OUTDIR/${ARM}_boot.txt" 2>&1
}

run_arm() {
  local extra_env="" fused_so="$FUSED_SO_DEFAULT" note=""
  ARM="$1"; TP="$2"; GRAPH="$3"; MODE="$4"; INT8XMX="$5"
  SO_KIND="$6"; F8M="$7"; CKPT_KIND="$8"
  export ARM TP GRAPH MODE

  if [ "$CKPT_KIND" = official ]; then
    export CKPT="$OFFI"
    tag="qwen36-35b-A3B-NVFP4"
  else
    export CKPT="$ORNI"
    tag="ornith-1.5-35b-A3B-NVFP4"
  fi
  if [ "$SO_KIND" = f8scale ]; then
    fused_so="$F8SCALE_SO"
  fi
  export FUSED_SO="$fused_so"
  extra_env="B70_FP8_CHANNEL_INT8XMX=${INT8XMX}"
  if [ -n "$F8M" ] && [ "$F8M" != "0" ]; then
    extra_env="${extra_env} B70_NVFP4_F8_SCALE_M_MAX=${F8M}"
  fi
  export B70_EXTRA_ENV="$extra_env"
  export KV_FP8=0 LANGONLY=1 MAXSEQS="${MAXSEQS:-2}" MAXLEN="${MAXLEN:-8192}"
  export UTIL="${UTIL:-0.90}" MTPTOK="${MTPTOK:-}" P2PACCESS=0
  export SERVED="${tag}-${MODE}-tp${TP}-g${GRAPH}-xmx${INT8XMX}-${ARM}"
  if [ "$GRAPH" = 1 ] && [ "$TP" = 2 ]; then
    export CAPSIZES="${CAPSIZES:-1,2,4,8}"
  else
    export CAPSIZES="${CAPSIZES:-}"
  fi

  log "===== ARM $ARM tp=$TP graph=$GRAPH mode=$MODE xmx=$INT8XMX so=$SO_KIND f8m=${F8M:-0} ckpt=$CKPT_KIND ====="
  stop_serve
  local t0 t1 hs=NA verdict=BOOTFAIL has_paris=NA has_391=NA bangs=NA tps=NA
  t0=$(date +%s)
  start_serve
  if ! wait_healthy; then
    note="not_healthy"
    docker logs "$NAME" >"$OUTDIR/${ARM}_docker.log" 2>&1 || true
    log "ARM $ARM BOOTFAIL"
  else
    t1=$(date +%s)
    hs=$((t1 - t0))
    log "ARM $ARM HEALTHY ${hs}s; G1..."
    if python3 "$PROBE" "http://${HOST}:${PORT}/v1" auto \
        >"$OUTDIR/${ARM}_g1.json" 2>"$OUTDIR/${ARM}_g1.err"; then
      verdict=GO
    else
      verdict=NO-GO
    fi
    if [ -s "$OUTDIR/${ARM}_g1.json" ]; then
      read -r has_paris has_391 bangs tps verdict_j < <(
        python3 -c "
import json
d=json.load(open('$OUTDIR/${ARM}_g1.json'))
print(d.get('has_paris'), d.get('has_391'), d.get('any_bangs'),
      (d.get('probes') or {}).get('paris',{}).get('tps'),
      d.get('verdict'))
")
      [ -n "$verdict_j" ] && verdict="$verdict_j"
    fi
    docker logs "$NAME" >"$OUTDIR/${ARM}_docker.log" 2>&1 || true
    grep -E 'nvfp4-shim|MXFP8|KV cache|channel-FP8|fused MoE|Error|DEVICE_LOST' \
      "$OUTDIR/${ARM}_docker.log" | tail -40 >"$OUTDIR/${ARM}_shim.txt" || true
    log "ARM $ARM $verdict paris=$has_paris 391=$has_391 bangs=$bangs tps=$tps ${hs}s"
  fi
  echo -e "${ARM}\t${TP}\t${GRAPH}\t${MODE}\t${INT8XMX}\t${SO_KIND}\t${F8M:-0}\t${CKPT_KIND}\t${SERVED}\t${hs}\t${verdict}\t${has_paris}\t${has_391}\t${bangs}\t${tps}\t${note}" \
    | tee -a "$SUMM"
  # persist last good arm for TP=2 / GRAPH follow-ons
  if [ "$verdict" = GO ]; then
    echo "$ARM $TP $GRAPH $MODE $INT8XMX $SO_KIND ${F8M:-0} $CKPT_KIND" >> "$OUTDIR/go_arms.txt"
  fi
  stop_serve
}

# --- matrix ---
# A2: planned next -- fused expert gemm, INT8XMX default on
run_arm A2 1 0 fused 1 fused 0 ornith
# A3: fused + disable channel-FP8->INT8XMX (tiled F.linear)
run_arm A3 1 0 fused 0 fused 0 ornith
# A4: emul + INT8XMX off (isolates 1e vs expert path)
run_arm A4 1 0 emul 0 fused 0 ornith
# CTRL: official Qwen3.6-35B NVFP4 fused, INT8XMX off (recipe/control)
run_arm CTRL 1 0 fused 0 fused 0 official

GO_COUNT=0
[ -f "$OUTDIR/go_arms.txt" ] && GO_COUNT=$(wc -l < "$OUTDIR/go_arms.txt")

if [ "${GO_COUNT:-0}" -gt 0 ]; then
  # pick last GO ornith arm if any, else last GO
  BEST=$(grep ' ornith$' "$OUTDIR/go_arms.txt" | tail -1 || true)
  [ -z "$BEST" ] && BEST=$(tail -1 "$OUTDIR/go_arms.txt")
  # go_arms: ARM TP GRAPH MODE INT8XMX SO F8 CKPT
  set -- $BEST
  B_ARM=$1; B_MODE=$4; B_XMX=$5; B_SO=$6; B_F8=$7; B_CK=$8
  log "BEST GO=$BEST -- GRAPH then TP=2"
  run_arm G1g 1 1 "$B_MODE" "$B_XMX" "$B_SO" "$B_F8" "$B_CK"
  run_arm T2e 2 0 "$B_MODE" "$B_XMX" "$B_SO" "$B_F8" "$B_CK"
  run_arm T2g 2 1 "$B_MODE" "$B_XMX" "$B_SO" "$B_F8" "$B_CK"
  # extra kernel: f8scale on the winning recipe, TP=1 eager
  run_arm F8S 1 0 "$B_MODE" "$B_XMX" f8scale 8 "$B_CK"
else
  log "no GO on TP=1 -- still run fused+f8scale and one TP=2 isolate"
  run_arm A5 1 0 fused 0 f8scale 8 ornith
  run_arm T2i 2 0 fused 0 fused 0 ornith
fi

log "DONE summary $SUMM"
column -t -s $'\t' "$SUMM" | tee -a "$LEDGER"
stop_serve
echo "sweep_complete $STAMP" | tee -a "$LEDGER"
