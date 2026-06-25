#!/usr/bin/env bash
# 120 -- W8A8-27B MTP x graph combinatorial sweep (perf + crash-soak).
#
# Drives the shelf recipe (rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/serve.sh) one cell
# at a time. Two passes:
#   perf : single-stream + c4 decode t/s (1000/TPOT via bin/35_sweep_bench.sh). ~2 min/cell. ALL cells.
#   soak : drive long generations until SOAK_TOKENS cumulative (default 50k, >2x the ~20-28k crash
#          threshold) or the engine dies. Only the cells tagged soak/both.
#
# Campaign: docs/20260625_w8a8_27b_mtp_graph_campaign.md
# RUN UNDER THE LEASE:  bin/gpu-run bash scripts/120_w8a8_mtp_graph_sweep.sh perf
# Wedge policy: B70_AUTO_RESET defaults 0 -> on a detected wedge the sweep HALTS and reports
#   (no reboot). Set B70_AUTO_RESET=1 for unattended self-heal (xe-reset -> reboot on this box).
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
RECIPE="$REPO/rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/serve.sh"
SERVED=qwen36-27b-w8a8-sqgptq-mtp
CKPT_CT=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft     # container path (vllm bench --tokenizer runs in-container)
export PORT="${PORT:-18080}"   # recipe default is 8000; export so serve.sh binds where we probe
MODE="${1:-perf}"                                      # perf | soak
SOAK_TOKENS="${SOAK_TOKENS:-50000}"
SOAK_MAXTOK="${SOAK_MAXTOK:-8192}"
BENCH_IN="${BENCH_IN:-512}"; BENCH_OUT="${BENCH_OUT:-256}"; BENCH_CONC="${BENCH_CONC:-1 4}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$REPO/agentic-eval/results/logs"
RESULTS="$LOGDIR/campaign_120_${MODE}_${TS}.tsv"
export B70_AUTO_RESET="${B70_AUTO_RESET:-0}"
export B70_LOGDIR="$LOGDIR"
export MAXLEN="${MAXLEN:-16384}"                       # fits SOAK_MAXTOK gen + headroom; decode t/s ~insensitive
STOP_SWEEP=0

# Execution order. Safe eager cell first (canary: validates serve/bench/teardown plumbing).
# Override by passing cell names after MODE, e.g.:
#   scripts/120 soak E_pw_drafteager_mtp3 B_none_mtp3   (E first -> gets a pristine card after a reboot)
ORDER=(A_eager_mtp3 repro_pw_mtp3 E_pw_drafteager_mtp3 B_none_mtp3 nomtp_pw_cap)
[ "$#" -gt 1 ] && ORDER=("${@:2}")

# Which cells get a soak in `soak` mode (the candidate fixes + a repro control).
soak_cell() { case "$1" in E_pw_drafteager_mtp3|B_none_mtp3|repro_pw_mtp3) return 0;; *) return 1;; esac; }

# Per-cell env. Reset all knobs, then set. Recipe defaults: GRAPH=1 CGMODE=PIECEWISE MTPTOK=3 PUSH_AR=1.
cell_env() {
  unset GRAPH CGMODE MTPTOK SPEC B70_NOMTP CAPSIZES PUSH_AR PUSH_AR_GRAPH
  case "$1" in
    A_eager_mtp3)         export GRAPH=0 MTPTOK=3 ;;                                   # shipped fix (baseline)
    repro_pw_mtp3)        export GRAPH=1 MTPTOK=3 ;;                                   # crashing winner (the bug)
    E_pw_drafteager_mtp3) export GRAPH=1 SPEC='{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' ;; # THE lever
    B_none_mtp3)          export GRAPH=1 CGMODE=NONE MTPTOK=3 ;;                       # no replay, keep compile
    nomtp_pw_cap)         export GRAPH=1 B70_NOMTP=1 ;;                                # captured, MTP off (control)
    *) echo "unknown cell $1" >&2; return 2 ;;
  esac
}

record() { printf '%s\t%s\t%s\t%s\n' "$(date +%H:%M:%S)" "$1" "$2" "$3" | tee -a "$RESULTS"; }

container() { docker ps --format '{{.Names}}' 2>/dev/null | grep -iE 'vllm|qwen' | head -1; }

soak_until_crash() {  # $1=cname ; echoes PASS:<tok> or CRASH:<tok>:<sig>
  local cname="$1" total=0 t0; t0=$(date +%s)
  local sig="" prompt
  prompt='You are an expert systems engineer. Write an exhaustive, deeply technical reference on distributed consensus, Raft and Paxos, fault tolerance, vector clocks, CRDTs, and quorum systems. Include detailed pseudocode and edge cases.'
  while [ "$total" -lt "$SOAK_TOKENS" ] && [ "$STOP_SWEEP" = 0 ]; do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 || { sig="health_fail"; break; }
    local resp ct
    resp=$(curl -s --max-time 600 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED\",\"prompt\":\"$prompt\",\"max_tokens\":$SOAK_MAXTOK,\"temperature\":0,\"ignore_eos\":true}" 2>&1)
    ct=$(printf '%s' "$resp" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+' | head -1)
    if [ -z "$ct" ] || [ "$ct" = 0 ]; then
      sig=$(docker logs "$cname" 2>&1 | grep -oE 'EngineDeadError|RPC call to sample_tokens timed out|Fatal Python error: Aborted|cancelled|DEVICE_LOST|OUT_OF_RESOURCES' | tail -1)
      [ -z "$sig" ] && sig="req_fail_nosig"
      break
    fi
    total=$((total + ct))
    echo "    [soak] +${ct} -> ${total}/${SOAK_TOKENS} tok  ($(( $(date +%s) - t0 ))s)" >&2
  done
  if [ "$total" -ge "$SOAK_TOKENS" ]; then echo "PASS:${total}"; else echo "CRASH:${total}:${sig:-unknown}"; fi
}

spec_accept() {  # pull latest SpecDecoding acceptance length from /metrics-driven log line
  local cname="$1"
  docker logs "$cname" 2>&1 | grep -oE 'Mean acceptance length: [0-9.]+' | tail -1 | grep -oE '[0-9.]+' | head -1
}

run_cell() {
  local label="$1"
  [ "$STOP_SWEEP" = 1 ] && return 0
  cell_env "$label" || return 0
  local slog="$LOGDIR/cell_${label}_serve_${TS}.log"
  echo "===================================================================="
  echo "== CELL $label  GRAPH=${GRAPH:-1} CGMODE=${CGMODE:-PIECEWISE} MTPTOK=${MTPTOK:-} SPEC=${SPEC:-} NOMTP=${B70_NOMTP:-0}"
  echo "===================================================================="
  if ! bash "$RECIPE" start >"$slog" 2>&1; then
    echo "[!] serve start FAILED for $label (tail):"; tail -8 "$slog"
    if grep -qiE 'WEDGED before serve|BOX MAY BE WEDGED' "$slog"; then STOP_SWEEP=1; record "$label" "WEDGE" "serve_blocked_wedge"; return 0; fi
    record "$label" "SERVE_FAIL" "see $(basename "$slog")"
    bash "$RECIPE" stop >>"$slog" 2>&1 || true
    return 0
  fi
  local cname; cname="$(container)"
  echo "   serving as container: ${cname:-<none>}"

  local dec1=NA dec4=NA accept=NA soak_res="-"
  # perf
  local blog="$LOGDIR/cell_${label}_bench_${TS}.log"
  local csv; csv=$(NAME="$cname" MODEL="$SERVED" LABEL="$label" TOKPATH="$CKPT_CT" PORT="$PORT" \
                   IN="$BENCH_IN" OUT="$BENCH_OUT" CONC="$BENCH_CONC" bash "$REPO/bin/35_sweep_bench.sh" 2>&1)
  echo "$csv" >"$blog"
  dec1=$(echo "$csv" | awk -F, '$1=="1"{print $6}' | head -1); dec1=${dec1:-NA}
  dec4=$(echo "$csv" | awk -F, '$1=="4"{print $6}' | head -1); dec4=${dec4:-NA}
  accept=$(spec_accept "$cname"); accept=${accept:-NA}

  # soak (soak mode + tagged cell only)
  if [ "$MODE" = soak ] && soak_cell "$label"; then
    echo "   soaking $label to ${SOAK_TOKENS} tok ..."
    soak_res=$(soak_until_crash "$cname")
    accept=$(spec_accept "$cname"); accept=${accept:-NA}
  fi

  local stoplog="$LOGDIR/cell_${label}_stop_${TS}.log"
  bash "$RECIPE" stop >"$stoplog" 2>&1 || true
  local wedge=""
  if grep -qiE 'BOX MAY BE WEDGED|WEDGED' "$stoplog" 2>/dev/null; then wedge=" [WEDGE -- HALTING SWEEP]"; STOP_SWEEP=1; fi
  record "$label" "${soak_res}" "decode1=${dec1} decode4=${dec4} accept_len=${accept}${wedge}"
}

echo "# campaign 120  mode=$MODE  $(date)" | tee "$RESULTS"
echo "# columns: time  cell  soak_result  notes(decode1/decode4 t/s, accept_len)" | tee -a "$RESULTS"
echo "# MAXLEN=$MAXLEN BENCH_IN=$BENCH_IN BENCH_OUT=$BENCH_OUT CONC='$BENCH_CONC' SOAK_TOKENS=$SOAK_TOKENS" | tee -a "$RESULTS"
for cell in "${ORDER[@]}"; do run_cell "$cell"; done
echo "=== campaign 120 ($MODE) done. results: $RESULTS ==="
[ "$STOP_SWEEP" = 1 ] && echo "[!] sweep halted early on a wedge -- recover (bin/xe-reset or reboot) before re-running."
echo "----- RESULTS -----"; cat "$RESULTS"
