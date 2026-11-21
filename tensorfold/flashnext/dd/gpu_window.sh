#!/usr/bin/env bash
# gpu_window.sh -- the ONE GPU window for the TensorFold hotschmoe-dd cutover (tensorfold/flashnext/GPU-TESTS.md).
# Owner: worker INT. Run as hotschmoe from any directory. Every GPU touch goes through bin/gpu-run; timed parts
# hold bin/cpu-run --quiet outside the lease. Port 18150 only; 18080/18124 are touched only by cutover/rollback.
#
#   gpu_window.sh preflight   CPU only, safe while Strata serves: files, frozen sha, ports, steps list, fixture
#   gpu_window.sh window      CONFIRM=yes: stop Strata, then regress + prod; on any HARD failure restart Strata
#   gpu_window.sh regress     G1: int-rf2-config quick bench on the frozen binary (bitwise p0 hash, C1-C8, PF2k)
#   gpu_window.sh prod        G2: production-config server on :18150 + every step in gpu_window_steps.tsv
#   gpu_window.sh cutover     CONFIRM=yes: install + start hotschmoe-dd-tensorfold, dd_functional.py via :18080
#                             with image; PASS -> enable TF / disable Strata; FAIL -> rollback
#   gpu_window.sh rollback    stop TF unit (its ExecStopPost runs xpu-health), re-enable + start Strata, verify :18080
# Env: TF_DD_CONFIG (default tf-dd-prod.env), WIN_TAG, WIN_DIR, WIN_STEPS, WIN_ONLY="ID ..." (subset of prod steps).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host=/mnt/vm_8tb/github/b70_ai_things
config=${TF_DD_CONFIG:-$here/tf-dd-prod.env}
cfg() { ( set +u; source "$config" >/dev/null 2>&1; eval "printf '%s' \"\${$1:-}\"" ); }
TF_BIN=$(cfg TF_BIN); TF_BIN_SHA256=$(cfg TF_BIN_SHA256); TF_COMMIT=$(cfg TF_COMMIT)
wt=$(cd "$(dirname "$TF_BIN")/../../.." 2>/dev/null && pwd || echo /nonexistent)
tag=${WIN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
win=${WIN_DIR:-/mnt/vm_8tb/b70/results/hotschmoe-dd/gpu-window/$tag}
steps=${WIN_STEPS:-$here/gpu_window_steps.tsv}
fixture=${TF_IMAGE_FIXTURE:-/mnt/vm_8tb/b70/results/hotschmoe-dd/restore-20261021/vision-fixture.png}
S=/mnt/vm_8tb/b70/results/tf-flashnext/workers/S
ref=$S/int-rf2-results.json
port=${WIN_TEST_PORT:-18150}
key=/mnt/vm_8tb/b70/secrets/dd_api_key
hold=/mnt/vm_8tb/b70/results/tf-flashnext/PRIORITY.hold
export B70_AGENT=hotschmoe-dd-tf B70_CPU_LOCK_TIMEOUT=${B70_CPU_LOCK_TIMEOUT:-1800} B70_GPU_LOCK_TIMEOUT=${B70_GPU_LOCK_TIMEOUT:-600}
ts() { date -u '+%F %T UTC'; }
mkdir -p "$win"
gate() { printf '%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$1" "$2" "$3" "${4:-}" | tee -a "$win/gates.tsv"; }  # id hard|soft PASS|FAIL detail
log() { echo "$(ts) [gpu_window] $*" | tee -a "$win/window.log"; }
die() { log "ABORT: $*"; exit 1; }
gpumd() { printf '%s [hotschmoe-dd-tf] %s\n' "$(ts)" "$*" >> "$host/tensorfold/flashnext/GPU.md"; }
wait_http() {  # url seconds [auth]
  local i; for ((i=0;i<$2;i++)); do
    if [[ -n ${3:-} ]]; then curl -fs --max-time 3 -H "Authorization: Bearer $(cat "$key")" "$1" >/dev/null && return 0
    else curl -fs --max-time 3 "$1" >/dev/null && return 0; fi
    sleep 1; done; return 1; }

preflight() {
  local bad=0 have
  [[ -x $TF_BIN ]] || { log "preflight: missing $TF_BIN"; bad=1; }
  have=$(sha256sum "$TF_BIN" 2>/dev/null | cut -d' ' -f1)
  [[ $have == "$TF_BIN_SHA256" ]] || { log "preflight: sha $have != env $TF_BIN_SHA256"; bad=1; }
  [[ $(git -C "$wt" rev-parse HEAD 2>/dev/null) == "$TF_COMMIT" ]] || { log "preflight: $wt HEAD != $TF_COMMIT"; bad=1; }
  [[ -z $(git -C "$wt" status --porcelain --untracked-files=no 2>/dev/null) ]] || { log "preflight: $wt dirty"; bad=1; }
  [[ -r $fixture ]] || { log "preflight: no image fixture $fixture"; bad=1; }
  [[ -r $ref && -r $S/expected-current.json ]] || { log "preflight: int-rf2 reference missing"; bad=1; }
  [[ -r $steps ]] || { log "preflight: no steps file $steps"; bad=1; }
  grep -q '@FREEZE' "$config" && { log "preflight: $config still has @FREEZE placeholders"; bad=1; }
  awk -F'\t' '!/^#/ && NF && (NF<5 || ($3!="hard" && $3!="soft") || $4+0<1 || $4+0>180) {print "bad step line " NR ": " $0; e=1} END{exit e}' "$steps" \
    || { log "preflight: steps file malformed (5 tab fields, hard|soft, timeout 1..180 s)"; bad=1; }
  ss -ltn "sport = :$port" | tail -n +2 | grep -q . && { log "preflight: :$port busy"; bad=1; }
  log "preflight: strata=$(systemctl --user is-active hotschmoe-dd) tf=$(systemctl --user is-active hotschmoe-dd-tensorfold 2>/dev/null) $("$host/bin/gpu-run" --status 2>/dev/null | grep -c HELD) card leases held; cpu: $("$host/bin/cpu-run" --status)"
  log "preflight: config=$config commit=$TF_COMMIT bin=$TF_BIN steps=$(grep -cv '^#' "$steps") rc=$bad"
  return $bad
}

regress() {  # G1 ~3 min: unchanged tools/flashnext_serve_bench.sh at the int-rf2 config, clean environment
  mkdir -p "$win/regress"; cp "$S/expected-current.json" "$win/regress/"
  log "G1 regress start (cards 0+1, ~3 min)"
  env -i HOME="$HOME" USER="${USER:-hotschmoe}" PATH=/usr/local/bin:/usr/bin:/bin XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
    B70_AGENT=hotschmoe-dd-tf B70_CPU_LOCK_TIMEOUT=900 FN_BENCH_ROOT="$win/regress" FN_BENCH_EXPECTED="$win/regress/expected-current.json" \
    S_SKIP_BUILD=1 FN_BENCH_QUICK=1 timeout 1200 bash "$wt/tools/flashnext_serve_bench.sh" "$wt" "regress" > "$win/regress/outer.log" 2>&1
  local rc=$?
  if ((rc)); then gate G1-regress hard FAIL "bench rc=$rc (oracle, concurrency==solo or health; see $win/regress)"; return 1; fi
  gate G1-regress hard PASS "p0 bitwise == int-rf2, concurrent == solo, health pre/post HEALTHY"
  local out; out=$(python3 "$here/gpu_window_checks.py" compare "$win/regress/regress-results.json" "$ref" 0.95); local c=$?
  gate G1-speed soft "$( ((c)) && echo FAIL || echo PASS)" "$out"
  return 0
}

prod_session() {  # runs INSIDE gpu-run (both cards): health, server, steps, teardown
  local d=$win/prod srv=$win/prod/server.log server_pid= rc_all=0
  mkdir -p "$d"
  finish() {
    trap - EXIT; set +e
    [[ -n $server_pid ]] && { kill -TERM "$server_pid" 2>/dev/null; timeout 60 tail --pid="$server_pid" -f /dev/null; kill -KILL "$server_pid" 2>/dev/null; }
    if grep -Eq 'DEVICE_LOST|WEDGED|HUNG' "$srv" "$d/health-pre.log" 2>/dev/null; then "$host/bin/xe-reset" > "$d/reset.log" 2>&1; fi
    timeout 150 ${WIN_TEST_HEALTH:-$host/bin/xpu-health} > "$d/health-post.log" 2>&1; local h=$?
    gate G2-health-post hard "$( ((h)) && echo FAIL || echo PASS)" "$(tail -1 "$d/health-post.log")"
    ((h)) && rc_all=1
    exit $rc_all
  }
  trap finish EXIT
  local health=${WIN_TEST_HEALTH:-$host/bin/xpu-health}  # WIN_TEST_* = CPU self-test hooks only (selftest.sh)
  timeout 150 $health > "$d/health-pre.log" 2>&1 || { gate G2-health-pre hard FAIL "$(tail -1 "$d/health-pre.log")"; rc_all=1; exit 1; }
  gate G2-health-pre hard PASS "$(tail -1 "$d/health-pre.log")"
  local t0=$SECONDS
  if [[ -n ${WIN_TEST_SERVER:-} ]]; then bash -c "$WIN_TEST_SERVER" > "$srv" 2>&1 &
  else TF_DD_CONFIG=$config TF_LISTEN=tcp TF_PORT=$port bash "$here/dd_launch_tf.sh" > "$srv" 2>&1 & fi
  server_pid=$!
  local ready=0 i
  for ((i=0;i<600;i++)); do
    curl -fs --max-time 2 "http://127.0.0.1:$port/v1/models" > "$d/models.json" && { ready=1; break; }
    kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
  if ((!ready)); then gate G2-load hard FAIL "not ready after $((SECONDS-t0)) s: $(tail -3 "$srv" | tr '\n' ' ' | cut -c1-300)"; rc_all=1; exit 1; fi
  gate G2-load hard PASS "ready in $((SECONDS-t0)) s; $(grep -m1 'Intel GPU engine' "$srv" | cut -c1-220)"
  export TF_URL=http://127.0.0.1:$port TF_PORT=$port TF_MODEL=hotschmoe-dd TF_WT=$wt TF_SERVER_LOG=$srv TF_FIXTURE=$fixture
  export TF_REF=$ref TF_HERE=$here TF_HOST=$host TF_WIN=$win
  local id owner kind tmo cmd rc
  while IFS=$'\t' read -r id owner kind tmo cmd; do
    [[ -z $id || $id == \#* ]] && continue
    [[ -n ${WIN_ONLY:-} && " $WIN_ONLY " != *" $id "* ]] && continue
    export TF_OUT=$d/$id; mkdir -p "$TF_OUT"
    kill -0 "$server_pid" 2>/dev/null || { gate "$id" "$kind" FAIL "server process gone before step"; rc_all=1; break; }
    log "step $id ($owner, $kind, <= $tmo s)"
    ( set -a; source "$config" >/dev/null 2>&1; set +a; cd "$TF_OUT"; timeout --kill-after=10 "$tmo" bash -c "$cmd" ) > "$TF_OUT/step.log" 2>&1
    rc=$?
    gate "$id" "$kind" "$( ((rc)) && echo FAIL || echo PASS)" "rc=$rc $(grep -h '^GATE' "$TF_OUT/step.log" | tail -3 | tr '\n' ' ' | cut -c1-600)"
    if ((rc)) && [[ $kind == hard ]]; then rc_all=1; log "HARD step $id failed: stopping the prod session"; break; fi
  done < "$steps"
  local out; out=$(python3 "$here/gpu_window_checks.py" logscan "$srv"); local c=$?
  gate G2-server-log hard "$( ((c)) && echo FAIL || echo PASS)" "$out"; ((c)) && rc_all=1
  exit $rc_all
}

prod() {  # G2 ~20 min: one lease, quiet CPU outside it
  log "G2 prod start (cards 0+1, <= ${WIN_LEASE_S:-2400} s, port $port)"; gpumd "claim cards 0+1 <= ${WIN_LEASE_S:-2400} s: gpu_window prod $tag (port $port${WIN_ONLY:+, steps $WIN_ONLY})"
  WIN_DIR=$win WIN_TAG=$tag TF_DD_CONFIG=$config WIN_STEPS=$steps "$host/bin/cpu-run" --quiet "$host/bin/gpu-run" timeout --signal=TERM --kill-after=30 ${WIN_LEASE_S:-2400} bash "$here/gpu_window.sh" _prod_session
  local rc=$?
  gpumd "released cards 0+1: gpu_window prod $tag rc=$rc"
  if ((rc == 124 || rc == 137)); then
    log "prod timed out: recovery health"; "$host/bin/gpu-run" timeout 300 bash -c '"$1/bin/xe-reset"; "$1/bin/xpu-health"' bash "$host" > "$win/prod/recovery.log" 2>&1
  fi
  return $rc
}

soak_backend() {  # the passive record follows the serving backend via a drop-in (removed again on rollback)
  local d=$HOME/.config/systemd/user/hotschmoe-dd-soak.service.d
  if [[ $1 == tf ]]; then mkdir -p "$d"; printf '[Service]\nExecStart=\nExecStart=/usr/bin/python3 %s/soak_monitor.py --backend tf\n' "$here" > "$d/backend-tf.conf"
  else rm -f "$d/backend-tf.conf"; fi
  systemctl --user daemon-reload; systemctl --user try-restart hotschmoe-dd-soak 2>/dev/null; log "soak monitor backend -> $1"
}

hard_failures() { awk -F'\t' '$3=="hard" && $4=="FAIL"' "$win/gates.tsv" 2>/dev/null | wc -l; }

strata_up() {
  log "starting Strata (hotschmoe-dd)"; systemctl --user enable hotschmoe-dd >/dev/null 2>&1; systemctl --user start hotschmoe-dd
  if wait_http http://127.0.0.1:18080/v1/models 600 key; then log "Strata answers on :18080"; return 0; fi
  log "Strata did NOT come back within 600 s: see /mnt/vm_8tb/b70/logs/hotschmoe-dd/server.log; try bin/xe-reset then start again"; return 1
}

window() {
  [[ ${CONFIRM:-} == yes ]] || die "window stops production Strata: rerun with CONFIRM=yes"
  preflight || die "preflight failed"
  echo "hotschmoe-dd-tf" > "$hold"; trap 'rm -f "$hold"' EXIT
  log "stopping Strata (hotschmoe-dd) for the window"; systemctl --user stop hotschmoe-dd
  for i in $(seq 60); do "$host/bin/gpu-run" --status 2>/dev/null | grep -q HELD || break; sleep 2; done
  regress; prod
  rm -f "$hold"
  local n; n=$(hard_failures)
  if ((n)); then log "$n HARD gate(s) failed -> no cutover; restoring Strata"; strata_up; return 1; fi
  log "all HARD gates PASS; soft results:"; awk -F'\t' '$3=="soft"' "$win/gates.tsv" | tee -a "$win/window.log"
  log "next: CONFIRM=yes WIN_DIR=$win $0 cutover   (or: $0 rollback to restore Strata now)"
}

functional_18080() {
  timeout 900 python3 "$host/strata/flash-next/dd/dd_functional.py" --image-fixture "$fixture" > "$win/functional-18080.json" 2>&1
}

cutover() {
  [[ ${CONFIRM:-} == yes ]] || die "cutover switches production: rerun with CONFIRM=yes"
  [[ -s $win/gates.tsv ]] || die "no gates.tsv in $win: run the window first (or pass WIN_DIR)"
  (( $(hard_failures) == 0 )) || die "hard gate failures in $win/gates.tsv"
  grep -q 'G2-load' "$win/gates.tsv" || die "no prod run recorded in $win"
  local u=$HOME/.config/systemd/user
  cp "$here/hotschmoe-dd-tensorfold.service" "$u/" && systemctl --user daemon-reload
  log "cutover: stop hotschmoe-dd, start hotschmoe-dd-tensorfold ($TF_COMMIT)"
  systemctl --user stop hotschmoe-dd
  TF_START=$SECONDS; systemctl --user start hotschmoe-dd-tensorfold
  if ! wait_http http://127.0.0.1:18080/v1/models 900 key; then gate C1-start hard FAIL "TF unit not answering on :18080"; rollback; return 1; fi
  gate C1-start hard PASS "TF behind :18080 in $((SECONDS-TF_START)) s"
  if ! functional_18080; then gate C2-functional-18080 hard FAIL "$(tail -c 600 "$win/functional-18080.json" | tr '\n' ' ')"; rollback; return 1; fi
  gate C2-functional-18080 hard PASS "$(tr -d '\n ' < "$win/functional-18080.json" | cut -c1-400)"
  systemctl --user disable hotschmoe-dd >/dev/null 2>&1; systemctl --user enable hotschmoe-dd-tensorfold >/dev/null 2>&1
  soak_backend tf
  log "CUTOVER DONE: hotschmoe-dd-tensorfold enabled, hotschmoe-dd disabled. Rollback: $0 rollback"
}

rollback() {
  log "rollback: stop hotschmoe-dd-tensorfold (ExecStopPost runs xpu-health)"
  systemctl --user stop hotschmoe-dd-tf-clamp hotschmoe-dd-tensorfold 2>/dev/null
  systemctl --user disable hotschmoe-dd-tensorfold >/dev/null 2>&1
  strata_up || return 1
  soak_backend strata
  if [[ ${ROLLBACK_FUNCTIONAL:-1} == 1 ]]; then functional_18080 && log "Strata functional PASS" || log "Strata functional FAIL: see $win/functional-18080.json"; fi
}

case ${1:-} in
  preflight) preflight ;;
  regress) regress ;;
  prod) prod ;;
  _prod_session) prod_session ;;
  window) window ;;
  cutover) cutover ;;
  rollback) rollback ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
