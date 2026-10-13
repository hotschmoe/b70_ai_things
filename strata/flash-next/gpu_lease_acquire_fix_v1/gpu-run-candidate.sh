#!/usr/bin/env bash
# gpu-run -- serialize Arc Pro B70 GPU access across parallel agents/containers, PER CARD.
#
# Why: agents edit/compile in parallel (CPU-bound, no GPU contention), but the actual GPU test/bench
# runs must NOT overlap ON THE SAME CARD. Overlap corrupts perf timings (noisy neighbor) and lets one
# buggy WIP kernel wedge a card under another run. This wraps any command in an flock on a per-card
# lockfile so exactly one run executes per card at a time. Editing/compiling stay fully parallel.
#
# Two B70s -> two locks (gpu.lock.0, gpu.lock.1). By DEFAULT a run locks BOTH cards (correct for a
# whole-box run: TP=2 / data-parallel / PP=2 use both cards, and a single-card run that does not say
# otherwise is treated conservatively as whole-box). Pass --card N to lock ONLY card N -- this is how
# you run a model on one card while leaving the other FREE for experiments on the other --card.
#
# Usage:
#   gpu-run <cmd> [args...]              # lock BOTH cards (whole box) -- TP=2 / DP / PP / default
#   gpu-run --card 0 <cmd>              # lock ONLY card 0  (leaves card 1 free for --card 1 runs)
#   gpu-run --card 1 <cmd>             # lock ONLY card 1
#   gpu-run --status                    # per-card holder (or "free")
# Pair --card N with the workload's own card pin (e.g. serve.sh DEVICE=N / ZE_AFFINITY_MASK=N).
#
# Cross-container: the lockfiles are host paths bind-mounted into every container, so all agents flock
# the SAME inodes and the lease is honored across containers on the same host.
#
# Env (override as needed):
#   B70_GPU_LOCK          lock BASE path    (default /mnt/vm_8tb/b70/gpu.lock; per-card = <base>.0/.1)
#   B70_GPU_LOCK_TIMEOUT  max wait seconds  (default 600; 0 = wait forever)
#   B70_AGENT             label for logs    (default: $HOSTNAME)
set -uo pipefail

LOCK="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"
TIMEOUT="${B70_GPU_LOCK_TIMEOUT:-600}"
AGENT="${B70_AGENT:-$(hostname)}"
L0="$LOCK.0"; L1="$LOCK.1"; O0="$L0.owner"; O1="$L1.owner"

card_status() {  # $1=label $2=ownerfile
  if [ -s "$2" ]; then echo "  card $1: HELD  $(cat "$2")"; else echo "  card $1: free"; fi
}

CARD=""
case "${1:-}" in
  --status|--who) echo "GPU lease:"; card_status 0 "$O0"; card_status 1 "$O1"; exit 0 ;;
  --card) CARD="${2:-}"; shift 2 || true ;;
  --card=*) CARD="${1#--card=}"; shift ;;
  -h|--help|"") awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
esac
[ -n "$CARD" ] && [ "$CARD" != 0 ] && [ "$CARD" != 1 ] && { echo "gpu-run: --card must be 0 or 1" >&2; exit 2; }

command -v flock >/dev/null 2>&1 || { echo "gpu-run: flock not found (install util-linux)" >&2; exit 127; }
lockdir="$(dirname "$LOCK")"
[ -d "$lockdir" ] || { echo "gpu-run: lock dir '$lockdir' does not exist (set B70_GPU_LOCK to a host path)." >&2; exit 1; }

# Which cards does this run need? --card N -> just N; default -> both (ordered 0 then 1, no deadlock).
case "$CARD" in
  0) CARDS=(0) ;;
  1) CARDS=(1) ;;
  *) CARDS=(0 1) ;;
esac

CMDLINE="$*"   # the actual user command (for the owner record), captured before acquire() shadows $*
OWNED=()
WAITER=""
RUNNING=0
INTERRUPTED=0
cleanup() { for f in "${OWNED[@]+"${OWNED[@]}"}"; do : > "$f" 2>/dev/null || true; done; }
interrupted() {
  local status="$1"
  # Keep the foreground workload and lease alive until its own retirement.
  # For immediate active cancellation, signal that owned workload directly.
  if [ "$RUNNING" = 1 ]; then INTERRUPTED="$status"; return; fi
  local waiter
  # jobs covers a signal between spawning the lock waiter and recording $!.
  for waiter in ${WAITER:-$(jobs -pr)}; do
    kill -TERM "$waiter" 2>/dev/null || true
    wait "$waiter" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT
trap 'interrupted 130' INT
trap 'interrupted 143' TERM

acquire() {  # $1=card 0|1 ; $2=fd ; $3=lockfile ; $4=ownerfile
  eval "exec $2>\"$3\"" || { echo "gpu-run: cannot open lock $3" >&2; exit 1; }
  flock -n "$2"
  local status=$?
  if [ "$status" = 1 ]; then
    echo "gpu-run: card $1 busy (held by: $(cat "$4" 2>/dev/null || echo unknown)); waiting up to ${TIMEOUT}s ..." >&2
    if [ "$TIMEOUT" = 0 ]; then flock "$2" & else flock -w "$TIMEOUT" "$2" & fi
    WAITER=$!
    wait "$WAITER"
    status=$?
    WAITER=""
    if [ "$status" != 0 ]; then
      echo "gpu-run: lock acquisition failed (exit $status) on card $1; command will not run" >&2
      if [ "$TIMEOUT" != 0 ] && [ "$status" = 1 ]; then exit 124; fi
      exit "$status"
    fi
  elif [ "$status" != 0 ]; then
    echo "gpu-run: lock acquisition failed (exit $status) on card $1; command will not run" >&2
    exit "$status"
  fi
  # Register cleanup before publishing ownership, including signal windows.
  OWNED+=("$4")
  printf '%s pid=%s since=%s cmd: %s\n' "$AGENT" "$$" "$(date '+%F %T')" "$CMDLINE" > "$4" 2>/dev/null || true
}

for c in "${CARDS[@]}"; do
  if [ "$c" = 0 ]; then acquire 0 8 "$L0" "$O0"; else acquire 1 9 "$L1" "$O1"; fi
done

echo "gpu-run: [$AGENT] acquired card(s) ${CARDS[*]} -> $*" >&2
start=$(date +%s)
RUNNING=1
if [ "$INTERRUPTED" != 0 ]; then exit "$INTERRUPTED"; fi
"$@"
rc=$?
RUNNING=0
if [ "$INTERRUPTED" != 0 ]; then rc="$INTERRUPTED"; fi
echo "gpu-run: [$AGENT] done in $(( $(date +%s) - start ))s (exit $rc)" >&2
exit $rc
