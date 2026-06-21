#!/usr/bin/env bash
# qrun NAME cmd... -- launch a GPU job DETACHED under gpu-run (flock lease), logging to
# results/NAME.log with a "QRUN_EXIT <rc>" sentinel appended on completion (pollable from afar).
# Run it under setsid over ssh so it survives the ssh session closing:
#   ssh host "cd /mnt/vm_8tb/b70 && SRC=.. OUT=.. setsid ./qrun.sh Q1_smoke bash 65_autoround_w8a8.sh"
# Job env (SRC/OUT/ITERS/...) is inherited from the caller through to gpu-run and the job script.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
NAME="${1:?usage: qrun NAME cmd...}"; shift
LOG="$ROOT/results/$NAME.log"
mkdir -p "$ROOT/results"
export B70_AGENT="loop-$NAME"
{
  echo "QRUN_START $NAME $(date '+%F %T') :: $*"
  "$ROOT/gpu-run" "$@"
  echo "QRUN_EXIT $? $NAME $(date '+%F %T')"
} >> "$LOG" 2>&1 &
echo "launched $NAME pid=$! -> $LOG"
