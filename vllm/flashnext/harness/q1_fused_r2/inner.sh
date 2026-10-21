#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
B=/mnt/vm_8tb/b70/build/vllm-flashnext
U=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace
export LD_LIBRARY_PATH="$B/o4/zelayers:$LD_LIBRARY_PATH"
exec timeout --signal=TERM --kill-after=15 240 "$U" -d --chrome-kernel-logging \
  --teardown-on-signal 10 --session "$Q1_UT_SESSION" --follow-child-process 0 \
  --output-dir-path "$Q1_PROFILE_OUT/trace" -o "$Q1_PROFILE_OUT/trace/timing.txt" \
  /opt/venv/bin/python "$S/probe.py" --output "$Q1_PROFILE_OUT" --repeats 12
