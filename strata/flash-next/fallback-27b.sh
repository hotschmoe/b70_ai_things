#!/usr/bin/env bash
# Stop Strata and start the retained 27B endpoint in a persistent user unit.
set -euo pipefail
repo=/mnt/vm_8tb/github/b70_ai_things
cd "$repo"
python3 vllm/int4/native5802_trial/serve.py check
if docker container inspect strata-ab-serve >/dev/null 2>&1; then
  docker stop -t 30 strata-ab-serve
fi
exec systemd-run --user --unit=b70-fallback-27b --collect \
  --property=WorkingDirectory="$repo" \
  --property=StandardOutput=append:/mnt/vm_8tb/b70/results/fallback-27b.log \
  --property=StandardError=append:/mnt/vm_8tb/b70/results/fallback-27b.log \
  /usr/bin/python3 "$repo/vllm/int4/native5802_trial/serve.py" start
