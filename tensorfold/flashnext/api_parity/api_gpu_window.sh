#!/usr/bin/env bash
# API quick GPU gate on port 18154 (worker API). Run under the lease:
#   B70_AGENT=API /mnt/vm_8tb/github/b70_ai_things/bin/gpu-run timeout 295 bash api_gpu_window.sh
# Starts the real server (b70/dd-api build, tf-dd-api.env), runs dd/api_gate.py all, stops it, runs xpu-health.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dd=/mnt/vm_8tb/github/b70_ai_things/tensorfold/flashnext/dd
out=${API_OUT:-/mnt/vm_8tb/b70/results/tf-flashnext/workers/API/gpu1}
port=18154
mkdir -p "$out"
TF_DD_CONFIG="$here/tf-dd-api.env" TF_LISTEN=tcp TF_PORT=$port bash "$dd/dd_launch_tf.sh" >"$out/server.log" 2>&1 &
spid=$!
ready=0
for i in $(seq 1 150); do
  curl -s -m 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $spid 2>/dev/null || break
  sleep 1
done
echo "ready=$ready after ~$i s" | tee "$out/window.log"
rc=1
if [[ $ready == 1 ]]; then
  for g in reasoning tools forced budget edge; do
    timeout 120 python3 "$dd/api_gate.py" $g "http://127.0.0.1:$port" 2>&1 | tee -a "$out/gates.log"
  done
  rc=0
fi
kill -TERM $spid 2>/dev/null; for i in $(seq 1 40); do kill -0 $spid 2>/dev/null || break; sleep 1; done; kill -KILL $spid 2>/dev/null
grep -c . "$out/server.log" | sed 's/^/server log lines: /'
timeout 60 /mnt/vm_8tb/github/b70_ai_things/bin/xpu-health 2>&1 | tail -3 | tee -a "$out/window.log"
exit $rc
