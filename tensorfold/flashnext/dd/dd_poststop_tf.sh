#!/usr/bin/env bash
# ExecStopPost for hotschmoe-dd-tensorfold.service: after the server exits, reset if the log shows
# a device loss, then run xpu-health under the lease. Never fails the unit (ExecStopPost=-).
set -uo pipefail
host=/mnt/vm_8tb/github/b70_ai_things
log=/mnt/vm_8tb/b70/logs/hotschmoe-dd/tf-server.log
ts() { date -u '+%F %T UTC'; }
echo "$(ts) [dd_poststop_tf] service_result=${SERVICE_RESULT:-?} exit_code=${EXIT_CODE:-?} exit_status=${EXIT_STATUS:-?}"
# Only inspect output of the last launch.
start=$(grep -n '\[dd_launch_tf\] variant=' "$log" 2>/dev/null | tail -1 | cut -d: -f1)
if [[ -n $start ]] && tail -n +"$start" "$log" | grep -Eq 'DEVICE_LOST|ZE_RESULT_ERROR_DEVICE_LOST|WEDGED|HUNG'; then
  echo "$(ts) [dd_poststop_tf] device loss in server log -> bin/xe-reset"
  timeout 420 "$host/bin/xe-reset"; echo "$(ts) [dd_poststop_tf] xe-reset rc=$?"
fi
# Short lease wait: if another job took the cards right after exit, skip (it runs its own health checks).
B70_GPU_LOCK_TIMEOUT=60 timeout 220 "$host/bin/gpu-run" timeout 150 "$host/bin/xpu-health"
rc=$?
echo "$(ts) [dd_poststop_tf] xpu-health rc=$rc (124 = lease busy or timeout: skipped; the next lease holder runs its own health)"
exit 0
