#!/usr/bin/env bash
# hotschmoe-dd: Strata Flash-Next daily driver (systemd: hotschmoe-dd.service).
# Default since 2026-11-20: Strata v0.2.16 port (hotschmoe-dd-0216.json, nl_launch_0216.sh).
# Roll back to night-final-v3: DD_CONFIG=$repo/strata/flash-next/dd/hotschmoe-dd.json DD_LAUNCHER=$repo/strata/flash-next/night/nl_launch.sh
# Backend on loopback 127.0.0.1:18124; the API-key frontdoor (hotschmoe-dd-frontdoor.service) serves 0.0.0.0:18080.
set -euo pipefail
repo=/mnt/vm_8tb/github/b70_ai_things
config=${DD_CONFIG:-$repo/strata/flash-next/dd/hotschmoe-dd-0216.json}
launcher=${DD_LAUNCHER:-$repo/strata/flash-next/dd/nl_launch_0216.sh}
for i in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 2; done   # boot: wait for docker
docker rm -f hotschmoe-dd >/dev/null 2>&1 || true
exec "$launcher" hotschmoe-dd "$config" 18124
