#!/usr/bin/env bash
# hotschmoe-dd: Strata Flash-Next night-final-v3 daily driver (systemd: hotschmoe-dd.service).
# Backend on loopback 127.0.0.1:18124; the API-key frontdoor (hotschmoe-dd-frontdoor.service) serves 0.0.0.0:18080.
set -euo pipefail
repo=/mnt/vm_8tb/github/b70_ai_things
config=${DD_CONFIG:-$repo/strata/flash-next/dd/hotschmoe-dd.json}
for i in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 2; done   # boot: wait for docker
docker rm -f hotschmoe-dd >/dev/null 2>&1 || true
exec "$repo/strata/flash-next/night/nl_launch.sh" hotschmoe-dd "$config" 18124
