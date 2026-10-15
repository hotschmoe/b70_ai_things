#!/usr/bin/env bash
# E5e: CPU llama-server, same prompts as E5d (is the SYCL garbage backend-specific?). Run under bin/cpu-run.
set -u; cd /mnt/vm_8tb/b70/build/backend-decision
docker rm -f bd-e5e >/dev/null 2>&1
bash e1-server.sh bd-e5e --alias hotschmoe-dd > e5e6-results/e5e-cpu-server.log 2>&1 &
for i in $(seq 1 200); do curl -sf 127.0.0.1:18190/health >/dev/null && break; sleep 2; done
timeout 900 python3 -I e5_discrim.py http://127.0.0.1:18190 e5e6-results/e5e-cpu.json; echo "rc=$?"
docker stop bd-e5e >/dev/null 2>&1; wait
