#!/usr/bin/env bash
set -u; cd /mnt/vm_8tb/b70/build/backend-decision
./e1-server.sh bd-e1-conc --spec-type draft-mtp -md /mtp/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf --spec-draft-n-max 3 > e1-srv-conc.log 2>&1 &
for i in $(seq 1 300); do curl -sf 127.0.0.1:18190/health >/dev/null && break; sleep 2; done
python3 -I e1_conc.py | tee e1-conc-mtp.txt
docker stop bd-e1-conc >/dev/null; wait
