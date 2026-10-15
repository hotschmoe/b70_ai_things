#!/usr/bin/env bash
# E1 MTP arm: evict phase on a fresh server, then control phase on a fresh server. Run under cpu-run.
set -u
cd /mnt/vm_8tb/b70/build/backend-decision
ARGS="--spec-type draft-mtp -md /mtp/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf --spec-draft-n-max 3"
for PH in evict control; do
  ./e1-server.sh bd-e1-mtp $ARGS > e1-srv-mtp-$PH.log 2>&1 &
  for i in $(seq 1 300); do curl -sf 127.0.0.1:18190/health >/dev/null && break; docker ps -q -f name=bd-e1-mtp | grep -q . || [ $i -lt 10 ] || { echo "server died"; tail -20 e1-srv-mtp-$PH.log; exit 1; }; sleep 2; done
  python3 -I e1_restore.py http://127.0.0.1:18190 e1-$PH-mtp.json 32 24 $PH | tee e1-$PH-mtp.txt
  docker stop bd-e1-mtp >/dev/null; wait
done
