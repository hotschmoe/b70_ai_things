#!/usr/bin/env bash
set -euo pipefail
cd /mnt/vm_8tb/b70/results/localmaxxing
tok() { curl -s 127.0.0.1:18124/metrics/prometheus | awk '/^llamacpp:tokens_predicted_total/{print $2}'; }
busy() { curl -s 127.0.0.1:18124/slots | python3 -c 'import json,sys; print(sum(1 for s in json.load(sys.stdin) if s.get("is_processing")))'; }
tpl=runs/ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/20261016T164951Z.json
for cid in code-v1 reasoning-v1; do
  echo "$cid busy_before=$(busy) tok_before=$(tok) $(date -u +%FT%TZ)"
  python3 lmx_verified.py $cid $tpl payload_$cid.json 512
  echo "$cid busy_after=$(busy) tok_after=$(tok) $(date -u +%FT%TZ)"
done
