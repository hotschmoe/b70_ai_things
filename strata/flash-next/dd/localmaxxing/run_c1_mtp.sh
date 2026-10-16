#!/usr/bin/env bash
# localmaxxing C1 MTP speed test against the live hotschmoe-dd Strata backend; token counter before/after
# detects an interleaved user request (the run is then discarded).
set -euo pipefail
cd /mnt/vm_8tb/b70/results/localmaxxing
tok() { curl -s 127.0.0.1:18124/metrics/prometheus | awk '/^llamacpp:tokens_predicted_total/{print $2}'; }
busy() { curl -s 127.0.0.1:18124/slots | python3 -c 'import json,sys; print(sum(1 for s in json.load(sys.stdin) if s.get("is_processing")))'; }
echo "busy_before=$(busy) tok_before=$(tok) $(date -u +%FT%TZ)"
~/.local/bin/lmx speed-test run strata --mode remote --base-url http://127.0.0.1:18124/v1 --served-model hotschmoe-dd \
  --hf-id ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --quantization IQ3_XXS --hardware hardware.json --backend sycl \
  --engine-version 0.1.44.1 --engine-commit 296b59d2 --engine-build "night-20261016 local SYCL patches (e753e9cb)" \
  --spec-decoding --spec-method MTP --spec-num-tokens 4 --max-tokens 512 --runs-dir runs ${LMX_EXTRA:-} --json > c1_mtp.json 2> c1_mtp.err
echo "busy_after=$(busy) tok_after=$(tok) $(date -u +%FT%TZ)"
