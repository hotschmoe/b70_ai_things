#!/usr/bin/env bash
set -u; cd /mnt/vm_8tb/b70/build/backend-decision; R=/mnt/vm_8tb/github/b70_ai_things; mkdir -p e3-results
echo "== eager skipped (graph run repeats eager)"
echo "== graph"; E3_TIMEOUT=180 ./e3-xccl.sh 1 | tee e3-results/graph.txt
echo "== post-health"; $R/bin/xpu-health > e3-results/post-health.txt 2>&1; echo "xpu-health rc=$?"
$R/bin/xpu-collective-health > e3-results/post-collective.txt 2>&1; echo "collective rc=$?"
