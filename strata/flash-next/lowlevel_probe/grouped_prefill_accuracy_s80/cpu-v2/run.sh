#!/usr/bin/env bash
# ROOT ONLY after explicit CPU-run GO; no DRI/network/GPU, no automatic build.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
R=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
command -v docker id mkdir tee >/dev/null
[[ -x "$P/build/reference_s80" && -w "$P" ]]
mkdir "$P/run-cpu"
mkdir "$P/run-cpu/tmp"
set +e
B70_AGENT=S80-CPU-fullGU-reference "$REPO/bin/cpu-run" docker run --rm --network none \
 -u "$(id -u):$(id -g)" -e TMPDIR=/out/tmp -e MKL_NUM_THREADS=4 -e OMP_NUM_THREADS=4 \
 -e MKL_DYNAMIC=FALSE -e OMP_DYNAMIC=FALSE -e MKL_INTERFACE_LAYER=LP64 \
 -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro -v "$P":/prep:ro \
 -v "$P/build":/build:ro -v "$P/run-cpu":/out -v "$P/run-cpu/tmp":/tmp \
 -v "$R/peer-grouped-gemm-s42-prep/run-clean/exports":/inputs:ro \
 --entrypoint bash "$IMAGE" /prep/run-container.sh 2>&1 | tee "$P/run-cpu/run.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$P/run-cpu/run.exit"
printf '%s\n' "${rc[1]}" > "$P/run-cpu/tee.exit"
[[ "${rc[0]}" = 0 && "${rc[1]}" = 0 ]]
