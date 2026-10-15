#!/usr/bin/env bash
# ROOT ONLY after exact-source/model review. CPU-only; never executed by preparer.
set -euo pipefail
P=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/hc-more-groups-s73-prep
R=${P%/*}
REPO=/mnt/vm_8tb/github/b70_ai_things
mkdir "$P/build"
mkdir "$P/build/tmp" "$P/build/cache"
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=S73-offline "$REPO/bin/cpu-run" env MAX_JOBS=4 docker run --rm --network none -u "$(id -u):$(id -g)" -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_DIR=/out/cache -e SYCL_CACHE_PERSISTENT=0 -e MAX_JOBS=4 -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro -v "$P":/src:ro -v "$P/build":/out -v "$R/iq4xs-rpw2-s59-prep/frozen-inputs":/frozen:ro -v "$R/iq4xs-rpw2-s59-prep/build":/common:ro --entrypoint bash sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2 /src/build-container.sh
