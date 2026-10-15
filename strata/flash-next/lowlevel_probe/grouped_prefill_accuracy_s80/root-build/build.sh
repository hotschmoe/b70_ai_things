#!/usr/bin/env bash
# ROOT ONLY after source/helper/model review. Known S42 dependency-complete link.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PREP=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/grouped-prefill-accuracy-s80-prep
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
mkdir "$HERE/build"
mkdir "$HERE/build/tmp"
docker image inspect "$IMAGE" > "$HERE/build/image.json"
set +e
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=S80-adapter-build "$REPO/bin/cpu-run" env MAX_JOBS=4 docker run --rm --network none -u "$(id -u):$(id -g)" \
 -e TMPDIR=/out/tmp -e MAX_JOBS=4 -e SYCL_CACHE_PERSISTENT=0 \
 -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro -v "$PREP":/src:ro -v "$HERE":/recipe:ro -v "$HERE/build":/out -v "$PREP/frozen-inputs":/frozen:ro \
 --entrypoint bash "$IMAGE" /recipe/build-container.sh 2>&1 | tee "$HERE/build/build.log"
rc=("${PIPESTATUS[@]}")
printf '%s\n' "${rc[0]}" > "$HERE/build/build.exit"
printf '%s\n' "${rc[1]}" > "$HERE/build/tee.exit"
[ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ]
