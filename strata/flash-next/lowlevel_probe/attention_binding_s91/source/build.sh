#!/usr/bin/env bash
# ROOT ONLY, after exact source/model review. This preparation task does not execute it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
mkdir "$HERE/build"; mkdir "$HERE/build/tmp" "$HERE/build/cache"
docker image inspect "$IMAGE" > "$HERE/build/image.json"
set +e
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=S91-score-reduce12-build "$REPO/bin/cpu-run" docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" \
 -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_PERSISTENT=0 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro -v "$HERE":/src:ro -v "$HERE/build":/out \
 --entrypoint bash "$IMAGE" /src/build-container.sh > "$HERE/build/build.log" 2>&1
rc=$?;set -e
printf '%s\n' "$rc" > "$HERE/build/build.exit"
exit "$rc"
