#!/usr/bin/env bash
# ROOT ONLY after source review; never run during another agent's quiet window.
set -euo pipefail
P=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
NATIVE=/mnt/vm_8tb/b70/build/strata-ab/src-ep
PTI=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
[[ $(id -u) == 1000 && $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
mkdir "$P/build"
mkdir "$P/build/tmp" "$P/build/cache"
set +e
B70_AGENT=S103-offline-build "$REPO/bin/cpu-run" docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" \
  -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_PERSISTENT=0 -e MAX_JOBS=4 \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -v "$P":/src:ro -v "$P/build":/out \
  -v "$NATIVE":/native:ro -v "$PTI":"$PTI":ro --entrypoint bash "$IMAGE" /src/build-container.sh 2>&1 | tee "$P/build/build.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$P/build/build.exit"
printf '%s\n' "${rc[1]}" > "$P/build/tee.exit"
[[ ${rc[0]} == 0 && ${rc[1]} == 0 ]]
