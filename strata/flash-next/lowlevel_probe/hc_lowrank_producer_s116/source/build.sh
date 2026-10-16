#!/usr/bin/env bash
# SOURCE ONLY. Root full-read + actual Opus/Fable CPU_BUILD_GO first.
set -euo pipefail
[[ $(id -u) == 1000 ]]
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
[[ ! -e "$HERE/build" ]]
mkdir "$HERE/build"
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
docker image inspect "$IMAGE" > "$HERE/build/image.json"
# cpu-run must wrap this script. No GPU lease/device required for offline AOT.
set +e
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" \
 --cap-drop ALL --security-opt no-new-privileges \
 -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_DIR=/out/cache \
 -e SYCL_CACHE_PERSISTENT=0 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro \
 -v "$HERE":/src:ro -v "$HERE/build":/out --entrypoint bash "$IMAGE" -c \
 'mkdir /out/tmp /out/cache; exec bash /src/build-container.sh' > "$HERE/build/build.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$HERE/build/build.exit"
exit "$code"
