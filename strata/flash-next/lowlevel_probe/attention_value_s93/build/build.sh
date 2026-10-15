#!/usr/bin/env bash
# ROOT ONLY after full script read, exact reviewed source application and CAP2051 PASS.
# No auto apply, run, ownership repair or retry.
set -euo pipefail
[[ $(id -u) == 1000 ]]
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
command -v docker python3 mkdir id tee >/dev/null
mkdir "$HERE/build"
mkdir "$HERE/build/tmp" "$HERE/build/cache"
docker image inspect "$IMAGE" > "$HERE/build/image.json"
python3 - "$HERE/build/image.json" "$IMAGE" <<'PYIMAGE'
import json,sys
assert json.load(open(sys.argv[1]))[0]['Id']==sys.argv[2]
PYIMAGE
set +e
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=S93-value-main-incremental "$REPO/bin/cpu-run" env MAX_JOBS=4 docker run --rm --network none --cpus 4 -u "1000:$(id -g)" \
 -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_DIR=/out/cache \
 -e SYCL_CACHE_PERSISTENT=0 -e MAX_JOBS=4 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -v /mnt/vm_8tb/b70/build/strata-ab:/work -v "$HERE":/prep:ro -v "$HERE/build":/out \
 --entrypoint bash "$IMAGE" /prep/build-container.sh 2>&1 | tee "$HERE/build/build.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$HERE/build/build.exit"
printf '%s\n' "${rc[1]}" > "$HERE/build/tee.exit"
[ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ]
