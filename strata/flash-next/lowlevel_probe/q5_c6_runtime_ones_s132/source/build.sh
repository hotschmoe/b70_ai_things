#!/usr/bin/env bash
# ROOT ONLY after exact source and actual model review. Offline CPU build.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
R=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
command -v docker python3 id mkdir tee >/dev/null
[[ -w "$HERE" && -x "$REPO/bin/cpu-run" ]]
mkdir "$HERE/build"
mkdir "$HERE/build/tmp" "$HERE/build/cache"
docker image inspect "$IMAGE" > "$HERE/build/image.json"
python3 - "$HERE/build/image.json" "$IMAGE" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))[0]['Id']==sys.argv[2]
PY
set +e
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=Q5-C6-runtime-ones-adapter-only-build "$REPO/bin/cpu-run" \
  env MAX_JOBS=4 docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" \
  -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_DIR=/out/cache \
  -e SYCL_CACHE_PERSISTENT=0 -e MAX_JOBS=4 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro -v "$HERE":/src:ro -v "$HERE/build":/out \
  -v "$R/iq4xs-rpw2-s59-prep/build":/common:ro \
  -v "$R/iq4xs-rpw2-s59-prep/frozen-inputs":/frozen:ro \
  --entrypoint bash "$IMAGE" /src/build-container.sh 2>&1 | tee "$HERE/build/build.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$HERE/build/build.exit"
printf '%s\n' "${rc[1]}" > "$HERE/build/tee.exit"
[ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ]
