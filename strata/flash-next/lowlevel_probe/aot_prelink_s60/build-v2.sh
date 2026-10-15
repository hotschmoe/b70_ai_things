#!/usr/bin/env bash
# ROOT ONLY after exact model/source review. No GPU device is mounted.
set -euo pipefail
HERE=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq4xs-rpw2-s59-prep
EXEC_DIR=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/aot-reuse-s60-execution
REPO=/mnt/vm_8tb/github/b70_ai_things
[ -d "$HERE/build/tmp" ] && [ -d "$HERE/build/cache" ]
[ ! -e "$HERE/build/build.exit" ]
EXPECTED_IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
docker image inspect "$EXPECTED_IMAGE" > "$HERE/build/image.json"
IMAGE=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[0]["Id"])' "$HERE/build/image.json")
[ "$IMAGE" = "$EXPECTED_IMAGE" ]
set +e
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=S60-supported-prelink "$REPO/bin/cpu-run" \
  env MAX_JOBS=4 docker run --rm --network none -u "$(id -u):$(id -g)" \
  -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_DIR=/out/cache \
  -e SYCL_CACHE_PERSISTENT=0 -e MAX_JOBS=4 \
  -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro -v "$HERE":/src:ro -v "$HERE/build":/out -v "$EXEC_DIR":/execution:ro \
  --entrypoint bash "$IMAGE" /execution/build-container-v2.sh 2>&1 | tee "$HERE/build/build.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$HERE/build/build.exit"
printf '%s\n' "${rc[1]}" > "$HERE/build/tee.exit"
[ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ]
