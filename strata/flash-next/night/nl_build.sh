#!/usr/bin/env bash
# Night-lab incremental build of the Strata SYCL engine in src-ep/build-sycl-ep; copies the exe to builds/strata-<tag>.
# Usage: nl_build.sh <tag>
set -euo pipefail
tag=${1:?tag}
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
CACHE=/mnt/vm_8tb/b70/cache/strata-aot-bmg-g31
OUT=/mnt/vm_8tb/b70/results/night-20261016/builds
mkdir -p "$OUT" "$OUT/tmp" "$OUT/cache"
git -C /mnt/vm_8tb/b70/build/strata-ab/src-ep diff > "$OUT/strata-$tag.diff" || true
git -C /mnt/vm_8tb/b70/build/strata-ab/src-ep rev-parse HEAD > "$OUT/strata-$tag.head"
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=night-build "$REPO/bin/cpu-run" docker run --rm --network none --cpus 8 -u "1000:$(id -g)" \
  -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_DIR=/out/cache \
  -e SYCL_CACHE_PERSISTENT=0 -e NEO_CACHE_PERSISTENT=1 -e NEO_CACHE_DIR=/offline-cache \
  -v /mnt/vm_8tb/b70/build/strata-ab:/work -v "$OUT":/out -v "$CACHE":/offline-cache \
  --entrypoint bash "$IMAGE" -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /work/src-ep/build-sycl-ep && time ninja -j6 strata && cp strata /out/strata-$tag" 2>&1 | tail -${NL_BUILD_TAIL:-6}
ls -la "$OUT/strata-$tag"
