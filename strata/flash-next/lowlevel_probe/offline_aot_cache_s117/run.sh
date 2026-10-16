#!/usr/bin/env bash
# FROZEN SOURCE PREPARATION. ROOT GO required before ONE future invocation.
set -euo pipefail
[[ $(id -u) == 1000 ]]
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOTDIR=$(dirname "$HERE")
OUT="$ROOTDIR/aot-cache-discriminator-run"
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
REPO=/mnt/vm_8tb/github/b70_ai_things
command -v docker python3 sha256sum tee >/dev/null
[[ ! -e "$OUT" && ! -e "$OUT.log" && ! -e "$OUT.exit" ]]
(cd "$HERE" && sha256sum -c expected-prep.sha256)
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
mkdir "$OUT" "$OUT/tmp"
docker image inspect "$IMAGE" > "$OUT.image.json"
set +e
B70_AGENT=offline-cache-tiny-discriminator B70_CPU_LOCK_TIMEOUT=0 "$REPO/bin/cpu-run" docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
 -e TMPDIR=/out/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e SYCL_CACHE_PERSISTENT=0 -e NEO_CACHE_STATS=1 \
 -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70:ro -v "$HERE":/prep:ro -v "$OUT":/out \
 --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && $(id -u) == 1000 ]]
command -v python3 ocloc ldd >/dev/null
exec python3 /prep/run-cache.py
' 2>&1 | tee "$OUT.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$OUT.exit"
printf '%s\n' "${rc[1]}" > "$OUT.tee-exit"
[[ ${rc[0]} == 0 && ${rc[1]} == 0 ]]
