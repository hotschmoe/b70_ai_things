#!/usr/bin/env bash
set -euo pipefail
[[ $(id -u) == 1000 ]]
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
R=$(dirname "$HERE")
OUT="$R/mtp-parallel-argmax-main-s114-audit"
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
command -v docker python3 mkdir tee >/dev/null
[[ ! -e "$OUT" && ! -e "$OUT.log" && ! -e "$OUT.exit" ]]
(cd "$HERE" && sha256sum -c expected-prep.sha256)
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
mkdir "$OUT"
set +e
B70_AGENT=S114-selected-main-audit B70_CPU_LOCK_TIMEOUT=0 "$REPO/bin/cpu-run" docker run --rm --network none --cpus 4 -u 1000:$(id -g)  -e TMPDIR="$OUT/tmp" -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e SYCL_CACHE_PERSISTENT=0  -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70:ro -v "$OUT":"$OUT"  --entrypoint python3 "$IMAGE" "$HERE/audit.py" "$@" 2>&1 | tee "$OUT.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$OUT.exit"
printf '%s\n' "${rc[1]}" > "$OUT.tee-exit"
[[ ${rc[0]} == 0 && ${rc[1]} == 0 ]]
