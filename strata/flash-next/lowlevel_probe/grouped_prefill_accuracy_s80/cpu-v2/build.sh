#!/usr/bin/env bash
# ROOT ONLY after source GO; this builds only and never runs the reference.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
command -v docker python3 id mkdir tee >/dev/null
[[ -w "$P" && -x "$REPO/bin/cpu-run" ]]
mkdir "$P/build"
mkdir "$P/build/tmp"
docker image inspect "$IMAGE" > "$P/build/image.json"
python3 - "$P/build/image.json" "$IMAGE" <<'CHECK'
import json,sys
assert json.load(open(sys.argv[1]))[0]['Id']==sys.argv[2]
CHECK
set +e
B70_AGENT=S80-CPU-reference-build "$REPO/bin/cpu-run" docker run --rm --network none \
 -u "$(id -u):$(id -g)" -e TMPDIR=/out/tmp \
 -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro -v "$P":/prep:ro \
 -v "$P/build":/out -v "$P/build/tmp":/tmp \
 --entrypoint bash "$IMAGE" /prep/build-container.sh 2>&1 | tee "$P/build/build.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$P/build/build.exit"
printf '%s\n' "${rc[1]}" > "$P/build/tee.exit"
[[ "${rc[0]}" = 0 && "${rc[1]}" = 0 ]]
