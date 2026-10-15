#!/usr/bin/env bash
# Root must explicitly select/authorize this build. No automatic runtime step.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
OUT=${1:-$P/build-s74}
command -v docker sha256sum id mkdir >/dev/null
[[ -x "$REPO/bin/cpu-run" && "$OUT" == /* ]]
# Fresh directory, owned by this caller, avoids inherited root-owned ninja/logs.
# mkdir fails if the path already exists: no output or old evidence is replaced.
mkdir -- "$OUT"
mkdir -- "$OUT/tmp"
[[ -w "$OUT" && -w "$OUT/tmp" ]]
exec "$REPO/bin/cpu-run" docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$P,dst=/src,readonly" \
  --mount "type=bind,src=$OUT,dst=/out" \
  --mount "type=bind,src=$OUT/tmp,dst=/tmp" \
  --workdir /out --entrypoint bash "$IMAGE" /src/build-container.sh
