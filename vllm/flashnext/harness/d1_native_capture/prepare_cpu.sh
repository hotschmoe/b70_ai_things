#!/bin/bash
# Caller must wrap the whole script in bin/cpu-run MAX_JOBS=4. No GPU.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_ROOT=/mnt/vm_8tb/b70/build/vllm-flashnext/d1/native_capture
OUT="$BUILD_ROOT/setup-r1"
EXPECTED=sha256:ea234fbf7123cba58b387f4046a0c32192dd5b6febead3684dbd8690882bf572
BASE_ALIAS=vllm-flashnext-xpu:d1-base-ea234fbf7123cba5
IMAGE=vllm-flashnext-xpu:d1-gdb-r1
[ ! -e "$OUT" ] || { echo "Refuse existing output $OUT"; exit 2; }
mkdir -p "$OUT" "$BUILD_ROOT/tmp"
export TMPDIR="$BUILD_ROOT/tmp" MAX_JOBS=4
actual=$(docker image inspect vllm-flashnext-xpu:py312 --format '{{.Id}}')
[ "$actual" = "$EXPECTED" ] || { echo 'Parent image changed; review required'; exit 3; }
docker image inspect vllm-flashnext-xpu:py312 --format '{{json .}}' > "$OUT/base-image.json"
docker tag "$EXPECTED" "$BASE_ALIAS"
[ "$(docker image inspect "$BASE_ALIAS" --format '{{.Id}}')" = "$EXPECTED" ]
docker build --build-arg "D1_BASE_IMAGE=$BASE_ALIAS" -t "$IMAGE" -f "$ROOT/Dockerfile.gdb" "$ROOT" > "$OUT/image-build.log" 2>&1
[ "$(docker image inspect "$BASE_ALIAS" --format '{{.Id}}')" = "$EXPECTED" ]
[ "$(docker image inspect vllm-flashnext-xpu:py312 --format '{{.Id}}')" = "$EXPECTED" ]
docker image inspect "$IMAGE" --format '{{json .}}' > "$OUT/debug-image.json"
timeout --signal=TERM --kill-after=3 25s docker run --rm --network none \
  --cap-add SYS_PTRACE --ulimit core=0 --memory 2g --pids-limit 256 --user 0 \
  -v "$ROOT:$ROOT:ro" -v "$OUT:$OUT" -w "$OUT" --entrypoint /usr/bin/python3 "$IMAGE" \
  "$ROOT/cpu_attach_check.py" "$ROOT" "$OUT/attach" > "$OUT/attach-check.log" 2>&1
docker run --rm --network none --entrypoint /bin/cat "$IMAGE" /opt/d1-gdb-package-delta.json > "$OUT/package-delta.json"
echo 'D1_CPU_DEBUG_IMAGE_AND_ATTACH_PASS'
cat "$OUT/attach/result.json"
