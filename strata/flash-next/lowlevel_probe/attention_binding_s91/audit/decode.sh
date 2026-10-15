#!/usr/bin/env bash
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
mkdir "$P/tmp"
docker image inspect --format '{{.Id}}' "$IMAGE" > "$P/decoder-image-id.txt"
[[ $(cat "$P/decoder-image-id.txt") == "$IMAGE" ]]
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" -e TMPDIR=/w/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -v "$P":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri ]]
command -v ocloc > /w/decoder-tool-path.txt
for arm in public private;do
 sha256sum "/w/$arm.zebin" >> /w/decoder-input-before.sha256
 mkdir "/w/$arm-disasm"
 ocloc disasm -file "/w/$arm.zebin" -device bmg-g31 -dump "/w/$arm-disasm" > "/w/$arm-disasm.log" 2> "/w/$arm-disasm.stderr"
 printf "%s\n" "$?" > "/w/$arm-disasm.exit"
 sha256sum "/w/$arm.zebin" >> /w/decoder-input-after.sha256
done
'
