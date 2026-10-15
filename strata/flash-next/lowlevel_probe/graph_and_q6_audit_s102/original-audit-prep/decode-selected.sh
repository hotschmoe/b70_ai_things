#!/usr/bin/env bash
# SOURCE ONLY: root first reads facts and decides whether one decoder is needed.
set -euo pipefail
[[ $# == 1 && $1 =~ ^[0-9a-f]{64}$ ]]
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
O=$(dirname -- "$P")/q6-a4-s102-audit
[[ -f "$O/facts.json" && ! -e "$O/disasm" ]]
printf '%s  %s\n' "$1" "$O/q6-c6-shortname.zebin" | sha256sum -c -
mkdir "$O/disasm" "$O/tmp"
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" -e TMPDIR=/w/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -v "$O":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && -w /w/tmp ]]
command -v ocloc > /w/decoder-tool-path.txt
sha256sum /w/q6-c6-shortname.zebin > /w/decoder-input-before.sha256
ocloc disasm -file /w/q6-c6-shortname.zebin -device bmg-g31 -dump /w/disasm > /w/ocloc.stdout 2> /w/ocloc.stderr
printf "%s\n" "$?" > /w/ocloc.exit
sha256sum /w/q6-c6-shortname.zebin > /w/decoder-input-after.sha256
cmp /w/decoder-input-before.sha256 /w/decoder-input-after.sha256
'
