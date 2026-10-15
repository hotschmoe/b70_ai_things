#!/usr/bin/env bash
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
O=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/routed-iq2-existing-isa-s97
command -v python3 docker c++filt >/dev/null
[[ ! -e "$O/selection.json" && ! -e "$O/disasm" ]]
python3 "$P/extract.py" > "$O/extract.stdout" 2> "$O/extract.stderr"
mkdir "$O/tmp" "$O/disasm"
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
docker image inspect --format '{{.Id}}' "$IMAGE" > "$O/image-id.txt"
[[ $(cat "$O/image-id.txt") == "$IMAGE" ]]
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" -e TMPDIR=/w/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -v "$O":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && -w /w && -w /w/tmp ]]
command -v ocloc sha256sum >/w/tool-paths.txt
sha256sum /w/qt17-shortname.zebin >/w/decoder-input-before.sha256
ocloc disasm -file /w/qt17-shortname.zebin -device bmg-g31 -dump /w/disasm >/w/ocloc.stdout 2>/w/ocloc.stderr
printf "%s\n" "$?" >/w/ocloc.exit
sha256sum /w/qt17-shortname.zebin >/w/decoder-input-after.sha256
cmp /w/decoder-input-before.sha256 /w/decoder-input-after.sha256
'
