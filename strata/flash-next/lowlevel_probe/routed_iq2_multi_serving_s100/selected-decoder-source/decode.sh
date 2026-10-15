#!/usr/bin/env bash
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
O=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/routed-iq2-multi-s100-candidate-isa
command -v python3 docker >/dev/null
python3 "$P/prepare.py"
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
docker image inspect --format '{{.Id}}' "$IMAGE" > "$O/image-id.txt"
[[ $(cat "$O/image-id.txt") == "$IMAGE" ]]
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" -e TMPDIR=/w/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -v "$O":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && -w /w/tmp ]]
command -v ocloc sha256sum >/w/tool-paths.txt
sha256sum /w/candidate-input.zebin >/w/decoder-input-before.sha256
set +e
ocloc disasm -file /w/candidate-input.zebin -device bmg-g31 -dump /w/disasm >/w/ocloc.stdout 2>/w/ocloc.stderr
decoder_rc=$?
set -e
printf "%s\n" "$decoder_rc" >/w/ocloc.exit
sha256sum /w/candidate-input.zebin >/w/decoder-input-after.sha256
cmp /w/decoder-input-before.sha256 /w/decoder-input-after.sha256
exit "$decoder_rc"
'
