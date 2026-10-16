#!/usr/bin/env bash
# Contingency recipe ONLY. Do not run after native-prefix reuse PASS.
set -euo pipefail
[[ ${ROOT_ONE_SELECTED_DECODE_GO:-0} == 1 ]]
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
O=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/routed-gu17-existing-audit
[[ -f "$O/HOLD.json" && ! -e "$O/isa-reuse.json" && ! -e "$O/disasm" ]]
# Root must first inspect the HOLD and separately authorize this single decode.
python3 "$P/prepare_decode.py" "$O"
mkdir "$O/tmp" "$O/disasm"
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
docker run --rm --network none --cpus 4 -u 1000:1000 -e TMPDIR=/w/tmp \
 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -v "$O":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && -w /w/tmp && $(command -v ocloc) == /usr/bin/ocloc ]]
sha256sum /usr/bin/ocloc > /w/tool.sha256
[[ $(cut -d" " -f1 /w/tool.sha256) == 7a418c1e7dfc62b6fb3895bea111647fa3e1ff600bb69f6be2a3114b90c41eac ]]
sha256sum /w/selected-short.zebin > /w/decoder-input-before.sha256
set +e
timeout 60 ocloc disasm -file /w/selected-short.zebin -device bmg-g31 -dump /w/disasm > /w/ocloc.stdout 2> /w/ocloc.stderr
rc=$?
set -e
printf "%s\n" "$rc" > /w/ocloc.exit
sha256sum /w/selected-short.zebin > /w/decoder-input-after.sha256
cmp /w/decoder-input-before.sha256 /w/decoder-input-after.sha256
exit "$rc"
'
