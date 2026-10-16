#!/usr/bin/env bash
set -euo pipefail
[[ $(id -u) == 1000 ]]
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
python3 "$HERE/prepare.py"
mkdir "$HERE/tmp"
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
 -e TMPDIR=/out/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -v "$HERE":/out --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && $(id -u) == 1000 ]]
[[ $(command -v ocloc) == /usr/bin/ocloc ]]
sha256sum /usr/bin/ocloc > /out/tool.sha256
sha256sum /out/sampler-alias.zebin /out/probability-alias.zebin > /out/inputs-before.sha256
for arm in sampler probability;do
 mkdir /out/$arm-disasm
 set +e
 ocloc disasm -file /out/$arm-alias.zebin -device bmg-g31 -dump /out/$arm-disasm > /out/$arm-ocloc.stdout 2> /out/$arm-ocloc.stderr
 code=$?
 printf "%s\n" "$code" > /out/$arm-decoder.exit
 set -e
 [[ $code == 0 && -s /out/$arm-disasm/.text.s114_$arm.asm ]]
 sha256sum /out/$arm-disasm/.text.s114_$arm.asm > /out/$arm-asm.sha256
done
sha256sum /out/sampler-alias.zebin /out/probability-alias.zebin > /out/inputs-after.sha256
cmp /out/inputs-before.sha256 /out/inputs-after.sha256
'
