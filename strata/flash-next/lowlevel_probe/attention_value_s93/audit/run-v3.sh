#!/usr/bin/env bash
# ROOT ONLY after real main build. CPU lease must enclose this whole script.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPECTED=${1:?root actual frozen main executable SHA256}
python3 "$P/audit-v3.py" "$EXPECTED" > "$P/audit-v3.stdout" 2> "$P/audit-v3.stderr"
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
docker image inspect --format '{{.Id}}' "$IMAGE" > "$P/decoder-image-id.txt"
[[ $(cat "$P/decoder-image-id.txt") == "$IMAGE" ]]
mkdir "$P/tmp-v3"
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" -e TMPDIR=/w/tmp-v3 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -v "$P":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri ]]
command -v ocloc > /w/decoder-tool-path.txt
while read -r arm;do
 [[ $arm == original || $arm == candidate || $arm == oldmerge || $arm == newmerge ]]
 sha256sum "/w/$arm.zebin" >> /w/decoder-input-before.sha256
 mkdir "/w/$arm-disasm"
 ocloc disasm -file "/w/$arm.zebin" -device bmg-g31 -dump "/w/$arm-disasm" > "/w/$arm-disasm.log" 2> "/w/$arm-disasm.stderr"
 printf "%s\n" "$?" > "/w/$arm-disasm.exit"
 sha256sum "/w/$arm.zebin" >> /w/decoder-input-after.sha256
done < /w/decode-todo.txt
'
python3 "$P/summarize.py" > "$P/summarize.stdout" 2> "$P/summarize.stderr"
