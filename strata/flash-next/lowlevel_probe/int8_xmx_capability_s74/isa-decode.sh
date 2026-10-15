#!/usr/bin/env bash
# Caller holds bin/cpu-run. Offline decoder only: no GPU device, network, compile.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
command -v python3 docker >/dev/null
python3 "$P/extract.py" > "$P/extract.stdout" 2> "$P/extract.stderr"
docker run --rm --network none -u "$(id -u):$(id -g)" \
  -v "$P":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
command -v ocloc > /w/tool-path.txt
mkdir /w/disasm
ocloc disasm -file /w/capability.zebin -device bmg-g31 -dump /w/disasm > /w/disasm.log 2> /w/disasm.stderr
printf "%s\n" "$?" > /w/disasm.exit
'
