#!/usr/bin/env bash
# Caller holds cpu-run. Only offline candidate decoder, never compile/GPU.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
python3 "$P/extract.py" > "$P/extract.stdout" 2> "$P/extract.stderr"
docker run --rm --network none -u "$(id -u):$(id -g)" \
  -v "$P":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
command -v ocloc > /w/tool-path.txt
mkdir /w/candidate-disasm
ocloc disasm -file /w/candidate.zebin -device bmg-g31 -dump /w/candidate-disasm > /w/disasm.log 2> /w/disasm.stderr
printf "%s\n" "$?" > /w/disasm.exit
'
