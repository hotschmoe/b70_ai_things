#!/usr/bin/env bash
# Caller MUST hold bin/cpu-run. No compile, GPU device or network.
set -euo pipefail
P=$(cd "$(dirname "$0")" && pwd)
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
python3 "$P/extract.py" > "$P/extract.stdout" 2> "$P/extract.stderr"
docker run --rm --network none -u "$(id -u):$(id -g)" \
  -v "$P":/w --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
command -v ocloc > /w/tool-path.txt
for arm in original candidate;do
  mkdir "/w/$arm-disasm"
  ocloc disasm -file "/w/$arm.zebin" -device bmg-g31 -dump "/w/$arm-disasm" > "/w/$arm-disasm.log" 2> "/w/$arm-disasm.stderr"
  printf "%s\n" "$?" > "/w/$arm-disasm.exit"
done
'
