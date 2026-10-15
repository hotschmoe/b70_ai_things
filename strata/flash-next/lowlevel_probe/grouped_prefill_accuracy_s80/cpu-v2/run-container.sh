#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
command -v sha256sum python3 >/dev/null
[[ -x /build/reference_s80 && -w /out ]]
sha256sum -c /prep/expected-container.sha256
python3 /prep/finalize.py pre /out /inputs /prep /build
TIMEFORMAT='CPU_reference_seconds=%R'
{ time /build/reference_s80 /inputs /prep/selected36.tsv /out/reference > /out/helper.stdout 2> /out/helper.stderr; } 2> /out/helper.time
sha256sum -c /prep/expected-container.sha256
python3 /prep/finalize.py post /out /inputs /prep /build
