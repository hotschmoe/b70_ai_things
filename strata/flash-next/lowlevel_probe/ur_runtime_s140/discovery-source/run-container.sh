#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ -x /probe/cross_queue_graph && -e /dev/dri ]]
cd /probe
sha256sum -c /probe/runtime.sha256
unset SYCL_UR_USE_LEVEL_ZERO_V2 UR_LOADER_USE_LEVEL_ZERO_V2 UR_L0_USE_DRIVER_INORDER_LISTS
python3 - <<'PYENV'
import json,os
from pathlib import Path
keys=['SYCL_UR_TRACE','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO','SYCL_UR_USE_LEVEL_ZERO_V2','UR_LOADER_USE_LEVEL_ZERO_V2','UR_L0_USE_DRIVER_INORDER_LISTS','ONEAPI_DEVICE_SELECTOR','ZE_AFFINITY_MASK']
Path('/out/environment.json').write_text(json.dumps({k:os.environ.get(k)for k in keys},indent=2)+'\n',encoding='ascii')
PYENV
mkdir /out/mode0
set +e
/probe/cross_queue_graph 0 /out/mode0/leaf > /out/mode0/leaf.log 2> /out/mode0/leaf.stderr
rc=$?
set -e
printf '%s\n' "$rc" > /out/mode0/leaf.exit
[[ $rc == 0 ]] || exit "$rc"
python3 /src/analyze.py /out 0 > /out/mode0/analysis.stdout 2> /out/mode0/analysis.stderr
python3 /driver/discovery.py /out > /out/discovery.stdout 2> /out/discovery.stderr
