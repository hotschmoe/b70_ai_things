#!/usr/bin/env bash
# Run a stock Strata configuration under the pair lease.
set -euo pipefail
repo=/mnt/vm_8tb/github/b70_ai_things
build=/mnt/vm_8tb/b70/build/strata-ab
results=/mnt/vm_8tb/b70/results
weights=$repo/models/files/qwen3.8-flash-next
nvme=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco
config=${1:?Pass an absolute serving config path}
cd "$repo"
exec bin/gpu-run docker run --rm -i --name strata-ab-serve --network host \
  --memory 90g --device /dev/dri --ulimit memlock=-1 \
  -v "$build:$build" -v "$results:$results" \
  -v "$weights/ista-gsq-rco-iq3-s:$weights/ista-gsq-rco-iq3-s:ro" \
  -v "$weights/ista-gsq-rco-iq3-xxs:$weights/ista-gsq-rco-iq3-xxs:ro" \
  -v "$nvme:$nvme:ro" \
  -e ONEAPI_DEVICE_SELECTOR='level_zero:*' -e SYCL_CACHE_PERSISTENT=0 \
  -e ZES_ENABLE_SYSMAN=1 -e STRATA_VERIFY_DEVICE_PLAN=1 \
  -e STRATA_VERIFY_NO_HOST=1 -e STRATA_MIRROR_MIB=0 \
  -e STRATA_STAGE_TRIM=1 -e STRATA_KV_WIDE_HOST=0 -e STRATA_STAGER_THREADS=12 \
  --entrypoint bash strata-ab-build:ocloc -lc \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd "$1/src"; exec "$1/venv/bin/python" sycl/serve/server_intel.py --engine strata --config "$2" --host 0.0.0.0 --port 18085' \
  bash "$build" "$config"
