#!/usr/bin/env bash
# Strata 0.2.16 launcher (copy of night/nl_launch.sh: dev:020 image, strata-0216 mount + venv, no STRATA_MIRROR_MIB=0).
# Run one Strata config under the pair GPU lease (foreground). Usage: nl_launch_0216.sh <container-name> <config.json> [port]
set -euo pipefail
repo=/mnt/vm_8tb/github/b70_ai_things
build=/mnt/vm_8tb/b70/build/strata-ab
results=/mnt/vm_8tb/b70/results
weights=$repo/models/files/qwen3.8-flash-next
nvme=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco
container=${1:?container}
config=$(realpath "${2:?config}")
port=${3:-18085}
image=${NL_IMAGE:-strata-sycl-dev:020}
source_dir=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cwd"])' "$config")
ipc_opts=()
if python3 -c 'import json,sys; sys.exit("--expert-parallel-device" not in json.load(open(sys.argv[1]))["args"])' "$config"; then
  ipc_opts=(--cap-add SYS_PTRACE)
fi
extra_env=()
# NL_ENV="K=V K2=V2" adds container-level env (engine env belongs in config.json "env").
for kv in ${NL_ENV:-}; do extra_env+=(-e "$kv"); done
cd "$repo"
exec bin/gpu-run docker run --rm -i --name "$container" --network host \
  "${ipc_opts[@]}" "${extra_env[@]}" \
  --memory 90g --device /dev/dri --ulimit memlock=-1 \
  --health-cmd "curl -fsS -o /dev/null http://127.0.0.1:$port/health" --health-interval 30s \
  --health-start-period 600s --health-retries 3 \
  -v "$build:$build" -v "$results:$results" -v "$config:$config:ro" \
  -v /mnt/vm_8tb/b70/build/strata-0216:/mnt/vm_8tb/b70/build/strata-0216 \
  -v "$weights/ista-gsq-rco-iq3-s:$weights/ista-gsq-rco-iq3-s:ro" \
  -v "$weights/ista-gsq-rco-iq3-xxs:$weights/ista-gsq-rco-iq3-xxs:ro" \
  -v "$nvme:$nvme:ro" \
  -e ONEAPI_DEVICE_SELECTOR='level_zero:*' -e SYCL_CACHE_PERSISTENT=0 \
  -e ZES_ENABLE_SYSMAN=1 -e STRATA_VERIFY_DEVICE_PLAN=1 \
  -e STRATA_VERIFY_NO_HOST=1 \
  -e STRATA_STAGE_TRIM=1 -e STRATA_KV_WIDE_HOST=0 -e STRATA_STAGER_THREADS=12 \
  --entrypoint bash "$image" -lc \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd "$3"; exec /mnt/vm_8tb/b70/build/strata-0216/venv/bin/python sycl/serve/server_intel.py --engine strata --config "$2" --host 127.0.0.1 --port "$4"' \
  bash "$build" "$config" "$source_dir" "$port"
