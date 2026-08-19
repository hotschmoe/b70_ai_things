#!/bin/bash
# intel/vllm:0.21.0-xpu wrapper. Image entrypoint sources oneAPI setvars;
# lib.sh --entrypoint vllm skips that and torch cannot load libccl.so.1.
# Default setvars pins CCL 2021.15 (PRE.10 device_fd). This image also
# ships 2021.17 against libsycl.so.8 -- use that for TP>1.
set +u
if [ -f /opt/intel/oneapi/setvars.sh ]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
fi
set -u
# D14: rebuilt 4ceafd1 (graph-replay) wins over in-image 2021.17 sched algos.
if [ -d /opt/ccl4ce/lib ]; then
  export CCL_ROOT=/opt/ccl4ce
else
  export CCL_ROOT=/opt/intel/oneapi/ccl/2021.17
fi
export LD_LIBRARY_PATH="${CCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
echo "=== intel021 wrapper CCL_ROOT=$CCL_ROOT ===" >&2
exec /opt/venv/bin/python3 -m vllm.entrypoints.cli.main "$@"
