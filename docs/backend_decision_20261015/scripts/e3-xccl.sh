#!/usr/bin/env bash
# E3(b) runner. Usage: e3-xccl.sh [graph 0|1]. Run as: cpu-run --quiet bin/gpu-run e3-xccl.sh
IMG="b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78"
CCL_ROOT=/opt/venv
timeout --signal=TERM --kill-after=10 ${E3_TIMEOUT:-180} docker run --rm --name bd-e3-xccl --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vm_8tb/b70/build/backend-decision/e3_xccl_sync.py:/opt/e3.py:ro \
  -e E3_GRAPH=${1:-0} -e CCL_ATL_TRANSPORT=ofi -e CCL_TOPO_P2P_ACCESS=0 -e CCL_LOG_LEVEL=warn \
  -e CCL_KERNEL_PATH=$CCL_ROOT/lib/ccl/kernels -e FI_TCP_IFACE=eth0 -e CCL_KVS_IFACE=eth0 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 -e ZE_AFFINITY_MASK=0,1 \
  -e LD_PRELOAD=$CCL_ROOT/lib/libccl.so.1.0 \
  -e LD_LIBRARY_PATH=$CCL_ROOT/lib:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  --entrypoint /opt/venv/bin/torchrun "$IMG" --standalone --nproc-per-node=2 /opt/e3.py 2>&1 | grep -E "E3_XCCL_RESULT|Error|error|DEVICE_LOST|Traceback" | head -20
rc=${PIPESTATUS[0]}; docker rm -f bd-e3-xccl >/dev/null 2>&1; echo "e3-xccl rc=$rc"; exit $rc
