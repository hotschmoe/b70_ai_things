#!/usr/bin/env bash
# Cross-card all-reduce microbench on 2x B70 (novel: no public Battlemage all-reduce numbers exist).
# GPU job -> wrap in gpu-run:  ./gpu-run bash 60_allreduce_bench.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
docker rm -f arbench 2>/dev/null || true
docker run --rm --name arbench --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$ROOT/allreduce_bench.py:/allreduce_bench.py:ro" \
  -e TMPDIR=/tmp_ssd \
  -e CCL_ENABLE_SYCL_KERNELS=0 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e CCL_ZE_IPC_EXCHANGE=pidfd \
  -e CCL_TOPO_P2P_ACCESS=0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint bash "$IMG" -lc '
    echo "=== link BEFORE ==="; for b in 0000:0a:00.0 0000:44:00.0; do echo " $b: none (host-only sysfs)"; done 2>/dev/null
    python /allreduce_bench.py
  '
