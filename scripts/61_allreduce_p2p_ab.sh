#!/usr/bin/env bash
# All-reduce P2P on/off A/B on kernel 7.0 (B70<->B70). Sibling of 60_allreduce_bench.sh; the new
# question after canAccessPeer flipped True on 7.0 (P2P_GPU H.11): does enabling CCL P2P access lift
# the host-staged ~1.16 GB/s (H.10) toward the ~15.8 GB/s Gen3 x16 wire?
# GPU job -> wrap in gpu-run:  ./gpu-run bash 61_allreduce_p2p_ab.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"

run_variant() {  # name P2P_ACCESS SYCL_KERNELS IPC
  local name="$1" p2p="$2" sycl="$3" ipc="$4"
  echo
  echo "############################################################"
  echo "## VARIANT $name :  CCL_TOPO_P2P_ACCESS=$p2p  CCL_ENABLE_SYCL_KERNELS=$sycl  IPC=$ipc"
  echo "############################################################"
  docker rm -f arbench 2>/dev/null || true
  docker run --rm --name arbench --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g \
    -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$ROOT/allreduce_bench.py:/allreduce_bench.py:ro" \
    -e TMPDIR=/tmp_ssd \
    -e CCL_ENABLE_SYCL_KERNELS="$sycl" -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e CCL_ZE_IPC_EXCHANGE="$ipc" \
    -e CCL_TOPO_P2P_ACCESS="$p2p" -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --entrypoint bash "$IMG" -lc 'python /allreduce_bench.py' 2>&1 \
    | grep -E "torch |===|bytes|^ *[0-9]+ |FAILED|no ipex|no oneccl|Error|error" || true
}

# Baselines (P2P off) then P2P on. Same image, only the P2P/IPC/SYCL knobs vary.
# NOTE: this oneCCL build only accepts CCL_ZE_IPC_EXCHANGE = sockets|pidfd (NOT drmfd from the old 6.18
# image). P2P is toggled by CCL_TOPO_P2P_ACCESS, independent of the IPC exchange mechanism -> use pidfd.
run_variant "A_p2pOFF_eager"  0 0 pidfd   # ~0.68 GB/s expected (H.10 eager)
run_variant "B_p2pOFF_sycl"   0 1 pidfd   # ~1.16 GB/s expected (H.10 captured-serve path)
run_variant "C_p2pON_eager"   1 0 pidfd
run_variant "D_p2pON_sycl"    1 1 pidfd   # THE question: does this climb toward the ~15.8 GB/s wire?

echo; echo "=== A/B DONE. Compare algbw_GB/s across variants at 16MB / 256MB. ==="
